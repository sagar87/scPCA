from functools import partial
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch
from anndata import AnnData  # type: ignore
from numpy.typing import NDArray
from patsy import dmatrix  # type: ignore
from torch.types import Device

from .models import dpca_guide, dpca_model, scpca_guide, scpca_model
from .train import SUBSAMPLE, SVILocalHandler
from .utils import get_rna_counts, get_states


class scPCA:
    """
    Single-cell Principal Component Analysis (scPCA) model.

    This class provides an interface to perform scPCA on single-cell data. It allows for
    the extraction of principal components while accounting for batch effects and other
    covariates.

    Parameters
    ----------
    adata :
        Anndata object containing the single-cell data.
    num_factors :
        Number of factors to fit.
    layers_key :
        Key to extract single-cell count matrix from adata.layers. If None, scPCA will
        try to extract the count matrix from adata.X. Default is None.
    design_formula :
        R style formula to construct the design matrix from adata.obs. If None, scPCA
        fits a normal PCA. Default is "1".
    intercept_formula :
        R style formula to extract batch information from adata.obs. Default is "1".
    subsampling :
        Number of cells to subsample for training. A larger number will result in a more
        accurate computation of the gradients, but will also increase the training time
        and memory usage. Default is 4096.
    device :
        Device to run the model on. A GPU is recommended. Default is GPU if available, else CPU.
    seed :
        Random seed for reproducibility. Default is None.
    model_kwargs :
        Additional keyword arguments for the model. Default values are provided.
    training_kwargs :
        Additional keyword arguments for training. Default is SUBSAMPLE.
    """

    def __init__(
        self,
        adata: AnnData,
        num_factors: int,
        layers_key: Union[str, None] = None,
        design_formula: str = "1",
        intercept_formula: str = "1",
        subsampling: int = 4096,
        device: Optional[Literal["cuda", "cpu"]] = None,
        seed: Optional[int] = None,
        model_kwargs: Dict[str, Any] = {
            "β_rna_sd": 0.01,
            "β_rna_mean": 3.0,
            "fixed_beta": True,
        },
        training_kwargs: Dict[str, Any] = SUBSAMPLE,
    ):
        self.adata = adata
        self.num_factors = num_factors
        self.layers_key = layers_key
        self.design_formula = design_formula
        self.intercept_formula = intercept_formula
        self.subsampling = min([subsampling, adata.shape[0]])
        self.device: Device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )
        self.seed: Optional[torch.Generator] = seed if seed is None else torch.manual_seed(self.seed)

        # prepare design and batch matrix
        self.design_matrix = dmatrix(design_formula, adata.obs)
        self.design_states = get_states(self.design_matrix)
        self.intercept_matrix = dmatrix(intercept_formula, adata.obs)
        self.intercept_states = get_states(self.intercept_matrix)

        #
        self.model_kwargs = model_kwargs
        self.training_kwargs = training_kwargs

        # setup data
        self.data = self._setup_data()
        self.handler = self._setup_handler()

    def _to_torch(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert numpy arrays of a dictionary to torch tensors.

        Parameters
        ----------
        data :
            Dictionary containing numpy arrays.

        Returns
        -------
            Dictionary with numpy arrays converted to torch tensors.
        """
        return {k: torch.tensor(v, device=self.device) if isinstance(v, np.ndarray) else v for k, v in data.items()}

    def _setup_data(self) -> Dict[str, Union[torch.Tensor, int, None]]:
        """
        Prepare the data for the scPCA model.

        Returns
        -------
            Dictionary containing tensors and other relevant information for the model.
        """
        X = get_rna_counts(self.adata, self.layers_key)
        X_size = np.log(X.sum(axis=1, keepdims=True))

        design: NDArray[np.float32] = np.asarray(self.design_states.encoding).astype(np.float32)
        design_idx = self.design_states.idx

        intercept: NDArray[np.float32] = np.asarray(self.intercept_states.encoding).astype(np.float32)
        intercept_idx = self.intercept_states.idx

        num_genes = X.shape[1]
        num_cells = X.shape[0]
        idx = np.arange(num_cells)

        data = dict(
            num_factors=self.num_factors,
            X=X,
            X_size=X_size,
            design=design,
            intercept=intercept,
            design_idx=design_idx,
            intercept_idx=intercept_idx,
            idx=idx,
            num_genes=num_genes,
            num_cells=num_cells,
        )
        return self._to_torch(data)

    def _setup_handler(self) -> SVILocalHandler:
        """
        Set up the handler for training the scPCA model.

        Returns
        -------
            Handler for training the model.
        """
        train_model = partial(
            scpca_model, subsampling=self.subsampling, device=self.device, **self.data, **self.model_kwargs
        )
        train_guide = partial(
            scpca_guide, subsampling=self.subsampling, device=self.device, **self.data, **self.model_kwargs
        )
        idx = self.data.pop("idx")
        predict_model = partial(scpca_model, subsampling=0, device=self.device, **self.data, **self.model_kwargs)
        predict_guide = partial(scpca_guide, subsampling=0, device=self.device, **self.data, **self.model_kwargs)

        return SVILocalHandler(
            model=train_model,
            guide=train_guide,
            predict_model=predict_model,
            predict_guide=predict_guide,
            idx=idx,
            **self.training_kwargs,
        )

    def fit(self, *args: Any, **kwargs: Any) -> None:
        """
        Fit the scPCA model to the data.

        Parameters
        ----------
        *args :
            Positional arguments for the fit method.
        **kwargs :
            Keyword arguments for the fit method.
        """
        self.handler.fit(*args, **kwargs)

    def _meta_to_anndata(self, model_key: str) -> None:
        """
        Store meta information in the AnnData object.

        Parameters
        ----------
        model_key :
            Key to store the model in the AnnData object.
        """
        res: Dict[str, Any] = {}
        res["design"] = self.design_states.sparse
        res["intercept"] = self.intercept_states.sparse

        res["design_index"] = self.design_states.idx
        res["intercept_index"] = self.intercept_states.idx

        res["loss"] = self.handler.loss
        res["model"] = {"num_factors": self.num_factors, "seed": self.seed, **self.model_kwargs}

        self.adata.uns[f"{model_key}"] = res

    def posterior_to_anndata(self, model_key: str, num_samples: int = 25) -> None:
        """
        Store the posterior samples in the AnnData object.

        Parameters
        ----------
        model_key :
            Key to store the model in the AnnData object.
        num_samples :
            Number of samples to draw from the posterior. Default is 25.
        """

        self._meta_to_anndata(model_key)
        adata = self.adata

        adata.varm[f"{model_key}_W_rna"] = (
            self.handler.predict_global_variable("W_lin", num_samples=num_samples).T.swapaxes(-1, -3).swapaxes(-1, -2)
        )
        adata.varm[f"{model_key}_V_rna"] = self.handler.predict_global_variable(
            "W_add", num_samples=num_samples
        ).T.swapaxes(-1, -2)

        α_rna = self.handler.predict_global_variable("α_rna", num_samples=num_samples).T

        if α_rna.ndim == 2:
            α_rna = np.expand_dims(α_rna, 1)

        adata.varm[f"{model_key}_α_rna"] = α_rna.swapaxes(-1, -2)

        σ_rna = self.handler.predict_global_variable("σ_rna", num_samples=num_samples).T

        if σ_rna.ndim == 2:
            σ_rna = np.expand_dims(σ_rna, 1)

        adata.varm[f"{model_key}_σ_rna"] = σ_rna.swapaxes(-1, -2)

        adata.obsm[f"X_{model_key}"] = self.handler.predict_local_variable("z", num_samples=num_samples).swapaxes(0, 1)

    def mean_to_anndata(self, model_key: str, num_samples: int = 25, num_split: int = 2048) -> None:
        """
        Store the posterior mean estimates in the AnnData object.

        Parameters
        ----------
        model_key :
            Key to store the model in the AnnData object.
        num_samples :
            Number of samples to draw from the posterior. Default is 25.
        num_split :
            Number of splits for the data. Default is 2048.
        """
        self._meta_to_anndata(model_key)
        adata = self.adata

        adata.layers[f"{model_key}_μ_rna"] = self.handler.predict_local_variable(
            "μ_rna", num_samples=num_samples, num_split=num_split
        ).mean(0)
        adata.layers[f"{model_key}_offset_rna"] = self.handler.predict_local_variable(
            "offset_rna", num_samples=num_samples, num_split=num_split
        ).mean(0)
        adata.obsm[f"X_{model_key}"] = self.handler.predict_local_variable(
            "z", num_samples=num_samples, num_split=num_split
        ).mean(0)
        adata.varm[f"{model_key}_W_rna"] = (
            self.handler.predict_global_variable("W_lin", num_samples=num_samples).mean(0).T
        )
        adata.varm[f"{model_key}_V_rna"] = (
            self.handler.predict_global_variable("W_add", num_samples=num_samples).mean(0).T
        )
        adata.varm[f"{model_key}_α_rna"] = self.handler.predict_global_variable("α_rna").mean(0).T
        adata.varm[f"{model_key}_σ_rna"] = self.handler.predict_global_variable("σ_rna").mean(0).T


class dPCA(scPCA):
    """
    Design Principal Component Analysis (dPCA) model.


    Parameters
    ----------
    adata :
        Anndata object containing the single-cell data.
    num_factors :
        Number of factors to fit. Default is 15.
    layers_key :
        Key to extract single-cell count matrix from adata.layers. If None, scPCA will
        try to extract the count matrix from adata.X. Default is None.
    batch_formula :
        R style formula to extract batch information from adata.obs. If None, scPCA
        assumes a single batch. Default is None.
    design_formula :
        R style formula to construct the design matrix from adata.obs. If None, scPCA
        fits a normal PCA. Default is None.
    subsampling :
        Number of cells to subsample for training. Default is 4096.
    device :
        Device to run the model on. A GPU is recommended. Default is GPU if available, else CPU.
    model_kwargs :
        Additional keyword arguments for the model. Default values are provided.
    training_kwargs :
        Additional keyword arguments for training. Default is SUBSAMPLE.
    """

    def __init__(
        self,
        adata: AnnData,
        num_factors: int,
        layers_key: Union[str, None] = None,
        design_formula: str = "1",
        intercept_formula: str = "1",
        subsampling: int = 4096,
        device: Optional[Literal["cuda", "cpu"]] = None,
        seed: Optional[int] = None,
        model_kwargs: Dict[str, Any] = {
            "z_sd": 1.0,
        },
        training_kwargs: Dict[str, Any] = SUBSAMPLE,
    ):
        self.adata = adata
        self.num_factors = num_factors
        self.layers_key = layers_key
        self.design_formula = design_formula
        self.intercept_formula = intercept_formula
        self.subsampling = min([subsampling, adata.shape[0]])
        self.device: Device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        )
        self.seed: Optional[torch.Generator] = seed if seed is None else torch.manual_seed(self.seed)

        # prepare design and batch matrix
        self.design_matrix = dmatrix(design_formula, self.adata)
        self.intercept_matrix = dmatrix(intercept_formula, self.adata)

        #
        self.model_kwargs = model_kwargs
        self.training_kwargs = training_kwargs

        # setup data
        self.data = self._setup_data()
        self.handler = self._setup_handler()

    def _setup_data(self) -> Dict[str, Union[torch.Tensor, int, None]]:
        """
        Sets up the data.
        """
        X = get_rna_counts(self.adata, self.layers_key)
        intercept_design: NDArray[np.float32] = np.asarray(self.intercept_matrix).astype(np.float32)
        loading_design: NDArray[np.float32] = np.asarray(self.design_matrix).astype(np.float32)

        num_obs = X.shape[0]
        idx = np.arange(num_obs)

        data = dict(
            X=X,
            intercept_design=intercept_design,
            loading_design=loading_design,
            idx=idx,
            num_obs=num_obs,
        )
        return self._to_torch(data)

    def _setup_handler(self) -> SVILocalHandler:
        """
        Sets up the handler for training the model.
        """
        train_model = partial(
            dpca_model,
            num_factors=self.num_factors,
            subsampling=self.subsampling,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        train_guide = partial(
            dpca_guide,
            num_factors=self.num_factors,
            subsampling=self.subsampling,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        idx = self.data.pop("idx")

        predict_model = partial(
            dpca_model,
            num_factors=self.num_factors,
            subsampling=0,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        predict_guide = partial(
            dpca_guide,
            num_factors=self.num_factors,
            subsampling=0,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        return SVILocalHandler(
            model=train_model,
            guide=train_guide,
            predict_model=predict_model,
            predict_guide=predict_guide,
            idx=idx,
            **self.training_kwargs,
        )

    def fit(self, *args: Any, **kwargs: Any) -> None:
        self.handler.fit(*args, **kwargs)

    def mean_to_anndata(self, model_key: str, num_samples: int = 25, num_split: int = 2048) -> None:
        adata = self.adata
        # set up unstructured metadata
        adata.uns[f"{model_key}"] = {}
        adata.uns[f"{model_key}"]["posterior"] = {
            "μ_rna": self.handler.predict_local_variable("μ_rna", num_samples=num_samples).mean(0),
            "σ": self.handler.predict_global_variable("σ", num_samples=num_samples).mean(0),
            "W_add": self.handler.predict_global_variable("W_add", num_samples=num_samples).mean(0).T,
        }

        W_fac = self.handler.predict_global_variable("W_fac", num_samples=num_samples).mean(0)

        # V = self.handler.predict_global_variable("W_add", num_samples=num_samples).mean(0)

        adata.uns[f"{model_key}"]["design"] = get_states(self.design_matrix)
        # adata.uns[f"{model_key}"]["batch"] = get_states(self.batch)
        adata.uns[f"{model_key}"]["model"] = self.model_kwargs

        adata.uns[f"{model_key}"]["W"] = {}
        adata.uns[f"{model_key}"]["V"] = {}
        adata.uns[f"{model_key}"]["state_vec"] = {}
        adata.uns[f"{model_key}"]["state_scale"] = {}
        adata.uns[f"{model_key}"]["state_unit"] = {}

        for i, (k, v) in enumerate(adata.uns[f"{model_key}"]["design"].items()):
            adata.uns[f"{model_key}"]["W"][k] = W_fac[v[-1]]
            adata.uns[f"{model_key}"]["state_vec"][k] = W_fac[v].sum(axis=0)
            adata.uns[f"{model_key}"]["state_scale"][k] = np.linalg.norm(
                W_fac[v].sum(axis=0), ord=2, axis=1, keepdims=True
            )
            adata.uns[f"{model_key}"]["state_unit"][k] = W_fac[v].sum(axis=0) / np.linalg.norm(
                W_fac[v].sum(axis=0), ord=2, axis=1, keepdims=True
            )

        # for k, v in get_states(self.intercept_matrix).items():
        #     adata.uns[f"{model_key}"]["V"][k] = V[v[-1]]

        adata.obsm[f"X_{model_key}"] = self.handler.predict_local_variable("z", num_samples=num_samples).mean(0)

        adata.varm[f"{model_key}"] = self.handler.predict_global_variable("W_fac", num_samples=num_samples).mean(0).T

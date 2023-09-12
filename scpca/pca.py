from functools import partial
from typing import Any, Dict, Optional, Union

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
    scPCA model.

    Parameters
    ----------
    adata: anndata.AnnData
        Anndata object with the single-cell data.
    num_factors: int (default: 15)
        Number of factors to fit.
    layers_key: str or None (default: None)
        Key to extract single-cell count matrix from adata.layers. If layers_key is None,
        scPCA will try to extract the count matrix from the adata.X.
    batch_formula: str or None (default: None)
        R style formula to extract batch information from adata.obs. If batch_formula is None,
        scPCA assumes a single batch. Usually `batch_column - 1`.
    design_formula: str or None (default: None)
        R style formula to construct the design matrix from adata.obs. If design_formula is None,
        scPCA fits a normal PCA.
    subsampling: int (default: 4096)
        Number of cells to subsample for training. A larger number will result in a more accurate
        computation of the gradients, but will also increase the training time and memory usage.
    device: torch.device (default: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        Device to run the model on. A GPU is highly recommended.
    model_key: str (default: "scpca")
        Key to store the model in the AnnData object.
    model_kwargs: dict
        Model parameters. See sccca.model.model for more details.
    training_kwargs: dict
        Training parameters. See sccca.handler for more details.
    """

    def __init__(
        self,
        adata: AnnData,
        num_factors: int,
        layers_key: Union[str, None] = None,
        design_formula: str = "1",
        intercept_formula: str = "1",
        subsampling: int = 4096,
        device: Device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed: Optional[int] = None,
        model_kwargs: Dict[str, Any] = {
            "β_rna_sd": 0.01,
            "β_rna_mean": 3.0,
            "fixed_beta": True,
            "intercept": True,
            "batch_beta": False,
            "horseshoe": False,
        },
        training_kwargs: Dict[str, Any] = SUBSAMPLE,
    ):
        self.seed = seed

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.adata = adata
        self.num_factors = num_factors
        self.layers_key = layers_key
        self.design_formula = design_formula
        self.intercept_formula = intercept_formula

        self.subsampling = min([subsampling, adata.shape[0]])
        self.device = device

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
        Helper method to convert numpy arrays of a dict to torch tensors.
        """
        return {k: torch.tensor(v, device=self.device) if isinstance(v, np.ndarray) else v for k, v in data.items()}

    def _setup_data(self) -> Dict[str, Union[torch.Tensor, int, None]]:
        """
        Sets up the data.
        """
        X = get_rna_counts(self.adata, self.layers_key)
        X_size = np.log(X.sum(axis=1, keepdims=True))

        design: NDArray[np.float32] = np.asarray(self.design_states.encoding).astype(np.float32)
        design_idx = self.design_states.idx

        batch: NDArray[np.float32] = np.asarray(self.intercept_states.encoding).astype(np.float32)
        batch_idx = self.intercept_states.idx

        num_genes = X.shape[1]
        num_cells = X.shape[0]
        num_batches = batch.shape[1]
        idx = np.arange(num_cells)

        data = dict(
            num_factors=self.num_factors,
            X=X,
            X_size=X_size,
            Y=None,
            Y_size=None,
            design=design,
            batch=batch,
            design_idx=design_idx,
            batch_idx=batch_idx,
            idx=idx,
            num_genes=num_genes,
            num_proteins=None,
            num_batches=num_batches,
            num_cells=num_cells,
        )
        return self._to_torch(data)

    def _setup_handler(self) -> SVILocalHandler:
        """
        Sets up the handler for training the model.
        """
        train_model = partial(
            scpca_model,
            subsampling=self.subsampling,
            minibatches=False,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        train_guide = partial(
            scpca_guide,
            subsampling=self.subsampling,
            minibatches=False,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        idx = self.data.pop("idx")

        predict_model = partial(
            scpca_model,
            subsampling=0,
            minibatches=True,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        predict_guide = partial(
            scpca_guide,
            subsampling=0,
            minibatches=True,
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

    def _meta_to_anndata(self, model_key: str) -> None:
        res: Dict[str, Any] = {}
        res["design"] = self.design_states.sparse
        res["intercept"] = self.intercept_states.sparse

        res["design_index"] = self.design_states.idx
        res["intercept_index"] = self.intercept_states.idx

        res["loss"] = self.handler.loss
        res["model"] = {"num_factors": self.num_factors, "seed": self.seed, **self.model_kwargs}

        self.adata.uns[f"{model_key}"] = res

    def posterior_to_anndata(self, model_key: str, num_samples: int = 25) -> None:
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
    dPCA model.


    Parameters
    ----------
    adata: anndata.AnnData
        Anndata object with the single-cell data.
    num_factors: int (default: 15)
        Number of factors to fit.
    layers_key: str or None (default: None)
        Key to extract single-cell count matrix from adata.layers. If layers_key is None,
        scPCA will try to extract the count matrix from the adata.X.
    batch_formula: str or None (default: None)
        R style formula to extract batch information from adata.obs. If batch_formula is None,
        scPCA assumes a single batch. Usually `batch_column - 1`.
    design_formula: str or None (default: None)
        R style formula to construct the design matrix from adata.obs. If design_formula is None,
        scPCA fits a normal PCA.
    subsampling: int (default: 4096)
        Number of cells to subsample for training. A larger number will result in a more accurate
        computation of the gradients, but will also increase the training time and memory usage.
    device: torch.device (default: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        Device to run the model on. A GPU is highly recommended.
    model_key: str (default: "scpca")
        Key to store the model in the AnnData object.
    model_kwargs: dict
        Model parameters. See sccca.model.model for more details.
    training_kwargs: dict
        Training parameters. See sccca.handler for more details.
    """

    def __init__(
        self,
        adata: AnnData,
        num_factors: int,
        layers_key: Union[str, None] = None,
        design_formula: str = "1",
        intercept_formula: str = "1",
        subsampling: int = 4096,
        device: Device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
        self.device = device

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
            minibatches=False,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        train_guide = partial(
            dpca_guide,
            num_factors=self.num_factors,
            subsampling=self.subsampling,
            minibatches=False,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        idx = self.data.pop("idx")

        predict_model = partial(
            dpca_model,
            num_factors=self.num_factors,
            subsampling=0,
            minibatches=True,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        predict_guide = partial(
            dpca_guide,
            num_factors=self.num_factors,
            subsampling=0,
            minibatches=True,
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

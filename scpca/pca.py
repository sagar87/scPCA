from functools import partial
from typing import Any, Dict, Literal, Optional, Sequence, Union

import numpy as np
import torch
from anndata import AnnData  # type: ignore
from numpy.typing import NDArray

from .models import dpca_guide, dpca_model, scpca_guide, scpca_model
from .train import DEFAULT, FactorModel, SVILocalHandler
from .utils.data import _get_rna_counts


class scPCA(FactorModel):
    """
    Single-cell Principal Component Analysis (scPCA) model.

    This class provides an interface to perform scPCA on single-cell data. It allows for
    the extraction of principal components while accounting for batch effects and other
    covariates.

    Parameters
    ----------
    adata
        Anndata object containing the single-cell data.
    num_factors
        Number of factors to fit.
    layers_key
        Key to extract single-cell count matrix from adata.layers. If None, scPCA will
        try to extract the count matrix from adata.X. Default is None.
    loadings_formula
        R style formula to construct the loadings design matrix from adata.obs. If None, scPCA
        fits a normal PCA. Default is "1".
    intercept_formula
        R style formula to construct the intercept design matrix from adata.obs. Default is "1",
        which fits a single mean offset for each gene across all cells.
    size_factor
        Optional size factor information for cells. Default is None, if no size factor
        is given scPCA computes simply computes the log sum of counts for each cell.
    subsampling
        Number of cells to subsample for training. Default is 4096.
    device
        Device to run the model on. A GPU is recommended. Default is GPU if available, else CPU.
    seed
        Random seed for reproducibility. Default is None.
    model_kwargs
        Additional keyword arguments for the model. Default values are provided.
    training_kwargs
        Additional keyword arguments for training. Default is DEFAULT.
    """

    def __init__(
        self,
        adata: AnnData,
        num_factors: int,
        layers_key: Union[str, None] = None,
        loadings_formula: str = "1",
        intercept_formula: str = "1",
        size_factor: Optional[Union[str, NDArray[np.float32]]] = None,
        subsampling: int = 4096,
        device: Optional[Literal["cuda", "cpu"]] = None,
        seed: Optional[int] = None,
        model_kwargs: Dict[str, Any] = {},
        training_kwargs: Dict[str, Any] = DEFAULT,
    ):
        super().__init__(
            adata=adata,
            num_factors=num_factors,
            layers_key=layers_key,
            loadings_formula=loadings_formula,
            intercept_formula=intercept_formula,
            subsampling=subsampling,
            device=device,
            seed=seed,
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
        )

        # setup data
        self.size_factor = size_factor
        self.data = self._setup_data()
        self.handler = self._setup_handler()

    def _setup_data(self) -> Dict[str, Union[torch.Tensor, int, float]]:
        """
        Prepare the data for the scPCA model.

        Returns
        -------
            Dictionary containing tensors and other relevant information for the model.
        """
        X = _get_rna_counts(self.adata, self.layers_key)

        if isinstance(self.size_factor, str):
            X_size = self.adata.obs[self.size_factor].values.reshape(-1, 1).astype(np.float32)
        elif isinstance(self.size_factor, np.ndarray):
            if self.size_factor.ndim == 1:
                X_size = self.size_factor.reshape(-1, 1)
            else:
                X_size = self.size_factor
        else:
            X_size = np.log(X.sum(axis=1, keepdims=True))

        design: NDArray[np.float32] = np.asarray(self.loadings_states.encoding).astype(np.float32)
        design_idx = self.loadings_states.idx

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

    def posterior_to_anndata(
        self, model_key: str, num_samples: int = 25, variables: Sequence[str] = ["W", "Z"]
    ) -> None:
        """
        Store the posterior samples in the AnnData object.

        Parameters
        ----------
        model_key
            Key to store the model in the AnnData object.
        num_samples
            Number of samples to draw from the posterior. Default is 25.
        variables
            List of variables for which the posterior mean estimates should be stored.
            Possible values include "W", "V", "μ", "Z", "α", "σ", and "offset".
            Default is ["W", "Z"].
        """

        self._meta_to_anndata(model_key)
        adata = self.adata

        if self.handler is not None:
            for var in variables:
                if var == "W":
                    adata.varm[f"W_{model_key}"] = (
                        self.handler.predict_global_variable("W_lin", num_samples=num_samples)
                        .T.swapaxes(-1, -3)
                        .swapaxes(-1, -2)
                    )
                if var == "V":
                    adata.varm[f"V_{model_key}"] = self.handler.predict_global_variable(
                        "W_add", num_samples=num_samples
                    ).T.swapaxes(-1, -2)
                if var == "Z":
                    adata.obsm[f"X_{model_key}"] = self.handler.predict_local_variable(
                        "z", num_samples=num_samples
                    ).swapaxes(0, 1)

                if var == "α":
                    α_rna = self.handler.predict_global_variable("α_rna", num_samples=num_samples).T
                    if α_rna.ndim == 2:
                        α_rna = np.expand_dims(α_rna, 1)
                    adata.varm[f"{model_key}_α_rna"] = α_rna.swapaxes(-1, -2)

                if var == "σ":
                    σ_rna = self.handler.predict_global_variable("σ_rna", num_samples=num_samples).T
                    if σ_rna.ndim == 2:
                        σ_rna = np.expand_dims(σ_rna, 1)
                    adata.varm[f"{model_key}_σ_rna"] = σ_rna.swapaxes(-1, -2)

    def mean_to_anndata(
        self, model_key: str, num_samples: int = 25, num_split: int = 2048, variables: Sequence[str] = ["W", "Z"]
    ) -> None:
        """
        Store the posterior mean estimates in the AnnData object for specified variables.

        This method retrieves the posterior mean estimates for the given variables and stores them in the AnnData object.
        The variables can include weights ("W"), loadings ("V"), means ("μ"), latent factors ("Z"), among others.

        Parameters
        ----------
        model_key
            Key to store the model results in the AnnData object.
        num_samples
            Number of samples to draw from the posterior. Default is 25.
        num_split
            Number of splits for the data. Default is 2048.
        variables
            List of variables for which the posterior mean estimates should be stored.
            Possible values include "W", "V", "μ", "Z", "α", "σ", and "offset".
            Default is ["W", "Z"].

        Returns
        -------
            The results are stored in the provided AnnData object.
        """
        self._meta_to_anndata(model_key)
        adata = self.adata
        if self.handler is not None:
            # sort factors
            if "Z" in variables or "W" in variables:
                z = self.handler.predict_local_variable(
                    "z", num_samples=num_samples, num_split=num_split, return_mean=True
                )
                idx = np.argsort(z.var(0))[::-1]

            for var in variables:
                if var == "Z":
                    adata.obsm[f"X_{model_key}"] = z[..., idx]

                if var == "W":
                    W = self.handler.predict_global_variable("W_lin", num_samples=num_samples, return_mean=True).T
                    adata.varm[f"W_{model_key}"] = W[:, idx, :]

                if var == "V":
                    adata.varm[f"V_{model_key}"] = self.handler.predict_global_variable(
                        "W_add", num_samples=num_samples, return_mean=True
                    ).T
                if var == "μ":
                    adata.layers[f"μ_{model_key}"] = self.handler.predict_local_variable(
                        "μ_rna", num_samples=num_samples, num_split=num_split, return_mean=True
                    )

                if var == "α":
                    adata.varm[f"α_{model_key}"] = self.handler.predict_global_variable(
                        "α_rna", num_samples=num_samples, return_mean=True
                    ).T

                if var == "σ":
                    adata.varm[f"σ_{model_key}"] = self.handler.predict_global_variable(
                        "σ_rna", num_samples=num_samples, return_mean=True
                    ).T

                if var == "offset":
                    adata.layers[f"offset_{model_key}"] = self.handler.predict_local_variable(
                        "offset_rna", num_samples=num_samples, num_split=num_split, return_mean=True
                    )


class dPCA(FactorModel):
    """
    Design Principal Component Analysis (dPCA) model.

    Parameters
    ----------
    adata
        Anndata object containing the data to analyse.
    num_factors
        Number of factors to fit.
    layers_key
        Key to extract data matrix from adata.layers. If None, dPCA will
        try to extract the matrix from adata.X. Default is None.
    loadings_formula
        R style formula to construct the loadings design matrix from adata.obs. If None, dPCA
        fits a normal PCA/factor model. Default is '1'.
    batch_formula
        R style formula to extract intercept design maxtrix from adata.obs. If None, dPCA
        assumes a single batch. Default is '1'.
    subsampling
        Number of obs to subsample for training. Default is 4096.
    device
        Device to run the model on. A GPU is recommended. Default is GPU if available, else CPU.
    model_kwargs
        Additional keyword arguments for the model. Default values are provided.
    training_kwargs
        Additional keyword arguments for training. Default is SUBSAMPLE.
    """

    def __init__(
        self,
        adata: AnnData,
        num_factors: int,
        layers_key: Union[str, None] = None,
        loadings_formula: str = "1",
        intercept_formula: str = "1",
        subsampling: int = 4096,
        device: Optional[Literal["cuda", "cpu"]] = None,
        seed: Optional[int] = None,
        model_kwargs: Dict[str, Any] = {
            "z_sd": 1.0,
        },
        training_kwargs: Dict[str, Any] = DEFAULT,
    ):
        super().__init__(
            adata=adata,
            num_factors=num_factors,
            layers_key=layers_key,
            loadings_formula=loadings_formula,
            intercept_formula=intercept_formula,
            subsampling=subsampling,
            device=device,
            seed=seed,
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
        )

        # setup data
        self.data = self._setup_data()
        self.handler = self._setup_handler()

    def _setup_data(self) -> Dict[str, Union[torch.Tensor, int, float]]:
        """
        Sets up the data.
        """
        X = _get_rna_counts(self.adata, self.layers_key)
        loading_design: NDArray[np.float32] = np.asarray(self.loadings_states.encoding).astype(np.float32)
        loading_idx = self.loadings_states.idx
        intercept_design: NDArray[np.float32] = np.asarray(self.intercept_states.encoding).astype(np.float32)
        intercept_idx = self.intercept_states.idx

        num_obs = X.shape[0]
        idx = np.arange(num_obs)

        data = dict(
            X=X,
            intercept_design=intercept_design,
            loading_design=loading_design,
            loading_idx=loading_idx,
            intercept_idx=intercept_idx,
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

    def posterior_to_anndata(
        self, model_key: str, num_samples: int = 25, variables: Sequence[str] = ["W", "Z"]
    ) -> None:
        """
        Store the posterior samples in the AnnData object.

        Parameters
        ----------
        model_key
            Key to store the model in the AnnData object.
        num_samples
            Number of samples to draw from the posterior. Default is 25.
        variables
            List of variables for which the posterior mean estimates should be stored.
            Possible values include "W", "V", "μ", "Z", "α", "σ", and "offset".
            Default is ["W", "Z"].
        """

        self._meta_to_anndata(model_key)
        adata = self.adata
        if self.handler is not None:
            for var in variables:
                if var == "W":
                    adata.varm[f"W_{model_key}"] = (
                        self.handler.predict_global_variable("W_lin", num_samples=num_samples)
                        .T.swapaxes(-1, -3)
                        .swapaxes(-1, -2)
                    )
                if var == "V":
                    adata.varm[f"V_{model_key}"] = self.handler.predict_global_variable(
                        "W_add", num_samples=num_samples
                    ).T.swapaxes(-1, -2)
                if var == "Z":
                    adata.obsm[f"X_{model_key}"] = self.handler.predict_local_variable(
                        "z", num_samples=num_samples
                    ).swapaxes(0, 1)

                if var == "α":
                    α_rna = self.handler.predict_global_variable("α_rna", num_samples=num_samples).T
                    if α_rna.ndim == 2:
                        α_rna = np.expand_dims(α_rna, 1)
                    adata.varm[f"{model_key}_α_rna"] = α_rna.swapaxes(-1, -2)

                if var == "σ":
                    σ_rna = self.handler.predict_global_variable("σ_rna", num_samples=num_samples).T
                    if σ_rna.ndim == 2:
                        σ_rna = np.expand_dims(σ_rna, 1)
                    adata.varm[f"{model_key}_σ_rna"] = σ_rna.swapaxes(-1, -2)

    def mean_to_anndata(
        self, model_key: str, num_samples: int = 25, num_split: int = 2048, variables: Sequence[str] = ["W", "Z"]
    ) -> None:
        """
        Store the posterior mean estimates in the AnnData object for specified variables.

        This method retrieves the posterior mean estimates for the given variables and stores them in the AnnData object.
        The variables can include weights ("W"), loadings ("V"), means ("μ"), latent factors ("Z"), among others.

        Parameters
        ----------
        model_key
            Key to store the model results in the AnnData object.
        num_samples
            Number of samples to draw from the posterior. Default is 25.
        num_split
            Number of splits for the data. Default is 2048.
        variables
            List of variables for which the posterior mean estimates should be stored.
            Possible values include "W", "V", "μ", "Z", "α", "σ", and "offset".
            Default is ["W", "Z"].

        Returns
        -------
            The results are stored in the provided AnnData object.
        """
        self._meta_to_anndata(model_key)
        adata = self.adata
        if self.handler is not None:
            for var in variables:
                if var == "W":
                    adata.varm[f"W_{model_key}"] = self.handler.predict_global_variable(
                        "W_lin", num_samples=num_samples, return_mean=True
                    ).T
                if var == "V":
                    adata.varm[f"V_{model_key}"] = self.handler.predict_global_variable(
                        "W_add", num_samples=num_samples, return_mean=True
                    ).T
                if var == "μ":
                    adata.layers[f"μ_{model_key}"] = self.handler.predict_local_variable(
                        "μ_rna", num_samples=num_samples, num_split=num_split, return_mean=True
                    )
                if var == "Z":
                    adata.obsm[f"X_{model_key}"] = self.handler.predict_local_variable(
                        "z", num_samples=num_samples, num_split=num_split, return_mean=True
                    )
                if var == "σ":
                    adata.varm[f"σ_{model_key}"] = self.handler.predict_global_variable(
                        "σ", num_samples=num_samples, return_mean=True
                    ).T

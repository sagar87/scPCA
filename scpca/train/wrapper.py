from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch
from anndata import AnnData  # type: ignore
from patsy import dmatrix  # type: ignore
from torch.types import Device

from ..utils.design import _get_states
from .local_handler import SVILocalHandler
from .settings import DEFAULT


class FactorModel:
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
        model_kwargs: Dict[str, Any] = {},
        training_kwargs: Dict[str, Any] = DEFAULT,
    ):
        self.adata = adata
        self.num_factors = num_factors
        self.layers_key = layers_key
        self.loadings_formula = loadings_formula
        self.intercept_formula = intercept_formula
        self.subsampling = min([subsampling, adata.shape[0]])
        self.device = self._set_device(device)
        self.seed = seed
        self._set_seed(seed)

        # prepare design and batch matrix
        self.loadings_matrix = dmatrix(loadings_formula, self.adata.obs)
        self.loadings_states = _get_states(self.loadings_matrix)
        self.intercept_matrix = dmatrix(intercept_formula, self.adata.obs)
        self.intercept_states = _get_states(self.intercept_matrix)
        #
        self.model_kwargs = model_kwargs
        self.training_kwargs = training_kwargs
        self.handler: Optional[SVILocalHandler] = None

    def _set_device(self, device: Optional[Literal["cuda", "cpu"]] = None) -> Device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

    def _set_seed(self, seed: Optional[int] = None) -> Optional[torch.Generator]:
        return seed if seed is None else torch.manual_seed(seed)

    def _to_torch(self, data: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, int, float]]:
        """
        Convert numpy arrays of a dictionary to torch tensors.

        Parameters
        ----------
        data
            Dictionary containing numpy arrays.

        Returns
        -------
            Dictionary with numpy arrays converted to torch tensors.
        """
        return {k: torch.tensor(v, device=self.device) if isinstance(v, np.ndarray) else v for k, v in data.items()}

    def fit(
        self, num_epochs: Optional[int] = None, lr: Optional[float] = None, *args: torch.Tensor, **kwargs: torch.Tensor
    ) -> None:
        if self.handler is not None:
            self.handler.fit(num_epochs, lr, *args, **kwargs)

    def _setup_data(self) -> Dict[str, Union[torch.Tensor, int, float]]:
        raise NotImplementedError()

    def _setup_handler(self) -> SVILocalHandler:
        raise NotImplementedError()

    def _meta_to_anndata(self, model_key: str) -> None:
        """
        Store meta information in the AnnData object.

        Parameters
        ----------
        model_key
            Key to store the model in the AnnData object.
        """
        res: Dict[str, Any] = {}
        res["loadings_states"] = self.loadings_states.sparse
        res["intercept_states"] = self.intercept_states.sparse

        res["loadings_index"] = self.loadings_states.idx
        res["intercept_index"] = self.intercept_states.idx

        res["loadings_formula"] = self.loadings_formula
        res["intercept_formula"] = self.intercept_formula

        res["model"] = {"num_factors": self.num_factors, "seed": self.seed, **self.model_kwargs}

        if self.handler is not None:
            res["loss"] = self.handler.loss

        self.adata.uns[f"{model_key}"] = res

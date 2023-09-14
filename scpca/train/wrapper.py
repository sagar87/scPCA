from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import torch
from anndata import AnnData  # type: ignore
from patsy import dmatrix  # type: ignore
from torch.types import Device

from ..utils import get_states
from .settings import DEFAULT


class ModelWrapper:
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
        training_kwargs: Dict[str, Any] = DEFAULT,
    ):
        self.adata = adata
        self.num_factors = num_factors
        self.layers_key = layers_key
        self.design_formula = design_formula
        self.intercept_formula = intercept_formula
        self.subsampling = min([subsampling, adata.shape[0]])
        self.device = self._set_device(device)
        self.seed = self._set_seed(seed)

        # prepare design and batch matrix
        self.design_matrix = dmatrix(design_formula, self.adata.obs)
        self.design_states = get_states(self.design_matrix)
        self.intercept_matrix = dmatrix(intercept_formula, self.adata.obs)
        self.intercept_states = get_states(self.intercept_matrix)
        #
        self.model_kwargs = model_kwargs
        self.training_kwargs = training_kwargs

    def _set_device(self, device: Optional[Literal["cuda", "cpu"]] = None) -> Device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

    def _set_seed(self, seed: Optional[int] = None) -> Optional[torch.Generator]:
        return seed if seed is None else torch.manual_seed(self.seed)

    def _to_torch(self, data: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, int, float]]:
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

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from anndata import AnnData  # type: ignore
from numpy.typing import NDArray
from scipy.sparse import issparse  # type: ignore

from ..logger import logger


def _get_rna_counts(adata: AnnData, layers_key: Optional[str] = None) -> NDArray[np.float32]:
    """
    Extracts RNA counts from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing RNA counts.
    layers_key : str or None, optional (default: None)
        Key to specify the layer from which to extract the counts.
        If `None`, the raw counts are extracted from `adata.X`.
        If a valid `layers_key` is provided, the counts are extracted from `adata.layers[layers_key]`.

    Returns
    -------
    X : np.ndarray
        RNA counts matrix as a NumPy array.

    Raises
    ------
    KeyError
        If `layers_key` is provided and not found in `adata.layers`.

    Notes
    -----
    - If `layers_key` is `None`, the function extracts the counts from the attribute `adata.X`,
      which is assumed to contain the raw counts.
    - If `layers_key` is provided, the function checks if it exists in `adata.layers`.
      If found, the counts are extracted from `adata.layers[layers_key]`.
      If not found, a `KeyError` is raised.

    The function first checks if the counts are stored as a sparse matrix (`issparse(X)`).
    If so, the sparse matrix is converted to a dense array using `X.toarray()`.

    Finally, the counts are returned as a NumPy array with the data type set to `np.float32`.
    """
    if layers_key is None:
        X = adata.X
    else:
        if layers_key not in adata.layers:
            raise KeyError("Spefied layers_key was not found in the AnnData object.")
        X = adata.layers[layers_key]

    if issparse(X):
        X = X.toarray()

    return np.asarray(X).astype(np.float32)


DESIGN_KEY = "design"


def _get_model_design(adata: AnnData, model_key: str, reverse: bool = False) -> Any:
    """
    Extracts the design dictionary from an AnnData object.

    This function retrieves the design dictionary associated with a specific model from the `uns` attribute of an AnnData
    object. The `uns` attribute is a dictionary-like storage for unstructured annotation data.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the model annotations.
    model_key : str
        The key identifying the model in the `uns` attribute.
    reverse : bool
        Whether to reverse the key/items in the returned dict.

    Returns
    -------
    dict
        The design dictionary associated with the specified model.

    Raises
    ------
    ValueError
        If the model key is not found in the `uns` attribute of the AnnData object.
    ValueError
        If the design mapping is not found in the model annotations.

    Example
    -------
    >>> adata = AnnData()
    >>> model_key = "my_model"
    >>> design_mapping = {"Intercept": 0, "stim": 1}
    >>> adata.uns[model_key] = {"design": design_mapping}
    >>> result = _get_model_design(adata, model_key)
    >>> print(result)
    {"Intercept": 0, "stim": 1}
    """

    if model_key not in adata.uns:
        raise ValueError(f"No model with the key {model_key} found.")

    model_dict = adata.uns[model_key]

    if DESIGN_KEY not in model_dict:
        raise ValueError("No design mapping found in model annotations.")

    model_design = model_dict[DESIGN_KEY]

    if reverse:
        model_design = {v: k for k, v in model_design.items()}

    return model_design


def _validate_sign(sign: Union[float, int]) -> Union[float, int]:
    """
    Validates if the provided sign is either 1.0 or -1.0.

    Parameters
    ----------
    sign :
        The value to validate.

    Returns
    -------
    Union[float, int]
        The validated sign.

    Raises
    ------
    TypeError
        If the sign is not of type float or int.
    ValueError
        If the absolute value of the sign is not 1.0.
    """
    if not isinstance(sign, (float, int)):
        raise TypeError("Sign must either be of float or integer type.")

    if np.abs(sign) != 1.0:
        raise ValueError("Sign must be either 1 or -1.")

    return sign


def _validate_states(states: Union[List[str], Tuple[str, str], str]) -> Tuple[str, str]:
    """
    Retrieve state indices from the model dictionary based on the provided states.

    Parameters
    ----------
    model_dict :
        Dictionary containing state mappings.
    states :
        Either a single state as a string or a list containing two states.

    Returns
    -------
    tuple
        A tuple containing two state names and their corresponding indices.

    Raises
    ------
    ValueError
        If the length of provided states in the list is not equal to 2.
    TypeError
        If the type of states is neither a list nor a string.
    """

    if isinstance(states, (list, tuple)):
        if len(states) != 2:
            raise ValueError("The length of provided states must equal 2.")
        state_a, state_b = states

    elif isinstance(states, str):
        logger.info("Only one state was provided, using 'Intercept' as base state.")
        state_a, state_b = "Intercept", states
    else:
        raise TypeError("The 'states' parameter must be either a list or a string.")

    return state_a, state_b

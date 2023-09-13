from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
from anndata import AnnData  # type: ignore
from numpy.typing import NDArray
from scipy.sparse import issparse  # type: ignore

from ..logger import logger

DESIGN_KEY = "design"


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


def _get_model_design(adata: AnnData, model_key: str, reverse: bool = False) -> Any:
    """
    Extracts the design dictionary from an AnnData object.

    This function retrieves the design dictionary associated with a specific model from the `uns` attribute of an AnnData
    object. The `uns` attribute is a dictionary-like storage for unstructured annotation data.

    Parameters
    ----------
    adata :
        The AnnData object containing the model annotations.
    model_key :
        The key identifying the model in the `uns` attribute.
    reverse :
        Whether to reverse the key/items in the returned dict.

    Returns
    -------
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


def _get_gene_idx(array: NDArray[np.float32], highest: int, lowest: int) -> NDArray[np.int64]:
    """
    Given an array of indices return the highest and/or lowest
    indices.

    Parameters
    ----------
    array: np.ndarray
        array in which to extract the highest/lowest indices
    highest: int
        number of top indices to extract
    lowest: int
        number of lowest indices to extract

    Returns
    -------
    np.ndarray
    """
    order = np.argsort(array)

    if highest == 0:
        gene_idx = order[:lowest]
    else:
        gene_idx = np.concatenate([order[:lowest], order[-highest:]])

    return gene_idx


def get_ordered_genes(
    adata: AnnData,
    model_key: str,
    state: str,
    factor: int,
    sign: Union[int, float] = 1.0,
    vector: str = "W_rna",
    highest: int = 10,
    lowest: int = 0,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Retrieve the ordered genes based on differential factor values.

    Parameters
    ----------
    adata :
        Annotated data object containing gene expression data.
    model_key :
        Key to identify the specific model.
    state :
        Name of the model state from which to extract genes.
    factor :
        Factor index for which differential factor values are calculated.
    sign :
        Sign multiplier for differential factor values. Default is 1.0.
    vector :
        Vector type from which to extract differential factor values. Default is "W_rna".
    highest :
        Number of genes with the highest differential factor values to retrieve. Default is 10.
    lowest :
        Number of genes with the lowest differential factor values to retrieve. Default is 0.
    ascending :
        Flag indicating whether to sort genes in ascending order based on differential factor values.
        Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the ordered genes along with their magnitude, differential factor values,
        type (lowest/highest), model state, factor index, and gene index.

    Raises
    ------
    ValueError
        If the specified model key or model state is not found in the provided AnnData object.
    """
    _ = _validate_sign(sign)
    model_design = _get_model_design(adata, model_key)
    state = model_design[state]
    weights = sign * adata.varm[f"{model_key}_{vector}"][..., factor, state]
    gene_idx = _get_gene_idx(weights, highest, lowest)

    magnitude = np.abs(weights[gene_idx])
    genes = adata.var_names.to_numpy()[gene_idx]

    return (
        pd.DataFrame(
            {
                "gene": genes,
                "magnitude": magnitude,
                "diff": weights[gene_idx],
                "type": ["lowest"] * lowest + ["highest"] * highest,
                "state": state,
                "factor": factor,
                "index": gene_idx,
            }
        )
        .sort_values(by="diff", ascending=ascending)
        .reset_index(drop=True)
        .rename(columns={"diff": "value"})
    )

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
import scanpy as sc  # type: ignore
from anndata import AnnData  # type: ignore
from numpy.typing import NDArray
from scipy.sparse import issparse  # type: ignore

from ..logger import logger

DESIGN_KEY = "loadings_states"


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
    array :
        array in which to extract the highest/lowest indices
    highest :
        number of top indices to extract
    lowest :
        number of lowest indices to extract

    Returns
    -------
    """
    order = np.argsort(array)

    if highest == 0:
        gene_idx = order[:lowest]
    else:
        gene_idx = np.concatenate([order[:lowest], order[-highest:]])

    return gene_idx


def state_loadings(
    adata: AnnData,
    model_key: str,
    state: Union[List[str], Tuple[str, str], str],
    factor: int,
    sign: Union[int, float] = 1.0,
    variable: str = "W",
    highest: int = 10,
    lowest: int = 0,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Extract and order genes from an AnnData object based on their loading weights.

    This function retrieves genes based on their association with a specific loading weights and state
    from a given model. It allows for the selection of genes with the highest and lowest
    loading weight values, and returns them in a sorted DataFrame.

    Parameters
    ----------
    adata :
        The annotated data matrix containing gene expression data.
    model_key :
        The key corresponding to the model in the AnnData object.
    state :
        The state from which to extract gene information.
    factor :
        The index of the factor based on which genes are ordered.
    sign :
        Multiplier to adjust the direction of factor values.
    variable :
        The type of vector from which factor values are extracted.
    highest :
        The number of top genes with the highest factor values to retrieve.
    lowest :
        The number of genes with the lowest factor values to retrieve.
    ascending :
        Whether to sort the genes in ascending order of factor values.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing genes ordered by their factor values. Columns include gene name,
        magnitude of association, weight, type (highest/lowest), state, factor index, and gene index.

    Raises
    ------
    ValueError
        If the provided model key or state is not present in the AnnData object.
    """
    _ = _validate_sign(sign)
    model_design = _get_model_design(adata, model_key)
    state = model_design[state]
    weights = sign * adata.varm[f"{variable}_{model_key}"][..., factor, state]
    gene_idx = _get_gene_idx(weights, highest, lowest)

    magnitude = np.abs(weights[gene_idx])
    genes = adata.var_names.to_numpy()[gene_idx]

    return (
        pd.DataFrame(
            {
                "gene": genes,
                "magnitude": magnitude,
                "weight": weights[gene_idx],
                "type": ["lowest"] * lowest + ["highest"] * highest,
                "state": state,
                "factor": factor,
                "index": gene_idx,
            }
        )
        .sort_values(by="weight", ascending=ascending)
        .reset_index(drop=True)
    )


def state_diff(
    adata: AnnData,
    model_key: str,
    states: Union[List[str], Tuple[str, str], str],
    factor: int,
    sign: Union[int, float] = 1.0,
    variable: str = "W",
    highest: int = 10,
    lowest: int = 0,
    ascending: bool = False,
    threshold: float = 1.96,
) -> pd.DataFrame:
    """
    Compute the dfference between two state loadings (logits) of a specified factor.

    Parameters
    ----------
    adata :
        Annotated data matrix.
    model_key :
        Key to access the model in the adata object.
    states :
        List containing two states for comparison. If a single `str` is provided
        the base state is assumed to be 'Intercept'.
    factor :
        Factor index to consider for the diff calculation.
    sign :
        Sign to adjust the difference, either -1.0 or 1.0, by default 1.0.
    variable :
        Vector key to access in the model, by default "W".
    highest :
        Number of highest diff genes to retrieve, by default 10.
    lowest :
        Number of lowest diff genes to retrieve, by default 0.
    ascending :
        Whether to sort the results in ascending order, by default `False`.
    threshold :
        Threshold for significance, by default 1.96.

    Returns
    -------
    pd.DataFrame
        DataFrame containing differential genes, their magnitudes, differences, types, states, factors, indices, and significance.

    Notes
    -----
    This function computes the differential genes between two states based on a given model.
    It first validates the sign, retrieves the model design, and computes
    the difference between the two states for a given factor. The function then
    retrieves the gene indices based on the highest and lowest differences and
    constructs a DataFrame with the results.
    """

    sign = _validate_sign(sign)
    states = _validate_states(states)

    model_design = _get_model_design(adata, model_key)
    state_a = model_design[states[0]]
    state_b = model_design[states[1]]
    a = adata.varm[f"{variable}_{model_key}"][..., factor, state_a]
    b = adata.varm[f"{variable}_{model_key}"][..., factor, state_b]

    # diff_factor = sign * (model_dict[vector][state_b][factor] - model_dict[vector][state_a][factor])
    diff_factor = sign * (b - a)

    gene_idx = _get_gene_idx(diff_factor, highest, lowest)

    magnitude = np.abs(diff_factor[gene_idx])
    genes = adata.var_names.to_numpy()[gene_idx]
    # is_significant = lambda x: x > norm().ppf(1 - significance_level) or x < norm().ppf(significance_level)
    df = (
        pd.DataFrame(
            {
                "gene": genes,
                "magnitude": magnitude,
                "difference": diff_factor[gene_idx],
                "type": ["lowest"] * lowest + ["highest"] * highest,
                "state": states[1] + "-" + states[0],
                "factor": factor,
                "index": gene_idx,
                states[0]: sign * a[gene_idx],
                states[1]: sign * b[gene_idx],
            }
        )
        .sort_values(by="difference", ascending=ascending)
        .reset_index(drop=True)
    )

    df["significant"] = df["magnitude"] > threshold

    return df


def umap(
    adata: AnnData, model_key: str, neighbors_kwargs: Dict[str, Any] = {}, umap_kwargs: Dict[str, Any] = {}
) -> None:
    """
    Performs UMAP dimensionality reduction on an AnnData object. Uses scanpy's
    UMAP function but stores the nearest neighbors graph and UMAP coordinates in the
    `anndata` object with the a `model_key` prefix.

    Parameters
    ----------
    adata :
        The AnnData object containing the data to be processed.
    model_key :
        The basis to use for the UMAP calculation.
    neighbors_kwargs :
        Additional keyword arguments to be passed to `sc.pp.neighbors` function.
        Default is an empty dictionary.
    umap_kwargs :
        Additional keyword arguments to be passed to `sc.tl.umap` function.
        Default is an empty dictionary.

    Returns
    -------
    None

    Notes
    -----
    This function performs UMAP dimensionality reduction on the input `adata` object
    using the specified embedding of the specified model. It first computes the neighbors graph using the
    `sc.pp.neighbors` function, with the option to provide additional keyword arguments
    via `neighbors_kwargs`. Then, it applies the UMAP algorithm using the `sc.tl.umap`
    function, with the option to provide additional keyword arguments via `umap_kwargs`.
    Finally, it stores the UMAP coordinates in the `obsm` attribute of the `adata` object
    under the key `"{model_key}_umap"` or `"X_{model_key}_umap"` respectively.

    Example
    -------
    >>> adata = AnnData(X)
    >>> umap(adata, model_key="pca", neighbors_kwargs={"n_neighbors": 10}, umap_kwargs={"min_dist": 0.5})

    """
    if f"X_{model_key}" in adata.obsm:
        model_key = f"X_{model_key}"
    elif model_key in adata.obsm:
        pass
    else:
        raise KeyError("Neither f'X_{model_key}' nor '{model_key}' in adata.obsm.")

    sc.pp.neighbors(adata, use_rep=f"{model_key}", key_added=f"{model_key}", **neighbors_kwargs)
    sc.tl.umap(adata, neighbors_key=f"{model_key}", **umap_kwargs)
    adata.obsm[f"{model_key}_umap"] = adata.obsm["X_umap"]
    del adata.obsm["X_umap"]

from typing import Optional

import numpy as np
from anndata import AnnData  # type: ignore
from numpy.types import NDArray  # type: ignore
from scipy.sparse import issparse  # type: ignore


def get_rna_counts(adata: AnnData, layers_key: Optional[str] = None) -> NDArray[np.float32]:
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

    return X.astype(np.float32)

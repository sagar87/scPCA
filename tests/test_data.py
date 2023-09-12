import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from scpca.utils.data import get_rna_counts


def test_get_rna_counts_with_X():
    # Test when layers_key is None, extracting from adata.X
    adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6]]))
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    result = get_rna_counts(adata)
    assert np.array_equal(result, expected)


def test_get_rna_counts_with_layers():
    # Test when layers_key is provided, extracting from adata.layers[layers_key]
    adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6]]), layers={"counts": np.array([[7, 8, 9], [10, 11, 12]])})
    expected = np.array([[7, 8, 9], [10, 11, 12]])
    result = get_rna_counts(adata, layers_key="counts")
    assert np.array_equal(result, expected)


def test_get_rna_counts_with_sparse_matrix():
    # Test when layers_key is provided, and X is a sparse matrix
    adata = AnnData(X=csr_matrix([[1, 0, 3], [0, 5, 0]]), layers={"counts": csr_matrix([[7, 0, 9], [0, 11, 0]])})
    expected = np.array([[7, 0, 9], [0, 11, 0]])
    result = get_rna_counts(adata, layers_key="counts")
    assert np.array_equal(result, expected)


def test_get_rna_counts_return_type():
    # Test the return type of the function
    adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6]]))
    result = get_rna_counts(adata)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_get_rna_counts_with_invalid_layers_key():
    # Test when invalid layers_key is provided
    adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6]]), layers={"counts": np.array([[7, 8, 9], [10, 11, 12]])})
    with pytest.raises(KeyError):
        get_rna_counts(adata, layers_key="invalid_key")

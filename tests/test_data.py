import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from scpca.utils.data import _get_rna_counts, _validate_sign, _validate_states


# validation functions
def test_valid_signs():
    assert _validate_sign(1) == 1
    assert _validate_sign(-1) == -1
    assert _validate_sign(1.0) == 1.0
    assert _validate_sign(-1.0) == -1.0


def test_invalid_signs():
    with pytest.raises(ValueError):
        _validate_sign(2)
    with pytest.raises(ValueError):
        _validate_sign(-2)
    with pytest.raises(ValueError):
        _validate_sign(0)


def test_invalid_types():
    with pytest.raises(TypeError):
        _validate_sign("1")
    with pytest.raises(TypeError):
        _validate_sign([1])
    with pytest.raises(TypeError):
        _validate_sign((1,))


def test_validate_states_with_list():
    state_a, state_b = _validate_states(["state1", "state2"])
    assert state_a == "state1"
    assert state_b == "state2"


def test_validate_states_with_tuple():
    state_a, state_b = _validate_states(("state1", "state2"))
    assert state_a == "state1"
    assert state_b == "state2"


def test_validate_states_with_single_state():
    state_a, state_b = _validate_states("state1")
    assert state_a == "Intercept"
    assert state_b == "state1"


def test_validate_states_with_invalid_list_length():
    with pytest.raises(ValueError, match="The length of provided states must equal 2."):
        _validate_states(["state1"])


def test_validate_states_with_invalid_type():
    with pytest.raises(TypeError, match="The 'states' parameter must be either a list or a string."):
        _validate_states(123)

    with pytest.raises(TypeError, match="The 'states' parameter must be either a list or a string."):
        _validate_states({"state1": "index1"})


def test_get_rna_counts_with_X():
    # Test when layers_key is None, extracting from adata.X
    adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6]]))
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    result = _get_rna_counts(adata)
    assert np.array_equal(result, expected)


def test_get_rna_counts_with_layers():
    # Test when layers_key is provided, extracting from adata.layers[layers_key]
    adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6]]), layers={"counts": np.array([[7, 8, 9], [10, 11, 12]])})
    expected = np.array([[7, 8, 9], [10, 11, 12]])
    result = _get_rna_counts(adata, layers_key="counts")
    assert np.array_equal(result, expected)


def test_get_rna_counts_with_sparse_matrix():
    # Test when layers_key is provided, and X is a sparse matrix
    adata = AnnData(X=csr_matrix([[1, 0, 3], [0, 5, 0]]), layers={"counts": csr_matrix([[7, 0, 9], [0, 11, 0]])})
    expected = np.array([[7, 0, 9], [0, 11, 0]])
    result = _get_rna_counts(adata, layers_key="counts")
    assert np.array_equal(result, expected)


def test_get_rna_counts_return_type():
    # Test the return type of the function
    adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6]]))
    result = _get_rna_counts(adata)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_get_rna_counts_with_invalid_layers_key():
    # Test when invalid layers_key is provided
    adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6]]), layers={"counts": np.array([[7, 8, 9], [10, 11, 12]])})
    with pytest.raises(KeyError):
        _get_rna_counts(adata, layers_key="invalid_key")

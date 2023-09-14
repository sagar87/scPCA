import numpy as np
from scipy.spatial.distance import cosine

from scpca.pca import dPCA
from scpca.train import TEST


def is_aligned(a, b, atol=0.001):
    """
    Checks if the vector is aligned, i.e. it has the same direction (cosine distance == 0)
    or is flipped by 180 degrees (cosine distance 2).
    """
    dist = cosine(a.squeeze(), b.squeeze())
    return np.isclose(0.0, dist, atol=atol) or np.isclose(2.0, dist, atol=atol)


def is_close_to_zero_vec(vec, atol=0.1):
    vec = vec.squeeze()
    true = np.zeros_like(vec)
    return np.allclose(vec, true, atol=atol)


def test_dpca_two_states(two_state_data):
    adata = two_state_data
    m = dPCA(adata, 1, design_formula="state", training_kwargs=TEST)
    m.fit(lr=0.01)
    m.fit(lr=0.001)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = adata.uns["m"]["design"]

    W = adata.varm["W_m"]
    assert is_aligned(adata.uns["true_axes"]["A"], W[..., design["Intercept"]])
    assert is_aligned(adata.uns["true_axes"]["B"], W[..., design["state[T.B]"]])
    assert is_close_to_zero_vec(adata.varm["V_m"])


def test_dpca_four_state(four_state_data):
    adata = four_state_data
    m = dPCA(adata, 1, design_formula="state", training_kwargs=TEST)
    m.fit(lr=0.01)
    m.fit(lr=0.001)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = adata.uns["m"]["design"]

    W = adata.varm["W_m"]
    assert is_aligned(adata.uns["true_axes"]["A"], W[..., design["Intercept"]])
    assert is_aligned(adata.uns["true_axes"]["B"], W[..., design["state[T.B]"]])
    assert is_aligned(adata.uns["true_axes"]["C"], W[..., design["state[T.C]"]])
    assert is_aligned(adata.uns["true_axes"]["D"], W[..., design["state[T.D]"]])

    assert is_close_to_zero_vec(adata.varm["V_m"])

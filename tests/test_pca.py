import numpy as np
from scipy.spatial.distance import cosine

from scpca.pca import dPCA, scPCA
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


def test_dpca_two_states(one_factorial_two_state_normal_data):
    adata = one_factorial_two_state_normal_data
    m = dPCA(adata, 1, design_formula="state", training_kwargs=TEST)
    m.fit(lr=0.01)
    m.fit(lr=0.001)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = adata.uns["m"]["design"]

    W = adata.varm["W_m"]
    assert is_aligned(adata.uns["true_axes"]["A"], W[..., design["Intercept"]])
    assert is_aligned(adata.uns["true_axes"]["B"], W[..., design["state[T.B]"]])
    assert is_close_to_zero_vec(adata.varm["V_m"])


def test_dpca_two_states_with_offset(one_factorial_two_state_normal_data_with_offset):
    adata = one_factorial_two_state_normal_data_with_offset
    m = dPCA(adata, 1, design_formula="state", intercept_formula="state-1", training_kwargs=TEST)
    m.fit(lr=0.01)
    m.fit(lr=0.001)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = adata.uns["m"]["design"]

    W = adata.varm["W_m"]
    assert is_aligned(adata.uns["true_axes"]["A"], W[..., design["Intercept"]])
    assert is_aligned(adata.uns["true_axes"]["B"], W[..., design["state[T.B]"]])
    assert np.allclose(adata.uns["true_offset"]["A"], adata.varm["V_m"][0], atol=0.3)
    assert np.allclose(adata.uns["true_offset"]["B"], adata.varm["V_m"][1], atol=0.3)


def test_dpca_four_state(one_factorial_four_state_normal_data):
    adata = one_factorial_four_state_normal_data
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


def test_scpca_two_states(one_factorial_two_state_poisson_data):
    adata = one_factorial_two_state_poisson_data
    m = scPCA(
        adata, 1, design_formula="state", size_factor="size_factor", training_kwargs=TEST, model_kwargs={"z_sd": 1.0}
    )
    m.fit(lr=0.01)
    m.fit(lr=0.001)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = adata.uns["m"]["design"]

    W = adata.varm["W_m"]
    assert is_aligned(adata.uns["true_axes"]["A"], W[..., design["Intercept"]])
    assert is_aligned(adata.uns["true_axes"]["B"], W[..., design["state[T.B]"]])
    assert is_close_to_zero_vec(adata.varm["V_m"])


def test_scpca_two_states_with_offset(one_factorial_two_state_poisson_data_with_offset):
    adata = one_factorial_two_state_poisson_data_with_offset
    m = scPCA(
        adata,
        1,
        design_formula="state",
        intercept_formula="state-1",
        size_factor="size_factor",
        training_kwargs=TEST,
        model_kwargs={"z_sd": 1.0},
    )
    m.fit(lr=0.01, num_epochs=500)
    m.fit(lr=0.001, num_epochs=500)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = adata.uns["m"]["design"]

    W = adata.varm["W_m"]
    assert is_aligned(adata.uns["true_axes"]["A"], W[..., design["Intercept"]])
    assert is_aligned(adata.uns["true_axes"]["B"], W[..., design["state[T.B]"]])
    assert np.allclose(adata.uns["true_offset"]["A"], adata.varm["V_m"][0], atol=0.3)
    assert np.allclose(adata.uns["true_offset"]["B"], adata.varm["V_m"][1], atol=0.3)


def test_scpca_four_state(one_factorial_four_state_poisson_data):
    adata = one_factorial_four_state_poisson_data
    m = scPCA(
        adata, 1, design_formula="state", size_factor="size_factor", training_kwargs=TEST, model_kwargs={"z_sd": 1.0}
    )
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

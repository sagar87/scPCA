import numpy as np
from scipy.spatial.distance import cosine

from scpca.pca import dPCA, scPCA
from scpca.train import DEFAULT, TEST
from scpca.utils.data import _get_model_design


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


def proj(A):
    return A @ np.linalg.inv(A.T @ A) @ A.T


def is_same_subspace(reference, inferred, atol=0.01):
    A = proj(inferred)
    return np.allclose(A @ reference, reference, atol=atol)


def test_dpca_two_states(one_factorial_two_state_normal_data):
    adata = one_factorial_two_state_normal_data
    m = dPCA(adata, 1, loadings_formula="state", training_kwargs=TEST)
    m.fit(lr=0.01)
    m.fit(lr=0.001)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = _get_model_design(adata, "m")

    W = adata.varm["W_m"]
    assert is_aligned(adata.uns["true_axes"]["A"], W[..., design["Intercept"]])
    assert is_aligned(adata.uns["true_axes"]["B"], W[..., design["state[T.B]"]])
    assert is_close_to_zero_vec(adata.varm["V_m"])


def test_dpca_two_states_with_offset(one_factorial_two_state_normal_data_with_offset):
    adata = one_factorial_two_state_normal_data_with_offset
    m = dPCA(adata, 1, loadings_formula="state", intercept_formula="state-1", training_kwargs=TEST)
    m.fit(lr=0.01)
    m.fit(lr=0.001)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = _get_model_design(adata, "m")

    W = adata.varm["W_m"]
    assert is_aligned(adata.uns["true_axes"]["A"], W[..., design["Intercept"]])
    assert is_aligned(adata.uns["true_axes"]["B"], W[..., design["state[T.B]"]])
    assert np.allclose(adata.uns["true_offset"]["A"], adata.varm["V_m"][0], atol=0.3)
    assert np.allclose(adata.uns["true_offset"]["B"], adata.varm["V_m"][1], atol=0.3)


def test_dpca_four_state(one_factorial_four_state_normal_data):
    adata = one_factorial_four_state_normal_data
    m = dPCA(adata, 1, loadings_formula="state", training_kwargs=TEST)
    m.fit(lr=0.01)
    m.fit(lr=0.001)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = _get_model_design(adata, "m")

    W = adata.varm["W_m"]
    assert is_aligned(adata.uns["true_axes"]["A"], W[..., design["Intercept"]])
    assert is_aligned(adata.uns["true_axes"]["B"], W[..., design["state[T.B]"]])
    assert is_aligned(adata.uns["true_axes"]["C"], W[..., design["state[T.C]"]])
    assert is_aligned(adata.uns["true_axes"]["D"], W[..., design["state[T.D]"]])

    assert is_close_to_zero_vec(adata.varm["V_m"])


def test_scpca_two_states(one_factorial_two_state_poisson_data):
    adata = one_factorial_two_state_poisson_data
    m = scPCA(
        adata, 1, loadings_formula="state", size_factor="size_factor", training_kwargs=TEST, model_kwargs={"z_sd": 1.0}
    )
    m.fit(lr=0.01)
    m.fit(lr=0.001)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = _get_model_design(adata, "m")

    W = adata.varm["W_m"]
    assert is_aligned(adata.uns["true_axes"]["A"], W[..., design["Intercept"]])
    assert is_aligned(adata.uns["true_axes"]["B"], W[..., design["state[T.B]"]])
    assert is_close_to_zero_vec(adata.varm["V_m"])


def test_scpca_two_states_with_offset(one_factorial_two_state_poisson_data_with_offset):
    adata = one_factorial_two_state_poisson_data_with_offset
    m = scPCA(
        adata,
        1,
        loadings_formula="state",
        intercept_formula="state-1",
        size_factor="size_factor",
        training_kwargs=TEST,
        model_kwargs={"z_sd": 1.0},
    )
    m.fit(lr=0.01, num_epochs=500)
    m.fit(lr=0.001, num_epochs=500)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = _get_model_design(adata, "m")

    W = adata.varm["W_m"]
    assert is_aligned(adata.uns["true_axes"]["A"], W[..., design["Intercept"]])
    assert is_aligned(adata.uns["true_axes"]["B"], W[..., design["state[T.B]"]])
    assert np.allclose(adata.uns["true_offset"]["A"], adata.varm["V_m"][0], atol=0.3)
    assert np.allclose(adata.uns["true_offset"]["B"], adata.varm["V_m"][1], atol=0.3)


def test_scpca_four_state(one_factorial_four_state_poisson_data):
    adata = one_factorial_four_state_poisson_data
    m = scPCA(
        adata, 1, loadings_formula="state", size_factor="size_factor", training_kwargs=TEST, model_kwargs={"z_sd": 1.0}
    )
    m.fit(lr=0.01)
    m.fit(lr=0.001)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = _get_model_design(adata, "m")

    W = adata.varm["W_m"]
    assert is_aligned(adata.uns["true_axes"]["A"], W[..., design["Intercept"]])
    assert is_aligned(adata.uns["true_axes"]["B"], W[..., design["state[T.B]"]])
    assert is_aligned(adata.uns["true_axes"]["C"], W[..., design["state[T.C]"]])
    assert is_aligned(adata.uns["true_axes"]["D"], W[..., design["state[T.D]"]])

    assert is_close_to_zero_vec(adata.varm["V_m"])


# 3 dimensional data


def test_dpca_two_states_3_dims(one_factorial_two_state_normal_three_dim_data):
    adata = one_factorial_two_state_normal_three_dim_data
    MODIFIED = {**DEFAULT}
    MODIFIED["num_epochs"] = 1000
    m = dPCA(adata, 2, loadings_formula="state", seed=353151, training_kwargs=MODIFIED)
    m.fit()
    m.fit(lr=0.01)
    m.fit(lr=0.001)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = _get_model_design(adata, "m")

    W = adata.varm["W_m"]
    # import pdb; pdb.set_trace()
    assert is_same_subspace(adata.uns["true_axes"]["A"].T, W[..., design["Intercept"]])
    assert is_same_subspace(adata.uns["true_axes"]["B"].T, W[..., design["state[T.B]"]])
    # assert is_close_to_zero_vec(adata.varm["V_m"])


def test_dpca_two_states_3_dims_with_offset(one_factorial_two_state_normal_three_dim_data_with_offset):
    adata = one_factorial_two_state_normal_three_dim_data_with_offset
    MODIFIED = {**DEFAULT}
    MODIFIED["num_epochs"] = 1000
    m = dPCA(adata, 2, loadings_formula="state", intercept_formula="state-1", seed=353151, training_kwargs=MODIFIED)
    m.fit()
    m.fit(lr=0.01)
    m.fit(lr=0.001)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = _get_model_design(adata, "m")

    W = adata.varm["W_m"]
    # import pdb; pdb.set_trace()
    assert np.allclose(adata.uns["true_offset"]["A"], adata.varm["V_m"][..., design["Intercept"]], atol=0.2)
    assert np.allclose(adata.uns["true_offset"]["B"], adata.varm["V_m"][..., design["state[T.B]"]], atol=0.2)
    assert is_same_subspace(adata.uns["true_axes"]["A"].T, W[..., design["Intercept"]], atol=0.1)
    assert is_same_subspace(adata.uns["true_axes"]["B"].T, W[..., design["state[T.B]"]], atol=0.1)


def test_dpca_two_states_3_dims_poisson(one_factorial_two_state_poisson_three_dim_data):
    adata = one_factorial_two_state_poisson_three_dim_data
    MODIFIED = {**DEFAULT}
    MODIFIED["num_epochs"] = 1000
    m = scPCA(adata, 2, loadings_formula="state", size_factor="size_factor", seed=65364, training_kwargs=MODIFIED)
    m.fit()
    m.fit(lr=0.01)
    m.fit(lr=0.001)
    m.mean_to_anndata("m", variables=["W", "V", "Z"])
    design = _get_model_design(adata, "m")

    W = adata.varm["W_m"]
    # import pdb; pdb.set_trace()
    assert is_same_subspace(adata.uns["true_axes"]["A"].T, W[..., design["Intercept"]])
    assert is_same_subspace(adata.uns["true_axes"]["B"].T, W[..., design["state[T.B]"]])
    # assert is_close_to_zero_vec(adata.varm["V_m"])


# def test_dpca_two_states_3_dims_poisson_with_offset(one_factorial_two_state_poisson_three_dim_data_with_offset):
#     adata = one_factorial_two_state_poisson_three_dim_data_with_offset
#     MODIFIED = {**DEFAULT}
#     MODIFIED["num_epochs"] = 1000
#     m = scPCA(adata, 2, loadings_formula="state", intercept_formula='state-1', training_kwargs=MODIFIED)
#     m.fit()
#     m.fit(lr=0.01)
#     m.fit(lr=0.001)
#     m.mean_to_anndata("m", variables=["W", "V", "Z"])
#     design = adata.uns["m"]["design"]

#     W = adata.varm["W_m"]
#     # import pdb; pdb.set_trace()
#     assert np.allclose(adata.uns['true_offset']['A'], adata.varm['V_m'][..., design["Intercept"]], atol=1.)
#     assert np.allclose(adata.uns['true_offset']['B'], adata.varm['V_m'][..., design["state[T.B]"]], atol=1.)

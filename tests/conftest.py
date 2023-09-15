import os
import string
from distutils import dir_util
from typing import NamedTuple, Optional

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from numpy.typing import NDArray
from scipy import stats
from scipy.sparse import csr_matrix


class SimulatedData(NamedTuple):
    W: NDArray[np.float32]
    Z: NDArray[np.float32]
    μ: NDArray[np.float32]
    X: NDArray[np.float32]
    size_factor: Optional[NDArray[np.float32]] = None


def principle_axis(rad):
    return np.array([np.cos(rad), np.sin(rad)]).reshape(1, -1)


def simulate_dataset(
    num_obs: int,
    num_dims: int,
    num_features: int,
    W: NDArray[np.float32] = None,
    z: NDArray[np.float32] = None,
    offset: NDArray[np.float32] = None,
    z_loc: float = 0.0,
    z_scale: float = 1.0,
    size_factor: float = 4.605170,
    σ: float = 1,
    noise: str = "normal",
) -> SimulatedData:
    if W is None:
        W = np.random.normal(0.0, 2.0, size=(num_dims, num_features))

    if offset is None:
        offset = np.zeros((1, num_features))

    if z is None:
        z = np.random.normal(z_loc, z_scale, size=(num_obs, num_dims))

    if noise == "normal":
        μ = np.dot(z, W) + offset
        X = stats.norm(loc=μ, scale=σ).rvs()
    if noise == "poisson":
        μ = size_factor + np.dot(z, W) + offset
        X = stats.poisson(np.exp(μ)).rvs()

    return SimulatedData(W, z, μ, X, size_factor)


def simulate_2d_data(angles=[np.pi / 8 * 1, np.pi / 8 * 3], num_obs=100, σ=0.1):
    data = []
    states = []

    for angle, state in zip(angles, string.ascii_uppercase):
        w = principle_axis(angle)
        D = simulate_dataset(num_obs, 1, 2, W=w, σ=σ)
        data.append(D)
        states.extend([state] * num_obs)

    X = np.concatenate([sub.X for sub in data])
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"state": states}),
        var=pd.DataFrame({"gene": list(string.ascii_lowercase[:2])}).set_index("gene"),
        dtype=np.float32,
    )
    adata.uns["true_axes"] = {state: vec.W for state, vec in zip(string.ascii_uppercase, data)}
    adata.obsm["X_true"] = np.concatenate([sub.Z for sub in data])
    return adata


def simulate_2d_poisson_data(angles=[np.pi / 8 * 1, np.pi / 8 * 3], size_factor=[4.6, 5.2], num_obs=100):
    latent_dim = 1
    num_features = 2
    data = []
    states = []
    size_factors = []

    for i, (angle, state) in enumerate(zip(angles, string.ascii_uppercase)):
        w = principle_axis(angle)
        D = simulate_dataset(
            num_obs, latent_dim, num_features, W=w, z_loc=0.0, z_scale=1.0, size_factor=size_factor[i], noise="poisson"
        )
        data.append(D)
        states.extend([state] * num_obs)
        size_factors.extend([size_factor[i]] * num_obs)

    X = np.concatenate([sub.X for sub in data])
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"state": states, "size_factor": size_factors}),
        var=pd.DataFrame({"gene": list(string.ascii_lowercase[:num_features])}).set_index("gene"),
        dtype=np.float32,
    )
    adata.uns["true_axes"] = {state: vec.W for state, vec in zip(string.ascii_uppercase, data)}
    adata.obsm["X_true"] = np.concatenate([sub.Z for sub in data])
    return adata


@pytest.fixture(scope="session")
def data_dir(tmpdir_factory):
    # img = compute_expensive_image()
    test_dir = os.path.join(os.path.dirname(__file__), "test_files")
    tmp_dir = tmpdir_factory.getbasetemp()

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmp_dir))

    return tmp_dir


@pytest.fixture(scope="module")
def test_sparse_anndata():
    counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)
    adata = ad.AnnData(counts)
    return adata


@pytest.fixture(scope="session", name="test_anndata")
def test_anndata(data_dir):
    adata = sc.read_h5ad(os.path.join(str(data_dir), "test_object.h5ad"))

    return adata


@pytest.fixture(scope="session", name="two_state_data")
def two_state_data(data_dir):
    adata = simulate_2d_data()
    return adata


@pytest.fixture(scope="session", name="four_state_data")
def four_state_data(data_dir):
    adata = simulate_2d_data([np.pi / 8 * 1, -np.pi / 8 * 1, np.pi / 8 * 2, np.pi / 8 * 5])
    return adata

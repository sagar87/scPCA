import os
import string
from distutils import dir_util
from typing import NamedTuple

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
    X: NDArray[np.float32]


def principle_axis(rad):
    return np.array([np.cos(rad), np.sin(rad)]).reshape(1, -1)


def simulate_dataset(
    num_obs: int,
    num_dims: int,
    num_features: int,
    W: NDArray[np.float32] = None,
    z: NDArray[np.float32] = None,
    offset: NDArray[np.float32] = None,
    σ: float = 1,
) -> SimulatedData:
    if W is None:
        W = np.random.normal(0.0, 2.0, size=(num_dims, num_features))

    if offset is None:
        offset = np.zeros((1, num_features))

    if z is None:
        z = np.random.normal(0.0, 1.0, size=(num_obs, num_dims))

    μ = np.dot(z, W) + offset
    X = stats.norm(loc=μ, scale=σ).rvs()

    return SimulatedData(W, z, X)


def simulate_2d_data(angles=[np.pi / 8 * 1, np.pi / 8 * 4], num_obs=100, σ=0.1):
    data = []
    states = []

    for angle, state in zip(angles, string.ascii_uppercase):
        w = principle_axis(angle)
        D = simulate_dataset(num_obs, 1, 2, W=w, σ=σ)
        data.append(D)
        states.extend([state] * num_obs)

    X = np.concatenate([sub.X for sub in data])
    adata = ad.AnnData(X=X, obs=pd.DataFrame({"state": states}), dtype=np.float32)
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

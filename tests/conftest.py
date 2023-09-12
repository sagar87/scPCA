import os
from distutils import dir_util

import anndata as ad
import numpy as np
import pytest
import scanpy as sc
from scipy.sparse import csr_matrix


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

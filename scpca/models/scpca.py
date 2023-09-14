from typing import Any

import pyro  # type: ignore
import torch
from pyro import deterministic, param, plate, sample
from pyro.distributions import Exponential, Gamma, GammaPoisson, Normal  # type: ignore
from pyro.distributions import TransformedDistribution as TD
from pyro.distributions.constraints import positive  # type: ignore
from pyro.distributions.transforms import ExpTransform  # type: ignore
from torch import Tensor, einsum, exp, ones, tensor, zeros
from torch.types import Device


def scpca_model(
    X: Tensor,
    X_size: Tensor,
    design: Tensor,
    intercept: Tensor,
    design_idx: Tensor,
    intercept_idx: Tensor,
    idx: Tensor,
    num_genes: int,
    num_cells: int,
    num_factors: int,
    device: Device,
    subsampling: int = 0,
    β_rna_mean: float = 3.0,
    β_rna_sd: float = 0.01,
    W_fac_sd: float = 1.0,
    z_sd: float = 0.1,
    fixed_beta: bool = True,
) -> Any:
    if subsampling > 0:
        cell_plate = plate("cells", num_cells, subsample_size=subsampling)
    else:
        cell_plate = plate("cells", num_cells, subsample=idx)

    gene_plate = plate("genes", num_genes)
    num_intercepts = intercept.shape[1]
    intercept_plate = plate("intercept", num_intercepts)

    num_states = design.shape[1]
    state_plate = plate("states", num_states)

    # rescale
    normed_design = design * design.sum(1, keepdim=True) ** -0.5
    normed_batch = intercept * intercept.sum(1, keepdim=True) ** -0.5

    # convert to concentration and rates
    β_rna_conc = β_rna_mean**2 / β_rna_sd**2
    β_rna_rate = β_rna_mean / β_rna_sd**2

    with state_plate:
        W_fac = sample(
            "W_fac",
            Normal(
                zeros((num_factors, num_genes), device=device),
                W_fac_sd * ones((num_factors, num_genes), device=device),
            ).to_event(2),
        )

    with intercept_plate:
        W_add = sample("W_add", Normal(zeros(num_genes, device=device), ones(num_genes, device=device)).to_event(1))

    if fixed_beta:
        with gene_plate:
            α_rna_inv = sample("α_rna_inv", Exponential(tensor(β_rna_mean, device=device)))
            α_rna = deterministic("α_rna", (1 / α_rna_inv).T)

    else:
        β_rna = sample("β_rna", Gamma(tensor(β_rna_conc, device=device), tensor(β_rna_rate, device=device)))

        with gene_plate:
            α_rna_inv = sample("α_rna_inv", Exponential(β_rna))
            α_rna = deterministic("α_rna", (1 / α_rna_inv).T)

    with cell_plate as ind:
        z = sample(
            "z",
            Normal(
                zeros(num_factors, device=device),
                z_sd * ones(num_factors, device=device),
            ).to_event(1),
        )

        intercept_mat = normed_batch[intercept_idx[ind]]
        design_indicator = design_idx[ind]
        cell_indicator = torch.arange(ind.shape[0])

        # construct bases for each column of in D
        W_lin = pyro.deterministic("W_lin", (normed_design.unsqueeze(2).unsqueeze(3) * W_fac.unsqueeze(0)).sum(1))
        Wz = einsum("cf,bfp->bcp", z, W_lin)
        intercept_rna = intercept_mat @ W_add
        offset_rna = pyro.deterministic("offset_rna", X_size[ind] + intercept_rna)

        μ_rna = deterministic("μ_rna", exp(offset_rna + Wz[design_indicator, cell_indicator]))
        deterministic("σ_rna", μ_rna**2 / α_rna * (1 + α_rna / μ_rna))
        return sample("rna", GammaPoisson(α_rna, α_rna / μ_rna).to_event(1), obs=X[ind])


def scpca_guide(
    X: Tensor,
    X_size: Tensor,
    design: Tensor,
    intercept: Tensor,
    design_idx: Tensor,
    intercept_idx: Tensor,
    idx: Tensor,
    num_genes: int,
    num_cells: int,
    num_factors: int,
    device: Device,
    subsampling: int = 0,
    β_rna_mean: float = 3.0,
    β_rna_sd: float = 0.1,
    W_fac_sd: float = 1.0,
    z_sd: float = 1.0,
    fixed_beta: bool = True,
) -> None:
    gene_plate = plate("genes", num_genes)

    num_intercepts = intercept.shape[1]
    intercept_plate = plate("intercept", num_intercepts)

    # design matrix
    num_states = design.shape[1]
    state_plate = plate("states", num_states)

    if subsampling > 0:
        cell_plate = plate("cells", num_cells, subsample_size=subsampling)
    else:
        cell_plate = plate("cells", num_cells, subsample=idx)

    W_fac_loc = param("W_fac_loc", zeros((num_states, num_factors, num_genes), device=device))
    W_fac_scale = param(
        "W_fac_scale",
        0.1 * ones((num_states, num_factors, num_genes), device=device),
        constraint=positive,
    )

    with state_plate:
        sample("W_fac", Normal(W_fac_loc, W_fac_scale).to_event(2))

    # intercept terms
    W_add_loc = param("W_add_loc", zeros((num_intercepts, num_genes), device=device))
    W_add_scale = param("W_add_scale", 0.1 * ones((num_intercepts, num_genes), device=device), constraint=positive)

    with intercept_plate:
        sample("W_add", Normal(W_add_loc, W_add_scale).to_event(1))

    # account for intercept in beta
    if not fixed_beta:
        β_rna_loc = param("β_rna_loc", zeros(1, device=device))
        β_rna_scale = param("β_rna_scale", ones(1, device=device), constraint=positive)

        sample("β_rna", TD(Normal(β_rna_loc, β_rna_scale), ExpTransform()))

    α_rna_loc = param("α_rna_loc", zeros((num_genes), device=device))
    α_rna_scale = param("α_rna_scale", 0.1 * ones((num_genes), device=device), constraint=positive)

    with gene_plate:
        sample("α_rna_inv", TD(Normal(α_rna_loc, α_rna_scale), ExpTransform()))

    z_loc = param("z_loc", zeros((num_cells, num_factors), device=device))
    z_scale = param("z_scale", 0.1 * ones((num_cells, num_factors), device=device), constraint=positive)

    with cell_plate as ind:
        sample("z", Normal(z_loc[ind], z_scale[ind]).to_event(1))

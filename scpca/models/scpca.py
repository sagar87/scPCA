import pyro  # type: ignore
import torch
from pyro import deterministic, param, plate, sample
from pyro.distributions import (  # type: ignore
    Exponential,
    Gamma,
    GammaPoisson,
    HalfCauchy,
    InverseGamma,
    Normal,
)
from pyro.distributions import TransformedDistribution as TD
from pyro.distributions.constraints import positive  # type: ignore
from pyro.distributions.transforms import ExpTransform  # type: ignore
from torch import Tensor, einsum, exp, ones, tensor, zeros
from torch.types import Device


def scpca_model(
    X: Tensor,
    X_size: Tensor,
    design: Tensor,
    batch: Tensor,
    design_idx: Tensor,
    batch_idx: Tensor,
    idx: Tensor,
    num_genes: int,
    num_batches: int,
    num_cells: int,
    num_factors: int,
    device: Device,
    subsampling: int = 0,
    β_rna_mean: float = 3.0,
    β_rna_sd: float = 0.01,
    W_fac_sd: float = 1.0,
    z_sd: float = 0.1,
    horseshoe: bool = False,
    fixed_beta: bool = False,
    minibatches: bool = False,
) -> None:
    gene_plate = plate("genes", num_genes)
    batch_plate = plate("batches", num_batches)

    num_groups = design.shape[1]
    group_plate = plate("groups", num_groups)

    if subsampling > 0:
        cell_plate = plate("cells", num_cells, subsample_size=subsampling)
    else:
        cell_plate = plate("cells", num_cells, subsample=idx)

    # factor matrices
    if horseshoe:
        with pyro.plate("modality", 2):
            tau = pyro.sample("tau", HalfCauchy(ones(1, device=device)))

    normed_design = design * design.sum(1, keepdim=True) ** -0.5
    normed_batch = batch * batch.sum(1, keepdim=True) ** -0.5

    with group_plate:
        W_fac = sample(
            "W_fac",
            Normal(
                zeros((num_factors, num_genes), device=device),
                W_fac_sd * ones((num_factors, num_genes), device=device),
            ).to_event(2),
        )

        if horseshoe:  # turns on sparsity priors
            W_del = sample("W_del", HalfCauchy(ones(1, device=device)))
            W_lam = sample("W_lam", HalfCauchy(ones(num_genes, device=device)).to_event(1))
            W_c = sample(
                "W_c",
                InverseGamma(
                    0.5 * ones(num_genes, device=device),
                    0.5 * ones(num_genes, device=device),
                ).to_event(1),
            )

            # print(W_tau.shape, W_lam.shape, W_fac.shape)
            W_gamma = tau[0] * W_del.reshape(-1, 1) * W_lam
            W_fac = deterministic("W_horse", W_fac * (torch.sqrt(W_c) * W_gamma) / torch.sqrt(W_c + W_gamma**2))

    with batch_plate:
        W_add = sample("W_add", Normal(zeros(num_genes, device=device), ones(num_genes, device=device)).to_event(1))

    β_rna_conc = β_rna_mean**2 / β_rna_sd**2
    β_rna_rate = β_rna_mean / β_rna_sd**2

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

        intercept_mat = normed_batch[batch_idx[ind]]
        design_indicator = design_idx[ind]
        cell_indicator = torch.arange(ind.shape[0])

        # construct bases for each column of in D
        W_lin = pyro.deterministic("W_lin", (normed_design.unsqueeze(2).unsqueeze(3) * W_fac.unsqueeze(0)).sum(1))

        # W_nrm = torch.linalg.norm(W_lin, dim=2, keepdims=True)
        # W_vec = pyro.deterministic("W_vec", W_lin / W_nrm)
        # z_vec = pyro.deterministic("z_vec", z * W_nrm[design_indicator].squeeze())

        Wz = einsum("cf,bfp->bcp", z, W_lin)
        intercept_rna = intercept_mat @ W_add
        offset_rna = pyro.deterministic("offset_rna", X_size[ind] + intercept_rna)

        μ_rna = deterministic("μ_rna", exp(offset_rna + Wz[design_indicator, cell_indicator]))
        α_rna_bat = α_rna
        deterministic("σ_rna", μ_rna**2 / α_rna_bat * (1 + α_rna_bat / μ_rna))
        sample("rna", GammaPoisson(α_rna_bat, α_rna_bat / μ_rna).to_event(1), obs=X[ind])


def scpca_guide(
    X: Tensor,
    X_size: Tensor,
    design: Tensor,
    batch: Tensor,
    design_idx: Tensor,
    batch_idx: Tensor,
    idx: Tensor,
    num_genes: int,
    num_batches: int,
    num_cells: int,
    num_factors: int,
    device: Device,
    subsampling: int = 0,
    β_rna_mean: float = 3.0,
    β_rna_sd: float = 0.1,
    W_fac_sd: float = 1.0,
    z_sd: float = 1.0,
    horseshoe: bool = False,
    fixed_beta: bool = False,
    minibatches: bool = False,
) -> None:
    gene_plate = plate("genes", num_genes)
    batch_plate = plate("batches", num_batches)

    # design matrix
    num_groups = design.shape[1]
    group_plate = plate("groups", num_groups)

    if subsampling > 0:
        cell_plate = plate("cells", num_cells, subsample_size=subsampling)
    else:
        cell_plate = plate("cells", num_cells, subsample=idx)

    W_fac_loc = param("W_fac_loc", zeros((num_groups, num_factors, num_genes), device=device))
    W_fac_scale = param(
        "W_fac_scale",
        0.1 * ones((num_groups, num_factors, num_genes), device=device),
        constraint=positive,
    )

    if horseshoe:
        tau_loc = param("tau_loc", zeros(2, device=device))
        tau_scale = param("tau_scale", 0.1 * ones(2, device=device), constraint=positive)

        with pyro.plate("modality", 2):
            sample(
                "tau",
                TD(Normal(tau_loc, tau_scale), ExpTransform()),
            )

        W_del_loc = param("W_del_loc", zeros(num_factors, device=device))
        W_del_scale = param("W_del_scale", 0.1 * ones(num_factors, device=device), constraint=positive)

        W_lam_loc = param("W_lam_loc", zeros((num_factors, num_genes), device=device))
        W_lam_scale = param("W_lam_scale", 0.1 * ones((num_factors, num_genes), device=device), constraint=positive)

        W_c_loc = param("W_c_loc", zeros((num_factors, num_genes), device=device))
        W_c_scale = param("W_c_scale", 0.1 * ones((num_factors, num_genes), device=device), constraint=positive)

    with group_plate:
        sample("W_fac", Normal(W_fac_loc, W_fac_scale).to_event(2))

        if horseshoe:
            sample("W_del", TD(Normal(W_del_loc, W_del_scale), ExpTransform()))
            sample("W_lam", TD(Normal(W_lam_loc, W_lam_scale), ExpTransform()).to_event(1))
            sample("W_c", TD(Normal(W_c_loc, W_c_scale), ExpTransform()).to_event(1))

    # intercept terms
    W_add_loc = param("W_add_loc", zeros((num_batches, num_genes), device=device))
    W_add_scale = param("W_add_scale", 0.1 * ones((num_batches, num_genes), device=device), constraint=positive)

    with batch_plate:
        sample("W_add", Normal(W_add_loc, W_add_scale).to_event(1))

    # account for batches in beta
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

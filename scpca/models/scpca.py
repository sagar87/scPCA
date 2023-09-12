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
    TransformedDistribution,
)
from pyro.distributions.constraints import less_than, positive  # type: ignore
from pyro.distributions.transforms import ExpTransform  # type: ignore
from torch import einsum, exp, ones, tensor, zeros


def scpca_model(
    X: torch.Tensor,
    Y: torch.Tensor,
    X_size: torch.Tensor,
    Y_size: torch.Tensor,
    design: torch.Tensor,
    batch: torch.Tensor,
    design_idx: torch.Tensor,
    batch_idx: torch.Tensor,
    idx: torch.Tensor,
    num_genes: int,
    num_proteins: int,
    num_batches: int,
    num_cells: int,
    num_factors: int,
    device: torch.device,
    subsampling: int = 0,
    β_rna_mean: float = 3.0,
    β_rna_sd: float = 0.01,
    W_fac_sd: float = 1.0,
    z_sd: float = 0.1,
    horseshoe: bool = False,
    batch_beta: bool = False,
    fixed_beta: bool = False,
    intercept: bool = True,
    constrain_alpha: bool = False,
    minibatches: bool = False,
) -> None:
    gene_plate = plate("genes", num_genes)

    if Y is not None:
        protein_plate = plate("proteins", num_proteins)

    # cell_plate = plate("cells", num_cells)
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
        # W_fac = sample("W_fac", Normal(zeros(num_genes, device=device), ones(num_genes, device=device)).to_event(2))
        W_fac = sample(
            "W_fac",
            Normal(
                zeros((num_factors, num_genes), device=device),
                W_fac_sd * ones((num_factors, num_genes), device=device),
            ).to_event(2),
        )
        # scale additive factors
        # W_fac = W_fac * torch.cat([torch.ones((1, num_factors, num_genes), device=device), W_fac_sd * torch.ones((num_groups-1, num_factors, num_genes), device=device)], 0)
        # torch.ones((num_groups, num_factors, num_genes), device=device)
        # print(W_fac.shape)11
        if Y is not None:
            V_fac = sample(
                "V_fac",
                Normal(
                    zeros((num_factors, num_proteins), device=device),
                    ones((num_factors, num_proteins), device=device),
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
            W_fac = deterministic(
                "W_horse",
                W_fac * (torch.sqrt(W_c) * W_gamma) / torch.sqrt(W_c + W_gamma**2),
            )

            V_del = sample("V_del", HalfCauchy(ones(1, device=device)))
            V_lam = sample("V_lam", HalfCauchy(ones(num_proteins, device=device)).to_event(1))
            V_c = sample(
                "V_c",
                InverseGamma(
                    0.5 * ones(num_proteins, device=device),
                    0.5 * ones(num_proteins, device=device),
                ).to_event(1),
            )

            V_gamma = tau[1] * V_del.reshape(-1, 1) * V_lam
            V_fac = deterministic(
                "V_horse",
                V_fac * (torch.sqrt(V_c) * V_gamma) / torch.sqrt(V_c + V_gamma**2),
            )

    # if design is not None:
    #     with group_plate:
    #         W_grp = sample("W_grp", Normal(zeros((num_factors, num_genes), device=device), ones((num_factors, num_genes), device=device)).to_event(2))
    #         # print('W_grp', W_grp.shape)
    #         # base = zeros((1, num_factors, num_genes), device=device)
    #             # print(Normal(zeros(num_genes, device=device), ones(num_genes, device=device)).to_event(1).batch_shape)
    #         W_grp_cat = torch.cat([W_fac.unsqueeze(0), W_grp], 0)

    if intercept:
        with batch_plate:
            W_add = sample(
                "W_add",
                Normal(zeros(num_genes, device=device), ones(num_genes, device=device)).to_event(1),
            )

            if Y is not None:
                V_add = sample(
                    "V_add",
                    Normal(
                        zeros(num_proteins, device=device),
                        ones(num_proteins, device=device),
                    ).to_event(1),
                )
    else:
        W_add = torch.zeros((num_batches, num_genes), device=device)

        if Y is not None:
            V_add = torch.zeros((num_batches, num_proteins), device=device)

    β_rna_conc = (β_rna_mean**2 / β_rna_sd**2,)
    β_rna_rate = (β_rna_mean / β_rna_sd**2,)

    if batch_beta:
        with batch_plate:
            β_rna = sample(
                "β_rna",
                Gamma(tensor(β_rna_conc, device=device), tensor(β_rna_rate, device=device)),
            )

            if Y is not None:
                β_prot = sample(
                    "β_prot",
                    Gamma(tensor(9.0, device=device), tensor(3.0, device=device)),
                )

        with gene_plate:
            α_rna_inv = sample("α_rna_inv", Exponential(β_rna).to_event(1))
            α_rna = deterministic("α_rna", (1 / α_rna_inv).T)

        # print('model', α_rna_inv.shape)
        if Y is not None:
            with protein_plate:
                α_prot_inv = sample("α_prot_inv", Exponential(β_prot).to_event(1))
                α_prot = deterministic("α_prot", (1 / α_prot_inv).T)
    elif fixed_beta:
        with gene_plate:
            α_rna_inv = sample("α_rna_inv", Exponential(tensor(β_rna_mean, device=device)))
            α_rna = deterministic("α_rna", (1 / α_rna_inv).T)

        if Y is not None:
            with protein_plate:
                α_prot_inv = sample("α_prot_inv", Exponential(β_prot))
                α_prot = deterministic("α_prot", (1 / α_prot_inv).T)
    else:
        β_rna = sample(
            "β_rna",
            Gamma(tensor(β_rna_conc, device=device), tensor(β_rna_rate, device=device)),
        )
        if Y is not None:
            β_prot = sample("β_prot", Gamma(tensor(9.0, device=device), tensor(3.0, device=device)))

        with gene_plate:
            α_rna_inv = sample("α_rna_inv", Exponential(β_rna))
            α_rna = deterministic("α_rna", (1 / α_rna_inv).T)

            # print('model', α_rna_inv.shape)

        if Y is not None:
            with protein_plate:
                α_prot_inv = sample("α_prot_inv", Exponential(β_prot))
                α_prot = deterministic("α_prot", (1 / α_prot_inv).T)

    if subsampling > 0 or minibatches:
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

            if Y is not None:
                V_lin = pyro.deterministic(
                    "V_lin", (normed_design.unsqueeze(2).unsqueeze(3) * V_fac.unsqueeze(0)).sum(1)
                )
                # V_nrm = torch.linalg.norm(V_lin, dim=2, keepdims=True)
                # V_vec = pyro.deterministic("V_vec", V_lin / V_nrm)

                Vz = einsum("cf,bfp->bcp", z, V_lin)
                intercept_prot = intercept_mat @ V_add
                offset_prot = pyro.deterministic("offset_prot", Y_size[ind] + intercept_prot)

                μ_prot = deterministic("μ_prot", exp(offset_prot + Vz[design_indicator, cell_indicator]))

            if batch_beta:
                α_rna_bat = intercept_mat @ α_rna

                if Y is not None:
                    α_prot_bat = intercept_mat @ α_prot
            else:
                α_rna_bat = α_rna

                if Y is not None:
                    α_prot_bat = α_prot

            deterministic("σ_rna", μ_rna**2 / α_rna_bat * (1 + α_rna_bat / μ_rna))

            sample(
                "rna",
                GammaPoisson(α_rna_bat, α_rna_bat / μ_rna).to_event(1),
                obs=X[ind],
            )
            if Y is not None:
                deterministic("σ_prot", μ_prot**2 / α_prot_bat * (1 + α_prot_bat / μ_prot))
                sample(
                    "prot",
                    GammaPoisson(α_prot_bat, α_prot_bat / μ_prot).to_event(1),
                    obs=Y[ind],
                )
    else:
        with cell_plate:
            z = sample(
                "z",
                Normal(
                    zeros(num_factors, device=device),
                    0.1 * ones(num_factors, device=device),
                ).to_event(1),
            )

            μ_rna = deterministic(
                "μ_rna",
                exp(
                    (
                        batch.T.unsqueeze(-1)
                        * (X_size.unsqueeze(0) + W_add.unsqueeze(1) + einsum("cf,bfp->bcp", z, W_fac.unsqueeze(0)))
                    ).sum(0)
                ),
            )

            if Y is not None:
                μ_prot = deterministic(
                    "μ_prot",
                    exp(
                        (
                            batch.T.unsqueeze(-1)
                            * (Y_size.unsqueeze(0) + V_add.unsqueeze(1) + einsum("cf,bfp->bcp", z, V_fac.unsqueeze(0)))
                        ).sum(0)
                    ),
                )

            if batch_beta:
                α_rna_bat = batch @ α_rna
                if Y is not None:
                    α_prot_bat = batch @ α_prot
            else:
                α_rna_bat = α_rna
                if Y is not None:
                    α_prot_bat = α_prot

            sample("rna", GammaPoisson(α_rna_bat, α_rna_bat / μ_rna).to_event(1), obs=X)

            if Y is not None:
                sample(
                    "prot",
                    GammaPoisson(α_prot_bat, α_prot_bat / μ_prot).to_event(1),
                    obs=Y,
                )


def scpca_guide(
    X: torch.Tensor,
    Y: torch.Tensor,
    X_size: torch.Tensor,
    Y_size: torch.Tensor,
    design: torch.Tensor,
    batch: torch.Tensor,
    design_idx: torch.Tensor,
    batch_idx: torch.Tensor,
    idx: torch.Tensor,
    num_genes: int,
    num_proteins: int,
    num_batches: int,
    num_cells: int,
    num_factors: int,
    device: torch.device,
    subsampling: int = 0,
    β_rna_mean: float = 3.0,
    β_rna_sd: float = 0.1,
    W_fac_sd: float = 1.0,
    z_sd: float = 1.0,
    horseshoe: bool = False,
    batch_beta: bool = False,
    fixed_beta: bool = False,
    intercept: bool = True,
    constrain_alpha: bool = False,
    minibatches: bool = False,
) -> None:
    gene_plate = plate("genes", num_genes)
    if Y is not None:
        protein_plate = plate("proteins", num_proteins)
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

    if Y is not None:
        V_fac_loc = param("V_fac_loc", zeros((num_groups, num_factors, num_proteins), device=device))
        V_fac_scale = param(
            "V_fac_scale",
            0.1 * ones((num_groups, num_factors, num_proteins), device=device),
            constraint=positive,
        )

    if horseshoe:
        tau_loc = param("tau_loc", zeros(2, device=device))
        tau_scale = param("tau_scale", 0.1 * ones(2, device=device), constraint=positive)

        with pyro.plate("modality", 2):
            sample(
                "tau",
                TransformedDistribution(Normal(tau_loc, tau_scale), ExpTransform()),
            )

        W_del_loc = param("W_del_loc", zeros(num_factors, device=device))
        W_del_scale = param("W_del_scale", 0.1 * ones(num_factors, device=device), constraint=positive)

        W_lam_loc = param("W_lam_loc", zeros((num_factors, num_genes), device=device))
        W_lam_scale = param(
            "W_lam_scale",
            0.1 * ones((num_factors, num_genes), device=device),
            constraint=positive,
        )

        W_c_loc = param("W_c_loc", zeros((num_factors, num_genes), device=device))
        W_c_scale = param(
            "W_c_scale",
            0.1 * ones((num_factors, num_genes), device=device),
            constraint=positive,
        )

        V_del_loc = param("V_del_loc", zeros(num_factors, device=device))
        V_del_scale = param("V_del_scale", 0.1 * ones(num_factors, device=device), constraint=positive)

        V_lam_loc = param("V_lam_loc", zeros((num_factors, num_proteins), device=device))
        V_lam_scale = param(
            "V_lam_scale",
            0.1 * ones((num_factors, num_proteins), device=device),
            constraint=positive,
        )

        V_c_loc = param("V_c_loc", zeros((num_factors, num_proteins), device=device))
        V_c_scale = param(
            "V_c_scale",
            0.1 * ones((num_factors, num_proteins), device=device),
            constraint=positive,
        )

    with group_plate:
        sample("W_fac", Normal(W_fac_loc, W_fac_scale).to_event(2))

        if Y is not None:
            sample("V_fac", Normal(V_fac_loc, V_fac_scale).to_event(2))

        if horseshoe:
            sample(
                "W_del",
                TransformedDistribution(Normal(W_del_loc, W_del_scale), ExpTransform()),
            )
            sample(
                "W_lam",
                TransformedDistribution(Normal(W_lam_loc, W_lam_scale), ExpTransform()).to_event(1),
            )
            sample(
                "W_c",
                TransformedDistribution(Normal(W_c_loc, W_c_scale), ExpTransform()).to_event(1),
            )

            sample(
                "V_del",
                TransformedDistribution(Normal(V_del_loc, V_del_scale), ExpTransform()),
            )
            sample(
                "V_lam",
                TransformedDistribution(Normal(V_lam_loc, V_lam_scale), ExpTransform()).to_event(1),
            )
            sample(
                "V_c",
                TransformedDistribution(Normal(V_c_loc, V_c_scale), ExpTransform()).to_event(1),
            )

    #     if design is not None:
    #         W_grp_loc = param("W_grp_loc", zeros((num_groups, num_factors, num_genes), device=device))
    #         W_grp_scale = param( "W_grp_scale", 0.1 * ones((num_groups, num_factors, num_genes), device=device), constraint=positive)

    #         with group_plate:
    #             sample("W_grp", Normal(W_grp_loc, W_grp_scale).to_event(2))

    # intercept terms
    if intercept:
        W_add_loc = param(
            "W_add_loc",
            zeros((num_batches, num_genes), device=device),
        )
        W_add_scale = param(
            "W_add_scale",
            0.1 * ones((num_batches, num_genes), device=device),
            constraint=positive,
        )

        if Y is not None:
            V_add_loc = param(
                "V_add_loc",
                zeros((num_batches, num_proteins), device=device),
            )
            V_add_scale = param(
                "V_add_scale",
                0.1 * ones((num_batches, num_proteins), device=device),
                constraint=positive,
            )

        with batch_plate:
            sample("W_add", Normal(W_add_loc, W_add_scale).to_event(1))

            if Y is not None:
                sample("V_add", Normal(V_add_loc, V_add_scale).to_event(1))

    # account for batches in beta
    if batch_beta:
        β_rna_loc = param("β_rna_loc", zeros(num_batches, device=device))
        β_rna_scale = param("β_rna_scale", ones(num_batches, device=device), constraint=positive)

        if Y is not None:
            β_prot_loc = param("β_prot_loc", zeros(num_batches, device=device))
            β_prot_scale = param("β_prot_scale", ones(num_batches, device=device), constraint=positive)

        with batch_plate:
            sample(
                "β_rna",
                TransformedDistribution(Normal(β_rna_loc, β_rna_scale), ExpTransform()),
            )

            if Y is not None:
                sample(
                    "β_prot",
                    TransformedDistribution(Normal(β_prot_loc, β_prot_scale), ExpTransform()),
                )
    elif fixed_beta:
        pass
    else:
        β_rna_loc = param("β_rna_loc", zeros(1, device=device))
        β_rna_scale = param("β_rna_scale", ones(1, device=device), constraint=positive)

        if Y is not None:
            β_prot_loc = param("β_prot_loc", zeros(1, device=device))
            β_prot_scale = param("β_prot_scale", ones(1, device=device), constraint=positive)

        sample(
            "β_rna",
            TransformedDistribution(Normal(β_rna_loc, β_rna_scale), ExpTransform()),
        )

        if Y is not None:
            sample(
                "β_prot",
                TransformedDistribution(Normal(β_prot_loc, β_prot_scale), ExpTransform()),
            )
        # print('guide', β_rna_loc.shape)

    if batch_beta:
        if constrain_alpha:
            α_rna_loc = param(
                "α_rna_loc",
                zeros((num_genes, num_batches), device=device),
                constraint=less_than(6.0),
            )
        else:
            α_rna_loc = param("α_rna_loc", zeros((num_genes, num_batches), device=device))

        α_rna_scale = param(
            "α_rna_scale",
            0.1 * ones((num_genes, num_batches), device=device),
            constraint=positive,
        )

        if Y is not None:
            α_prot_loc = param("α_prot_loc", zeros((num_proteins, num_batches), device=device))
            α_prot_scale = param(
                "α_prot_scale",
                0.1 * ones((num_proteins, num_batches), device=device),
                constraint=positive,
            )

        with gene_plate:
            sample(
                "α_rna_inv",
                TransformedDistribution(Normal(α_rna_loc, α_rna_scale), ExpTransform()).to_event(1),
            )

        if Y is not None:
            with protein_plate:
                sample(
                    "α_prot_inv",
                    TransformedDistribution(Normal(α_prot_loc, α_prot_scale), ExpTransform()).to_event(1),
                )
    else:
        if constrain_alpha:
            α_rna_loc = param(
                "α_rna_loc",
                zeros((num_genes), device=device),
                constraint=less_than(6.0),
            )
        else:
            α_rna_loc = param("α_rna_loc", zeros((num_genes), device=device))

        α_rna_scale = param("α_rna_scale", 0.1 * ones((num_genes), device=device), constraint=positive)

        if Y is not None:
            α_prot_loc = param("α_prot_loc", zeros((num_proteins), device=device))
            α_prot_scale = param(
                "α_prot_scale",
                0.1 * ones((num_proteins), device=device),
                constraint=positive,
            )

        with gene_plate:
            sample(
                "α_rna_inv",
                TransformedDistribution(Normal(α_rna_loc, α_rna_scale), ExpTransform()),
            )

        if Y is not None:
            with protein_plate:
                sample(
                    "α_prot_inv",
                    TransformedDistribution(Normal(α_prot_loc, α_prot_scale), ExpTransform()),
                )

    z_loc = param("z_loc", zeros((num_cells, num_factors), device=device))
    z_scale = param(
        "z_scale",
        0.1 * ones((num_cells, num_factors), device=device),
        constraint=positive,
    )

    with cell_plate as ind:
        sample("z", Normal(z_loc[ind], z_scale[ind]).to_event(1))

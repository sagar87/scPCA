from pyro import deterministic, param, plate, sample  # type: ignore
from pyro.distributions import (  # type: ignore
    HalfCauchy,
    Normal,
    TransformedDistribution,
)
from pyro.distributions.constraints import positive  # type: ignore
from pyro.distributions.transforms import ExpTransform  # type: ignore
from torch import Tensor, einsum, ones, zeros
from torch.types import Device


def dpca_model(
    X: Tensor,
    intercept_design: Tensor,
    loading_design: Tensor,
    idx: Tensor,
    num_factors: int,
    num_obs: int,
    device: Device,
    z_sd: float = 1.0,
    variance: str = "diagonal",
    subsampling: int = 0,
    intercept: bool = True,
    minibatches: bool = False,
) -> None:
    if subsampling > 0:
        obs_plate = plate("obs", num_obs, subsample_size=subsampling)
    else:
        obs_plate = plate("obs", num_obs, subsample=idx)

    # factor_plate = plate("factor", num_factors)

    num_features = X.shape[1]
    feature_plate = plate("features", num_features)

    num_batches = intercept_design.shape[1]
    batch_plate = plate("batches", num_batches)

    num_groups = loading_design.shape[1]
    group_plate = plate("groups", num_groups)

    with group_plate:
        # W_fac = sample("W_fac", Normal(zeros(num_genes, device=device), ones(num_genes, device=device)).to_event(2))
        W_fac = sample(
            "W_fac",
            Normal(
                zeros((num_factors, num_features), device=device),
                ones((num_factors, num_features), device=device),
            ).to_event(2),
        )

    with batch_plate:
        W_add = sample(
            "W_add",
            Normal(zeros(num_features, device=device), ones(num_features, device=device)).to_event(1),
        )

    if variance == "diagonal":
        with feature_plate:
            σ = sample("σ", HalfCauchy(1.0))
    elif variance == "isotropic":
        σ = sample("σ", HalfCauchy(1.0))

    # print(W_fac.shape, W_add.shape)

    if subsampling > 0 or minibatches:
        with obs_plate as ind:
            z = sample(
                "z",
                Normal(
                    zeros(num_factors, device=device),
                    z_sd * ones(num_factors, device=device),
                ).to_event(1),
            )
            # print(z.shape, loading_design[ind].T.unsqueeze(-1).shape)
            Wz = (loading_design[ind].T.unsqueeze(-1) * einsum("cf,bfp->bcp", z, W_fac)).sum(0)
            intercept = intercept_design[ind] @ W_add

            μ = deterministic("μ_rna", intercept + Wz)
            sample("rna", Normal(μ, σ).to_event(1), obs=X[ind])


def dpca_guide(
    X: Tensor,
    intercept_design: Tensor,
    loading_design: Tensor,
    idx: Tensor,
    num_factors: int,
    num_obs: int,
    device: Device,
    z_sd: float = 1.0,
    variance: str = "diagonal",
    subsampling: int = 0,
    intercept: bool = True,
    minibatches: bool = False,
) -> None:
    if subsampling > 0:
        obs_plate = plate("obs", num_obs, subsample_size=subsampling)
    else:
        obs_plate = plate("obs", num_obs, subsample=idx)

    # factor_plate = plate("factor", num_factors)

    num_features = X.shape[1]
    feature_plate = plate("features", num_features)

    num_batches = intercept_design.shape[1]
    batch_plate = plate("batches", num_batches)

    num_groups = loading_design.shape[1]
    group_plate = plate("groups", num_groups)

    W_fac_loc = param("W_fac_loc", zeros((num_groups, num_factors, num_features), device=device))
    W_fac_scale = param(
        "W_fac_scale",
        0.1 * ones((num_groups, num_factors, num_features), device=device),
        constraint=positive,
    )

    with group_plate:
        sample("W_fac", Normal(W_fac_loc, W_fac_scale).to_event(2))

    W_add_loc = param(
        "W_add_loc",
        zeros((num_batches, num_features), device=device),
    )
    W_add_scale = param(
        "W_add_scale",
        0.1 * ones((num_batches, num_features), device=device),
        constraint=positive,
    )

    with batch_plate:
        sample("W_add", Normal(W_add_loc, W_add_scale).to_event(1))

    if variance == "diagonal":
        σ_loc = param("σ_loc", zeros(num_features, device=device))
        σ_scale = param("σ_scale", 0.1 * ones(num_features, device=device), constraint=positive)
        with feature_plate:
            sample("σ", TransformedDistribution(Normal(σ_loc, σ_scale), ExpTransform()))
    elif variance == "isotropic":
        σ_loc = param("σ_loc", zeros(1, device=device))
        σ_scale = param("σ_scale", 0.1 * ones(1, device=device), constraint=positive)
        sample("σ", TransformedDistribution(Normal(σ_loc, σ_scale), ExpTransform()))

    z_loc = param("z_loc", zeros((num_obs, num_factors), device=device))
    z_scale = param(
        "z_scale",
        0.1 * ones((num_obs, num_factors), device=device),
        constraint=positive,
    )

    with obs_plate as ind:
        sample("z", Normal(z_loc[ind], z_scale[ind]).to_event(1))

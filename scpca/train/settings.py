import pyro  # type: ignore

DEFAULT = dict(
    num_epochs=5000,
    optimizer=pyro.optim.ClippedAdam,
    optimizer_kwargs={"lr": 0.01, "betas": (0.95, 0.999)},
    scheduler=None,
    loss=pyro.infer.TraceMeanField_ELBO,
    loss_kwargs={"num_particles": 1},
)


TEST = dict(
    num_epochs=200,
    optimizer=pyro.optim.ClippedAdam,
    optimizer_kwargs={"lr": 0.1, "betas": (0.95, 0.999)},
    scheduler=None,
    loss=pyro.infer.TraceMeanField_ELBO,
    loss_kwargs={"num_particles": 1},
)

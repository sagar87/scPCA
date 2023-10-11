from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pyro  # type: ignore
import torch
from numpy.typing import NDArray
from pyro.infer import Predictive, Trace_ELBO  # type: ignore
from torch.cuda import empty_cache
from tqdm import tqdm  # type: ignore

from scpca.train.handler import SVIBaseHandler


class SVILocalHandler(SVIBaseHandler):
    """
    Extend SVIBaseHandler to enable to use a separate model and guide
    for prediction. Assumes theat model and guide accept an idx argument
    that is an torch array of indices.
    """

    def __init__(
        self,
        model: Callable[[Any], Any],
        guide: Callable[[Any], Any],
        loss: Trace_ELBO = pyro.infer.TraceMeanField_ELBO,
        optimizer: Callable[[Any], Any] = torch.optim.Adam,
        scheduler: Callable[[Any], Any] = pyro.optim.ReduceLROnPlateau,
        seed: Optional[int] = None,
        num_epochs: int = 30000,
        log_freq: int = 10,
        optimizer_kwargs: Dict[str, Any] = {"lr": 1e-2},
        scheduler_kwargs: Dict[str, Any] = {"factor": 0.99},
        loss_kwargs: Dict[str, Any] = {"num_particles": 1},
        predict_model: Optional[Callable[[Any], Any]] = None,
        predict_guide: Optional[Callable[[Any], Any]] = None,
        idx: Any = None,
    ):
        super().__init__(
            model=model,
            guide=guide,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            seed=seed,
            num_epochs=num_epochs,
            log_freq=log_freq,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
            loss_kwargs=loss_kwargs,
        )
        self.predict_model = predict_model
        self.predict_guide = predict_guide
        self.idx = idx

    def predict(
        self, return_sites: List[str], num_samples: int = 25, *args: Union[str, List[str]], **kwargs: Any
    ) -> Dict[str, NDArray[np.float32]]:
        if self.params is not None:
            pyro.clear_param_store()
            pyro.get_param_store().set_state(self.params)

        predictive = Predictive(
            self.predict_model,
            guide=self.predict_guide,
            num_samples=num_samples,
            return_sites=return_sites,
        )

        posterior = predictive(*args, **kwargs)
        posterior_predictive: Dict[str, NDArray[np.float32]] = self._to_numpy(posterior)
        empty_cache()

        return posterior_predictive

    def predict_global_variable(self, var: str, num_samples: int = 25, return_mean: bool = False) -> Any:
        """
        Sample global variables from the posterior.

        Parameters
        ----------
        var
            Name of the variable to sample.
        num_samples
            Number of samples to draw.
        return_mean
            If true posterior samples are averaged directly after sampling.
        """

        posterior_predictive = self.predict([var], num_samples=num_samples, idx=self.idx[0:1])
        return posterior_predictive[var].mean(0) if return_mean else posterior_predictive[var]

    def predict_local_variable(
        self, var: str, num_samples: int = 25, num_split: int = 2048, obs_dim: int = 1, return_mean: bool = False
    ) -> NDArray[np.float32]:
        """
        Sample local variables from the posterior. In order to
        avoid memory issues, the sampling is performed in batches.

        Parameters
        ----------
        var
            Name of the variable to sample.
        num_samples
            Number of samples to draw.
        num_split
            The parameter determines the size of the batches. The actual
            batch size is total number of observations divided by num_split.
        obs_dim
            The dimension of the observations. After sampling, the output
            is concatenated along this dimension.
        return_mean
            If true posterior samples are averaged directly after sampling.
        """
        split_obs = torch.split(self.idx, num_split)
        obs_dim = 0 if return_mean else obs_dim
        # create status bar
        pbar = tqdm(range(len(split_obs)))

        results = []
        for i in pbar:
            posterior_predictive = self.predict([var], num_samples=num_samples, idx=split_obs[i])
            post_pred = posterior_predictive[var].mean(0) if return_mean else posterior_predictive[var]
            results.append(post_pred)
            # update status bar
            pbar.set_description(f"Predicting {var} for obs {split_obs[i].min()}-{split_obs[i].max()}.")

        return np.concatenate(results, obs_dim)

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np  # type: ignore
import pyro  # type: ignore
import torch
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
        checkpoint_freq: int = 500,
        to_numpy: bool = True,
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
            checkpoint_freq=checkpoint_freq,
            to_numpy=to_numpy,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
            loss_kwargs=loss_kwargs,
        )
        self.predict_model = predict_model
        self.predict_guide = predict_guide
        self.idx = idx

    def predict(
        self, return_sites: List[str], num_samples: int = 25, *args: Union[str, List[str]], **kwargs: Any
    ) -> None:
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
        self.posterior = self._to_numpy(posterior) if self.to_numpy else posterior
        empty_cache()

    def predict_global_variable(self, var: str, num_samples: int = 25) -> np.ndarray:
        """
        Sample global variables from the posterior.

        Parameters
        ----------
        var : str
            Name of the variable to sample.
        num_samples : int
            Number of samples to draw.
        """

        self.predict([var], num_samples=num_samples, idx=self.idx[0:1])

        return self.posterior[var]

    def predict_local_variable(
        self,
        var: str,
        num_samples: int = 25,
        num_split: int = 2048,
        obs_dim: int = 1,
    ) -> np.ndarray:
        """
        Sample local variables from the posterior. In order to
        avoid memory issues, the sampling is performed in batches.

        Parameters
        ----------
        var : str
            Name of the variable to sample.
        num_samples : int
            Number of samples to draw.
        num_split : int
            The parameter determines the size of the batches. The actual
            batch size is total number of observations divided by num_split.
        obs_dim : int
            The dimension of the observations. After sampling, the output
            is concatenated along this dimension.
        """
        split_obs = torch.split(self.idx, num_split)

        # create status bar
        pbar = tqdm(range(len(split_obs)))

        results = []
        for i in pbar:
            self.predict([var], num_samples=num_samples, idx=split_obs[i])
            results.append(self.posterior[var])
            # update status bar
            pbar.set_description(f"Predicting {var} for obs {split_obs[i].min()}-{split_obs[i].max()}.")

        return np.concatenate(results, obs_dim)

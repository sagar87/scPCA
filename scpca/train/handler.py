from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Union

import numpy as np  # type: ignore
import pyro  # type: ignore
import torch
from pyro.infer import SVI, Predictive, Trace_ELBO  # type: ignore
from torch.cuda import empty_cache  # type: ignore
from tqdm import tqdm  # type: ignore


class SVIBaseHandler:
    """
    Helper object that abstracts some of numpyros complexities. Inspired
    by an implementation of Florian Wilhelm.
    :param model: A numpyro model.
    :param guide: A numpyro guide.
    :param loss: Loss function, defaults to Trace_ELBO.
    :param lr: Learning rate, defaults to 0.01.
    :param rng_key: Random seed, defaults to 254.
    :param num_epochs: Number of epochs to train the model, defaults to 5000.
    :param num_samples: Number of posterior samples.
    :param log_func: Logging function, defaults to print.
    :param log_freq: Frequency of logging, defaults to 0 (no logging).
    :param to_numpy: Convert the posterior distribution to numpy array(s),
        defaults to True.
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
        checkpoint_freq: int = -1,
        to_numpy: bool = True,
        optimizer_kwargs: Dict[str, Any] = {"lr": 1e-2},
        scheduler_kwargs: Dict[str, Any] = {"factor": 0.99},
        loss_kwargs: Dict[str, Any] = {"num_particles": 1},
    ):
        pyro.clear_param_store()
        if seed is not None:
            pyro.set_rng_seed(seed)
        self.model = model
        self.guide = guide
        self.loss = loss(**loss_kwargs)
        self.scheduler = False if scheduler is None else True
        self.seed = seed

        if self.scheduler:
            self.optimizer = scheduler(
                {
                    "optimizer": optimizer,
                    "optim_args": optimizer_kwargs,
                    **scheduler_kwargs,
                }
            )
        else:
            self.optimizer = optimizer(optimizer_kwargs)

        self.svi = self._init_svi()
        self.log_freq = log_freq
        self.checkpoint_freq = checkpoint_freq
        self.learning_rates: DefaultDict[str, List[float]] = defaultdict(list)
        self.gradient_norms: DefaultDict[str, List[float]] = defaultdict(list)

        self.checkpoints: Dict[str, torch.Tensor] = {}
        self.num_epochs = num_epochs

        self.loss = None
        self.to_numpy = to_numpy
        self.best_elbo = None
        self.steps = 0

    def _update_state(self, loss: Union[List[float], np.ndarray]) -> None:
        if isinstance(loss, list):
            loss = np.asarray(loss)
        self.loss = loss if self.loss is None else np.concatenate([self.loss, loss])

    def _to_numpy(self, posterior: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in posterior.items()}

    def _init_svi(self) -> SVI:
        """
        Initialises the SVI.
        """
        return SVI(self.model, self.guide, self.optimizer, loss=self.loss)

    def _set_lr(self, lr: float) -> None:
        # this works for CLIPPED adam
        # checl if optim_objs is presetn
        if hasattr(self.optimizer, "optim_objs"):
            for opt in self.optimizer.optim_objs.values():
                for group in opt.param_groups:
                    group["lr"] = lr
        else:
            pass
            # for group in self.optimizer.optim_objs.values():
            #     group['lr'] = lr

    def _get_learning_rate(self) -> Any:
        """
        Extracts the learning rate from the first parameter.
        TODO: return a dict(lr: [params])
        """
        for name, param in self.optimizer.get_state().items():
            if "optimizer" in param:
                lr = param["optimizer"]["param_groups"][0]["lr"]
            else:
                lr = param["param_groups"][0]["lr"]
            break
        return lr

    def _track_learning_rate(self) -> None:
        """
        Tracks the learning rate during training.
        """
        for name, param in self.optimizer.get_state().items():
            if "optimizer" in param:
                self.learning_rates[name].append(param["optimizer"]["param_groups"][0]["lr"])
            else:
                self.learning_rates[name].append(param["param_groups"][0]["lr"])

    def _track_gradient_norms(self) -> None:
        # Register hooks to monitor gradient norms.

        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(lambda g, name=name: self.gradient_norms[name].append(g.norm().item()))

    def _fit(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> np.ndarray:
        losses = []
        pbar = tqdm(range(self.steps, self.steps + self.num_epochs))
        failure = False

        previous_elbo = 0
        best_elbo = np.inf if self.best_elbo is None else self.best_elbo
        delta = 0

        try:
            for i in pbar:
                current_elbo = self.svi.step(*args, **kwargs)
                losses.append(current_elbo)

                if i == 0:
                    self._track_gradient_norms()
                    best_elbo = current_elbo
                else:
                    improvement = best_elbo - current_elbo
                    if improvement > 0:
                        best_elbo = current_elbo

                if i % self.log_freq == 0:
                    lr = self._get_learning_rate()
                    pbar.set_description(
                        f"Epoch: {i} | lr: {lr:.2E} | ELBO: {current_elbo:.0f} | Δ_{self.log_freq}: {delta:.2f} | Best: {best_elbo:.0f}"
                    )
                    if i > 0:
                        delta = previous_elbo - current_elbo
                    previous_elbo = current_elbo

                if self.checkpoint_freq > 0 and i % self.checkpoint_freq == 0:
                    self.checkpoints[i] = pyro.get_param_store().get_state()

                if self.scheduler:
                    if issubclass(
                        self.optimizer.pt_scheduler_constructor,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    ):
                        self.optimizer.step(current_elbo)
                    else:
                        self.optimizer.step()

        except KeyboardInterrupt:
            # TODO: use warning
            print("Stoping training ...")
            failure = True

        if failure:
            self.steps += i
        else:
            self.steps += self.num_epochs

        # update max elbo
        self.best_elbo = best_elbo

        return losses

    def fit(
        self, num_epochs: Optional[int] = None, lr: Optional[float] = None, *args: torch.Tensor, **kwargs: torch.Tensor
    ) -> None:
        if num_epochs is not None:
            self.num_epochs = num_epochs

        if lr is not None:
            self._set_lr(lr)

        loss = self._fit(*args, **kwargs)
        self._update_state(loss)
        self.params = pyro.get_param_store().get_state()

    def predict(
        self, return_sites: List[str], num_samples: int = 25, *args: Union[str, List[str]], **kwargs: Any
    ) -> None:
        if self.params is not None:
            pyro.clear_param_store()
            pyro.get_param_store().set_state(self.params)

        predictive = Predictive(
            self.model,
            guide=self.guide,
            num_samples=num_samples,
            return_sites=return_sites,
        )
        posterior = predictive(*args, **kwargs)
        self.posterior = self._to_numpy(posterior) if self.to_numpy else posterior
        empty_cache()

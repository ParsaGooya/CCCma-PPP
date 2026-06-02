import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import dataclasses
from loss.loss_abc import lossABC

from cccma_ppp.models.normalized_flows import NormalizedFlowModel


@dataclasses.dataclass
class BetaAnnealing:
    """
    Schedule for annealing the KL-divergence weight (beta) during training.
    """

    beta: float = 1
    beta_min: float = 0
    num_epoch_to_warmup: int = 0
    num_epochs_to_hold: int = 0

    def __post_init__(self):
        """
        Validate annealing configuration.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If beta_min is invalid.
        """

        self.built = False
        assert self.beta_min >= 0
        if self.num_epoch_to_warmup == 0:
            self.num_epochs_to_hold = 0

    def build(self, num_batches):
        """
        Initialize annealing schedule based on number of batches.

        Parameters
        ----------
        num_batches : int
            Number of batches per epoch.

        Returns
        -------
        None
        """

        self.num_batches = num_batches
        if self.num_epochs_to_hold == 0:
            self.range_epochs = (self.num_epoch_to_warmup) * self.num_batches
        else:
            self.range_epochs = (
                self.num_epoch_to_warmup + self.num_epochs_to_hold
            ) * self.num_batches

        if self.range_epochs == 0:
            self.range_epochs = 1e-20

        self.built = True

    def __call__(self, step):
        """
        Compute beta value at a given training step.

        Parameters
        ----------
        step : int
            Current training step.

        Returns
        -------
        float
            Annealed beta value.

        Raises
        ------
        AssertionError
            If scheduler is not initialized.
        """

        assert self.built, "make sure beta finder has is built."

        if self.num_epochs_to_hold == 0:
            return self.beta_min + (self.beta - self.beta_min) * min(
                (step / self.range_epochs), 1
            )
        else:
            cycle_pos = step % self.range_epochs
            return self.beta_min + (self.beta - self.beta_min) * min(
                (cycle_pos / (self.num_epoch_to_warmup * self.num_batches)), 1
            )


class KLD(lossABC):
    """
    Kullback-Leibler divergence loss with optional flow-based prior.
    """

    def __init__(self, reduction="mean", prior_flow=None):
        """
        Initialize KLD loss.

        Parameters
        ----------
        reduction : str, optional
            Reduction method ('mean' or 'sum').
        prior_flow : NormalizedFlowModel, optional
            Flow-based prior model.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If reduction is incompatible with flow-based prior.
        """

        super().__init__()
        self.reduction = reduction
        self.prior_flow = prior_flow
        if self.prior_flow is not None:
            assert self.reduction.lower() == "sum", (
                "reduction must be sum for prior flow to work."
            )

    def forward(
        self,
        mu,
        log_var,
        cond_mu=None,
        cond_log_var=None,
        print_loss=False,
    ):
        """
        Compute KL divergence between posterior and prior distributions.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of posterior distribution.
        log_var : torch.Tensor
            Log variance of posterior distribution.
        cond_mu : torch.Tensor, optional
            Mean of conditional prior.
        cond_log_var : torch.Tensor, optional
            Log variance of conditional prior.
        print_loss : bool, optional
            Whether to print loss value.

        Returns
        -------
        torch.Tensor
            Computed KL divergence.

        Raises
        ------
        AssertionError
            If input tensor shapes are inconsistent.
        """

        assert mu.shape == log_var.shape
        var = torch.exp(log_var) + 1e-4

        if self.prior_flow is None:
            if all([cond_mu is not None, cond_log_var is not None]):
                assert cond_mu.shape == cond_log_var.shape
                cond_var = torch.exp(cond_log_var) + 1e-4
                KLD = kl_divergence(
                    Normal(mu, torch.sqrt(var)), Normal(cond_mu, torch.sqrt(cond_var))
                )

            else:
                KLD = kl_divergence(
                    Normal(mu, torch.sqrt(var)),
                    Normal(torch.zeros_like(mu), torch.ones_like(log_var)),
                )

        else:
            opts = dict(device=mu.device, dtype=mu.dtype)
            base_dist = Normal(torch.zeros_like(mu), torch.ones_like(log_var))

            posterior_samples = self.sample(
                mu, torch.sqrt(var), sample_size=self.prior_flow.flow_sample_size
            )
            posterior_logprob = (
                Normal(mu, torch.sqrt(var))
                .log_prob(posterior_samples)
                .mean(0)
                .sum(-1)
                .to(**opts)
            )

            if cond_mu is None:
                condition = None
            else:
                # condition = cond_mu
                # condition = condition.unsqueeze(0).expand(*posterior_samples.shape[0:-1], condition.shape[-1])
                condition = cond_mu.unsqueeze(0).expand(
                    posterior_samples.shape[0], *cond_mu.shape
                )
                condition = torch.flatten(condition, start_dim=0, end_dim=1)

            # e_samples, log_det =
            flow_output = self.prior_flow(
                torch.flatten(posterior_samples, start_dim=0, end_dim=1),
                condition=condition,
            )

            e_samples = torch.unflatten(
                flow_output.e_samples, dim=0, sizes=posterior_samples.shape[:2]
            ).to(**opts)
            prior_logprob = base_dist.log_prob(e_samples).mean(0).sum(-1).to(**opts)

            log_det = (
                torch.unflatten(
                    flow_output.log_det, dim=0, sizes=posterior_samples.shape[:2]
                )
                .mean(0)
                .to(**opts)
            )
            KLD = (posterior_logprob - prior_logprob - log_det).mean()

        KLD = self._aggregate(KLD)
        if print_loss:
            self._print_loss(KLD)

        return KLD

    def _aggregate(self, loss):
        """
        Apply reduction to KL divergence values.

        Parameters
        ----------
        loss : torch.Tensor
            Raw KL divergence values.

        Returns
        -------
        torch.Tensor
            Reduced loss.
        """

        if self.prior_flow is None:
            if self.reduction == "mean":
                KLD = loss.mean()
            if self.reduction == "sum":
                KLD = loss.sum(dim=-1).mean()
        else:
            KLD = loss.clamp(0)

        return KLD

    def _print_loss(self, loss):
        """
        Print formatted KL divergence loss.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value.

        Returns
        -------
        None
        """

        print(f"KLD : {loss.item():.5f}")

    def sample(self, mu, log_var, sample_size=1, std=1):
        """
        Sample from posterior distribution.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of distribution.
        log_var : torch.Tensor
            Log variance of distribution.
        sample_size : int, optional
            Number of samples.
        std : float, optional
            Standard deviation scaling.

        Returns
        -------
        torch.Tensor
            Sampled values.
        """

        var = torch.exp(log_var) + 1e-4
        out = mu + torch.sqrt(var) * self._get_normal(var, std).sample((sample_size,))

        return out

    def _get_normal(self, ref_tensor, std=1):
        """
        Create a standard normal distribution matching a reference tensor.

        Parameters
        ----------
        ref_tensor : torch.Tensor
            Reference tensor for shape.
        std : float, optional
            Standard deviation.

        Returns
        -------
        torch.distributions.Normal
            Normal distribution instance.
        """

        return torch.distributions.Normal(
            torch.zeros_like(ref_tensor), torch.ones_like(ref_tensor) * std
        )

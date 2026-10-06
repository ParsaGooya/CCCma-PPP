import torch
from torch.distributions import Normal, kl_divergence
import dataclasses

from cccma_ppp.loss.loss_abc import lossABC, Reduction


from cccma_ppp.architectures.normalized_flows import NormalizedFlowModel


@dataclasses.dataclass
class BetaAnnealing:
    """
    Configure annealing of the Kullback-Leibler divergence weight.

    Parameters
    ----------
    beta : float
        Target beta value reached after warmup.
    beta_min : float
        Initial beta value used at the beginning of each warmup period.
    num_epoch_to_warmup : int
        Number of epochs over which beta increases from ``beta_min`` to ``beta``.
    num_epochs_to_hold : int
        Number of epochs for which beta remains at its target value before a new
        annealing cycle begins.
    """

    beta: float = 1
    beta_min: float = 0
    num_epoch_to_warmup: int = 0
    num_epochs_to_hold: int = 0

    def __post_init__(self):
        """
        Initialize and validate the beta annealing configuration.

        When no warmup is configured, the hold period is disabled.

        Raises
        ------
        AssertionError
            If ``beta_min`` is negative.
        """
        self.built = False
        assert self.beta_min >= 0
        if self.num_epoch_to_warmup == 0:
            self.num_epochs_to_hold = 0

    def build(self, num_batches):
        """
        Initialize the beta schedule for a training data loader.

        Epoch-based warmup and hold durations are converted to batch-step durations.

        Parameters
        ----------
        num_batches : int
            Number of training batches in each epoch.
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

    def __call__(self, step: int):
        """
        Return the beta value for a training step.

        Without a hold period, beta increases linearly from ``beta_min`` to ``beta``
        and remains at ``beta``. With a hold period, the schedule repeats cyclically,
        with linear warmup followed by a period at the target beta value.

        Parameters
        ----------
        step : int
            Current training step.

        Returns
        -------
        float
            Beta value associated with the training step.

        Raises
        ------
        AssertionError
            If the annealing schedule has not been built before use.
        """
        assert self.built, "make sure beta finder has been built."

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
    Compute Kullback-Leibler divergence for latent distributions.

    Supports divergence against a standard normal prior, a condition-dependent
    normal prior, or a prior represented by a normalizing flow.

    Parameters
    ----------
    reduction : Reduction
        Method used to aggregate latent Kullback-Leibler divergence values.
    """

    def __init__(
        self,
        reduction: Reduction = "mean",
    ):
        """
        Initialize the Kullback-Leibler divergence loss.

        Parameters
        ----------
        reduction : Reduction
            Method used to aggregate latent divergence values.
        """
        super().__init__()
        self.reduction = reduction
        self._has_prior_flow = False

    def forward(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        cond_mu: torch.Tensor | None = None,
        cond_log_var: torch.Tensor | None = None,
        prior_flow: NormalizedFlowModel = None,
        print_loss=False,
    ) -> torch.Tensor:
        """
        Compute the Kullback-Leibler divergence for a latent distribution.

        When conditional distribution parameters are provided, the posterior is
        compared with the conditional Gaussian prior. Without conditional parameters,
        a standard normal prior is used. When a normalizing flow is provided, the
        divergence is estimated from posterior samples and the flow Jacobian.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the posterior Gaussian distribution.
        log_var : torch.Tensor
            Log-variance of the posterior Gaussian distribution.
        cond_mu : torch.Tensor | None
            Mean of the conditional Gaussian prior.
        cond_log_var : torch.Tensor | None
            Log-variance of the conditional Gaussian prior.
        prior_flow : NormalizedFlowModel | None
            Optional normalizing flow defining the latent prior.
        print_loss : bool
            Whether to print the aggregated Kullback-Leibler divergence.

        Returns
        -------
        torch.Tensor
            Aggregated Kullback-Leibler divergence.

        Raises
        ------
        AssertionError
            If ``mu`` and ``log_var`` have different shapes, or if both conditional
            distribution parameters are provided with incompatible shapes.
        """
        if prior_flow is not None:
            self._has_prior_flow = True

        assert mu.shape == log_var.shape
        var = torch.exp(log_var) + 1e-4

        if prior_flow is None:
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
                mu, torch.sqrt(var), sample_size=prior_flow.flow_sample_size
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
                condition = cond_mu.unsqueeze(0).expand(
                    posterior_samples.shape[0], *cond_mu.shape
                )
                condition = torch.flatten(condition, start_dim=0, end_dim=1)

            flow_output = prior_flow(
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
        Aggregate Kullback-Leibler divergence values.

        For analytic Gaussian divergence, values are reduced according to the
        configured reduction method. Flow-based estimates are clamped to be
        non-negative.

        Parameters
        ----------
        loss : torch.Tensor
            Unaggregated or flow-estimated Kullback-Leibler divergence.

        Returns
        -------
        torch.Tensor
            Aggregated Kullback-Leibler divergence.
        """
        if not self._has_prior_flow:
            if self.reduction == "mean":
                KLD = loss.mean()
            if self.reduction == "sum":
                KLD = loss.sum(dim=-1).mean()
        else:
            KLD = loss.clamp(0)

        return KLD

    def _print_loss(self, loss):
        """
        Print the Kullback-Leibler divergence.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar Kullback-Leibler divergence to print.
        """
        print(f"KLD : {loss.item():.5f}")

    def sample(self, mu, log_var, sample_size=1, std=1):
        """
        Draw samples from a Gaussian distribution.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the Gaussian distribution.
        log_var : torch.Tensor
            Log-variance parameter used to construct the sampling distribution.
        sample_size : int
            Number of samples to draw.
        std : float
            Standard deviation of the normal noise used for sampling.

        Returns
        -------
        torch.Tensor
            Samples drawn from the Gaussian distribution.
        """
        var = torch.exp(log_var) + 1e-4
        out = mu + torch.sqrt(var) * self._get_normal(var, std).sample((sample_size,))

        return out

    def _get_normal(self, ref_tensor, std=1):
        """
        Construct a zero-mean normal distribution matching a reference tensor.

        Parameters
        ----------
        ref_tensor : torch.Tensor
            Reference tensor defining the shape, data type, and device of the
            distribution parameters.
        std : float
            Standard deviation of the normal distribution.

        Returns
        -------
        torch.distributions.Normal
            Zero-mean normal distribution with the requested standard deviation.
        """
        return torch.distributions.Normal(
            torch.zeros_like(ref_tensor), torch.ones_like(ref_tensor) * std
        )

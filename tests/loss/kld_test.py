import pytest
import torch


from dataclasses import dataclass
from cccma_ppp.loss.kld import BetaAnnealing, KLD


@dataclass
class DummyCvaeOutput:
    output: object = None
    mu: object = None
    log_var: object = None
    cond_mu: object = None
    cond_log_var: object = None


def test_beta_no_warmup():
    beta = BetaAnnealing(beta=1.0, beta_min=0.0, num_epoch_to_warmup=0)
    beta.build(num_batches=10)

    assert beta(0) == pytest.approx(0.0)

    assert beta(1) == pytest.approx(1.0)


@pytest.mark.pruned
def test_beta_linear_increase():
    beta = BetaAnnealing(beta=1.0, beta_min=0.0, num_epoch_to_warmup=2)
    beta.build(num_batches=10)

    assert beta(0) == pytest.approx(0.0)
    assert beta(10) == pytest.approx(0.5)
    assert beta(20) == pytest.approx(1.0)


def test_beta_with_hold_cycle():
    beta = BetaAnnealing(
        beta=1.0,
        beta_min=0.0,
        num_epoch_to_warmup=2,
        num_epochs_to_hold=1,
    )
    beta.build(num_batches=10)

    assert beta(5) < 1.0

    assert beta(25) == pytest.approx(1.0)


@pytest.mark.pruned
def test_beta_requires_build():
    beta = BetaAnnealing()
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        beta(0)


@pytest.mark.pruned
def test_kld_standard_normal():
    kld = KLD()

    mu = torch.zeros(10, 5)
    logvar = torch.zeros(10, 5)

    loss = kld(mu, logvar)

    assert loss.item() < 1e-5


@pytest.mark.pruned
def test_kld_nonzero_mean():
    kld = KLD()

    mu = torch.ones(10, 5)
    logvar = torch.zeros(10, 5)

    loss = kld(mu, logvar)

    assert loss.item() > 0


def test_kld_with_condition():
    kld = KLD()

    mu = torch.ones(10, 5)
    logvar = torch.zeros(10, 5)

    cond_mu = torch.zeros(10, 5)
    cond_logvar = torch.zeros(10, 5)

    loss = kld(mu, logvar, cond_mu=cond_mu, cond_log_var=cond_logvar)

    assert loss.item() > 0


@pytest.mark.pruned
def test_kld_shape_mismatch():
    kld = KLD()

    mu = torch.zeros(10, 5)
    logvar = torch.zeros(10, 4)

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        kld(mu, logvar)


@pytest.mark.pruned
def test_kld_sum_reduction():
    kld = KLD(reduction="sum")

    mu = torch.ones(10, 5)
    logvar = torch.zeros(10, 5)

    loss = kld(mu, logvar)

    assert loss.item() > 0


class DummyFlowOutput:
    def __init__(self, e_samples, log_det):
        self.e_samples = e_samples
        self.log_det = log_det


class DummyFlow:
    def __init__(self):
        self.flow_sample_size = 2

    def __call__(self, x, condition=None):

        return DummyFlowOutput(
            e_samples=x,
            log_det=torch.zeros(x.shape[0]),
        )


def test_kld_with_flow():
    kld = KLD()

    mu = torch.zeros(4, 3)
    logvar = torch.zeros(4, 3)

    flow = DummyFlow()

    loss = kld(mu, logvar, prior_flow=flow)

    assert loss.item() >= 0


@pytest.mark.pruned
def test_kld_with_flow_and_condition():
    kld = KLD()

    mu = torch.zeros(4, 3)
    logvar = torch.zeros(4, 3)
    cond_mu = torch.ones(4, 3)

    flow = DummyFlow()

    loss = kld(
        mu,
        logvar,
        cond_mu=cond_mu,
        prior_flow=flow,
    )

    assert loss.item() >= 0


@pytest.mark.pruned
def test_flow_clamps_loss():
    kld = KLD()

    dummy_loss = torch.tensor(-1.0)
    kld._has_prior_flow = True

    result = kld._aggregate(dummy_loss)

    assert result.item() == 0


@pytest.mark.pruned
def test_mean_aggregation():
    kld = KLD(reduction="mean")
    loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = kld._aggregate(loss)

    assert result.item() == pytest.approx(loss.mean().item())


def test_sum_aggregation():
    kld = KLD(reduction="sum")
    loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = kld._aggregate(loss)

    expected = loss.sum(dim=-1).mean()
    assert result.item() == pytest.approx(expected.item())
import pytest
import torch

from cccma_ppp.loss.loss_abc import lossABC


class ValidLoss(lossABC):
    def forward(
        self, data, target, generative_modeling=False, generator=False, print_loss=False
    ):
        loss = (data - target) ** 2
        loss = self._aggregate(loss)

        if print_loss:
            self._print_loss(loss)
        return loss

    def _aggregate(self, loss):
        return loss.mean()

    def _print_loss(self, loss):

        print(loss.item())


class MissingForward(lossABC):
    def _aggregate(self, loss):
        return loss

    def _print_loss(self, loss):
        pass


class MissingAggregate(lossABC):
    def forward(
        self, data, target, generative_modeling=False, generator=False, print_loss=False
    ):
        return data

    def _print_loss(self, loss):
        pass


class MissingPrint(lossABC):
    def forward(
        self, data, target, generative_modeling=False, generator=False, print_loss=False
    ):
        return data

    def _aggregate(self, loss):
        return loss


def test_cannot_instantiate_abstract():

    with pytest.raises(TypeError):
        lossABC()


def test_missing_forward():

    with pytest.raises(TypeError):
        MissingForward()


def test_missing_aggregate():

    with pytest.raises(TypeError):
        MissingAggregate()


def test_missing_print():

    with pytest.raises(TypeError):
        MissingPrint()


def test_valid_subclass_forward():

    model = ValidLoss()

    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[1.0, 1.0]])

    loss = model(x, y)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar


def test_valid_subclass_correct_value():

    model = ValidLoss()

    x = torch.tensor([[2.0, 2.0]])
    y = torch.tensor([[0.0, 0.0]])

    # (2^2 + 2^2) / 2 = 4
    loss = model(x, y)

    assert loss.item() == pytest.approx(4.0)


def test_print_loss(capsys):

    model = ValidLoss()

    x = torch.ones(1, 2)
    y = torch.zeros(1, 2)

    model(x, y, print_loss=True)

    captured = capsys.readouterr()
    assert captured.out.strip() != ""


def test_backward_pass():

    model = ValidLoss()

    x = torch.tensor([[2.0, 2.0]], requires_grad=True)
    y = torch.zeros_like(x)

    loss = model(x, y)
    loss.backward()

    assert x.grad is not None
    assert torch.all(x.grad != 0)

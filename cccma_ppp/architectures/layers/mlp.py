import torch.nn as nn

from cccma_ppp.architectures.layers.generic import ActivationName, _build_activation


def build_mlp(
    dims: list[int],
    *,
    activation: ActivationName = "relu",
    dropout_rate: float | None = None,
    batch_normalization: bool = False,
    activate_final: bool = False,
) -> nn.Sequential:
    """
    Build a multilayer perceptron from a sequence of layer dimensions.

    Each pair of consecutive dimensions defines a linear layer. Activation,
    dropout, and batch normalization are applied after each non-final linear
    layer and optionally after the final layer.

    Parameters
    ----------
    dims : list[int]
        Dimensions of the input, hidden, and output layers.
    activation : ActivationName
        Activation function applied after eligible linear layers.
    dropout_rate : float | None
        Dropout probability applied after each activation. If ``None``, dropout
        is disabled.
    batch_normalization : bool
        Whether to apply batch normalization after activation and dropout.
    activate_final : bool
        Whether to apply activation, dropout, and batch normalization after the
        final linear layer.

    Returns
    -------
    nn.Sequential
        Sequential module containing the constructed multilayer perceptron.
    """
    layers = []

    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        is_final = i == len(dims) - 2

        layers.append(nn.Linear(in_dim, out_dim))

        if not is_final or activate_final:
            layers.append(_build_activation(activation))

            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))

            if batch_normalization:
                layers.append(nn.BatchNorm1d(out_dim))

    return nn.Sequential(*layers)

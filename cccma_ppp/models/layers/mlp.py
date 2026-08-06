import torch.nn as nn

from cccma_ppp.models.layers.generic import ActivationName, _build_activation


def build_mlp(
    dims: list[int],
    *,
    activation: ActivationName = "relu",
    dropout_rate: float | None = None,
    batch_normalization: bool = False,
    activate_final: bool = False,
) -> nn.Sequential:
    """
    Document this function.

    Parameters
    ----------
    dims : list[int]
        Description not yet provided.
    activation : ActivationName
        Description not yet provided.
    dropout_rate : float | None
        Description not yet provided.
    batch_normalization : bool
        Description not yet provided.
    activate_final : bool
        Description not yet provided.

    Returns
    -------
    nn.Sequential
        Description not yet provided.
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

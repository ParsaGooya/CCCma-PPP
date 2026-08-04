import torch
import numpy as np


from cccma_ppp.models.models_abc import (
    modelConfigABC,
    deterministicmodelsABC,
    DeterministicRequest,
)

from cccma_ppp.models.layers.mlp import build_mlp
from cccma_ppp.core.deterministic_module import deterministicOutput


from typing import ClassVar
import dataclasses
from typing import Literal

from cccma_ppp.core.selectors import deterministicModelSelector

from cccma_ppp.models.layers.generic import (
    InitMethod,
    ActivationName,
    _validate_dropout,
)


AppendMode = Literal[1, 2, 3]


@deterministicModelSelector.register("mlp")
@dataclasses.dataclass
class AutoencoderConfig(modelConfigABC):
    encoder_hidden_dims: list
    decoder_hidden_dims: list = None
    batch_normalization: bool = False
    dropout_rate: float = None
    append_mode: AppendMode = 1
    init_method: InitMethod = "trunc_normal"
    activation: ActivationName = "relu"

    NUM_INPUT_DIMS: ClassVar[int] = 2
    NUM_OUTPUT_DIMS: ClassVar[int] = 2
    GENERATOR: ClassVar[None] = None

    def __post_init__(self):

        _validate_dropout(self.dropout_rate)

        if self.decoder_hidden_dims is None:
            if len(self.encoder_hidden_dims) == 1:
                self.decoder_hidden_dims = []
            else:
                self.decoder_hidden_dims = self.encoder_hidden_dims[::-1][1:]

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):

        return Autoencoder(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class Autoencoder(deterministicmodelsABC):
    def __init__(
        self,
        config: AutoencoderConfig,
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple | None = None,
        added_features_dim: int = None,
    ):

        super().__init__()

        self.config = config
        self.batch_normalization = config.batch_normalization
        self.dropout_rate = config.dropout_rate
        self.init_method = config.init_method
        self.append_mode = config.append_mode
        self.latent_size = config.encoder_hidden_dims[-1]
        self.encoder_hidden_dims = config.encoder_hidden_dims
        self.decoder_hidden_dims = config.decoder_hidden_dims

        if output_shape is None:
            output_shape = input_shape.copy()

        if not len(output_shape) == self.config.NUM_OUTPUT_DIMS:
            raise RuntimeError(
                f"MLP models should create {self.config.NUM_OUTPUT_DIMS}D outputs"
            )

        self._validate_checkpoint_compatibility(
            input_shape=input_shape,
            output_shape=output_shape,
        )

        self.input_shape = np.prod(input_shape)
        self.output_shape = np.prod(output_shape)
        self.added_features_dim = added_features_dim

        if self.added_features_dim is None:
            self.added_features_dim = 0

        if (self.append_mode == 2) or (self.append_mode == 3):
            decoder_dims = [
                self.latent_size + self.added_features_dim,
                *self.decoder_hidden_dims,
                self.output_shape,
            ]
        else:
            decoder_dims = [
                self.latent_size,
                *self.decoder_hidden_dims,
                self.output_shape,
            ]

        if (self.append_mode == 1) or (self.append_mode == 3):
            encoder_dims = [
                self.input_shape + self.added_features_dim,
                *self.encoder_hidden_dims,
            ]
        else:
            encoder_dims = [self.input_shape, *self.encoder_hidden_dims]

        self.encoder = build_mlp(
            encoder_dims,
            activation=self.config.activation,
            dropout_rate=self.dropout_rate,
            batch_normalization=self.batch_normalization,
            activate_final=True,
        )

        self.decoder = build_mlp(
            decoder_dims,
            activation=self.config.activation,
            dropout_rate=self.dropout_rate,
            batch_normalization=self.batch_normalization,
            activate_final=False,
        )

        if self.config.checkpoint_config is not None:
            self._load_state_dict(self.config.checkpoint_config)
        else:
            self._initialize_weights(self.init_method)

    def forward(self, request: DeterministicRequest) -> deterministicOutput:
        x = request.input
        x_mask = request.input_mask
        added_features = request.added_features

        if x_mask is not None:
            x = x * x_mask

        B, C = x.shape[:2]
        x = x.flatten(start_dim=1)

        if added_features is not None:
            x_features = added_features.flatten(start_dim=1)
        else:
            x_features = None

        if x_features is not None:
            if self.append_mode == 1:
                out = self.encoder(torch.cat([x, x_features], dim=-1))
                out = self.decoder(out)

            elif self.append_mode == 2:
                out = self.encoder(x)
                out = self.decoder(torch.cat([out, x_features], dim=-1))

            elif self.append_mode == 3:
                out = self.encoder(torch.cat([x, x_features], dim=-1))
                out = self.decoder(torch.cat([out, x_features], dim=-1))

        else:
            out = self.encoder(x)
            out = self.decoder(out)

        return deterministicOutput(output=out.view(B, C, -1))

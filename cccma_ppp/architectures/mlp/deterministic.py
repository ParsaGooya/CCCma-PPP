import torch
import numpy as np


from cccma_ppp.architectures.models_abc import (
    modelConfigABC,
    deterministicmodelsABC,
    DeterministicRequest,
)

from cccma_ppp.architectures.layers.mlp import build_mlp
from cccma_ppp.core.models.deterministic import deterministicOutput


from typing import ClassVar
import dataclasses
from typing import Literal

from cccma_ppp.core.selectors import deterministicModelSelector

from cccma_ppp.architectures.layers.generic import (
    InitMethod,
    ActivationName,
    _validate_dropout,
)


AppendMode = Literal[1, 2, 3]


@deterministicModelSelector.register("mlp")
@dataclasses.dataclass
class AutoencoderConfig(modelConfigABC):
    """
    Configuration for a deterministic multilayer perceptron autoencoder.

    Parameters
    ----------
    encoder_hidden_dims : list
        Sizes of the hidden layers in the encoder. The final dimension defines
        the size of the latent representation.
    decoder_hidden_dims : list | None
        Sizes of the hidden layers in the decoder. If ``None``, they are derived
        from ``encoder_hidden_dims`` in reverse order.
    batch_normalization : bool
        Whether to apply batch normalization in the encoder and decoder.
    dropout_rate : float | None
        Dropout probability used in the encoder and decoder. If ``None``,
        dropout is disabled.
    append_mode : AppendMode
        Controls where additional features are concatenated: ``1`` for the
        encoder input, ``2`` for the decoder input, or ``3`` for both.
    init_method : InitMethod
        Weight initialization method used when no checkpoint is loaded.
    activation : ActivationName
        Activation function used in the encoder and decoder.
    """

    encoder_hidden_dims: list
    decoder_hidden_dims: list = None
    batch_normalization: bool = False
    dropout_rate: float = None
    append_mode: AppendMode = 1
    init_method: InitMethod = "default"
    activation: ActivationName = "relu"

    NUM_INPUT_DIMS: ClassVar[int] = 2
    NUM_OUTPUT_DIMS: ClassVar[int] = 2

    GENERATOR: ClassVar[None] = None

    def __post_init__(self):
        """
        Validate and complete the autoencoder configuration.

        If decoder hidden dimensions are not provided, they are derived from the
        encoder hidden dimensions in reverse order.

        Raises
        ------
        ValueError
            If ``dropout_rate`` is not ``None`` and is outside the interval [0, 1].
        """
        _validate_dropout(self.dropout_rate)

        if self.decoder_hidden_dims is None:
            if len(self.encoder_hidden_dims) == 1:
                self.decoder_hidden_dims = []
            else:
                self.decoder_hidden_dims = self.encoder_hidden_dims[::-1][1:]

    @property
    def EXPECTS_MASK(self) -> bool:
        """
        Indicate whether the model requires mask inputs.

        Returns
        -------
        bool
            ``False``, since masks are optional for the MLP autoencoder.
        """
        return False

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Build an autoencoder from this configuration.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of the model input excluding the batch dimension.
        output_shape : np.ndarray | None
            Shape of the model output excluding the batch dimension. If ``None``,
            ``input_shape`` is used.
        added_features_dim : int | None
            Number of additional features that may be concatenated with the encoder
            input, decoder input, or both according to ``append_mode``.

        Returns
        -------
        Autoencoder
            Constructed deterministic MLP autoencoder.
        """
        return Autoencoder(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class Autoencoder(deterministicmodelsABC):
    """
    Deterministic multilayer perceptron autoencoder.

    The model flattens its input, encodes it into a latent representation, and
    decodes the representation into the requested output shape. Additional
    features can be provided to the encoder, decoder, or both.

    Parameters
    ----------
    config : AutoencoderConfig
        Configuration defining the encoder, decoder, feature concatenation, and
        weight initialization.
    input_shape : np.ndarray | tuple
        Shape of the model input excluding the batch dimension.
    output_shape : np.ndarray | tuple | None
        Shape of the model output excluding the batch dimension. If ``None``,
        ``input_shape`` is used.
    added_features_dim : int | None
        Number of additional features supplied to the model.
    """

    def __init__(
        self,
        config: AutoencoderConfig,
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple | None = None,
        added_features_dim: int = None,
    ):
        """
        Initialize a deterministic MLP autoencoder.

        Parameters
        ----------
        config : AutoencoderConfig
            Configuration defining the encoder, decoder, feature concatenation, and
            weight initialization.
        input_shape : np.ndarray | tuple
            Shape of the model input excluding the batch dimension.
        output_shape : np.ndarray | tuple | None
            Shape of the model output excluding the batch dimension. If ``None``,
            ``input_shape`` is used.
        added_features_dim : int | None
            Number of additional features supplied to the model.

        Raises
        ------
        RuntimeError
            If ``output_shape`` does not have the number of dimensions required by
            the MLP model.
        """
        super().__init__(config)

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
        """
        Encode an input and reconstruct it through the decoder.

        The input is optionally masked before being flattened. Additional features
        are concatenated with the encoder input, decoder input, or both according to
        ``append_mode``.

        Parameters
        ----------
        request : DeterministicRequest
            Request containing the model input, optional input mask, and optional
            additional features.

        Returns
        -------
        deterministicOutput
            Model output containing the decoded tensor reshaped to preserve the
            input batch and channel dimensions.
        """
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

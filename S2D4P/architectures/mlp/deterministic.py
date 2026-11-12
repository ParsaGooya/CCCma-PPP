import torch
import numpy as np


from S2D4P.architectures.architecture_abc import (
    ArchConfigABC,
    deterministicArchABC,
    DeterministicRequest,
)

from S2D4P.architectures.layers.mlp import build_mlp
from S2D4P.core.models.deterministic import deterministicOutput


from typing import ClassVar
import dataclasses
from typing import Literal

from S2D4P.core.selectors import deterministicArchSelector

from S2D4P.architectures.layers.generic import (
    InitMethod,
    ActivationName,
    _validate_dropout,
)


AppendMode = Literal[1, 2, 3]


@deterministicArchSelector.register("mlp")
@dataclasses.dataclass
class AutoencoderConfig(ArchConfigABC):
    """
    Configuration for an MLP-based deterministic autoencoder.

    Parameters
    ----------
    encoder_hidden_dims : list
        Hidden-layer sizes used by the encoder. The final value determines the
        latent representation size.
    decoder_hidden_dims : list | None
        Hidden-layer sizes used by the decoder. If ``None``, they are derived
        from the encoder hidden dimensions in reverse order.
    batch_normalization : bool
        Whether MLP layers use batch normalization.
    dropout_rate : float | None
        Dropout probability applied within MLP layers. If ``None``, dropout is
        disabled.
    append_mode : AppendMode
        Controls where additional features are concatenated. Mode 1 adds them to
        the encoder input, mode 2 adds them to the decoder input, and mode 3 adds
        them to both.
    init_method : InitMethod
        Weight initialization method used when a checkpoint is not loaded.
    activation : ActivationName
        Activation function used by the encoder and decoder.
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
        Resolve defaults and validate the autoencoder configuration.

        When decoder dimensions are omitted, they are derived from the encoder hidden
        dimensions in reverse order.
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
        Indicate whether the architecture expects spatial masks.

        Returns
        -------
        bool
            ``False`` because the MLP architecture does not require spatial masks.
        """
        return False

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Build an MLP-based deterministic autoencoder.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of each model input excluding the batch dimension.
        output_shape : np.ndarray | None
            Shape of each model output excluding the batch dimension. If ``None``,
            the input shape is used.
        added_features_dim : int | None
            Number of additional features supplied alongside each sample.

        Returns
        -------
        Autoencoder
            Constructed deterministic autoencoder.
        """
        return Autoencoder(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class Autoencoder(deterministicArchABC):
    """
    MLP-based deterministic autoencoder architecture.

    The architecture flattens each input sample, encodes it into a latent
    representation, and decodes that representation into the requested output
    shape. Additional features can be supplied to the encoder, decoder, or both.

    Parameters
    ----------
    config : AutoencoderConfig
        Configuration defining the encoder, decoder, feature-appending behavior,
        and initialization.
    input_shape : np.ndarray | tuple
        Shape of each model input excluding the batch dimension.
    output_shape : np.ndarray | tuple | None
        Shape of each model output excluding the batch dimension. If ``None``,
        the input shape is used.
    added_features_dim : int | None
        Number of additional features supplied alongside each sample.
    """

    def __init__(
        self,
        config: AutoencoderConfig,
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple | None = None,
        added_features_dim: int = None,
    ):
        """
        Initialize the MLP-based deterministic autoencoder.

        Encoder and decoder dimensions are constructed according to ``append_mode``.
        Model parameters are restored from a configured checkpoint when available;
        otherwise, the requested initialization method is applied.

        Parameters
        ----------
        config : AutoencoderConfig
            Configuration defining the architecture.
        input_shape : np.ndarray | tuple
            Shape of each model input excluding the batch dimension.
        output_shape : np.ndarray | tuple | None
            Shape of each model output excluding the batch dimension. If ``None``,
            the input shape is used.
        added_features_dim : int | None
            Number of additional features supplied alongside each sample.

        Raises
        ------
        RuntimeError
            If ``output_shape`` does not have the number of dimensions required by
            the MLP architecture.
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
        Run the deterministic autoencoder forward pass.

        An optional input mask is applied before the input is flattened. Additional
        features are then concatenated to the encoder input, decoder input, or both
        according to ``append_mode``.

        Parameters
        ----------
        request : DeterministicRequest
            Forward request containing the model input, optional input mask, and
            optional additional features.

        Returns
        -------
        deterministicOutput
            Deterministic model output reshaped to batch, channel, and flattened
            output dimensions.
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

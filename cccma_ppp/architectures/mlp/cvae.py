import torch
import torch.nn as nn
import numpy as np
from typing import ClassVar
import dataclasses

from cccma_ppp.architectures.models_abc import (
    cVAEmodelsABC,
    cVAEmodelConfigABC,
    cVAEForwardRequest,
    cVAEPredictRequest,
)
from cccma_ppp.architectures.layers.mlp import build_mlp
from cccma_ppp.core.models.cvae import cVAEOutput

from cccma_ppp.architectures.layers.generic import (
    InitMethod,
    ActivationName,
    _validate_dropout,
)

from cccma_ppp.core.selectors import cVAEModelSelector


@cVAEModelSelector.register("mlp")
@dataclasses.dataclass
class cVAE_MLPConfig(cVAEmodelConfigABC):
    """
    Configuration for a multilayer perceptron conditional variational autoencoder.

    Parameters
    ----------
    encoder_hidden_dims : list
        Sizes of the hidden layers in the recognition network.
    latent_size : int
        Size of the latent representation.
    condition_embedding_dims : list | None
        Sizes of the hidden layers used to encode the conditioning input. If
        ``None``, ``encoder_hidden_dims`` is used.
    condition_embedding_size : int | None
        Size of the encoded conditioning representation. If ``None``,
        ``latent_size`` is used.
    decoder_hidden_dims : list | None
        Sizes of the hidden layers in the decoder. If ``None``, they are derived
        from ``encoder_hidden_dims`` in reverse order.
    condition_dependant_latent : bool
        Whether the latent prior depends on the conditioning input.
    condemb_to_decoder : bool
        Whether to concatenate the condition embedding with latent samples before
        decoding.
    batch_normalization : bool
        Whether to apply batch normalization in the MLP networks.
    dropout_rate : float | None
        Dropout probability used in the MLP networks. If ``None``, dropout is
        disabled.
    init_method : InitMethod
        Weight initialization method used when no checkpoint is loaded.
    activation : ActivationName
        Activation function used in the MLP networks.
    """

    encoder_hidden_dims: list
    latent_size: int
    condition_embedding_dims: list | None = None
    condition_embedding_size: int | None = None
    decoder_hidden_dims: list = None
    condition_dependant_latent: bool = False
    condemb_to_decoder: bool = True
    batch_normalization: bool = False
    dropout_rate: float = None
    init_method: InitMethod = "default"
    activation: ActivationName = "relu"

    NUM_INPUT_DIMS: ClassVar[int] = 2
    NUM_OUTPUT_DIMS: ClassVar[int] = 2

    GENERATOR: ClassVar[None] = None

    def __post_init__(self):
        """
        Validate and complete the MLP cVAE configuration.

        Missing decoder dimensions, condition embedding dimensions, and condition
        embedding size are inferred from the encoder and latent configuration.

        Raises
        ------
        ValueError
            If ``dropout_rate`` is outside the interval [0, 1], or if the condition
            embedding is excluded from the decoder while using a condition-independent
            latent representation.
        """
        _validate_dropout(self.dropout_rate)

        if self.decoder_hidden_dims is None:
            if len(self.encoder_hidden_dims) == 0:
                self.decoder_hidden_dims = []
            else:
                self.decoder_hidden_dims = self.encoder_hidden_dims[::-1][1:]

        if self.condition_embedding_dims is None:
            self.condition_embedding_dims = self.encoder_hidden_dims

        if self.condition_embedding_size is None:
            self.condition_embedding_size = self.latent_size

        if not self.condition_dependant_latent:
            if not self.condemb_to_decoder:
                raise ValueError(
                    "condition embedding has to be passed to decoder for cVAE when latent is not condition dependant."
                )

    @property
    def EXPECTS_MASK(self) -> bool:
        """
        Indicate whether the model requires mask inputs.

        Returns
        -------
        bool
            ``False``, since masks are optional for the MLP cVAE.
        """
        return False

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Build an MLP conditional variational autoencoder from this configuration.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of the conditioning input excluding the batch dimension.
        output_shape : np.ndarray | None
            Shape of the model output excluding the batch dimension. If ``None``,
            ``input_shape`` is used.
        added_features_dim : int | None
            Number of additional conditioning features supplied to the model.

        Returns
        -------
        cVAE_MLP
            Constructed MLP conditional variational autoencoder.
        """
        return cVAE_MLP(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class cVAE_MLP(cVAEmodelsABC):
    """
    Multilayer perceptron conditional variational autoencoder.

    The model encodes conditioning inputs into an embedding, infers posterior
    latent parameters from targets and conditioning information, and decodes
    latent samples into model outputs.

    Parameters
    ----------
    config : cVAE_MLPConfig
        Configuration defining the MLP architecture and latent representation.
    input_shape : np.ndarray | tuple
        Shape of the conditioning input excluding the batch dimension.
    output_shape : np.ndarray | tuple | None
        Shape of the model output excluding the batch dimension. If ``None``,
        ``input_shape`` is used.
    added_features_dim : int | None
        Number of additional conditioning features supplied to the model.
    """

    def __init__(
        self,
        config: cVAE_MLPConfig,
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple | None = None,
        added_features_dim: int = None,
    ):
        """
        Initialize an MLP conditional variational autoencoder.

        Parameters
        ----------
        config : cVAE_MLPConfig
            Configuration defining the encoder, decoder, condition embedding, latent
            representation, and weight initialization.
        input_shape : np.ndarray | tuple
            Shape of the conditioning input excluding the batch dimension.
        output_shape : np.ndarray | tuple | None
            Shape of the model output excluding the batch dimension. If ``None``,
            ``input_shape`` is used.
        added_features_dim : int | None
            Number of additional conditioning features supplied to the model.

        Raises
        ------
        RuntimeError
            If ``output_shape`` does not have the number of dimensions required by
            the MLP model.
        """
        super().__init__(config)

        self.encoder_hidden_dims = config.encoder_hidden_dims
        self.latent_size = config.latent_size
        self.decoder_hidden_dims = config.decoder_hidden_dims
        self.condition_embedding_dims = config.condition_embedding_dims
        self.condition_embedding_size = config.condition_embedding_size
        self.condition_dependant_latent = config.condition_dependant_latent
        self.condemb_to_decoder = config.condemb_to_decoder
        self.init_method = config.init_method

        self.batch_normalization = config.batch_normalization
        self.dropout_rate = config.dropout_rate

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

        if self.condemb_to_decoder:
            self.add_condition_size = self.condition_embedding_size
        else:
            self.add_condition_size = 0

        decoder_dims = [
            self.latent_size + self.add_condition_size + self.added_features_dim,
            *self.decoder_hidden_dims,
            self.output_shape,
        ]

        condition_embedding_dims = [
            self.input_shape + self.added_features_dim,
            *self.condition_embedding_dims,
        ]

        self.embedding = build_mlp(
            condition_embedding_dims,
            activation=self.config.activation,
            dropout_rate=self.dropout_rate,
            batch_normalization=self.batch_normalization,
            activate_final=True,
        )

        if self.condition_dependant_latent and not self.condition_dependant_flow:
            self.condition_mu = nn.Linear(
                condition_embedding_dims[-1], self.condition_embedding_size
            )
            self.condition_log_var = nn.Linear(
                condition_embedding_dims[-1], self.condition_embedding_size
            )
        else:
            self.embedding.append(
                nn.Linear(condition_embedding_dims[-1], self.condition_embedding_size)
            )

        self.add_condition_size = self.condition_embedding_size

        encoder_dims = [
            self.output_shape + self.add_condition_size + self.added_features_dim,
            *self.encoder_hidden_dims,
        ]

        self.encoder = build_mlp(
            encoder_dims,
            activation=self.config.activation,
            dropout_rate=self.dropout_rate,
            batch_normalization=self.batch_normalization,
            activate_final=True,
        )

        self.mu = nn.Linear(encoder_dims[-1], self.latent_size)
        self.log_var = nn.Linear(encoder_dims[-1], self.latent_size)

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

    def forward(self, request: cVAEForwardRequest) -> cVAEOutput:
        """
        Perform a training forward pass through the conditional variational autoencoder.

        The conditioning input is embedded, posterior latent parameters are inferred
        from the target, latent samples are drawn, and the samples are decoded into
        model outputs.

        Parameters
        ----------
        request : cVAEForwardRequest
            Forward-pass request containing targets, conditioning inputs, optional
            masks and added features, latent sample size, and optional posterior
            variance limits.

        Returns
        -------
        cVAEOutput
            Model output containing decoded samples, posterior latent parameters,
            sampled latent values, and condition-distribution parameters.
        """
        x = request.target
        x_mask = request.target_mask
        condition = request.condition
        condition_mask = request.condition_mask
        added_features = request.added_features
        latent_sample_size = request.latent_sample_size
        posterior_variance_limits = request.posterior_variance_limits

        self._shape_model_output = x.shape

        cond_mu, cond_log_var = self._condition(
            condition=condition,
            condition_mask=condition_mask,
            added_features=added_features,
        )

        mu, log_var = self._recognition(
            x=x, x_mask=x_mask, condition=cond_mu, added_features=added_features
        )

        if posterior_variance_limits is not None:
            log_var = torch.clamp(
                log_var,
                min=posterior_variance_limits[0].type_as(mu),
                max=posterior_variance_limits[1].type_as(mu),
            )

        latent_samples = self._sample(mu, log_var, latent_sample_size)

        self._shape_model_output = (latent_sample_size, *self._shape_model_output)

        out = self._generate(
            latent_samples=latent_samples,
            condition=cond_mu,
            added_features=added_features,
        )

        out = out.view(self._shape_model_output)

        return cVAEOutput(
            output=out,
            mu=mu,
            log_var=log_var,
            samples=latent_samples,
            cond_mu=cond_mu,
            cond_log_var=cond_log_var,
        )

    def predict(
        self,
        request: cVAEPredictRequest,
    ) -> cVAEOutput:
        """
        Generate predictions by sampling from the conditional latent prior.

        Parameters
        ----------
        request : cVAEPredictRequest
            Prediction request containing conditioning inputs, optional added
            features, and the requested number of latent samples.

        Returns
        -------
        cVAEOutput
            Model output containing generated predictions and condition-distribution
            parameters.
        """
        condition = request.condition
        added_features = request.added_features
        latent_sample_size = request.latent_sample_size

        B, C = condition.shape[:2]
        _shape_model_output = (latent_sample_size, B, C, -1)

        latent_samples, cond_mu, cond_log_var = self._sample_prior(request)

        output = self._generate(
            latent_samples, condition=cond_mu, added_features=added_features
        )

        return cVAEOutput(
            output=output.view(_shape_model_output),
            mu=None,
            log_var=None,
            samples=None,
            cond_mu=cond_mu,
            cond_log_var=cond_log_var,
        )

    def _recognition(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None,
        condition: torch.Tensor = None,
        added_features: torch.Tensor = None,
    ) -> tuple[torch.Tensor]:
        """
        Infer posterior latent distribution parameters from target data.

        The target is optionally masked and flattened before condition embeddings and
        additional features are concatenated and passed through the recognition
        network.

        Parameters
        ----------
        x : torch.Tensor
            Target tensor used to infer the posterior latent distribution.
        x_mask : torch.Tensor | None
            Optional mask applied element-wise to the target before encoding.
        condition : torch.Tensor | None
            Encoded conditioning representation concatenated with the target.
        added_features : torch.Tensor | None
            Additional features concatenated with the recognition-network input.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Posterior mean and log-variance tensors.
        """
        if x_mask is not None:
            x = x * x_mask

        x = x.flatten(start_dim=1)

        if added_features is not None:
            x_features = added_features.flatten(start_dim=1)
        else:
            x_features = None

        if condition is not None:
            x = torch.cat([x, condition], dim=-1)

        if x_features is not None:
            x = torch.cat([x, x_features], dim=-1)

        out = self.encoder(x)
        mu = self.mu(out)
        log_var = self.log_var(out)

        return mu, log_var

    def _condition(
        self,
        condition: torch.Tensor,
        condition_mask: torch.Tensor = None,
        added_features: torch.Tensor = None,
    ) -> tuple[torch.Tensor]:
        """
        Encode conditioning data into a latent conditioning representation.

        The conditioning input is optionally masked and concatenated with additional
        features before being passed through the condition embedding network.

        Parameters
        ----------
        condition : torch.Tensor
            Conditioning tensor to encode.
        condition_mask : torch.Tensor | None
            Optional mask applied element-wise to the conditioning tensor.
        added_features : torch.Tensor | None
            Additional features concatenated with the flattened conditioning input.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Condition mean representation and, when using a condition-dependent
            latent distribution, its log-variance. Otherwise, the second element is
            ``None``.
        """
        if added_features is not None:
            x_features = added_features.flatten(start_dim=1)
        else:
            x_features = None

        if condition_mask is not None:
            condition = condition * condition_mask

        condition = condition.flatten(start_dim=1)
        if x_features is not None:
            condition = torch.cat([condition, x_features], dim=-1)

        cond_in = self.embedding(condition)
        if self.condition_dependant_latent and not self.condition_dependant_flow:
            cond_mu = self.condition_mu(cond_in)
            cond_log_var = self.condition_log_var(cond_in)
        else:
            cond_mu = cond_in
            cond_log_var = None

        return cond_mu, cond_log_var

    def _generate(
        self,
        latent_samples: torch.Tensor,
        condition: torch.Tensor = None,
        added_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Decode latent samples into model outputs.

        Additional features and, when configured, the condition embedding are
        broadcast across latent samples and concatenated with them before decoding.

        Parameters
        ----------
        latent_samples : torch.Tensor
            Latent samples with leading dimensions corresponding to sample and batch.
        condition : torch.Tensor | None
            Encoded conditioning representation to include in the decoder input when
            configured.
        added_features : torch.Tensor | None
            Additional features to include in the decoder input.

        Returns
        -------
        torch.Tensor
            Decoded outputs with leading dimensions corresponding to latent sample
            and batch.
        """
        latent_sample_size, batch_size = latent_samples.shape[:-1]

        x_features = (
            added_features.flatten(start_dim=1) if added_features is not None else None
        )
        cond_mu = condition.flatten(start_dim=1) if condition is not None else None

        if x_features is not None:
            latent_samples = torch.cat(
                [
                    latent_samples,
                    x_features.unsqueeze(0).expand(
                        (latent_sample_size, *x_features.shape)
                    ),
                ],
                dim=-1,
            )

        if all([cond_mu is not None, self.condemb_to_decoder]):
            latent_samples = torch.cat(
                [
                    latent_samples,
                    cond_mu.unsqueeze(0).expand(latent_sample_size, *cond_mu.shape),
                ],
                dim=-1,
            )

        feature_size = latent_samples.shape[-1]

        latent_samples = latent_samples.reshape(
            latent_sample_size * batch_size, feature_size
        )
        out = self.decoder(latent_samples)

        return out.reshape(latent_sample_size, batch_size, -1)

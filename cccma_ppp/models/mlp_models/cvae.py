import torch
import torch.nn as nn
import numpy as np
from typing import ClassVar
import dataclasses

from cccma_ppp.models.models_abc import (
    cVAEmodelsABC,
    cVAEmodelConfigABC,
    cVAEForwardRequest,
    cVAEPredictRequest,
)
from cccma_ppp.models.layers.utils import _get_normal
from cccma_ppp.models.layers.mlp import build_mlp
from cccma_ppp.core.cVAE_module import cVAEOutput

from cccma_ppp.models.layers.generic import (
    InitMethod,
    ActivationName,
    _validate_dropout,
)

from cccma_ppp.core.selectors import cVAEModelSelector


@cVAEModelSelector.register("mlp")
@dataclasses.dataclass
class cVAE_MLPConfig(cVAEmodelConfigABC):
    encoder_hidden_dims: list
    latent_size: int
    condition_embedding_dims: list | None = None
    condition_embedding_size: int | None = None
    decoder_hidden_dims: list = None
    condition_dependant_latent: bool = False
    condemb_to_decoder: bool = True
    batch_normalization: bool = False
    dropout_rate: float = None
    init_method: InitMethod = "trunc_normal"
    activation: ActivationName = "relu"

    NUM_INPUT_DIMS: ClassVar[int] = 2
    NUM_OUTPUT_DIMS: ClassVar[int] = 2

    GENERATOR: ClassVar[None] = None

    def __post_init__(self):

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

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):

        return cVAE_MLP(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class cVAE_MLP(cVAEmodelsABC):
    def __init__(
        self,
        config: cVAE_MLPConfig,
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple | None = None,
        added_features_dim: int = None,
    ):

        super().__init__()

        self.config = config

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

        self.condition_dependant_flow = getattr(
            config, "condition_dependant_flow", False
        )

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

        x = request.target
        x_mask = request.target_mask
        condition = request.condition
        condition_mask = request.condition_mask
        added_features = request.added_features
        sample_size = request.sample_size
        min_posterior_variance = request.min_posterior_variance

        self._shape_model_output = x.shape

        cond_mu, cond_log_var = self._condition(
            condition=condition,
            condition_mask=condition_mask,
            added_features=added_features,
        )

        mu, log_var = self._recognition(
            x=x, x_mask=x_mask, condition=cond_mu, added_features=added_features
        )

        if min_posterior_variance is not None:
            log_var = torch.clamp(
                log_var, min=min_posterior_variance.type_as(mu), max=None
            )

        latent_samples = self._sample(mu, log_var, sample_size)

        self._shape_model_output = (sample_size, *self._shape_model_output)

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

        condition = request.condition
        condition_mask = request.condition_mask
        added_features = request.added_features
        prior_flow = request.prior_flow
        latent_samples = request.latent_samples
        nstds = request.nstds
        sample_size = request.sample_size

        B, C = condition.shape[:2]
        latent_ref_tensor = torch.zeros(
            (B, self.latent_size), device=condition.device, dtype=condition.dtype
        )
        _shape_model_output = (sample_size, B, C, -1)

        cond_mu, cond_log_var = self._condition(
            condition=condition,
            condition_mask=condition_mask,
            added_features=added_features,
        )

        if latent_samples is None:
            if self.condition_dependant_latent and not self.condition_dependant_flow:
                latent_samples = self._sample(
                    cond_mu, cond_log_var, sample_size, std=nstds
                )

            else:
                latent_samples = _get_normal(latent_ref_tensor, std=nstds).sample(
                    (sample_size,)
                )

            if prior_flow is not None:
                cond = None
                batch_size, feature_size = latent_samples.shape[1:]

                if prior_flow.condition_size is not None:
                    cond = (
                        cond_mu.unsqueeze(0)
                        .expand(sample_size, -1, -1)
                        .reshape(sample_size * batch_size, -1)
                    )

                latent_samples = latent_samples.reshape(
                    sample_size * batch_size, feature_size
                )

                flow_output = prior_flow.inverse(latent_samples, cond)
                latent_samples = flow_output.e_samples
                latent_samples = latent_samples.reshape(sample_size, batch_size, -1)
        else:
            expected_shape = (sample_size, *latent_ref_tensor.shape)
            if not latent_samples.shape == expected_shape:
                raise ValueError(
                    f"Got user specified latent_samples of shape ({latent_samples.shape}) "
                    f"but expected shape {(expected_shape)}"
                )

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

        sample_size, batch_size = latent_samples.shape[:-1]

        x_features = (
            added_features.flatten(start_dim=1) if added_features is not None else None
        )
        cond_mu = condition.flatten(start_dim=1) if condition is not None else None

        if x_features is not None:
            latent_samples = torch.cat(
                [
                    latent_samples,
                    x_features.unsqueeze(0).expand((sample_size, *x_features.shape)),
                ],
                dim=-1,
            )

        if all([cond_mu is not None, self.condemb_to_decoder]):
            latent_samples = torch.cat(
                [
                    latent_samples,
                    cond_mu.unsqueeze(0).expand(sample_size, *cond_mu.shape),
                ],
                dim=-1,
            )

        feature_size = latent_samples.shape[-1]

        latent_samples = latent_samples.reshape(sample_size * batch_size, feature_size)
        out = self.decoder(latent_samples)

        return out.reshape(sample_size, batch_size, -1)

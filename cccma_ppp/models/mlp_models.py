import torch
import torch.nn as nn
import numpy as np
from cccma_ppp.models.models_abc import cVAEmodelsABC, deterministicmodelsABC, modelConfigABC
from cccma_ppp.core.selectors import deterministicModelSelector, cVAEModelSelector
from cccma_ppp.models.normalized_flows import NormalizedFlowModel
import dataclasses
from cccma_ppp.core.cVAE_module import cVAEOutput
from cccma_ppp.core.deterministic_module import deterministicOutput







@cVAEModelSelector.register('mlp')
@dataclasses.dataclass
class cVAE_MLPConfig(modelConfigABC):
        encoder_hidden_dims : list
        latent_size : int
        decoder_hidden_dims: list =None
        condition_embedding_dims : list= None
        condition_dependant_latent:bool  = False
        condemb_to_decoder: bool = True
        batch_normalization : bool =False
        dropout_rate: float = None
        init_method : str =  'trunc_normal'

        def __post_init__(self):
            super().__init__()
            self.condition_dependant_flow = False

        def build(self):
            return cVAE_MLP(self)


class cVAE_MLP(cVAEmodelsABC):
    NUM_OUTPUT_DIMS = 1
    GENERATOR = False
    def __init__(self,   ## do not buuld anything here to keep it lightweight during config parsing
                    config : cVAE_MLPConfig):

        super().__init__()
        self.generative_modeling = True
        
        self.config = config

        self.encoder_hidden_dims = config.encoder_hidden_dims
        self.latent_size = config.latent_size
        self.decoder_hidden_dims = config.decoder_hidden_dims
        self.condition_embedding_dims = config.condition_embedding_dims
        self.condition_dependant_latent = config.condition_dependant_latent
        self.condition_dependant_flow = config.condition_dependant_flow
        self.condemb_to_decoder = config.condemb_to_decoder
        self.init_method = config.init_method

        self.batch_normalization = config.batch_normalization
        self.dropout_rate = config.dropout_rate

        if self.condition_embedding_dims is not None:
            self.condition_embedding_size = self.condition_embedding_dims[-1]
            self.condition_embedding_dims = self.condition_embedding_dims[:-1]
        else:
            self.condemb_to_decoder = False

        if self.dropout_rate is not None:
            assert self.dropout_rate <= 1 and self.dropout_rate >= 0, (
                "drop out rate must be between 0 and 1"
            )

        if self.decoder_hidden_dims is None:
            if len(self.encoder_hidden_dims) == 0:
                self.decoder_hidden_dims = []
            else:
                self.decoder_hidden_dims = self.encoder_hidden_dims[::-1][1:]

        if self.condition_dependant_latent:
            if not self.condition_dependant_flow:
                assert self.latent_size == self.condition_embedding_size, (f"for condition dependent latent when prior flow is off, "
                                                                            f"condition embedding size ({self.condition_embedding_dims + [f'**{self.condition_embedding_size}**']}) "
                                                                        f"must equal latent size ({self.latent_size}).")

        else:
            if self.condition_embedding_dims is not None:
                assert self.condemb_to_decoder, (
                    "condition embedding has to be passed to decoder for cVAE when latent is not condition dependant."
                )

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):

        assert len(output_shape) == self.NUM_OUTPUT_DIMS, f'MLP models should creat {self.NUM_OUTPUT_DIMS}D outputs'
        if output_shape is None:
            output_shape = input_shape.copy()

        if self.config.checkpoint_config is not None:
            assert input_shape == self.config.checkpoint_config.checkpoint_input_shape, f'the requested input shape ({input_shape}) does not match the loaded model : {self.config.checkpoint_config.checkpoint_input_shape}'
            assert output_shape == self.config.checkpoint_config.checkpoint_output_shape, f'the requested output shape ({output_shape}) does not match the loaded model : {self.config.checkpoint_config.checkpoint_output_shape}'

        self.input_shape = np.prod(input_shape)
        self.output_shape =  np.prod(output_shape)
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

        if self.condition_embedding_dims is not None:
            condition_embedding_dims = [
                self.input_shape + self.added_features_dim,
                *self.condition_embedding_dims,
            ]
            layers = []
            for i in range(len(condition_embedding_dims) - 1):
                layers.append(
                    nn.Linear(
                        condition_embedding_dims[i], condition_embedding_dims[i + 1]
                    )
                )
                layers.append(nn.ReLU())
                if self.dropout_rate is not None:
                    layers.append(nn.Dropout(self.dropout_rate))
                if self.batch_normalization:
                    layers.append(nn.BatchNorm1d(condition_embedding_dims[i + 1]))

            if (self.condition_dependant_latent and not self.condition_dependant_flow):
                self.condition_mu = nn.Linear(condition_embedding_dims[-1], self.condition_embedding_size)
                self.condition_log_var = nn.Linear(condition_embedding_dims[-1], self.condition_embedding_size)
            else:
                layers.append(
                    nn.Linear(
                        condition_embedding_dims[-1], self.condition_embedding_size
                    )
                )

            self.embedding = nn.Sequential(*layers)

            self.add_condition_size = self.condition_embedding_size

        encoder_dims = [
            self.output_shape + self.add_condition_size + self.added_features_dim,
            *self.encoder_hidden_dims,
        ]
        ##remember cVAE should reconstruct the target

        layers = []
        for i in range(len(encoder_dims) - 1):
            layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            layers.append(nn.ReLU())
            if self.dropout_rate is not None:
                layers.append(nn.Dropout(self.dropout_rate))
            if self.batch_normalization:
                layers.append(nn.BatchNorm1d(encoder_dims[i + 1]))
        self.encoder = nn.Sequential(*layers)

        self.mu = nn.Linear(encoder_dims[-1], self.latent_size)
        self.log_var = nn.Linear(encoder_dims[-1], self.latent_size)

        layers = []
        for i in range(len(decoder_dims) - 1):
            layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i <= (len(decoder_dims) - 3):
                layers.append(nn.ReLU())
                if self.dropout_rate is not None:
                    layers.append(nn.Dropout(self.dropout_rate))
                if self.batch_normalization:
                    layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
        self.decoder = nn.Sequential(*layers)

        

        if self.config.checkpoint_config is not None:
            self._load_state_dict(self.config.checkpoint_config)

        else:
            self._initialize_weights()



    def forward(self,
                x : torch.Tensor,
                added_features : torch.Tensor= None,
                condition : torch.Tensor= None,
                sample_size = 1,
                min_posterior_variance = None) -> cVAEOutput:

        x_in = x[0] if isinstance(x, (tuple, list)) else x
        self._shape_model_output = x_in.shape  ##cVAE autoencodes the input
        B = x_in.shape[0]
        opts = dict(device=x_in.device, dtype=x_in.dtype)
        del x_in

        cond_mu, cond_log_var = self._condition(
            condition=condition, added_features=added_features
        )

        mu, log_var = self._recognition(
            x=x, condition=cond_mu, added_features=added_features
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

        return cVAEOutput(output = out,
                                  mu = mu,
                                  log_var = log_var,
                                  cond_mu = cond_mu,
                                  cond_log_var = cond_log_var)

    def predict(self,
                condition : torch.Tensor= None,
                added_features : torch.Tensor= None,
                prior_flow : NormalizedFlowModel | None = None,
                sample_size = 1) -> cVAEOutput:

        cond_in = condition[0] if isinstance(condition, (tuple, list)) else condition
        B, C = cond_in.shape[:2]
        latent_ref_tensor = torch.zeros(
            (B, self.latent_size), device=cond_in.device, dtype=cond_in.dtype
        )
        _shape_model_output = (sample_size, B, C, -1)
        del cond_in

        cond_mu, cond_log_var = self._condition(
            condition=condition, added_features=added_features
        )

        if self.condition_dependant_latent and not self.condition_dependant_flow:
            latent_samples = self._sample(cond_mu, cond_log_var, sample_size)

            if (self.condition_dependant_latent and not self.condition_dependant_flow):
                latent_samples = self._sample(cond_mu, cond_log_var, sample_size)

        if prior_flow is not None:
            cond = cond_mu if prior_flow.condition_size is not None else None

            batch_size, feature_size = latent_samples.shape[1:]
            latent_samples = latent_samples.reshape(
                sample_size * batch_size, feature_size
            )

            flow_output = prior_flow.inverse(latent_samples, cond)
            latent_samples = flow_output.e_samples
            latent_samples = latent_samples.reshape(sample_size, batch_size, -1)

        output = self._generate(
            latent_samples, condition=cond_mu, added_features=added_features
        )

        return cVAEOutput(
            output=output.view(_shape_model_output),
            mu=None,
            log_var=None,
            cond_mu=cond_mu,
            cond_log_var=cond_log_var,
        )

                flow_output = prior_flow.inverse(latent_samples, cond )
                latent_samples =  flow_output.e_samples
                latent_samples = latent_samples.reshape(sample_size, batch_size, -1)


            output =  self._generate(latent_samples, condition = cond_mu, added_features = added_features)

            return  cVAEOutput(output = output.view(_shape_model_output),
                                  mu = None,
                                  log_var = None,
                                  cond_mu = cond_mu,
                                  cond_log_var = cond_log_var)

    def _recognition(self, x : torch.Tensor,
                        condition : torch.Tensor= None,
                        added_features : torch.Tensor= None)-> tuple[torch.Tensor]:

        if isinstance(x, (tuple, list)):
            x_in, x_mask = x
        else:
            x_in, x_mask = x, None

        if x_mask is not None:
            x_in = x_in * x_mask

        x_in = x_in.flatten(start_dim=1)

        if added_features is not None:
            x_features = added_features.flatten(start_dim=1)
        else:
            x_features = None

        if condition is not None:
            x_in = torch.cat([x_in, condition], dim=-1)

        if x_features is not None:
            x_in = torch.cat([x_in, x_features], dim=-1)

        out = self.encoder(x_in)
        mu = self.mu(out)
        log_var = self.log_var(out)

        return mu, log_var

    def _condition(
        self,
        condition: torch.Tensor = None,
        added_features: torch.Tensor = None,
    ) -> tuple[torch.Tensor]:

        if self.condition_embedding_dims is not None:
            if added_features is not None:
                x_features = added_features.flatten(start_dim=1)
            else:
                x_features = None

            if isinstance(condition, (tuple, list)):
                cond_in, cond_mask = condition
            else:
                cond_in, cond_mask = condition, None

            if cond_mask is not None:
                cond_in = cond_in * cond_mask

            cond_in = cond_in.flatten(start_dim=1)
            if x_features is not None:
                cond_in = torch.cat([cond_in, x_features], dim=-1)

            cond_in = self.embedding(cond_in)
            if (self.condition_dependant_latent and not self.condition_dependant_flow):
                    cond_mu = self.condition_mu(cond_in)
                    cond_log_var =self.condition_log_var(cond_in)
            else:
                cond_mu = cond_in
                cond_log_var = None

        else:
            cond_mu = None
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

    def _sample(self, mu, log_var, sample_size=1, std=1):

        var = torch.exp(log_var) + 1e-4
        out = mu + torch.sqrt(var) * self._get_normal(var, std).sample((sample_size,))

        return out

    def _get_normal(self, ref_tensor, std=1):
        return torch.distributions.Normal(
            torch.zeros_like(ref_tensor), torch.ones_like(ref_tensor) * std
        )


@deterministicModelSelector.register("mlp")
@dataclasses.dataclass
class AutoencoderConfig(modelConfigABC):
    encoder_hidden_dims: list
    decoder_hidden_dims: list = None
    batch_normalization: bool = False
    dropout_rate: float = None
    append_mode = 1
    init_method: str = "trunc_normal"

    def __post_init__(self):
        super().__init__()

    def build(self):
        return Autoencoder(self)



@deterministicModelSelector.register('mlp')
@dataclasses.dataclass
class AutoencoderConfig(modelConfigABC):
            encoder_hidden_dims : list
            decoder_hidden_dims: list =None
            batch_normalization : bool =False
            dropout_rate: float = None
            append_mode=1
            init_method : str =  'trunc_normal'

            def __post_init__(self):
                super().__init__()

            def build(self):
                return Autoencoder(self)


class Autoencoder( deterministicmodelsABC):

    NUM_OUTPUT_DIMS = 1
    GENERATOR = False
    def __init__(self, 
                 config : AutoencoderConfig): ## do not buuld anything here to keep it lightweight during config parsing

        super().__init__()
        self.config = config
        self.batch_normalization = config.batch_normalization
        self.dropout_rate = config.dropout_rate
        self.init_method = config.init_method
        self.append_mode = config.append_mode
        self.latent_size = config.encoder_hidden_dims[-1]
        self.encoder_hidden_dims = config.encoder_hidden_dims

        if config.decoder_hidden_dims is None:
            if len(self.encoder_hidden_dims) == 1:
                self.decoder_hidden_dims = []
            else:
                self.decoder_hidden_dims = self.encoder_hidden_dims[::-1][1:]



    def build(self,
              input_shape : np.ndarray,
              output_shape: np.ndarray|None = None,
              added_features_dim : int = None):

        assert len(output_shape) == self.NUM_OUTPUT_DIMS, f'MLP models should creat {self.NUM_OUTPUT_DIMS}D outputs'

        if output_shape is None:
            output_shape = input_shape.copy()

        if self.config.checkpoint_config is not None:
            assert input_shape == self.config.checkpoint_config.checkpoint_input_shape, f'the requested input shape ({input_shape}) does not match the loaded model : {self.config.checkpoint_config.checkpoint_input_shape}'
            assert output_shape == self.config.checkpoint_config.checkpoint_output_shape, f'the requested output shape ({output_shape}) does not match the loaded model : {self.config.checkpoint_config.checkpoint_output_shape}'

        self.input_shape = np.prod(input_shape)
        self.output_shape =  np.prod(output_shape)
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

        layers = []
        for i in range(len(encoder_dims) - 1):
            layers.append(nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            layers.append(nn.ReLU())
            if self.dropout_rate is not None:
                layers.append(nn.Dropout(self.dropout_rate))
            if self.batch_normalization:
                layers.append(nn.BatchNorm1d(encoder_dims[i + 1]))

        self.encoder = nn.Sequential(*layers)

        layers = []
        for i in range(len(decoder_dims) - 1):
            layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i <= (len(decoder_dims) - 3):
                layers.append(nn.ReLU())
                if self.dropout_rate is not None:
                    layers.append(nn.Dropout(self.dropout_rate))
                if self.batch_normalization:
                    layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))

        self.decoder = nn.Sequential(*layers)

        if self.config.checkpoint_config is not None:
            self._load_state_dict( self.config.checkpoint_config)
        else:
            self._initialize_weights()



    def forward(self,
                x : torch.Tensor,
                added_features = None) -> deterministicOutput:

        if isinstance(x, (tuple, list)):
            x_in, x_mask = x
        else:
            x_in, x_mask = x, None

        if x_mask is not None:
            x_in = x_in * x_mask

        B, C = x_in.shape[:2]
        x_in = x_in.flatten(start_dim=1)

        if added_features is not None:
            x_features = added_features.flatten(start_dim=1)
        else:
            x_features = None

        if (type(x) == list) or (type(x) == tuple):
            if self.append_mode == 1:  # append at encoder
                out = self.encoder(torch.cat([x_in, x_features], dim=-1))
                out = self.decoder(out)

            elif self.append_mode == 2:  # append at decoder
                out = self.encoder(x_in)
                out = self.decoder(torch.cat([out, x_features], dim=-1))

            elif self.append_mode == 3:  # append at encoder and decoder
                out = self.encoder(torch.cat([x_in, x_features], dim=-1))
                out = self.decoder(torch.cat([out, x_features], dim=-1))

        else:
            out = self.encoder(x_in)
            out = self.decoder(out)

        return deterministicOutput(output = out.view(B, C, -1))




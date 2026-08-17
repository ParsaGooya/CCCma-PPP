import dataclasses
from dataclasses import field
from typing import ClassVar
import numpy as np
import math
import warnings
import torch
import torch.nn as nn

from cccma_ppp.core.selectors import cVAEModelSelector, deterministicModelSelector
from cccma_ppp.core.models.cvae import cVAEOutput

from cccma_ppp.configs import save_deterministic_guess_only

from cccma_ppp.architectures.models_abc import (
    cVAEmodelConfigABC,
    cVAEmodelsABC,
    cVAEForwardRequest,
    cVAEPredictRequest,
    DeterministicRequest,
    GENERATORConfig,
)

from cccma_ppp.architectures.layers.generic import (
    InitMethod,
    UpsamplingMethod,
    MaskPoolingMethod,
    OutputActivation,
    NormalizationMethod,
    AlignmentMethod,
)
from cccma_ppp.architectures.layers.utils import (
    _broadcast_mask,
    _resize_tensor,
)

from cccma_ppp.architectures.layers.unet import (
    build_conv_block,
    UpBlock,
    DownBlock,
    UNetOutput,
    UNetOutputSIC
)


from cccma_ppp.architectures.layers.conv import (
    ConvBlockConfig,
    PartialConvBlockConfig,
    ConvNeXtBlockConfig,
    TensorMask,
    LatentVector,
)

from cccma_ppp.architectures.unet.utils import _unet_config_checks, _repeat_tensor_mask
from cccma_ppp.architectures.unet.deterministic import UNetConfig

@dataclasses.dataclass
class UNetSelector(deterministicModelSelector):
    """
    Document this class.
    
    Parameters
    ----------
    type : str
        Description not yet provided.
    share_output_block : bool
        Description not yet provided.
    """
    type: str = "unet"
    share_output_block: bool = True


@cVAEModelSelector.register("unet")
@dataclasses.dataclass
class cVAEUNetConfig(cVAEmodelConfigABC):
    """
    Document this class.
    
    Parameters
    ----------
    channels : list[int]
        Description not yet provided.
    latent_size : int
        Description not yet provided.
    condition_embedding_channels : list | None
        Description not yet provided.
    condition_embedding_size : int | None
        Description not yet provided.
    latent_normalization : NormalizationMethod | None
        Description not yet provided.
    condition_dependant_latent : bool
        Description not yet provided.
    condemb_to_decoder : bool
        Description not yet provided.
    deterministic_guess_config : UNetSelector | None
        Description not yet provided.
    block_config : ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig
        Description not yet provided.
    upsampling_method : UpsamplingMethod
        Description not yet provided.
    upsampling_alignment_method : AlignmentMethod
        Description not yet provided.
    transpose_kernel_sizes : list[int | tuple[int, int]] | int
        Description not yet provided.
    add_skip_latent : bool
        Description not yet provided.
    mask_pooling : MaskPoolingMethod
        Description not yet provided.
    mask_fraction_threshold : float
        Description not yet provided.
    output_activation : OutputActivation
        Description not yet provided.
    output_block_hidden_channels : int | None
        Description not yet provided.
    init_method : InitMethod
        Description not yet provided.
    GENERATOR : GENERATORConfig | None
        Description not yet provided.
    """
    channels: list[int]
    latent_size: int
    condition_embedding_channels: list | None = None
    condition_embedding_size: int | None = None

    latent_normalization: NormalizationMethod | None = "layer"
    condition_dependant_latent: bool = False
    condemb_to_decoder: bool = True
    deterministic_guess_config: UNetSelector | None = None

    block_config: ConvBlockConfig | PartialConvBlockConfig | ConvNeXtBlockConfig = (
        field(default_factory=ConvBlockConfig)
    )

    upsampling_method: UpsamplingMethod = "bilinear"
    upsampling_alignment_method: AlignmentMethod = "padd"
    transpose_kernel_sizes: list[int | tuple[int, int]] | int = 3

    add_skip_latent: bool = False

    mask_pooling: MaskPoolingMethod = "any"
    mask_fraction_threshold: float = 0.5

    output_activation: OutputActivation = "identity"
    output_block_hidden_channels: int | None = None

    init_method: InitMethod = "default"
    GENERATOR: GENERATORConfig | None = None

    NUM_INPUT_DIMS: ClassVar[int] = 3
    NUM_OUTPUT_DIMS: ClassVar[int] = 3

    def __post_init__(self) -> None:
        """
        Document this function.
        """
        _unet_config_checks(self)
        n_up_blocks = len(self.channels) - 1

        if isinstance(self.transpose_kernel_sizes, int):
            self.transpose_kernel_sizes = [self.transpose_kernel_sizes] * n_up_blocks

        if self.condition_embedding_channels is None:
            self.condition_embedding_channels = self.channels

        if self.condition_embedding_size is None:
            self.condition_embedding_size = self.latent_size

        self._resolve_deterministic_guess()

    def _resolve_deterministic_guess(self):
        """
        Document this function.
        
        Raises
        ------
        TypeError
            Description not yet provided.
        ValueError
            Description not yet provided.
        """
        if self.deterministic_guess_config is not None:
            self.share_output_block = self.deterministic_guess_config.share_output_block
            self.freeze_deterministic = self.deterministic_guess_config.checkpoint_config.freeze_weights

            self.deterministic_guess_config = self.deterministic_guess_config.get_model_config()

            if not isinstance(self.deterministic_guess_config, UNetConfig):
                raise TypeError(
                    "cVAEUNet requires deterministic_guess_config to resolve to a "
                    "UNetConfig or one of its subclasses, got "
                    f"{type(self.deterministic_guess_config).__name__}."
                )

            if (self.deterministic_guess_config.channels[0] !=
                self.channels[0]):
                raise ValueError(
                    "The cVAE UNet model and the configured deterministic guess " \
                    "model must have the same number of channels before output block " \
                    f"for summation of the deterministic guess at that level. Expected : {self.channels[0]} " \
                    f"got {self.deterministic_guess_config.channels[0]}"
                )

            if self.deterministic_guess_config.GENERATOR is not None:

                raise ValueError(
                    "The deterministic guess UNet model cannot have GENERATOR on. " \
                )  

            if self.share_output_block:
                if any([(
                    self.output_activation
                    != self.deterministic_guess_config.output_activation
                    ),
                    (
                    self.output_block_hidden_channels
                    != self.deterministic_guess_config.output_block_hidden_channels
                    ),
                    (getattr(self, "clip_output", None)
                    != getattr(self.deterministic_guess_config, "clip_output", None)
                    ),
                ]):
                    raise ValueError(
                        "With share_output_block=True, the cVAE and deterministic "
                        "guess must use compatible output-block settings "
                        "(activation, hidden channels, and clip_output)."
                    )

    
        else:
            self.share_output_block = False    
            self.freeze_deterministic = False         

    @property
    def EXPECTS_MASK(self) -> bool:
        """
        Document this function.
        
        Returns
        -------
        bool
            Description not yet provided.
        """
        return (
            isinstance(self.block_config, PartialConvBlockConfig)
            or getattr(self.block_config, "use_partial_conv", False)
        )
    
    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int | None = None,
    ):
        """
        Document this function.
        
        Parameters
        ----------
        input_shape : np.ndarray
            Description not yet provided.
        output_shape : np.ndarray | None
            Description not yet provided.
        added_features_dim : int | None
            Description not yet provided.
        
        Returns
        -------
        Any
            Description not yet provided.
        """
        return cVAEUNet(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class cVAEUNet(cVAEmodelsABC):
    """
    Document this class.
    
    Parameters
    ----------
    config : cVAEUNetConfig
        Description not yet provided.
    input_shape : np.ndarray | tuple
        Description not yet provided.
    output_shape : np.ndarray | tuple | None
        Description not yet provided.
    added_features_dim : int | None
        Description not yet provided.
    """
    def __init__(
        self,
        config: cVAEUNetConfig,
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple | None = None,
        added_features_dim: int | None = None,
    ):
        """
        Document this function.
        
        Parameters
        ----------
        config : cVAEUNetConfig
            Description not yet provided.
        input_shape : np.ndarray | tuple
            Description not yet provided.
        output_shape : np.ndarray | tuple | None
            Description not yet provided.
        added_features_dim : int | None
            Description not yet provided.
        
        Raises
        ------
        RuntimeError
            Description not yet provided.
        ValueError
            Description not yet provided.
        """
        super().__init__(config)

        self.init_method = config.init_method
        self.added_features_dim = added_features_dim or 0
        self.latent_size = config.latent_size
        self.condition_embedding_channels = config.condition_embedding_channels
        self.condition_embedding_size = config.condition_embedding_size
        self.condition_dependant_latent = config.condition_dependant_latent
        self.condemb_to_decoder = config.condemb_to_decoder 
        self.deterministic_guess_config = config.deterministic_guess_config
        self.share_output_block = config.share_output_block
        self.freeze_deterministic = config.freeze_deterministic
        self.deterministic_guess = None


        if output_shape is None:
            output_shape = input_shape

        if len(input_shape) != config.NUM_INPUT_DIMS:
            raise RuntimeError(
                f"UNet expects {config.NUM_INPUT_DIMS}D input shapes "
                f"(channel, height, width), got {input_shape}."
            )

        if len(output_shape) != config.NUM_OUTPUT_DIMS:
            raise RuntimeError(
                f"UNet expects {config.NUM_OUTPUT_DIMS}D output shapes "
                f"(channel, height, width), got {output_shape}."
            )

        min_spatial_size = int(min(input_shape[-2:]))

        if min_spatial_size < 1:
            raise ValueError(
                "UNet spatial dimensions must be positive, "
                f"got {tuple(input_shape[-2:])}."
            )

        n_down_blocks = len(config.channels) - 1
        max_down_blocks = math.floor(math.log2(min_spatial_size))

        if n_down_blocks > max_down_blocks:
            raise ValueError(
                "The requested UNet depth is too large for the input spatial "
                f"shape {tuple(input_shape[-2:])}. The configuration defines "
                f"{n_down_blocks} downsampling blocks, but the minimum spatial "
                f"dimension permits at most {max_down_blocks}. "
                f"Use no more than {max_down_blocks + 1} channel levels."
            )

        self._validate_checkpoint_compatibility(
            input_shape=input_shape,
            output_shape=output_shape,
        )

        self.input_shape = input_shape
        self.output_shape = output_shape

        output_channels = output_shape[0]
        output_spatial_shape = output_shape[1:]
        channels = config.channels
        
        recognition_input_channels = (
            output_shape[0] + input_shape[0] + self.added_features_dim
        )

        self.recognition = Recognition(
            input_channels=recognition_input_channels,
            input_spatial_shape=tuple(output_shape[-2:]),
            channels=channels,
            latent_size=self.latent_size,
            config=config,
        )

        condition_input_channels = input_shape[0] + self.added_features_dim
        get_log_var = (
            self.condition_dependant_latent and not self.condition_dependant_flow
        )

        self.condition = Recognition(
            input_channels=condition_input_channels,
            input_spatial_shape=tuple(input_shape[-2:]),
            channels=self.condition_embedding_channels,
            latent_size=self.condition_embedding_size,
            config=config,
            get_log_var=get_log_var,
        )

        self.upsampling_shapes = list(reversed(self.recognition.spatial_shapes))

        reversed_channels = list(reversed(channels))
        if self.condemb_to_decoder:
            self.add_condition_size = self.condition_embedding_size
        else:
            self.add_condition_size = 0
        generation_latent_size = self.latent_size + self.add_condition_size

        self.generation = Generation(
            latent_size=generation_latent_size,
            channels=reversed_channels,
            resize_shapes=self.upsampling_shapes,
            config=config,
        )

        if self.deterministic_guess_config is not None:

            self.deterministic_guess = self.deterministic_guess_config.build(
                input_shape=input_shape,
                output_shape=output_shape,
                added_features_dim=added_features_dim,
            )

        if self.share_output_block: 
            
            self.output = self.deterministic_guess.output_block

        elif self.freeze_deterministic:

            for param in self.deterministic_guess.output_block.parameters():
                param.requires_grad = True

            self.output = self.deterministic_guess.output_block

        else:

            self.output = self._build_output(
                in_channels=reversed_channels[-1],
                out_channels=output_channels,
            )

        if config.checkpoint_config is not None:
            self._load_state_dict(config.checkpoint_config)
        else:
            self._initialize_weights(
                config.init_method,
                exclude=(self.deterministic_guess,))

        self._predict_called: bool = False

    def _build_output(
        self,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """
        Document this function.
        
        Parameters
        ----------
        in_channels : int
            Description not yet provided.
        out_channels : int
            Description not yet provided.
        
        Returns
        -------
        nn.Module
            Description not yet provided.
        """
        return UNetOutput(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=self.config.output_block_hidden_channels,
            activation=self.config.output_activation,
        )

    def _prepare_input(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None,
        condition: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
        added_features: torch.Tensor | None = None,
    ) -> TensorMask:
        """
        Document this function.
        
        Parameters
        ----------
        x : torch.Tensor
            Description not yet provided.
        x_mask : torch.Tensor | None
            Description not yet provided.
        condition : torch.Tensor | None
            Description not yet provided.
        condition_mask : torch.Tensor | None
            Description not yet provided.
        added_features : torch.Tensor | None
            Description not yet provided.
        
        Returns
        -------
        TensorMask
            Description not yet provided.
        """
        x_mask = _broadcast_mask(x_mask, x)
        if condition is not None:
            condition = _resize_tensor(
                condition,
                x.shape[-2:],
            )

            x = torch.cat([x, condition], dim=1)

            if x_mask is not None:
                if condition_mask is not None:
                    condition_mask = _broadcast_mask(condition_mask, condition)
                else:
                    condition_mask = torch.ones_like(condition)

                x_mask = torch.cat(
                    [x_mask, condition_mask],
                    dim=1,
                )

        if added_features is not None:
            added_features = _resize_tensor(
                added_features,
                x.shape[-2:],
                mode="nearest",
            )

            feature_mask = (
                torch.ones_like(added_features) if x_mask is not None else None
            )

            x = torch.cat([x, added_features], dim=1)

            if x_mask is not None:
                x_mask = torch.cat(
                    [x_mask, feature_mask],
                    dim=1,
                )

        return TensorMask(tensor=x, mask=x_mask)

    def forward(self, request: cVAEForwardRequest) -> cVAEOutput:
        """
        Document this function.
        
        Parameters
        ----------
        request : cVAEForwardRequest
            Description not yet provided.
        
        Returns
        -------
        cVAEOutput
            Description not yet provided.
        """
        x = request.target
        x_mask = request.target_mask
        condition = request.condition
        condition_mask = request.condition_mask
        added_features = request.added_features
        latent_sample_size = request.latent_sample_size
        posterior_variance_limits = request.posterior_variance_limits
        num_output_samples = request.output_sample_size

        batch_size = x.shape[0]

        if self.training and self.config.GENERATOR is not None:
            num_output_samples = self.config.GENERATOR.num_training_noise_samples

        cond_mu, cond_log_var = self._condition(
            condition=condition,
            condition_mask=condition_mask,
            added_features=added_features,
        )

        mu, log_var = self._recognition(
            x=x,
            x_mask=x_mask,
            condition=condition,
            condition_mask=condition_mask,
            added_features=added_features,
        )

        if posterior_variance_limits is not None:
            log_var = torch.clamp(
                log_var, 
                min=posterior_variance_limits[0].type_as(mu), 
                max=posterior_variance_limits[1].type_as(mu),
            )

        latent_samples = self._sample(mu, log_var, latent_sample_size)

        out = self._generate(
            latent_samples=latent_samples,
            condition_embedding=cond_mu,
            num_output_samples=num_output_samples,
        )

        if num_output_samples > 0:
            sample_sizes = (num_output_samples, latent_sample_size, batch_size)

        else:
            sample_sizes = (latent_sample_size, batch_size)

        deterministic_guess = self._deterministic_guess(
                input = condition,
                input_mask = condition_mask,
                added_features = added_features,
        )

        if deterministic_guess is not None:
            out = out + deterministic_guess

        output = self._output_block(out, 
                                    sample_sizes)
                                    
                                    
        return cVAEOutput(
            output=output,
            mu=mu,
            log_var=log_var,
            samples=latent_samples,
            cond_mu=cond_mu,
            cond_log_var=cond_log_var,
        )

    def predict(
        self,
        request: cVAEPredictRequest,
        deterministic_guess_only: bool = save_deterministic_guess_only
    ) -> cVAEOutput:
        """
        Document this function.
        
        Parameters
        ----------
        request : cVAEPredictRequest
            Description not yet provided.
        deterministic_guess_only : bool
            Description not yet provided.
        
        Returns
        -------
        cVAEOutput
            Description not yet provided.
        
        Warns
        -----
        UserWarning
            Description not yet provided.
        """
        num_output_samples = request.output_sample_size
        latent_sample_size = request.latent_sample_size

        latent_samples, cond_mu, cond_log_var = self._sample_prior(request)
        batch_size = cond_mu.shape[0]

        out = self._generate(
            latent_samples,
            condition_embedding=cond_mu,
            num_output_samples=num_output_samples,
        )

        if deterministic_guess_only:
            latent_sample_size = 1
            num_output_samples = 0

        if num_output_samples > 0:
            sample_sizes = (num_output_samples, latent_sample_size, batch_size)

        else:
            sample_sizes = (latent_sample_size, batch_size)

        deterministic_guess = self._deterministic_guess(
                input = request.condition,
                input_mask = request.condition_mask,
                added_features = request.added_features,
        )

        if deterministic_guess is not None:
            if deterministic_guess_only:
                if not self._predict_called:
                    warnings.warn("======================================================\n"
                                  "YOU ARE SAVING DETERMINISTIC GUESS BRANCH OUTPUT ONLY! \n"
                                  "======================================================")
                out = deterministic_guess.unsqueeze(0)
            else:
                out = out + deterministic_guess

        output = self._output_block(out, 
                                    sample_sizes)
        self._predict_called = True
        return cVAEOutput(
            output=output,
            mu=None,
            log_var=None,
            samples=None,
            cond_mu=cond_mu,
            cond_log_var=cond_log_var,
            deterministic_guess=deterministic_guess_only
        )

    def _recognition(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None,
        condition: torch.Tensor = None,
        condition_mask: torch.Tensor = None,
        added_features: torch.Tensor = None,
    ) -> tuple[torch.Tensor]:
        """
        Document this function.
        
        Parameters
        ----------
        x : torch.Tensor
            Description not yet provided.
        x_mask : torch.Tensor | None
            Description not yet provided.
        condition : torch.Tensor
            Description not yet provided.
        condition_mask : torch.Tensor
            Description not yet provided.
        added_features : torch.Tensor
            Description not yet provided.
        
        Returns
        -------
        tuple[torch.Tensor]
            Description not yet provided.
        """
        input = self._prepare_input(
            x=x,
            x_mask=x_mask,
            condition=condition,
            condition_mask=condition_mask,
            added_features=added_features,
        )

        latent = self.recognition(input)

        return latent.mu, latent.log_var

    def _condition(
        self,
        condition: torch.Tensor,
        condition_mask: torch.Tensor = None,
        added_features: torch.Tensor = None,
    ) -> tuple[torch.Tensor]:
        """
        Document this function.
        
        Parameters
        ----------
        condition : torch.Tensor
            Description not yet provided.
        condition_mask : torch.Tensor
            Description not yet provided.
        added_features : torch.Tensor
            Description not yet provided.
        
        Returns
        -------
        tuple[torch.Tensor]
            Description not yet provided.
        """
        input = self._prepare_input(
            x=condition,
            x_mask=condition_mask,
            added_features=added_features,
        )

        condition_embedding = self.condition(input)

        cond_mu = condition_embedding.mu
        cond_log_var = condition_embedding.log_var

        return cond_mu, cond_log_var

    def _generate(
        self,
        latent_samples: torch.Tensor,
        condition_embedding: torch.Tensor | None = None,
        num_output_samples: int = 0,
    ) -> torch.Tensor:
        """
        Document this function.
        
        Parameters
        ----------
        latent_samples : torch.Tensor
            Description not yet provided.
        condition_embedding : torch.Tensor | None
            Description not yet provided.
        num_output_samples : int
            Description not yet provided.
        
        Returns
        -------
        torch.Tensor
            Description not yet provided.
        """
        latent_sample_size, batch_size = latent_samples.shape[:-1]

        if all([condition_embedding is not None, self.condemb_to_decoder]):
            latent_samples = torch.cat(
                [
                    latent_samples,
                    condition_embedding.unsqueeze(0).expand(
                        latent_sample_size, *condition_embedding.shape
                    ),
                ],
                dim=-1,
            )

        feature_size = latent_samples.shape[-1]

        latent_samples = latent_samples.reshape(latent_sample_size * batch_size, feature_size)
        out = self.generation(latent_samples, num_output_samples)

        if num_output_samples > 0:
            return out.reshape(
                num_output_samples,
                latent_sample_size,
                batch_size,
                *out.shape[2:],
            )

        return out.reshape(latent_sample_size, batch_size, *out.shape[1:])

    def _deterministic_guess(
            self,
            input: torch.Tensor,
            input_mask: torch.Tensor | None = None,
            added_features: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """
        Document this function.
        
        Parameters
        ----------
        input : torch.Tensor
            Description not yet provided.
        input_mask : torch.Tensor | None
            Description not yet provided.
        added_features : torch.Tensor | None
            Description not yet provided.
        
        Returns
        -------
        torch.Tensor | None
            Description not yet provided.
        """
        if self.deterministic_guess is None:
            return

        request = DeterministicRequest(
            input,
            input_mask,
            added_features,
        )   

        return self.deterministic_guess.forward_decoder(request) 


    def _output_block(
        self,
        input: torch.Tensor,
        sample_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        """
        Document this function.
        
        Parameters
        ----------
        input : torch.Tensor
            Description not yet provided.
        sample_sizes : tuple[int, ...]
            Description not yet provided.
        
        Returns
        -------
        torch.Tensor
            Description not yet provided.
        """
        input = input.reshape(
            math.prod(sample_sizes),
            *input.shape[len(sample_sizes):],
        )

        output = self.output(input)

        return output.reshape(
            *sample_sizes,
            *output.shape[1:],
        )

class Recognition(nn.Module):
    """
    Document this class.
    
    Parameters
    ----------
    input_channels : int
        Description not yet provided.
    input_spatial_shape : tuple[int, int]
        Description not yet provided.
    channels : list[int]
        Description not yet provided.
    latent_size : int
        Description not yet provided.
    config : cVAEUNetConfig
        Description not yet provided.
    get_log_var : bool
        Description not yet provided.
    """
    def __init__(
        self,
        input_channels: int,
        input_spatial_shape: tuple[int, int],
        channels: list[int],
        latent_size: int,
        config: cVAEUNetConfig,
        *,
        get_log_var: bool = True,
    ):
        """
        Document this function.
        
        Parameters
        ----------
        input_channels : int
            Description not yet provided.
        input_spatial_shape : tuple[int, int]
            Description not yet provided.
        channels : list[int]
            Description not yet provided.
        latent_size : int
            Description not yet provided.
        config : cVAEUNetConfig
            Description not yet provided.
        get_log_var : bool
            Description not yet provided.
        """
        super().__init__()

        self.initial_mapping = build_conv_block(
            input_channels,
            channels[0],
            config.block_config,
        )

        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    channels[index],
                    channels[index + 1],
                    block_config=config.block_config,
                    return_skip=False,
                    mask_pooling=config.mask_pooling,
                    mask_fraction_threshold=config.mask_fraction_threshold,
                )
                for index in range(len(channels) - 1)
            ]
        )

        self.spatial_shapes = self._get_spatial_shapes(input_spatial_shape)

        bottleneck_spatial_shape = self.spatial_shapes[-1]
        bottleneck_output_shape = (
            channels[-1],
            *bottleneck_spatial_shape,
        )

        self.bottleneck = build_conv_block(
            channels[-1],
            channels[-1],
            config.block_config,
            latent_size=latent_size,
            block_output_shape=bottleneck_output_shape,
            get_log_var=get_log_var,
            latent_normalization=config.latent_normalization,
        )

    def _get_spatial_shapes(
        self,
        input_shape: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """
        Document this function.
        
        Parameters
        ----------
        input_shape : tuple[int, int]
            Description not yet provided.
        
        Returns
        -------
        list[tuple[int, int]]
            Description not yet provided.
        """
        shape = tuple(input_shape)
        shapes = [shape]

        for block in self.down_blocks:
            shape = block.output_shape(shape)
            shapes.append(shape)

        return shapes

    def forward(
        self,
        input: TensorMask,
    ) -> LatentVector:
        """
        Document this function.
        
        Parameters
        ----------
        input : TensorMask
            Description not yet provided.
        
        Returns
        -------
        LatentVector
            Description not yet provided.
        """
        input = self.initial_mapping(input)

        for down_block in self.down_blocks:
            input = down_block(input)

        return self.bottleneck(input)


class Generation(nn.Module):
    """
    Document this class.
    
    Parameters
    ----------
    latent_size : int
        Description not yet provided.
    channels : list[int]
        Description not yet provided.
    resize_shapes : list[tuple]
        Description not yet provided.
    config : cVAEUNetConfig
        Description not yet provided.
    """
    def __init__(
        self,
        latent_size: int,
        channels: list[int],
        resize_shapes: list[tuple],
        config: cVAEUNetConfig,
    ):
        """
        Document this function.
        
        Parameters
        ----------
        latent_size : int
            Description not yet provided.
        channels : list[int]
            Description not yet provided.
        resize_shapes : list[tuple]
            Description not yet provided.
        config : cVAEUNetConfig
            Description not yet provided.
        """
        super().__init__()
        self.bottleneck_dim = channels[0]
        self.bottleneck_shape = resize_shapes[0]
        self.resize_shapes = resize_shapes[1:]
        self.config = config

        self.combine_latent = nn.Linear(
            latent_size, self.bottleneck_dim * np.prod(self.bottleneck_shape)
        )

        input_channels = self.bottleneck_dim
        generator_enabled = config.GENERATOR is not None
        inject_noise_in_block = (
            generator_enabled and config.GENERATOR.noise_level != "low"
        )

        up_blocks: list[nn.Module] = []
        for index, out_channels in enumerate(channels[1:]):
            inject_noise = generator_enabled and (
                config.GENERATOR.noise_level != "medium" or index == len(self.resize_shapes) - 1
            )

            up_blocks.append(
                UpBlock(
                    input_channels=input_channels,
                    skip_channels=None,
                    out_channels=out_channels,
                    block_config=config.block_config,
                    upsampling_method=config.upsampling_method,
                    skip_alignment_method=config.upsampling_alignment_method,
                    transpose_kernel_size=config.transpose_kernel_sizes[index],
                    inject_noise=inject_noise,
                    inject_noise_in_block=inject_noise_in_block,
                )
            )
            input_channels = out_channels

        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(
        self,
        latent_samples: torch.Tensor,
        num_output_samples: int = 0,
    ) -> torch.Tensor:
        """
        Document this function.
        
        Parameters
        ----------
        latent_samples : torch.Tensor
            Description not yet provided.
        num_output_samples : int
            Description not yet provided.
        
        Returns
        -------
        torch.Tensor
            Description not yet provided.
        """
        batch_size = latent_samples.shape[0]
        x = self.combine_latent(latent_samples)
        x = x.reshape(-1, self.bottleneck_dim, *self.bottleneck_shape)

        input = TensorMask(tensor=x, mask=None)

        if self.config.GENERATOR is not None and num_output_samples > 0:
            input = _repeat_tensor_mask(
                input,
                repeats=num_output_samples,
            )

        for up_block, resize_shape in zip(
            self.up_blocks,
            self.resize_shapes,
            strict=True,
        ):
            input = up_block(input, skip=None, resize_shape=resize_shape)

        output = input.tensor
        if self.config.GENERATOR is not None and num_output_samples > 0:
            output = output.reshape(
                batch_size,
                num_output_samples,
                *output.shape[1:],
            ).transpose(0, 1)

        return output




@cVAEModelSelector.register("unetsic")
@dataclasses.dataclass
class cVAEUNetSICEConfig(cVAEUNetConfig):
    """
    Document this class.
    
    Parameters
    ----------
    clip_output : bool
        Description not yet provided.
    """
    clip_output: bool = False
    
    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int | None = None,
    ):
        """
        Document this function.
        
        Parameters
        ----------
        input_shape : np.ndarray
            Description not yet provided.
        output_shape : np.ndarray | None
            Description not yet provided.
        added_features_dim : int | None
            Description not yet provided.
        
        Returns
        -------
        Any
            Description not yet provided.
        """
        return cVAEUNetSIC(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

class cVAEUNetSIC(cVAEUNet):
    """
    Document this class.
    """
    def _build_output(
        self,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """
        Document this function.
        
        Parameters
        ----------
        in_channels : int
            Description not yet provided.
        out_channels : int
            Description not yet provided.
        
        Returns
        -------
        nn.Module
            Description not yet provided.
        """
        return UNetOutputSIC(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=self.config.output_block_hidden_channels,
            activation=self.config.output_activation,
            clip_output=self.config.clip_output
        )


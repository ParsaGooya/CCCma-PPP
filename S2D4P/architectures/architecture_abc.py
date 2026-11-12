import abc
import numpy as np
import torch
import torch.nn as nn
from typing import final, ClassVar
from timm.models.layers import trunc_normal_
import gc
from pathlib import Path
import dataclasses

from S2D4P.architectures.layers.utils import _get_normal
from S2D4P.core.core_abc import OutputABC
from S2D4P.generic.runtime import RuntimeContext
from S2D4P.architectures.layers.generic import (
    ActivationName,
    InitMethod,
    NoiseLevel,
)
from S2D4P.architectures.layers.utils import _sample


@dataclasses.dataclass
class GENERATORConfig:
    """
    Configuration for generator-based stochastic output sampling.

    Parameters
    ----------
    noise_level : NoiseLevel
        Level controlling where generator noise is injected into the architecture.
    num_training_noise_samples : int
        Number of stochastic output samples generated during training.
    num_validation_noise_samples : int | None
        Number of stochastic output samples generated during validation. If
        ``None``, ``num_training_noise_samples`` is used.
    """

    noise_level: NoiseLevel = "full"
    num_training_noise_samples: int = 10
    num_validation_noise_samples: int | None = None

    def __post_init__(self):
        """
        Resolve and validate generator sampling settings.

        When the validation sample count is omitted, the training sample count is
        used.

        Raises
        ------
        ValueError
            If fewer than two validation output samples are configured.
        """

        if self.num_validation_noise_samples is None:
            self.num_validation_noise_samples = self.num_training_noise_samples

        if self.num_validation_noise_samples <= 1:
            raise ValueError(
                "For Generator training, at least two output ensemble samples are required."
            )


@dataclasses.dataclass
class CheckpointConfig:
    """
    Configuration for loading architecture weights from a checkpoint.

    Parameters
    ----------
    load_path : Path | str
        Path to the checkpoint file.
    checkpoint_input_shape : np.ndarray
        Input shape used by the checkpointed architecture.
    checkpoint_output_shape : np.ndarray
        Output shape used by the checkpointed architecture.
    checkpoint_input_var_metadata : dict
        Input-variable metadata associated with the checkpoint.
    checkpoint_output_var_metadata : dict
        Output-variable metadata associated with the checkpoint.
    strict : bool
        Whether checkpoint parameters must exactly match the current architecture
        state dictionary.
    freeze_weights : bool
        Whether loaded architecture parameters are frozen after loading.
    """

    load_path: Path | str
    checkpoint_input_shape: np.ndarray
    checkpoint_output_shape: np.ndarray
    checkpoint_input_var_metadata: dict
    checkpoint_output_var_metadata: dict
    strict: bool = True
    freeze_weights: bool = False


class flowABC(nn.Module, abc.ABC):
    """
    Abstract base class for invertible normalizing-flow transformations.
    """

    @abc.abstractmethod
    def forward(self, x, condition=None):
        """
        Apply the forward flow transformation.

        Parameters
        ----------
        x : Any
            Values to transform through the flow.
        condition : Any
            Optional conditioning information supplied to the flow.

        Returns
        -------
        Any
            Forward flow output.
        """
        pass

    @abc.abstractmethod
    def inverse(self, z, condition=None):
        """
        Apply the inverse flow transformation.

        Parameters
        ----------
        z : Any
            Values to transform through the inverse flow.
        condition : Any
            Optional conditioning information supplied to the inverse flow.

        Returns
        -------
        Any
            Inverse flow output.
        """
        pass


class ArchConfigABC(abc.ABC):
    """
    Abstract base configuration for model architectures.

    Attributes
    ----------
    activation : ActivationName
        Activation function used by the architecture.
    GENERATOR : GENERATORConfig | None
        Optional configuration for generator-based stochastic sampling.
    EXPECTS_MASK : bool
        Whether the architecture expects spatial validity masks.
    """

    activation: ActivationName
    NUM_INPUT_DIMS: ClassVar[int | None]
    NUM_OUTPUT_DIMS: ClassVar[int | None]
    GENERATOR: GENERATORConfig | None
    EXPECTS_MASK: bool

    def __init_subclass__(cls):
        """
        Initialize configuration state for an architecture subclass.

        Each subclass begins without an attached checkpoint configuration.
        """
        super().__init_subclass__()
        cls.checkpoint_config = None

    @final
    def _add_checkpoint_config(self, checkpoint_config: CheckpointConfig) -> None:
        """
        Attach checkpoint-loading information to the architecture configuration.

        Parameters
        ----------
        checkpoint_config : CheckpointConfig
            Checkpoint configuration to associate with the architecture.
        """
        self.checkpoint_config = checkpoint_config

    @final
    @property
    def NUM_INPUT_DIMS(self) -> int:
        """
        Return the number of input dimensions expected by the architecture.

        Returns
        -------
        int
            Number of input dimensions expected by the architecture.
        """
        return self.NUM_INPUT_DIMS

    @final
    @property
    def NUM_OUTPUT_DIMS(self) -> int:
        """
        Return the number of output dimensions produced by the architecture.

        Returns
        -------
        int
            Number of output dimensions produced by the architecture.
        """
        return self.NUM_OUTPUT_DIMS

    @final
    @property
    def GENERATOR(self) -> bool:
        """
        Return the generator configuration associated with the architecture.

        Returns
        -------
        GENERATORConfig | None
            Generator configuration, or ``None`` when stochastic generator behavior
            is disabled.
        """
        return getattr(self, "GENERATOR", None)

    @property
    @abc.abstractmethod
    def EXPECTS_MASK(self) -> bool:
        """
        Indicate whether the architecture expects spatial validity masks.

        Returns
        -------
        bool
            Whether spatial validity masks are required by the architecture.
        """
        pass

    @abc.abstractmethod
    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
        **kwargs,
    ):
        """
        Build an architecture from this configuration.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of each model input excluding the batch dimension.
        output_shape : np.ndarray | None
            Shape of each model output excluding the batch dimension.
        added_features_dim : int | None
            Number of additional features supplied with each sample.
        kwargs : Any
            Additional architecture-specific keyword arguments.

        Returns
        -------
        Any
            Constructed architecture.
        """
        pass


class ArchABC(nn.Module, abc.ABC):
    """
    Abstract base class for model architectures.

    Provides shared checkpoint validation, parameter initialization, device
    resolution, and checkpoint-loading utilities.

    Attributes
    ----------
    generative_modeling : bool
        Whether the architecture performs generative modeling.
    """

    generative_modeling: bool

    @final
    def _validate_checkpoint_compatibility(
        self,
        *,
        input_shape: np.ndarray,
        output_shape: np.ndarray,
    ) -> None:
        """
        Validate current data and shapes against checkpoint metadata.

        Validation is skipped when no checkpoint configuration is attached.

        Parameters
        ----------
        input_shape : np.ndarray
            Input shape requested for the current architecture.
        output_shape : np.ndarray
            Output shape requested for the current architecture.

        Raises
        ------
        RuntimeError
            If the requested input or output shape differs from the checkpoint, or
            current input or target metadata are incompatible with checkpoint
            metadata.
        """
        checkpoint = self.config.checkpoint_config

        if checkpoint is None:
            return

        if not np.array_equal(
            input_shape,
            checkpoint.checkpoint_input_shape,
        ):
            raise RuntimeError(
                f"Requested input shape {input_shape} does not match "
                f"checkpoint input shape {checkpoint.checkpoint_input_shape}."
            )

        if not np.array_equal(
            output_shape,
            checkpoint.checkpoint_output_shape,
        ):
            raise RuntimeError(
                f"Requested output shape {output_shape} does not match "
                f"checkpoint output shape {checkpoint.checkpoint_output_shape}."
            )

        if (
            RuntimeContext.INPUT_VAR_METADATA
            != checkpoint.checkpoint_input_var_metadata
        ):
            raise RuntimeError(
                "Checkpoint input metadata is incompatible with the "
                "current input variables or preprocessing pipeline."
            )

        if (
            RuntimeContext.TARGET_VAR_METADATA
            != checkpoint.checkpoint_output_var_metadata
        ):
            raise RuntimeError(
                "Checkpoint target metadata is incompatible with the "
                "current target variables or preprocessing pipeline."
            )

    @abc.abstractmethod
    def forward(self):
        """
        Run the architecture forward pass.

        Returns
        -------
        Any
            Architecture-specific model output.
        """
        pass

    @final
    def _initialize_weights(
        self, init_method="trunc_normal", exclude: tuple[nn.Module, ...] = ()
    ) -> None:
        """
        Initialize trainable architecture parameters.

        Modules listed in ``exclude`` and their descendants are skipped. Supported
        layers are initialized using the requested initialization method.

        Parameters
        ----------
        init_method : InitMethod
            Parameter initialization method to apply.
        exclude : tuple[nn.Module, ...]
            Modules excluded from parameter initialization.
        """
        excluded = {id(module) for module in exclude}

        def visit(module: nn.Module):
            """
            Recursively initialize a module and its children.

            Parameters
            ----------
            module : nn.Module
                Module whose parameters and child modules are initialized.
            """
            if id(module) in excluded:
                return

            weights_init(module, method=init_method)

            for child in module.children():
                visit(child)

        visit(self)

    @final
    def _get_device(self) -> torch.device:
        """
        Return the device currently used by the architecture.

        The device of the first parameter is preferred, followed by the first
        registered buffer. CPU is returned when the architecture contains neither.

        Returns
        -------
        torch.device
            Device associated with the architecture.
        """
        param = next(self.parameters(), None)

        if param is not None:
            return param.device

        buffer = next(self.buffers(), None)

        if buffer is not None:
            return buffer.device

        return torch.device("cpu")

    @final
    def _load_state_dict(self, checkpoint_config: CheckpointConfig):
        """
        Load architecture parameters from a checkpoint.

        Only state-dictionary entries belonging to the architecture are retained and
        their ``arch.`` prefix is removed before loading. Parameters can optionally be
        frozen after restoration.

        Parameters
        ----------
        checkpoint_config : CheckpointConfig
            Configuration specifying the checkpoint, strict loading behavior, and
            whether loaded parameters are frozen.

        Raises
        ------
        FileNotFoundError
            If the configured checkpoint file does not exist.
        """
        if not Path(checkpoint_config.load_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_config.load_path}"
            )

        checkpoint = torch.load(
            Path(checkpoint_config.load_path),
            map_location=self._get_device(),
            weights_only=True,
        )["module"]

        model_state_dict = {
            key.removeprefix("arch."): value
            for key, value in checkpoint.items()
            if key.startswith("arch.")
        }

        self.load_state_dict(model_state_dict, strict=checkpoint_config.strict)

        del checkpoint
        gc.collect()

        if checkpoint_config.freeze_weights:
            for param in self.parameters():
                param.requires_grad = False


@dataclasses.dataclass
class DeterministicRequest:
    """
    Container for a deterministic architecture request.

    Parameters
    ----------
    input : torch.Tensor
        Model input tensor.
    input_mask : torch.Tensor | None
        Optional validity mask associated with the input.
    added_features : torch.Tensor | None
        Optional additional features supplied with the input.
    output_sample_size : int
        Number of stochastic output samples requested.
    chunk_output_samples : bool
        Whether multiple stochastic outputs are generated separately rather than
        by repeating decoder features in a single pass.
    """

    input: torch.Tensor
    input_mask: torch.Tensor | None = None
    added_features: torch.Tensor | None = None
    output_sample_size: int = 1
    chunk_output_samples: bool = False


class deterministicArchABC(ArchABC):
    """
    Abstract base class for deterministic model architectures.

    Parameters
    ----------
    config : ArchConfigABC
        Architecture configuration.
    """

    def __init__(self, config: ArchConfigABC):
        """
        Initialize a deterministic architecture.

        Parameters
        ----------
        config : ArchConfigABC
            Architecture configuration.
        """
        super().__init__()
        self.config = config
        self.generative_modeling = False

    @abc.abstractmethod
    def forward(
        self,
        request: DeterministicRequest,
    ) -> OutputABC:
        """
        Run the deterministic architecture forward pass.

        Parameters
        ----------
        request : DeterministicRequest
            Request containing inputs, masks, additional features, and output sampling
            settings.

        Returns
        -------
        OutputABC
            Architecture-specific deterministic model output.
        """
        pass


class cVAEArchConfigABC(ArchConfigABC):
    """
    Abstract base configuration for cVAE architectures.

    Attributes
    ----------
    latent_size : int
        Size of the latent representation.
    condition_dependant_latent : bool
        Whether the latent prior depends on the conditioning input.
    condition_embedding_size : int
        Size of the conditioning embedding.
    condition_embedding_dims : list
        Dimensions used to construct the conditioning embedding.
    condemb_to_decoder : bool
        Whether the conditioning embedding is supplied directly to the decoder.
    """

    latent_size: int
    condition_dependant_latent: bool
    condition_embedding_size: int
    condition_embedding_dims: list
    condemb_to_decoder: bool

    def _resolve_flow_settings(self, condition_dependant_flow: bool = False):
        """
        Resolve and validate prior-flow settings.

        Parameters
        ----------
        condition_dependant_flow : bool
            Whether the prior flow is conditioned on the conditioning embedding.

        Returns
        -------
        cVAEArchConfigABC
            This configuration with the flow setting applied.

        Raises
        ------
        ValueError
            If a condition-dependent Gaussian latent prior is used without a
            condition-dependent flow and the conditioning embedding size differs from
            the latent size.
        """
        self.condition_dependant_flow = condition_dependant_flow

        if self.condition_dependant_latent and not self.condition_dependant_flow:
            if self.latent_size != self.condition_embedding_size:
                raise ValueError(
                    f"for condition dependent latent when prior flow is off, "
                    f"condition embedding size ({self.condition_embedding_size}) "
                    f"must equal latent size ({self.latent_size})."
                )

        return self


@dataclasses.dataclass
class cVAEForwardRequest:
    """
    Container for a cVAE training forward request.

    Parameters
    ----------
    target : torch.Tensor
        Target tensor used to infer the posterior latent distribution.
    condition : torch.Tensor
        Conditioning tensor supplied to the cVAE.
    target_mask : torch.Tensor | None
        Optional validity mask associated with the target.
    condition_mask : torch.Tensor | None
        Optional validity mask associated with the conditioning input.
    added_features : torch.Tensor | None
        Optional additional features supplied to the architecture.
    latent_sample_size : int
        Number of posterior latent samples to generate.
    output_sample_size : int
        Number of stochastic outputs generated for each latent sample.
    posterior_variance_limits : list[torch.Tensor, torch.Tensor] | None
        Optional lower and upper limits applied to posterior log-variance.
    """

    target: torch.Tensor
    condition: torch.Tensor
    target_mask: torch.Tensor | None = None
    condition_mask: torch.Tensor | None = None
    added_features: torch.Tensor | None = None
    latent_sample_size: int = 1
    output_sample_size: int = 1
    posterior_variance_limits: list[torch.Tensor, torch.Tensor] | None = None


@dataclasses.dataclass
class cVAEPredictRequest:
    """
    Container for a cVAE prediction request.

    Parameters
    ----------
    condition : torch.Tensor
        Conditioning tensor supplied to the cVAE.
    condition_mask : torch.Tensor | None
        Optional validity mask associated with the conditioning input.
    added_features : torch.Tensor | None
        Optional additional features supplied to the architecture.
    prior_flow : flowABC | None
        Optional normalizing flow used to transform prior latent samples.
    latent_samples : torch.Tensor | None
        Optional user-provided latent samples. If ``None``, samples are generated
        from the configured prior.
    nstds : int
        Standard-deviation multiplier used during latent sampling.
    latent_sample_size : int
        Number of latent samples generated for each input.
    output_sample_size : int
        Number of stochastic outputs generated for each latent sample.
    chunk_size : int | None
        Number of latent samples processed per generation chunk.
    """

    condition: torch.Tensor
    condition_mask: torch.Tensor | None = None
    added_features: torch.Tensor | None = None
    prior_flow: flowABC | None = None
    latent_samples: torch.Tensor | None = None
    nstds: int = 1
    latent_sample_size: int = 1
    output_sample_size: int = 0
    chunk_size: int | None = None


class cVAEArchABC(ArchABC):
    """
    Abstract base class for conditional variational autoencoder architectures.

    Provides shared latent sampling and prior-sampling behavior for cVAE
    architectures.

    Parameters
    ----------
    config : cVAEArchConfigABC
        cVAE architecture configuration.
    """

    def __init__(self, config: cVAEArchConfigABC):
        """
        Initialize a cVAE architecture.

        Parameters
        ----------
        config : cVAEArchConfigABC
            cVAE architecture configuration.
        """
        super().__init__()
        self.generative_modeling = True
        self.config = config
        self.condition_dependant_flow = self.config.condition_dependant_flow

    @abc.abstractmethod
    def forward(
        self,
        request: cVAEForwardRequest,
    ) -> OutputABC:
        """
        Run the cVAE training forward pass.

        Parameters
        ----------
        request : cVAEForwardRequest
            Request containing target, conditioning, masking, and sampling
            information.

        Returns
        -------
        OutputABC
            Architecture-specific cVAE output.
        """
        pass

    @abc.abstractmethod
    def predict(
        self,
        request: cVAEPredictRequest,
    ) -> OutputABC:
        """
        Generate predictions from the cVAE prior.

        Parameters
        ----------
        request : cVAEPredictRequest
            Request containing conditioning and latent sampling information.

        Returns
        -------
        OutputABC
            Architecture-specific generated output.
        """
        pass

    @abc.abstractmethod
    def _recognition(self) -> tuple[torch.Tensor, ...]:
        """
        Infer posterior latent distribution parameters.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Architecture-specific posterior latent parameters.
        """
        pass

    @abc.abstractmethod
    def _condition(self) -> tuple[torch.Tensor, ...]:
        """
        Encode conditioning information into latent parameters or features.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Architecture-specific conditioning representation.
        """
        pass

    @abc.abstractmethod
    def _generate(self) -> torch.Tensor:
        """
        Decode latent values into model outputs.

        Returns
        -------
        torch.Tensor
            Generated model output.
        """
        pass

    @final
    def _sample(self, mu, log_var, sample_size=1, std=1):
        """
        Draw samples from a diagonal Gaussian latent distribution.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent distribution.
        log_var : torch.Tensor
            Log-variance of the latent distribution.
        sample_size : int
            Number of latent samples to draw.
        std : float
            Standard-deviation multiplier applied to the sampling noise.

        Returns
        -------
        torch.Tensor
            Latent samples with a leading sample dimension.
        """
        var = torch.exp(log_var) + 1e-4
        return _sample(mu, var, sample_size, std)

    @final
    def _sample_prior(
        self, request: cVAEPredictRequest
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Sample latent values from the configured prior.

        The conditioning representation is computed first. Samples are then drawn
        from a condition-dependent Gaussian prior, a standard normal prior, or
        provided directly by the request. An optional normalizing flow is applied in
        the inverse direction to prior samples.

        Parameters
        ----------
        request : cVAEPredictRequest
            Prediction request defining conditioning information, latent sampling,
            optional user-provided samples, and an optional prior flow.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
            Latent samples, conditioning latent mean or embedding, and optional
            conditioning log-variance.

        Raises
        ------
        ValueError
            If user-provided latent samples do not have the expected shape.
        """
        condition = request.condition
        latent_samples = request.latent_samples
        latent_sample_size = request.latent_sample_size
        condition_mask = request.condition_mask
        added_features = request.added_features
        prior_flow = request.prior_flow
        nstds = request.nstds

        B, C = condition.shape[:2]
        latent_ref_tensor = torch.zeros(
            (B, self.latent_size), device=condition.device, dtype=condition.dtype
        )

        cond_mu, cond_log_var = self._condition(
            condition=condition,
            condition_mask=condition_mask,
            added_features=added_features,
        )

        if latent_samples is None:
            if self.condition_dependant_latent and not self.condition_dependant_flow:
                latent_samples = self._sample(
                    cond_mu, cond_log_var, latent_sample_size, std=nstds
                )

            else:
                latent_samples = _get_normal(latent_ref_tensor, std=nstds).sample(
                    (latent_sample_size,)
                )

            if prior_flow is not None:
                cond = None
                batch_size, feature_size = latent_samples.shape[1:]

                if prior_flow.condition_size is not None:
                    cond = (
                        cond_mu.unsqueeze(0)
                        .expand(latent_sample_size, -1, -1)
                        .reshape(latent_sample_size * batch_size, -1)
                    )

                latent_samples = latent_samples.reshape(
                    latent_sample_size * batch_size, feature_size
                )

                flow_output = prior_flow.inverse(latent_samples, cond)
                latent_samples = flow_output.e_samples
                latent_samples = latent_samples.reshape(
                    latent_sample_size, batch_size, -1
                )
        else:
            expected_shape = (latent_sample_size, *latent_ref_tensor.shape)
            if not latent_samples.shape == expected_shape:
                raise ValueError(
                    f"Got user specified latent_samples of shape ({latent_samples.shape}) "
                    f"but expected shape {(expected_shape)}"
                )

        return latent_samples, cond_mu, cond_log_var


def weights_init(
    m: nn.Module,
    method: InitMethod = "default",
):
    """
    Initialize parameters of supported neural-network layers.

    Only linear and one-, two-, or three-dimensional convolutional layers are
    modified. Biases are initialized to zero when trainable.

    Parameters
    ----------
    m : nn.Module
        Module whose parameters are initialized.
    method : InitMethod
        Initialization method. ``"default"`` leaves existing parameters
        unchanged.

    Raises
    ------
    NotImplementedError
        If the requested initialization method is unsupported.
    """
    if not isinstance(
        m,
        (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
        ),
    ):
        return

    if method == "default":
        return

    elif method == "kaiming":
        initializer = lambda t: nn.init.kaiming_uniform_(
            t,
            a=math.sqrt(5),
            mode="fan_in",
            nonlinearity="leaky_relu",
        )

    elif method == "xavier":
        initializer = nn.init.xavier_uniform_

    elif method == "trunc_normal":
        initializer = lambda t: trunc_normal_(
            t,
            std=0.02,
        )

    else:
        raise NotImplementedError(
            f'Initialization method "{method}" is not implemented.'
        )

    if m.weight is not None and m.weight.requires_grad:
        initializer(m.weight)

    if m.bias is not None and m.bias.requires_grad:
        nn.init.zeros_(m.bias)

import abc
import math
import numpy as np
import torch
import torch.nn as nn
from typing import final, ClassVar
from timm.models.layers import trunc_normal_
import gc
from pathlib import Path
import dataclasses

from cccma_ppp.architectures.layers.utils import _get_normal
from cccma_ppp.core.core_abc import OutputABC
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.architectures.layers.generic import (
    ActivationName,
    InitMethod,
    NoiseLevel,
)
from cccma_ppp.architectures.layers.utils import _sample


@dataclasses.dataclass
class GENERATORConfig:
    """
    Configuration for stochastic generator behavior.

    Parameters
    ----------
    noise_level : NoiseLevel
        Level controlling where noise is injected within generator-enabled
        architectures.
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
        Complete and validate the generator sampling configuration.

        If the validation sample count is not specified, it is set to the training
        sample count.

        Raises
        ------
        ValueError
            If ``num_validation_noise_samples`` is less than two.
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
    Configuration for loading model weights from a checkpoint.

    Parameters
    ----------
    load_path : Path | str
        Path to the checkpoint file.
    checkpoint_input_shape : np.ndarray
        Input shape expected by the checkpointed model.
    checkpoint_output_shape : np.ndarray
        Output shape expected by the checkpointed model.
    checkpoint_input_var_metadata : dict
        Input-variable metadata associated with the checkpoint.
    checkpoint_output_var_metadata : dict
        Output-variable metadata associated with the checkpoint.
    strict : bool
        Whether checkpoint parameters must exactly match the model state
        dictionary when loading.
    freeze_weights : bool
        Whether to disable gradient updates for all model parameters after
        loading the checkpoint.
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
    Abstract base class for invertible flow transformations.

    Subclasses define transformations between data and latent representations,
    optionally conditioned on additional information.
    """

    @abc.abstractmethod
    def forward(self, x, condition=None):
        """
        Transform data into the flow's latent representation.

        Parameters
        ----------
        x : Any
            Input to transform.
        condition : Any
            Optional conditioning information used by the transformation.
        """
        pass

    @abc.abstractmethod
    def inverse(self, z, condition=None):
        """
        Transform latent values through the inverse flow mapping.

        Parameters
        ----------
        z : Any
            Latent representation to transform.
        condition : Any
            Optional conditioning information used by the inverse transformation.
        """
        pass


class modelConfigABC(abc.ABC):
    """
    Abstract base configuration for model architectures.

    Attributes
    ----------
    activation : ActivationName
        Activation function configured for the model.
    GENERATOR : GENERATORConfig | None
        Optional configuration controlling stochastic generator behavior.
    EXPECTS_MASK : bool
        Whether the model architecture expects validity-mask information.
    """

    activation: ActivationName
    NUM_INPUT_DIMS: ClassVar[int | None]
    NUM_OUTPUT_DIMS: ClassVar[int | None]
    GENERATOR: GENERATORConfig | None
    EXPECTS_MASK: bool

    def __init_subclass__(cls):
        """
        Initialize checkpoint state for a model configuration subclass.

        Each subclass is initialized without an associated checkpoint configuration.
        """
        super().__init_subclass__()
        cls.checkpoint_config = None

    @final
    def _add_checkpoint_config(self, checkpoint_config: CheckpointConfig) -> None:
        """
        Attach checkpoint-loading information to this model configuration.

        Parameters
        ----------
        checkpoint_config : CheckpointConfig
            Checkpoint configuration to associate with the model.
        """
        self.checkpoint_config = checkpoint_config

    @final
    @property
    def NUM_INPUT_DIMS(self) -> int:
        """
        Return the number of dimensions expected by model inputs.

        Returns
        -------
        int
            Required number of input dimensions excluding the batch dimension.
        """
        return self.NUM_INPUT_DIMS

    @final
    @property
    def NUM_OUTPUT_DIMS(self) -> int:
        """
        Return the number of dimensions produced by model outputs.

        Returns
        -------
        int
            Required number of output dimensions excluding the batch dimension.
        """
        return self.NUM_OUTPUT_DIMS

    @final
    @property
    def GENERATOR(self) -> bool:
        """
        Return the generator configuration associated with the model.

        Returns
        -------
        GENERATORConfig | None
            Generator configuration, or ``None`` when generator behavior is
            disabled.
        """
        return getattr(self, "GENERATOR", None)

    @property
    @abc.abstractmethod
    def EXPECTS_MASK(self) -> bool:
        """
        Indicate whether the model architecture expects validity-mask information.

        Returns
        -------
        bool
            Whether validity masks are used by the model.
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
        Build a model from this configuration.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of the model input excluding the batch dimension.
        output_shape : np.ndarray | None
            Shape of the model output excluding the batch dimension.
        added_features_dim : int | None
            Number of additional features supplied to the model.
        **kwargs : Any
            Additional model-specific construction arguments.

        Returns
        -------
        modelABC
            Model constructed from this configuration.
        """
        pass


class modelABC(nn.Module, abc.ABC):
    """
    Abstract base class for model architectures.

    Attributes
    ----------
    generative_modeling : bool
        Whether the model performs generative rather than deterministic modeling.
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
        Validate the current model configuration against checkpoint metadata.

        Checks that the requested input and output shapes and the current variable
        metadata match those associated with the checkpoint.

        Parameters
        ----------
        input_shape : np.ndarray
            Input shape requested for the current model.
        output_shape : np.ndarray
            Output shape requested for the current model.

        Raises
        ------
        RuntimeError
            If the requested input or output shape differs from the checkpoint shape,
            or if the current input or target variable metadata is incompatible with
            the checkpoint metadata.
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
        Perform a forward pass through the model.

        Returns
        -------
        OutputABC
            Model-specific output produced by the forward pass.
        """
        pass

    @final
    def _initialize_weights(
        self, init_method="trunc_normal", exclude: tuple[nn.Module, ...] = ()
    ) -> None:
        """
        Initialize trainable model weights recursively.

        Modules explicitly listed in ``exclude`` are skipped together with their
        descendants.

        Parameters
        ----------
        init_method : InitMethod
            Initialization method applied to supported trainable layers.
        exclude : tuple[nn.Module, ...]
            Modules to exclude from weight initialization.
        """
        excluded = {id(module) for module in exclude}

        def visit(module: nn.Module):
            """
            Initialize a module and recursively visit its children.

            Parameters
            ----------
            module : nn.Module
                Module to initialize unless it belongs to the excluded set.
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
        Determine the device on which the model is stored.

        Returns
        -------
        torch.device
            Device of the first model parameter, or the first model buffer when no
            parameters are present. Returns the CPU device if neither exists.
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
        Load model parameters from a checkpoint.

        Only checkpoint entries belonging to the model are loaded, with the
        ``model.`` prefix removed. Model parameters can optionally be frozen after
        loading.

        Parameters
        ----------
        checkpoint_config : CheckpointConfig
            Configuration specifying the checkpoint path and loading behavior.

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
            key.removeprefix("model."): value
            for key, value in checkpoint.items()
            if key.startswith("model.")
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
    Container for inputs to a deterministic model.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor supplied to the model.
    input_mask : torch.Tensor | None
        Optional validity mask associated with the input.
    added_features : torch.Tensor | None
        Optional additional features supplied to the model.
    output_sample_size : int
        Number of stochastic outputs requested when generator behavior is
        enabled.
    chunk_output_samples : bool
        Whether stochastic output samples are generated separately rather than
        by repeating the decoder input in a single pass.
    """

    input: torch.Tensor
    input_mask: torch.Tensor | None = None
    added_features: torch.Tensor | None = None
    output_sample_size: int = 1
    chunk_output_samples: bool = False


class deterministicmodelsABC(modelABC):
    """
    Abstract base class for deterministic model architectures.

    Parameters
    ----------
    config : modelConfigABC
        Configuration defining the deterministic model.
    """

    def __init__(self, config: modelConfigABC):
        """
        Initialize a deterministic model.

        Parameters
        ----------
        config : modelConfigABC
            Configuration defining the model architecture and behavior.
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
        Perform a forward pass through the deterministic model.

        Parameters
        ----------
        request : DeterministicRequest
            Request containing model inputs and optional sampling information.

        Returns
        -------
        OutputABC
            Model-specific deterministic output.
        """
        pass


class cVAEmodelConfigABC(modelConfigABC):
    """
    Abstract base configuration for conditional variational autoencoders.

    Attributes
    ----------
    latent_size : int
        Size of the latent representation.
    condition_dependant_latent : bool
        Whether the latent prior depends on the conditioning input.
    condition_embedding_size : int
        Size of the conditioning representation.
    condition_embedding_dims : list
        Dimensions used to construct the conditioning representation.
    condemb_to_decoder : bool
        Whether the condition embedding is supplied directly to the decoder.
    """

    latent_size: int
    condition_dependant_latent: bool
    condition_embedding_size: int
    condition_embedding_dims: list
    condemb_to_decoder: bool

    def _resolve_flow_settings(self, condition_dependant_flow: bool = False):
        """
        Configure conditional flow behavior and validate latent dimensions.

        Parameters
        ----------
        condition_dependant_flow : bool
            Whether the prior flow depends on the conditioning representation.

        Returns
        -------
        cVAEmodelConfigABC
            This configuration instance with the flow setting applied.

        Raises
        ------
        ValueError
            If a condition-dependent latent distribution is used without a
            condition-dependent flow and the condition embedding size differs from
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
    Container for inputs to a cVAE training forward pass.

    Parameters
    ----------
    target : torch.Tensor
        Target tensor used to infer the posterior latent distribution.
    condition : torch.Tensor
        Conditioning tensor supplied to the model.
    target_mask : torch.Tensor | None
        Optional validity mask associated with the target.
    condition_mask : torch.Tensor | None
        Optional validity mask associated with the conditioning input.
    added_features : torch.Tensor | None
        Optional additional features supplied to the model.
    latent_sample_size : int
        Number of latent samples drawn from the posterior for each input.
    output_sample_size : int
        Number of stochastic outputs generated for each latent sample when
        generator behavior is enabled.
    posterior_variance_limits : list[torch.Tensor, torch.Tensor] | None
        Optional lower and upper bounds applied to posterior log-variance values.
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
    Container for inputs to cVAE prediction.

    Parameters
    ----------
    condition : torch.Tensor
        Conditioning tensor supplied to the model.
    condition_mask : torch.Tensor | None
        Optional validity mask associated with the conditioning input.
    added_features : torch.Tensor | None
        Optional additional features supplied to the model.
    prior_flow : flowABC | None
        Optional flow used to transform samples from the latent prior.
    latent_samples : torch.Tensor | None
        Optional user-provided latent samples. If ``None``, samples are drawn
        from the configured prior.
    nstds : int
        Scale factor applied to the standard deviation used for latent sampling.
    latent_sample_size : int
        Number of latent samples used for each conditioning input.
    output_sample_size : int
        Number of stochastic outputs requested for each latent sample.
    chunk_size : int | None
        Maximum number of latent samples processed together during chunked
        generation.
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


class cVAEmodelsABC(modelABC):
    """
    Abstract base class for conditional variational autoencoder models.

    Parameters
    ----------
    config : cVAEmodelConfigABC
        Configuration defining the cVAE architecture and latent behavior.
    """

    def __init__(self, config: cVAEmodelConfigABC):
        """
        Initialize a conditional variational autoencoder.

        Parameters
        ----------
        config : cVAEmodelConfigABC
            Configuration defining the cVAE architecture and latent behavior.
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
        Perform a training forward pass through the cVAE.

        Parameters
        ----------
        request : cVAEForwardRequest
            Request containing targets, conditioning inputs, and sampling settings.

        Returns
        -------
        OutputABC
            Model-specific cVAE output.
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
            Request containing conditioning inputs and prior-sampling settings.

        Returns
        -------
        OutputABC
            Model-specific generated output.
        """
        pass

    @abc.abstractmethod
    def _recognition(self) -> tuple[torch.Tensor, ...]:
        """
        Infer posterior latent distribution parameters.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Tensors describing the posterior latent distribution.
        """
        pass

    @abc.abstractmethod
    def _condition(self) -> tuple[torch.Tensor, ...]:
        """
        Encode conditioning information for latent sampling and generation.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Tensors describing the encoded conditioning representation.
        """
        pass

    @abc.abstractmethod
    def _generate(self) -> torch.Tensor:
        """
        Generate model outputs from latent representations.

        Returns
        -------
        torch.Tensor
            Generated model output.
        """
        pass

    @final
    def _sample(self, mu, log_var, sample_size=1, std=1):
        """
        Sample latent values using the reparameterization procedure.

        The latent variance is obtained by exponentiating ``log_var`` and adding a
        small positive offset for numerical stability.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the latent Gaussian distribution.
        log_var : torch.Tensor
            Log-variance of the latent Gaussian distribution.
        sample_size : int
            Number of latent samples to draw.
        std : float
            Standard deviation multiplier applied to the sampling noise.

        Returns
        -------
        torch.Tensor
            Sampled latent values.
        """
        var = torch.exp(log_var) + 1e-4
        return _sample(mu, var, sample_size, std)

    @final
    def _sample_prior(
        self, request: cVAEPredictRequest
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Obtain latent samples for prediction.

        Conditioning information is encoded before latent samples are obtained.
        Samples may come from a condition-dependent Gaussian prior, a standard
        Gaussian prior, user-provided latent values, or an optional inverse flow
        transformation.

        Parameters
        ----------
        request : cVAEPredictRequest
            Prediction request specifying conditioning inputs, sampling parameters,
            optional latent samples, and an optional prior flow.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
            Latent samples, condition mean representation, and optional condition
            log-variance.

        Raises
        ------
        ValueError
            If user-provided latent samples do not have the expected sample, batch,
            and latent dimensions.
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
    Initialize the weights of a supported neural-network layer.

    Initialization is applied to linear and convolutional layers. Unsupported
    module types and the ``"default"`` method leave parameters unchanged. Biases
    are initialized to zero when trainable.

    Parameters
    ----------
    m : nn.Module
        Module whose parameters are to be initialized.
    method : InitMethod
        Weight initialization method to apply.

    Raises
    ------
    NotImplementedError
        If ``method`` is not a supported initialization method.
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

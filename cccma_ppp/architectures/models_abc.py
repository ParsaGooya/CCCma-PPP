import abc
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
from cccma_ppp.architectures.layers.generic import ActivationName, InitMethod, NoiseLevel
from cccma_ppp.architectures.layers.utils import _sample


@dataclasses.dataclass
class GENERATORConfig:
    noise_level: NoiseLevel = "full"
    num_training_noise_samples: int = 10
    num_validation_noise_samples: int | None = None

    def __post_init__(self):
        if self.num_validation_noise_samples is None:
            self.num_validation_noise_samples = self.num_training_noise_samples


@dataclasses.dataclass
class CheckpointConfig:
    """
    Configuration for loading model checkpoints.

    Parameters
    ----------
    load_path : pathlib.Path or str
        Path to checkpoint file.
    checkpoint_input_shape : np.ndarray
        Input shape used during training.
    checkpoint_output_shape : np.ndarray
        Output shape used during training.
    checkpoint_input_var_metadata : dict
        Metadata describing input variables and preprocessing.
    checkpoint_output_var_metadata : dict
        Metadata describing target variables and preprocessing.
    strict : bool, optional
        Whether to strictly enforce state_dict matching.
    freeze_weights : bool, optional
        Whether to freeze model parameters after loading.
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
    Abstract base class for flow-based models.
    """

    @abc.abstractmethod
    def forward(self, x, condition=None):
        """
        Apply forward transformation.

        Parameters
        ----------
        x : torch.Tensor
        condition : torch.Tensor or None, optional

        Returns
        -------
        torch.Tensor
        """

        pass

    @abc.abstractmethod
    def inverse(self, z, condition=None):
        """
        Apply inverse transformation.

        Parameters
        ----------
        z : torch.Tensor
        condition : torch.Tensor or None, optional

        Returns
        -------
        torch.Tensor
        """

        pass


class modelConfigABC(abc.ABC):
    """
    Abstract base class for model configuration.
    """

    activation: ActivationName
    NUM_INPUT_DIMS: ClassVar[int | None]
    NUM_OUTPUT_DIMS: ClassVar[int | None]
    GENERATOR: GENERATORConfig | None
    EXPECTS_MASK: bool

    def __init_subclass__(cls):
        """
        Validate subclass configuration.

        Returns
        -------
        None
        """

        super().__init_subclass__()
        cls.checkpoint_config = None

    @final
    def _add_checkpoint_config(self, checkpoint_config: CheckpointConfig) -> None:
        """
        Attach checkpoint configuration.

        Parameters
        ----------
        checkpoint_config : CheckpointConfig

        Returns
        -------
        None
        """

        self.checkpoint_config = checkpoint_config

    @final
    @property
    def NUM_INPUT_DIMS(self) -> int:
        """
        Return number of input dims in
        the selected architecture.

        Return
        -------
        int
        """

        return self.NUM_INPUT_DIMS

    @final
    @property
    def NUM_OUTPUT_DIMS(self) -> int:
        """
        Return number of output dims in
        the selected architecture.

        Return
        -------
        int
        """

        return self.NUM_OUTPUT_DIMS

    @final
    @property
    def GENERATOR(self) -> bool:
        """
        Check if the selected architecture
        has a GENERATOR.

        Return
        -------
        bool
        """

        return getattr(self, "GENERATOR", None)

    
    @property
    @abc.abstractmethod
    def EXPECTS_MASK(self) -> bool:
        """
        Check if the selected architecture
        EXPECTS_MASK.

        Return
        -------
        bool
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
        Construct model instance.

        Returns
        -------
        modelABC
        """

        pass


class modelABC(nn.Module, abc.ABC):
    """
    Abstract base class for all models.
    """

    generative_modeling: bool

    @final
    def _validate_checkpoint_compatibility(
        self,
        *,
        input_shape: np.ndarray,
        output_shape: np.ndarray,
    ) -> None:
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
        Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """

        pass

    @final
    def _initialize_weights(self, 
                            init_method="trunc_normal",
                            exclude: tuple[nn.Module, ...] = ()
    ) -> None:
        """
        Initialize model weights.

        Returns
        -------
        None
        """
        excluded = {id(module) for module in exclude}

        def visit(module: nn.Module):
            if id(module) in excluded:
                return

            weights_init(module, method=init_method)

            for child in module.children():
                visit(child)

        visit(self)

    @final
    def _get_device(self) -> torch.device:
        """
        Get model device.

        Returns
        -------
        torch.device
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
        Load model weights from checkpoint.

        Parameters
        ----------
        checkpoint_config : CheckpointConfig

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
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
            # self.requires_grad_(False)


@dataclasses.dataclass
class DeterministicRequest:
    """
    Deterministic forward method arguments.

    Parameters
    ----------
    input : torch.Tensor
        Model input.
    input_mask : torch.Tensor
        Model input mask.
    added_features : torch.Tensor or None
        Additional features.
    output_sample_size : int, optional
        Number of output samples for models with GENERATOR.
    """

    input: torch.Tensor
    input_mask: torch.Tensor | None = None
    added_features: torch.Tensor | None = None
    output_sample_size: int = 0


class deterministicmodelsABC(modelABC):
    """
    Base class for deterministic models.
    """

    def __init__(self, config: modelConfigABC):
        """
        Initialize deterministic model.

        Parameters
        ----------
        config : modelConfigABC
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
        Generate samples.

        Returns
        -------
        torch.Tensor
        """

        pass


class cVAEmodelConfigABC(modelConfigABC):
    """
    Abstract base class for cVAE model configurations.
    """

    latent_size: int
    condition_dependant_latent: bool
    condition_embedding_size: int
    condition_embedding_dims: list
    condemb_to_decoder: bool

    def _resolve_flow_settings(self, condition_dependant_flow: bool = False):
        """
        Configure flow-related settings.

        Parameters
        ----------
        condition_dependant_flow : bool, optional

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If configuration is inconsistent.
        """

        self.condition_dependant_flow = condition_dependant_flow

        if (self.condition_dependant_latent and 
            not self.condition_dependant_flow):
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
    cVAE forward arguments.

    Parameters
    ----------
    target : torch.Tensor
        Target to reconstruct.
    condition : torch.Tensor
        Conditioning input.
    target_mask : torch.Tensor
        Mask for target to reconstruct.
    condition_mask : torch.Tensor
        Conditioning input mask.
    added_features : torch.Tensor or None
        Additional features.
    latent_sample_size : int, optional
        Number of latent samples.
    output_sample_size : int, optional
        Number of output samples for models with GENERATOR.
    posterior_variance_limits : list of two torch.Tensors or None, optional
        Minimum variance constraint.
    """

    target: torch.Tensor
    condition: torch.Tensor
    target_mask: torch.Tensor | None = None
    condition_mask: torch.Tensor | None = None
    added_features: torch.Tensor | None = None
    latent_sample_size: int = 1
    output_sample_size: int = 0
    posterior_variance_limits: list[torch.Tensor, torch.Tensor] | None = None


@dataclasses.dataclass
class cVAEPredictRequest:
    """
    cVAE predict arguments.

    Parameters
    ----------
    condition : torch.Tensor
        Conditioning input.
    condition_mask : torch.Tensor
        Conditioning input mask.
    target : torch.Tensor
        Target to reconstruct.
    target_mask : torch.Tensor
        Mask for target to reconstruct.
    added_features : torch.Tensor or None
        Additional features.
    prior_flow : NormalizedFlowModel or None
        Optional flow-based prior.
    latent_samples: torch.Tensor or None
        latent_samples pre-specified by user
    nstds: int
        Adjust the spread in prior samples.
    latent_sample_size : int, optional
        Number of latent samples.
    output_sample_size : int, optional
        Number of output samples for models with GENERATOR.
    """

    condition: torch.Tensor
    condition_mask: torch.Tensor | None = None
    added_features: torch.Tensor | None = None
    prior_flow: flowABC | None = None
    latent_samples: torch.Tensor | None = None
    nstds: int = 1
    latent_sample_size: int = 1
    output_sample_size: int = 0


class cVAEmodelsABC(modelABC):
    """
    Base class for conditional variational autoencoders.
    """

    def __init__(self, config: cVAEmodelConfigABC):
        """
        Initialize cVAE model.

        Parameters
        ----------
        config : modelConfigABC
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
        Generate samples.

        Returns
        -------
        torch.Tensor
        """

        pass

    @abc.abstractmethod
    def predict(
        self,
        request: cVAEPredictRequest,
    ) -> OutputABC:
        """
        Generate samples.

        Returns
        -------
        torch.Tensor
        """

        pass

    @abc.abstractmethod
    def _recognition(self) -> tuple[torch.Tensor, ...]:
        """
        Encode to latent distribution.

        Returns
        -------
        tuple
        """

        pass

    @abc.abstractmethod
    def _condition(self) -> tuple[torch.Tensor, ...]:
        """
        Compute condition representation.

        Returns
        -------
        tuple
            Tuple of condition tensors.
        """

        pass

    @abc.abstractmethod
    def _generate(self) -> torch.Tensor:
        """
        Decode latent representation.

        Returns
        -------
        torch.Tensor
        """

        pass

    @final
    def _sample(self, mu, log_var, sample_size=1, std=1):
        """
        Sample latent variables from Gaussian distribution.

        Parameters
        ----------
        mu : torch.Tensor
        log_var : torch.Tensor
        sample_size : int, optional
        std : float, optional

        Returns
        -------
        torch.Tensor
            Sampled latent variables.
        """

        var = torch.exp(log_var) + 1e-4
        return _sample(mu, var, sample_size, std)

    @final
    def _sample_prior(self, 
                      request: cVAEPredictRequest
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:


        condition= request.condition
        latent_samples= request.latent_samples
        latent_sample_size= request.latent_sample_size
        condition_mask= request.condition_mask
        added_features= request.added_features
        prior_flow= request.prior_flow
        nstds= request.nstds


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
                        cond_mu.unsqueeze(0)  # [1, B, C]
                        .expand(latent_sample_size, -1, -1)  # [S, B, C]
                        .reshape(latent_sample_size * batch_size, -1)
                    )

                latent_samples = latent_samples.reshape(
                    latent_sample_size * batch_size, feature_size
                )

                flow_output = prior_flow.inverse(latent_samples, cond)
                latent_samples = flow_output.e_samples
                latent_samples = latent_samples.reshape(latent_sample_size, batch_size, -1)
        else:
            expected_shape = (latent_sample_size, *latent_ref_tensor.shape)
            if not latent_samples.shape == expected_shape:
                raise ValueError(
                    f"Got user specified latent_samples of shape ({latent_samples.shape}) "
                    f"but expected shape {(expected_shape)}"
                )

        return  latent_samples, cond_mu, cond_log_var


def weights_init(
    m: nn.Module,
    method: InitMethod = "default",
):
    """
    Initialize trainable weights.

    Parameters
    ----------
    m : nn.Module
        Module whose parameters will be initialized.
    method : {"default", "kaiming", "xavier", "trunc_normal"}, optional
        Initialization method. "default" preserves PyTorch's built-in
        initialization.

    Raises
    ------
    NotImplementedError
        If an unsupported initialization method is requested.
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
        # Preserve PyTorch's initialization.
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
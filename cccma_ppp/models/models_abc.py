import abc
import numpy as np
import torch
import torch.nn as nn
from typing import final, ClassVar
from timm.models.layers import trunc_normal_
import gc
from pathlib import Path
import dataclasses

from cccma_ppp.core.core_abc import OutputABC
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.models.layers.generic import ActivationName, InitMethod, NoiseLevel
from cccma_ppp.models.layers.utils import _sample


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
    load_path: Path | str
    checkpoint_input_shape: np.ndarray
    checkpoint_output_shape: np.ndarray
    checkpoint_input_var_metadata: dict
    checkpoint_output_var_metadata: dict
    strict: bool = True
    freeze_weights: bool = False


class flowABC(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x, condition=None):

        pass

    @abc.abstractmethod
    def inverse(self, z, condition=None):

        pass


class modelConfigABC(abc.ABC):
    activation: ActivationName
    NUM_INPUT_DIMS: ClassVar[int | None]
    NUM_OUTPUT_DIMS: ClassVar[int | None]
    GENERATOR: GENERATORConfig | None

    def __init_subclass__(cls):

        super().__init_subclass__()
        cls.checkpoint_config = None

    @final
    def _add_checkpoint_config(self, checkpoint_config: CheckpointConfig) -> None:

        self.checkpoint_config = checkpoint_config

    @final
    def _validate_checkpoint_compatibility(
        self,
        *,
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple,
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
    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
        **kwargs,
    ):

        pass


class modelABC(nn.Module, abc.ABC):
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

        pass

    @final
    def _initialize_weights(self, init_method="trunc_normal"):

        self.apply(lambda m: weights_init(m, method=init_method))

    @final
    def _get_device(self) -> torch.device:

        param = next(self.parameters(), None)

        if param is not None:
            return param.device

        buffer = next(self.buffers(), None)

        if buffer is not None:
            return buffer.device

        return torch.device("cpu")

    @final
    def _load_state_dict(self, checkpoint_config: CheckpointConfig):

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
    input: torch.Tensor
    input_mask: torch.Tensor | None = None
    added_features: torch.Tensor | None = None
    output_sample_size: int = 0


class deterministicmodelsABC(modelABC):
    def __init__(self):

        super().__init__()
        self.generative_modeling = False

    @abc.abstractmethod
    def forward(
        self,
        request: DeterministicRequest,
    ) -> OutputABC:

        pass


class cVAEmodelConfigABC(modelConfigABC):
    latent_size: int
    condition_dependant_latent: bool
    condition_embedding_size: int
    condition_embedding_dims: list
    condemb_to_decoder: bool

    def _resolve_flow_settings(self, condition_dependant_flow: bool = False):

        self.condition_dependant_flow = condition_dependant_flow

        if self.condition_dependant_latent:
            if not self.condition_dependant_flow:
                if self.latent_size != self.condition_embedding_size:
                    raise ValueError(
                        f"for condition dependent latent when prior flow is off, "
                        f"condition embedding size ({self.condition_embedding_size}) "
                        f"must equal latent size ({self.latent_size})."
                    )

        return self


@dataclasses.dataclass
class cVAEForwardRequest:
    target: torch.Tensor
    condition: torch.Tensor
    target_mask: torch.Tensor | None = None
    condition_mask: torch.Tensor | None = None
    added_features: torch.Tensor | None = None
    sample_size: int = 1
    output_sample_size: int = 0
    min_posterior_variance: torch.Tensor | None = None


@dataclasses.dataclass
class cVAEPredictRequest:
    condition: torch.Tensor
    condition_mask: torch.Tensor | None = None
    added_features: torch.Tensor | None = None
    prior_flow: flowABC | None = None
    latent_samples: torch.Tensor | None = None
    nstds: int = 1
    sample_size: int = 1
    output_sample_size: int = 0


class cVAEmodelsABC(modelABC):
    def __init__(self):

        super().__init__()
        self.generative_modeling = True

    @abc.abstractmethod
    def forward(
        self,
        request: cVAEForwardRequest,
    ) -> OutputABC:

        pass

    @abc.abstractmethod
    def predict(
        self,
        request: cVAEPredictRequest,
    ) -> OutputABC:

        pass

    @abc.abstractmethod
    def _recognition(self) -> tuple[torch.Tensor, ...]:

        pass

    @abc.abstractmethod
    def _condition(self) -> tuple[torch.Tensor, ...]:

        pass

    @abc.abstractmethod
    def _generate(self) -> torch.Tensor:

        pass

    @final
    def _sample(self, mu, log_var, sample_size=1, std=1):

        var = torch.exp(log_var) + 1e-4
        return _sample(mu, var, sample_size, std)


def weights_init(m, method: InitMethod = "xavier"):

    if not isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return

    if method == "xavier":

        def initializer(t):
            nn.init.xavier_uniform_(t)

    elif method == "trunc_normal":

        def initializer(t):
            trunc_normal_(t, std=0.02)

    else:
        raise NotImplementedError(
            'initiliazation methods besied "trunc_normal" and "xavier" are not implementd.'
        )

    if hasattr(m, "weight") and m.weight is not None:
        if m.weight.requires_grad:
            initializer(m.weight)

    if hasattr(m, "bias") and m.bias is not None:
        if m.bias.requires_grad:
            nn.init.constant_(m.bias, 0)

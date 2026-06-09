import abc
import numpy as np
import torch
import torch.nn as nn
from typing import final
from timm.models.layers import trunc_normal_
import gc
from pathlib import Path
import dataclasses


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
    NUM_OUTPUT_DIMS = 2
    GENERATOR = False

    def __init__(self):
        self.checkpoint_config: CheckpointConfig | None = None

    @final
    def _add_checkpoint_config(self, checkpoint_config: CheckpointConfig) -> None:
        self.checkpoint_config = checkpoint_config

    @abc.abstractmethod
    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
        **kwargs,
    ):
        pass


class cVAEmodelConfigABC(modelConfigABC, abc.ABC):
    def __init__(
        self,
        latent_size: int,
        condition_dependant_latent: bool = False,
        condition_embedding_size: int = None,
    ):

        super().__init__()
        self.condition_dependant_latent = condition_dependant_latent
        self.latent_size = latent_size
        self.condition_embedding_size = condition_embedding_size

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


class modelABC(nn.Module, abc.ABC):
    def __init__(self, config: modelConfigABC):
        super().__init__()
        self.init_method: str = "trunc_normal"
        self.NUM_OUTPUT_DIMS = config.NUM_OUTPUT_DIMS
        self.GENERATOR = config.GENERATOR

    @abc.abstractmethod
    def forward(self, x):
        pass

    @final
    def _initialize_weights(self):
        self.apply(lambda m: weights_init(m, method=self.init_method))

            
                                                                                     
                                                     

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


class deterministicmodelsABC(modelABC, abc.ABC):
    def __init__(self, config: modelConfigABC):
        super().__init__(config)
        self.generative_modeling = False


class cVAEmodelsABC(modelABC, abc.ABC):
    def __init__(self, config: modelConfigABC):
        super().__init__(config)
        self.generative_modeling = True

    @abc.abstractmethod
    def predict(self, x):
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


def weights_init(m, method="xavier"):

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

import abc
import numpy as np
import torch
import torch.nn as nn
from typing import final
from timm.models.layers import trunc_normal_
import gc
from pathlib import Path
import dataclasses
from typing import Literal


InitMethod = Literal["trunc_normal", "xavier"]


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

    NUM_OUTPUT_DIMS = None
    GENERATOR = False

    def __init__(self):
        """
        Initialize model configuration.

        Returns
        -------
        None
        """
        self.checkpoint_config: CheckpointConfig | None = None

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


class cVAEmodelConfigABC(modelConfigABC, abc.ABC):
    """
    Abstract base class for cVAE model configurations.
    """

    def __init__(
        self,
        latent_size: int,
        condition_dependant_latent: bool = False,
        condition_embedding_size: int = None,
    ):
        """
        Initialize cVAE configuration.

        Parameters
        ----------
        latent_size : int
        condition_dependant_latent : bool, optional
        condition_embedding_size : int or None
        """

        super().__init__()
        self.condition_dependant_latent = condition_dependant_latent
        self.latent_size = latent_size
        self.condition_embedding_size = condition_embedding_size

    def _resolve_flow_settings(
        self,
        condition_dependant_flow: bool = False,
    ):
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
    """
    Abstract base class for all models.
    """

    def __init__(self, config: modelConfigABC):
        """
        Initialize base model.

        Parameters
        ----------
        config : modelConfigABC
        """

        super().__init__()
        self.init_method: InitMethod = "trunc_normal"
        self.NUM_OUTPUT_DIMS = config.NUM_OUTPUT_DIMS
        self.GENERATOR = config.GENERATOR

    @abc.abstractmethod
    def forward(self, x):
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
    def _initialize_weights(self):
        """
        Initialize model weights.

        Returns
        -------
        None
        """
        self.apply(lambda m: weights_init(m, method=self.init_method))

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


class deterministicmodelsABC(modelABC, abc.ABC):
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
        super().__init__(config)
        self.generative_modeling = False


class cVAEmodelsABC(modelABC, abc.ABC):
    """
    Base class for conditional variational autoencoders.
    """

    def __init__(self, config: modelConfigABC):
        """
        Initialize cVAE model.

        Parameters
        ----------
        config : modelConfigABC
        """
        super().__init__(config)
        self.generative_modeling = True

    @abc.abstractmethod
    def predict(self, x):
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


def weights_init(m, method: InitMethod = "xavier"):
    """
    Initialize layer weights.

    Parameters
    ----------
    m : torch.nn.Module
    method : {"xavier", "trunc_normal"}, optional

    Returns
    -------
    None

    Raises
    ------
    NotImplementedError
    """

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

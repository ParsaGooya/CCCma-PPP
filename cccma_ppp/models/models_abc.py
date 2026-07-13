import abc
import numpy as np
import torch
import torch.nn as nn
from typing import final, Literal, ClassVar
from timm.models.layers import trunc_normal_
import gc
from pathlib import Path
import dataclasses

from cccma_ppp.core.core_abc import OutputABC

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
    NUM_INPUT_DIMS: ClassVar[int | None]
    NUM_OUTPUT_DIMS: ClassVar[int | None]
    GENERATOR: ClassVar[bool]

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


class cVAEmodelConfigABC(modelConfigABC):
    """
    Abstract base class for cVAE model configurations.
    """

    latent_size: int
    condition_dependant_latent: bool
    condition_embedding_size: int

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
    def _initialize_weights(self, init_method="trunc_normal"):
        """
        Initialize model weights.

        Returns
        -------
        None
        """

        self.apply(lambda m: weights_init(m, method=init_method))

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


class deterministicmodelsABC(modelABC):
    """
    Base class for deterministic models.
    """

    def __init__(self):
        """
        Initialize deterministic model.

        Parameters
        ----------
        config : modelConfigABC
        """

        super().__init__()
        self.generative_modeling = False


@dataclasses.dataclass
class cVAEPredictRequest:
    """
    cVAE predict method arguments.

    Parameters
    ----------
    condition : torch.Tensor 
        Conditioning input.
    added_features : torch.Tensor or None
        Additional features.
    prior_flow : NormalizedFlowModel or None
        Optional flow-based prior.
    latent_samples: torch.Tensor or None
        latent_samples pre-specified by user
    sample_size : int, optional
        Number of samples.
    """
    condition: torch.Tensor
    added_features: torch.Tensor | None = None
    prior_flow: flowABC | None = None
    latent_samples: torch.Tensor | None = None
    nstds: int = 1
    sample_size: int = 1

class cVAEmodelsABC(modelABC):
    """
    Base class for conditional variational autoencoders.
    """

    def __init__(self):
        """
        Initialize cVAE model.

        Parameters
        ----------
        config : modelConfigABC
        """

        super().__init__()
        self.generative_modeling = True

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

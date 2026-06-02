import torch
import numpy as np
import pathlib as Path
from pathlib import Path
import dataclasses

from cccma_ppp.loss.loss_abc import lossABC
from cccma_ppp.loss.loss import Losspipeline
from cccma_ppp.core.registery import *
from cccma_ppp.core.core_abc import moduleABC
from cccma_ppp.core.selectors import *
from cccma_ppp.data.dataloader import BatchData


@ModuleSelector.register("deterministic")
@ModuleSelector.register("default")
@dataclasses.dataclass
class deterministicConfig:
    """
    Configuration for constructing deterministic model modules.
    """

    ModelConfig: deterministicModelSelector | None = None
    load_dir: str | None = None

    def __post_init__(self):
        """
        Validate configuration inputs.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If neither model configuration nor load directory is provided.
        """

        if self.load_dir is None:
            assert self.ModelConfig is not None, (
                "provide loading dir or model configurations"
            )
        else:
            RuntimeWarning(
                f"all model config overwritten by the loaded model: \n {self.load_dir}"
            )

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Build and return a deterministic module instance.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of input data.
        output_shape : np.ndarray or None, optional
            Shape of output data.
        added_features_dim : int, optional
            Dimension of additional features.

        Returns
        -------
        deterministic
            Constructed deterministic module.
        """

        return deterministic(self).build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


@dataclasses.dataclass
class deterministicOutput:
    """
    Container for deterministic model outputs.
    """

    output: torch.Tensor


class deterministic(moduleABC):
    """
    Deterministic model wrapper implementing module interface.
    """

    def __init__(self, config: deterministicConfig | None = None):
        """
        Initialize deterministic module.

        Parameters
        ----------
        config : deterministicConfig or None, optional
            Configuration for the module.

        Returns
        -------
        None
        """

        super().__init__()
        self.config = config
        self.model = self.config.ModelConfig.get_model()

        self.built = False
        self.criterion = None

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Build the underlying deterministic model.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of input data.
        output_shape : np.ndarray or None, optional
            Shape of output data.
        added_features_dim : int, optional
            Dimension of additional features.

        Returns
        -------
        deterministic
            Built module instance.
        """

        self.model.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

        self.build = True
        if self.config.load_dir is not None:
            self._load_from_state(self.config.load_dir)

        return self

    def init_loss_function(self, reconstruction_loss: Losspipeline):
        """
        Initialize the loss function for training.

        Parameters
        ----------
        reconstruction_loss : Losspipeline
            Reconstruction loss pipeline.

        Returns
        -------
        None
        """

        self.criterion = reconstruction_loss

    def _compute_loss(self, data: BatchData):
        """
        Compute loss for a batch of data.

        Parameters
        ----------
        data : BatchData
            Input batch data.

        Returns
        -------
        tuple
            Total loss tensor and dictionary of loss components.

        Raises
        ------
        AssertionError
            If loss function is not initialized.
        """

        assert self.criterion is not None, (
            "crieterion should be specified before training is possible. Hint: call .init_loss_function() method in your module first."
        )

        output = self.forward(data)

        if isinstance(data.target, (tuple, list)):
            target, target_mask = data.target
        else:
            target, target_mask = data.target, None

        total_loss, indiv_losses = self.criterion(
            output.output, target, target_mask=target_mask, print_loss=False
        )

        losses_dict = {"total_loss": total_loss.item()}

        for key, value in indiv_losses.items():
            losses_dict[key] = value

        return total_loss, losses_dict

    def forward(self, data: BatchData):
        """
        Perform forward pass of the model.

        Parameters
        ----------
        data : BatchData
            Input batch data.

        Returns
        -------
        deterministicOutput
            Model predictions.
        """

        return deterministicOutput(
            output=self.model(x=data.input, added_features=data.added_features)
        )

    def predict(self, data: BatchData):
        """
        Generate predictions from the model.

        Parameters
        ----------
        data : BatchData
            Input batch data.

        Returns
        -------
        deterministicOutput
            Model predictions.
        """

        return deterministicOutput(
            output=self.model(x=data.input, added_features=data.added_features)
        )

    def _get_device(self):
        """
        Retrieve device of module parameters or buffers.

        Returns
        -------
        torch.device
            Device where module resides.
        """

        param = next(self.parameters(), None)

        if param is not None:
            return param.device

        buffer = next(self.buffers(), None)

        if buffer is not None:
            return buffer.device

        return torch.device("cpu")

    def _save_state_dict(self, save_path: Path | str):
        """
        Save model state dictionary to disk.

        Parameters
        ----------
        save_path : Path or str
            Directory to save checkpoint.

        Returns
        -------
        None
        """

        path = Path(save_path) / f"deterministic_module.pt"
        torch.save(self.state_dict(), path)

    def _load_from_state(self, load_path: Path | str, strict: bool = True):
        """
        Load model state dictionary from disk.

        Parameters
        ----------
        load_path : Path or str
            Path to checkpoint file.
        strict : bool, optional
            Whether to enforce strict state loading.

        Returns
        -------
        dict
            Loaded checkpoint data.

        Raises
        ------
        AssertionError
            If module is not built before loading.
        FileNotFoundError
            If checkpoint file does not exist.
        """

        assert self.built, (
            "module stgate should be built for torch to load the weights into. Hint: call .build() method first."
        )

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(
            Path(load_path), map_location=self.device, weights_only=False
        )
        self.load_state_dict(checkpoint, strict=strict)

        return checkpoint

import torch
import numpy as np
from pathlib import Path
import dacite
import dataclasses
import gc
import warnings

from cccma_ppp.loss import Losspipeline
from cccma_ppp.core.core_abc import moduleABC, moduleConfigABC, OutputABC
from cccma_ppp.core.selectors import (
    ModuleSelector,
    deterministicModelSelector,
    _load_config_from_checkpoint,
)
from cccma_ppp.train.dataloader import BatchData
from cccma_ppp.generic.runtime import RuntimeContext


@dataclasses.dataclass
class deterministicOutput(OutputABC):
    """
    Container for deterministic model outputs.

    Parameters
    ----------
    output : torch.Tensor
        Model predictions.
    """

    output: torch.Tensor


@ModuleSelector.register("deterministic")
@ModuleSelector.register("default")
@dataclasses.dataclass
class deterministicConfig(moduleConfigABC):
    """
    Configuration for deterministic models.

    Parameters
    ----------
    ModelConfig : deterministicModelSelector or None
        Model configuration selector.
    load_dir : str or None
        Path to checkpoint for loading configuration.
    """

    ModelConfig: deterministicModelSelector | None = None
    load_dir: str | None = None

    def __post_init__(self):
        """
        Validate and initialize configuration.

        Raises
        ------
        ValueError
            If neither `ModelConfig` nor `load_dir` is provided.
        """

        if self.load_dir is None:
            if self.ModelConfig is None:
                raise ValueError("provide loading dir or model configurations")
        else:
            self._load_from_checkpoint(self.load_dir)
            warnings.warn(
                f"all module config overwritten by the saved module: \n {self.load_dir}"
            )

        self.model_config = self.ModelConfig.get_model_config()

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Construct deterministic module instance.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of input data.
        output_shape : np.ndarray or None, optional
            Shape of output data.
        added_features_dim : int, optional
            Number of additional input features.

        Returns
        -------
        deterministic
            Initialized deterministic module.
        """

        return deterministic(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

    def _load_from_checkpoint(self, load_path: Path | str):
        """
        Load configuration from checkpoint.

        Parameters
        ----------
        load_path : pathlib.Path or str
            Path to checkpoint file.

        Returns
        -------
        deterministicConfig
            Updated configuration instance.
        """

        checkpoint_module, checkpoint_config = _load_config_from_checkpoint(
            Path(load_path)
        )

        self.ModelConfig = dacite.from_dict(
            data_class=deterministicModelSelector,
            data=checkpoint_module.get("ModelConfig"),
            config=dacite.Config(strict=True),
        )

        del checkpoint_config, checkpoint_module
        gc.collect()
        return self


class deterministic(moduleABC):
    """
    Deterministic model module.

    Parameters
    ----------
    config : deterministicConfig
        Configuration object.
    input_shape : np.ndarray
        Shape of input data.
    output_shape : np.ndarray or None
        Shape of output data.
    added_features_dim : int or None
        Number of additional features.
    """

    def __init__(
        self,
        config: deterministicConfig,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Initialize deterministic module and underlying model.

        Parameters
        ----------
        config : deterministicConfig
            Configuration instance.
        input_shape : np.ndarray
            Input shape.
        output_shape : np.ndarray or None
            Output shape.
        added_features_dim : int or None
            Additional feature dimension.

        Raises
        ------
        RuntimeError
            If checkpoint metadata or shapes are inconsistent with current setup.
        """

        super().__init__()
        self.config = config
        self.model_config = config.model_config

        self.criterion = None

        if output_shape is None:
            output_shape = input_shape.copy()

        if self.config.load_dir is not None:
            if (
                RuntimeContext.INPUT_VAR_METADATA
                != self.config.checkpoint_config.checkpoint_input_var_metadata
            ):
                raise RuntimeError(
                    "the loaded module was not trained for the consistent input variables or preprocessing steps"
                )
            if (
                RuntimeContext.TARGET_VAR_METADATA
                != self.config.checkpoint_config.checkpoint_output_var_metadata
            ):
                raise RuntimeError(
                    "the loaded module was not trained for the consistent output variables or preprocessing steps"
                )

            if input_shape != self.config.checkpoint_config.checkpoint_input_shape:
                raise RuntimeError(
                    f"the requested input shape ({input_shape}) does not match the loaded module : {self.config.checkpoint_config.checkpoint_input_shape}"
                )
            if output_shape != self.config.checkpoint_config.checkpoint_output_shape:
                raise RuntimeError(
                    f"the requested output shape ({output_shape}) does not match the loaded module : {self.config.checkpoint_config.checkpoint_output_shape}"
                )

        self.model = self.model_config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

        if self.config.load_dir is not None:
            self._load_state_dict(self.config.load_dir)

    def init_loss_function(self, reconstruction_loss: Losspipeline):
        """
        Initialize loss function.

        Parameters
        ----------
        reconstruction_loss : Losspipeline
            Loss pipeline used for training.

        Returns
        -------
        None
        """

        self.criterion = reconstruction_loss.to(self._get_device())

    def _compute_loss(self, data: BatchData):
        """
        Compute training loss.

        Parameters
        ----------
        data : BatchData
            Input batch.

        Returns
        -------
        tuple
            Total loss tensor and dictionary of loss components.

        Raises
        ------
        RuntimeError
            If loss function has not been initialized.
        """

        if self.criterion is None:
            raise RuntimeError(
                "Criterion should be specified before training is possible. Hint: call .init_loss_function() method in your module first."
            )

        output = self.forward(data)

        target = data.target
        target_mask = data.target_mask

        total_loss, indiv_losses = self.criterion(
            output.output, target, target_mask=target_mask, print_loss=False
        )

        losses_dict = {"total_loss": total_loss.item()}

        for key, value in indiv_losses.items():
            losses_dict[key] = value

        return total_loss, losses_dict

    def forward(self, data: BatchData) -> deterministicOutput:
        """
        Perform forward pass.

        Parameters
        ----------
        data : BatchData
            Input batch.

        Returns
        -------
        deterministicOutput
            Model predictions.
        """

        return self.model(
            x=data.input, x_mask=data.input_mask, added_features=data.added_features
        )

    def predict(self, data: BatchData) -> deterministicOutput:
        """
        Perform inference.

        Parameters
        ----------
        data : BatchData
            Input batch.

        Returns
        -------
        deterministicOutput
            Model predictions.
        """

        return self.forward(data)

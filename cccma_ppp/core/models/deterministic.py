import torch
import numpy as np
from pathlib import Path
import dacite
import dataclasses
import gc
import warnings

from cccma_ppp.loss.loss import Losspipeline
from cccma_ppp.core.core_abc import moduleABC, moduleConfigABC, OutputABC
from cccma_ppp.architectures.models_abc import DeterministicRequest
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
    Document this class.

    Parameters
    ----------
    output : torch.Tensor
        Description not yet provided.
    """

    output: torch.Tensor


@ModuleSelector.register("deterministic")
@ModuleSelector.register("default")
@dataclasses.dataclass
class deterministicConfig(moduleConfigABC):
    """
    Document this class.

    Parameters
    ----------
    ModelConfig : deterministicModelSelector | None
        Description not yet provided.
    load_dir : str | None
        Description not yet provided.
    """

    ModelConfig: deterministicModelSelector | None = None
    load_dir: str | None = None

    def __post_init__(self):
        """
        Document this function.

        Raises
        ------
        ValueError
            Description not yet provided.

        Warns
        -----
        UserWarning
            Description not yet provided.
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
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple | None = None,
        added_features_dim: int = None,
    ):
        """
        Document this function.

        Parameters
        ----------
        input_shape : np.ndarray | tuple
            Description not yet provided.
        output_shape : np.ndarray | tuple | None
            Description not yet provided.
        added_features_dim : int
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return deterministic(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

    def _load_from_checkpoint(self, load_path: Path | str):
        """
        Document this function.

        Parameters
        ----------
        load_path : Path | str
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
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
    Document this class.

    Parameters
    ----------
    config : deterministicConfig
        Description not yet provided.
    input_shape : np.ndarray
        Description not yet provided.
    output_shape : np.ndarray | None
        Description not yet provided.
    added_features_dim : int
        Description not yet provided.
    """

    def __init__(
        self,
        config: deterministicConfig,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Document this function.

        Parameters
        ----------
        config : deterministicConfig
            Description not yet provided.
        input_shape : np.ndarray
            Description not yet provided.
        output_shape : np.ndarray | None
            Description not yet provided.
        added_features_dim : int
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
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

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.added_features_dim = added_features_dim

        self.model = self.model_config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

        if self.config.load_dir is not None:
            self._load_state_dict(self.config.load_dir)

    def init_loss_function(self, reconstruction_loss: Losspipeline):
        """
        Document this function.

        Parameters
        ----------
        reconstruction_loss : Losspipeline
            Description not yet provided.
        """
        self.criterion = reconstruction_loss.to(self._get_device())

    def _compute_loss(self, data: BatchData):
        """
        Document this function.

        Parameters
        ----------
        data : BatchData
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        data : BatchData
            Description not yet provided.

        Returns
        -------
        deterministicOutput
            Description not yet provided.
        """
        generator = self.model_config.GENERATOR
        output_sample_size = 1
        if not self.training and generator is not None:
            output_sample_size = generator.num_validation_noise_samples

        return self.model(
            DeterministicRequest(
                input=data.input,
                input_mask=data.input_mask,
                added_features=data.added_features,
                output_sample_size=output_sample_size,
            )
        )

    def predict(
        self, data: BatchData, 
        output_sample_size: int = 1,
        chunk_output_samples: bool = False
    ) -> deterministicOutput:
        """
        Document this function.

        Parameters
        ----------
        data : BatchData
            Description not yet provided.
        output_sample_size : int
            Description not yet provided.

        Returns
        -------
        deterministicOutput
            Description not yet provided.
        """
        return self.forward(
            DeterministicRequest(
                input=data.input,
                input_mask=data.input_mask,
                added_features=data.added_features,
                output_sample_size=output_sample_size,
                chunk_output_samples=chunk_output_samples
            )
        )

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
    Container for the output of a deterministic model.

    Parameters
    ----------
    output : torch.Tensor
        Tensor produced by the deterministic model.
    """

    output: torch.Tensor


@ModuleSelector.register("deterministic")
@ModuleSelector.register("default")
@dataclasses.dataclass
class deterministicConfig(moduleConfigABC):
    """
    Configuration for a deterministic modeling module.

    Parameters
    ----------
    ModelConfig : deterministicModelSelector | None
        Selector defining the underlying deterministic model architecture.
        Required when a checkpoint directory is not provided.
    load_dir : str | None
        Path to a saved module from which the model configuration is restored.
    """

    ModelConfig: deterministicModelSelector | None = None
    load_dir: str | None = None

    def __post_init__(self):
        """
        Resolve and validate the deterministic module configuration.

        When ``load_dir`` is provided, the saved model configuration is restored from
        the checkpoint and replaces the provided model configuration.

        Raises
        ------
        ValueError
            If neither a model configuration nor a checkpoint directory is provided.

        Warns
        -----
        UserWarning
            If the module configuration is restored from ``load_dir`` and overwrites
            the provided configuration.
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
        Build a deterministic modeling module.

        Parameters
        ----------
        input_shape : np.ndarray | tuple
            Shape of the model input excluding the batch dimension.
        output_shape : np.ndarray | tuple | None
            Shape of the model output excluding the batch dimension. If ``None``,
            ``input_shape`` is used.
        added_features_dim : int | None
            Number of additional features supplied to the model.

        Returns
        -------
        deterministic
            Constructed deterministic modeling module.
        """
        return deterministic(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

    def _load_from_checkpoint(self, load_path: Path | str):
        """
        Restore the model configuration from a saved checkpoint.

        Parameters
        ----------
        load_path : Path | str
            Path to the saved checkpoint.

        Returns
        -------
        deterministicConfig
            This configuration instance populated with the saved model settings.
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
    Training and inference module for deterministic models.

    The module builds the selected deterministic architecture, manages checkpoint
    loading, and provides reconstruction-loss computation for training.

    Parameters
    ----------
    config : deterministicConfig
        Configuration defining the deterministic model and optional checkpoint.
    input_shape : np.ndarray
        Shape of the model input excluding the batch dimension.
    output_shape : np.ndarray | None
        Shape of the model output excluding the batch dimension. If ``None``,
        ``input_shape`` is used.
    added_features_dim : int | None
        Number of additional features supplied to the model.
    """

    def __init__(
        self,
        config: deterministicConfig,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Initialize a deterministic modeling module.

        Parameters
        ----------
        config : deterministicConfig
            Configuration defining the model architecture and optional checkpoint.
        input_shape : np.ndarray
            Shape of the model input excluding the batch dimension.
        output_shape : np.ndarray | None
            Shape of the model output excluding the batch dimension. If ``None``,
            ``input_shape`` is used.
        added_features_dim : int | None
            Number of additional features supplied to the model.

        Raises
        ------
        RuntimeError
            If checkpoint input or output metadata is incompatible with the current
            runtime configuration, or if the requested input or output shape differs
            from that associated with the checkpoint.
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
        Initialize the reconstruction loss used for training.

        Parameters
        ----------
        reconstruction_loss : Losspipeline
            Reconstruction loss pipeline used to compare model outputs with targets.
        """
        self.criterion = reconstruction_loss.to(self._get_device())

    def _compute_loss(self, data: BatchData):
        """
        Compute the reconstruction loss for a batch.

        Parameters
        ----------
        data : BatchData
            Batch containing model inputs, targets, masks, and optional additional
            features.

        Returns
        -------
        tuple[torch.Tensor, dict]
            Total differentiable loss and a dictionary containing scalar loss
            components for reporting.

        Raises
        ------
        RuntimeError
            If the reconstruction loss has not been initialized before computing the
            training objective.
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
        Perform a forward pass through the deterministic model.

        During evaluation, generator-enabled models use the configured number of
        validation noise samples.

        Parameters
        ----------
        data : BatchData
            Batch containing model inputs, masks, and optional additional features.

        Returns
        -------
        deterministicOutput
            Output produced by the deterministic model.
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
        self,
        data: BatchData,
        output_sample_size: int = 1,
        chunk_output_samples: bool = False,
    ) -> deterministicOutput:
        """
        Generate deterministic or generator-enabled predictions for a batch.

        Parameters
        ----------
        data : BatchData
            Batch containing model inputs, masks, and optional additional features.
        output_sample_size : int
            Number of stochastic outputs requested when generator behavior is
            enabled.
        chunk_output_samples : bool
            Whether stochastic output samples are generated separately rather than
            by repeating the decoder input in a single pass.

        Returns
        -------
        deterministicOutput
            Model predictions for the requested batch and sampling configuration.
        """
        return self.forward(
            DeterministicRequest(
                input=data.input,
                input_mask=data.input_mask,
                added_features=data.added_features,
                output_sample_size=output_sample_size,
                chunk_output_samples=chunk_output_samples,
            )
        )

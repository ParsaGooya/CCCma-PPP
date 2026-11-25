import torch
import numpy as np
from pathlib import Path
import dacite
import dataclasses
import gc
import warnings

from S2D4P.loss.loss import Losspipeline
from S2D4P.core.core_abc import moduleABC, moduleConfigABC, OutputABC
from S2D4P.architectures.architecture_abc import DeterministicRequest
from S2D4P.core.selectors import (
    ModuleSelector,
    deterministicArchSelector,
    _load_config_from_checkpoint,
)
from S2D4P.train.dataloader import BatchData

@dataclasses.dataclass
class deterministicOutput(OutputABC):
    """
    Container for deterministic model output.

    Parameters
    ----------
    output : torch.Tensor
        Model predictions produced by the deterministic architecture.
    """

    output: torch.Tensor


@ModuleSelector.register("deterministic")
@ModuleSelector.register("default")
@dataclasses.dataclass
class deterministicConfig(moduleConfigABC):
    """
    Configuration for a deterministic model module.

    Parameters
    ----------
    Architecture : deterministicArchSelector | None
        Selector defining the deterministic model architecture.
    load_dir : str | None
        Path to a saved module checkpoint. If provided, the saved architecture
        configuration is used instead of the explicitly configured architecture.
    """

    Architecture: deterministicArchSelector | None = None
    load_dir: str | None = None

    def __post_init__(self):
        """
        Resolve the deterministic module configuration.

        When a checkpoint is provided, the saved architecture configuration replaces
        the explicitly configured architecture.

        Raises
        ------
        ValueError
            If neither an architecture configuration nor a checkpoint path is
            provided.

        Warns
        -----
        UserWarning
            If a checkpoint is loaded and its saved module configuration replaces the
            current settings.
        """
        if self.load_dir is None:
            if self.Architecture is None:
                raise ValueError("provide loading dir or architecture configurations")
        else:
            self._load_from_checkpoint(self.load_dir)
            warnings.warn(
                f"all module config overwritten by the saved module: \n {self.load_dir}"
            )

        self.architecture_config = self.Architecture.get_architecture_config()

    def build(
        self,
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple | None = None,
        added_features_dim: int = None,
    ):
        """
        Build a deterministic model module.

        Parameters
        ----------
        input_shape : np.ndarray | tuple
            Shape of each model input excluding the batch dimension.
        output_shape : np.ndarray | tuple | None
            Shape of each model output excluding the batch dimension. If ``None``,
            the input shape is used.
        added_features_dim : int | None
            Number of additional features supplied with each sample.

        Returns
        -------
        deterministic
            Constructed deterministic model module.
        """
        return deterministic(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

    def _load_from_checkpoint(self, load_path: Path | str):
        """
        Restore the deterministic architecture configuration from a checkpoint.

        Parameters
        ----------
        load_path : Path | str
            Path to the checkpoint containing the saved module configuration.

        Returns
        -------
        deterministicConfig
            This configuration populated with the checkpoint architecture settings.
        """
        checkpoint_module, checkpoint_config = _load_config_from_checkpoint(
            Path(load_path)
        )

        self.Architecture = dacite.from_dict(
            data_class=deterministicArchSelector,
            data=checkpoint_module.get("Architecture"),
            config=dacite.Config(strict=True),
        )

        del checkpoint_config, checkpoint_module
        gc.collect()
        return self


class deterministic(moduleABC):
    """
    Deterministic model module.

    Wraps a configured deterministic architecture and reconstruction loss while
    providing common training-loss, forward, and prediction interfaces.

    Parameters
    ----------
    config : deterministicConfig
        Configuration defining the deterministic architecture.
    input_shape : np.ndarray
        Shape of each model input excluding the batch dimension.
    output_shape : np.ndarray | None
        Shape of each model output excluding the batch dimension. If ``None``,
        the input shape is used.
    added_features_dim : int | None
        Number of additional features supplied with each sample.
    """

    def __init__(
        self,
        config: deterministicConfig,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Initialize the deterministic model module.

        Builds the selected architecture and validates current data metadata and
        shapes against checkpoint information when loading a saved model.

        Parameters
        ----------
        config : deterministicConfig
            Configuration defining the deterministic module.
        input_shape : np.ndarray
            Shape of each model input excluding the batch dimension.
        output_shape : np.ndarray | None
            Shape of each model output excluding the batch dimension. If ``None``,
            the input shape is used.
        added_features_dim : int | None
            Number of additional features supplied with each sample.

        Raises
        ------
        RuntimeError
            If current input or target metadata, input shape, or output shape are
            incompatible with the loaded checkpoint.
        """
        super().__init__()
        
        self.model_setup(config, input_shape, output_shape, added_features_dim)

        self.arch = self.architecture_config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

        if self.config.load_dir is not None:
            self._load_state_dict(self.config.load_dir)

    def init_loss_function(self, reconstruction_loss: Losspipeline):
        """
        Initialize the reconstruction loss.

        Parameters
        ----------
        reconstruction_loss : Losspipeline
            Reconstruction loss pipeline used during deterministic training.
        """
        self.criterion = reconstruction_loss.to(self._get_device())

    def _compute_loss(self, data: BatchData):
        """
        Compute the deterministic training loss for a batch.

        Model predictions are compared with the target using the configured
        reconstruction loss and optional target validity mask.

        Parameters
        ----------
        data : BatchData
            Training batch containing inputs, targets, masks, and optional additional
            features.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor | float]]
            Differentiable total loss and individual loss terms used for logging.

        Raises
        ------
        RuntimeError
            If the reconstruction loss has not been initialized.
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
        Run the deterministic model forward pass.

        During validation, generator-enabled architectures use the configured number
        of validation noise samples.

        Parameters
        ----------
        data : BatchData
            Batch containing model inputs, masks, and optional additional features.

        Returns
        -------
        deterministicOutput
            Model predictions produced by the configured architecture.
        """
        generator = self.architecture_config.GENERATOR
        output_sample_size = 1
        if not self.training and generator is not None:
            output_sample_size = generator.num_validation_noise_samples

        return self.arch(
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
        Generate deterministic or generator-sampled predictions.

        Parameters
        ----------
        data : BatchData
            Batch containing model inputs, masks, and optional additional features.
        output_sample_size : int
            Number of output samples requested from the architecture.
        chunk_output_samples : bool
            Whether multiple stochastic output samples are generated separately
            rather than by repeating decoder features in a single pass.

        Returns
        -------
        deterministicOutput
            Generated model predictions.
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

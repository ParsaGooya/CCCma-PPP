import torch
from pathlib import Path
import dataclasses
from typing import ClassVar

from S2D4P.core.trainer import clear_memory
from S2D4P.generic.distributed import Distributed
from S2D4P.generic.aggregator import RunningCovariance
from S2D4P.core.core_abc import moduleABC
from S2D4P.core.models.deterministic import deterministicOutput
from S2D4P.inference.predictors.predictor_abc import (
    PredictorABC,
    save_batch_to_netcdf,
)
from S2D4P.data.dataloader import BatchDataABC


@dataclasses.dataclass
class DeterministicPredictorConfig:
    """
    Configuration for deterministic model inference.

    Parameters
    ----------
    chunk_output_samples : bool
        Whether multiple stochastic output samples are generated separately
        instead of repeating decoder features in a single pass.
    """

    chunk_output_samples: bool = False
    _type: ClassVar[str] = "deterministic"

    def build(
        self,
        module: moduleABC,
        distributed: Distributed,
        output_dir: Path | str,
        num_output_sampling: int = 1,
    ):
        """
        Build a deterministic predictor.

        Parameters
        ----------
        module : moduleABC
            Trained deterministic model module used for inference.
        distributed : Distributed
            Distributed runtime used for device placement and rank information.
        output_dir : Path | str
            Directory in which inference outputs and training statistics are stored.
        num_output_sampling : int
            Number of output samples generated for each input.

        Returns
        -------
        DetermninisticPredictor
            Configured deterministic predictor.
        """
        return DetermninisticPredictor(
            self, module, distributed, output_dir, num_output_sampling
        )


class DetermninisticPredictor(PredictorABC):
    """
    Run inference for a deterministic model.

    Supports direct model prediction, generator-based output sampling, and
    sampling based on residual covariance estimated from training data.

    Parameters
    ----------
    config : DeterministicPredictorConfig
        Configuration controlling deterministic inference behavior.
    module : moduleABC
        Trained deterministic model module used for inference.
    distributed : Distributed
        Distributed runtime used for device placement and rank information.
    output_dir : Path | str
        Directory in which inference outputs and training statistics are stored.
    num_output_sampling : int
        Number of output samples generated for each input.
    """

    def __init__(
        self,
        config: DeterministicPredictorConfig,
        module: moduleABC,
        distributed: Distributed,
        output_dir: Path | str,
        num_output_sampling: int = 1,
    ):
        """
        Initialize the deterministic predictor.

        When the architecture does not provide generator-based sampling, multiple
        requested outputs are produced using covariance estimated from training
        residuals.

        Parameters
        ----------
        config : DeterministicPredictorConfig
            Configuration controlling deterministic inference.
        module : moduleABC
            Trained deterministic model module used for inference.
        distributed : Distributed
            Distributed runtime used for device placement and rank information.
        output_dir : Path | str
            Directory in which inference outputs and training statistics are stored.
        num_output_sampling : int
            Number of output samples generated for each input.

        Raises
        ------
        ValueError
            If ``num_output_sampling`` is not positive.
        """
        self.config = config
        self.module = module
        self.num_output_sampling = num_output_sampling
        self.num_output_covariance_sampling = 1
        self.output_dir = Path(output_dir)

        if num_output_sampling <= 0:
            raise ValueError("num_output_sampling must be larger than 1.")

        if module.architecture_config.GENERATOR is None:
            self.num_output_covariance_sampling = num_output_sampling

        self.distributed = distributed
        self.device = distributed.device

        if self.extract_training_vars:
            self._stats = {
                "residual": RunningCovariance(self.distributed),
            }

        self.output_sampler = None
        self._batch_counter = 0

    @property
    def extract_training_vars(self):
        """
        Indicate whether training residual statistics must be extracted.

        Returns
        -------
        bool
            ``True`` when multiple outputs require covariance statistics derived from
            training residuals.
        """

        return self.num_output_covariance_sampling > 1

    @torch.no_grad()
    def _infer_on_batch(
        self,
        batch: BatchDataABC,
        _getting_train_stats: bool = False,
    ) -> deterministicOutput | dict[str, RunningCovariance]:
        """
        Run deterministic inference or statistic extraction on one batch.

        When training statistics are requested, deterministic residuals are used to
        update running covariance statistics. During prediction, output samples are
        generated either by the architecture itself or by adding sampled decoder
        noise based on training residuals.

        Parameters
        ----------
        batch : BatchDataABC
            Batch containing model inputs, optional targets, masks, metadata, and
            additional features.
        _getting_train_stats : bool
            Whether to update training residual statistics instead of producing
            inference output.

        Returns
        -------
        deterministicOutput | dict[str, RunningCovariance]
            Model output during inference or updated residual covariance statistics
            during training-statistic extraction.

        Raises
        ------
        RuntimeError
            If training residual statistics are requested but the batch does not
            contain targets.
        """
        clear_memory()
        self.raw_module.eval()

        with torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            if _getting_train_stats:
                if batch.target is None:
                    raise RuntimeError(
                        "to save the posterior variables the dataset must contain the target prediction."
                    )
                output = self.raw_module.forward(data=batch)

            if _getting_train_stats:
                stats = self._update_train_stats(output, batch)
                return stats

            if self.num_output_covariance_sampling > 1:
                num_output_sampling = 1
            else:
                num_output_sampling = self.num_output_sampling

            output = self.raw_module.predict(
                data=batch,
                output_sample_size=num_output_sampling,
                chunk_output_samples=self.config.chunk_output_samples,
            )

            if self.num_output_covariance_sampling > 1:
                sample_size = (output.output.shape[0],)
                reshape_size = output.output.shape[1:]
                output = self.add_decoder_noise(
                    output,
                    self.num_output_covariance_sampling,
                    sample_size,
                    reshape_size,
                )

            self._batch_to_netcdf(output, batch.metadata)
            return output

    def _update_train_stats(
        self,
        output: deterministicOutput,
        data: BatchDataABC,
    ) -> dict[str, RunningCovariance]:
        """
        Update covariance statistics from deterministic training residuals.

        Parameters
        ----------
        output : deterministicOutput
            Deterministic model predictions for the batch.
        data : BatchDataABC
            Batch containing the corresponding target values.

        Returns
        -------
        dict[str, RunningCovariance]
            Updated running covariance statistics for flattened prediction residuals.
        """
        prediction = output.output
        target = data.target
        residual = target - prediction
        residual = residual.reshape(residual.shape[0], -1)

        self.stats["residual"].update(residual)

        return self.stats

    def _batch_to_netcdf(
        self,
        output: deterministicOutput,
        metadata: list[dict],
    ):
        """
        Save a batch of deterministic predictions to a temporary NetCDF file.

        The output-sample dimension is retained for later aggregation, and the
        temporary filename includes the distributed rank and batch number.

        Parameters
        ----------
        output : deterministicOutput
            Deterministic predictions to save.
        metadata : list[dict]
            Per-sample coordinate metadata used to construct the NetCDF output.
        """
        prediction = output.output

        if self.num_output_sampling == 1:
            prediction = prediction.unsqueeze(0)

        num_output_dims = self.raw_module.architecture_config.NUM_OUTPUT_DIMS
        extra_dims_sorted = ["output_samples"]
        assign_coords = None
        attrs = {"source": "predictions"}

        save_name = f"rank{self.distributed.rank}_{self._batch_counter:08d}.nc"

        save_batch_to_netcdf(
            prediction,
            metadata,
            num_output_dims,
            save_name,
            self.temp_save_dir,
            extra_dims_sorted,
            assign_coords,
            attrs,
        )

        self._batch_counter += 1

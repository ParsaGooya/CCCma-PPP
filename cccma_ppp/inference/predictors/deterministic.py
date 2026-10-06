import torch
from pathlib import Path
import dataclasses
from typing import ClassVar

from cccma_ppp.core.trainer import clear_memory
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.aggregator import RunningCovariance
from cccma_ppp.core.core_abc import moduleABC
from cccma_ppp.core.models.deterministic import deterministicOutput
from cccma_ppp.inference.predictors.predictor_abc import (
    PredictorABC,
    save_batch_to_netcdf,
)
from cccma_ppp.data_modules.dataloader import BatchDataABC


@dataclasses.dataclass
class DeterministicPredictorConfig:
    """
    Configuration for deterministic inference.

    Parameters
    ----------
    chunk_output_samples : bool
        Whether stochastic output samples are generated separately rather than
        by repeating the decoder input in a single pass.
    """

    chunk_output_samples: bool = (False,)
    _type: ClassVar[str] = "deterministic"

    def build(
        self,
        module: moduleABC,
        distributed: Distributed,
        output_dir: Path | str,
        num_output_sampling: int = 1,
    ):
        """
        Build a deterministic predictor from this configuration.

        Parameters
        ----------
        module : moduleABC
            Trained deterministic module used for inference.
        distributed : Distributed
            Distributed runtime associated with inference.
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
    Perform inference with a deterministic model.

    The predictor supports multiple output samples, empirical residual covariance
    sampling for models without generator behavior, extraction of training
    residual statistics, and saving predictions to temporary NetCDF files.

    Parameters
    ----------
    config : DeterministicPredictorConfig
        Configuration controlling deterministic inference behavior.
    module : moduleABC
        Trained deterministic module used for inference.
    distributed : Distributed
        Distributed runtime associated with inference.
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
        Initialize a deterministic predictor.

        Parameters
        ----------
        config : DeterministicPredictorConfig
            Configuration controlling deterministic inference behavior.
        module : moduleABC
            Trained deterministic module used for inference.
        distributed : Distributed
            Distributed runtime associated with inference.
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

        if module.model_config.GENERATOR is None:
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
            ``True`` when multiple output samples require empirical residual
            covariance sampling.
        """

        return self.num_output_covariance_sampling > 1

    @torch.no_grad()
    def _infer_on_batch(
        self,
        batch: BatchDataABC,
        _getting_train_stats: bool = False,
    ) -> deterministicOutput | dict[str, RunningCovariance]:
        """
        Perform deterministic inference or training-statistic extraction for a batch.

        When training statistics are requested, model residuals are accumulated for
        subsequent covariance-based output sampling. Otherwise, predictions are
        generated and optional decoder noise is added before the batch is saved.

        Parameters
        ----------
        batch : BatchDataABC
            Batch containing model inputs, optional targets, and metadata.
        _getting_train_stats : bool
            Whether to collect training residual statistics instead of generating
            inference outputs.

        Returns
        -------
        deterministicOutput | dict[str, RunningCovariance]
            Deterministic model output, or running training-statistic accumulators
            when ``_getting_train_stats`` is enabled.

        Raises
        ------
        RuntimeError
            If training-statistic extraction is requested but the batch does not
            contain target data.
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
        Update training residual statistics from deterministic predictions.

        Residuals between target and predicted values are flattened for each batch
        element and accumulated into the running covariance estimator.

        Parameters
        ----------
        output : deterministicOutput
            Deterministic model output used to calculate residuals.
        data : BatchDataABC
            Training batch containing the corresponding target values.

        Returns
        -------
        dict[str, RunningCovariance]
            Updated running covariance accumulators.
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

        The output-sample dimension is added when necessary, and sample metadata are
        used to construct the corresponding output coordinates.

        Parameters
        ----------
        output : deterministicOutput
            Deterministic model output to save.
        metadata : list[dict]
            Per-sample metadata used to construct output coordinates.
        """
        prediction = output.output

        if self.num_output_sampling == 1:
            prediction = prediction.unsqueeze(0)

        num_output_dims = self.raw_module.model_config.NUM_OUTPUT_DIMS
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

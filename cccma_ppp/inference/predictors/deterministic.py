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
    Document this class.
    """

    _type: ClassVar[str] = "deterministic"

    def build(
        self,
        module: moduleABC,
        distributed: Distributed,
        output_dir: Path | str,
        num_output_sampling: int = 1,
    ):
        """
        Document this function.

        Parameters
        ----------
        module : moduleABC
            Description not yet provided.
        distributed : Distributed
            Description not yet provided.
        output_dir : Path | str
            Description not yet provided.
        num_output_sampling : int
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return DetermninisticPredictor(
            self, module, distributed, output_dir, num_output_sampling
        )


class DetermninisticPredictor(PredictorABC):
    """
    Document this class.

    Parameters
    ----------
    config : DeterministicPredictorConfig
        Description not yet provided.
    module : moduleABC
        Description not yet provided.
    distributed : Distributed
        Description not yet provided.
    output_dir : Path | str
        Description not yet provided.
    num_output_sampling : int
        Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        config : DeterministicPredictorConfig
            Description not yet provided.
        module : moduleABC
            Description not yet provided.
        distributed : Distributed
            Description not yet provided.
        output_dir : Path | str
            Description not yet provided.
        num_output_sampling : int
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
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
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """

        return self.num_output_covariance_sampling > 1

    @torch.no_grad()
    def _infer_on_batch(
        self,
        batch: BatchDataABC,
        _getting_train_stats: bool = False,
    ) -> deterministicOutput | dict[str, RunningCovariance]:
        """
        Document this function.

        Parameters
        ----------
        batch : BatchDataABC
            Description not yet provided.
        _getting_train_stats : bool
            Description not yet provided.

        Returns
        -------
        deterministicOutput | dict[str, RunningCovariance]
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
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
                data=batch, output_sample_size=num_output_sampling
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
        Document this function.

        Parameters
        ----------
        output : deterministicOutput
            Description not yet provided.
        data : BatchDataABC
            Description not yet provided.

        Returns
        -------
        dict[str, RunningCovariance]
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        output : deterministicOutput
            Description not yet provided.
        metadata : list[dict]
            Description not yet provided.
        """
        prediction = output.output

        if self.num_output_sampling == 1:
            prediction = prediction.unsqueeze(0)

        num_output_dims = self.raw_module.model_config.NUM_OUTPUT_DIMS
        extra_dims_sorted = ["output_samples"]
        assign_coords = None
        save_name = (
            f"prediction_rank{self.distributed.rank}_{self._batch_counter:08d}.nc"
        )

        save_batch_to_netcdf(
            prediction,
            metadata,
            num_output_dims,
            save_name,
            self.temp_save_dir,
            extra_dims_sorted,
            assign_coords,
        )

        self._batch_counter += 1

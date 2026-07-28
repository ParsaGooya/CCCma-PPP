import torch
from pathlib import Path
import dataclasses
import torch.nn.functional as F
from typing import Callable, ClassVar
import warnings

from cccma_ppp.core.trainer import clear_memory
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.aggregator import RunningCovariance
from cccma_ppp.core.core_abc import moduleABC
from cccma_ppp.core.cVAE_module import cVAEOutput
from cccma_ppp.core.deterministic_module import deterministicOutput
from cccma_ppp.inference.predictor_abc import PredictorABC, save_batch_to_netcdf
from cccma_ppp.data_modules.dataloader import BatchDataABC


@dataclasses.dataclass
class cVAEPredictorConfig:
    num_latent_samples: int
    nstds: float = 1.0
    infer_latent_samples_from_training: bool = False
    save_latent: bool = False

    _type: ClassVar[str] = "cvae"

    def __post_init__(self) -> None:
        if self.num_latent_samples < 1:
            raise ValueError("num_latent_samples must be at least 1.")

        if self.nstds <= 0:
            raise ValueError("nstds must be positive.")

        if self.save_latent:
            warnings.warn(
                "\n===================================================\n"
                + "save_latent is True. No predictions will be saved! \n"
                + "===================================================\n"
            )

    def build(
        self,
        module: moduleABC,
        distributed: Distributed,
        output_dir: Path | str,
        num_output_sampling: int | None = None,
    ):

        return cVAEPredictor(self, module, distributed, output_dir, num_output_sampling)


class cVAEPredictor(PredictorABC):
    def __init__(
        self,
        config: cVAEPredictorConfig,
        module: moduleABC,
        distributed: Distributed,
        output_dir: Path | str,
        num_output_sampling: int | None = None,
    ):

        self.config = config
        self.module = module
        self.num_output_sampling = num_output_sampling
        self.num_output_covariance_sampling = None
        self.output_dir = Path(output_dir)

        if num_output_sampling <= 0:
            raise ValueError("num_output_sampling must be larger than 1.")

        if module.config.GENERATOR is None:
            self.num_output_covariance_sampling = num_output_sampling or 0

        self.num_latent_samples = config.num_latent_samples
        self.infer_latent_samples_from_training = (
            config.infer_latent_samples_from_training
        )
        self.nstds = config.nstds
        self.save_latent = config.save_latent

        self.distributed = distributed
        self.device = distributed.device

        if self.extract_training_vars:
            self._stats = {}
            if self.extract_training_residuals:
                self._stats["residual"] = RunningCovariance(self.distributed)
            if self.extract_posterior_samples:
                self._stats["samples"] = RunningCovariance(self.distributed)

        self.latent_sampler = None
        self.output_sampler = None
        self._batch_counter = 0

    @property
    def extract_training_vars(self):
        return any([self.extract_posterior_samples, self.extract_training_residuals])

    @property
    def extract_posterior_samples(self):
        return self.infer_latent_samples_from_training

    @property
    def extract_training_residuals(self):
        return (self.num_output_covariance_sampling or 0) > 0

    @torch.no_grad()
    def _infer_on_batch(
        self,
        batch: BatchDataABC,
        _getting_train_stats: bool = False,
    ) -> cVAEOutput | dict[str, RunningCovariance]:

        clear_memory()
        self.raw_module.eval()
        latent_samples = None

        with torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            if _getting_train_stats or self.save_latent:
                if batch.target is None:
                    raise RuntimeError(
                        "to save the posterior variables the dataset must contain the target prediction."
                    )
                output = self.raw_module.forward(data=batch, sample_size=1)

            if _getting_train_stats:
                stats = self._update_train_stats(output, batch)
                return stats

            if self.save_latent:
                self._batch_to_netcdf(output, batch.metadata)
                return output

            if self.infer_latent_samples_from_training:
                latent_samples = self._get_latent_samples_based_on_train(data=batch)

            output = self.raw_module.predict(
                data=batch,
                sample_size=self.num_latent_samples,
                nstds=self.nstds,
                latent_samples=latent_samples,
                output_sample_size=self.num_output_sampling,
            )

            if self.num_output_covariance_sampling > 0:
                sample_size = output.output.shape[:2]  # N x B
                reshape_size = output.output.shape[2:]  # C x ...
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
        output: cVAEOutput,
        data: BatchDataABC,
    ) -> dict[str, RunningCovariance]:

        if self.extract_posterior_samples:
            if output.samples is None:
                raise RuntimeError(
                    "cVAEOutput.samples is required for training latent stats."
                )

            samples = output.samples.reshape(-1, output.samples.shape[-1])
            self.stats["samples"].update(samples)

        if self.extract_training_residuals:
            prediction = output.output
            target = data.target
            residual = target - prediction[0]
            residual = residual.reshape(residual.shape[0], -1)

            self.stats["residual"].update(residual)

        return self.stats

    def _get_latent_samples_based_on_train(self, data: BatchDataABC):
        if self.infer_latent_samples_from_training and self.latent_sampler is None:
            self.latent_sampler = self.build_latent_sampler()

        x = data.input
        batch_size = x.shape[0]

        sample_size = (self.num_latent_samples, batch_size)

        latent_samples = self.latent_sampler(sample_size, self.nstds)

        return latent_samples.to(self.device)

    def build_latent_sampler(self) -> Callable[..., torch.Tensor]:
        stats_path = self.output_dir / "training_variable_stats.pt"

        if not stats_path.exists():
            raise ValueError(
                "Training statistics based on the trained model must be saved "
                "to disk first."
            )

        stats = torch.load(
            stats_path,
            map_location="cpu",
            weights_only=True,
        )

        if not all(
            [
                stats.get("samples_mean", None) is not None,
                stats.get("samples_cov", None) is not None,
            ]
        ):
            raise ValueError("The loaded training stats is not for a cVAE model.")

        def _sampler(sample_size: int | tuple[int, ...], std: float):
            return self._sample(
                stats["samples_mean"],
                stats["samples_cov"],
                sample_size,
                std,
            )

        return _sampler

    def _batch_to_netcdf(
        self,
        output: cVAEOutput,
        metadata: list[dict],
    ):
        attrs = None
        if self.save_latent:
            latent_vars = {
                "mu": output.mu,
                "log_var": output.log_var,
                "samples": output.samples,
                "cond_mu": output.cond_mu,
                "cond_log_var": output.cond_log_var,
            }

            latent_vars = {
                name: value for name, value in latent_vars.items() if value is not None
            }

            if not latent_vars:
                raise RuntimeError("No latent variables are available to save.")

            max_latent_dim = max(value.shape[-1] for value in latent_vars.values())

            prepared = []

            for name, value in latent_vars.items():
                value = value.detach().cpu().squeeze()

                pad_size = max_latent_dim - value.shape[-1]

                if pad_size > 0:
                    value = F.pad(
                        value,
                        pad=(0, pad_size),
                        mode="constant",
                        value=float("-inf"),
                    )

                prepared.append(value.unsqueeze(-2))

            prediction = torch.cat(prepared, dim=-2)

            assign_coords = {
                "channels": list(latent_vars),
            }

            num_output_dims = 1
            extra_dims_sorted = []
            save_name = (
                f"latent_rank{self.distributed.rank}_{self._batch_counter:08d}.nc"
            )

            if self.infer_latent_samples_from_training:
                attrs = {"infer_latent_samples_from_training": True}

        else:
            prediction = output.output.detach().cpu()

            if self.num_output_sampling is None:
                prediction = prediction.unsqueeze(0)

            num_output_dims = self.raw_module.model_config.NUM_OUTPUT_DIMS
            extra_dims_sorted = ["output_samples", "latent_samples"]
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
            attrs,
        )

        self._batch_counter += 1


@dataclasses.dataclass
class DeterministicPredictorConfig:
    _type: ClassVar[str] = "deterministic"

    def build(
        self,
        module: moduleABC,
        distributed: Distributed,
        output_dir: Path | str,
        num_output_sampling: int | None = None,
    ):

        return DetermninisticPredictor(
            self, module, distributed, output_dir, num_output_sampling
        )


class DetermninisticPredictor(PredictorABC):
    def __init__(
        self,
        config: cVAEPredictorConfig,
        module: moduleABC,
        distributed: Distributed,
        output_dir: Path | str,
        num_output_sampling: int | None = None,
    ):

        self.config = config
        self.module = module
        self.num_output_sampling = num_output_sampling
        self.num_output_covariance_sampling = None
        self.output_dir = Path(output_dir)

        if num_output_sampling <= 0:
            raise ValueError("num_output_sampling must be larger than 1.")

        if module.config.GENERATOR is None:
            self.num_output_covariance_sampling = num_output_sampling or 0

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
        return (self.num_output_covariance_sampling or 0) > 0

    @torch.no_grad()
    def _infer_on_batch(
        self,
        batch: BatchDataABC,
        _getting_train_stats: bool = False,
    ) -> deterministicOutput | dict[str, RunningCovariance]:

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

            output = self.raw_module.predict(
                data=batch, output_sample_size=self.num_output_sampling
            )

            if self.num_output_covariance_sampling > 0:
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

        prediction = output.output
        target = data.target
        residual = target - prediction
        residual = residual.reshape(residual.shape[0], -1)

        self.stats["residual"].update(residual)

        return self.stats

    def _batch_to_netcdf(
        self,
        output: cVAEOutput,
        metadata: list[dict],
    ):

        prediction = output.output.detach().cpu()

        if self.num_output_sampling is None:
            prediction = prediction.unsqueeze(0)

        num_output_dims = self.raw_module.config.NUM_OUTPUT_DIMS
        extra_dims_sorted = ["output_samples", "latent_samples"]
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

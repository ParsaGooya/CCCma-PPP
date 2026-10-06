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
from cccma_ppp.core.models.cvae import cVAEOutput
from cccma_ppp.inference.predictors.predictor_abc import (
    PredictorABC,
    save_batch_to_netcdf,
)
from cccma_ppp.data_modules.dataloader import BatchDataABC


@dataclasses.dataclass
class cVAEPredictorConfig:
    """
    Configuration for cVAE inference.

    Parameters
    ----------
    num_latent_samples : int
        Number of latent samples generated for each input sample.
    nstds : float
        Standard-deviation multiplier used when sampling latent values.
    chunk_size : int | None
        Maximum number of latent samples processed together during chunked
        generation. If ``None``, latent samples are processed together.
    infer_latent_samples_from_training : bool
        Whether latent samples are drawn from the mean and covariance estimated
        from posterior samples collected on the training data.
    save_latent : bool
        Whether to save latent variables instead of model predictions.
    """

    num_latent_samples: int
    nstds: float = 1.0
    chunk_size: int | None = None
    infer_latent_samples_from_training: bool = False
    save_latent: bool = False

    _type: ClassVar[str] = "cvae"

    def __post_init__(self) -> None:
        """
        Validate the cVAE predictor configuration.

        Raises
        ------
        ValueError
            If ``num_latent_samples`` is less than one or ``nstds`` is not positive.

        Warns
        -----
        UserWarning
            If ``save_latent`` is enabled, indicating that predictions will not be
            saved.
        """
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
        num_output_sampling: int = 1,
    ):
        """
        Build a cVAE predictor from this configuration.

        Parameters
        ----------
        module : moduleABC
            Trained cVAE module used for inference.
        distributed : Distributed
            Distributed runtime associated with inference.
        output_dir : Path | str
            Directory in which inference outputs and training statistics are stored.
        num_output_sampling : int
            Number of output samples generated for each latent sample.

        Returns
        -------
        cVAEPredictor
            Configured cVAE predictor.
        """
        return cVAEPredictor(self, module, distributed, output_dir, num_output_sampling)


class cVAEPredictor(PredictorABC):
    """
    Perform inference with a conditional variational autoencoder.

    The predictor supports prior or training-derived latent sampling, multiple
    latent and output samples, extraction of posterior and residual statistics,
    and saving predictions or latent variables to temporary NetCDF files.

    Parameters
    ----------
    config : cVAEPredictorConfig
        Configuration controlling latent sampling and inference behavior.
    module : moduleABC
        Trained cVAE module used for inference.
    distributed : Distributed
        Distributed runtime associated with inference.
    output_dir : Path | str
        Directory in which inference outputs and training statistics are stored.
    num_output_sampling : int
        Number of output samples generated for each latent sample.
    """

    def __init__(
        self,
        config: cVAEPredictorConfig,
        module: moduleABC,
        distributed: Distributed,
        output_dir: Path | str,
        num_output_sampling: int = 1,
    ):
        """
        Initialize a cVAE predictor.

        Parameters
        ----------
        config : cVAEPredictorConfig
            Configuration controlling latent sampling and inference behavior.
        module : moduleABC
            Trained cVAE module used for inference.
        distributed : Distributed
            Distributed runtime associated with inference.
        output_dir : Path | str
            Directory in which inference outputs and training statistics are stored.
        num_output_sampling : int
            Number of output samples generated for each latent sample.

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
        """
        Indicate whether training-derived statistics must be extracted.

        Returns
        -------
        bool
            ``True`` if posterior latent statistics or training residual statistics
            are required.
        """
        return any([self.extract_posterior_samples, self.extract_training_residuals])

    @property
    def extract_posterior_samples(self):
        """
        Indicate whether posterior latent samples must be collected from training data.

        Returns
        -------
        bool
            Whether latent sampling during inference is based on posterior statistics
            extracted from the training data.
        """
        return self.infer_latent_samples_from_training

    @property
    def extract_training_residuals(self):
        """
        Indicate whether training residual statistics must be collected.

        Returns
        -------
        bool
            ``True`` when multiple output covariance samples are required.
        """
        return self.num_output_covariance_sampling > 1

    @torch.no_grad()
    def _infer_on_batch(
        self,
        batch: BatchDataABC,
        _getting_train_stats: bool = False,
    ) -> cVAEOutput | dict[str, RunningCovariance]:
        """
        Perform cVAE inference or training-statistic extraction for one batch.

        Depending on the predictor configuration, the method can collect training
        statistics, save latent variables, sample latent values from training-derived
        statistics, generate predictions, and add decoder noise based on training
        residual covariance.

        Parameters
        ----------
        batch : BatchDataABC
            Batch containing model inputs, optional targets, and metadata.
        _getting_train_stats : bool
            Whether to collect training-derived latent or residual statistics instead
            of generating inference outputs.

        Returns
        -------
        cVAEOutput | dict[str, RunningCovariance]
            cVAE inference output, or running training-statistic accumulators when
            ``_getting_train_stats`` is enabled.

        Raises
        ------
        RuntimeError
            If a target is required for latent saving or training-statistic
            extraction but is absent from the batch.
        """
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
                output = self.raw_module.forward(data=batch, latent_sample_size=1)

            if _getting_train_stats:
                stats = self._update_train_stats(output, batch)
                return stats

            if self.save_latent:
                self._batch_to_netcdf(output, batch.metadata)
                return output

            if self.infer_latent_samples_from_training:
                latent_samples = self._get_latent_samples_based_on_train(data=batch)

            if self.num_output_covariance_sampling > 1:
                num_output_sampling = 1
            else:
                num_output_sampling = self.num_output_sampling

            output = self.raw_module.predict(
                data=batch,
                latent_sample_size=self.num_latent_samples,
                nstds=self.nstds,
                latent_samples=latent_samples,
                output_sample_size=num_output_sampling,
                chunk_size=self.config.chunk_size,
            )

            if self.num_output_covariance_sampling > 1:
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
        """
        Update training-derived statistics from a cVAE output.

        Posterior latent samples are accumulated when requested. Training residuals
        are accumulated when output covariance sampling requires an empirical
        residual distribution.

        Parameters
        ----------
        output : cVAEOutput
            cVAE output containing predictions and latent variables.
        data : BatchDataABC
            Training batch containing the corresponding target values.

        Returns
        -------
        dict[str, RunningCovariance]
            Updated running covariance accumulators.

        Raises
        ------
        RuntimeError
            If posterior sample statistics are requested but the cVAE output does not
            contain latent samples.
        """
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
        """
        Draw latent samples from training-derived posterior statistics.

        The latent sampler is initialized from saved training statistics on first use
        and then used to generate the requested number of samples for the current
        batch.

        Parameters
        ----------
        data : BatchDataABC
            Batch whose size determines the number of latent samples required.

        Returns
        -------
        torch.Tensor
            Latent samples on the predictor device.
        """
        if self.infer_latent_samples_from_training and self.latent_sampler is None:
            self.latent_sampler = self.build_latent_sampler()

        x = data.input
        batch_size = x.shape[0]

        sample_size = (self.num_latent_samples, batch_size)

        latent_samples = self.latent_sampler(sample_size, self.nstds)

        return latent_samples.to(self.device)

    def build_latent_sampler(self) -> Callable[..., torch.Tensor]:
        """
        Build a sampler from posterior statistics estimated on training data.

        The saved posterior mean and covariance are loaded and used to define a
        callable that draws latent samples with a configurable standard-deviation
        multiplier.

        Returns
        -------
        Callable[..., torch.Tensor]
            Callable that generates latent samples from the saved posterior
            distribution.

        Raises
        ------
        ValueError
            If the training-statistics file does not exist or does not contain the
            posterior sample mean and covariance required for cVAE sampling.
        """
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
            """
            Draw latent samples from the saved training distribution.

            Parameters
            ----------
            sample_size : int | tuple[int, ...]
                Number and shape of samples to draw.
            std : float
                Standard-deviation multiplier applied during sampling.

            Returns
            -------
            torch.Tensor
                Samples drawn using the saved posterior mean and covariance.
            """
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
        """
        Save a batch of predictions or latent variables to a temporary NetCDF file.

        When latent saving is enabled, available latent tensors are padded to a common
        latent dimension and combined along a channel dimension. Otherwise, generated
        predictions are prepared with their output- and latent-sample dimensions
        before being written.

        Parameters
        ----------
        output : cVAEOutput
            cVAE output containing predictions and latent variables to save.
        metadata : list[dict]
            Per-sample metadata used to construct output coordinates.

        Raises
        ------
        RuntimeError
            If latent saving is requested but no latent variables are available in
            ``output``.
        """
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
            attrs = {"source": "latent"}

            if self.infer_latent_samples_from_training:
                attrs["infer_latent_samples_from_training"] = True

        else:
            prediction = output.output

            if self.num_output_sampling == 1:
                prediction = prediction.unsqueeze(0)

            num_output_dims = self.raw_module.model_config.NUM_OUTPUT_DIMS
            extra_dims_sorted = ["output_samples", "latent_samples"]
            assign_coords = None
            attrs = {"source": "predictions", "nstds": self.nstds}

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

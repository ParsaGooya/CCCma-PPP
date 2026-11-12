import torch
from pathlib import Path
import dataclasses
import torch.nn.functional as F
from typing import Callable, ClassVar
import warnings

from S2D4P.core.trainer import clear_memory
from S2D4P.generic.distributed import Distributed
from S2D4P.generic.aggregator import RunningCovariance
from S2D4P.core.core_abc import moduleABC
from S2D4P.core.models.cvae import cVAEOutput
from S2D4P.inference.predictors.predictor_abc import (
    PredictorABC,
    save_batch_to_netcdf,
)
from S2D4P.data.dataloader import BatchDataABC


@dataclasses.dataclass
class cVAEPredictorConfig:
    """
    Configuration for conditional variational autoencoder inference.

    Parameters
    ----------
    num_latent_samples : int
        Number of latent samples generated for each input sample.
    nstds : float
        Standard-deviation multiplier used when sampling latent variables.
    chunk_size : int | None
        Number of latent samples processed per generation chunk. If ``None``, all
        latent samples are processed together.
    infer_latent_samples_from_training : bool
        Whether latent samples are drawn from a Gaussian distribution estimated
        from posterior samples collected on the training data.
    save_latent : bool
        Whether latent variables are saved instead of model predictions.
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
        Build a cVAE predictor.

        Parameters
        ----------
        module : moduleABC
            Trained cVAE module used for inference.
        distributed : Distributed
            Distributed runtime used for device placement and rank information.
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
    Run inference for a conditional variational autoencoder.

    Supports prior sampling, latent distributions estimated from training
    posteriors, output residual covariance sampling, latent-variable export, and
    temporary NetCDF prediction output.

    Parameters
    ----------
    config : cVAEPredictorConfig
        Configuration controlling latent sampling and output behavior.
    module : moduleABC
        Trained cVAE module used for inference.
    distributed : Distributed
        Distributed runtime used for device placement and rank information.
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
        Initialize the cVAE predictor.

        Configures latent and output sampling behavior and initializes running
        covariance accumulators when training-derived latent or residual statistics
        are required.

        Parameters
        ----------
        config : cVAEPredictorConfig
            Configuration controlling cVAE inference.
        module : moduleABC
            Trained cVAE module used for inference.
        distributed : Distributed
            Distributed runtime used for device placement and rank information.
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

        if module.architecture_config.GENERATOR is None:
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
            ``True`` when posterior latent statistics or training residual statistics
            are required.
        """
        return any([self.extract_posterior_samples, self.extract_training_residuals])

    @property
    def extract_posterior_samples(self):
        """
        Indicate whether posterior latent statistics must be extracted.

        Returns
        -------
        bool
            Whether inference uses a latent distribution estimated from training
            posterior samples.
        """
        return self.infer_latent_samples_from_training

    @property
    def extract_training_residuals(self):
        """
        Indicate whether training residual statistics must be extracted.

        Returns
        -------
        bool
            ``True`` when multiple output samples require residual covariance
            sampling.
        """
        return self.num_output_covariance_sampling > 1

    @torch.no_grad()
    def _infer_on_batch(
        self,
        batch: BatchDataABC,
        _getting_train_stats: bool = False,
    ) -> cVAEOutput | dict[str, RunningCovariance]:
        """
        Run cVAE inference or statistic extraction on one batch.

        When training statistics are requested, posterior variables are computed and
        used to update running covariance statistics. Latent-saving mode writes
        posterior variables instead of predictions. Otherwise, predictions are
        generated from either the configured prior or a training-derived latent
        distribution.

        Parameters
        ----------
        batch : BatchDataABC
            Batch containing conditioning inputs, optional targets, masks, metadata,
            and additional features.
        _getting_train_stats : bool
            Whether to update training-derived latent or residual statistics instead
            of generating inference output.

        Returns
        -------
        cVAEOutput | dict[str, RunningCovariance]
            cVAE output during prediction or latent saving, or updated running
            covariance statistics while extracting training statistics.

        Raises
        ------
        RuntimeError
            If a target is required for posterior-variable extraction but is absent
            from the batch.
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
        Update training-derived latent and residual statistics.

        Posterior latent samples are accumulated when training-based latent sampling
        is enabled. Prediction residuals are accumulated when output covariance
        sampling is required.

        Parameters
        ----------
        output : cVAEOutput
            cVAE output containing posterior samples and predictions.
        data : BatchDataABC
            Training batch containing the corresponding target values.

        Returns
        -------
        dict[str, RunningCovariance]
            Updated running covariance accumulators.

        Raises
        ------
        RuntimeError
            If posterior sample statistics are required but ``output.samples`` is
            unavailable.
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
        Sample latent vectors from training-derived posterior statistics.

        The latent sampler is constructed lazily from previously saved training
        statistics and reused for subsequent batches.

        Parameters
        ----------
        data : BatchDataABC
            Inference batch whose size determines the number of latent samples needed.

        Returns
        -------
        torch.Tensor
            Latent samples with leading latent-sample and batch dimensions, moved to
            the inference device.
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
        Build a sampler from training posterior statistics.

        The saved posterior mean and covariance matrix are loaded and used to create
        a callable that samples a multivariate latent distribution.

        Returns
        -------
        Callable[..., torch.Tensor]
            Callable accepting a sample shape and standard-deviation multiplier and
            returning latent samples.

        Raises
        ------
        ValueError
            If training statistics have not been saved or the stored statistics do
            not contain posterior sample mean and covariance values.
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
            Draw latent samples from the training-derived distribution.

            Parameters
            ----------
            sample_size : int | tuple[int, ...]
                Number or shape of latent samples to generate.
            std : float
                Standard-deviation multiplier applied during sampling.

            Returns
            -------
            torch.Tensor
                Samples drawn using the stored posterior mean and covariance matrix.
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
        Save batch predictions or latent variables to a temporary NetCDF file.

        In latent-saving mode, available posterior and conditional latent variables
        are padded to a common latent size and stored as channels. Otherwise, model
        predictions are stored with output-sample and latent-sample dimensions for
        later aggregation.

        Parameters
        ----------
        output : cVAEOutput
            cVAE output containing predictions or latent variables to save.
        metadata : list[dict]
            Per-sample coordinate metadata used to construct the NetCDF output.

        Raises
        ------
        RuntimeError
            If latent-saving mode is enabled but no latent variables are available.
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

            num_output_dims = self.raw_module.architecture_config.NUM_OUTPUT_DIMS
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

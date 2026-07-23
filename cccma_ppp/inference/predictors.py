import torch
from pathlib import Path
import dataclasses
import torch.nn.functional as F
from typing import Callable
import warnings

from cccma_ppp.core.trainer import clear_memory
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.aggregator import RunningCovariance
from cccma_ppp.core.selectors import PredictorSelector
from cccma_ppp.core.core_abc import moduleABC
from cccma_ppp.core.cVAE_module import cVAEOutput
from cccma_ppp.core.deterministic_module import deterministicOutput
from cccma_ppp.inference.predictor_abc import PredictorABC, save_batch_to_netcdf
from cccma_ppp.data_modules.dataloader import BatchDataABC


@PredictorSelector.register("cvae")
@dataclasses.dataclass
class cVAEPredictorConfig:
    """
    Configuration for a conditional variational autoencoder predictor.

    Parameters
    ----------
    num_latent_samples : int, optional
        Number of latent samples generated for each input sample.
    nstds : float, optional
        Standard-deviation scale applied when sampling latent variables.
    infer_latent_samples_from_training : bool, optional
        Whether to sample latent variables from statistics extracted from the
        training data.
    save_latent : bool, optional
        Whether to save latent variables instead of model predictions.

    """

    num_latent_samples: int = 1
    nstds: float = 1.0
    infer_latent_samples_from_training: bool = False
    save_latent: bool = False

    def __post_init__(self) -> None:
        """
        Validate the cVAE predictor configuration.

        Warns
        -----
        UserWarning
            If ``save_latent`` is enabled because predictions will not be saved.

        Raises
        ------
        ValueError
            If ``num_latent_samples`` is less than one.
        ValueError
            If ``nstds`` is not positive.

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
        num_output_covariance_sampling: int = 0,
    ):
        """
        Construct a cVAE predictor.

        Parameters
        ----------
        module : moduleABC
            Trained cVAE module used to generate predictions.
        distributed : Distributed
            Distributed execution context.
        output_dir : pathlib.Path or str
            Directory in which predictions and training statistics are stored.
        num_output_covariance_sampling : int, optional
            Number of decoder-noise samples generated from the residual covariance.

        Returns
        -------
        cVAEPredictor
            Configured cVAE predictor.

        """

        return cVAEPredictor(
            self, module, distributed, output_dir, num_output_covariance_sampling
        )


class cVAEPredictor(PredictorABC):
    """
    Generate and save predictions from a conditional variational autoencoder.

    Parameters
    ----------
    config : cVAEPredictorConfig
        Predictor configuration.
    module : moduleABC
        Trained cVAE module used for inference.
    distributed : Distributed
        Distributed execution context.
    output_dir : pathlib.Path or str
        Directory in which predictions and statistics are stored.
    num_output_covariance_sampling : int, optional
        Number of decoder-noise samples generated from the residual covariance.

    """

    def __init__(
        self,
        config: cVAEPredictorConfig,
        module: moduleABC,
        distributed: Distributed,
        output_dir: Path | str,
        num_output_covariance_sampling: int = 0,
    ):
        """
        Initialize the cVAE predictor.

        Parameters
        ----------
        config : cVAEPredictorConfig
            Predictor configuration.
        module : moduleABC
            Trained cVAE module used for inference.
        distributed : Distributed
            Distributed execution context.
        output_dir : pathlib.Path or str
            Directory in which predictions and statistics are stored.
        num_output_covariance_sampling : int, optional
            Number of decoder-noise samples generated from the residual covariance.

        """

        self.config = config
        self.module = module
        self.num_output_covariance_sampling = num_output_covariance_sampling
        self.output_dir = Path(output_dir)

        self.num_latent_samples = config.num_latent_samples
        self.infer_latent_samples_from_training = (
            config.infer_latent_samples_from_training
        )
        self.nstds = config.nstds
        self.save_latent = config.save_latent

        self.distributed = distributed
        self.device = distributed.device

        if self.extract_training_vars:
            self._stats = {
                "samples": RunningCovariance(self.distributed),
                "residual": RunningCovariance(self.distributed),
            }

        self.latent_sampler = None
        self.output_sampler = None
        self._batch_counter = 0

    @property
    def extract_training_vars(self):
        """
        Indicate whether training-variable statistics must be extracted.

        Returns
        -------
        bool
            ``True`` when latent sampling from training statistics or output
            covariance sampling is enabled.

        """
        return any(
            [
                self.infer_latent_samples_from_training,
                self.num_output_covariance_sampling > 0,
            ]
        )

    @torch.no_grad()
    def _infer_on_batch(
        self,
        batch: BatchDataABC,
        _getting_train_stats: bool = False,
    ) -> cVAEOutput | dict[str, RunningCovariance]:
        """
        Perform cVAE inference on a batch.

        Parameters
        ----------
        batch : BatchDataABC
            Batch containing model inputs, targets, masks, metadata, and optional
            additional features.
        _getting_train_stats : bool, optional
            Whether to update and return training-variable statistics instead of
            saving predictions.

        Returns
        -------
        cVAEOutput or dict of str to RunningCovariance
            Model output during inference or updated running statistics during
            training-statistics extraction.

        Raises
        ------
        RuntimeError
            If targets are unavailable when posterior variables or training
            statistics are requested.
        ValueError
            If required saved training statistics are unavailable or invalid.

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
        """
        Update latent-sample and residual statistics.

        Parameters
        ----------
        output : cVAEOutput
            Model output containing posterior samples and predictions.
        data : BatchDataABC
            Batch containing the target values associated with the predictions.

        Returns
        -------
        dict of str to RunningCovariance
            Updated latent-sample and residual covariance accumulators.

        Raises
        ------
        RuntimeError
            If the model output does not contain posterior samples.

        """

        if output.samples is None:
            raise RuntimeError(
                "cVAEOutput.samples is required for training latent stats."
            )

        samples = output.samples.reshape(-1, output.samples.shape[-1])
        self.stats["samples"].update(samples)

        prediction = output.output
        target = data.target
        residual = target - prediction[0]
        residual = residual.reshape(residual.shape[0], -1)

        self.stats["residual"].update(residual)

        return self.stats

    def _get_latent_samples_based_on_train(
        self,
        data: BatchDataABC,
    ):
        """
        Sample latent variables from training-data statistics.

        Parameters
        ----------
        data : BatchDataABC
            Batch whose size determines the latent-sample shape.

        Returns
        -------
        torch.Tensor
            Latent samples on the predictor device.

        Raises
        ------
        ValueError
            If the required training statistics are unavailable or invalid.

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
        Construct a latent sampler from saved training statistics.

        Returns
        -------
        callable
            Function that samples latent variables from the saved training mean
            and covariance.

        Raises
        ------
        ValueError
            If the training-statistics file does not exist.
        ValueError
            If the saved statistics do not contain the cVAE latent mean and
            covariance.

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

        def _sampler(
            sample_size: int | tuple[int, ...],
            std: float,
        ):
            """
            Sample latent variables from the training distribution.

            Parameters
            ----------
            sample_size : int or tuple of int
                Number or shape of latent samples to generate.
            std : float
                Standard-deviation scale applied to the latent distribution.

            Returns
            -------
            torch.Tensor
                Sampled latent variables.

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
        Save cVAE predictions or latent variables as a NetCDF file.

        Parameters
        ----------
        output : cVAEOutput
            Model output containing predictions and optional latent variables.
        metadata : list of dict
            Coordinate metadata associated with each sample in the batch.

        Raises
        ------
        RuntimeError
            If latent-variable saving is enabled but no latent variables are
            available.

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
            save_name = (
                f"latent_rank{self.distributed.rank}_{self._batch_counter:08d}.nc"
            )

            if self.infer_latent_samples_from_training:
                attrs = {"infer_latent_samples_from_training": True}

        else:
            prediction = output.output.detach().cpu()

            if self.num_output_covariance_sampling == 0:
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


@PredictorSelector.register("default")
@PredictorSelector.register("deterministic")
@dataclasses.dataclass
class DeterministicPredictorConfig:
    """
    Configuration for a deterministic predictor.

    """

    def build(
        self,
        module: moduleABC,
        distributed: Distributed,
        output_dir: Path | str,
        num_output_covariance_sampling: int = 0,
    ):
        """
        Construct a deterministic predictor.

        Parameters
        ----------
        module : moduleABC
            Trained deterministic module used to generate predictions.
        distributed : Distributed
            Distributed execution context.
        output_dir : pathlib.Path or str
            Directory in which predictions and training statistics are stored.
        num_output_covariance_sampling : int, optional
            Number of output-noise samples generated from the residual covariance.

        Returns
        -------
        DetermninisticPredictor
            Configured deterministic predictor.

        """

        return DetermninisticPredictor(
            self, module, distributed, output_dir, num_output_covariance_sampling
        )


class DetermninisticPredictor(PredictorABC):
    """
    Generate and save predictions from a deterministic model.

    Parameters
    ----------
    config : DeterministicPredictorConfig
        Predictor configuration.
    module : moduleABC
        Trained deterministic module used for inference.
    distributed : Distributed
        Distributed execution context.
    output_dir : pathlib.Path or str
        Directory in which predictions and statistics are stored.
    num_output_covariance_sampling : int, optional
        Number of output-noise samples generated from the residual covariance.

    """

    def __init__(
        self,
        config: DeterministicPredictorConfig,
        module: moduleABC,
        distributed: Distributed,
        output_dir: Path | str,
        num_output_covariance_sampling: int = 0,
    ):
        """
        Initialize the deterministic predictor.

        Parameters
        ----------
        config : DeterministicPredictorConfig
            Predictor configuration.
        module : moduleABC
            Trained deterministic module used for inference.
        distributed : Distributed
            Distributed execution context.
        output_dir : pathlib.Path or str
            Directory in which predictions and statistics are stored.
        num_output_covariance_sampling : int, optional
            Number of output-noise samples generated from the residual covariance.

        """

        self.config = config
        self.module = module
        self.num_output_covariance_sampling = num_output_covariance_sampling
        self.output_dir = Path(output_dir)

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
        Indicate whether residual statistics must be extracted.

        Returns
        -------
        bool
            ``True`` when output covariance sampling is enabled.

        """
        return self.num_output_covariance_sampling > 0

    @torch.no_grad()
    def _infer_on_batch(
        self,
        batch: BatchDataABC,
        _getting_train_stats: bool = False,
    ) -> deterministicOutput | dict[str, RunningCovariance]:
        """
        Perform deterministic inference on a batch.

        Parameters
        ----------
        batch : BatchDataABC
            Batch containing model inputs, targets, metadata, and optional
            additional features.
        _getting_train_stats : bool, optional
            Whether to update and return residual statistics instead of saving
            predictions.

        Returns
        -------
        deterministicOutput or dict of str to RunningCovariance
            Model output during inference or updated residual statistics during
            training-statistics extraction.

        Raises
        ------
        RuntimeError
            If targets are unavailable when training statistics are requested.
        ValueError
            If required saved training statistics are unavailable.

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

            output = self.raw_module.predict(data=batch)

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
        """
        Update residual statistics from deterministic predictions.

        Parameters
        ----------
        output : deterministicOutput
            Deterministic model predictions.
        data : BatchDataABC
            Batch containing the corresponding target values.

        Returns
        -------
        dict of str to RunningCovariance
            Updated residual covariance accumulator.

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
        Save deterministic predictions as a NetCDF file.

        Parameters
        ----------
        output : deterministicOutput
            Deterministic model predictions.
        metadata : list of dict
            Coordinate metadata associated with each sample in the batch.

        """

        prediction = output.output.detach().cpu()

        if self.num_output_covariance_sampling == 0:
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

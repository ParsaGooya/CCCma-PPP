import abc
import torch
from typing import final
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Callable

from cccma_ppp.data_modules.dataloader import BatchDataABC
from cccma_ppp.core.core_abc import OutputABC
from cccma_ppp.generic.aggregator import RunningCovariance


class PredictorABC(abc.ABC):
    """
    Abstract base class for inference predictors.

    Provides shared functionality for saving predictions, collecting
    training-derived statistics, sampling output noise, and accessing the
    underlying model module.

    Attributes
    ----------
    output_dir : str | Path
        Directory in which prediction outputs and training statistics are stored.
    output_sampler : Callable[..., torch.Tensor] | None
        Sampler used to generate decoder noise from training residual statistics.
    num_output_covariance_sampling : int | None
        Number of output samples generated from the empirical residual
        distribution.
    """

    output_dir: str | Path
    output_sampler: Callable[..., torch.Tensor] | None
    num_output_covariance_sampling: int | None

    @property
    def temp_save_dir(self):
        """
        Return the directory used for temporary prediction files.

        Returns
        -------
        Path
            Temporary output directory located under ``output_dir``.
        """
        return Path(self.output_dir) / "_temp"

    @property
    @abc.abstractmethod
    def extract_training_vars(self) -> bool:
        """
        Indicate whether training-derived statistics must be extracted.

        Returns
        -------
        bool
            Whether the predictor requires statistics calculated from training data.
        """
        pass

    @property
    def stats(self) -> dict[str, RunningCovariance]:
        """
        Return the training-statistic accumulators.

        Returns
        -------
        dict[str, RunningCovariance] | None
            Mapping from statistic names to running covariance accumulators when
            training statistics are required, otherwise ``None``.
        """
        if self.extract_training_vars:
            return self._stats

    @abc.abstractmethod
    def _infer_on_batch(self, batch: BatchDataABC, _getting_train_stats: bool = False):
        """
        Perform inference or training-statistic extraction for a batch.

        Parameters
        ----------
        batch : BatchDataABC
            Batch containing model inputs, optional targets, and metadata.
        _getting_train_stats : bool
            Whether to collect training-derived statistics instead of performing
            standard inference.

        Returns
        -------
        Any
            Predictor-specific inference output or training statistics.
        """
        pass

    @abc.abstractmethod
    def _batch_to_netcdf(self, output: OutputABC, metadata: list[dict]):
        """
        Save a batch of predictor outputs to a temporary NetCDF file.

        Parameters
        ----------
        output : OutputABC
            Model output to save.
        metadata : list[dict]
            Per-sample metadata used to construct output coordinates.
        """
        pass

    @abc.abstractmethod
    def _update_train_stats(self, output: OutputABC, batch: BatchDataABC):
        """
        Update training-derived statistics from a model output.

        Parameters
        ----------
        output : OutputABC
            Model output used to calculate the required statistics.
        batch : BatchDataABC
            Batch containing the corresponding model inputs and targets.

        Returns
        -------
        dict[str, RunningCovariance]
            Updated running covariance accumulators.
        """
        pass

    @final
    @property
    def raw_module(self):
        """
        Return the underlying unwrapped model module.

        Returns
        -------
        Any
            Module wrapped by distributed data parallel execution when applicable,
            or the original module otherwise.
        """
        if isinstance(self.module, torch.nn.parallel.DistributedDataParallel):
            return self.module.module
        return self.module

    @final
    def add_decoder_noise(
        self,
        output: OutputABC,
        num_output_samples: int,
        sample_size: tuple,
        reshape_size: tuple,
    ) -> OutputABC:
        """
        Add covariance-based noise to model predictions.

        Noise is sampled from the empirical residual distribution estimated from
        training data and added as an additional output-sample dimension.

        Parameters
        ----------
        output : OutputABC
            Model output whose predictions receive sampled decoder noise.
        num_output_samples : int
            Number of noisy output samples to generate.
        sample_size : tuple
            Leading sample dimensions used when drawing residual noise.
        reshape_size : tuple
            Shape of each predicted sample used to reshape the sampled noise.

        Returns
        -------
        OutputABC
            Model output with covariance-based decoder noise added to its predictions.
        """
        if self.output_sampler is None:
            self.output_sampler = self.build_output_sampler()

        prediction = output.output

        noise = self.output_sampler((num_output_samples, *sample_size))
        noise = noise.reshape(num_output_samples, *sample_size, *reshape_size).to(
            device=prediction.device,
            dtype=prediction.dtype,
        )

        prediction = prediction.unsqueeze(0) + noise
        output.output = prediction
        return output

    @final
    def build_output_sampler(self) -> Callable[..., torch.Tensor]:
        """
        Build a decoder-noise sampler from training residual statistics.

        The residual covariance is loaded from saved training statistics and combined
        with a zero-mean vector to define the sampling distribution.

        Returns
        -------
        Callable[..., torch.Tensor]
            Callable that generates samples from the empirical residual distribution.

        Raises
        ------
        ValueError
            If the training-statistics file does not exist.
        """
        stats_path = self.output_dir / "training_variable_stats.pt"

        if not stats_path.exists():
            raise ValueError(
                "Training statistics based on the trained model must be saved "
                "to disk first."
            )

        stats = torch.load(stats_path, map_location=self.device)

        def _sampler(sample_size: int | tuple[int, ...]):
            """
            Draw decoder-noise samples from the training residual distribution.

            Parameters
            ----------
            sample_size : int | tuple[int, ...]
                Number and shape of residual samples to draw.

            Returns
            -------
            torch.Tensor
                Samples drawn from the zero-mean residual distribution.
            """
            return self._sample(
                torch.zeros_like(stats["residual_mean"]),
                stats["residual_cov"],
                sample_size,
            )

        return _sampler

    @final
    def _get_multinormal(
        self,
        mu: torch.Tensor,
        cov: torch.Tensor,
        std: float = 1.0,
    ) -> torch.distributions.MultivariateNormal:
        """
        Construct a multivariate normal distribution.

        The covariance matrix is scaled by ``std`` and progressively increasing
        diagonal jitter is added when necessary to obtain a valid covariance matrix.

        Parameters
        ----------
        mu : torch.Tensor
            Mean vector of the multivariate normal distribution.
        cov : torch.Tensor
            Covariance matrix of the multivariate normal distribution.
        std : float
            Standard-deviation multiplier applied to the covariance.

        Returns
        -------
        torch.distributions.MultivariateNormal
            Multivariate normal distribution on the predictor device.

        Raises
        ------
        RuntimeError
            If a valid multivariate normal distribution cannot be constructed after
            repeated increases in diagonal jitter.
        ValueError
            If ``std`` is not positive.
        """
        mu = mu.detach().to(self.device).float()
        cov = cov.detach().to(self.device).float()

        if std <= 0:
            raise ValueError(f"std must be positive, got {std}.")

        cov = cov * (std**2)

        jitter = 1e-6
        eye = torch.eye(cov.shape[-1], device=self.device, dtype=cov.dtype)

        for _ in range(5):
            try:
                return torch.distributions.MultivariateNormal(
                    loc=mu,
                    covariance_matrix=cov + jitter * eye,
                )

            except ValueError:
                jitter *= 10

        raise RuntimeError(
            "Could not construct MultivariateNormal from covariance. "
            "Covariance may be singular or not positive definite."
        )

    @final
    def _sample(
        self,
        mu: torch.Tensor,
        cov: torch.Tensor,
        sample_size: int | tuple[int, ...] = 1,
        std: float = 1.0,
    ) -> torch.Tensor:
        """
        Draw samples from a multivariate normal distribution.

        Parameters
        ----------
        mu : torch.Tensor
            Mean vector of the distribution.
        cov : torch.Tensor
            Covariance matrix of the distribution.
        sample_size : int | tuple[int, ...]
            Number and shape of samples to draw.
        std : float
            Standard-deviation multiplier applied to the distribution.

        Returns
        -------
        torch.Tensor
            Sampled values on the predictor device.
        """
        dist = self._get_multinormal(mu=mu, cov=cov, std=std)

        if isinstance(sample_size, int):
            sample_size = (sample_size,)

        samples = dist.sample(sample_size)
        return samples.to(self.device)


def save_batch_to_netcdf(
    prediction: torch.Tensor,
    metadata: list[dict],
    num_output_dims: int,
    save_name: str,
    save_dir: str | Path,
    extra_dims_sorted: list[str] | None = None,
    assign_coords: dict = None,
    attrs: dict = None,
):
    """
    Save a batch of model predictions to a NetCDF file.

    Predictions are converted to an xarray DataArray with optional leading sample
    dimensions. Per-sample metadata are represented as a multi-index, which is
    unstacked into separate output dimensions before saving.

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction tensor to save.
    metadata : list[dict]
        Per-sample metadata used to construct batch coordinates.
    num_output_dims : int
        Number of dimensions in each model output, including the channel
        dimension.
    save_name : str
        Name of the NetCDF file.
    save_dir : str | Path
        Directory in which the NetCDF file is saved.
    extra_dims_sorted : list[str] | None
        Names of additional leading sample dimensions in ``prediction``.
    assign_coords : dict | None
        Additional coordinates to assign to the resulting DataArray.
    attrs : dict | None
        Attributes assigned to the resulting DataArray.

    Raises
    ------
    ValueError
        If the prediction does not have the expected number of dimensions or the
        number of metadata entries does not match the prediction batch size.
    """
    prediction = prediction.detach().float().cpu()

    if extra_dims_sorted is None:
        extra_dims_sorted = []

    coords = {}
    for i, dim in enumerate(extra_dims_sorted):
        coords[dim] = np.arange(1, prediction.shape[i] + 1)

    batch_size = prediction.shape[len(extra_dims_sorted)]
    channel_size = prediction.shape[len(extra_dims_sorted) + 1]
    spatial_shape = prediction.shape[len(extra_dims_sorted) + 2 :]

    expected_dims = num_output_dims + 1 + len(extra_dims_sorted)

    if prediction.ndim != expected_dims:
        raise ValueError(
            f"Expected prediction with {expected_dims} dimensions, got {prediction.shape}."
        )

    if len(metadata) != batch_size:
        raise ValueError(
            f"metadata length ({len(metadata)}) does not match batch size ({batch_size})."
        )

    dims = (
        extra_dims_sorted
        + ["batch", "channels"]
        + [f"output_dim_{i}" for i in range(num_output_dims - 1)]
    )

    index_keys = list(metadata[0].keys())

    batch_index = pd.MultiIndex.from_tuples(
        [tuple(m[k] for k in index_keys) for m in metadata],
        names=index_keys,
    )

    coords["batch"] = batch_index
    coords["channels"] = np.arange(1, channel_size + 1)

    output_keys = list()

    for i, size in enumerate(spatial_shape):
        coords[f"output_dim_{i}"] = np.arange(size)
        output_keys.append(f"output_dim_{i}")

    da = xr.DataArray(
        prediction.numpy(),
        dims=dims,
        coords=coords,
        name="prediction",
    )

    if assign_coords is not None:
        da = da.assign_coords(assign_coords)

    save_path = Path(save_dir) / save_name

    if attrs is not None:
        da.attrs = attrs

    da = da.unstack("batch").transpose(
        *extra_dims_sorted, *index_keys, ..., *output_keys
    )

    da.to_netcdf(save_path)

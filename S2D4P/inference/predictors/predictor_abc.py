import abc
import torch
from typing import final
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Callable

from S2D4P.data.dataloader import BatchDataABC
from S2D4P.core.core_abc import OutputABC
from S2D4P.generic.aggregator import RunningCovariance


class PredictorABC(abc.ABC):
    """
    Abstract base class for inference predictors.

    Defines shared functionality for training-statistic extraction, covariance-
    based output sampling, batch inference, and temporary NetCDF output.

    Attributes
    ----------
    output_dir : str | Path
        Directory in which inference outputs and training statistics are stored.
    output_sampler : Callable[..., torch.Tensor] | None
        Sampler constructed from training residual statistics for generating
        stochastic output perturbations.
    num_output_covariance_sampling : int | None
        Number of output samples generated from the training residual covariance.
    """

    output_dir: str | Path
    output_sampler: Callable[..., torch.Tensor] | None
    num_output_covariance_sampling: int | None

    @property
    def temp_save_dir(self):
        """
        Return the directory used for temporary batch predictions.

        Returns
        -------
        Path
            ``_temp`` directory under the configured output directory.
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
        Return training-statistic accumulators.

        Returns
        -------
        dict[str, RunningCovariance] | None
            Running covariance accumulators required by the predictor, or ``None``
            when no training-derived statistics are required.
        """
        if self.extract_training_vars:
            return self._stats

    @abc.abstractmethod
    def _infer_on_batch(self, batch: BatchDataABC, _getting_train_stats: bool = False):
        """
        Run inference or training-statistic extraction on one batch.

        Parameters
        ----------
        batch : BatchDataABC
            Batch containing model inputs, targets, masks, metadata, and optional
            additional features.
        _getting_train_stats : bool
            Whether to extract training-derived statistics instead of performing
            normal inference.

        Returns
        -------
        Any
            Predictor-specific model output or updated training statistics.
        """
        pass

    @abc.abstractmethod
    def _batch_to_netcdf(self, output: OutputABC, metadata: list[dict]):
        """
        Save predictor output for one batch to a temporary NetCDF file.

        Parameters
        ----------
        output : OutputABC
            Model output to save.
        metadata : list[dict]
            Per-sample coordinate metadata used to construct the NetCDF output.
        """
        pass

    @abc.abstractmethod
    def _update_train_stats(self, output: OutputABC, batch: BatchDataABC):
        """
        Update training-derived statistics from a model output.

        Parameters
        ----------
        output : OutputABC
            Model output used to calculate training statistics.
        batch : BatchDataABC
            Batch containing the corresponding inputs and targets.

        Returns
        -------
        Any
            Predictor-specific updated training statistics.
        """
        pass

    @final
    @property
    def raw_module(self):
        """
        Return the underlying unwrapped model module.

        Returns
        -------
        moduleABC
            Module wrapped by distributed data parallel execution when applicable, or
            the original module otherwise.
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
        Add covariance-sampled noise to decoder predictions.

        A residual sampler is built lazily from saved training statistics. Independent
        noise samples are reshaped to the model output dimensions and added to the
        prediction tensor.

        Parameters
        ----------
        output : OutputABC
            Model output whose prediction tensor is perturbed.
        num_output_samples : int
            Number of noisy output samples to generate.
        sample_size : tuple
            Leading sample dimensions used when drawing residual noise.
        reshape_size : tuple
            Output dimensions into which each sampled residual vector is reshaped.

        Returns
        -------
        OutputABC
            Model output with its prediction tensor replaced by the sampled output
            ensemble.
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
        Build an output-noise sampler from training residual statistics.

        The saved residual covariance is used to construct a zero-mean multivariate
        distribution for stochastic perturbation of model outputs.

        Returns
        -------
        Callable[..., torch.Tensor]
            Callable accepting a sample shape and returning residual noise samples.

        Raises
        ------
        ValueError
            If training-derived statistics have not been saved.
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
            Draw output noise from the training residual distribution.

            Parameters
            ----------
            sample_size : int | tuple[int, ...]
                Number or shape of residual samples to generate.

            Returns
            -------
            torch.Tensor
                Zero-mean samples drawn from the stored residual covariance.
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
        diagonal jitter is added when necessary to obtain a valid positive-definite
        covariance matrix.

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
        ValueError
            If ``std`` is not positive.
        RuntimeError
            If a valid multivariate normal distribution cannot be constructed after
            adding numerical jitter to the covariance matrix.
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
            Number or shape of samples to draw.
        std : float
            Standard-deviation multiplier applied to the covariance.

        Returns
        -------
        torch.Tensor
            Samples drawn from the distribution and placed on the predictor device.
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
    Save a batch of model predictions to NetCDF.

    Prediction tensors are converted to an xarray DataArray with explicit sample,
    channel, and output dimensions. Per-sample metadata are represented as a
    MultiIndex and unstacked into coordinate dimensions before saving.

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction tensor containing optional leading sample dimensions followed
        by batch, channel, and model output dimensions.
    metadata : list[dict]
        Per-sample coordinate metadata. Each dictionary must contain the same
        coordinate fields.
    num_output_dims : int
        Number of model output dimensions, including the channel dimension.
    save_name : str
        Filename of the NetCDF output.
    save_dir : str | Path
        Directory in which the NetCDF file is written.
    extra_dims_sorted : list[str] | None
        Names assigned to leading prediction dimensions, such as
        ``"output_samples"`` or ``"latent_samples"``.
    assign_coords : dict | None
        Additional coordinates assigned to the resulting DataArray.
    attrs : dict | None
        Attributes attached to the resulting DataArray.

    Raises
    ------
    ValueError
        If the prediction tensor has an unexpected number of dimensions or the
        metadata length does not match the prediction batch size.
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

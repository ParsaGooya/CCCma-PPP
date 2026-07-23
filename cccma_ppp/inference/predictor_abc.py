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
    Abstract base class for model predictors.

    Attributes
    ----------
    output_dir : str or pathlib.Path
        Directory in which prediction outputs and training statistics are
        stored.
    output_sampler : callable or None
        Callable used to sample decoder noise from training statistics.

    """

    output_dir: str | Path
    output_sampler: Callable[..., torch.Tensor] | None

    @property
    def temp_save_dir(self):
        """
        Return the temporary prediction directory.

        Returns
        -------
        pathlib.Path
            Temporary directory within the configured output directory.

        """
        return Path(self.output_dir) / "_temp"

    @property
    @abc.abstractmethod
    def extract_training_vars(self) -> bool:
        """
        Indicate whether training-variable statistics should be extracted.

        Returns
        -------
        bool
            ``True`` if training-variable statistics should be collected,
            otherwise ``False``.

        """
        pass

    @property
    def stats(self) -> dict[str, RunningCovariance]:
        """
        Return the running training-variable statistics.

        Returns
        -------
        dict of str to RunningCovariance or None
            Running covariance accumulators when training-variable extraction is
            enabled, otherwise ``None``.

        """
        if self.extract_training_vars:
            return self._stats

    @abc.abstractmethod
    def _infer_on_batch(
        self,
        batch: BatchDataABC,
        _getting_train_stats: bool = False,
    ):
        """
        Perform inference on a batch.

        Parameters
        ----------
        batch : BatchDataABC
            Batch on which inference is performed.
        _getting_train_stats : bool, optional
            Whether inference is being performed to collect training-variable
            statistics.

        Returns
        -------
        object
            Inference result. The concrete return type depends on the predictor
            implementation.

        """
        pass

    @abc.abstractmethod
    def _batch_to_netcdf(
        self,
        output: OutputABC,
        metadata: list[dict],
    ):
        """
        Save a batch of model outputs to a NetCDF file.

        Parameters
        ----------
        output : OutputABC
            Model output to save.
        metadata : list of dict
            Coordinate metadata associated with each sample in the batch.

        """
        pass

    @abc.abstractmethod
    def _update_train_stats(
        self,
        output: OutputABC,
        batch: BatchDataABC,
    ):
        """
        Update training-variable statistics from a batch.

        Parameters
        ----------
        output : OutputABC
            Model output used to update the statistics.
        batch : BatchDataABC
            Input batch associated with the model output.

        """
        pass

    @final
    @property
    def raw_module(self):
        """
        Return the underlying model module.

        Returns
        -------
        moduleABC
            Module wrapped by distributed data parallelism, or the original module
            when distributed data parallelism is not used.

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
        Add sampled decoder noise to a model output.

        Parameters
        ----------
        output : OutputABC
            Model output to which decoder noise is added.
        num_output_samples : int
            Number of noisy output samples to generate.
        sample_size : tuple
            Shape passed to the configured output sampler.
        reshape_size : tuple
            Additional shape used to align sampled noise with the prediction.

        Returns
        -------
        OutputABC
            Model output containing the noise-perturbed predictions.

        Raises
        ------
        ValueError
            If the required training statistics are unavailable when constructing
            the output sampler.

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
        Construct a sampler from stored residual statistics.

        Returns
        -------
        callable
            Callable that samples zero-mean multivariate-normal noise using the
            stored residual covariance matrix.

        Raises
        ------
        ValueError
            If the training-variable statistics file does not exist.

        """
        stats_path = self.output_dir / "training_variable_stats.pt"

        if not stats_path.exists():
            raise ValueError(
                "Training statistics based on the trained model must be saved "
                "to disk first."
            )

        stats = torch.load(stats_path, map_location=self.device)

        def _sampler(
            sample_size: int | tuple[int, ...],
        ):
            """
            Sample noise from the residual covariance distribution.

            Parameters
            ----------
            sample_size : int or tuple of int
                Number or shape of samples to generate.

            Returns
            -------
            torch.Tensor
                Sampled residual noise.

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
        Construct a numerically stable multivariate-normal distribution.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the multivariate-normal distribution.
        cov : torch.Tensor
            Covariance matrix of the multivariate-normal distribution.
        std : float, optional
            Scale applied to the distribution's standard deviation.

        Returns
        -------
        torch.distributions.MultivariateNormal
            Multivariate-normal distribution on the predictor device.

        Raises
        ------
        ValueError
            If ``std`` is not positive.
        RuntimeError
            If a valid multivariate-normal distribution cannot be constructed
            after adding progressively increasing diagonal jitter.

        """
        mu = mu.detach().to(self.device).float()
        cov = cov.detach().to(self.device).float()

        if std <= 0:
            raise ValueError(f"std must be positive, got {std}.")

        # Widen/narrow spread. Scaling std by k means covariance scales by k**2.
        cov = cov * (std**2)

        # Numerical stability for nearly-singular covariance matrices.
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
        Sample from a multivariate-normal distribution.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the multivariate-normal distribution.
        cov : torch.Tensor
            Covariance matrix of the multivariate-normal distribution.
        sample_size : int or tuple of int, optional
            Number or shape of samples to generate.
        std : float, optional
            Scale applied to the distribution's standard deviation.

        Returns
        -------
        torch.Tensor
            Samples generated on the predictor device.

        Raises
        ------
        ValueError
            If ``std`` is not positive.
        RuntimeError
            If the multivariate-normal distribution cannot be constructed.

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
    Save a batch of predictions as a NetCDF file.

    The batch dimension is converted to a multi-index using the supplied
    metadata and subsequently unstacked into individual sample dimensions.

    Parameters
    ----------
    prediction : torch.Tensor
        Prediction tensor containing optional extra dimensions, followed by
        batch, channel, and output dimensions. The tensor must reside on the
        CPU before conversion to NumPy.
    metadata : list of dict
        Coordinate metadata for each sample in the batch.
    num_output_dims : int
        Number of spatial or output dimensions in the prediction tensor.
    save_name : str
        Name of the NetCDF file.
    save_dir : str or pathlib.Path
        Directory in which the NetCDF file is saved.
    extra_dims_sorted : list of str or None, optional
        Names of additional leading prediction dimensions, in tensor order.
    assign_coords : dict or None, optional
        Additional coordinates assigned to the prediction data array.
    attrs : dict or None, optional
        Attributes attached to the prediction data array.

    Raises
    ------
    ValueError
        If the prediction tensor does not have the expected number of
        dimensions.
    ValueError
        If the number of metadata entries does not match the batch size.

    """

    if extra_dims_sorted is None:
        extra_dims_sorted = []

    coords = {}
    for i, dim in enumerate(extra_dims_sorted):
        coords[dim] = np.arange(1, prediction.shape[i] + 1)

    batch_size = prediction.shape[len(extra_dims_sorted)]
    channel_size = prediction.shape[len(extra_dims_sorted) + 1]
    spatial_shape = prediction.shape[len(extra_dims_sorted) + 2 :]

    expected_dims = num_output_dims + 2 + len(extra_dims_sorted)

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
        + [f"output_dim_{i}" for i in range(num_output_dims)]
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

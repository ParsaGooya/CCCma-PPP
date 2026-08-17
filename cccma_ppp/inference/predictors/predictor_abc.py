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
    Document this class.
    
    Attributes
    ----------
    output_dir : str | Path
        Description not yet provided.
    output_sampler : Callable[..., torch.Tensor] | None
        Description not yet provided.
    num_output_covariance_sampling : int | None
        Description not yet provided.
    """
    output_dir: str | Path
    output_sampler: Callable[..., torch.Tensor] | None
    num_output_covariance_sampling: int | None

    @property
    def temp_save_dir(self):
        """
        Document this function.
        
        Returns
        -------
        Any
            Description not yet provided.
        """
        return Path(self.output_dir) / "_temp"

    @property
    @abc.abstractmethod
    def extract_training_vars(self) -> bool:
        """
        Document this function.
        """
        pass

    @property
    def stats(self) -> dict[str, RunningCovariance]:
        """
        Document this function.
        
        Returns
        -------
        dict[str, RunningCovariance]
            Description not yet provided.
        """
        if self.extract_training_vars:
            return self._stats

    @abc.abstractmethod
    def _infer_on_batch(self, batch: BatchDataABC, _getting_train_stats: bool = False):
        """
        Document this function.
        
        Parameters
        ----------
        batch : BatchDataABC
            Description not yet provided.
        _getting_train_stats : bool
            Description not yet provided.
        """
        pass

    @abc.abstractmethod
    def _batch_to_netcdf(self, output: OutputABC, metadata: list[dict]):
        """
        Document this function.
        
        Parameters
        ----------
        output : OutputABC
            Description not yet provided.
        metadata : list[dict]
            Description not yet provided.
        """
        pass

    @abc.abstractmethod
    def _update_train_stats(self, output: OutputABC, batch: BatchDataABC):
        """
        Document this function.
        
        Parameters
        ----------
        output : OutputABC
            Description not yet provided.
        batch : BatchDataABC
            Description not yet provided.
        """
        pass

    @final
    @property
    def raw_module(self):
        """
        Document this function.
        
        Returns
        -------
        Any
            Description not yet provided.
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
        Document this function.
        
        Parameters
        ----------
        output : OutputABC
            Description not yet provided.
        num_output_samples : int
            Description not yet provided.
        sample_size : tuple
            Description not yet provided.
        reshape_size : tuple
            Description not yet provided.
        
        Returns
        -------
        OutputABC
            Description not yet provided.
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
        Document this function.
        
        Returns
        -------
        Callable[..., torch.Tensor]
            Description not yet provided.
        
        Raises
        ------
        ValueError
            Description not yet provided.
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
            Document this function.
            
            Parameters
            ----------
            sample_size : int | tuple[int, ...]
                Description not yet provided.
            
            Returns
            -------
            Any
                Description not yet provided.
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
        Document this function.
        
        Parameters
        ----------
        mu : torch.Tensor
            Description not yet provided.
        cov : torch.Tensor
            Description not yet provided.
        std : float
            Description not yet provided.
        
        Returns
        -------
        torch.distributions.MultivariateNormal
            Description not yet provided.
        
        Raises
        ------
        RuntimeError
            Description not yet provided.
        ValueError
            Description not yet provided.
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
        Document this function.
        
        Parameters
        ----------
        mu : torch.Tensor
            Description not yet provided.
        cov : torch.Tensor
            Description not yet provided.
        sample_size : int | tuple[int, ...]
            Description not yet provided.
        std : float
            Description not yet provided.
        
        Returns
        -------
        torch.Tensor
            Description not yet provided.
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
    Document this function.
    
    Parameters
    ----------
    prediction : torch.Tensor
        Description not yet provided.
    metadata : list[dict]
        Description not yet provided.
    num_output_dims : int
        Description not yet provided.
    save_name : str
        Description not yet provided.
    save_dir : str | Path
        Description not yet provided.
    extra_dims_sorted : list[str] | None
        Description not yet provided.
    assign_coords : dict
        Description not yet provided.
    attrs : dict
        Description not yet provided.
    
    Raises
    ------
    ValueError
        Description not yet provided.
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

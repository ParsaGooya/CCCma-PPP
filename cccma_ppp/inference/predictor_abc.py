import abc
import torch
from typing import final
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path

from cccma_ppp.data_modules.dataloader import BatchDataABC
from cccma_ppp.core import OutputABC
from cccma_ppp.generic import RunningCovariance

class PredictorABC(abc.ABC):
    
    @property
    @abc.abstractmethod
    def extract_training_vars(self) -> bool:
        pass

    @property
    def stats(self) -> dict[str, RunningCovariance]:
        if self.extract_training_vars:
            return self._stats
    
    @abc.abstractmethod
    def _infer_on_batch(self, 
                        batch: BatchDataABC, 
                        _getting_train_stats: bool = False):
        pass

    @abc.abstractmethod
    def _batch_to_netcdf(self, 
                         output: OutputABC,
                         metadata: list[dict]):
        pass

    @abc.abstractmethod
    def _update_train_stats(self, 
                            output: OutputABC, 
                            batch: BatchDataABC):
        pass

    @final
    @property
    def raw_module(self):
        """
        Access underlying module (unwrap DDP if needed).

        Returns
        -------
        moduleABC
            Raw model instance.
        """

        if isinstance(self.module, torch.nn.parallel.DistributedDataParallel):
            return self.module.module
        return self.module
    
    @final
    def build_output_sampler(self):
        stats_path = self.output_dir / "training_variable_stats.pt"

        if not stats_path.exists():
            raise ValueError(
                "Training statistics based on the trained model must be saved "
                "to disk first."
            )

        stats = torch.load(stats_path, map_location=self.device)

        def _sampler(sample_size: int | tuple[int, ...]):
            return self._sample(
                torch.zeros_like(stats["residual_mean"]),
                stats["residual_cov"],
                sample_size
            )

        return _sampler

    @final
    def _get_multinormal(
        self,
        mu: torch.Tensor,
        cov: torch.Tensor,
        std: float = 1.0,
    ) -> torch.distributions.MultivariateNormal:
        mu = mu.detach().to(self.device).float()
        cov = cov.detach().to(self.device).float()

        if std <= 0:
            raise ValueError(f"std must be positive, got {std}.")

        # Widen/narrow spread. Scaling std by k means covariance scales by k**2.
        cov = cov * (std ** 2)

        # Numerical stability for nearly-singular covariance matrices.
        jitter = 1e-6
        eye = torch.eye(cov.shape[-1], device=self.device, dtype=cov.dtype)

        for _ in range(5):
            try:
                return torch.distributions.MultivariateNormal(
                    loc=mu,
                    covariance_matrix=cov + jitter * eye,
                )
            except RuntimeError:
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
        dist = self._get_multinormal(mu=mu, cov=cov, std=std)

        if isinstance(sample_size, int):
            sample_size = (sample_size,)

        samples = dist.sample(sample_size)
        return samples.to(self.device)






def _batch_to_netcdf(
    prediction: torch.Tensor,
    metadata: list[dict],
    num_output_dims: int,
    save_name: str,
    save_dir: str | Path,
    extra_dims_sorted: list[str] | None = None,
    assign_coords: dict = None
):

    if extra_dims_sorted is None:
        extra_dims_sorted = []

    coords = {}
    for i, dim in enumerate(extra_dims_sorted):
        coords[dim] = np.arange(prediction.shape[i])
    
    batch_size = prediction.shape[len(extra_dims_sorted)]
    channel_size = prediction.shape[len(extra_dims_sorted) + 1]
    spatial_shape = prediction.shape[len(extra_dims_sorted) + 2:]

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
    coords["channels"] =  np.arange(channel_size)

    for i, size in enumerate(spatial_shape):
        coords[f"output_dim_{i}"] = np.arange(size)

    da = xr.DataArray(
        prediction.numpy(),
        dims=dims,
        coords=coords,
        name="prediction",
    )

    if assign_coords is not None:
        da = da.assign_coords(assign_coords)

    save_path = (Path(save_dir) / save_name)

    da.to_netcdf(save_path)

import abc
from typing import ClassVar, final
import xarray as xr
import numpy as np

from cccma_ppp.data_modules.utils import add_lead_times

from cccma_ppp.configs import (
    required_sample_dimensions,
    realization_dim,
    lead_time_unit,
    lead_time_resolution,
)

init_time_dim, lead_time_dim = required_sample_dimensions


class PreprocessModuleABC(abc.ABC):
    """
    Document this class.

    Attributes
    ----------
    large_ensemble : bool
        Description not yet provided.
    fitted : bool
        Description not yet provided.
    """

    large_ensemble: bool
    fitted: bool
    dims: tuple[str, ...]
    frequency: str | None

    lead_time_resolution: ClassVar[lead_time_unit] = lead_time_resolution
    init_time_dim: ClassVar[int] = init_time_dim
    lead_time_dim: ClassVar[int] = lead_time_dim
    realization_dim: ClassVar[int] = realization_dim
    supported_frequencies: ClassVar = {None, "year", "month", "day"}

    @abc.abstractmethod
    def fit(self, data):
        """
        Document this function.

        Parameters
        ----------
        data : Any
            Description not yet provided.
        """
        pass

    @abc.abstractmethod
    def transform(self, data, **kwargs):
        """
        Document this function.

        Parameters
        ----------
        data : Any
            Description not yet provided.
        **kwargs : Any
            Description not yet provided.
        """
        pass

    @abc.abstractmethod
    def inverse_transform(self, data, **kwargs):
        """
        Document this function.

        Parameters
        ----------
        data : Any
            Description not yet provided.
        **kwargs : Any
            Description not yet provided.
        """
        pass

    @final
    def _get_reduction_dims(
        self,
        data: xr.Dataset | xr.DataArray,
    ) -> tuple[str, ...] | None:
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Description not yet provided.

        Returns
        -------
        tuple[str, ...] | None
            Description not yet provided.
        """
        reduction_dims = self.dims
        self.large_ensemble = False
        missing_dims = set(reduction_dims) - set(data.dims)
        if missing_dims:
            raise ValueError(
                f"Dimensions {missing_dims} in preprocessing config dims"
                f"not found in data. Available: {set(data.dims)}"
            )

        if (
            self.realization_dim in data.dims
            and len(reduction_dims) > 0
            and self.realization_dim not in reduction_dims
        ):
            self.large_ensemble = True
            reduction_dims = (self.realization_dim, *reduction_dims)

        if len(reduction_dims) == 0:
            return None

        return reduction_dims

    @final
    def _extract_temporal_coordinate(
        self,
        init_time: xr.DataArray,
        frequency: str | None,
    ) -> xr.DataArray:
        """
        Extract temporal coordinate based on frequency.

        Single source of truth for temporal extraction logic.

        Parameters
        ----------
        init_time : xr.DataArray
            Initialization time coordinate to extract from.
        frequency : str | None
            Temporal frequency: 'year', 'month', 'day', or None.

        Returns
        -------
        xr.DataArray
            Extracted temporal coordinate matching frequency.

        Raises
        ------
        ValueError
            If frequency is not supported.
        """
        if frequency is None:
            return init_time

        if frequency == "year":
            return init_time.dt.year
        elif frequency == "month":
            return init_time.dt.month
        elif frequency == "day":
            return init_time.dt.dayofyear
        else:
            raise ValueError(
                f"Unsupported frequency {frequency!r}. "
                f"Must be one of {self.supported_frequencies}."
            )

    @final
    def _add_grouping_coordinate(
        self,
        data: xr.Dataset | xr.DataArray,
    ) -> xr.Dataset | xr.DataArray:
        """
        Document this function.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Description not yet provided.

        Returns
        -------
        xr.Dataset | xr.DataArray
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        """
        if self.frequency is None:
            return data

        init_time = data[self.init_time_dim]
        grouping_coord = self._extract_temporal_coordinate(init_time, self.frequency)

        return data.assign_coords({self.frequency: grouping_coord})

    @final
    def _align_stat_for_transform(
        self,
        data: xr.DataArray | xr.Dataset,
        stat: xr.DataArray | xr.Dataset,
    ) -> xr.DataArray:
        """
        Document this function.

        Parameters
        ----------
        data : xr.DataArray | xr.Dataset
            Description not yet provided.
        stat : xr.DataArray | xr.Dataset
            Description not yet provided.

        Returns
        -------
        xr.DataArray
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        ValueError
            Description not yet provided.
        """
        if self.frequency is None or self.init_time_dim not in self.dims:
            return stat

        if self.frequency in data.coords:
            temporal_indexer = data.coords[self.frequency]

        else:
            if self.init_time_dim not in data.coords:
                raise ValueError(
                    f"Data must contain either the auxiliary coordinate "
                    f"{self.frequency!r} or the initialization-time "
                    f"coordinate {self.init_time_dim!r}."
                )

            init_time = data[self.init_time_dim]
            temporal_indexer = self._extract_temporal_coordinate(init_time, self.frequency)

        return stat.sel({self.frequency: temporal_indexer})

    @final
    def _get_inverse_target_time(
        self,
        data: xr.DataArray | xr.Dataset,
    ) -> xr.DataArray:
        """
        Document this function.

        Parameters
        ----------
        data : xr.DataArray | xr.Dataset
            Description not yet provided.

        Returns
        -------
        xr.DataArray
            Description not yet provided.
        
        Notes
        ------
        Assumes UTC timezones.
        """
        init_times = np.asarray(data[self.init_time_dim].values)

        if self.lead_time_dim not in data.dims:
            return data[self.init_time_dim]

        lead_times = np.asarray(data[self.lead_time_dim].values)

        init_grid, lead_grid = np.meshgrid(
            init_times,
            lead_times,
            indexing="ij",
        )

        target_times = add_lead_times(
            init_times=init_grid.reshape(-1),
            lead_times=lead_grid.reshape(-1),
            lead_time_resolution=self.lead_time_resolution,
        )

        if np.isnat(target_times).any():
            raise ValueError(
                f"add_lead_times produced NaT (Not a Time) values. "
                f"Check that init_times and lead_times are compatible."
            )

        if target_times.shape != init_grid.reshape(-1).shape:
            raise RuntimeError(
                f"add_lead_times returned wrong shape. "
                f"Expected {init_grid.reshape(-1).shape}, got {target_times.shape}"
            )

        target_times = target_times.reshape(init_grid.shape)

        return xr.DataArray(
            target_times,
            dims=(
                self.init_time_dim,
                self.lead_time_dim,
            ),
            coords={
                self.init_time_dim: data[self.init_time_dim],
                self.lead_time_dim: data[self.lead_time_dim],
            },
        )

    @final
    def _check_fitted(self) -> None:
        """
        Document this function.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        """
        if not self.fitted:
            raise RuntimeError(
                "The preprocessor must be fitted before calling "
                "'transform' or 'inverse_transform'."
            )

    @final
    def _check_zeros(self, data: xr.Dataset | xr.DataArray):

        zero_range_vars = []
        if isinstance(data, xr.Dataset):
            for var in data.data_vars:
                if (data[var] == 0).any():
                    zero_range_vars.append(var)
        else:
            if (data == 0).any():
                zero_range_vars.append(data.name or "data")

        if zero_range_vars:
            raise ValueError(
                f"{zero_range_vars} variables have zero values that cause undefined behaviour in .transform()."
            )

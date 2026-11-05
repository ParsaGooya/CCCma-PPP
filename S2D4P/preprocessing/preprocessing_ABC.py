import abc
from typing import ClassVar, final
import xarray as xr
import numpy as np

from S2D4P.data.utils import add_lead_times

from S2D4P.configs import (
    required_sample_dimensions,
    realization_dim,
    lead_time_unit,
    lead_time_resolution,
)

init_time_dim, lead_time_dim = required_sample_dimensions


class PreprocessModuleABC(abc.ABC):
    """
    Abstract base class for preprocessing modules.

    Defines the common interface for fitting, transforming, and inverse
    transforming xarray data, together with shared utilities for temporal
    grouping, reduction dimensions, and validation.

    Attributes
    ----------
    large_ensemble : bool
        Whether the realization dimension is implicitly included in reductions
        because it is present in the data but absent from the configured
        reduction dimensions.
    fitted : bool
        Whether the preprocessor has been fitted.
    dims : tuple[str, ...]
        Dimensions over which preprocessing statistics are calculated.
    frequency : str | None
        Optional temporal grouping frequency. Supported values are ``"year"``,
        ``"month"``, ``"day"``, and ``None``.
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
        Fit the preprocessing module to data.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data used to estimate preprocessing statistics.

        Returns
        -------
        Any
            Implementation-specific fit result.
        """
        pass

    @abc.abstractmethod
    def _reset_fit(self):
        """
        Reset fitted preprocessing state.

        Returns
        -------
        Any
            Implementation-specific reset result.
        """
        pass

    @abc.abstractmethod
    def transform(self, data, **kwargs):
        """
        Apply the fitted preprocessing transformation.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data to transform.
        kwargs : Any
            Additional implementation-specific transformation arguments.

        Returns
        -------
        xr.Dataset | xr.DataArray
            Transformed data.
        """
        pass

    @abc.abstractmethod
    def inverse_transform(self, data, **kwargs):
        """
        Reverse the fitted preprocessing transformation.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Transformed data to restore.
        kwargs : Any
            Additional implementation-specific inverse-transformation arguments.

        Returns
        -------
        xr.Dataset | xr.DataArray
            Inverse-transformed data.
        """
        pass

    @final
    def _get_reduction_dims(
        self,
        data: xr.Dataset | xr.DataArray,
    ) -> tuple[str, ...] | None:
        """
        Resolve dimensions used to calculate preprocessing statistics.

        Configured reduction dimensions are validated against the input data. When
        the data contain a realization dimension that is not explicitly reduced, the
        realization dimension is added automatically and the data are marked as a
        large ensemble.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data whose dimensions are used to resolve reduction dimensions.

        Returns
        -------
        tuple[str, ...] | None
            Dimensions over which preprocessing statistics are reduced, or ``None``
            when no reduction dimensions are configured.

        Raises
        ------
        ValueError
            If a configured reduction dimension is absent from the input data.
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
        Add the configured temporal grouping coordinate.

        When temporal grouping is enabled, a year, month, or day-of-year coordinate
        is derived from initialization time and assigned to the data.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data to which the temporal grouping coordinate is added.

        Returns
        -------
        xr.Dataset | xr.DataArray
            Data containing the grouping coordinate, or the original data when no
            temporal frequency is configured.
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
        Align fitted temporal statistics with data being transformed.

        When preprocessing statistics are grouped by time, the statistic values
        corresponding to each input initialization time are selected using an existing
        auxiliary temporal coordinate or one derived from initialization time.

        Parameters
        ----------
        data : xr.DataArray | xr.Dataset
            Data being transformed.
        stat : xr.DataArray | xr.Dataset
            Fitted preprocessing statistic to align with ``data``.

        Returns
        -------
        xr.DataArray | xr.Dataset
            Statistic aligned with the temporal coordinates of the input data.

        Raises
        ------
        ValueError
            If temporal alignment is required but neither the configured grouping
            coordinate nor initialization-time coordinate is available.
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
            temporal_indexer = self._extract_temporal_coordinate(
                init_time, self.frequency
            )

        return stat.sel({self.frequency: temporal_indexer})

    @final
    def _get_inverse_target_time(
        self,
        data: xr.DataArray | xr.Dataset,
    ) -> xr.DataArray:
        """
        Calculate target times required for inverse temporal preprocessing.

        When lead times are present, target times are calculated for every
        initialization-time and lead-time combination. Otherwise, initialization time
        is returned directly.

        Parameters
        ----------
        data : xr.DataArray | xr.Dataset
            Data containing initialization-time and optional lead-time coordinates.

        Returns
        -------
        xr.DataArray
            Target-time coordinate indexed by initialization time and lead time, or
            the initialization-time coordinate when no lead-time dimension exists.

        Raises
        ------
        ValueError
            If calculated target times contain invalid ``NaT`` values.
        RuntimeError
            If calculated target times have an unexpected shape.

        Notes
        -----
        Python datetime values are assumed to use UTC timezones.
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
                "add_lead_times produced NaT (Not a Time) values. "
                "Check that init_times and lead_times are compatible."
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
        Verify that the preprocessing module has been fitted.

        Raises
        ------
        RuntimeError
            If the preprocessor has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError(
                "The preprocessor must be fitted before calling "
                "'transform' or 'inverse_transform'."
            )

    @final
    def _check_zeros(self, data: xr.Dataset | xr.DataArray):
        """
        Verify that data do not contain zero values.

        This validation is intended for preprocessing transformations for which zero
        values would cause undefined behavior.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data to inspect for zero values.

        Raises
        ------
        ValueError
            If any variable contains one or more zero values.
        """
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

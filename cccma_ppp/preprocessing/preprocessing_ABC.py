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
    Abstract base class for preprocessing modules.

    Defines the common interface and utilities for fitting, transforming, and
    inverse transforming xarray data, including temporal grouping and
    dimension-reduction behavior.

    Attributes
    ----------
    large_ensemble : bool
        Whether the realization dimension is automatically included when
        calculating preprocessing statistics.
    fitted : bool
        Whether the preprocessing module has been fitted.
    dims : tuple[str, ...]
        Dimensions over which preprocessing statistics are calculated.
    frequency : str | None
        Temporal frequency used to group preprocessing statistics. Supported
        values are ``"year"``, ``"month"``, ``"day"``, and ``None``.
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
        data : Any
            Data used to estimate preprocessing parameters.
        """
        pass

    @abc.abstractmethod
    def transform(self, data, **kwargs):
        """
        Apply the fitted preprocessing transformation.

        Parameters
        ----------
        data : Any
            Data to transform.
        kwargs : Any
            Additional transformation-specific keyword arguments.

        Returns
        -------
        Any
            Transformed data.
        """
        pass

    @abc.abstractmethod
    def inverse_transform(self, data, **kwargs):
        """
        Apply the inverse preprocessing transformation.

        Parameters
        ----------
        data : Any
            Data to transform back to its original representation.
        kwargs : Any
            Additional inverse-transformation-specific keyword arguments.

        Returns
        -------
        Any
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

        Configured dimensions must exist in the input data. When a realization
        dimension is present but is not explicitly included in the configured
        reduction dimensions, it is added automatically and the data are marked as a
        large ensemble.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data whose dimensions are used to resolve the reduction dimensions.

        Returns
        -------
        tuple[str, ...] | None
            Dimensions over which preprocessing statistics are reduced, or ``None``
            when no reduction dimensions are configured.

        Raises
        ------
        ValueError
            If a configured reduction dimension is not present in the input data.
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
        Extract a temporal coordinate for a specified frequency.

        Parameters
        ----------
        init_time : xr.DataArray
            Initialization-time coordinate from which temporal values are extracted.
        frequency : str | None
            Temporal frequency to extract. Supported values are ``"year"``,
            ``"month"``, ``"day"``, and ``None``.

        Returns
        -------
        xr.DataArray
            Original initialization times when ``frequency`` is ``None``, or the
            corresponding year, month, or day-of-year values.

        Raises
        ------
        ValueError
            If ``frequency`` is not supported.
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
        Add a temporal grouping coordinate to data.

        When a preprocessing frequency is configured, the corresponding year, month,
        or day-of-year values are derived from initialization time and assigned as an
        auxiliary coordinate.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data to which the temporal grouping coordinate is added.

        Returns
        -------
        xr.Dataset | xr.DataArray
            Data with the temporal grouping coordinate added, or the original data
            when no temporal frequency is configured.
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
        Align temporal preprocessing statistics with input data.

        When preprocessing statistics are grouped temporally, the appropriate
        statistic is selected for each input initialization time using either an
        existing auxiliary temporal coordinate or one derived from initialization
        time.

        Parameters
        ----------
        data : xr.DataArray | xr.Dataset
            Data to which the preprocessing statistic will be applied.
        stat : xr.DataArray | xr.Dataset
            Preprocessing statistic to align with the input data.

        Returns
        -------
        xr.DataArray | xr.Dataset
            Statistic aligned with the temporal coordinates of ``data``.

        Raises
        ------
        ValueError
            If temporal alignment is required but neither the configured auxiliary
            temporal coordinate nor the initialization-time coordinate is available.
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
        Calculate target times associated with initialization and lead times.

        If the data contain a lead-time dimension, target times are calculated for
        every initialization-time and lead-time combination. Otherwise, the existing
        initialization-time coordinate is returned.

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
            If target-time calculation produces invalid ``NaT`` values.
        RuntimeError
            If the calculated target times do not have the expected shape.

        Notes
        -----
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
            If the preprocessing module has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError(
                "The preprocessor must be fitted before calling "
                "'transform' or 'inverse_transform'."
            )

    @final
    def _check_zeros(self, data: xr.Dataset | xr.DataArray):
        """
        Check data for zero values that would make a transformation undefined.

        For datasets, each variable is checked independently. For a DataArray, the
        array itself is checked.

        Parameters
        ----------
        data : xr.Dataset | xr.DataArray
            Data to inspect for zero values.

        Raises
        ------
        ValueError
            If one or more variables contain zero values.
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

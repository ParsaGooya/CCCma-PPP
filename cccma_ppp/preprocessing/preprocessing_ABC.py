import abc
from typing import ClassVar, final
import xarray as xr
from cccma_ppp.configs import (required_sample_dimensions,
                               realization_dim,
                               lead_time_unit,
                               lead_time_resolution)

init_time_dim, lead_time_dim = required_sample_dimensions
class PreprocessModuleABC(abc.ABC):
    """
    Abstract base class for preprocessing modules.

    Classvar:
        lead_time_resolution : {"month", "day"}, optional
            Temporal unit represented by one lead-time increment.
    """
    
    large_ensemble: bool
    fitted: bool

    lead_time_resolution: ClassVar[lead_time_unit] = lead_time_resolution
    init_time_dim: ClassVar[int] = init_time_dim
    lead_time_dim: ClassVar[int] = lead_time_dim
    realization_dim: ClassVar[int] = realization_dim
    supported_frequencies: ClassVar = {None, "year", "month", "day"}

    
    @abc.abstractmethod
    def fit(self, data):
        """
        Fit preprocessing parameters to data.

        Parameters
        ----------
        data : object
            Input data used to estimate preprocessing statistics.

        Returns
        -------
        None
        """

        pass

    @abc.abstractmethod
    def transform(self, data, **kwargs):
        """
        Apply preprocessing transformation.

        Parameters
        ----------
        data : object
            Input data to transform.
        **kwargs : dict
            Additional arguments for transformation.

        Returns
        -------
        object
            Transformed data.
        """

        pass

    @abc.abstractmethod
    def inverse_transform(self, data, **kwargs):
        """
        Reverse preprocessing transformation.

        Parameters
        ----------
        data : object
            Transformed data.
        **kwargs : dict
            Additional arguments for inverse transformation.

        Returns
        -------
        object
            Data in original representation.
        """

        pass

    @final
    def _get_reduction_dims(
        self,
        data: xr.Dataset | xr.DataArray,
    ) -> tuple[str, ...] | None:
        """
        Resolve the dimensions reduced during fitting.
        """
        reduction_dims = self.dims

        if (
            self.realization_dim in data.dims
            and reduction_dims is not None
            and self.realization_dim not in reduction_dims
        ):
            self.large_ensemble = True
            reduction_dims = (self.realization_dim, *reduction_dims)

        return reduction_dims
    
    @final
    def _add_grouping_coordinate(
        self,
        data: xr.Dataset | xr.DataArray,
    ) -> xr.Dataset | xr.DataArray:
        """
        Derive and attach the temporal coordinate used during fitting.

        """
        if (self.frequency is None or self.init_time_dim not in self.dims):
            return data

        init_time = data[self.init_time_dim]

        if self.frequency == "year":
            grouping_coord = init_time.dt.year

        elif self.frequency == "month":
            grouping_coord = init_time.dt.month

        elif self.frequency == "day":
            grouping_coord = init_time.dt.dayofyear

        else:
            raise RuntimeError(
                f"Unexpected temporal frequency {self.frequency!r}."
            )

        return data.assign_coords(
            {self.frequency: grouping_coord}
        )

    @final
    def _align_stat_for_transform(
        self,
        data: xr.DataArray,
        stat: xr.DataArray,
    ) -> xr.DataArray:
        """
        Align a fitted statistic for forward transformation.

        This alignment depends only on initialization time. 
        """
        if (self.frequency is None or self.init_time_dim not in self.dims):
            return stat

        # If the temporal auxilary coordinates were added while loading data
        # transform survives sample selection.
        if self.frequency in data.coords:
            temporal_indexer = data.coords[self.frequency]

        else:
            # Generic fallback: derive the indexer from init_time.

            if self.init_time_dim not in data.coords:
                raise ValueError(
                    f"Data must contain either the auxiliary coordinate "
                    f"{self.frequency!r} or the initialization-time "
                    f"coordinate {init_time_dim!r}."
                )

            init_time = data[self.init_time_dim]

            if self.frequency == "year":
                temporal_indexer = init_time.dt.year

            elif self.frequency == "month":
                temporal_indexer = init_time.dt.month

            elif self.frequency == "day":
                temporal_indexer = init_time.dt.dayofyear

            else:
                raise RuntimeError(
                    f"Unexpected temporal frequency {self.frequency!r}."
                )

        return stat.sel(
            {self.frequency: temporal_indexer}
        )


    @final
    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError(
                "The preprocessor must be fitted before calling "
                "'transform' or 'inverse_transform'."
            )
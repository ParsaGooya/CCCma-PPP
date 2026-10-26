import numpy as np
import xarray as xr
import dataclasses
from typing import final, Literal
import cftime

from cccma_ppp.data_modules.data.data_abc import DataConfigABC
from cccma_ppp.preprocessing.preprocessing import PreprocessingPipeline
from cccma_ppp.configs import (
    lead_time_unit,
    lead_time_resolution,
    required_sample_dimensions,
    model_data_allowed_dimensions,
    model_data_required_dimensions,
    observation_data_allowed_dimensions,
    observation_data_required_dimensions,
    condition_data_allowed_dimensions,
    condition_data_required_dimensions,
)

spatialmethod = Literal["uniform", "cosine_lat"]
init_time_dim, lead_time_dim = required_sample_dimensions


@dataclasses.dataclass
class ModelDataConfig(DataConfigABC):

    """
    Configuration for model prediction data.

    Extends `DataConfigABC` with model-specific dimension requirements
    and forecast temporal coverage. The forecast time range is constructed
    from the initialization times, maximum lead time, and configured
    lead-time resolution.

    All configuration parameters and data management methods are inherited
    from `DataConfigABC`.

    See Also
    --------
    DataConfigABC : Base class defining shared dataset configuration,
        validation, loading, and preprocessing functionality.
    """


    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    realization_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str | None = None
    file_type: str = "*.nc"
    rename_dict: dict[str, str] = None
    drop_vars_list: list[str] = None

    def __post_init__(self) -> None:

        """
        Initialize the model configuration and construct its forecast time range.

        Performs base-class initialization and constructs `time_range` from
        the initialization-time coordinates and maximum available lead time.
        """
        super().__init__()

        self.time_range = build_time_range(
            init_time=self.coords[self.init_time_dim],
            n_lead_times=self.coords[self.lead_time_dim].max().item(),
            lead_time_resolution=lead_time_resolution,
        )

    @property
    @final
    def TYPE(self) -> str:
        """
        Define the type of data condifured with this class.
        """
        return "model"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Return the dimensions permitted for model data.
        """
        return model_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Return the dimensions required for model data.

        Notes
        ----
        Model data must have initialization time and lead time dimensions
        and spatial dimensions but may also include additional dimensions such 
        and ensemble member. The allowed and required dimensions are defined in pipelne configs.
        """
        return model_data_required_dimensions


@dataclasses.dataclass
class ObsDataConfig(DataConfigABC):

    """
    Configuration for observational data.

    Extends `DataConfigABC` with observation-specific dimension
    requirements and temporal validation. Observational data must
    have the same temporal frequency as the configured forecast
    lead-time resolution.

    The observation time range is constructed from the available
    observation times.

    All configuration parameters and data management methods are inherited
    from `DataConfigABC`.

    See Also
    --------
    DataConfigABC : Base class defining shared dataset configuration,
        validation, loading, and preprocessing functionality.
    """

    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    realization_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str | None = None
    file_type: str = "*.nc"
    rename_dict: dict[str, str] = None
    drop_vars_list: list[str] = None

    def __post_init__(self):

        """
        Initialize the observation configuration and validate temporal resolution.

        Performs base-class initialization and verifies that the inferred
        observation frequency matches the configured lead-time resolution.
        Constructs `time_range` from the observation time coordinates.

        Raises
        ------
        RuntimeError
            If the observation frequency differs from the configured
            lead-time resolution.
        """

        super().__init__()

        if self.init_time_frequency != self.lead_time_resolution:
            raise RuntimeError(
                "Observation data must have same temporal frequency as the "
                f"lead time. Got {self.init_time_frequency} vs {self.lead_time_resolution}"
            )

        self.time_range = build_time_range(init_time=self.coords[init_time_dim], 
                                           lead_time_resolution=self.init_time_frequency)

    @final
    @property
    def TYPE(self) -> str:
        """
        Define the type of data condifured with this class.
        """
        return "observation"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Return the dimensions permitted for observation data.
        """
        return observation_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Return the dimensions required for observation data.

        Notes
        ----
        Observation data must have initialization time dimension and spatial dimensions,
        The allowed and requied dimensions are defined in pipelne configs.
        """
        return observation_data_required_dimensions


@dataclasses.dataclass
class ConditionDataConfig(DataConfigABC):

    """
    Configuration for auxiliary conditioning datasets.

    Extends `DataConfigABC` with dimension requirements for conditioning
    variables used by the pipeline.

    Conditioning data may have a defined temporal range or be independent
    of initialization time. When temporal coverage is available, the
    time range is constructed from the initialization times and maximum
    lead time. Otherwise, `time_range` is set to None.

    All configuration parameters and data management methods are inherited
    from `DataConfigABC`.

    See Also
    --------
    DataConfigABC : Base class defining shared dataset configuration,
        validation, loading, and preprocessing functionality.
    """


    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    realization_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str | None = None
    file_type: str = "*.nc"
    rename_dict: dict[str, str] = None
    drop_vars_list: list[str] = None

    def __post_init__(self):
        """
        Document this function.
        """
        super().__init__()

        if self.info.start_time is not None and self.info.final_time is not None:
            self.time_range = build_time_range(
                init_time=self.coords[init_time_dim],
                n_lead_times=self.coords[lead_time_dim].max().item(),
                lead_time_resolution=lead_time_resolution,
            )
        else:
            self.time_range = None

    @final
    @property
    def TYPE(self) -> str:
        """
        Define the type of data condifured with this class.
        """
        return "condition"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Return the dimensions permitted for condition data.
        """
        return condition_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Return the dimensions required for condition data.

        Notes
        ----
        Conditioning data could have initialization time and lead time dimensions
        but only need spatial dimensions as static conditioning. May also include additional dimensions such 
        and ensemble member. The allowed and required dimensions are defined in pipelne configs.
        """
        return condition_data_required_dimensions


def build_time_range(
    init_time: xr.DataArray,
    n_lead_times: int = 1,
    lead_time_resolution: lead_time_unit = "month",
) -> xr.CFTimeIndex | np.ndarray:
    """
    Build a continuous temporal extent from init times and lead times.

    Returns the inclusive continuous range [min(init_time),
    max(init_time) + max_lead_time]. This is NOT a list of valid times
    in the data, only the temporal extent covered.

    Use for temporal validation (e.g., checking if all samples
    fall within expected range), not for exact time matching.
 

    Parameters
    ----------
    init_time : xr.DataArray
        Description not yet provided.
    n_lead_times : int
        Description not yet provided.
    lead_time_resolution : lead_time_unit
        Description not yet provided.

    Returns
    -------
    xr.CFTimeIndex | np.ndarray
        Description not yet provided.

    Raises
    ------
    TypeError
        Description not yet provided.
    ValueError
        Description not yet provided.
    """
    
    if init_time.size == 0:
        raise ValueError("'init_time' cannot be empty.")

    if init_time.ndim != 1:
        raise ValueError("'init_time' must be one-dimensional.")

    if n_lead_times < 1:
        raise ValueError("'n_lead_times' must be at least 1.")

    first_value = init_time.values[0]

    is_cftime = isinstance(first_value, cftime.datetime)
    is_datetime64 = np.issubdtype(init_time.dtype, np.datetime64)

    if not (is_cftime or is_datetime64):
        raise TypeError(
            "'init_time' must contain either numpy.datetime64 "
            "or cftime.datetime objects."
        )

    frequencies = {
        "month": "MS",
        "day": "D",
    }
    if lead_time_resolution not in frequencies:
        raise ValueError(
            f"Unsupported lead-time resolution {lead_time_resolution!r}. "
            f"Must be one of {list(frequencies.keys())}"
        )
    frequency = frequencies[lead_time_resolution]

    start_time = init_time.min().item()
    final_init_time = init_time.max().item()

    calendar = final_init_time.calendar if is_cftime else "proleptic_gregorian"

    final_time = xr.date_range(
        start=final_init_time,
        periods=n_lead_times,
        freq=frequency,
        calendar=calendar,
        use_cftime=is_cftime,
    )[-1]

    return xr.date_range(
        start=start_time,
        end=final_time,
        freq=frequency,
        calendar=calendar,
        use_cftime=is_cftime,
    )

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
    Configuration for model forecast data.

    Parameters
    ----------
    paths : str
        Directory containing the model data files.
    names : list[str]
        Names of data variables to load.
    preprocessing_pipeline : PreprocessingPipeline
        Pipeline used to preprocess the model data.
    realization_list : list | None
        Realizations to select from ensemble data. If ``None``, no explicit
        realization subset is selected.
    ensemble_mean : bool | None
        Whether ensemble data are reduced to their ensemble mean.
    concat_dim : str
        Dimension along which data from multiple files are concatenated.
    file_type : str
        File pattern used to discover model data files.
    rename_dict : dict[str, str] | None
        Mapping used to rename dataset variables, dimensions, or coordinates.
    """

    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    realization_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = init_time_dim
    file_type: str = "*.nc"
    rename_dict: dict[str, str] = None

    def __post_init__(self) -> None:
        """
        Initialize the model data configuration.

        After resolving and validating the configured data, constructs the continuous
        temporal range covered by the initialization times and available lead times.
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
        Return the model data type identifier.

        Returns
        -------
        str
            ``"model"``.
        """

        return "model"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Return the dimensions permitted for model data.

        Returns
        -------
        frozenset[str]
            Names of all dimensions allowed in model datasets.
        """
        return model_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Return the dimensions required for model data.

        Returns
        -------
        frozenset[str]
            Names of dimensions that must be present in model datasets.
        """

        return model_data_required_dimensions


@dataclasses.dataclass
class ObsDataConfig(DataConfigABC):
    """
    Configuration for observation data.

    Parameters
    ----------
    paths : str
        Directory containing the observation data files.
    names : list[str]
        Names of data variables to load.
    preprocessing_pipeline : PreprocessingPipeline
        Pipeline used to preprocess the observation data.
    realization_list : list | None
        Realizations to select from ensemble data. If ``None``, no explicit
        realization subset is selected.
    ensemble_mean : bool | None
        Whether ensemble data are reduced to their ensemble mean.
    concat_dim : str
        Dimension along which data from multiple files are concatenated.
    file_type : str
        File pattern used to discover observation data files.
    rename_dict : dict[str, str] | None
        Mapping used to rename dataset variables, dimensions, or coordinates.
    """

    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    realization_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = init_time_dim
    file_type: str = "*.nc"
    rename_dict: dict[str, str] = None

    def __post_init__(self):
        """
        Initialize and validate the observation data configuration.

        Observation frequency is required to match the configured lead-time
        resolution. A continuous temporal range is then constructed from the
        observation initialization times.

        Raises
        ------
        RuntimeError
            If the observation initialization-time frequency does not match the
            configured lead-time resolution.
        """
        super().__init__()

        if self.init_time_frequency != self.lead_time_resolution:
            raise RuntimeError(
                "Observation data must have same temporal frequency as the "
                f"lead time. Got {self.init_time_frequency} vs {self.lead_time_resolution}"
            )

        self.time_range = build_time_range(
            init_time=self.coords[init_time_dim],
            lead_time_resolution=self.init_time_frequency,
        )

    @final
    @property
    def TYPE(self) -> str:
        """
        Return the observation data type identifier.

        Returns
        -------
        str
            ``"observation"``.
        """
        return "observation"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Return the dimensions permitted for observation data.

        Returns
        -------
        frozenset[str]
            Names of all dimensions allowed in observation datasets.
        """
        return observation_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Return the dimensions required for observation data.

        Returns
        -------
        frozenset[str]
            Names of dimensions that must be present in observation datasets.
        """
        return observation_data_required_dimensions


@dataclasses.dataclass
class ConditionDataConfig(DataConfigABC):
    """
    Configuration for conditioning data.

    Parameters
    ----------
    paths : str
        Directory containing the conditioning data files.
    names : list[str]
        Names of data variables to load.
    preprocessing_pipeline : PreprocessingPipeline
        Pipeline used to preprocess the conditioning data.
    realization_list : list | None
        Realizations to select from ensemble data. If ``None``, no explicit
        realization subset is selected.
    ensemble_mean : bool | None
        Whether ensemble data are reduced to their ensemble mean.
    concat_dim : str
        Dimension along which data from multiple files are concatenated.
    file_type : str
        File pattern used to discover conditioning data files.
    rename_dict : dict[str, str] | None
        Mapping used to rename dataset variables, dimensions, or coordinates.
    """

    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    realization_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = init_time_dim
    file_type: str = "*.nc"
    rename_dict: dict[str, str] = None

    def __post_init__(self):
        """
        Initialize the conditioning data configuration.

        When temporal coverage is available, constructs the continuous temporal range
        covered by the initialization times and available lead times. Otherwise,
        ``time_range`` is set to ``None``.
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
        Return the conditioning data type identifier.

        Returns
        -------
        str
            ``"condition"``.
        """
        return "condition"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Return the dimensions permitted for conditioning data.

        Returns
        -------
        frozenset[str]
            Names of all dimensions allowed in conditioning datasets.
        """
        return condition_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Return the dimensions required for conditioning data.

        Returns
        -------
        frozenset[str]
            Names of dimensions that must be present in conditioning datasets.
        """
        return condition_data_required_dimensions


def build_time_range(
    init_time: xr.DataArray,
    n_lead_times: int = 1,
    lead_time_resolution: lead_time_unit = "month",
) -> xr.CFTimeIndex | np.ndarray:
    """
    Build the continuous temporal extent covered by initialization and lead times.

    The returned range extends from the earliest initialization time through the
    latest initialization time plus the available lead-time extent. It represents
    the continuous temporal coverage rather than the exact valid times present in
    the source data.

    Parameters
    ----------
    init_time : xr.DataArray
        One-dimensional array of initialization times represented by either
        ``numpy.datetime64`` or ``cftime.datetime`` values.
    n_lead_times : int
        Number of lead-time positions included in the temporal extent.
    lead_time_resolution : lead_time_unit
        Temporal resolution used to advance lead times. Supported values are
        ``"month"`` and ``"day"``.

    Returns
    -------
    xr.CFTimeIndex | np.ndarray
        Continuous date range spanning the full temporal extent at the requested
        lead-time resolution.

    Raises
    ------
    TypeError
        If ``init_time`` does not contain ``numpy.datetime64`` or
        ``cftime.datetime`` values.
    ValueError
        If ``init_time`` is empty or not one-dimensional, if ``n_lead_times`` is
        less than one, or if ``lead_time_resolution`` is unsupported.
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

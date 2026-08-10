import numpy as np
import xarray as xr
import dataclasses
from typing import final, Literal
import cftime
import datetime

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
    Configuration for model (predictor) dataset.

    Parameters
    ----------
    paths : str
        Path to dataset files.
    names : list of str
        Variables to load.
    preprocessing_pipeline : PreprocessingPipeline, optional
        Preprocessing steps applied to data.
    realization_list : list or None, optional
        Specific ensemble members to select.
    ensemble_mean : bool or None, optional
        Whether to compute ensemble mean.
    concat_dim : str, optional
        Dimension along which files are concatenated.
    file_type : str, optional
        File pattern (e.g., "*.nc").
    rename_dict : dict or None, optional
        Variable renaming mapping.
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
    rename_dict: dict = None

    def __post_init__(self) -> None:
        """
        Initialize and validate model dataset configuration.

        Returns
        -------
        None
        """
        super().__init__()

        self.time_range = build_time_range(
            init_time=self.info.coords[self.init_time_dim],
            n_lead_times=self.info.coords[self.lead_time_dim].max().item(),
            lead_time_resolution=lead_time_resolution,
        )

    @property
    @final
    def TYPE(self) -> str:
        """
        Dataset type identifier.

        Returns
        -------
        str
            Always "model".
        """
        return "model"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Allowed dataset dimensions.

        Returns
        -------
        frozenset of str
        """
        return model_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Required dataset dimensions.

        Returns
        -------
        frozenset of str
        """
        return model_data_required_dimensions


@dataclasses.dataclass
class ObsDataConfig(DataConfigABC):
    """
    Configuration for observation (target) dataset.

    Parameters
    ----------
    paths : str
        Path(s) to observation data files.
    names : list of str
        Variable names in the dataset.
    preprocessing_pipeline : PreprocessingPipeline, optional
        Optional preprocessing pipeline applied to observations.
    realization_list : list or None, optional
        List of ensemble members, if applicable.
    ensemble_mean : bool or None, optional
        Whether to compute ensemble mean.
    concat_dim : str, optional
        Dimension along which to concatenate data.
    file_type : str, optional
        File format/type of the dataset.
    rename_dict : dict or None, optional
        Mapping of variable renames.
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
    rename_dict: dict = None

    def __post_init__(self):
        """
        Initialize observation dataset configuration.

        Resolves dataset paths, extracts metadata, and defines year range.

        Returns
        -------
        None
        """
        super().__init__()

        self.time_range = build_time_range(self.info.coords[init_time_dim])

    @final
    @property
    def TYPE(self):
        """
        Dataset type identifier.

        Returns
        -------
        str
            Always "observation".
        """
        return "observation"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Allowed dataset dimensions.

        Returns
        -------
        frozenset of str
        """
        return observation_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Required dataset dimensions.

        Returns
        -------
        frozenset of str
        """
        return observation_data_required_dimensions


@dataclasses.dataclass
class ConditionDataConfig(DataConfigABC):
    """
    Configuration for conditioning dataset.

    Parameters
    ----------
    paths : str
    names : list of str
    preprocessing_pipeline : PreprocessingPipeline, optional
    realization_list : list or None, optional
    ensemble_mean : bool or None, optional
    concat_dim : str, optional
    file_type : str, optional
    rename_dict : dict or None, optional
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
    rename_dict: dict = None

    def __post_init__(self):
        """
        Initialize condition dataset configuration.

        Returns
        -------
        None
        """
        super().__init__()

        if self.info.start_time is not None and self.info.final_time is not None:
            self.time_range = build_time_range(
                init_time=self.info.coords[init_time_dim],
                n_lead_times=self.info.coords[lead_time_dim].max().item(),
                lead_time_resolution=lead_time_resolution,
            )

    @final
    @property
    def TYPE(self):
        """
        Dataset type identifier.

        Returns
        -------
        str
            Always "condition".
        """
        return "condition"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Allowed dataset dimensions.

        Returns
        -------
        frozenset of str
        """
        return condition_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset:
        """
        Required dataset dimensions.

        Returns
        -------
        frozenset of str
        """
        return condition_data_required_dimensions





def build_time_range(
    init_time: xr.DataArray,
    n_lead_times: int = 1,
    lead_time_resolution: lead_time_unit = "month",
) -> xr.CFTimeIndex | np.ndarray:
    """
    Build the complete time range covered by initialization and lead times.

    Lead time is assumed to be one-based:
        lead_time=1 -> initialization period
        lead_time=2 -> next period
    """
    if init_time.size == 0:
        raise ValueError("'init_time' cannot be empty.")

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

    frequency = {
        "month": "MS",
        "day": "D",
    }[lead_time_resolution]

    start_time = init_time.min().item()
    final_init_time = init_time.max().item()

    calendar = (
        final_init_time.calendar
        if is_cftime
        else "proleptic_gregorian"
    )

    # The last valid time is n_lead_times - 1 periods after the
    # final initialization time.
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
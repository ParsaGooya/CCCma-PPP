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
    Document this class.

    Parameters
    ----------
    paths : str
        Description not yet provided.
    names : list[str]
        Description not yet provided.
    preprocessing_pipeline : PreprocessingPipeline
        Description not yet provided.
    realization_list : list | None
        Description not yet provided.
    ensemble_mean : bool | None
        Description not yet provided.
    concat_dim : str
        Description not yet provided.
    file_type : str
        Description not yet provided.
    rename_dict : dict
        Description not yet provided.
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
        Document this function.
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
        Document this function.

        Returns
        -------
        str
            Description not yet provided.
        """
        return "model"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Document this function.

        Returns
        -------
        frozenset[str]
            Description not yet provided.
        """
        return model_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Document this function.

        Returns
        -------
        frozenset[str]
            Description not yet provided.
        """
        return model_data_required_dimensions


@dataclasses.dataclass
class ObsDataConfig(DataConfigABC):
    """
    Document this class.

    Parameters
    ----------
    paths : str
        Description not yet provided.
    names : list[str]
        Description not yet provided.
    preprocessing_pipeline : PreprocessingPipeline
        Description not yet provided.
    realization_list : list | None
        Description not yet provided.
    ensemble_mean : bool | None
        Description not yet provided.
    concat_dim : str
        Description not yet provided.
    file_type : str
        Description not yet provided.
    rename_dict : dict
        Description not yet provided.
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
        Document this function.
        """
        super().__init__()

        self.time_range = build_time_range(self.info.coords[init_time_dim])

    @final
    @property
    def TYPE(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return "observation"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Document this function.

        Returns
        -------
        frozenset[str]
            Description not yet provided.
        """
        return observation_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Document this function.

        Returns
        -------
        frozenset[str]
            Description not yet provided.
        """
        return observation_data_required_dimensions


@dataclasses.dataclass
class ConditionDataConfig(DataConfigABC):
    """
    Document this class.

    Parameters
    ----------
    paths : str
        Description not yet provided.
    names : list[str]
        Description not yet provided.
    preprocessing_pipeline : PreprocessingPipeline
        Description not yet provided.
    realization_list : list | None
        Description not yet provided.
    ensemble_mean : bool | None
        Description not yet provided.
    concat_dim : str
        Description not yet provided.
    file_type : str
        Description not yet provided.
    rename_dict : dict
        Description not yet provided.
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
        Document this function.
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
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return "condition"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        """
        Document this function.

        Returns
        -------
        frozenset[str]
            Description not yet provided.
        """
        return condition_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset:
        """
        Document this function.

        Returns
        -------
        frozenset
            Description not yet provided.
        """
        return condition_data_required_dimensions


def build_time_range(
    init_time: xr.DataArray,
    n_lead_times: int = 1,
    lead_time_resolution: lead_time_unit = "month",
) -> xr.CFTimeIndex | np.ndarray:
    """
    Document this function.

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

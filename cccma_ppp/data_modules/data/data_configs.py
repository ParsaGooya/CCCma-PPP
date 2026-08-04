import numpy as np
import dataclasses
from typing import final, Literal

from cccma_ppp.data_modules.data.data_abc import DataConfigABC
from cccma_ppp.preprocessing.preprocessing import PreprocessingPipeline
from cccma_ppp.configs import (
    model_data_allowed_dimensions,
    model_data_required_dimensions,
    observation_data_allowed_dimensions,
    observation_data_required_dimensions,
    condition_data_allowed_dimensions,
    condition_data_required_dimensions,
)

spatialmethod = Literal["uniform", "cosine_lat"]


@dataclasses.dataclass
class ModelDataConfig(DataConfigABC):
    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
    file_type: str = "*.nc"
    rename_dict: dict = None

    def __post_init__(self) -> None:
        super().__init__()

        self.year_range = np.arange(
            self.info.start_year,
            self.info.final_year + self.info.sizes["lead_time"] // 12,
        )

    @property
    @final
    def TYPE(self) -> str:
        return "model"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        return model_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        return model_data_required_dimensions


@dataclasses.dataclass
class ObsDataConfig(DataConfigABC):
    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
    file_type: str = "*.nc"
    rename_dict: dict = None

    def __post_init__(self):
        super().__init__()

        self.year_range = np.arange(self.info.start_year, self.info.final_year + 1)

    @final
    @property
    def TYPE(self):
        return "observation"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        return observation_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        return observation_data_required_dimensions


@dataclasses.dataclass
class ConditionDataConfig(DataConfigABC):
    paths: str
    names: list[str]
    preprocessing_pipeline: PreprocessingPipeline = dataclasses.field(
        default_factory=PreprocessingPipeline
    )
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
    file_type: str = "*.nc"
    rename_dict: dict = None

    def __post_init__(self):
        super().__init__()

        if self.info.start_year is not None and self.info.final_year is not None:
            self.year_range = np.arange(
                self.info.start_year,
                self.info.final_year + self.info.sizes["lead_time"] // 12,
            )

    @final
    @property
    def TYPE(self):
        return "condition"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        return condition_data_allowed_dimensions

    @final
    @classmethod
    def _required_dims(cls) -> frozenset:
        return condition_data_required_dimensions

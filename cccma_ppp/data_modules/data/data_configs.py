import numpy as np
import dataclasses
from typing import final, Literal

from cccma_ppp.data_modules.data import DataConfigABC
from cccma_ppp.preprocessing import PreprocessingPipeline


spatialmethod = Literal['uniform', 'cosine_lat']


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

    def __post_init__(self):
        super().__init__()
        self._check_ensemble = False
        if self.ensemble_list is not None:
            self._check_ensemble = True

        self._resolve_data()
        self.info = self._get_ds_info()
        self.year_range = np.arange(
            self.info.start_year,
            self.info.final_year + self.info.sizes["lead_time"] // 12,
        )

    @final
    @property
    def TYPE(self):
        return "model"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        return frozenset({"year", "lead_time", "ensembles", "lat", "lon"}) 

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        return frozenset({"lead_time", "ensembles", "lat", "lon"})  


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
        self._check_ensemble = False
        if self.ensemble_list is not None:
            self._check_ensemble = True

        self._resolve_data()
        self.info = self._get_ds_info()
        self.year_range = np.arange(self.info.start_year, self.info.final_year + 1)

    @final
    @property
    def TYPE(self):
        return "observation"

    @final
    @classmethod
    def _allowed_dims(cls) -> frozenset[str]:
        return frozenset({"year", "month", "ensembles", "lat", "lon"}) 

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        return frozenset({"month", "ensembles", "lat", "lon"})


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
        self._check_ensemble = False
        if self.ensemble_list is not None:
            self._check_ensemble = True

        self._resolve_data()
        self.info = self._get_ds_info()
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
        return frozenset({"year", "lead_time", "ensembles", "lat", "lon"}) 

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        return frozenset({"lat", "lon"}) 




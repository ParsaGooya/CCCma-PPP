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
    ensemble_list : list | None
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
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
    file_type: str = "*.nc"
    rename_dict: dict = None

    def __post_init__(self) -> None:
        """
        Document this function.
        """
        super().__init__()

        self.year_range = np.arange(
            self.info.start_year,
            self.info.final_year + self.info.sizes["lead_time"] // 12,
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
    ensemble_list : list | None
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
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
    file_type: str = "*.nc"
    rename_dict: dict = None

    def __post_init__(self):
        """
        Document this function.
        """
        super().__init__()

        self.year_range = np.arange(self.info.start_year, self.info.final_year + 1)

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
    ensemble_list : list | None
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
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
    file_type: str = "*.nc"
    rename_dict: dict = None

    def __post_init__(self):
        """
        Document this function.
        """
        super().__init__()

        if self.info.start_year is not None and self.info.final_year is not None:
            self.year_range = np.arange(
                self.info.start_year,
                self.info.final_year + self.info.sizes["lead_time"] // 12,
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

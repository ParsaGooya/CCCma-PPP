import numpy as np
import dataclasses
from typing import final, Literal

from cccma_ppp.data_modules.data import DataConfigABC
from cccma_ppp.preprocessing import PreprocessingPipeline


spatialmethod = Literal["uniform", "cosine_lat"]

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
    ensemble_list : list or None, optional
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
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
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
        self._check_ensemble = False
        if self.ensemble_list is not None:
            self._check_ensemble = True

        self._resolve_data()
        self.info = self._get_ds_info()
        self.year_range = np.arange(
            self.info.start_year,
            self.info.final_year + self.info.sizes["lead_time"] // 12,
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
        return frozenset({"year", "lead_time", "ensembles", "lat", "lon"})

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Required dataset dimensions.

        Returns
        -------
        frozenset of str
        """
        return frozenset({"lead_time", "ensembles", "lat", "lon"})


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
    ensemble_list : list or None, optional
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
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
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
        self._check_ensemble = False
        if self.ensemble_list is not None:
            self._check_ensemble = True

        self._resolve_data()
        self.info = self._get_ds_info()
        self.year_range = np.arange(self.info.start_year, self.info.final_year + 1)

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
        return frozenset({"year", "month", "ensembles", "lat", "lon"})

    @final
    @classmethod
    def _required_dims(cls) -> frozenset[str]:
        """
        Required dataset dimensions.

        Returns
        -------
        frozenset of str
        """
        return frozenset({"month", "ensembles", "lat", "lon"})


@dataclasses.dataclass
class ConditionDataConfig(DataConfigABC):
    """
    Configuration for conditioning dataset.

    Parameters
    ----------
    paths : str
    names : list of str
    preprocessing_pipeline : PreprocessingPipeline, optional
    ensemble_list : list or None, optional
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
    ensemble_list: list | None = None
    ensemble_mean: bool | None = True
    concat_dim: str = "year"
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
        return frozenset({"year", "lead_time", "ensembles", "lat", "lon"})

    @final
    @classmethod
    def _required_dims(cls) -> frozenset:
        """
        Required dataset dimensions.

        Returns
        -------
        frozenset of str
        """
        return frozenset({"lat", "lon"})

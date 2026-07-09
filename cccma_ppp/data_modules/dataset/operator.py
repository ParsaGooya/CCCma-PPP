import numpy as np
from pathlib import Path
import xarray as xr

from cccma_ppp.data_modules.data.data_configs import DataConfigABC
from cccma_ppp.data_modules.dataset.config_abc import DatasetConfigABC
from cccma_ppp.data_modules.utils import WeightsConfig
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


class DatasetOperator:
    """
    Class for managing dataset-level operations.

    Parameters
    ----------
    config : DatasetConfigABC
        Dataset configuration.
    """

    def __init__(self, config: DatasetConfigABC):
        """
        Initialize dataset operator.

        Parameters
        ----------
        config : DatasetConfigABC

        Returns
        -------
        None
        """
        self.config = config

    @property
    def config_observation(self):
        """
        Observation dataset configuration if available.

        Returns
        -------
        ObsDataConfig or None
            Observation configuration if present, otherwise None.
        """
        if hasattr(self.config, "observation"):
            return self.config.observation

    def _fit_preprocessors(
        self,
        train_years: np.ndarray | list | tuple,
        save=False,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ):
        """
        Fit preprocessing pipelines for all datasets.

        Parameters
        ----------
        train_years : array-like
            Training years used for fitting.
        save : bool, optional
            Whether to persist fitted pipelines.
        save_path : pathlib.Path or str or None, optional
        save_name : str or None, optional

        Returns
        -------
        None

        Notes
        -----
        Applies fitting to model, observation, and condition datasets.
        """

        if self.config.model is not None:
            selection = {
                "year": train_years,
                "lead_time": self.config.lead_months,
            }
            if self.config.model.info.coords["ensembles"] is not None:
                selection["ensembles"] = self.config.model.info.coords["ensembles"]

            self.config.model._fit_preprocessor_pipeline(
                selection=selection,
                mask=True,
                save=save,
                save_path=save_path,
                save_name=save_name,
            )

        if self.config_observation is not None:
            selection = {"year": train_years}
            if self.config_observation.info.coords["ensembles"] is not None:
                selection["ensembles"] = self.config_observation.info.coords[
                    "ensembles"
                ]

            self.config_observation._fit_preprocessor_pipeline(
                selection=selection, save=save, save_path=save_path, save_name=save_name
            )

        if self.config.effective_condition is not None:
            if self.config.condition_method == "static":
                selection = {}
            else:
                selection = {
                    "year": train_years,
                    "lead_time": self.config.lead_months,
                }
                if self.config.effective_condition.info.coords["ensembles"] is not None:
                    selection["ensembles"] = (
                        self.config.effective_condition.info.coords["ensembles"]
                    )

            self.config.effective_condition._fit_preprocessor_pipeline(
                selection=selection,
                mask=True,
                save=save,
                save_path=save_path,
                save_name=save_name,
            )

        self.config._fitted_preprocessors = True

    def _load_fitted_preprocessors(self, load_dir: Path | str | None = None):
        """
        Load fitted preprocessing pipelines.

        Parameters
        ----------
        load_dir : pathlib.Path or str or None

        Returns
        -------
        None
        """

        if self.config.model is not None:
            self.config.model._load_preprocessor_pipeline(load_dir)

        if self.config_observation is not None:
            self.config_observation._load_preprocessor_pipeline(load_dir)

        if self.config.effective_condition is not None:
            self.config.effective_condition._load_preprocessor_pipeline(load_dir)

        self.config._fitted_preprocessors = True

    def _add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):
        """
        Add a fitted preprocessor to all relevant pipelines.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
        index : int, optional

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If preprocessor type is invalid.
        AssertionError
            If preprocessor is not fitted.
        """

        if not isinstance(preprocessor, PreprocessModuleABC):
            raise TypeError(
                f"preprocessor must be an instance of ProcessorConfig, "
                f"got {type(preprocessor)}"
            )
        assert preprocessor.fitted, "The preprocessor must be fitted"

        if self.config.model is not None:
            self.config.model.preprocessing_pipeline.add_fitted_preprocessor(
                preprocessor, index=index
            )
        if self.config_observation is not None:
            self.config_observation.preprocessing_pipeline.add_fitted_preprocessor(
                preprocessor, index=index
            )
        if self.config.effective_condition is not None:
            self.config.effective_condition.preprocessing_pipeline.add_fitted_preprocessor(
                preprocessor, index=index
            )

    def get_weights(
        self,
        config: WeightsConfig | None = None,
        save: bool = True,
        save_path: Path | str | None = None,
        save_name: str | None = None,
    ) -> xr.DataArray:
        """
        Compute spatial and variable weights.

        Parameters
        ----------
        config : WeightsConfig or None, optional
            Configuration controlling weight computation.
        save : bool, optional
            Whether to save computed weights.
        save_path : pathlib.Path or str or None, optional
            Directory where weights should be saved.
        save_name : str or None, optional
            Filename for saved weights.

        Returns
        -------
        xr.DataArray
            Computed spatial and variable weights.

        Raises
        ------
        ValueError
            If no valid dataset is available.
        RuntimeError
            If variable weights do not match expected variables.
        """
        if config is None:
            config = WeightsConfig()

        if self.config_observation is not None:
            target_coords = self.config_observation.info.coords.copy()
        elif self.config.model is not None:
            target_coords = self.config.model.info.coords.copy()
        else:
            raise ValueError(
                "No model or observation data is availablle. "
                "Weights could not be generated"
            )

        if "ensembles" in target_coords:
            del target_coords["ensembles"]

        from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove

        if self.config_observation is not None:
            pipeline = self.config_observation.preprocessing_pipeline
        else:
            pipeline = self.config.model.preprocessing_pipeline

        checklist = [
            isinstance(item, Flattennanremove) for item in pipeline.fitted_preprocessors
        ]

        weights = config.build_weights(
            target_coords,
            Flattennanremover=pipeline.get_preprocessors("flattener")
            if any(checklist)
            else None,
            save=save,
            save_path=save_path,
            save_name=save_name,
        )

        if "channels" in weights.dims:
            if self.config_observation is not None:
                error_msg = f"inconsistent variable weights {weights.channels.values} for taget variables {self.config_observation.names}"
                if not weights.channels.values == self.config_observation.names:
                    raise RuntimeError(error_msg)
            else:
                error_msg = f"inconsistent variable weights {weights.channels.values} for taget variables {self.config.model.names}"
                if not weights.channels.values == self.config.model.names:
                    raise RuntimeError(error_msg)

        return weights

    def get_input_var_metadata(self) -> dict:
        """
        Retrieve metadata for input variables.

        Returns
        -------
        dict
            Variable names and preprocessing steps.
        """

        metadata = dict(variables=list(), preprocessors=list())

        if self.config.effective_condition is None:
            metadata = self._update_metadata_with_dataconfig_metadata(
                metadata, self.config.model
            )

        else:
            if not self.config._using_model_data_as_condition:
                metadata = self._update_metadata_with_dataconfig_metadata(
                    metadata, self.config.model
                )
                metadata = self._update_metadata_with_dataconfig_metadata(
                    metadata, self.config.effective_condition
                )
            else:
                metadata = self._update_metadata_with_dataconfig_metadata(
                    metadata, self.config.effective_condition
                )

        return metadata

    def get_target_var_metadata(self):
        """
        Retrieve metadata for target variables.

        Returns
        -------
        dict

        Raises
        ------
        ValueError
            If no target data is available.
        """

        metadata = dict(variables=list(), preprocessors=list())

        if self.config_observation is None:
            if self.config.model is None:
                raise ValueError(
                    "No model or observation data is availablle. "
                    "target variable metadata could not be generated"
                )

            metadata = self._update_metadata_with_dataconfig_metadata(
                metadata, self.config.model
            )
        else:
            metadata = self._update_metadata_with_dataconfig_metadata(
                metadata, self.config_observation
            )

        return metadata

    def _update_metadata_with_dataconfig_metadata(
        self, metadata: dict, dataconfig: DataConfigABC
    ):
        """
        Update metadata with dataset configuration information.

        Parameters
        ----------
        metadata : dict
        dataconfig : DataConfigABC

        Returns
        -------
        dict
        """
        preprocessor_names = [
            processor[0] for processor in dataconfig.preprocessing_pipeline.pipeline
        ]
        for var in dataconfig.names:
            metadata["variables"].append(var)
            metadata["preprocessors"].append(preprocessor_names)

        return metadata


def _get_time_features(
    config: DatasetConfigABC,
    year: int,
    lead_time: int,
    input: xr.DataArray,
) -> np.ndarray | None:
    """
    Generate time-based features.

    Parameters
    ----------
    config : DatasetConfigABC
        Configuration containing feature definitions.
    year : int
        Base year of sample.
    lead_time : int
        Lead time in months.
    input : xr.DataArray
        Input data used for shape alignment.

    Returns
    -------
    np.ndarray or None
        Array of time features.

    Broadcasts features to match spatial input dimensions if needed.
    """

    if config.time_features is not None:
        time_features_list = np.array([config.time_features]).flatten()
        feature_indices = {
            "year": 0,
            "lead_time": 1,
            "month_sin": 2,
            "month_cos": 3,
        }

        target_time = year + lead_time // 12
        target_month = lead_time

        y = (target_time - np.min(config.get_common_time)) / (
            np.max(config.get_common_time) - np.min(config.get_common_time)
        )
        lt = lead_time / max(config.lead_months)
        msin = np.sin(2 * np.pi * target_month / 12.0)
        mcos = np.cos(2 * np.pi * target_month / 12.0)

        time_features = np.stack([y, lt, msin, mcos])
        time_features = time_features[
            ..., [feature_indices[k] for k in time_features_list]
        ]

        if input.ndim > 2:
            time_features = np.broadcast_to(
                time_features[(...,) + (None,) * (input.ndim - 1)],
                (time_features.shape[0],) + input.shape[1:],
            )

        return time_features

import numpy as np
import xarray as xr
import torch
import dataclasses
import warnings
from pathlib import Path

from cccma_ppp.data_modules.dataset.config_abc import (
    DatasetConfigABC,
    lead_months_config,
)
from cccma_ppp.data_modules.dataset.dataset_abc import (
    DatasetABC,
    AddedTimeFeatures,
)
from cccma_ppp.data_modules.dataset.operator import DatasetOperator
from cccma_ppp.data_modules.data.data_configs import (
    ModelDataConfig,
    ObsDataConfig,
    ConditionDataConfig,
)

from cccma_ppp.data_modules.utils import _unwrap_data_variables


from cccma_ppp.configs import supported_NN_dimensions_sorted, required_sample_dimensions


@dataclasses.dataclass
class TrainDatasetConfig(DatasetConfigABC):
    """
    Configuration for training dataset construction.

    Parameters
    ----------
    model : ModelDataConfig
        Model dataset configuration.
    observation : ObsDataConfig or None, optional
        Observation dataset configuration.
    condition : ConditionDataConfig or None, optional
        Conditioning dataset configuration.
    condition_method : str or None, optional
        Method for conditioning (e.g., "cross_ensemble", "same_member", "static").
    lead_months : array-like or None, optional
        Lead months to use.
    """

    model: ModelDataConfig
    observation: ObsDataConfig | None = None
    condition: ConditionDataConfig | None = None
    condition_method: str = None
    lead_months: lead_months_config | None = None

    def __post_init__(self):
        """
        Initialize and validate dataset configuration.

        Returns
        -------
        self
        """
        super().__init__()

        self._check_observation()

    def _check_model(self):
        return super()._check_model()

    def _check_condition(self):
        return super()._check_condition()

    @property
    def num_input_lead_months(self):
        return super().num_input_lead_months

    def _check_observation(self):
        """
        Validate observation dataset configuration.

        Returns
        -------
        self

        Raises
        ------
        AssertionError
            If required observation data is missing.
        """
        if self.observation is not None:
            for dim in [
                dim
                for dim in supported_NN_dimensions_sorted
                if dim in self.observation.info.coords
            ]:
                if dim in self.model.info.coords:
                    if not self.observation.info.coords[dim].equals(
                        self.model.info.coords[dim]
                    ):
                        warnings.warn(
                            "\n=====================================================================\n"
                            f"model and observation data do not have the same {dim} cooridnates.\n"
                            "=====================================================================\n"
                        )

                else:
                    warnings.warn(
                        "\n======================================================================\n"
                        f"observation data has NN dim {dim} which is not present in model data.\n"
                        "======================================================================\n"
                    )

        else:
            if self.condition_method is None:
                raise ValueError(
                    "No target observation is specified. Specify condition_method!"
                )

        return self

    @property
    def effective_input(self):
        return self.model

    @property
    def ds_operator(self):
        """
        Access dataset operator.

        Returns
        -------
        DatasetOperator
        """

        return DatasetOperator(self)

    @property
    def get_common_time(self):
        """
        Compute common time range.

        Returns
        -------
        np.ndarray
        """

        if self.observation is None:
            return self.model.year_range

        else:
            return np.intersect1d(self.model.year_range, self.observation.year_range)

    @property
    def available_times(self):
        """
        Available training years.

        Returns
        -------
        np.ndarray
        """

        return np.intersect1d(
            self.model.info.coords["year"].values, self.get_common_time
        )

    def fit_preprocessors(
        self,
        train_years,
        save=False,
        save_path=None,
        save_name=None,
    ):
        """
        Fit preprocessing pipeline.

        Returns
        -------
        None
        """
        self.ds_operator.fit_preprocessors(
            train_years=train_years,
            save=save,
            save_path=save_path,
            save_name=save_name,
        )

    def load_fitted_preprocessors(self, load_dir: Path | str | None = None):
        """
        Load fitted preprocessors.

        Returns
        -------
        None
        """
        self.ds_operator.load_fitted_preprocessors(load_dir)

    def add_fitted_preprocessor(self, preprocessor, index=0):
        """
        Add fitted preprocessor.

        Parameters
        ----------
        preprocessor : PreprocessModuleABC
        index : int, optional

        Returns
        -------
        None
        """

        self.ds_operator.add_fitted_preprocessor(preprocessor, index)

    def build_dataset(
        self,
        years: np.ndarray,
        time_features: AddedTimeFeatures,
        mask: xr.DataArray | None = None,
        return_metadata: bool = False,
        load: bool = False,
    ):
        """
        Construct training dataset.

        Returns
        -------
        TrainDataset
        """
        return TrainDataset(
            config=self,
            requested_years=years,
            time_features=time_features,
            mask=mask,
            return_metadata=return_metadata,
            load=load,
        )


@dataclasses.dataclass
class TrainDataset(DatasetABC):
    """
    Training dataset for model learning.

    Parameters
    ----------
    config : TrainDatasetConfig
    requested_years : array-like
    mask : xr.DataArray or None, optional
    return_metadata : bool, optional
    """

    config: TrainDatasetConfig
    requested_years: list[int] | tuple[int, ...] | np.ndarray
    time_features: AddedTimeFeatures
    mask: xr.DataArray | None = None
    return_metadata: bool = False
    load: bool = False

    def __post_init__(self):
        """
        Initialize dataset and load required data.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If preprocessors are not fitted.
        ValueError
            If requested years are invalid.
        """
        super().__init__()

        if self.config.observation is not None:
            self.observation_dataset = self._load_xarray_data(
                self.config.observation, load=self.load
            )

        self.obs_indexes = self.get_obs_indexes(self.sample_coords)

    @property
    def _autoencoding_model_data(self):
        """
        Whether model is used for autoencoding.

        Returns
        -------
        bool
        """
        return self.config.observation is None

    @property
    def _load_model(self):
        """
        Determine whether the model dataset should be loaded.

        Returns
        -------
        bool
            True if the model dataset needs to be loaded.

        Notes
        -----
        This returns ``True`` in either of the following cases:

        - A condition dataset different from the model dataset is provided
        (i.e., ``_using_model_data_as_condition`` is ``False``), regardless
        of whether observations are provided.
        - No observation dataset is provided, meaning the model data is being
        autoencoded (the condition method is already validated in the
        configuration), regardless of whether a standalone condition dataset
        is provided.
        """
        return any(
            [
                self._autoencoding_model_data,
                not self.config._using_model_data_as_condition,
            ]
        )

    @property
    def _write_condition_to_input(self):
        """
        Determine whether the condition data replaces the model input.

        Returns
        -------
        bool
            True if the condition data should be used as the sole input to the
            machine learning model.

        Notes
        -----
        This returns ``True`` in either of the following cases:

        - No standalone condition dataset is provided, but a condition method is specified. In this case, ``_using_model_data_as_condition`` is ``True`` and the condition is derived from the model data. The model dataset will only be loaded if required.
        - A standalone condition dataset is provided, but no observation dataset is available. In this case, the model data is being autoencoded, so both the model and condition datasets must be loaded.
        """
        if self.config._using_model_data_as_condition:
            return True
        else:
            if self._autoencoding_model_data:
                return True

        return False

    @property
    def _concat_condition_to_input(self):
        """
        Determine whether the condition data should be concatenated to the input.

        Returns
        -------
        bool
            True if the condition data should be concatenated to the model input.

        Notes
        -----
        This returns ``True`` when all of the following datasets are available:

        - A standalone condition dataset
        - A model dataset
        - An observation dataset

        In this case, ``_write_condition_to_input`` is ``False`` and ``effective_condition`` is available separately from the model input.
        """

        return (
            self._write_condition_to_input is False
            and self.config.effective_condition is not None
        )

    def get_obs_indexes(
        self,
        sample_coords: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray] | None:
        """
        Compute positional indexes for the observation dataset.

        Parameters
        ----------
        sample_coords : dict[str, np.ndarray]
            Sampling coordinate values for the model dataset. Must contain
            ``year`` and ``lead_time``.

        Returns
        -------
        dict[str, np.ndarray] or None
            Positional observation indexes for each sample, or ``None`` when
            no observation dataset is available.

        Raises
        ------
        ValueError
            if a corresponding observation coordinate cannot be found.
        """
        if self.observation_dataset is None:
            return None

        time_dim, lead_time_dim = required_sample_dimensions
        model_years = np.asarray(sample_coords[time_dim])
        lead_times = np.asarray(sample_coords[lead_time_dim])

        offset_years, months = np.divmod(lead_times - 0.5, 12)

        observation_coords = {
            "year": model_years + offset_years,
            "month": months + 0.5,
        }

        indexes = {
            dim: self.observation_dataset.indexes[dim].get_indexer(values)
            for dim, values in observation_coords.items()
        }

        missing_values = {
            dim: observation_coords[dim][positions == -1]
            for dim, positions in indexes.items()
            if np.any(positions == -1)
        }

        if missing_values:
            raise ValueError(
                "Some observation coordinates were not found in the observation "
                f"dataset: {missing_values}"
            )

        return indexes

    def get_target_shape(self):
        """
        Determine target shape.

        Returns
        -------
        tuple
        """

        from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove

        if self.observation_dataset is not None:
            checklist = [
                isinstance(item, Flattennanremove)
                for item in self.config.observation.preprocessing_pipeline.fitted_preprocessors
            ]

            len_names = len(self.config.observation.names)

            if any(checklist):
                out_shape = (
                    self.config.observation.preprocessing_pipeline.get_preprocessors(
                        "flattener"
                    ).final_locations.shape
                )

            else:
                out_shape = tuple(
                    self.config.observation.info.coords[dim].size
                    for dim in supported_NN_dimensions_sorted
                    if dim in self.config.observation.info.coords
                )

            return tuple([len_names, *out_shape])

        else:
            return self.get_input_shape()

    def _index_observation_dataset(self, ind: int) -> xr.DataArray | None:
        """
        Select and preprocess one observation sample.

        Parameters
        ----------
        ind : int
            Sample index.

        Returns
        -------
        xr.DataArray or None
            Preprocessed observation sample, or ``None`` when no observation
            dataset is available.
        """

        if self.observation_dataset is None:
            return None

        selection = {
            dim: [int(indexes[ind])] for dim, indexes in self.obs_indexes.items()
        }

        if "ensembles" in self.observation_dataset.dims:
            selection["ensembles"] = [
                np.random.randint(self.observation_dataset.sizes["ensembles"])
            ]

        obs = self.observation_dataset.isel(**selection)
        obs = self.config.observation.preprocessing_pipeline.transform(obs)

        return _unwrap_data_variables(obs)

    def __getitem__(self, ind):
        """
        Retrieve dataset sample.

        Parameters
        ----------
        ind : int

        Returns
        -------
        dict or tuple
            Sample dictionary, optionally with metadata.
        """
        selection = {dim: value[ind] for dim, value in self.sample_coords.items()}

        condition = self._index_condition_dataset(ind)
        target = self._index_observation_dataset(ind)
        input = self._index_model_dataset(ind)

        if self._autoencoding_model_data:
            target = input

        if self._write_condition_to_input:
            input = condition

        elif self._concat_condition_to_input:
            input = xr.concat([input, condition], dim="channels")

        time_features_array = self.time_features(selection, input)

        input_array, target_array = self._compute(
            input.data,
            target.data,
        )

        datadict = dict(
            input=torch.as_tensor(input_array, dtype=torch.float32),
            target=torch.as_tensor(target_array, dtype=torch.float32),
            added_features=torch.tensor(time_features_array, dtype=torch.float32)
            if time_features_array is not None
            else None,
        )

        if self.return_metadata:
            return datadict, selection
        else:
            return datadict

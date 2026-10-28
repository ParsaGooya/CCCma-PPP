import numpy as np
import xarray as xr
import torch
import dataclasses
import warnings
import cftime
import datetime
from pathlib import Path
from collections.abc import Sequence

from cccma_ppp.data_modules.dataset.dataset_abc import (
    DatasetConfigABC,
    lead_time_config,
)
from cccma_ppp.data_modules.dataset.dataset_abc import (
    DatasetABC,
    AddedTimeFeatures,
)
from cccma_ppp.data_modules.dataset.operator import DatasetOperator
from cccma_ppp.data_modules.data.data_configs import (
    ForecastDataConfig,
    RefDataConfig,
    ConditionDataConfig,
)

from cccma_ppp.data_modules.utils import add_lead_times, _unwrap_data_variables


from cccma_ppp.configs import lead_time_resolution


@dataclasses.dataclass
class TrainDatasetConfig(DatasetConfigABC):
    """
    Document this class.

    Parameters
    ----------
    forecast : ForecastDataConfig
        Description not yet provided.
    reference : RefDataConfig | None
        Description not yet provided.
    condition : ConditionDataConfig | None
        Description not yet provided.
    condition_method : str
        Description not yet provided.
    lead_times : lead_time_config | None
        Description not yet provided.
    """

    forecast: ForecastDataConfig
    reference: RefDataConfig | None = None
    condition: ConditionDataConfig | None = None
    condition_method: str = None
    lead_times: lead_time_config | None = None

    def __post_init__(self):
        """
        Document this function.
        """
        super().__init__()

        self._check_reference()

    def _check_reference(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.

        Warns
        -----
        UserWarning
            Description not yet provided.
        """
        if self.reference is not None:
            for dim in [
                dim
                for dim in self.supported_NN_dimensions
                if dim in self.reference.coords
            ]:
                if dim in self.forecast.coords:
                    if not self.reference.coords[dim].equals(self.forecast.coords[dim]):
                        warnings.warn(
                            "\n=====================================================================\n"
                            f"Forecast and reference data do not have the same {dim} cooridnates.\n"
                            "=====================================================================\n"
                        )

                else:
                    warnings.warn(
                        "\n======================================================================\n"
                        f"reference data has NN dim {dim} which is not present in forecast data.\n"
                        "======================================================================\n"
                    )

                if (
                    self.forecast.info.time_coords_type
                    != self.reference.info.time_coords_type
                ):
                    raise ValueError(
                        "reference data and forecast data must have the same"
                        " cftime/datetime type time coordinates."
                    )

        else:
            if self.condition_method is None:
                raise ValueError(
                    "No target reference is specified. Specify condition_method!"
                )

        return self

    @property
    def effective_input(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return self.forecast

    @property
    def ds_operator(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return DatasetOperator(self)

    @property
    def get_common_time(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        if self.reference is None:
            return self.forecast.time_range

        else:
            return self.forecast.time_range.intersection(self.reference.time_range)

    @property
    def available_times(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        forecast_times = self.forecast.coords[self.init_time_dim].to_index()
        common_times = self.get_common_time
        time_freq = self.forecast.init_time_frequency

        if time_freq == "year":
            min_year = forecast_times.year.min()
            max_year = forecast_times.year.max()

            return common_times[
                (common_times.year >= min_year) & (common_times.year <= max_year)
            ]

        elif time_freq == "month" and self.lead_time_resolution == "day":
            min_time = forecast_times.min()
            max_time = forecast_times.max()

            min_month = min_time.year * 12 + min_time.month
            max_month = max_time.year * 12 + max_time.month

            common_months = common_times.year * 12 + common_times.month

            return common_times[
                (common_months >= min_month) & (common_months <= max_month)
            ]

        return common_times[
            (common_times >= forecast_times.min()) & (common_times <= forecast_times.max())
        ]

    def fit_preprocessors(
        self,
        train_times: (
            Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
            | np.ndarray
            | xr.DataArray
        ),
        save: bool = False,
        save_path: str | Path | None = None,
        save_name: str | None = None,
    ):
        """
        Document this function.

        Parameters
        ----------
        train_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray | slice
            Description not yet provided.
        save : bool
            Description not yet provided.
        save_path : str | Path | None
            Description not yet provided.
        save_name : str | None
            Description not yet provided.
        """
        self.ds_operator.fit_preprocessors(
            train_times=train_times,
            save=save,
            save_path=save_path,
            save_name=save_name,
        )

    def load_fitted_preprocessors(self, load_dir: Path | str | None = None):
        """
        Document this function.

        Parameters
        ----------
        load_dir : Path | str | None
            Description not yet provided.
        """
        self.ds_operator.load_fitted_preprocessors(load_dir)

    def add_fitted_preprocessor(self, preprocessor, index=0):
        """
        Document this function.

        Parameters
        ----------
        preprocessor : Any
            Description not yet provided.
        index : Any
            Description not yet provided.
        """
        self.ds_operator.add_fitted_preprocessor(preprocessor, index)

    def build_dataset(
        self,
        times: (
            Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
            | np.ndarray
            | xr.DataArray
        ),
        time_features: AddedTimeFeatures,
        mask: xr.DataArray | None = None,
        return_metadata: bool = False,
        load: bool = False,
    ):
        """
        Document this function.

        Parameters
        ----------
        times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
            Description not yet provided.
        time_features : AddedTimeFeatures
            Description not yet provided.
        mask : xr.DataArray | None
            Description not yet provided.
        return_metadata : bool
            Description not yet provided.
        load : bool
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return TrainDataset(
            config=self,
            requested_times=times,
            time_features=time_features,
            mask=mask,
            return_metadata=return_metadata,
            load=load,
        )


@dataclasses.dataclass
class TrainDataset(DatasetABC):
    """
    Document this class.

    Parameters
    ----------
    config : TrainDatasetConfig
        Description not yet provided.
    requested_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
        Description not yet provided.
    time_features : AddedTimeFeatures
        Description not yet provided.
    mask : xr.DataArray | None
        Description not yet provided.
    return_metadata : bool
        Description not yet provided.
    load : bool
        Description not yet provided.
    """

    config: TrainDatasetConfig
    requested_times: (
        Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
        | np.ndarray
        | xr.DataArray
    )
    time_features: AddedTimeFeatures
    mask: xr.DataArray | None = None
    return_metadata: bool = False
    load: bool = False

    def __post_init__(self):
        """
        Document this function.
        """
        super().__init__()

        if self.config.reference is not None:
            self.config.reference.open_xarray_data(
                load=self.load, add_time_auxiliary_coords=True
            )

        self.obs_indexes = self.get_obs_indexes(self.sample_coords)

    @property
    def _autoencoding_forecast_data(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return self.config.reference is None

    @property
    def _load_forecast(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return any(
            [
                self._autoencoding_forecast_data,
                not self.config._using_forecast_data_as_condition,
            ]
        )

    @property
    def _write_condition_to_input(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        if self.config._using_forecast_data_as_condition:
            return True
        else:
            if self._autoencoding_forecast_data:
                return True

        return False

    @property
    def _concat_condition_to_input(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
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
        Document this function.

        Parameters
        ----------
        sample_coords : dict[str, np.ndarray]
            Description not yet provided.

        Returns
        -------
        dict[str, np.ndarray] | None
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if self.config.reference is None:
            return None

        forecast_times = np.asarray(sample_coords[self.config.init_time_dim])
        lead_times = np.asarray(sample_coords[self.config.lead_time_dim])

        reference_times = add_lead_times(
            init_times=forecast_times,
            lead_times=lead_times,
            lead_time_resolution=lead_time_resolution,
        )

        reference_coords = {
            self.config.init_time_dim: reference_times,
        }

        indexes = {
            dim: self.config.reference.indexes[dim].get_indexer(values)
            for dim, values in reference_coords.items()
        }

        missing_values = {
            dim: reference_coords[dim][positions == -1]
            for dim, positions in indexes.items()
            if np.any(positions == -1)
        }

        if missing_values:
            raise ValueError(
                "Some reference coordinates were not found in the reference "
                f"dataset: {missing_values}"
            )

        return indexes

    def get_target_shape(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        if self.config.reference is not None:

            len_names = len(self.config.reference.names)

            if self.config.reference.preprocessing_pipeline.has_flattener:
                flattener = (
                    self.config.reference.preprocessing_pipeline.get_flattener()
                )
                out_shape = flattener.final_locations.shape

            else:
                out_shape = tuple(
                    self.config.reference.coords[dim].size
                    for dim in self.config.supported_NN_dimensions
                    if dim in self.config.reference.coords
                )

            return tuple([len_names, *out_shape])

        else:
            return self.get_input_shape()

    def _index_reference_dataset(self, ind: int) -> xr.DataArray | None:
        """
        Document this function.

        Parameters
        ----------
        ind : int
            Description not yet provided.

        Returns
        -------
        xr.DataArray | None
            Description not yet provided.
        """
        if self.config.reference is None:
            return None

        selection = {dim: [indexes[ind]] for dim, indexes in self.obs_indexes.items()}

        if (
            not self.config.reference.ensemble_mean
            and self.config.realization_dim in self.config.reference.dims
        ):
            selection[self.config.realization_dim] = [
                np.random.randint(
                    self.config.reference.data.sizes[self.config.realization_dim]
                )
            ]

        obs = self.config.reference.isel(**selection)

        return _unwrap_data_variables(obs)

    def __getitem__(self, ind):
        """
        Document this function.

        Parameters
        ----------
        ind : Any
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
        selection = {dim: value[ind] for dim, value in self.sample_coords.items()}

        condition = self._index_condition_dataset(ind)
        target = self._index_reference_dataset(ind)
        input = self._index_forecast_dataset(ind)

        if self._autoencoding_forecast_data:
            target = input

        if self._write_condition_to_input:
            input = condition

        elif self._concat_condition_to_input:
            input = xr.concat([input, condition], dim="channels")

        added_features_array = self.time_features(ind, input)

        input_array, target_array = self._compute(
            input.data,
            target.data,
        )

        datadict = dict(
            input=torch.as_tensor(input_array, dtype=torch.float32),
            target=torch.as_tensor(target_array, dtype=torch.float32),
            added_features=torch.tensor(added_features_array, dtype=torch.float32)
            if added_features_array is not None
            else None,
        )

        if self.return_metadata:
            return datadict, selection
        else:
            return datadict

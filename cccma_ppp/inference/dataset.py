import dataclasses
from pathlib import Path
from collections.abc import Sequence
import numpy as np
import xarray as xr
import torch
import cftime
import datetime

from cccma_ppp.data_modules.dataset.dataset_abc import (
    DatasetConfigABC,
    DatasetABC,
    lead_time_config,
    AddedTimeFeatures,
)

from cccma_ppp.data_modules.dataset.operator import DatasetOperator

from cccma_ppp.data_modules.data.data_configs import (
    ModelDataConfig,
    ConditionDataConfig,
)

from cccma_ppp.train.dataset import TrainDatasetConfig
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


@dataclasses.dataclass
class InferenceDatasetConfig(DatasetConfigABC):
    model: ModelDataConfig | None = None
    condition: ConditionDataConfig | None = None
    condition_method: str = None
    lead_times: lead_time_config | None = None

    def __post_init__(self):

        super().__init__()

    @property
    def effective_input(self):
        if self.model is not None:
            return self.model
        else:
            return self.condition

    @property
    def ds_operator(self):
        return DatasetOperator(self)

    @property
    def available_times(self):

        time_ranges = list()
        if self.condition is not None:
            time_ranges.append(
                self.condition.info.coords[self.init_time_dim].to_index(),
            )
        if self.model is not None:
            time_ranges.append(
                self.model.info.coords[self.init_time_dim].to_index(),
            )

        common = time_ranges[0]
        for tr in time_ranges[1:]:
            common = common.intersection(tr)

        return common

    def load_fitted_preprocessors(self, load_dir: Path | str | None = None):
        self.ds_operator.load_fitted_preprocessors(load_dir)

    def add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):

        self.ds_operator.add_fitted_preprocessor(preprocessor, index)

    def build_dataset(
        self,
        times: (
            Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
            | np.ndarray
            | xr.DataArray
        ),
        time_features: AddedTimeFeatures,
        return_metadata: bool = False,
        load: bool = False,
    ):
        return InferenceDataset(
            config=self,
            requested_times=times,
            time_features=time_features,
            return_metadata=return_metadata,
            load=load,
        )


@dataclasses.dataclass
class InferenceDataset(DatasetABC):
    config: InferenceDatasetConfig
    requested_times:(
        Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
        | np.ndarray
        | xr.DataArray
    )
    time_features: AddedTimeFeatures
    return_metadata: bool = False
    load: bool = False

    mask: xr.DataArray | None = dataclasses.field(
        init=False,
        default=None,
    )

    def __post_init__(self):
        super().__init__()

    @property
    def _load_model(self):

        return all(
            [
                not self.config._using_model_data_as_condition,
                self.config.model is not None,
            ]
        )

    @property
    def _write_condition_to_input(self):

        return any(
            [self.config._using_model_data_as_condition, self.config.model is None]
        )

    @property
    def _concat_condition_to_input(self):

        return (
            self._write_condition_to_input is False
            and self.config.condition is not None
        )

    def __getitem__(self, ind):

        selection = {dim: value[ind] for dim, value in self.sample_coords.items()}

        condition = self._index_condition_dataset(ind)
        input = self._index_model_dataset(ind)

        if self._write_condition_to_input:
            input = condition

        elif self._concat_condition_to_input:
            input = xr.concat([input, condition], dim="channels")

        added_features_array = self.time_features(ind, input)

        (input_array,) = self._compute(input.data)

        datadict = dict(
            input=torch.as_tensor(input_array, dtype=torch.float32),
            added_features=torch.tensor(added_features_array, dtype=torch.float32)
            if added_features_array is not None
            else None,
        )

        if self.return_metadata:
            return datadict, selection
        else:
            return datadict


def _from_train(
    train_dataset_config: TrainDatasetConfig,
) -> "InferenceDatasetConfig":
    import copy

    kwargs = {
        "condition_method": train_dataset_config.condition_method,
        "lead_times": copy.deepcopy(train_dataset_config.lead_times),
    }

    train_has_observation = train_dataset_config.observation is not None
    train_has_condition = train_dataset_config.effective_condition is not None
    train_condition_from_model = train_dataset_config._using_model_data_as_condition

    if train_has_observation and not train_has_condition:
        kwargs["model"] = copy.deepcopy(train_dataset_config.model)

    elif train_condition_from_model:
        kwargs["model"] = copy.deepcopy(train_dataset_config.model)

    elif train_has_condition:
        if train_has_observation:
            kwargs["model"] = copy.deepcopy(train_dataset_config.model)
        kwargs["condition"] = copy.deepcopy(train_dataset_config.condition)

    else:
        raise ValueError(
            "Could not infer inference dataset config from training dataset config."
        )

    return InferenceDatasetConfig(**kwargs)

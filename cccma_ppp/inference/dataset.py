import dataclasses
from pathlib import Path
import numpy as np
import xarray as xr
import torch

from cccma_ppp.data_modules.dataset import (
    DatasetConfigABC, 
    DatasetABC,
    lead_months_config, 
    DatasetOperator,
    AddedTimeFeatures,
)

from cccma_ppp.data_modules.data import (
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
    lead_months: lead_months_config | None = None

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

        year_ranges = list()
        if self.condition is not None:
            year_ranges.append(
                self.condition.info.coords["year"].values,
                )
        if self.model is not None:
            year_ranges.append(
                self.model.info.coords["year"].values,
                )

        common = year_ranges[0]
        for yr in year_ranges[1:]:
            common = np.intersect1d(common, yr)

        return common

    def load_fitted_preprocessors(
        self, load_dir: Path | str | None = None
    ):
        self.ds_operator.load_fitted_preprocessors(load_dir)

    def add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):

        self.ds_operator.add_fitted_preprocessor(preprocessor, index)


    def build_dataset(
        self,
        years: np.ndarray,
        time_features: AddedTimeFeatures,
        return_metadata: bool = False,
        load: bool = False
    ):
        return InferenceDataset(
            config=self,
            requested_years=years,
            time_features=time_features,
            return_metadata=return_metadata,
            load=load
        )




@dataclasses.dataclass
class InferenceDataset(DatasetABC):
    config: InferenceDatasetConfig
    requested_years: list[int] | tuple[int, ...] | np.ndarray
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

        return all([not self.config._using_model_data_as_condition,
                    self.config.model is not None])
    
    @property
    def _write_condition_to_input(self):

        return any([self.config._using_model_data_as_condition, 
                        self.config.model is None])
    
    @property
    def _concat_condition_to_input(self):

        return (self._write_condition_to_input is False and
                self.config.condition is not None)


    def __getitem__(self, ind):

        selection = {dim : value[ind]
            for dim, value in self.sample_coords.items()
        }
        
        condition = self._index_condition_dataset(ind)
        input = self._index_model_dataset(ind)
        
        if self._write_condition_to_input:
            input = condition

        elif self._concat_condition_to_input:
            input = xr.concat([input, condition], dim="channels")

        time_features_array = self.time_features( 
                                           selection, 
                                           input)

        input_array = self._compute(input.data)

        datadict = dict(
            input=torch.as_tensor(input_array, dtype=torch.float32),
            added_features=torch.as_tensor(time_features_array, dtype=torch.float32)
            if time_features_array is not None
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
        "lead_months": copy.deepcopy(train_dataset_config.lead_months),
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
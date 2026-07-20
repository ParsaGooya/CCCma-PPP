import dataclasses
from pathlib import Path
import numpy as np
import xarray as xr
import torch
import dask

from cccma_ppp.data_modules.dataset.config_abc import (
    DatasetConfigABC,
    lead_months_config,
)
from cccma_ppp.data_modules.dataset.dataset_abc import DatasetABC
from cccma_ppp.data_modules.dataset.operator import DatasetOperator, _get_time_features

from cccma_ppp.data_modules.data.data_configs import (
    ModelDataConfig,
    ConditionDataConfig,
)

from cccma_ppp.data_modules.utils import (
    suppress_stderr,
)

from cccma_ppp.train.dataset import TrainDatasetConfig
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


@dataclasses.dataclass
class InferenceDatasetConfig(DatasetConfigABC):
    model: ModelDataConfig | None = None
    condition: ConditionDataConfig | None = None
    condition_method: str = None
    time_features: list[str] | None = None
    lead_months: lead_months_config | None = None

    def __post_init__(self):
        self._fitted_preprocessors: bool = False
        self._effective_condition: ConditionDataConfig | ModelDataConfig | None = None

        super().__init__()

        self._check_model()
        self._check_condition()

    def _check_model(self):
        if self.model is not None:
            if self.condition_method == "same_member":
                if self.model.ensemble_mean:
                    raise ValueError(
                        "for same member coniditioning the model data should not be ensemble mean."
                    )

        return self

    def _check_condition(self):
        if self.effective_condition is not None:
            if self.condition_method is None:
                raise ValueError(
                    "You must specify condition_method for conditioning dataset!"
                )

            if self.condition_method in ["cross_ensemble", "same_member"]:
                if self.effective_condition.ensemble_mean:
                    raise ValueError(
                        "condition ensemble_mean cannot be True for cross_ensemble or same_member conditioning."
                    )
                if self.effective_condition.info.coords.get("ensembles") is None:
                    raise ValueError(
                        "For cross_ensemble or same_member conditioning an ensembles dim must exist in the condition."
                    )
            elif self.condition_method == "ensemble_mean":
                if self.effective_condition.ensemble_mean is not True:
                    raise ValueError(
                        "Ensemble mean must be True for ensemble_mean conditioning."
                    )
            else:
                if self.effective_condition.ensemble_list is not None:
                    raise ValueError(
                        'For "static" conditioning fields cannot specify ensemble list.'
                    )
                if self._using_model_data_as_condition:
                    raise ValueError(
                        "'static' conditioning method cannot point to the same model data!"
                    )

        else:
            if self.condition_method == "static":
                raise ValueError(
                    "For static conditioning method condition dataset must be specified!"
                )

        return self

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
    def num_input_lead_months(self) -> int:
        if self.model is not None:
            return self.model.info.sizes["lead_time"]

        return self.condition.info.sizes["lead_time"]

    @property
    def get_common_time(self):
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

    @property
    def available_times(self):

        return self.get_common_time

    def _load_fitted_preprocessors(self, load_dir: Path | str | None = None):
        self.ds_operator._load_fitted_preprocessors(load_dir)

    def _add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):

        self.ds_operator._add_fitted_preprocessor(preprocessor, index)

    def build_dataset(
        self, years: np.ndarray, return_metadata: bool = False, load: bool = False
    ):
        return InferenceDataset(
            config=self,
            requested_years=years,
            return_metadata=return_metadata,
            load=load,
        )


@dataclasses.dataclass
class InferenceDataset(DatasetABC):
    config: InferenceDatasetConfig
    requested_years: list[int] | tuple[int] | np.ndarray
    return_metadata: bool = False
    load: bool = False

    mask = dataclasses.field(
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

        time_features = _get_time_features(self.config, selection, input)

        with suppress_stderr():
            input = dask.compute(
                input.data,
            )

        datadict = dict(
            input=torch.as_tensor(input, dtype=torch.float32),
            added_features=torch.as_tensor(time_features, dtype=torch.float32)
            if time_features is not None
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
        "time_features": copy.deepcopy(train_dataset_config.time_features),
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

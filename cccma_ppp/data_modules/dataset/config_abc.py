
import abc
from typing import ClassVar, final
import dataclasses
import numpy as np

from cccma_ppp.data_modules.data import (
    ModelDataConfig,
    ConditionDataConfig
)


@dataclasses.dataclass
class lead_months_config:
    list_months: list | None = None
    start: int = 1
    end: int = None

    def __post_init__(self):
        if self.list_months is None:
            if self.end is None:
                raise ValueError(
                    'Provide a list of lead_months to train on,' \
                    'or specify the start-end pair to choose a slice.' \
                )

    def build_lead_months(self): 
        return self.list_months or np.arange(self.start, self.end + 1)

class DatasetConfigABC(abc.ABC):

    _VALID_CONDITION_METHODS: ClassVar[frozenset[str]] = frozenset({'ensemble_mean' , 'cross_ensemble' , 'same_member',  'static'})
    _VALID_TIME_FEATURES: ClassVar[frozenset[str]] = frozenset({'year', 'lead_time', 'month_sin', 'month_cos'}) 

    model: ModelDataConfig | None
    condition: ConditionDataConfig | None
    condition_method: str | None
    time_features: list[str] | None
    lead_months: lead_months_config | None
    _effective_condition: ConditionDataConfig | ModelDataConfig | None

    def __init__(self):
        self._check_required_input_source()
        self._check_condition_method()
        self._check_time_features()
        self._check_model_vs_condition()
        self._resolve_lead_months()
        self._resolve_condition()

        if self.lead_months is None:
            self.lead_months = np.arange(1, self.num_input_lead_months + 1)
        if not max(self.lead_months) <= self.num_input_lead_months:
            raise ValueError(
            f"Maximum available lead months is {self.num_input_lead_months}"
        )

    @final
    def _check_required_input_source(self):
        if self.model is None and self.condition is None:
            raise ValueError(
                "For a PPP dataset to create an input, either model or "
                "condition data must be provided."
            )
        return self

    @final
    def _check_model_vs_condition(self):
        if all([self.condition is not None,
            self.model is not None, 
            not self._using_model_data_as_condition]):
            if self.condition_method != "static":

                if not set(self.model.year_range).issubset(
                    set(self.condition.year_range)):

                    raise ValueError('Condition data should be available'
                    ' on the same time period as model data.')

                if not (self.model.info.sizes['lead_time'] <=
                        self.condition.info.sizes['lead_time']):

                    raise ValueError('Condition data should be available'
                    ' on the same lead_times as model data.')

                if hasattr(self, 'observation'):
                    if self.observation is not None:
                        if not self.condition.info.coords["lat"].equals(
                                self.model.info.coords["lat"]
                            ):
                            raise ValueError(
                                "model and condition data do not have the same latitudes cooridnates." /
                                "when bias correcting to observations"
                            )
                        if not self.condition.info.coords["lon"].equals(
                            self.model.info.coords["lon"]
                            ):
                            raise ValueError(
                                "model and condition data do not have the same longitudes cooridnates." /
                                "when bias correcting to observations"
                            )
                

    @final
    def _check_condition_method(self):
        if self.condition_method is not None:
  
            if self.condition_method not in self._VALID_CONDITION_METHODS:
                raise ValueError(
                    f'Invalid condition_method: {self.condition_method}. '
                    f'Must be a in {sorted(self._VALID_CONDITION_METHODS)}.'
            )
        return self
    
    @final
    def _check_time_features(self):
        if self.time_features is not None:
  
            invalid = set(self.time_features) - self._VALID_TIME_FEATURES
            if invalid:
                raise ValueError(
                    f'Invalid time features: {sorted(invalid)}. '
                    f'Must be a subset of {sorted(self._VALID_TIME_FEATURES)}.'
            )
        return self

    @final
    def _resolve_lead_months(self):
        if self.lead_months is not None:
            self.lead_months = self.lead_months.build_lead_months() 
    
    @abc.abstractmethod
    def _check_model(self):
        pass

    @abc.abstractmethod
    def _check_condition(self):
        pass

    @property
    @abc.abstractmethod
    def ds_operator(self):
        pass

    @property
    @abc.abstractmethod
    def num_input_lead_months(self) -> int:
        pass
    
    @final
    @property
    def _using_model_data_as_condition(self) -> bool:
        '''
        Check if the model data is going to be used as a condition.If so, we can avoid loading both model and condition data
        if not necessary. This should be true if:

           1- No condition data is provided but condition_method is (subject to condition_method not being "static")

           2- The provided condition data points to the same files and variables and ensemble members.
        
        '''
        if self.condition is None:
            return self.condition_method in {'ensemble_mean', 'cross_ensemble', 'same_member'}
        
        elif self.model is not None: 
            return (
                self.condition.paths == self.model.paths
                and self.condition.names == self.model.names
                and self.condition.ensemble_list == self.model.ensemble_list
            )
        
        return False
        
    @final
    @property
    def effective_condition(self) -> ConditionDataConfig | ModelDataConfig | None:
        return self._effective_condition
    
    @final
    def _model_as_condition(self) -> ModelDataConfig:
        ensemble_mean = self.condition_method == 'ensemble_mean'
        return  ModelDataConfig(
                paths=self.model.paths,
                names=self.model.names,
                preprocessing_pipeline=self.model.preprocessing_pipeline,
                ensemble_list=self.model.ensemble_list,
                concat_dim=self.model.concat_dim,
                file_type=self.model.file_type,
                ensemble_mean=ensemble_mean,
                rename_dict=self.model.rename_dict,
            )
    
    @final
    def _resolve_condition(self):
        """Resolve the conditioning dataset, falling back to model data if no condition is provided but condition_method is."""
        if self.condition is not None:
            self._effective_condition = self.condition
        elif self._using_model_data_as_condition:
            self._effective_condition = self._model_as_condition()
        else:
            self._effective_condition = None

        return self


    @abc.abstractmethod
    def build_dataset(self):
        pass

    

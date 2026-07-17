import dataclasses
from pathlib import Path
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch
import dask

from cccma_ppp.data_modules.dataset import (
    DatasetConfigABC, 
    lead_months_config, 
    DatasetOperator,
    _get_time_features,
    _build_chunks
)

from cccma_ppp.data_modules.data import (
    DataConfigABC,
    ModelDataConfig,
    ConditionDataConfig,
)

from cccma_ppp.data_modules import (
    _unwrap_data_variables,
    _load_xarray_data,
    _create_train_mask,
    suppress_stderr,
)

from cccma_ppp.train.dataset import TrainDatasetConfig
from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC
from cccma_ppp.configs import supported_NN_dimensions_sorted

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
                if not self.effective_condition.ensemble_mean is True:
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
    def available_inference_years(self):

        return self.get_common_time

    def _load_fitted_preprocessors(
        self, load_dir: Path | str | None = None
    ):
        self.ds_operator._load_fitted_preprocessors(load_dir)

    def _add_fitted_preprocessor(self, preprocessor: PreprocessModuleABC, index=0):

        self.ds_operator._add_fitted_preprocessor(preprocessor, index)


    def build_dataset(
        self,
        years: np.ndarray,
        return_metadata: bool = False,
        load: bool = False
    ):
        return InferenceDataset(
            config=self,
            requested_years=years,
            return_metadata=return_metadata,
            load=load
        )




@dataclasses.dataclass
class InferenceDataset(Dataset):
    config: InferenceDatasetConfig
    requested_years: list[int] | tuple[int] | np.ndarray
    return_metadata: bool = False
    load: bool = False

    def __post_init__(self):
        if not self.config._fitted_preprocessors:
            raise RuntimeError(
                "Make sure to fit preprocessors first!. Hint:  TrainDatasetConfig._fit_preprocessors()"
            )
        if not set(self.requested_years).issubset(set(self.config.available_inference_years)):
            raise ValueError(
                "the requested years are not common to model and condition data."
            )

        self.model_dataset = self.condition_dataset = None

        if self.config.model is not None:
            self.model_dataset = self._load_xarray_data(self.config.model,
                                                        load = self.load)

        if self.config.effective_condition is not None:
            self.condition_dataset = self._load_xarray_data(self.config.effective_condition,
                                                            load = self.load)

        self.mask = self._prepare_mask()
        self.sample_coords = self.get_sampling_coords()
        self.model_indexes = self.get_model_indexes(self.sample_coords)
        self.cond_indexes = self.get_cond_indexes(self.sample_coords)


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


    def _prepare_mask(self):
        mask = _create_train_mask(
            years=self.config.available_inference_years,
            lead_times=np.arange(1, self.config.effective_input.info.sizes["lead_time"] + 1),
        )
        mask = xr.full_like(mask, fill_value=False)

        mask = mask.sel(year=self.requested_years).sel(
            lead_time= self.config.lead_months
        )
        if all(
            [
                not self.config.effective_input.ensemble_mean,
                self.config.effective_input.info.coords.get("ensembles") is not None,
            ]
        ):
            mask = mask.expand_dims(
                ensembles=len(self.config.effective_input.info.coords["ensembles"]), axis=0
            )
            mask = mask.assign_coords(
                ensembles=self.config.effective_input.info.coords["ensembles"]
            )

        mask = mask.where(~mask)

        return mask

    def _load_xarray_data(self, 
                          config: DataConfigABC, 
                          load: bool = False):
        

        return _load_xarray_data(
            config.list_paths,
            names=config.names,
            ensemble_mean=config.ensemble_mean,
            selection={"ensembles": config.info.coords["ensembles"]}
            if config.info.coords.get("ensembles") is not None
            else None,
            concat_dim=config.concat_dim,
            rename_dict=config.rename_dict,
            load=load
        )

    def get_sampling_coords(self):
        """
        Compute coordinates for sampling the datasets.

        Returns
        -------
        dict
        """

        mask = (
            self.mask.stack(batch=dict(self.mask.sizes).keys())
            .transpose("batch", ...)
            .dropna(dim="batch")
            .batch.values
        )
        mask = tuple(map(np.array, zip(*mask)))
        indexes = {
            key: mask[ind]
            for ind, key in enumerate(self.mask.sizes)
        }

        return indexes
    
    def get_model_indexes(
        self,
        sample_coords: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray] | None:
        """
        Convert sampling coordinates to positional model-dataset indexes.

        Parameters
        ----------
        sample_coords : dict[str, np.ndarray]
            Mapping from dimension names to coordinate values for each sample.

        Returns
        -------
        dict[str, np.ndarray] or None
            Positional indexes for each dimension, or None if no model dataset
            is loaded.
        """
        if not self._load_model:
            return None

        indexes = {
            dim: self.model_dataset.indexes[dim].get_indexer(values)
            for dim, values in sample_coords.items()
        }

        missing = {
            dim: sample_coords[dim][positions == -1]
            for dim, positions in indexes.items()
            if np.any(positions == -1)
        }

        if missing:
            raise ValueError(
                f"Some sampling coordinates were not found in the model dataset: {missing}"
            )

        return indexes



    def get_cond_indexes(
        self,
        sample_coords: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray] | None:
        """
        Compute positional indexes for the conditioning dataset.

        Parameters
        ----------
        sample_coords : dict[str, np.ndarray]
            Sampling coordinate values for the model dataset.

        Returns
        -------
        dict[str, np.ndarray] or None
            Positional conditioning indexes for each sample, or ``None`` when
            no conditioning dataset is available or the condition is static.

        Raises
        ------
        ValueError
            If required sampling coordinates are missing, if ``same_member`` is
            requested without ensemble coordinates, or if any conditioning
            coordinates cannot be found.
        """
        if (
            self.condition_dataset is None
            or self.config.condition_method == "static"
        ):
            
            return None
        
        condition_coords = {
            "year": np.asarray(sample_coords["year"]),
            "lead_time": np.asarray(sample_coords["lead_time"]),
        }

        if self.config.condition_method == "same_member":
            if "ensembles" not in sample_coords:
                raise ValueError(
                    "'same_member' conditioning requires ensemble coordinates "
                    "in sample_coords."
                )

            condition_coords["ensembles"] = np.asarray(
                sample_coords["ensembles"]
            )


        indexes = {
            dim: self.condition_dataset.indexes[dim].get_indexer(values)
            for dim, values in condition_coords.items()
        }

        missing_values = {
            dim: condition_coords[dim][positions == -1]
            for dim, positions in indexes.items()
            if np.any(positions == -1)
        }

        if missing_values:
            raise ValueError(
                "Some conditioning coordinates were not found in the "
                f"conditioning dataset: {missing_values}"
            )

        return indexes

    def get_input_shape(self): ##need extra check for both model and condition

        from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove

        checklist = [
            isinstance(item, Flattennanremove)
            for item in self.config.effective_input.preprocessing_pipeline.fitted_preprocessors
        ]

        len_names = len(self.config.effective_input.names)
        if self._concat_condition_to_input:
            len_names += len(self.config.effective_condition.names)

        if any(checklist):
            return  (self.config.effective_input.preprocessing_pipeline
                        .get_preprocessors("flattener")
                        .final_locations.size * len_names,)
        else:

            return tuple(
                self.config.effective_input.info.coords[dim].size 
                for dim in supported_NN_dimensions_sorted  
                if dim in self.config.effective_input.info.coords)
        

    def get_added_features_dim(self):

        return 0 if self.config.time_features is None else len(self.config.time_features)

    def _index_condition_dataset(self, ind: int) -> xr.DataArray | None:
        """
        Select and preprocess one conditioning sample.

        Parameters
        ----------
        ind : int
            Sample index.

        Returns
        -------
        xr.DataArray or None
            Preprocessed conditioning sample, or ``None`` when no conditioning
            dataset is available.
        """
        if self.condition_dataset is None:
            return None

        if self.config.condition_method == "static":
            selection = {}

        else:
            selection = {
                dim: [int(indexes[ind])]
                for dim, indexes in self.cond_indexes.items()
            }

            if self.config.condition_method == "cross_ensemble":
                selection["ensembles"] = [
                    np.random.randint(self.condition_dataset.sizes["ensembles"])
                ]


        condition = self.condition_dataset.isel(**selection)

        condition = (
            self.config.effective_condition.preprocessing_pipeline.transform(
                condition
            )
        )

        return _unwrap_data_variables(condition)
        
    def _index_model_dataset(self, ind: int) -> xr.DataArray | None:
        """
        Select and preprocess one model sample.

        Parameters
        ----------
        ind : int
            Sample index.

        Returns
        -------
        xr.DataArray or None
            Preprocessed model sample, or ``None`` when model data are not loaded.
        """
        if not self._load_model:
            return None

        selection = {
            dim: [int(indexes[ind])]
            for dim, indexes in self.model_indexes.items()
        }

        model = self.model_dataset.isel(**selection)
        model = self.config.model.preprocessing_pipeline.transform(model)

        return _unwrap_data_variables(model)

    def __getitem__(self, ind):
        year = float(self.sample_coords["year"][ind])
        lead_time = float(self.sample_coords["lead_time"][ind])
        selection = dict(year=year, lead_time=lead_time)

        if self.sample_coords.get("ensembles") is not None:
            selection['ensemble_id'] = self.sample_coords["ensembles"][ind]
        
        condition = self._index_condition_dataset(ind)
        input = self._index_model_dataset(ind)
        
        if self._write_condition_to_input:
            input = condition

        elif self._concat_condition_to_input:
            input = xr.concat([input, condition], dim="channels")

        time_features = _get_time_features(self.config, year, lead_time, input)

        with suppress_stderr():
            input = dask.compute(input.data,)


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

    def __len__(self):
        return len(self.sample_coords.get(list(self.sample_coords.keys())[0]))




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
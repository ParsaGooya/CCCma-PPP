import abc
from typing import ClassVar, final
import dataclasses
import numpy as np
from torch.utils.data import Dataset
import xarray as xr

from cccma_ppp.data_modules.data import ModelDataConfig, ConditionDataConfig, DataConfigABC
from cccma_ppp.configs import (supported_NN_dimensions_sorted,
                               required_sample_dimensions,
                               optional_sample_dimensions)

from cccma_ppp.data_modules import (
    _unwrap_data_variables,
    _load_xarray_data,
    _create_train_mask,
)

@dataclasses.dataclass
class lead_months_config:
    """
    Configuration for selecting lead months.

    Parameters
    ----------
    list_months : list of int or None, optional
        Explicit list of lead months.
    start : int, optional
        Start of lead month range (inclusive).
    end : int or None, optional
        End of lead month range (inclusive).
    """

    list_months: list | None = None
    start: int = 1
    end: int = None

    def __post_init__(self):
        """
        Validate lead month configuration.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If neither list nor range is properly specified.
        """
        if self.list_months is None:
            if self.end is None:
                raise ValueError(
                    "Provide a list of lead_months to train on,"
                    "or specify the start-end pair to choose a slice."
                )

    def build_lead_months(self):
        """
        Construct lead month array.

        Returns
        -------
        np.ndarray or list
            Lead months defined either explicitly or as a range.
        """
        return self.list_months or np.arange(self.start, self.end + 1)


class DatasetConfigABC(abc.ABC):
    """
    Abstract base class for dataset configuration.

    Attributes
    ----------
    model : ModelDataConfig or None
    condition : ConditionDataConfig or None
    condition_method : str or None
    time_features : list of str or None
    lead_months : lead_months_config or None
    """

    _VALID_CONDITION_METHODS: ClassVar[frozenset[str]] = frozenset(
        {"ensemble_mean", "cross_ensemble", "same_member", "static"}
    )
    _VALID_TIME_FEATURES: ClassVar[frozenset[str]] = frozenset(
        {"year", "lead_time", "month_sin", "month_cos"}
    )

    model: ModelDataConfig | None
    condition: ConditionDataConfig | None
    condition_method: str | None
    time_features: list[str] | None
    lead_months: lead_months_config | None
    _effective_condition: ConditionDataConfig | ModelDataConfig | None

    def __init__(self):
        """
        Initialize dataset configuration.

        Returns
        -------
        None
        """
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
        """
        Ensure at least one input source is provided.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If both model and condition are missing.
        """
        if self.model is None and self.condition is None:
            raise ValueError(
                "For a PPP dataset to create an input, either model or "
                "condition data must be provided."
            )
        return self

    @final
    def _check_model_vs_condition(self):
        """
        Validate compatibility between model and condition datasets.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If condition data does not span the same time range as model data for non static condition methods.
        ValueError
            If condition data does not provide sufficient lead-time coverage for non static condition methods.
        ValueError
            If condition data and model data do not have similar ensembles for same_member condition methods.            
        ValueError
            If spatial coordinates (lat/lon) between model and condition differ
            when observation-based correction is applied.
        """
        if all(
            [
                self.condition is not None,
                self.model is not None,
                not self._using_model_data_as_condition,
            ]
        ):
            if self.condition_method != "static":

                for dim in [dim for dim in self.model.info.coords 
                    if dim in required_sample_dimensions
                    ]:
                    
                    if self.condition.info.coords.get(dim) is None:
                    
                        raise ValueError(
                            "Condition data should be available"
                            f" on the same dimestions as model data."
                        )
                
                    if not set(self.model.info.coords[dim].values).issubset(
                        set(self.condition.info.coords[dim].values)
                    ):
                        raise ValueError(
                            "Condition data should be available"
                            f" on the same {dim} coordinates as model data."
                        )

            
            if self.condition_method == "same_member":

                if any([self.model.info.coords.get("ensembles") is None, 
                        self.effective_condition.info.coords.get("ensembles") is None]):

                    raise ValueError(
                                "Condition data and model data must have ensembles "
                                "dims and coords."
                            )                   

                if not self.model.info.coords["ensembles"].equals(
                    self.condition.info.coords["ensembles"]):
                
                    raise ValueError(
                        "Condition data should have the same ensemble members"
                        "as model data for same_member conditioning."
                    )
                            
            if getattr(self, "observation", None) is not None:

                for dim in [dim for dim in self.model.info.coords 
                    if dim in supported_NN_dimensions_sorted]:
                

                    if self.condition.info.coords.get(dim, None) is None:
                        raise ValueError(
                            f"model and condition data must have the same NN dims."
                            / "when bias correcting to observations"
                        )

                    if not self.condition.info.coords.get(dim).equals(
                        self.model.info.coords.get(dim)
                    ):
                        raise ValueError(
                            f"model and condition data do not have the same {dim} cooridnates."
                            / "when bias correcting to observations"
                        )


    @final
    def _check_condition_method(self):
        """
        Validate conditioning method.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If condition method is not supported.
        """
        if self.condition_method is not None:
            if self.condition_method not in self._VALID_CONDITION_METHODS:
                raise ValueError(
                    f"Invalid condition_method: {self.condition_method}. "
                    f"Must be a in {sorted(self._VALID_CONDITION_METHODS)}."
                )
        return self

    @final
    def _check_time_features(self):
        """
        Validate time feature selection.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If invalid time features are specified.
        """
        if self.time_features is not None:
            invalid = set(self.time_features) - self._VALID_TIME_FEATURES
            if invalid:
                raise ValueError(
                    f"Invalid time features: {sorted(invalid)}. "
                    f"Must be a subset of {sorted(self._VALID_TIME_FEATURES)}."
                )
        return self

    @final
    def _resolve_lead_months(self):
        """
        Resolve lead month configuration.

        Returns
        -------
        None
        """
        if (self.lead_months is not None and
            isinstance(self.lead_months, lead_months_config)):
            self.lead_months = self.lead_months.build_lead_months()

    @abc.abstractmethod
    def _check_model(self):
        """
        Validate model configuration.

        Returns
        -------
        self
        """
        pass

    @abc.abstractmethod
    def _check_condition(self):
        """
        Validate condition configuration.

        Returns
        -------
        self
        """
        pass

    @property
    @abc.abstractmethod
    def available_times(self):
        """
        Available times for dataset creation.

        Returns
        -------
        np.ndarray
        """
        pass
                                  

    @property
    @abc.abstractmethod
    def ds_operator(self):
        """
        Dataset operator instance.

        Returns
        -------
        DatasetOperator
        """
        pass

    @property
    @abc.abstractmethod
    def num_input_lead_months(self) -> int:
        """
        Number of input lead months available in the dataset.

        Returns
        -------
        int
            Total number of lead months used as input to the model.
        """
        pass


    @property
    @abc.abstractmethod
    def effective_input(self) -> ConditionDataConfig | ModelDataConfig | None:

        pass


    @final
    @property
    def _using_model_data_as_condition(self) -> bool:
        """
        Determine whether the model data is reused as the condition.

        Returns
        -------
        bool
            True if the condition data is derived from or identical to the
            model data.

        Notes
        -----
        When this returns ``True``, loading separate model and condition
        datasets can be avoided when unnecessary.

        This returns ``True`` in either of the following cases:

        1. No condition dataset is provided, but a ``condition_method`` is
        specified (except when ``condition_method == "static"``).

        2. A condition dataset is provided, but it references the same files,
        variables, and ensemble members as the model dataset.
        """
        if self.condition is None:
            return self.condition_method in {
                "ensemble_mean",
                "cross_ensemble",
                "same_member",
            }

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
        """
        Effective conditioning dataset.

        Returns
        -------
        ConditionDataConfig or ModelDataConfig or None
        """
        return self._effective_condition

    @final
    def _model_as_condition(self) -> ModelDataConfig:
        """
        Create model-based condition configuration.

        Returns
        -------
        ModelDataConfig
        """
        ensemble_mean = self.condition_method == "ensemble_mean"
        return ModelDataConfig(
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
        """
        Resolve effective condition dataset.

        Returns
        -------
        self
        """
        if self.condition is not None:
            self._effective_condition = self.condition
        elif self._using_model_data_as_condition:
            self._effective_condition = self._model_as_condition()
        else:
            self._effective_condition = None

        return self

    @abc.abstractmethod
    def build_dataset(self):
        """
        Build dataset instance.

        Returns
        -------
        Dataset
        """
        pass






class DatasetABC(Dataset, abc.ABC):

    config: DatasetConfigABC
    requested_years: list[int] | tuple[int, ...] | np.ndarray
    mask: xr.DataArray | None 
    return_metadata: bool 
    load: bool 
    model_dataset: xr.DataArray | None
    observation_dataset: xr.DataArray | None
    condition_dataset: xr.DataArray | None

    def __init__(self):

        self._check_init()
        self._resolve_mask()
        self._prepare_sampling_mask(self._sampling_selectors)

        self.model_dataset = None
        self.condition_dataset = None
        self.observation_dataset = None

        if self._load_model:
            self.model_dataset = self._load_xarray_data(self.config.model,
                                                        load = self.load)

        if self.config.effective_condition is not None:
            self.condition_dataset = self._load_xarray_data(self.config.effective_condition,
                                                            load = self.load)

        self.sample_coords = self.get_sampling_coords()
        self.model_indexes = self.get_model_indexes(self.sample_coords)
        self.cond_indexes = self.get_cond_indexes(self.sample_coords)


    @final
    def _check_init(self):

        if not self.config._fitted_preprocessors:
            raise RuntimeError(
                "Make sure to fit preprocessors first!. Hint:  TrainDatasetConfig._fit_preprocessors()"
            )
        if not set(self.requested_years).issubset(
            set(self.config.available_times)
        ):
            raise ValueError(
                "the requested years are not common to input and target data."
            )

    @final
    def _resolve_mask(self):
        
        if self.mask is None:

            mask = _create_train_mask(
                time=self.config.available_times,
                lead_times=np.arange(1, self.config.effective_input.info.sizes["lead_time"] + 1),
            )
            self.mask = xr.full_like(mask, fill_value=False)               


        missing = set(required_sample_dimensions) - self.mask.dims

        if missing:
            raise ValueError(f"The mask must have {required_sample_dimensions} dims. Current dims: {missing}")    

    @property
    def _sampling_selectors(self) -> dict:

        time_dim, lead_time_dim = required_sample_dimensions
        return {
        time_dim : self.requested_years,
        lead_time_dim : self.config.lead_months
        }

    @property
    @abc.abstractmethod
    def _load_model(self) -> bool:

        pass

    @property
    @abc.abstractmethod
    def _write_condition_to_input(self):

        pass

    @property
    @abc.abstractmethod
    def _concat_condition_to_input(self):

        pass

    @final
    def _prepare_sampling_mask(self, selectors: dict):


        missing = set(required_sample_dimensions) - selectors.keys()

        if missing:
            raise ValueError(f"No selectors provided for dimensions: {missing}")

        mask = self.mask.sel({
            dim: selectors[dim]
            for dim in required_sample_dimensions
        })

        for dim in optional_sample_dimensions:
            if dim not in self.config.effective_input.info.coords:
                continue

            if dim == "ensembles" and self.config.effective_input.ensemble_mean:
                continue

            coords = self.config.effective_input.info.coords[dim]

            mask = mask.expand_dims({dim: coords}, axis=0)
        

        self.mask = mask.where(~mask)

        return self

    @final
    def _load_xarray_data(self, 
                          config: DataConfigABC, 
                          load: bool = False):
        """
        Load dataset from xarray sources.

        Returns
        -------
        xr.DataArray
        """

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

    @final
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

    @final
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

    @final
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
            dim: np.asarray(values)
            for dim, values in sample_coords.items()
            if dim in self.condition_dataset.dims
            and dim != "ensembles"
        }

        if self.config.condition_method == "same_member":
            if "ensembles" not in sample_coords:
                raise ValueError(
                    "'same_member' conditioning requires ensemble coordinates."
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

    @final
    def get_input_shape(self) -> tuple:
        """
        Determine input shape.

        Returns
        -------
        tuple
        """

        from cccma_ppp.preprocessing.utils_preprocessing import Flattennanremove

        checklist = [
            isinstance(item, Flattennanremove)
            for item in self.config.effective_input.preprocessing_pipeline.fitted_preprocessors
        ]

        len_names = len(self.config.effective_input.names)
        if self._concat_condition_to_input:
            len_names += len(self.config.effective_condition.names)

        if any(checklist):
            return (
                self.config.effective_input.preprocessing_pipeline.get_preprocessors(
                    "flattener"
                ).final_locations.size
                * len_names,
            )

        else:
            return tuple(
                self.config.effective_input.info.coords[dim].size 
                for dim in supported_NN_dimensions_sorted  
                if dim in self.config.effective_input.info.coords)
                
            
    @final
    def get_added_features_dim(self):
        """
        Number of additional features.

        Returns
        -------
        int
        """

        return (
            0 if self.config.time_features is None else len(self.config.time_features)
        )
    
    @final
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
        

    @final
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

    @final
    def __len__(self):
        """
        Dataset length.

        Returns
        -------
        int
        """
        return len(next(iter(self.sample_coords.values())))

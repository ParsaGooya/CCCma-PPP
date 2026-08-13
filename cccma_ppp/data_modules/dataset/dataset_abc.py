import abc
from typing import ClassVar, final
import dataclasses
import numpy as np
import datetime
import cftime
from torch.utils.data import Dataset
import xarray as xr
import dask
from collections.abc import Sequence

from cccma_ppp.data_modules.data.data_configs import (
    ModelDataConfig,
    ConditionDataConfig,
)

from cccma_ppp.configs import (
    lead_time_resolution,
    supported_NN_dimensions_sorted,
    required_sample_dimensions,
    realization_dim,
)

from cccma_ppp.data_modules.utils import (
    _validate_time_sequence,
    _unwrap_data_variables,
    _create_train_mask,
    suppress_stderr,
    add_lead_times
)

init_time_dim, lead_time_dim = required_sample_dimensions
@dataclasses.dataclass
class lead_time_config:
    """
    Configuration for selecting lead times.

    Parameters
    ----------
    list_lead_times : list of int or None, optional
        Explicit list of lead times.
    start : int, optional
        Start of lead time range (inclusive).
    end : int or None, optional
        End of lead time range (inclusive).
    """

    list_lead_times: list | None = None
    start: int = 1
    end: int = None

    def __post_init__(self):
        """
        Validate lead time configuration.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If neither list nor range is properly specified.
        """
        if self.list_lead_times is None:
            if self.end is None:
                raise ValueError(
                    "Provide a list of lead_times to train on,"
                    "or specify the start-end pair to choose a slice."
                )

    def build_lead_times(self):
        """
        Construct lead times array.

        Returns
        -------
        np.ndarray or list
            Lead times defined either explicitly or as a range.
        """
        return self.list_lead_times or np.arange(self.start, self.end + 1)


class DatasetConfigABC(abc.ABC):
    """
    Abstract base class for dataset configuration.

    Attributes
    ----------
    model : ModelDataConfig or None
    condition : ConditionDataConfig or None
    condition_method : str or None
    lead_times : lead_times_config or None
    """

    _VALID_CONDITION_METHODS: ClassVar[frozenset[str]] = frozenset(
        {"ensemble_mean", "cross_ensemble", "same_member", "static"}
    )

    model: ModelDataConfig | None
    condition: ConditionDataConfig | None
    condition_method: str | None
    lead_times: lead_time_config | None
    _effective_condition: ConditionDataConfig | ModelDataConfig | None

    init_time_dim: ClassVar[str] = init_time_dim
    lead_time_dim: ClassVar[str] = lead_time_dim
    realization_dim: ClassVar[str] = realization_dim
    lead_time_resolution: ClassVar[str] = lead_time_resolution 
    supported_NN_dimensions: ClassVar[tuple] = supported_NN_dimensions_sorted

    def __init__(self):
        """
        Initialize dataset configuration.

        Returns
        -------
        None
        """
        self._fitted_preprocessors: bool = False
        self._effective_condition: ConditionDataConfig | ModelDataConfig | None = None

        self._check_required_input_source()
        self._check_condition_method()
        self._check_model_vs_condition()

        self._resolve_lead_times()
        self._resolve_condition()

        self._check_model()
        self._check_condition()

        if self.lead_times is None:
            self.lead_times = self.input_lead_times
        if not set(self.lead_times).issubset(set(self.input_lead_times)):
            raise ValueError(
                f"The requested lead times are not available: must be in {self.input_lead_times}"
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
            If condition data and model data do not have the same cftime/datetime type coords for init_time_dim.
        ValueError
            If condition data does not span the same time range as model data for non static condition methods.
        ValueError
            If condition data does not provide sufficient lead-time coverage for non static condition methods.
        ValueError
            If condition data and model data do not have similar ensemble members for same_member condition methods.
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
            if self.condition_method.lower() != "static":
                for dim in [
                    dim
                    for dim in self.model.coords
                    if dim in (self.init_time_dim, self.lead_time_dim)
                ]:
                    if self.condition.coords.get(dim) is None:
                        raise ValueError(
                            "Condition data should be available"
                            f" on the same {dim} dimestions as model data."
                        )

                    if not set(self.model.coords[dim].values).issubset(
                        set(self.condition.coords[dim].values)
                    ):
                        raise ValueError(
                            "Condition data should be available"
                            f" on the same {dim} coordinates as model data."
                        )
                
                if self.model.info.time_coords_type != self.condition.info.time_coords_type:

                    raise ValueError(
                        "Condition data and model data must have the same"
                        f" cftime/datetime type time coordinates."
                    )           

            if self.condition_method.lower() == "same_member":
                if any(
                    [
                        self.model.coords.get(self.realization_dim) is None,
                        self.effective_condition.coords.get(self.realization_dim) is None,
                    ]
                ):
                    raise ValueError(
                        f"Condition data and model data must have {self.realization_dim} "
                        "dims and coords."
                    )

                if not self.model.coords[self.realization_dim].equals(
                    self.condition.coords[self.realization_dim]
                ):
                    raise ValueError(
                        "Condition data should have the same ensemble members"
                        "as model data for same_member conditioning."
                    )

            if getattr(self, "observation", None) is not None:
                for dim in [
                    dim
                    for dim in self.model.coords
                    if dim in self.supported_NN_dimensions
                ]:
                    if self.condition.coords.get(dim, None) is None:
                        raise ValueError(
                            "model and condition data must have the same NN dims."
                            / "when bias correcting to observations"
                        )

                    if not self.condition.coords.get(dim).equals(
                        self.model.coords.get(dim)
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
            if self.condition_method.lower() not in self._VALID_CONDITION_METHODS:
                raise ValueError(
                    f"Invalid condition_method: {self.condition_method}. "
                    f"Must be a in {sorted(self._VALID_CONDITION_METHODS)}."
                )
        return self

    def _check_model(self):
        if self.model is not None:
            if self.condition_method.lower() == "same_member":
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

            if self.condition_method.lower() in ["cross_ensemble", "same_member"]:
                if self.effective_condition.ensemble_mean:
                    raise ValueError(
                        "condition ensemble_mean cannot be True for cross_ensemble or same_member conditioning."
                    )
                if self.effective_condition.coords.get(self.realization_dim) is None:
                    raise ValueError(
                        f"For cross_ensemble or same_member conditioning a {self.realization_dim} dim must exist in the condition."
                    )
            elif self.condition_method.lower() == "ensemble_mean":
                if self.effective_condition.ensemble_mean is not True:
                    raise ValueError(
                        "Ensemble mean must be True for ensemble_mean conditioning."
                    )
            else:
                if self.effective_condition.realization_list is not None:
                    raise ValueError(
                        'For "static" conditioning fields cannot specify realization list.'
                    )
                if self._using_model_data_as_condition:
                    raise ValueError(
                        "'static' conditioning method cannot point to the same model data!"
                    )

            if self.condition_method.lower() == "static":
                checklist = [
                    dim in self.effective_condition.coords
                    for dim in ((self.init_time_dim, self.lead_time_dim, self.realization_dim))
                ]
                if any(checklist):
                    raise ValueError(
                        "For static condition method the condition dataset cannot have"
                        f"any of the sampling dimensions and coords "
                        f"{((self.init_time_dim, self.lead_time_dim, self.realization_dim))}"
                    )

        elif self.condition_method is not None:
            if self.condition_method.lower() == "static":
                raise ValueError(
                    "For static conditioning method condition dataset must be specified!"
                )

        return self

    @final
    def _resolve_lead_times(self):
        """
        Resolve lead time configuration.

        Returns
        -------
        None
        """
        if self.lead_times is not None and isinstance(
            self.lead_times, lead_time_config
        ):
            self.lead_times = self.lead_times.build_lead_times()

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
    def input_lead_times(self) -> int:
        """
        Coords of lead times in input dataset.

        Returns
        -------
        int
        """
        return self.effective_input.coords[self.lead_time_dim].values

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
            return self.condition_method.lower() in {
                "ensemble_mean",
                "cross_ensemble",
                "same_member",
            }

        elif self.model is not None:
            return (
                self.condition.paths == self.model.paths
                and self.condition.names == self.model.names
                and self.condition.realization_list == self.model.realization_list
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
        ensemble_mean = self.condition_method.lower() == "ensemble_mean"
        return ModelDataConfig(
            paths=self.model.paths,
            names=self.model.names,
            preprocessing_pipeline=self.model.preprocessing_pipeline,
            realization_list=self.model.realization_list,
            concat_dim=self.model.concat_dim,
            file_type=self.model.file_type,
            ensemble_mean=ensemble_mean,
            rename_dict=self.model.rename_dict,
        )

    @final
    def get_input_times(self, requested_times: (Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
            | np.ndarray
            | xr.DataArray
        )
    ):
        """
        Select the available input times within the requested_times

        Parameters
        ----------
        requested_times : array-like
            Training times used for fitting. Values must be either NumPy
            datetime64 or cftime datetime objects.

        Returns
        -------
        xr.DataArray containing input times within requested_times.           
        """
        missing = [
            t for t in requested_times.values
            if t not in self.available_times
        ]

        if missing:
            raise ValueError(
                f"The following requested_times are unavailable: {missing}"
            )

        if not isinstance(requested_times, xr.DataArray):
            requested_times = xr.DataArray(
                requested_times,
                dims=(self.init_time_dim,),
                coords={self.init_time_dim: requested_times},
            )
        
        input_times = self.effective_input.coords[self.init_time_dim].to_index()
        return requested_times.sel(
            {self.init_time_dim: requested_times.to_index().intersection(input_times)}
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


@dataclasses.dataclass
class AddedTimeFeatures:
    reference_config: DatasetConfigABC
    time_features: list[str] | None = None

    def __post_init__(self):

        self.time_features_array: np.ndarray | None = None
        self.lead_time_resolution = self.reference_config.lead_time_resolution
        self.init_time_dim = self.reference_config.init_time_dim
        self.lead_time_dim = self.reference_config.lead_time_dim

        self.min_time_ref = self.reference_config.get_common_time.min()
        self.max_time_ref = self.reference_config.get_common_time.max()
        self.time_span_ref =  self.max_time_ref - self.min_time_ref 
        
        self.feature_indices = {
            self.init_time_dim: 0,
            self.lead_time_dim: 1,
            "month_sin": 2,
            "month_cos": 3,
            "day_sin": 4,
            "day_cos": 5,
        }

        requested_features = tuple(self.time_features or ())

        unsupported = set(requested_features) - set(self.feature_indices)

        if unsupported:
            raise ValueError(
                f"Unsupported time features: {unsupported}. "
                f"Supported features are: {set(self.feature_indices)}"
            )

        self.time_features = tuple(
            feature
            for feature in self.feature_indices
            if feature in requested_features
        )


    @staticmethod
    def _days_in_year(
        time: np.datetime64 | datetime.datetime | cftime.datetime,
    ) -> int:
        """Return the number of days in the timestamp's calendar year."""
        if isinstance(time, cftime.datetime):
            calendar = time.calendar

            if calendar == "360_day":
                return 360

            if calendar in {"noleap", "365_day"}:
                return 365

            if calendar in {"all_leap", "366_day"}:
                return 366

            return 366 if time.is_leap_year else 365

        year = int(xr.DataArray(time).dt.year.item())

        return (
            np.datetime64(f"{year + 1}-01-01")
            - np.datetime64(f"{year}-01-01")
        ).astype("timedelta64[D]").astype(int)


    def build_time_features(
        self,
        sample_coords: dict[str, np.ndarray],
    ) -> "AddedTimeFeatures":
        """
        Precompute temporal features for every sampled coordinate pair.

        The resulting array has shape:

            (n_samples, n_selected_features)
        """

        required_dims = {
            self.init_time_dim,
            self.lead_time_dim,
        }

        missing = required_dims - sample_coords.keys()

        if missing:
            raise ValueError(
                "The provided sample coordinates are missing required "
                f"dimensions: {missing}."
            )

        init_times = np.asarray(
            sample_coords[self.init_time_dim]
        )
        lead_times = np.asarray(
            sample_coords[self.lead_time_dim]
        )
        
        if not self.time_features:
            return self

        requested = set(self.time_features)
        target_times = None
        target_time_da = None
        calculated_features: dict[str, np.ndarray] = {}

        if requested:
            target_times = add_lead_times(
                init_times=init_times,
                lead_times=lead_times,
                lead_time_resolution=self.lead_time_resolution,
            )

        if self.init_time_dim in requested:
            normalized_times = np.asarray(
                [
                    (time - self.min_time_ref) / self.time_span_ref
                    for time in target_times
                ],
                dtype=np.float32,
            )

            calculated_features[self.init_time_dim] = normalized_times
                
        if self.lead_time_dim in requested:
            
            normalized_lead_times = (
                lead_times.astype(np.float32)
                / float(np.max(self.reference_config.lead_times))
            )

            calculated_features[self.lead_time_dim] = normalized_lead_times

        if bool(
            requested
            & {"month_sin", "month_cos","day_sin", "day_cos"}
        ):   
 
        
            target_time_da = xr.DataArray(
                target_times,
                dims=("sample",),
            )

        if requested & {"month_sin", "month_cos"}:
            
            target_month = np.asarray(
                target_time_da.dt.month.values,
                dtype=np.float32,
            )
            
            if "month_sin" in requested:
                calculated_features["month_sin"] = np.sin(2 * np.pi * (target_month - 1) / 12.0)
            if "month_cos" in requested:
                calculated_features["month_cos"] = np.cos(2 * np.pi * (target_month - 1) / 12.0)           

        if requested & {"day_sin", "day_cos"}:

            target_days = np.asarray(
                target_time_da.dt.dayofyear.values,
                dtype=np.float32,
            )

            days_in_year = np.asarray(
                [
                    self._days_in_year(time)
                    for time in target_times
                ],
                dtype=np.float32,
            )
            if "day_sin" in requested:
                calculated_features["day_sin"] = np.sin(2 * np.pi * (target_days - 1) / days_in_year)
            if "day_cos" in requested:
                calculated_features["day_cos"] = np.cos(2 * np.pi * (target_days - 1) / days_in_year)

        self.time_features_array = np.stack(
            [
                calculated_features[feature]
                for feature in self.time_features
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

        return self
        
    
    def __call__(
        self,
        ind: int,
        input: xr.DataArray,
    ) -> np.ndarray | None:

        if self.time_features is None:
            return

        if self.time_features_array is None:
            raise RuntimeError(
                "Time-feature indexes must be built before indexing. "
                "Call 'build_indexes(sample_coords)' first."
            )
        if not 0 <= ind < len(self.time_features_array):
            raise IndexError(
                f"Time-feature index {ind} is out of bounds for "
                f"{len(self.time_features_array)} samples."
            )
            
        time_features = self.time_features_array[ind]

        if input.ndim > 2:
            time_features = np.broadcast_to(
                time_features[(...,) + (None,) * (input.ndim - 1)],
                (time_features.shape[0],) + input.shape[1:],
            ).copy()

        return time_features

    def __len__(self):
        return len(self.time_features)

    def __eq__(self, other):
        if not isinstance(other, AddedTimeFeatures):
            return NotImplemented

        return (
            all(self.reference_config.lead_times == other.reference_config.lead_times)
            and all(
                self.reference_config.get_common_time
                == other.reference_config.get_common_time
            )
            and (type(self.reference_config) is type(other.reference_config))
            and (self.time_features == other.time_features)
        )


class DatasetABC(Dataset, abc.ABC):
    config: DatasetConfigABC
    requested_times: (
        Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
        | np.ndarray
        | xr.DataArray
    )
    mask: xr.DataArray | None
    time_features: AddedTimeFeatures
    return_metadata: bool
    load: bool

    def __init__(self):

        self._check_init()
        self._resolve_mask()
        self._prepare_sampling_mask(self._sampling_times_selectors)

        if self._load_model:
            self.config.model.open_xarray_data(
                load=self.load, add_time_auxiliary_coords=True
            )

        if self.config.effective_condition is not None:
            self.config.effective_condition.open_xarray_data(
                load=self.load, add_time_auxiliary_coords=True
            )

        self.sample_coords = self.get_sampling_coords()

        self.model_indexes = self.get_model_indexes(self.sample_coords)
        self.cond_indexes = self.get_cond_indexes(self.sample_coords)

        self.time_features = dataclasses.replace(self.time_features)
        self.time_features.build_time_features(self.sample_coords)


    @final
    def _check_init(self):

        _validate_time_sequence(self.requested_times)

        if not self.config._fitted_preprocessors:
            raise RuntimeError(
                "Make sure to fit preprocessors first!. Hint:  TrainDatasetConfig._fit_preprocessors()"
            )
 
        missing = [
            t for t in self.requested_times.values
            if t not in self.config.available_times
        ]

        if missing:
            raise ValueError(
                f"The following requested initialization times are unavailable: {missing}"
            )

    @final
    def _resolve_mask(self):

        if self.mask is None:
            mask = _create_train_mask(
                init_times=self.config.available_times,
                lead_times=self.config.input_lead_times,
                lead_time_resolution=lead_time_resolution
            )
            self.mask = xr.full_like(mask, fill_value=False)

        missing = set((self.config.init_time_dim, self.config.lead_time_dim)) - set(self.mask.dims)

        if missing:
            raise ValueError(
                f"The mask must have {(self.config.init_time_dim, self.config.lead_time_dim)} dims. Current dims: {missing}"
            )

    @property
    def _sampling_times_selectors(self) -> dict:

        return {self.config.init_time_dim: self.config.get_input_times(self.requested_times), self.config.lead_time_dim: self.config.lead_times}

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
    def _prepare_sampling_mask(self, sampling_times_selectors: dict):

        missing = set((self.config.init_time_dim, self.config.lead_time_dim)) - sampling_times_selectors.keys()

        if missing:
            raise ValueError(f"No selectors provided for dimensions: {missing}")

        mask = self.mask.sel(
            {dim: sampling_times_selectors[dim] for dim in (self.config.init_time_dim, self.config.lead_time_dim)}
        )

        if (not self.config.effective_input.ensemble_mean
            and self.config.realization_dim in self.config.effective_input.coords):
                
            coords = self.config.effective_input.coords[self.config.realization_dim]

            mask = mask.expand_dims({self.config.realization_dim: coords}, axis=0)

        self.mask = mask.where(~mask)

        return self

    @final
    def get_sampling_coords(self):
        """
        Compute coordinates for sampling the datasets.

        Returns
        -------
        dict
        """

        sample_dims = tuple(self.mask.sizes)

        stacked_mask = (
            self.mask
            .stack(batch=sample_dims)
            .transpose("batch", ...)
            .dropna(dim="batch")
        )

        return {
            dim: np.asarray(stacked_mask.coords[dim].values)
            for dim in sample_dims
        }

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
            dim: self.config.model.indexes[dim].get_indexer(values)
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
            self.config.effective_condition is None
            or self.config.condition_method.lower() == "static"
        ):
            return None

        condition_coords = {
            dim: np.asarray(values)
            for dim, values in sample_coords.items()
            if dim in self.config.effective_condition.dims and dim != self.config.realization_dim 
        }

        if self.config.condition_method.lower() == "same_member":
            if self.config.realization_dim not in sample_coords:
                raise ValueError(
                    f"'same_member' conditioning requires {self.config.realization_dim} coordinates."
                )

            condition_coords[self.config.realization_dim] = np.asarray(sample_coords[self.config.realization_dim])

        indexes = {
            dim: self.config.effective_condition.indexes[dim].get_indexer(values)
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
            in_shape = (
                self.config.effective_input.preprocessing_pipeline.get_preprocessors(
                    "flattener"
                ).final_locations.shape
            )

        else:
            in_shape = tuple(
                self.config.effective_input.coords[dim].size
                for dim in self.config.supported_NN_dimensions
                if dim in self.config.effective_input.coords
            )

        return tuple([len_names, *in_shape])

    @final
    def get_added_features_dim(self):

        return len(self.time_features)

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
        if self.config.effective_condition is None:
            return None

        if self.config.condition_method.lower() == "static":
            selection = {}

        else:
            selection = {
                dim: [indexes[ind]] for dim, indexes in self.cond_indexes.items()
            }

            if self.config.condition_method.lower() == "cross_ensemble":
                selection[self.config.realization_dim] = [
                    np.random.randint(self.config.effective_condition.sizes[self.config.realization_dim])
                ]

        condition = self.config.effective_condition.isel(**selection)
        
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
            dim: [int(indexes[ind])] for dim, indexes in self.model_indexes.items()
        }

        model = self.config.model.isel(**selection)

        return _unwrap_data_variables(model)

    @staticmethod
    def _compute(*arrays) -> tuple:
        with suppress_stderr(), dask.config.set(scheduler="synchronous"):
            return dask.compute(*arrays)

    @final
    def __len__(self):
        """
        Dataset length.

        Returns
        -------
        int
        """
        return len(next(iter(self.sample_coords.values())))

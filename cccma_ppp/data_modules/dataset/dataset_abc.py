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
    add_lead_times,
)

init_time_dim, lead_time_dim = required_sample_dimensions


@dataclasses.dataclass
class lead_time_config:

    """
    Configuration for selecting forecast lead times.

    Supports either an explicit list of lead times or an inclusive range
    defined by starting and ending lead times. Lead times are one-based.

    Parameters
    ----------
    list_lead_times : list | None, default None
        Explicit list of lead times to select. If provided, `start` and
        `end` are ignored.
    start : int, default 1
        First lead time in the selected range. Used only when
        `list_lead_times` is None.
    end : int | None, default None
        Last lead time in the selected range, inclusive. Required when
        `list_lead_times` is None.
    """

    list_lead_times: list | None = None
    start: int = 1
    end: int = None

    def __post_init__(self):

        """
        Validate the lead-time selection configuration.

        Ensures that either an explicit list of lead times or an ending
        lead time for range-based selection is provided.

        Raises
        ------
        ValueError
            If neither `list_lead_times` nor `end` is specified.
        """

        if self.list_lead_times is None:
            if self.end is None:
                raise ValueError(
                    "Provide a list of lead_times to train on,"
                    "or specify the start-end pair to choose a slice."
                )

    def build_lead_times(self):

        """
        Construct the selected forecast lead times.

        Returns the explicitly specified lead times, if available.
        Otherwise, generates an inclusive sequence from `start` to `end`.

        Returns
        -------
        list | np.ndarray
            Selected lead times. Returns the original list when
            `list_lead_times` is provided.
        """

        if self.list_lead_times is not None:
            return self.list_lead_times
        return np.arange(self.start, self.end + 1)


class DatasetConfigABC(abc.ABC):

    """
    Abstract base class for configuring pipeline datasets.

    Defines the shared interface for selecting model and conditioning data,
    resolving conditioning methods, validating compatibility between data
    sources, and selecting forecast lead times. Train and Inference datasets 
    inherit from this class to implement specific data-loading.

    Supports ensemble-mean, cross-ensemble, same-member, and static
    conditioning. Model data can also serve as the conditioning source
    when a separate conditioning dataset is not provided.

    Subclasses must implement `available_times`, `ds_operator`,
    `effective_input`, and `build_dataset`. The following attributes
    must be defined in subclasses including both Train and Inference pipelines:

    Attributes
    ----------
    model : ModelDataConfig | None
        Configuration for the model prediction data. Could only be 
        None for Inference pipeline where the
        conditioning dataset must provide the required input.
    condition : ConditionDataConfig | None
        Optional configuration for auxiliary conditioning data.
        If None, model data may be reused as conditioning data,
        depending on the selected conditioning method.
    condition_method : str | None
        Method used to associate conditioning data with model inputs.
        Supported methods are "ensemble_mean", "cross_ensemble",
        "same_member", and "static".
    lead_times : lead_time_config | list | np.ndarray | None
        Requested forecast lead times. Accepts a lead-time configuration
        or an explicit sequence. If None, all available input lead times
        are selected.
    """


    _VALID_CONDITION_METHODS: ClassVar[frozenset[str]] = frozenset(
        {"ensemble_mean", "cross_ensemble", "same_member", "static"}
    )

    model: ModelDataConfig | None
    condition: ConditionDataConfig | None
    condition_method: str | None
    lead_times: lead_time_config | None

    init_time_dim: ClassVar[str] = init_time_dim
    lead_time_dim: ClassVar[str] = lead_time_dim
    realization_dim: ClassVar[str] = realization_dim
    lead_time_resolution: ClassVar[str] = lead_time_resolution
    supported_NN_dimensions: ClassVar[tuple] = supported_NN_dimensions_sorted

    def __init__(self):

        """
        Initialize the dataset configuration and validate its input sources.

        Verifies that at least one input source is available, validates the
        conditioning method, and resolves the effective conditioning dataset.
        At least one of the model or conditioning datasets must be provided. 
        If both are provided, the conditioning method must be compatible.

        Checks compatibility between model and conditioning data, including
        temporal coordinates, ensemble realizations, and spatial dimensions
        where applicable.

        Resolves the requested lead times and verifies that they are available
        in the effective input dataset.

        Raises
        ------
        ValueError
            If no input source is provided, the conditioning configuration
            is invalid, model and conditioning data are incompatible, or
            the requested lead times are unavailable.
        """

        self._fitted_preprocessors: bool = False
        self._effective_condition: ConditionDataConfig | ModelDataConfig | None = None

        self._check_required_input_source()
        self._check_condition_method()
       
        self._resolve_condition()
        self._check_model_vs_condition()

        self._resolve_lead_times()
        
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
        Verify that at least one input data source is configured.

        Returns
        -------
        DatasetConfigABC
            The current dataset configuration instance.

        Raises
        ------
        ValueError
            If both `model` and `condition` are None.
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
        Validate compatibility between model and conditioning datasets.

        When separate model and conditioning datasets are provided,
        verifies that non-static conditioning data contain all required
        model initialization-time and lead-time dimensions and coordinates and 
        use the same datetime or cftime representation.

        For non ensemble-mean conditioning, ensures that the effective conditioning 
        dataset contains realization dimension and coordinates.

        For same-member conditioning, verifies that both datasets have
        identical ensemble realization coordinates.

        When observational targets is available, also requires matching
        neural-network dimensions and coordinates between model and
        conditioning datasets to be able to concatenate them for bias correction.

        Raises
        ------
        ValueError
            If required temporal coordinates are missing or incompatible,
            same-member conditioning uses different ensemble realizations,
            or spatial coordinates differ when observational targets
            are provided.
        """

        if all(
            [
                self.condition is not None,
                self.model is not None,
                not self._using_model_data_as_condition,
            ]
        ):
            if self.effective_condition_method != "static":
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

                if (
                    self.model.info.time_coords_type
                    != self.condition.info.time_coords_type
                ):
                    raise ValueError(
                        "Condition data and model data must have the same"
                        " cftime/datetime type time coordinates."
                    )

            if self.effective_condition_method == "same_member":
                if any(
                    [
                        self.model.coords.get(self.realization_dim) is None,
                        self.effective_condition.coords.get(self.realization_dim)
                        is None,
                    ]
                ):
                    raise ValueError(
                        f"Condition data and model data must have {self.realization_dim} "
                        "dims and coords for same_member conditioning."
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
                            "model and condition data must have the same NN dims. "
                            "when bias correcting to observations"
                        )

                    if not self.condition.coords.get(dim).equals(
                        self.model.coords.get(dim)
                    ):

                        raise ValueError(
                            f"model and condition data do not have the same {dim} cooridnates. "
                            f"when bias correcting to observations. Got {self.condition.coords.get(dim)} "
                            f"vs {self.model.coords.get(dim)}"
                        )

    @final
    def _check_condition_method(self):

        """
        Validate the selected conditioning method.

        Returns
        -------
        DatasetConfigABC
            The current dataset configuration instance.

        Raises
        ------
        ValueError
            If the effective conditioning method is not None and is not
            one of "ensemble_mean", "cross_ensemble", "same_member",
            or "static".
        """

        if self.effective_condition_method is not None:
            if self.effective_condition_method not in self._VALID_CONDITION_METHODS:
                raise ValueError(
                    f"Invalid condition_method: {self.effective_condition_method}. "
                    f"Must be a in {sorted(self._VALID_CONDITION_METHODS)}."
                )
        return self

    def _check_model(self):

        """
        Validate model configuration requirements for the conditioning method.

        Returns
        -------
        DatasetConfigABC
            The current dataset configuration instance.

        Raises
        ------
        ValueError
            If same-member conditioning is requested while the model
            configuration specifies ensemble averaging.
        """

        if self.model is not None:
            
            if self.effective_condition_method == "same_member":
                if self.model.ensemble_mean:
                    raise ValueError(
                        "for same member coniditioning the model data should not be ensemble mean."
                    )

        return self

    def _check_condition(self):

        """
        Validate the effective conditioning data and its conditioning method.

        Checks that the conditioning dataset is compatible with the selected
        conditioning method. Cross-ensemble and same-member conditioning
        require individual ensemble realizations, whereas ensemble-mean
        conditioning requires ensemble averaging.

        Static conditioning requires an explicitly provided conditioning
        dataset without initialization-time, lead-time, or realization
        coordinates.

        Returns
        -------
        DatasetConfigABC
            The current dataset configuration instance.

        Raises
        ------
        ValueError
            If the conditioning method is missing, the ensemble
            configuration is incompatible with the selected method,
            or static conditioning violates its data requirements.
        """

        if self.effective_condition is not None:
            if self.effective_condition_method is None:
                raise ValueError(
                    "You must specify condition_method for conditioning dataset!"
                )

            if self.effective_condition_method in ["cross_ensemble", "same_member"]:
                if self.effective_condition.ensemble_mean:
                    raise ValueError(
                        "condition ensemble_mean cannot be True for cross_ensemble or same_member conditioning."
                    )
                if self.effective_condition.coords.get(self.realization_dim) is None:
                    raise ValueError(
                        f"For cross_ensemble or same_member conditioning a {self.realization_dim} dim must exist in the condition."
                    )
            elif self.effective_condition_method == "ensemble_mean":
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

            if self.effective_condition_method == "static":
                checklist = [
                    dim in self.effective_condition.coords
                    for dim in (
                        (self.init_time_dim, self.lead_time_dim, self.realization_dim)
                    )
                ]
                if any(checklist):
                    raise ValueError(
                        "For static condition method the condition dataset cannot have"
                        f"any of the sampling dimensions and coords "
                        f"{((self.init_time_dim, self.lead_time_dim, self.realization_dim))}"
                    )

        elif self.effective_condition_method is not None:
            if self.effective_condition_method == "static":
                raise ValueError(
                    "For static conditioning method condition dataset must be specified!"
                )

        return self

    @final
    def _resolve_lead_times(self):

        """
        Resolve the requested lead-time configuration.

        If `lead_times` is a `lead_time_config` instance, replaces it
        with the sequence returned by its `build_lead_times` method.
        Otherwise, leaves the existing lead-time selection unchanged.
        """

        if self.lead_times is not None and isinstance(
            self.lead_times, lead_time_config
        ):
            self.lead_times = self.lead_times.build_lead_times()

    @property
    @abc.abstractmethod
    def available_times(self):

        """
        Return the time span available for dataset construction.
        This is different than input time because it should be the 
        common time range resolved by the forecast and if available,
        the observation data. Forecasts range is based on initialization 
        time and lead time, while observation data has one time resolution.

        For instance, for decadal predictions, the initial times are the 
        start of the years, but the available times are the span of coverage 
        based on initial_time / lead_time span which has a monthly resolution.

        Subclasses must implement this property to determine which
        initialization times satisfy their input and target data
        requirements.
        """
        pass

    @property
    @abc.abstractmethod
    def ds_operator(self):

        """
        Return the dataset-specific operator.

        Subclasses must implement this property to provide the operator
        used for their dataset construction and data access.
        
        Recommended implementaion is 
        from cccma_ppp.data_modules.dataset.operator import DatasetOperator.
        """
        pass

    @property
    def input_lead_times(self) -> np.ndarray:

        """
        Return the forecast lead times available in the effective input.

        Returns
        -------
        np.ndarray
            Lead-time coordinate values of the effective input dataset.
        """

        return self.effective_input.coords[self.lead_time_dim].values

    @property
    @abc.abstractmethod
    def effective_input(self) -> ConditionDataConfig | ModelDataConfig | None:

        """
        Return the input data that effectively goes into the dataset.
        For Inference and Train this shall differ based on what is 
        provided in the config.

        Subclasses must implement this property to identify the data
        source used as the primary input.

        Returns
        -------
        ConditionDataConfig | ModelDataConfig | None
            Effective input configuration.
        """

        pass

    @final
    @property
    def _using_model_data_as_condition(self) -> bool:

        """
        Determine whether the same model data are used as the conditioning source.

        When no separate conditioning dataset is provided, model data
        are reused for ensemble-mean, cross-ensemble, or same-member
        conditioning. 

        When both configurations are provided, compares their paths,
        variable names, and realization selections. If those point to 
        the same data, the model data are considered to be used as the conditioning source.

        Returns
        -------
        bool
            True if model data are used as conditioning data;
            otherwise, False.
        """

        if self.condition is None:
            
            return self.effective_condition_method in {
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
        Return the resolved conditioning data. 
        Resolved at __init__ time based on the provided 
        model and condition configs.

        Returns
        -------
        ConditionDataConfig | ModelDataConfig | None
            Explicit or model-derived conditioning configuration,
            or None if no conditioning dataset is required.
        """

        return self._effective_condition

    @final
    @property
    def effective_condition_method(self) -> str:

        """
        Return the normalized conditioning method.

        Returns
        -------
        str | None
            Lowercase conditioning method, or None if no method
            is specified.
        """

        return (None 
                if self.condition_method is None 
                else self.condition_method.lower())

    @final
    def _model_as_condition(self) -> ModelDataConfig:

        """
        Create a data object from the model dataset to be used as condition.

        Constructs a new `ModelDataConfig` using the existing model's
        data sources, variable selection, preprocessing pipeline,
        ensemble realizations, and file configuration.

        Ensemble averaging is determined from if conditioning is ensemble-mean.

        Returns
        -------
        ModelDataConfig
            Model configuration adapted for use as conditioning data.
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
    def get_input_times(
        self,
        requested_times: (
            Sequence[np.datetime64 | datetime.datetime | cftime.datetime]
            | np.ndarray
            | xr.DataArray
        ),
    ):

        """
        Resolve requested initialization times against the effective input.

        Validates that all requested times are available to the dataset,
        and selects those present in the effective input's
        initialization-time coordinates to iterate over.

        The intention for this code is the dataset available temporal coverage
        is based on the model/condition/observation common time span. But, the
        exact initial times to be iterated over is not necessarily the same as the available times. 
        See documentation for `available_times` for more details.

        Parameters
        ----------
        requested_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
            Requested initialization times. Non-xarray inputs are converted
            to a DataArray indexed by the initialization-time dimension.

        Returns
        -------
        xr.DataArray
            Requested initialization times that are present in the
            effective input dataset, preserving their original order.

        Raises
        ------
        ValueError
            If any requested initialization time is not included in
            `available_times`.
        """

        if not isinstance(requested_times, xr.DataArray):
            requested_times = xr.DataArray(
                requested_times,
                dims=(self.init_time_dim,),
                coords={self.init_time_dim: requested_times},
            )
            
        missing = [t for t in requested_times.values if t not in self.available_times]

        if missing:
            raise ValueError(
                f"The following requested_times are unavailable: {missing}"
            )

        input_times = self.effective_input.coords[self.init_time_dim].to_index()
        return requested_times.sel(
            {self.init_time_dim: requested_times.to_index().intersection(input_times)}
        )

    @final
    def _resolve_condition(self):

        """
        Resolve the effective conditioning data source.

        Uses the explicitly configured conditioning dataset when available.
        Otherwise, creates a model-derived conditioning configuration if
        the selected method requires model data as conditioning input.

        If neither applies, the effective conditioning dataset is None.

        Returns
        -------
        DatasetConfigABC
            The current dataset configuration instance.
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
        Construct the dataset from this validated configuration.

        Subclasses must implement this method to assemble the required
        model, conditioning, and optional observational data according
        to their dataset-specific sampling and preprocessing procedures.
        """
        pass


@dataclasses.dataclass
class AddedTimeFeatures:
    """
    Document this class.

    Parameters
    ----------
    reference_config : DatasetConfigABC
        Description not yet provided.
    time_features : list[str] | None
        Description not yet provided.
    """

    reference_config: DatasetConfigABC
    time_features: list[str] | None = None

    def __post_init__(self):
        """
        Document this function.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        self.time_features_array: np.ndarray | None = None
        self.lead_time_resolution = self.reference_config.lead_time_resolution
        self.init_time_dim = self.reference_config.init_time_dim
        self.lead_time_dim = self.reference_config.lead_time_dim

        self.min_time_ref = self.reference_config.get_common_time.min()
        self.max_time_ref = self.reference_config.get_common_time.max()
        self.time_span_ref = self.max_time_ref - self.min_time_ref

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
            feature for feature in self.feature_indices if feature in requested_features
        )

    @staticmethod
    def _days_in_year(
        time: np.datetime64 | datetime.datetime | cftime.datetime,
    ) -> int:
        """
        Document this function.

        Parameters
        ----------
        time : np.datetime64 | datetime.datetime | cftime.datetime
            Description not yet provided.

        Returns
        -------
        int
            Description not yet provided.
        """
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
            (np.datetime64(f"{year + 1}-01-01") - np.datetime64(f"{year}-01-01"))
            .astype("timedelta64[D]")
            .astype(int)
        )

    def build_time_features(
        self,
        sample_coords: dict[str, np.ndarray],
    ) -> "AddedTimeFeatures":
        """
        Document this function.

        Parameters
        ----------
        sample_coords : dict[str, np.ndarray]
            Description not yet provided.

        Returns
        -------
        'AddedTimeFeatures'
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
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

        init_times = np.asarray(sample_coords[self.init_time_dim])
        lead_times = np.asarray(sample_coords[self.lead_time_dim])

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

            if self.time_span_ref == 0:
                raise ValueError(
                    "Cannot normalize time with zero-length reference span. "
                    "Ensure reference_config.get_common_time has range > 0."
                )

            normalized_times = np.asarray(
                [
                    (time - self.min_time_ref) / self.time_span_ref
                    for time in target_times
                ],
                dtype=np.float32,
            )

            calculated_features[self.init_time_dim] = normalized_times

        if self.lead_time_dim in requested:
            max_lead_time = float(np.max(self.reference_config.lead_times))

            if max_lead_time == 0:
                raise ValueError(
                    "Cannot normalize lead time when maximum is zero. "
                    "Ensure at least one lead time > 0."
                )
            
            normalized_lead_times = lead_times.astype(np.float32) / max_lead_time

            calculated_features[self.lead_time_dim] = normalized_lead_times

        if bool(requested & {"month_sin", "month_cos", "day_sin", "day_cos"}):
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
                calculated_features["month_sin"] = np.sin(
                    2 * np.pi * (target_month - 1) / 12.0
                )
            if "month_cos" in requested:
                calculated_features["month_cos"] = np.cos(
                    2 * np.pi * (target_month - 1) / 12.0
                )

        if requested & {"day_sin", "day_cos"}:
            target_days = np.asarray(
                target_time_da.dt.dayofyear.values,
                dtype=np.float32,
            )

            days_in_year = np.asarray(
                [self._days_in_year(time) for time in target_times],
                dtype=np.float32,
            )
            if "day_sin" in requested:
                calculated_features["day_sin"] = np.sin(
                    2 * np.pi * (target_days - 1) / days_in_year
                )
            if "day_cos" in requested:
                calculated_features["day_cos"] = np.cos(
                    2 * np.pi * (target_days - 1) / days_in_year
                )

        self.time_features_array = np.stack(
            [calculated_features[feature] for feature in self.time_features],
            axis=-1,
        ).astype(np.float32, copy=False)

        return self

    def __call__(
        self,
        ind: int,
        input: xr.DataArray,
    ) -> np.ndarray | None:
        """
        Document this function.

        Parameters
        ----------
        ind : int
            Description not yet provided.
        input : xr.DataArray
            Description not yet provided.

        Returns
        -------
        np.ndarray | None
            Description not yet provided.

        Raises
        ------
        IndexError
            Description not yet provided.
        RuntimeError
            Description not yet provided.
        """
        if not self.time_features:
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
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return len(self.time_features)

    def __eq__(self, other):
        """
        Document this function.

        Parameters
        ----------
        other : Any
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.
        """
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
    """
    Document this class.

    Attributes
    ----------
    config : DatasetConfigABC
        Description not yet provided.
    requested_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
        Description not yet provided.
    mask : xr.DataArray | None
        Description not yet provided.
    time_features : AddedTimeFeatures
        Description not yet provided.
    return_metadata : bool
        Description not yet provided.
    load : bool
        Description not yet provided.
    """

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

    def __init__(self, seed: int | None = None):
        """
        Document this function.
        """
        self._check_init()
        self._resolve_mask()
        self._prepare_sampling_mask(self._sampling_times_selectors)
        self._rng = np.random.default_rng(seed)

        if self._load_model:
            self.config.model.open_xarray_data(
                load=self.load, add_time_auxiliary_coords=True
            )

        if self.config.effective_condition is not None:
            self.config.effective_condition.open_xarray_data(
                load=self.load, add_time_auxiliary_coords= (
                    self.config.effective_condition_method != 'static'
                )
            )

        self.sample_coords = self.get_sampling_coords()

        self.model_indexes = self.get_model_indexes(self.sample_coords)
        self.cond_indexes = self.get_cond_indexes(self.sample_coords)

        self.time_features = dataclasses.replace(self.time_features)
        self.time_features.build_time_features(self.sample_coords)

    @final
    def _check_init(self):
        """
        Document this function.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        ValueError
            Description not yet provided.
        """
        _validate_time_sequence(self.requested_times)

        if not self.config._fitted_preprocessors:
            raise RuntimeError(
                "Make sure to fit preprocessors first!. Hint:  TrainDatasetConfig._fit_preprocessors()"
            )

        missing = [
            t
            for t in self.requested_times.values
            if t not in self.config.available_times
        ]

        if missing:
            raise ValueError(
                f"The following requested initialization times are unavailable: {missing}"
            )

    @final
    def _resolve_mask(self):
        """
        Document this function.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if self.mask is None:
            mask = _create_train_mask(
                init_times=self.config.available_times,
                lead_times=self.config.input_lead_times,
                lead_time_resolution=lead_time_resolution,
            )
            self.mask = xr.full_like(mask, fill_value=False)

        missing = set((self.config.init_time_dim, self.config.lead_time_dim)) - set(
            self.mask.dims
        )

        if missing:
            raise ValueError(
                f"The mask must have {(self.config.init_time_dim, self.config.lead_time_dim)} dims. Current dims: {missing}"
            )

    @property
    def _sampling_times_selectors(self) -> dict:
        """
        Document this function.

        Returns
        -------
        dict
            Description not yet provided.
        """
        return {
            self.config.init_time_dim: self.config.get_input_times(
                self.requested_times
            ),
            self.config.lead_time_dim: self.config.lead_times,
        }

    @property
    @abc.abstractmethod
    def _load_model(self) -> bool:
        """
        Document this function.
        """
        pass

    @property
    @abc.abstractmethod
    def _write_condition_to_input(self):
        """
        Document this function.
        """
        pass

    @property
    @abc.abstractmethod
    def _concat_condition_to_input(self):
        """
        Document this function.
        """
        pass

    @final
    def _prepare_sampling_mask(self, sampling_times_selectors: dict):
        """
        Document this function.

        Parameters
        ----------
        sampling_times_selectors : dict
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        missing = (
            set((self.config.init_time_dim, self.config.lead_time_dim))
            - sampling_times_selectors.keys()
        )

        if missing:
            raise ValueError(f"No selectors provided for dimensions: {missing}")

        mask = self.mask.sel(
            {
                dim: sampling_times_selectors[dim]
                for dim in (self.config.init_time_dim, self.config.lead_time_dim)
            }
        )

        if (
            not self.config.effective_input.ensemble_mean
            and self.config.realization_dim in self.config.effective_input.coords
        ):
            coords = self.config.effective_input.coords[self.config.realization_dim]

            mask = mask.expand_dims({self.config.realization_dim: coords}, axis=0)

        self.mask = mask.where(~mask)

        return self

    @final
    def get_sampling_coords(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        sample_dims = tuple(self.mask.sizes)

        stacked_mask = (
            self.mask.stack(batch=sample_dims)
            .transpose("batch", ...)
            .dropna(dim="batch")
        )

        return {dim: np.asarray(stacked_mask.coords[dim].values) for dim in sample_dims}

    @final
    def get_model_indexes(
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
        if (
            self.config.effective_condition is None
            or self.config.effective_condition_method == "static"
        ):
            return None

        condition_coords = {
            dim: np.asarray(values)
            for dim, values in sample_coords.items()
            if dim in self.config.effective_condition.dims
            and dim != self.config.realization_dim
        }

        if self.config.effective_condition_method == "same_member":
            if self.config.realization_dim not in sample_coords:
                raise ValueError(
                    f"'same_member' conditioning requires {self.config.realization_dim} coordinates."
                )

            condition_coords[self.config.realization_dim] = np.asarray(
                sample_coords[self.config.realization_dim]
            )

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
        Document this function.

        Returns
        -------
        tuple
            Description not yet provided.
        """

        len_names = len(self.config.effective_input.names)
        if self._concat_condition_to_input:
            len_names += len(self.config.effective_condition.names)

        if self.config.effective_input.preprocessing_pipeline.has_flattener:
            flattener = self.config.effective_input.preprocessing_pipeline.get_flattener()
            in_shape = (
                flattener.final_locations.shape
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
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return len(self.time_features)

    @final
    def _index_condition_dataset(self, ind: int) -> xr.DataArray | None:
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
        if self.config.effective_condition is None:
            return None

        if self.config.effective_condition_method == "static":
            selection = {}

        else:
            selection = {
                dim: [indexes[ind]] for dim, indexes in self.cond_indexes.items()
            }

            if self.config.effective_condition_method == "cross_ensemble":
                selection[self.config.realization_dim] = [
                    self._rng.integers(
                        self.config.effective_condition.sizes[
                            self.config.realization_dim
                        ]
                    )
                ]

        condition = self.config.effective_condition.isel(**selection)

        return _unwrap_data_variables(condition)

    @final
    def _index_model_dataset(self, ind: int) -> xr.DataArray | None:
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
        if not self._load_model:
            return None

        selection = {
            dim: [int(indexes[ind])] for dim, indexes in self.model_indexes.items()
        }

        model = self.config.model.isel(**selection)

        return _unwrap_data_variables(model)

    @staticmethod
    def _compute(*arrays) -> tuple:
        """
        Document this function.

        Parameters
        ----------
        *arrays : Any
            Description not yet provided.

        Returns
        -------
        tuple
            Description not yet provided.
        """
        with suppress_stderr(), dask.config.set(scheduler="synchronous"):
            return dask.compute(*arrays)

    @final
    def __len__(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return len(next(iter(self.sample_coords.values())))

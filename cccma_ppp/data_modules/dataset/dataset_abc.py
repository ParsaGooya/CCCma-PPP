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
    Configuration for selecting lead times.

    Lead times can be specified explicitly or generated as an inclusive sequence
    from ``start`` through ``end``.

    Parameters
    ----------
    list_lead_times : list | None
        Explicit lead times to select. If ``None``, lead times are generated from
        ``start`` and ``end``.
    start : int
        First lead time in the generated sequence.
    end : int | None
        Final lead time in the generated sequence.
    """

    list_lead_times: list | None = None
    start: int = 1
    end: int = None

    def __post_init__(self):
        """
        Validate the lead-time selection configuration.

        Raises
        ------
        ValueError
            If neither an explicit list of lead times nor an ending lead time is
            provided.
        """
        if self.list_lead_times is None:
            if self.end is None:
                raise ValueError(
                    "Provide a list of lead_times to train on,"
                    "or specify the start-end pair to choose a slice."
                )

    def build_lead_times(self):
        """
        Build the configured lead-time selection.

        Returns
        -------
        list | np.ndarray
            Explicitly configured lead times, or an inclusive sequence from
            ``start`` through ``end``.
        """
        if self.list_lead_times is not None:
            return self.list_lead_times
        return np.arange(self.start, self.end + 1)


class DatasetConfigABC(abc.ABC):
    """
    Abstract base configuration for PPP datasets.

    Provides common validation and resolution of model and conditioning data,
    conditioning strategies, lead-time selection, and available sampling times.

    Attributes
    ----------
    model : ModelDataConfig | None
        Model data configuration available to the dataset.
    condition : ConditionDataConfig | None
        Explicit conditioning data configuration.
    condition_method : str | None
        Method used to construct conditioning samples.
    lead_times : lead_time_config | None
        Lead-time selection used to construct dataset samples.
    _effective_condition : ConditionDataConfig | ModelDataConfig | None
        Resolved conditioning data configuration after accounting for cases where
        model data are reused as conditioning data.
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
        Initialize and validate a dataset configuration.

        Resolves the effective conditioning source and lead times, validates model and
        conditioning compatibility, and ensures all requested lead times are
        available from the effective input.

        Raises
        ------
        ValueError
            If no input source is available, the conditioning configuration is
            invalid or incompatible with the model data, or a requested lead time is
            unavailable.
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
        Validate that an input data source is configured.

        Returns
        -------
        DatasetConfigABC
            This configuration instance after validation.

        Raises
        ------
        ValueError
            If neither model nor conditioning data are provided.
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
        Validate compatibility between model and conditioning data.

        For temporally varying conditioning data, required model sampling coordinates
        must be available in the conditioning data and use a compatible time
        representation. Additional realization and neural-network dimension checks
        are performed when required by the conditioning method and dataset type.

        Raises
        ------
        ValueError
            If model and conditioning data have incompatible sampling coordinates,
            time representations, ensemble members, or neural-network dimensions.
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
                            f"when bias correcting to observations"
                        )

    @final
    def _check_condition_method(self):
        """
        Validate the configured conditioning method.

        Returns
        -------
        DatasetConfigABC
            This configuration instance after validation.

        Raises
        ------
        ValueError
            If the conditioning method is not one of the supported methods.
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
        Validate model data against the selected conditioning method.

        Returns
        -------
        DatasetConfigABC
            This configuration instance after validation.

        Raises
        ------
        ValueError
            If same-member conditioning is requested while the model data are reduced
            to an ensemble mean.
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
        Validate the effective conditioning data and method.

        Checks ensemble requirements for ensemble-based conditioning and ensures that
        static conditioning data do not contain sampling dimensions.

        Returns
        -------
        DatasetConfigABC
            This configuration instance after validation.

        Raises
        ------
        ValueError
            If conditioning data are provided without a conditioning method, ensemble
            settings are incompatible with the selected method, static conditioning
            contains sampling dimensions, or static conditioning is otherwise
            configured inconsistently.
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
        Resolve a lead-time configuration into explicit lead-time values.

        If ``lead_times`` is a ``lead_time_config`` instance, it is replaced by the
        lead-time sequence produced by that configuration.
        """
        if self.lead_times is not None and isinstance(
            self.lead_times, lead_time_config
        ):
            self.lead_times = self.lead_times.build_lead_times()

    @property
    @abc.abstractmethod
    def available_times(self):
        """
        Return the initialization times available for dataset sampling.

        Returns
        -------
        Any
            Initialization times available to the concrete dataset configuration.
        """
        pass

    @property
    @abc.abstractmethod
    def ds_operator(self):
        """
        Return the dataset-specific operation used to combine or select data.

        Returns
        -------
        Any
            Dataset operation defined by the concrete dataset configuration.
        """
        pass

    @property
    def input_lead_times(self) -> np.ndarray:
        """
        Return lead times available from the effective input data.

        Returns
        -------
        np.ndarray
            Lead-time coordinate values available for input sampling.
        """
        return self.effective_input.coords[self.lead_time_dim].values

    @property
    @abc.abstractmethod
    def effective_input(self) -> ConditionDataConfig | ModelDataConfig | None:
        """
        Return the data configuration used as the effective model input.

        Returns
        -------
        ConditionDataConfig | ModelDataConfig | None
            Data configuration used to construct model inputs.
        """
        pass

    @final
    @property
    def _using_model_data_as_condition(self) -> bool:
        """
        Determine whether model data are used as conditioning data.

        Returns
        -------
        bool
            ``True`` when the conditioning configuration resolves to the same model
            data source, otherwise ``False``.
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
        Return the resolved conditioning data configuration.

        Returns
        -------
        ConditionDataConfig | ModelDataConfig | None
            Effective conditioning data, or ``None`` when conditioning is not used.
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
            Lowercase conditioning method, or ``None`` when no method is configured.
        """
        return None if self.condition_method is None else self.condition_method.lower()

    @final
    def _model_as_condition(self) -> ModelDataConfig:
        """
        Construct conditioning data from the configured model data.

        The resulting configuration shares the model data source and preprocessing
        pipeline while adjusting ensemble averaging according to the conditioning
        method.

        Returns
        -------
        ModelDataConfig
            Model data configuration adapted for use as conditioning data.
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
        Resolve requested initialization times available from the effective input.

        Requested times are checked against the complete set of available dataset
        times and then restricted to times present in the effective input data.

        Parameters
        ----------
        requested_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
            Initialization times requested for sampling.

        Returns
        -------
        xr.DataArray
            Requested initialization times that are present in the effective input
            data.

        Raises
        ------
        ValueError
            If any requested initialization time is unavailable from the dataset.
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

        Explicit conditioning data are used when provided. Otherwise, model data are
        adapted for conditioning when required by the conditioning method.

        Returns
        -------
        DatasetConfigABC
            This configuration instance with the effective conditioning source
            resolved.
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
        Build a dataset from this configuration.

        Returns
        -------
        DatasetABC
            Dataset constructed from the concrete configuration.
        """
        pass


@dataclasses.dataclass
class AddedTimeFeatures:
    """
    Construct additional temporal features for dataset samples.

    Temporal features can represent normalized valid time, normalized lead time,
    or cyclical month and day-of-year components.

    Parameters
    ----------
    reference_config : DatasetConfigABC
        Dataset configuration providing the temporal range, lead times, and
        dimension names used to construct and normalize features.
    time_features : list[str] | None
        Temporal features to generate. If ``None`` or empty, no temporal features
        are added.
    """

    reference_config: DatasetConfigABC
    time_features: list[str] | None = None

    def __post_init__(self):
        """
        Initialize and validate the requested temporal features.

        Temporal normalization bounds and supported feature names are obtained from
        the reference dataset configuration.

        Raises
        ------
        ValueError
            If one or more requested temporal features are unsupported.
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
        Return the number of days in the year containing a time value.

        The result accounts for standard, leap-year, and supported CF calendar
        conventions.

        Parameters
        ----------
        time : np.datetime64 | datetime.datetime | cftime.datetime
            Time value whose year length is required.

        Returns
        -------
        int
            Number of days in the corresponding calendar year.
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
        Build temporal features for a collection of sample coordinates.

        Valid times are derived from initialization and lead times. Requested
        features are then normalized or encoded cyclically and stored in sample
        order.

        Parameters
        ----------
        sample_coords : dict[str, np.ndarray]
            Sampling coordinates containing initialization times and lead times.

        Returns
        -------
        AddedTimeFeatures
            This instance with the temporal feature array populated.

        Raises
        ------
        ValueError
            If required sampling coordinates are missing, the reference temporal span
            is zero when time normalization is requested, or the maximum configured
            lead time is zero when lead-time normalization is requested.
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
        Return temporal features for a dataset sample.

        For multidimensional model inputs, temporal features are spatially broadcast
        to match the non-channel dimensions of the input.

        Parameters
        ----------
        ind : int
            Index of the sample whose temporal features are requested.
        input : xr.DataArray
            Model input used to determine whether temporal features require spatial
            broadcasting.

        Returns
        -------
        np.ndarray | None
            Temporal features for the selected sample, or ``None`` when no features
            are configured.

        Raises
        ------
        IndexError
            If ``ind`` is outside the available sample range.
        RuntimeError
            If temporal features have not been built before indexing.
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
        Return the number of configured temporal features.

        Returns
        -------
        int
            Number of temporal features.
        """
        return len(self.time_features)

    def __eq__(self, other):
        """
        Compare two temporal feature configurations.

        Configurations are considered equal when they use the same dataset
        configuration type, lead times, common temporal reference, and requested
        features.

        Parameters
        ----------
        other : Any
            Object to compare with this temporal feature configuration.

        Returns
        -------
        bool | NotImplemented
            ``True`` when the configurations are equivalent, ``False`` when they are
            not, or ``NotImplemented`` for objects of another type.
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
    Abstract base class for sampled PPP datasets.

    Handles validation of requested times, sampling-mask construction, opening
    model and conditioning data, coordinate-to-index conversion, temporal feature
    generation, and indexed sample access support.

    Attributes
    ----------
    config : DatasetConfigABC
        Configuration defining the dataset sources and sampling behavior.
    requested_times : Sequence[np.datetime64 | datetime.datetime | cftime.datetime] | np.ndarray | xr.DataArray
        Initialization times requested for dataset sampling.
    mask : xr.DataArray | None
        Mask identifying unavailable or excluded sample combinations.
    time_features : AddedTimeFeatures
        Temporal feature generator associated with the dataset.
    return_metadata : bool
        Whether dataset samples include their sampling metadata.
    load : bool
        Whether opened xarray data are loaded into memory.
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
        Initialize dataset sampling state.

        Validates initialization settings, prepares the sampling mask, opens required
        model and conditioning data, resolves sample coordinates and positional
        indexes, and constructs temporal features for the resulting samples.
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
                load=self.load, add_time_auxiliary_coords=True
            )

        self.sample_coords = self.get_sampling_coords()

        self.model_indexes = self.get_model_indexes(self.sample_coords)
        self.cond_indexes = self.get_cond_indexes(self.sample_coords)

        self.time_features = dataclasses.replace(self.time_features)
        self.time_features.build_time_features(self.sample_coords)

    @final
    def _check_init(self):
        """
        Validate dataset initialization requirements.

        Raises
        ------
        RuntimeError
            If the configured preprocessing pipelines have not been fitted.
        ValueError
            If the requested initialization-time sequence is invalid or contains
            times unavailable from the dataset configuration.
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
        Resolve and validate the sampling mask.

        When no mask is supplied, a mask spanning all available initialization and
        lead times is created and initialized as fully available.

        Raises
        ------
        ValueError
            If the mask does not contain both the initialization-time and lead-time
            dimensions.
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
        Return coordinate selectors used to construct the sampling mask.

        Returns
        -------
        dict
            Initialization-time and lead-time selections used for dataset sampling.
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
        Indicate whether model data must be opened and indexed.

        Returns
        -------
        bool
            Whether the concrete dataset requires model data.
        """
        pass

    @property
    @abc.abstractmethod
    def _write_condition_to_input(self):
        """
        Indicate whether conditioning data replace the primary model input.

        Returns
        -------
        bool
            Whether conditioning data are written directly to the model input.
        """
        pass

    @property
    @abc.abstractmethod
    def _concat_condition_to_input(self):
        """
        Indicate whether conditioning variables are concatenated with model inputs.

        Returns
        -------
        bool
            Whether conditioning variables contribute additional input channels.
        """
        pass

    @final
    def _prepare_sampling_mask(self, sampling_times_selectors: dict):
        """
        Restrict the sampling mask to requested times and lead times.

        The mask is selected using the provided temporal coordinates and expanded
        across model realizations when individual ensemble members are sampled.
        Invalid entries are represented as missing values for subsequent stacking.

        Parameters
        ----------
        sampling_times_selectors : dict
            Coordinate selections for initialization time and lead time.

        Returns
        -------
        DatasetABC
            This dataset instance with its sampling mask prepared.

        Raises
        ------
        ValueError
            If selectors for initialization time or lead time are missing.
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
        Extract coordinates for all valid dataset samples.

        The sampling dimensions are stacked into a single batch dimension and masked
        sample combinations are removed.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from each sampling dimension to its coordinate values in sample
            order.
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
        Map sampling coordinates to positional indexes in the model dataset.

        Parameters
        ----------
        sample_coords : dict[str, np.ndarray]
            Sampling coordinates for which model indexes are required.

        Returns
        -------
        dict[str, np.ndarray] | None
            Positional indexes for each sampling dimension, or ``None`` when model
            data are not loaded by the dataset.

        Raises
        ------
        ValueError
            If one or more sampling coordinates cannot be found in the model dataset.
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
        Map sampling coordinates to positional indexes in the conditioning dataset.

        Static conditioning requires no positional indexes. Same-member conditioning
        also maps realization coordinates so that model and conditioning ensemble
        members remain aligned.

        Parameters
        ----------
        sample_coords : dict[str, np.ndarray]
            Sampling coordinates for which conditioning indexes are required.

        Returns
        -------
        dict[str, np.ndarray] | None
            Positional indexes for applicable conditioning dimensions, or ``None``
            when no indexed conditioning data are required.

        Raises
        ------
        ValueError
            If same-member conditioning lacks realization coordinates or if one or
            more sampling coordinates cannot be found in the conditioning dataset.
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
        Determine the shape of a model input sample.

        The first dimension corresponds to input variables, including concatenated
        conditioning variables when configured. Remaining dimensions are determined
        from a fitted flattener or supported neural-network dimensions.

        Returns
        -------
        tuple
            Input shape excluding the batch dimension.
        """

        len_names = len(self.config.effective_input.names)
        if self._concat_condition_to_input:
            len_names += len(self.config.effective_condition.names)

        if self.config.effective_input.preprocessing_pipeline.has_flattener:
            flattener = (
                self.config.effective_input.preprocessing_pipeline.get_flattener()
            )
            in_shape = flattener.final_locations.shape

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
        Return the number of additional temporal features.

        Returns
        -------
        int
            Number of configured temporal features.
        """
        return len(self.time_features)

    @final
    def _index_condition_dataset(self, ind: int) -> xr.DataArray | None:
        """
        Retrieve conditioning data for a sample.

        Static conditioning returns the full static field. Other conditioning methods
        use precomputed positional indexes, while cross-ensemble conditioning samples
        a realization randomly.

        Parameters
        ----------
        ind : int
            Sample index to retrieve.

        Returns
        -------
        xr.DataArray | None
            Selected and preprocessed conditioning variables, or ``None`` when no
            conditioning data are configured.
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
        Retrieve model data for a sample.

        Parameters
        ----------
        ind : int
            Sample index to retrieve.

        Returns
        -------
        xr.DataArray | None
            Selected and preprocessed model variables, or ``None`` when model data
            are not required.
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
        Compute one or more lazy arrays synchronously.

        Dask computation is performed with the synchronous scheduler while standard
        error output is suppressed.

        Parameters
        ----------
        *arrays : Any
            Lazy objects to compute.

        Returns
        -------
        tuple
            Computed values corresponding to the supplied objects.
        """
        with suppress_stderr(), dask.config.set(scheduler="synchronous"):
            return dask.compute(*arrays)

    @final
    def __len__(self):
        """
        Return the number of valid dataset samples.

        Returns
        -------
        int
            Number of entries represented by the sampling coordinates.
        """
        return len(next(iter(self.sample_coords.values())))

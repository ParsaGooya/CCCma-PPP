import abc
from typing import ClassVar, final
import dataclasses
import numpy as np

from cccma_ppp.data_modules.data import ModelDataConfig, ConditionDataConfig


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
            If condition data does not span the same time range as model data.
        ValueError
            If condition data does not provide sufficient lead-time coverage.
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
                if not set(self.model.year_range).issubset(
                    set(self.condition.year_range)
                ):
                    raise ValueError(
                        "Condition data should be available"
                        " on the same time period as model data."
                    )

                if not (
                    self.model.info.sizes["lead_time"]
                    <= self.condition.info.sizes["lead_time"]
                ):
                    raise ValueError(
                        "Condition data should be available"
                        " on the same lead_times as model data."
                    )

            if getattr(self, "observation", False) is not None:
                if not self.condition.info.coords["lat"].equals(
                    self.model.info.coords["lat"]
                ):
                    raise ValueError(
                        "model and condition data do not have the same latitudes cooridnates."
                        / "when bias correcting to observations"
                    )
                if not self.condition.info.coords["lon"].equals(
                    self.model.info.coords["lon"]
                ):
                    raise ValueError(
                        "model and condition data do not have the same longitudes cooridnates."
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
        if self.lead_months is not None:
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

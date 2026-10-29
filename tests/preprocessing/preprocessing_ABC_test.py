import numpy as np
import pytest
import xarray as xr

from S2D4P.preprocessing.preprocessing_ABC import (
    PreprocessModuleABC,
)


class ConcretePreprocessor(PreprocessModuleABC):
    def _reset_fit(self):
        self.fitted = False

    def __init__(
        self,
        *,
        dims=None,
        frequency=None,
        fitted=False,
    ):
        self.dims = dims
        self.frequency = frequency
        self.fitted = fitted
        self.large_ensemble = False

    def fit(self, data):
        self.fitted = True
        return self

    def transform(self, data, **kwargs):
        self._check_fitted()
        return data

    def inverse_transform(self, data, **kwargs):
        self._check_fitted()
        return data


def make_time_data(
    *,
    times=None,
    include_realization=False,
):
    if times is None:
        times = np.asarray(
            [
                "2000-01-01",
                "2000-02-15",
                "2001-03-20",
            ],
            dtype="datetime64[ns]",
        )

    if include_realization:
        values = np.arange(
            2 * len(times),
            dtype=float,
        ).reshape(
            2,
            len(times),
        )

        return xr.DataArray(
            values,
            dims=(
                PreprocessModuleABC.realization_dim,
                PreprocessModuleABC.init_time_dim,
            ),
            coords={
                PreprocessModuleABC.realization_dim: [
                    0,
                    1,
                ],
                PreprocessModuleABC.init_time_dim: times,
            },
        )

    return xr.DataArray(
        np.arange(
            len(times),
            dtype=float,
        ),
        dims=(PreprocessModuleABC.init_time_dim,),
        coords={
            PreprocessModuleABC.init_time_dim: times,
        },
    )


class TestPreprocessModuleABC:
    @pytest.mark.pruned
    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            PreprocessModuleABC()

    @pytest.mark.pruned
    def test_class_configuration(self):
        assert PreprocessModuleABC.init_time_dim is not None
        assert PreprocessModuleABC.lead_time_dim is not None
        assert PreprocessModuleABC.realization_dim is not None
        assert PreprocessModuleABC.lead_time_resolution is not None
        assert PreprocessModuleABC.supported_frequencies == {
            None,
            "year",
            "month",
            "day",
        }

    @pytest.mark.pruned
    def test_concrete_fit_returns_self(self):
        preprocessor = ConcretePreprocessor()

        result = preprocessor.fit(
            xr.DataArray(
                [
                    1.0,
                    2.0,
                ],
                dims=("samples",),
            )
        )

        assert result is preprocessor
        assert preprocessor.fitted is True

    @pytest.mark.pruned
    def test_concrete_transform_returns_data(self):
        preprocessor = ConcretePreprocessor(
            fitted=True,
        )
        data = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=("samples",),
        )

        result = preprocessor.transform(data)

        assert result is data

    @pytest.mark.pruned
    def test_concrete_inverse_transform_returns_data(self):
        preprocessor = ConcretePreprocessor(
            fitted=True,
        )
        data = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=("samples",),
        )

        result = preprocessor.inverse_transform(data)

        assert result is data


class TestGetReductionDims:
    @pytest.mark.pruned
    def test_none_dimensions_are_preserved(self):
        preprocessor = ConcretePreprocessor(
            dims=(),
        )
        data = xr.DataArray(
            np.ones(
                (
                    2,
                    3,
                )
            ),
            dims=(
                PreprocessModuleABC.realization_dim,
                "samples",
            ),
        )

        result = preprocessor._get_reduction_dims(data)

        assert result is None
        assert preprocessor.large_ensemble is False

    @pytest.mark.pruned
    def test_dimensions_without_realization_are_preserved(self):
        preprocessor = ConcretePreprocessor(
            dims=("samples",),
        )
        data = xr.DataArray(
            np.ones(3),
            dims=("samples",),
        )

        result = preprocessor._get_reduction_dims(data)

        assert result == ("samples",)
        assert preprocessor.large_ensemble is False

    @pytest.mark.pruned
    def test_realization_dimension_is_added(self):
        preprocessor = ConcretePreprocessor(
            dims=("samples",),
        )
        data = xr.DataArray(
            np.ones(
                (
                    2,
                    3,
                )
            ),
            dims=(
                PreprocessModuleABC.realization_dim,
                "samples",
            ),
        )

        result = preprocessor._get_reduction_dims(data)

        assert result == (
            PreprocessModuleABC.realization_dim,
            "samples",
        )
        assert preprocessor.large_ensemble is True

    @pytest.mark.pruned
    def test_existing_realization_dimension_is_not_duplicated(self):
        preprocessor = ConcretePreprocessor(
            dims=(
                PreprocessModuleABC.realization_dim,
                "samples",
            ),
        )
        data = xr.DataArray(
            np.ones(
                (
                    2,
                    3,
                )
            ),
            dims=(
                PreprocessModuleABC.realization_dim,
                "samples",
            ),
        )

        result = preprocessor._get_reduction_dims(data)

        assert result == (
            PreprocessModuleABC.realization_dim,
            "samples",
        )
        assert preprocessor.large_ensemble is False

    @pytest.mark.pruned
    def test_list_dimensions_are_converted_when_realization_is_added(self):
        preprocessor = ConcretePreprocessor(
            dims=[
                "samples",
            ],
        )
        data = xr.DataArray(
            np.ones(
                (
                    2,
                    3,
                )
            ),
            dims=(
                PreprocessModuleABC.realization_dim,
                "samples",
            ),
        )

        result = preprocessor._get_reduction_dims(data)

        assert result == (
            PreprocessModuleABC.realization_dim,
            "samples",
        )
        assert isinstance(
            result,
            tuple,
        )

    @pytest.mark.pruned
    def test_preprocessor_dimensions_are_not_mutated(self):
        preprocessor = ConcretePreprocessor(
            dims=("samples",),
        )
        data = xr.DataArray(
            np.ones(
                (
                    2,
                    3,
                )
            ),
            dims=(
                PreprocessModuleABC.realization_dim,
                "samples",
            ),
        )

        preprocessor._get_reduction_dims(data)

        assert preprocessor.dims == ("samples",)

    @pytest.mark.pruned
    def test_dataset_input_is_supported(self):
        preprocessor = ConcretePreprocessor(
            dims=("samples",),
        )
        data = xr.Dataset(
            {
                "tas": (
                    (
                        PreprocessModuleABC.realization_dim,
                        "samples",
                    ),
                    np.ones(
                        (
                            2,
                            3,
                        )
                    ),
                )
            }
        )

        result = preprocessor._get_reduction_dims(data)

        assert result == (
            PreprocessModuleABC.realization_dim,
            "samples",
        )
        assert preprocessor.large_ensemble is True


class TestAddGroupingCoordinate:
    def test_none_frequency_returns_original_data(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency=None,
        )
        data = make_time_data()

        result = preprocessor._add_grouping_coordinate(data)

        assert result is data

    @pytest.mark.pruned
    def test_missing_time_reduction_dimension_returns_original_data(self):
        preprocessor = ConcretePreprocessor(
            dims=(
                "latitude",
                "longitude",
            ),
            frequency="month",
        )
        data = make_time_data()

        result = preprocessor._add_grouping_coordinate(data)

        assert "month" in result.coords

    @pytest.mark.pruned
    def test_year_coordinate_is_added(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="year",
        )
        data = make_time_data()

        result = preprocessor._add_grouping_coordinate(data)

        assert "year" in result.coords
        np.testing.assert_array_equal(
            result["year"].values,
            np.asarray(
                [
                    2000,
                    2000,
                    2001,
                ]
            ),
        )
        assert result["year"].dims == (PreprocessModuleABC.init_time_dim,)

    @pytest.mark.pruned
    def test_month_coordinate_is_added(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="month",
        )
        data = make_time_data()

        result = preprocessor._add_grouping_coordinate(data)

        assert "month" in result.coords
        np.testing.assert_array_equal(
            result["month"].values,
            np.asarray(
                [
                    1,
                    2,
                    3,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_day_coordinate_is_added(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="day",
        )
        data = make_time_data()

        result = preprocessor._add_grouping_coordinate(data)

        assert "day" in result.coords
        np.testing.assert_array_equal(
            result["day"].values,
            np.asarray(
                [
                    1,
                    46,
                    79,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_unexpected_frequency_raises(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="hour",
        )
        data = make_time_data()

        with pytest.raises(
            ValueError,
            match="Unsupported frequency 'hour'",
        ):
            preprocessor._add_grouping_coordinate(data)

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "frequency",
        [
            "year",
            "month",
            "day",
        ],
    )
    def test_dataset_input_is_supported(
        self,
        frequency,
    ):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency=frequency,
        )

        data_array = make_time_data()
        data = xr.Dataset(
            {
                "tas": data_array,
                "pr": data_array + 1,
            }
        )

        result = preprocessor._add_grouping_coordinate(data)

        assert frequency in result.coords
        assert result[frequency].dims == (PreprocessModuleABC.init_time_dim,)


class TestAlignStatForTransform:
    @pytest.mark.pruned
    def test_none_frequency_returns_original_statistic(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency=None,
        )
        data = make_time_data()
        stat = xr.DataArray(2.0)

        result = preprocessor._align_stat_for_transform(
            data,
            stat,
        )

        assert result is stat

    @pytest.mark.pruned
    def test_missing_time_reduction_dimension_returns_statistic(self):
        preprocessor = ConcretePreprocessor(
            dims=(
                "latitude",
                "longitude",
            ),
            frequency="month",
        )
        data = make_time_data()
        stat = xr.DataArray(
            np.arange(
                1,
                13,
                dtype=float,
            ),
            dims=("month",),
            coords={
                "month": np.arange(
                    1,
                    13,
                )
            },
        )

        result = preprocessor._align_stat_for_transform(
            data,
            stat,
        )

        assert result is stat

    @pytest.mark.pruned
    def test_uses_existing_auxiliary_coordinate(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="month",
        )

        data = make_time_data().assign_coords(
            month=(
                PreprocessModuleABC.init_time_dim,
                [
                    3,
                    1,
                    2,
                ],
            )
        )
        stat = xr.DataArray(
            np.arange(
                10,
                130,
                10,
                dtype=float,
            ),
            dims=("month",),
            coords={
                "month": np.arange(
                    1,
                    13,
                )
            },
        )

        result = preprocessor._align_stat_for_transform(
            data,
            stat,
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    30.0,
                    10.0,
                    20.0,
                ]
            ),
        )
        assert result.dims == (PreprocessModuleABC.init_time_dim,)

    @pytest.mark.pruned
    def test_builds_year_indexer_from_initialization_time(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="year",
        )
        data = make_time_data()
        stat = xr.DataArray(
            [
                10.0,
                20.0,
            ],
            dims=("year",),
            coords={
                "year": [
                    2000,
                    2001,
                ]
            },
        )

        result = preprocessor._align_stat_for_transform(
            data,
            stat,
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    10.0,
                    10.0,
                    20.0,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_builds_month_indexer_from_initialization_time(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="month",
        )
        data = make_time_data()
        stat = xr.DataArray(
            np.arange(
                1,
                13,
                dtype=float,
            ),
            dims=("month",),
            coords={
                "month": np.arange(
                    1,
                    13,
                )
            },
        )

        result = preprocessor._align_stat_for_transform(
            data,
            stat,
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    1.0,
                    2.0,
                    3.0,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_builds_day_indexer_from_initialization_time(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="day",
        )
        data = make_time_data()
        stat = xr.DataArray(
            np.arange(
                1,
                367,
                dtype=float,
            ),
            dims=("day",),
            coords={
                "day": np.arange(
                    1,
                    367,
                )
            },
        )

        result = preprocessor._align_stat_for_transform(
            data,
            stat,
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    1.0,
                    46.0,
                    79.0,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_missing_temporal_coordinates_raises(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="month",
        )

        data = xr.DataArray(
            np.ones(3),
            dims=(PreprocessModuleABC.init_time_dim,),
        )
        stat = xr.DataArray(
            np.arange(
                1,
                13,
                dtype=float,
            ),
            dims=("month",),
            coords={
                "month": np.arange(
                    1,
                    13,
                )
            },
        )

        with pytest.raises(
            ValueError,
            match=(
                "Data must contain either the auxiliary coordinate "
                "'month' or the initialization-time coordinate"
            ),
        ):
            preprocessor._align_stat_for_transform(
                data,
                stat,
            )

    def test_unexpected_frequency_without_auxiliary_coordinate_raises(
        self,
    ):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="hour",
        )
        data = make_time_data()
        stat = xr.DataArray(
            np.arange(
                24,
                dtype=float,
            ),
            dims=("hour",),
            coords={"hour": np.arange(24)},
        )

        with pytest.raises(
            ValueError,
            match="Unsupported frequency 'hour'",
        ):
            preprocessor._align_stat_for_transform(
                data,
                stat,
            )

    @pytest.mark.pruned
    def test_unexpected_frequency_uses_existing_auxiliary_coordinate(
        self,
    ):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="hour",
        )
        data = make_time_data().assign_coords(
            hour=(
                PreprocessModuleABC.init_time_dim,
                [
                    1,
                    2,
                    3,
                ],
            )
        )
        stat = xr.DataArray(
            np.arange(
                24,
                dtype=float,
            ),
            dims=("hour",),
            coords={"hour": np.arange(24)},
        )

        result = preprocessor._align_stat_for_transform(
            data,
            stat,
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    1.0,
                    2.0,
                    3.0,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_preserves_non_temporal_statistic_dimensions(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="month",
        )
        data = make_time_data()

        stat = xr.DataArray(
            np.arange(
                24,
                dtype=float,
            ).reshape(
                12,
                2,
            ),
            dims=(
                "month",
                "channels",
            ),
            coords={
                "month": np.arange(
                    1,
                    13,
                ),
                "channels": [
                    "tas",
                    "pr",
                ],
            },
        )

        result = preprocessor._align_stat_for_transform(
            data,
            stat,
        )

        assert result.dims == (
            PreprocessModuleABC.init_time_dim,
            "channels",
        )
        np.testing.assert_array_equal(
            result["channels"].values,
            np.asarray(
                [
                    "tas",
                    "pr",
                ]
            ),
        )

    @pytest.mark.pruned
    def test_dataset_input_can_be_aligned(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="month",
        )

        data_array = make_time_data()
        data = xr.Dataset(
            {
                "tas": data_array,
            }
        )
        stat = xr.DataArray(
            np.arange(
                1,
                13,
                dtype=float,
            ),
            dims=("month",),
            coords={
                "month": np.arange(
                    1,
                    13,
                )
            },
        )

        result = preprocessor._align_stat_for_transform(
            data,
            stat,
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    1.0,
                    2.0,
                    3.0,
                ]
            ),
        )


class TestCheckFitted:
    @pytest.mark.pruned
    def test_unfitted_preprocessor_raises(self):
        preprocessor = ConcretePreprocessor(
            fitted=False,
        )

        with pytest.raises(
            RuntimeError,
            match=(
                "The preprocessor must be fitted before calling "
                "'transform' or 'inverse_transform'"
            ),
        ):
            preprocessor._check_fitted()

    @pytest.mark.pruned
    def test_fitted_preprocessor_is_accepted(self):
        preprocessor = ConcretePreprocessor(
            fitted=True,
        )

        assert preprocessor._check_fitted() is None

    @pytest.mark.pruned
    def test_transform_checks_fitted_state(self):
        preprocessor = ConcretePreprocessor(
            fitted=False,
        )

        with pytest.raises(
            RuntimeError,
            match="must be fitted",
        ):
            preprocessor.transform(
                xr.DataArray(
                    [
                        1.0,
                    ],
                    dims=("samples",),
                )
            )

    def test_inverse_transform_checks_fitted_state(self):
        preprocessor = ConcretePreprocessor(
            fitted=False,
        )

        with pytest.raises(
            RuntimeError,
            match="must be fitted",
        ):
            preprocessor.inverse_transform(
                xr.DataArray(
                    [
                        1.0,
                    ],
                    dims=("samples",),
                )
            )

    @pytest.mark.pruned
    def test_fit_enables_transform(self):
        preprocessor = ConcretePreprocessor(
            fitted=False,
        )
        data = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=("samples",),
        )

        preprocessor.fit(data)

        assert preprocessor.transform(data) is data

    @pytest.mark.pruned
    def test_fit_enables_inverse_transform(self):
        preprocessor = ConcretePreprocessor(
            fitted=False,
        )
        data = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=("samples",),
        )

        preprocessor.fit(data)

        assert preprocessor.inverse_transform(data) is data


from unittest.mock import patch


class TestAdditionalReductionDims:
    def test_missing_configured_dimension_raises(self):
        preprocessor = ConcretePreprocessor(
            dims=(
                "latitude",
                "longitude",
            ),
        )
        data = xr.DataArray(
            np.ones(
                (
                    2,
                    3,
                )
            ),
            dims=(
                "latitude",
                "samples",
            ),
        )

        with pytest.raises(
            ValueError,
            match="longitude",
        ):
            preprocessor._get_reduction_dims(data)

    @pytest.mark.pruned
    def test_missing_dimension_error_lists_available_dimensions(self):
        preprocessor = ConcretePreprocessor(
            dims=("missing",),
        )
        data = xr.DataArray(
            np.ones(
                (
                    2,
                    3,
                )
            ),
            dims=(
                "latitude",
                "longitude",
            ),
        )

        with pytest.raises(
            ValueError,
            match="Available",
        ):
            preprocessor._get_reduction_dims(data)

    @pytest.mark.pruned
    def test_large_ensemble_flag_is_reset_between_calls(self):
        preprocessor = ConcretePreprocessor(
            dims=("samples",),
        )

        ensemble_data = xr.DataArray(
            np.ones(
                (
                    2,
                    3,
                )
            ),
            dims=(
                PreprocessModuleABC.realization_dim,
                "samples",
            ),
        )
        nonensemble_data = xr.DataArray(
            np.ones(3),
            dims=("samples",),
        )

        preprocessor._get_reduction_dims(
            ensemble_data,
        )
        assert preprocessor.large_ensemble is True

        result = preprocessor._get_reduction_dims(
            nonensemble_data,
        )

        assert result == ("samples",)
        assert preprocessor.large_ensemble is False

    @pytest.mark.pruned
    def test_empty_dimensions_with_no_realization_returns_none(self):
        preprocessor = ConcretePreprocessor(
            dims=(),
        )
        data = xr.DataArray(
            np.ones(3),
            dims=("samples",),
        )

        result = preprocessor._get_reduction_dims(
            data,
        )

        assert result is None
        assert preprocessor.large_ensemble is False

    @pytest.mark.pruned
    def test_realization_only_reduction_is_preserved(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.realization_dim,),
        )
        data = xr.DataArray(
            np.ones(
                (
                    2,
                    3,
                )
            ),
            dims=(
                PreprocessModuleABC.realization_dim,
                "samples",
            ),
        )

        result = preprocessor._get_reduction_dims(
            data,
        )

        assert result == (PreprocessModuleABC.realization_dim,)
        assert preprocessor.large_ensemble is False

    @pytest.mark.pruned
    def test_multiple_reduction_dimensions_preserve_order(self):
        preprocessor = ConcretePreprocessor(
            dims=(
                "longitude",
                "latitude",
            ),
        )
        data = xr.DataArray(
            np.ones(
                (
                    2,
                    3,
                    4,
                )
            ),
            dims=(
                PreprocessModuleABC.realization_dim,
                "latitude",
                "longitude",
            ),
        )

        result = preprocessor._get_reduction_dims(
            data,
        )

        assert result == (
            PreprocessModuleABC.realization_dim,
            "longitude",
            "latitude",
        )


class TestExtractTemporalCoordinate:
    def test_none_returns_original_coordinate(self):
        preprocessor = ConcretePreprocessor()
        init_time = make_time_data()[PreprocessModuleABC.init_time_dim]

        result = preprocessor._extract_temporal_coordinate(
            init_time,
            None,
        )

        assert result is init_time

    @pytest.mark.pruned
    def test_extracts_year(self):
        preprocessor = ConcretePreprocessor()
        init_time = make_time_data()[PreprocessModuleABC.init_time_dim]

        result = preprocessor._extract_temporal_coordinate(
            init_time,
            "year",
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    2000,
                    2000,
                    2001,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_extracts_month(self):
        preprocessor = ConcretePreprocessor()
        init_time = make_time_data()[PreprocessModuleABC.init_time_dim]

        result = preprocessor._extract_temporal_coordinate(
            init_time,
            "month",
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    1,
                    2,
                    3,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_extracts_day_of_year(self):
        preprocessor = ConcretePreprocessor()
        init_time = make_time_data()[PreprocessModuleABC.init_time_dim]

        result = preprocessor._extract_temporal_coordinate(
            init_time,
            "day",
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    1,
                    46,
                    79,
                ]
            ),
        )

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "frequency",
        [
            "hour",
            "week",
            "",
            "MONTH",
        ],
    )
    def test_rejects_unsupported_frequency(
        self,
        frequency,
    ):
        preprocessor = ConcretePreprocessor()
        init_time = make_time_data()[PreprocessModuleABC.init_time_dim]

        with pytest.raises(
            ValueError,
            match="Unsupported frequency",
        ):
            preprocessor._extract_temporal_coordinate(
                init_time,
                frequency,
            )


class TestAdditionalGroupingCoordinate:
    @pytest.mark.pruned
    def test_existing_grouping_coordinate_is_replaced(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="month",
        )
        data = make_time_data().assign_coords(
            month=(
                PreprocessModuleABC.init_time_dim,
                [
                    12,
                    12,
                    12,
                ],
            )
        )

        result = preprocessor._add_grouping_coordinate(
            data,
        )

        np.testing.assert_array_equal(
            result["month"].values,
            np.asarray(
                [
                    1,
                    2,
                    3,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_original_data_is_not_modified(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="month",
        )
        data = make_time_data()

        result = preprocessor._add_grouping_coordinate(
            data,
        )

        assert "month" not in data.coords
        assert "month" in result.coords
        assert result is not data

    @pytest.mark.pruned
    def test_missing_initialization_coordinate_raises(self):
        preprocessor = ConcretePreprocessor(
            dims=("samples",),
            frequency="month",
        )
        data = xr.DataArray(
            np.ones(3),
            dims=("samples",),
        )

        with pytest.raises(
            KeyError,
        ):
            preprocessor._add_grouping_coordinate(
                data,
            )

    @pytest.mark.pruned
    def test_grouping_coordinate_preserves_initialization_dimension(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="year",
        )
        data = make_time_data()

        result = preprocessor._add_grouping_coordinate(
            data,
        )

        assert result["year"].dims == (PreprocessModuleABC.init_time_dim,)
        assert (
            result.sizes[PreprocessModuleABC.init_time_dim]
            == data.sizes[PreprocessModuleABC.init_time_dim]
        )


class TestAdditionalAlignStatistic:
    @pytest.mark.pruned
    def test_frequency_is_ignored_when_init_time_not_reduced(self):
        preprocessor = ConcretePreprocessor(
            dims=("latitude",),
            frequency="month",
        )
        data = make_time_data()
        stat = xr.DataArray(
            np.arange(
                12,
                dtype=float,
            ),
            dims=("month",),
            coords={
                "month": np.arange(
                    1,
                    13,
                )
            },
        )

        result = preprocessor._align_stat_for_transform(
            data,
            stat,
        )

        assert result is stat

    @pytest.mark.pruned
    def test_existing_auxiliary_coordinate_takes_precedence(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="month",
        )
        data = make_time_data().assign_coords(
            month=(
                PreprocessModuleABC.init_time_dim,
                [
                    12,
                    11,
                    10,
                ],
            )
        )
        stat = xr.DataArray(
            np.arange(
                1,
                13,
                dtype=float,
            ),
            dims=("month",),
            coords={
                "month": np.arange(
                    1,
                    13,
                )
            },
        )

        result = preprocessor._align_stat_for_transform(
            data,
            stat,
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    12.0,
                    11.0,
                    10.0,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_existing_auxiliary_coordinate_avoids_temporal_extraction(
        self,
    ):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="month",
        )
        data = make_time_data().assign_coords(
            month=(
                PreprocessModuleABC.init_time_dim,
                [
                    1,
                    2,
                    3,
                ],
            )
        )
        stat = xr.DataArray(
            np.arange(
                1,
                13,
                dtype=float,
            ),
            dims=("month",),
            coords={
                "month": np.arange(
                    1,
                    13,
                )
            },
        )

        with patch.object(
            preprocessor,
            "_extract_temporal_coordinate",
        ) as extract:
            result = preprocessor._align_stat_for_transform(
                data,
                stat,
            )

        extract.assert_not_called()
        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    1.0,
                    2.0,
                    3.0,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_dataset_statistic_is_aligned(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="month",
        )
        data = make_time_data()
        stat = xr.Dataset(
            {
                "mean": (
                    "month",
                    np.arange(
                        1,
                        13,
                        dtype=float,
                    ),
                ),
                "std": (
                    "month",
                    np.arange(
                        101,
                        113,
                        dtype=float,
                    ),
                ),
            },
            coords={
                "month": np.arange(
                    1,
                    13,
                )
            },
        )

        result = preprocessor._align_stat_for_transform(
            data,
            stat,
        )

        assert isinstance(
            result,
            xr.Dataset,
        )
        np.testing.assert_array_equal(
            result["mean"].values,
            np.asarray(
                [
                    1.0,
                    2.0,
                    3.0,
                ]
            ),
        )
        np.testing.assert_array_equal(
            result["std"].values,
            np.asarray(
                [
                    101.0,
                    102.0,
                    103.0,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_missing_statistical_group_raises_key_error(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="month",
        )
        data = make_time_data(
            times=np.asarray(
                [
                    "2000-12-01",
                ],
                dtype="datetime64[ns]",
            )
        )
        stat = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=("month",),
            coords={
                "month": [
                    1,
                    2,
                ]
            },
        )

        with pytest.raises(
            KeyError,
        ):
            preprocessor._align_stat_for_transform(
                data,
                stat,
            )


class TestInverseTargetTime:
    @pytest.mark.pruned
    def test_without_lead_dimension_returns_initialization_coordinate(
        self,
    ):
        preprocessor = ConcretePreprocessor()
        data = make_time_data()

        result = preprocessor._get_inverse_target_time(
            data,
        )

        assert result.identical(data[PreprocessModuleABC.init_time_dim])

    @pytest.mark.pruned
    def test_builds_target_time_grid(self):
        preprocessor = ConcretePreprocessor()
        data = xr.DataArray(
            np.zeros(
                (
                    2,
                    3,
                )
            ),
            dims=(
                PreprocessModuleABC.init_time_dim,
                PreprocessModuleABC.lead_time_dim,
            ),
            coords={
                PreprocessModuleABC.init_time_dim: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                PreprocessModuleABC.lead_time_dim: [
                    1,
                    2,
                    3,
                ],
            },
        )

        result = preprocessor._get_inverse_target_time(
            data,
        )

        assert result.dims == (
            PreprocessModuleABC.init_time_dim,
            PreprocessModuleABC.lead_time_dim,
        )
        assert result.shape == (
            2,
            3,
        )
        np.testing.assert_array_equal(
            result[PreprocessModuleABC.init_time_dim].values,
            data[PreprocessModuleABC.init_time_dim].values,
        )
        np.testing.assert_array_equal(
            result[PreprocessModuleABC.lead_time_dim].values,
            data[PreprocessModuleABC.lead_time_dim].values,
        )

    def test_flattens_init_and_lead_time_grids_before_calling_helper(
        self,
    ):
        preprocessor = ConcretePreprocessor()
        data = xr.DataArray(
            np.zeros(
                (
                    2,
                    2,
                )
            ),
            dims=(
                PreprocessModuleABC.init_time_dim,
                PreprocessModuleABC.lead_time_dim,
            ),
            coords={
                PreprocessModuleABC.init_time_dim: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                PreprocessModuleABC.lead_time_dim: [
                    1,
                    2,
                ],
            },
        )

        expected = np.asarray(
            [
                "2000-01-01",
                "2000-02-01",
                "2001-01-01",
                "2001-02-01",
            ],
            dtype="datetime64[ns]",
        )

        with patch(
            "cccma_ppp.preprocessing.preprocessing_ABC.add_lead_times",
            return_value=expected,
        ) as add_times:
            result = preprocessor._get_inverse_target_time(
                data,
            )

        kwargs = add_times.call_args.kwargs

        np.testing.assert_array_equal(
            kwargs["init_times"],
            np.asarray(
                [
                    "2000-01-01",
                    "2000-01-01",
                    "2001-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            ),
        )
        np.testing.assert_array_equal(
            kwargs["lead_times"],
            np.asarray(
                [
                    1,
                    2,
                    1,
                    2,
                ]
            ),
        )
        assert kwargs["lead_time_resolution"] == preprocessor.lead_time_resolution
        assert result.shape == (
            2,
            2,
        )

    def test_nat_target_times_raise(self):
        preprocessor = ConcretePreprocessor()
        data = xr.DataArray(
            np.zeros(
                (
                    1,
                    2,
                )
            ),
            dims=(
                PreprocessModuleABC.init_time_dim,
                PreprocessModuleABC.lead_time_dim,
            ),
            coords={
                PreprocessModuleABC.init_time_dim: np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                PreprocessModuleABC.lead_time_dim: [
                    1,
                    2,
                ],
            },
        )

        with patch(
            "cccma_ppp.preprocessing.preprocessing_ABC.add_lead_times",
            return_value=np.asarray(
                [
                    "2000-01-01",
                    "NaT",
                ],
                dtype="datetime64[ns]",
            ),
        ):
            with pytest.raises(
                ValueError,
                match="produced NaT",
            ):
                preprocessor._get_inverse_target_time(
                    data,
                )

    @pytest.mark.pruned
    def test_wrong_target_time_shape_raises(self):
        preprocessor = ConcretePreprocessor()
        data = xr.DataArray(
            np.zeros(
                (
                    2,
                    2,
                )
            ),
            dims=(
                PreprocessModuleABC.init_time_dim,
                PreprocessModuleABC.lead_time_dim,
            ),
            coords={
                PreprocessModuleABC.init_time_dim: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                PreprocessModuleABC.lead_time_dim: [
                    1,
                    2,
                ],
            },
        )

        with patch(
            "cccma_ppp.preprocessing.preprocessing_ABC.add_lead_times",
            return_value=np.asarray(
                [
                    "2000-01-01",
                    "2000-02-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            ),
        ):
            with pytest.raises(
                RuntimeError,
                match="returned wrong shape",
            ):
                preprocessor._get_inverse_target_time(
                    data,
                )

    @pytest.mark.pruned
    def test_target_times_are_reshaped_in_init_lead_order(self):
        preprocessor = ConcretePreprocessor()
        data = xr.DataArray(
            np.zeros(
                (
                    2,
                    2,
                )
            ),
            dims=(
                PreprocessModuleABC.init_time_dim,
                PreprocessModuleABC.lead_time_dim,
            ),
            coords={
                PreprocessModuleABC.init_time_dim: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                PreprocessModuleABC.lead_time_dim: [
                    1,
                    2,
                ],
            },
        )

        target_times = np.asarray(
            [
                "2000-01-01",
                "2000-02-01",
                "2001-01-01",
                "2001-02-01",
            ],
            dtype="datetime64[ns]",
        )

        with patch(
            "cccma_ppp.preprocessing.preprocessing_ABC.add_lead_times",
            return_value=target_times,
        ):
            result = preprocessor._get_inverse_target_time(
                data,
            )

        np.testing.assert_array_equal(
            result.values,
            target_times.reshape(
                2,
                2,
            ),
        )

    @pytest.mark.pruned
    def test_dataset_input_is_supported(self):
        preprocessor = ConcretePreprocessor()
        data = xr.Dataset(
            {
                "tas": (
                    (
                        PreprocessModuleABC.init_time_dim,
                        PreprocessModuleABC.lead_time_dim,
                    ),
                    np.zeros(
                        (
                            1,
                            2,
                        )
                    ),
                )
            },
            coords={
                PreprocessModuleABC.init_time_dim: np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                PreprocessModuleABC.lead_time_dim: [
                    1,
                    2,
                ],
            },
        )

        result = preprocessor._get_inverse_target_time(
            data,
        )

        assert isinstance(
            result,
            xr.DataArray,
        )
        assert result.dims == (
            PreprocessModuleABC.init_time_dim,
            PreprocessModuleABC.lead_time_dim,
        )


class TestCheckZeros:
    @pytest.mark.pruned
    def test_nonzero_dataarray_is_accepted(self):
        preprocessor = ConcretePreprocessor()
        data = xr.DataArray(
            [
                1.0,
                2.0,
                -1.0,
            ],
            dims=("samples",),
            name="values",
        )

        assert preprocessor._check_zeros(data) is None

    @pytest.mark.pruned
    def test_named_dataarray_with_zero_raises(self):
        preprocessor = ConcretePreprocessor()
        data = xr.DataArray(
            [
                1.0,
                0.0,
                2.0,
            ],
            dims=("samples",),
            name="temperature",
        )

        with pytest.raises(
            ValueError,
            match="temperature",
        ):
            preprocessor._check_zeros(data)

    @pytest.mark.pruned
    def test_unnamed_dataarray_with_zero_uses_data_name(self):
        preprocessor = ConcretePreprocessor()
        data = xr.DataArray(
            [
                1.0,
                0.0,
            ],
            dims=("samples",),
        )

        with pytest.raises(
            ValueError,
            match=r"\['data'\]",
        ):
            preprocessor._check_zeros(data)

    @pytest.mark.pruned
    def test_dataset_with_no_zeros_is_accepted(self):
        preprocessor = ConcretePreprocessor()
        data = xr.Dataset(
            {
                "tas": (
                    "samples",
                    [
                        1.0,
                        2.0,
                    ],
                ),
                "pr": (
                    "samples",
                    [
                        3.0,
                        4.0,
                    ],
                ),
            }
        )

        assert preprocessor._check_zeros(data) is None

    @pytest.mark.pruned
    def test_dataset_reports_one_zero_variable(self):
        preprocessor = ConcretePreprocessor()
        data = xr.Dataset(
            {
                "tas": (
                    "samples",
                    [
                        1.0,
                        0.0,
                    ],
                ),
                "pr": (
                    "samples",
                    [
                        3.0,
                        4.0,
                    ],
                ),
            }
        )

        with pytest.raises(
            ValueError,
            match="tas",
        ):
            preprocessor._check_zeros(data)

    def test_dataset_reports_all_zero_variables(self):
        preprocessor = ConcretePreprocessor()
        data = xr.Dataset(
            {
                "tas": (
                    "samples",
                    [
                        1.0,
                        0.0,
                    ],
                ),
                "pr": (
                    "samples",
                    [
                        0.0,
                        4.0,
                    ],
                ),
                "psl": (
                    "samples",
                    [
                        5.0,
                        6.0,
                    ],
                ),
            }
        )

        with pytest.raises(
            ValueError,
        ) as error:
            preprocessor._check_zeros(data)

        assert "tas" in str(error.value)
        assert "pr" in str(error.value)
        assert "psl" not in str(error.value)

    @pytest.mark.pruned
    def test_nan_values_are_not_treated_as_zeros(self):
        preprocessor = ConcretePreprocessor()
        data = xr.DataArray(
            [
                1.0,
                np.nan,
                2.0,
            ],
            dims=("samples",),
            name="values",
        )

        assert preprocessor._check_zeros(data) is None

    @pytest.mark.pruned
    def test_negative_values_are_not_treated_as_zeros(self):
        preprocessor = ConcretePreprocessor()
        data = xr.DataArray(
            [
                -2.0,
                -1.0,
            ],
            dims=("samples",),
            name="values",
        )

        assert preprocessor._check_zeros(data) is None

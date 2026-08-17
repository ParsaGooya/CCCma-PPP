import numpy as np
import pytest
import xarray as xr

from cccma_ppp.preprocessing.preprocessing_ABC import (
    PreprocessModuleABC,
)


class ConcretePreprocessor(PreprocessModuleABC):
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

    def test_unexpected_frequency_raises(self):
        preprocessor = ConcretePreprocessor(
            dims=(PreprocessModuleABC.init_time_dim,),
            frequency="hour",
        )
        data = make_time_data()

        with pytest.raises(
            RuntimeError,
            match="Unexpected temporal frequency 'hour'",
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
            RuntimeError,
            match="Unexpected temporal frequency 'hour'",
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

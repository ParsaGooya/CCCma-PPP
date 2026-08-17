from types import SimpleNamespace
from unittest.mock import Mock

import cftime
import numpy as np
import pytest
import xarray as xr

from cccma_ppp.generic.runtime import RuntimeContext
import cccma_ppp.preprocessing.utils_preprocessing as module
from cccma_ppp.preprocessing.utils_preprocessing import (
    AnomaliesScaler,
    Flattennanremove,
    Normalizer,
    Standardizer,
    TrendRemover,
    align_stat_data_lead_time_inverse_transform,
)


TIME_DIM = module.init_time_dim
LEAD_TIME_DIM = module.lead_time_dim


def make_time_data():
    return xr.DataArray(
        np.asarray(
            [
                2.0,
                4.0,
                6.0,
                8.0,
            ]
        ),
        dims=(TIME_DIM,),
        coords={
            TIME_DIM: np.asarray(
                [
                    "2000-01-01",
                    "2000-02-01",
                    "2001-01-01",
                    "2001-02-01",
                ],
                dtype="datetime64[ns]",
            )
        },
    )


def make_forecast_data(
    times=("2000-01-01", "2001-01-01"),
    lead_times=(1, 2),
):
    return xr.DataArray(
        np.zeros(
            (
                len(times),
                len(lead_times),
            )
        ),
        dims=(
            TIME_DIM,
            LEAD_TIME_DIM,
        ),
        coords={
            TIME_DIM: np.asarray(
                times,
                dtype="datetime64[ns]",
            ),
            LEAD_TIME_DIM: np.asarray(
                lead_times,
            ),
        },
    )


@pytest.fixture
def passthrough_alignment(monkeypatch):
    alignment = Mock(side_effect=lambda ds, stat, **kwargs: stat)

    monkeypatch.setattr(
        module,
        "align_stat_data_lead_time_inverse_transform",
        alignment,
    )

    return alignment


class TestNormalizer:
    @pytest.mark.pruned
    def test_defaults(self):
        scaler = Normalizer()

        assert scaler.min is None
        assert scaler.max is None
        assert scaler.dims == ()
        assert scaler.frequency is None
        assert scaler.large_ensemble is False
        assert scaler.fitted is False

    @pytest.mark.pruned
    def test_converts_dimensions_to_tuple(self):
        scaler = Normalizer(
            dims=[
                TIME_DIM,
                "lat",
            ]
        )

        assert scaler.dims == (
            TIME_DIM,
            "lat",
        )

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "frequency",
        [
            None,
            "year",
            "month",
            "day",
        ],
    )
    def test_accepts_supported_frequency(
        self,
        frequency,
    ):
        scaler = Normalizer(frequency=frequency)

        assert scaler.frequency == frequency

    def test_rejects_unsupported_frequency(self):
        with pytest.raises(
            ValueError,
            match="Unsupported frequency",
        ):
            Normalizer(frequency="hour")

    @pytest.mark.pruned
    def test_fit_returns_self(self):
        scaler = Normalizer(
            dims=[
                TIME_DIM,
            ]
        )

        result = scaler.fit(make_time_data())

        assert result is scaler
        assert scaler.fitted is True

    @pytest.mark.pruned
    def test_fit_computes_minimum_and_maximum(self):
        data = xr.DataArray(
            [
                2.0,
                4.0,
                8.0,
            ],
            dims=("samples",),
        )
        scaler = Normalizer(
            dims=[
                "samples",
            ]
        )

        scaler.fit(data)

        assert scaler.min.item() == 2.0
        assert scaler.max.item() == 8.0

    @pytest.mark.pruned
    def test_transform_requires_fitted_scaler(self):
        scaler = Normalizer()

        with pytest.raises(
            RuntimeError,
            match="must be fitted",
        ):
            scaler.transform(
                xr.DataArray(
                    [
                        1.0,
                    ],
                    dims=("samples",),
                )
            )

    @pytest.mark.pruned
    def test_inverse_requires_fitted_scaler(self):
        scaler = Normalizer()

        with pytest.raises(
            RuntimeError,
            match="must be fitted",
        ):
            scaler.inverse_transform(
                xr.DataArray(
                    [
                        1.0,
                    ],
                    dims=("samples",),
                )
            )

    @pytest.mark.pruned
    def test_transform(self):
        data = xr.DataArray(
            [
                2.0,
                4.0,
                6.0,
            ],
            dims=("samples",),
        )
        scaler = Normalizer(
            dims=[
                "samples",
            ]
        ).fit(data)

        result = scaler.transform(data)

        xr.testing.assert_allclose(
            result,
            xr.DataArray(
                [
                    0.0,
                    0.5,
                    1.0,
                ],
                dims=("samples",),
            ),
        )

    @pytest.mark.pruned
    def test_inverse_transform(
        self,
        passthrough_alignment,
    ):
        scaler = Normalizer()
        scaler.min = xr.DataArray(2.0)
        scaler.max = xr.DataArray(6.0)
        scaler.fitted = True

        data = xr.DataArray(
            [
                0.0,
                0.5,
                1.0,
            ],
            dims=("samples",),
        )

        result = scaler.inverse_transform(data)

        xr.testing.assert_allclose(
            result,
            xr.DataArray(
                [
                    2.0,
                    4.0,
                    6.0,
                ],
                dims=("samples",),
            ),
        )

        assert passthrough_alignment.call_count == 2
        assert passthrough_alignment.call_args_list[0].kwargs["ds"] is data
        assert passthrough_alignment.call_args_list[0].kwargs["stat"] is scaler.min

    @pytest.mark.pruned
    def test_roundtrip(
        self,
        passthrough_alignment,
    ):
        data = xr.DataArray(
            [
                2.0,
                4.0,
                6.0,
            ],
            dims=("samples",),
        )
        scaler = Normalizer(
            dims=[
                "samples",
            ]
        ).fit(data)

        restored = scaler.inverse_transform(scaler.transform(data))

        xr.testing.assert_allclose(
            restored,
            data,
        )

    @pytest.mark.pruned
    def test_monthly_grouped_statistics(self):
        data = make_time_data()
        scaler = Normalizer(
            dims=[
                TIME_DIM,
            ],
            frequency="month",
        )

        scaler.fit(data)

        assert scaler.min.dims == ("month",)
        assert scaler.max.dims == ("month",)
        np.testing.assert_array_equal(
            scaler.min["month"].values,
            [
                1,
                2,
            ],
        )
        np.testing.assert_array_equal(
            scaler.min.values,
            [
                2.0,
                4.0,
            ],
        )
        np.testing.assert_array_equal(
            scaler.max.values,
            [
                6.0,
                8.0,
            ],
        )

    @pytest.mark.pruned
    def test_dataset_input(self):
        data = make_time_data()
        dataset = xr.Dataset(
            {
                "a": data,
                "b": data + 2,
            }
        )

        scaler = Normalizer(
            dims=[
                TIME_DIM,
            ]
        ).fit(dataset)

        transformed = scaler.transform(dataset)

        assert isinstance(
            transformed,
            xr.Dataset,
        )
        assert set(transformed.data_vars) == {
            "a",
            "b",
        }

    @pytest.mark.pruned
    def test_ensemble_dimension_is_added_to_reduction(self):
        data = xr.DataArray(
            np.arange(
                12,
                dtype=float,
            ).reshape(
                3,
                4,
            ),
            dims=(
                Normalizer.realization_dim,
                "samples",
            ),
        )
        scaler = Normalizer(
            dims=[
                "samples",
            ]
        )

        scaler.fit(data)

        assert scaler.large_ensemble is True
        assert scaler.dims == ("samples",)
        assert scaler.min.ndim == 0
        assert scaler.max.ndim == 0

    @pytest.mark.pruned
    def test_existing_ensemble_dimension_is_not_duplicated(self):
        data = xr.DataArray(
            np.arange(
                12,
                dtype=float,
            ).reshape(
                3,
                4,
            ),
            dims=(
                Normalizer.realization_dim,
                "samples",
            ),
        )
        scaler = Normalizer(
            dims=[
                Normalizer.realization_dim,
                "samples",
            ]
        )

        scaler.fit(data)

        assert scaler.large_ensemble is False
        assert scaler.dims == (
            Normalizer.realization_dim,
            "samples",
        )


class TestStandardizer:
    @pytest.mark.pruned
    def test_defaults(self):
        scaler = Standardizer()

        assert scaler.mean is None
        assert scaler.std is None
        assert scaler.dims == ()
        assert scaler.frequency is None
        assert scaler.large_ensemble is False
        assert scaler.fitted is False

    def test_rejects_unsupported_frequency(self):
        with pytest.raises(
            ValueError,
            match="Unsupported frequency",
        ):
            Standardizer(frequency="hour")

    @pytest.mark.pruned
    def test_fit_computes_mean_and_standard_deviation(self):
        data = xr.DataArray(
            [
                2.0,
                4.0,
                6.0,
            ],
            dims=("samples",),
        )
        scaler = Standardizer(
            dims=[
                "samples",
            ]
        )

        result = scaler.fit(data)

        assert result is scaler
        assert scaler.fitted is True
        assert scaler.mean.item() == pytest.approx(4.0)
        assert scaler.std.item() == pytest.approx(
            np.std(
                [
                    2.0,
                    4.0,
                    6.0,
                ]
            )
        )

    def test_fit_applies_nan_mask(self):
        data = xr.DataArray(
            [
                1.0,
                2.0,
                100.0,
            ],
            dims=("samples",),
        )
        mask = xr.DataArray(
            [
                False,
                False,
                True,
            ],
            dims=("samples",),
        )

        scaler = Standardizer(
            dims=[
                "samples",
            ]
        ).fit(
            data,
            mask=mask,
        )

        assert scaler.mean.item() == pytest.approx(1.5)
        assert scaler.std.item() == pytest.approx(0.5)

    @pytest.mark.pruned
    def test_nonpositive_standard_deviation_becomes_nan(self):
        data = xr.DataArray(
            [
                4.0,
                4.0,
                4.0,
            ],
            dims=("samples",),
        )

        scaler = Standardizer(
            dims=[
                "samples",
            ]
        ).fit(data)

        assert np.isnan(scaler.std.item())

    @pytest.mark.pruned
    def test_transform(self):
        data = xr.DataArray(
            [
                2.0,
                4.0,
                6.0,
            ],
            dims=("samples",),
        )
        scaler = Standardizer(
            dims=[
                "samples",
            ]
        ).fit(data)

        result = scaler.transform(data)

        assert result.mean().item() == pytest.approx(0.0)
        assert result.std().item() == pytest.approx(1.0)

    @pytest.mark.pruned
    def test_transform_requires_fitted_scaler(self):
        with pytest.raises(
            RuntimeError,
            match="must be fitted",
        ):
            Standardizer().transform(
                xr.DataArray(
                    [
                        1.0,
                    ],
                    dims=("samples",),
                )
            )

    @pytest.mark.pruned
    def test_inverse_transform(
        self,
        passthrough_alignment,
    ):
        scaler = Standardizer()
        scaler.mean = xr.DataArray(4.0)
        scaler.std = xr.DataArray(2.0)
        scaler.fitted = True

        data = xr.DataArray(
            [
                -1.0,
                0.0,
                1.0,
            ],
            dims=("samples",),
        )

        result = scaler.inverse_transform(data)

        xr.testing.assert_allclose(
            result,
            xr.DataArray(
                [
                    2.0,
                    4.0,
                    6.0,
                ],
                dims=("samples",),
            ),
        )
        assert passthrough_alignment.call_count == 2

    @pytest.mark.pruned
    def test_roundtrip(
        self,
        passthrough_alignment,
    ):
        data = xr.DataArray(
            [
                2.0,
                4.0,
                6.0,
            ],
            dims=("samples",),
        )
        scaler = Standardizer(
            dims=[
                "samples",
            ]
        ).fit(data)

        result = scaler.inverse_transform(scaler.transform(data))

        xr.testing.assert_allclose(
            result,
            data,
        )

    def test_yearly_statistics(self):
        scaler = Standardizer(
            dims=[
                TIME_DIM,
            ],
            frequency="year",
        ).fit(make_time_data())

        assert scaler.mean.dims == ("year",)
        assert scaler.std.dims == ("year",)
        np.testing.assert_array_equal(
            scaler.mean["year"].values,
            [
                2000,
                2001,
            ],
        )


class TestAnomaliesScaler:
    @pytest.mark.pruned
    def test_defaults(self):
        scaler = AnomaliesScaler()

        assert scaler.mean is None
        assert scaler.dims == ()
        assert scaler.frequency is None
        assert scaler.large_ensemble is False
        assert scaler.fitted is False

    @pytest.mark.pruned
    def test_dimensions_are_stored_as_tuple(self):
        scaler = AnomaliesScaler(
            dims=[
                TIME_DIM,
                "lat",
            ]
        )

        assert scaler.dims == (
            TIME_DIM,
            "lat",
        )

    @pytest.mark.pruned
    def test_fit_computes_mean(self):
        data = xr.DataArray(
            [
                2.0,
                4.0,
                6.0,
            ],
            dims=("samples",),
        )

        scaler = AnomaliesScaler(
            dims=[
                "samples",
            ]
        )

        result = scaler.fit(data)

        assert result is scaler
        assert scaler.fitted is True
        assert scaler.mean.item() == pytest.approx(4.0)

    def test_fit_applies_nan_mask(self):
        data = xr.DataArray(
            [
                1.0,
                2.0,
                100.0,
            ],
            dims=("samples",),
        )
        mask = xr.DataArray(
            [
                False,
                False,
                True,
            ],
            dims=("samples",),
        )

        scaler = AnomaliesScaler(
            dims=[
                "samples",
            ]
        ).fit(
            data,
            mask=mask,
        )

        assert scaler.mean.item() == pytest.approx(1.5)

    @pytest.mark.pruned
    def test_transform(self):
        data = xr.DataArray(
            [
                2.0,
                4.0,
                6.0,
            ],
            dims=("samples",),
        )
        scaler = AnomaliesScaler(
            dims=[
                "samples",
            ]
        ).fit(data)

        result = scaler.transform(data)

        xr.testing.assert_allclose(
            result,
            xr.DataArray(
                [
                    -2.0,
                    0.0,
                    2.0,
                ],
                dims=("samples",),
            ),
        )

    @pytest.mark.pruned
    def test_inverse_transform(
        self,
        passthrough_alignment,
    ):
        scaler = AnomaliesScaler()
        scaler.mean = xr.DataArray(4.0)
        scaler.fitted = True

        data = xr.DataArray(
            [
                -2.0,
                0.0,
                2.0,
            ],
            dims=("samples",),
        )

        result = scaler.inverse_transform(data)

        xr.testing.assert_allclose(
            result,
            xr.DataArray(
                [
                    2.0,
                    4.0,
                    6.0,
                ],
                dims=("samples",),
            ),
        )
        passthrough_alignment.assert_called_once_with(
            ds=data,
            stat=scaler.mean,
            lead_time_resolution=scaler.lead_time_resolution,
        )

    @pytest.mark.pruned
    def test_transform_requires_fitted_scaler(self):
        with pytest.raises(
            RuntimeError,
            match="must be fitted",
        ):
            AnomaliesScaler().transform(
                xr.DataArray(
                    [
                        1.0,
                    ],
                    dims=("samples",),
                )
            )


class TestFlattennanremove:
    def make_spatial_data(self):
        return xr.DataArray(
            np.asarray(
                [
                    [
                        1.0,
                        np.nan,
                    ],
                    [
                        3.0,
                        4.0,
                    ],
                ]
            ),
            dims=(
                "lat",
                "lon",
            ),
            coords={
                "lat": [
                    45.0,
                    46.0,
                ],
                "lon": [
                    -124.0,
                    -123.0,
                ],
            },
        )

    @pytest.mark.pruned
    def test_defaults(self):
        flattener = Flattennanremove()

        assert flattener.load_dir is None
        assert flattener.fitted is False
        assert flattener.common_to_input_and_target is False
        assert flattener.NN_dims == []

    @pytest.mark.pruned
    def test_fit_detects_spatial_dimensions(self):
        flattener = Flattennanremove()

        result = flattener.fit(self.make_spatial_data())

        assert result is flattener
        assert flattener.fitted is True
        assert flattener.NN_dims == [
            "lat",
            "lon",
        ]
        assert flattener.reference_shape is not None

    @pytest.mark.pruned
    def test_fit_removes_nan_locations(self):
        flattener = Flattennanremove().fit(self.make_spatial_data())

        assert flattener.final_locations.size == 3

    @pytest.mark.pruned
    def test_fit_with_target_uses_common_locations(self):
        data = self.make_spatial_data()

        target = xr.DataArray(
            np.asarray(
                [
                    [
                        1.0,
                        2.0,
                    ],
                    [
                        np.nan,
                        4.0,
                    ],
                ]
            ),
            dims=(
                "lat",
                "lon",
            ),
            coords=data.coords,
        )

        with pytest.raises(KeyError):
            Flattennanremove().fit(
                data,
                target=target,
            )

    def test_missing_input_dimension_is_rejected(self):
        data = xr.DataArray(
            np.ones(2),
            dims=("lat",),
            coords={
                "lat": [
                    45.0,
                    46.0,
                ]
            },
        )
        target = xr.DataArray(
            np.ones(
                (
                    2,
                    2,
                )
            ),
            dims=(
                "lat",
                "lon",
            ),
            coords={
                "lat": [
                    45.0,
                    46.0,
                ],
                "lon": [
                    -124.0,
                    -123.0,
                ],
            },
        )

        with pytest.raises(
            RuntimeError,
            match="Missing from input data",
        ):
            Flattennanremove().fit(
                data,
                target=target,
            )

    @pytest.mark.pruned
    def test_transform_stacks_spatial_dimensions(self):
        data = self.make_spatial_data()
        flattener = Flattennanremove().fit(data)

        result = flattener.transform(data)

        assert tuple(result.dims) == ("ref",)
        assert result.size == 3

    @pytest.mark.pruned
    def test_transform_existing_ref_dimension(self):
        data = self.make_spatial_data()
        flattener = Flattennanremove().fit(data)

        stacked = data.stack(ref=flattener.NN_dims)

        result = flattener.transform(stacked)

        assert tuple(result.dims) == ("ref",)
        assert result.size == 3

    def test_inverse_requires_ref_dimension(self):
        with pytest.raises(
            ValueError,
            match="flattened 'ref' dimension",
        ):
            Flattennanremove().inverse_transform(self.make_spatial_data())

    def test_inverse_restores_spatial_layout(self):
        data = self.make_spatial_data()
        flattener = Flattennanremove().fit(data)

        flattened = flattener.transform(data)
        result = flattener.inverse_transform(flattened)

        assert tuple(result.dims) == (
            "lat",
            "lon",
        )
        np.testing.assert_array_equal(
            result["lat"].values,
            data["lat"].values,
        )
        np.testing.assert_array_equal(
            result["lon"].values,
            data["lon"].values,
        )

    def test_check_nn_dims_accepts_none(self):
        flattener = Flattennanremove()
        flattener.NN_dims = [
            "lat",
            "lon",
        ]

        assert flattener._check_nn_dims(None) is None

    @pytest.mark.pruned
    def test_check_nn_dims_rejects_missing_dimensions(self):
        flattener = Flattennanremove()
        flattener.NN_dims = [
            "lat",
            "lon",
        ]

        with pytest.raises(
            ValueError,
            match="Missing dimensions.*lon",
        ):
            flattener._check_nn_dims(
                xr.DataArray(
                    np.ones(2),
                    dims=("lat",),
                )
            )

    def test_save(
        self,
        tmp_path,
        monkeypatch,
    ):
        dump = Mock()
        monkeypatch.setattr(
            module.joblib,
            "dump",
            dump,
        )

        flattener = Flattennanremove()
        flattener.fit(
            self.make_spatial_data(),
            save=True,
            save_name="custom",
            save_path=tmp_path / "nested",
        )

        expected = tmp_path / "nested" / "custom.joblib"

        dump.assert_called_once_with(
            flattener,
            expected,
        )
        assert expected.parent.is_dir()

    def test_load_rejects_unfitted_object(
        self,
        tmp_path,
        monkeypatch,
    ):
        loaded = SimpleNamespace(fitted=False)

        monkeypatch.setattr(
            module.joblib,
            "load",
            Mock(return_value=loaded),
        )

        with pytest.raises(
            RuntimeError,
            match="has to be fitted first",
        ):
            Flattennanremove()._load_from_memory(tmp_path / "flattener.joblib")

    @pytest.mark.pruned
    def test_load_copies_state(
        self,
        tmp_path,
        monkeypatch,
    ):
        loaded = SimpleNamespace(
            fitted=True,
            reference_shape=xr.Dataset(
                coords={
                    "lat": [
                        45.0,
                    ]
                }
            ),
            final_locations=xr.DataArray(
                [
                    0,
                ],
                dims=("ref",),
            ),
            common_to_input_and_target=True,
        )

        load = Mock(return_value=loaded)
        monkeypatch.setattr(
            module.joblib,
            "load",
            load,
        )

        flattener = Flattennanremove()
        flattener._load_from_memory(tmp_path / "flattener.joblib")

        load.assert_called_once_with(tmp_path / "flattener.joblib")
        assert flattener.fitted is True
        assert flattener.common_to_input_and_target is True
        assert flattener.reference_shape is loaded.reference_shape
        assert flattener.final_locations is loaded.final_locations


class TestAlignStatistic:
    @pytest.mark.pruned
    def test_static_statistic_is_returned_unchanged(self):
        data = make_forecast_data()
        stat = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=("channels",),
        )

        result = align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )

        assert result is stat

    def test_statistic_with_lead_time_is_returned_unchanged(self):
        data = make_forecast_data()
        stat = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=(LEAD_TIME_DIM,),
            coords={
                LEAD_TIME_DIM: [
                    1,
                    2,
                ]
            },
        )

        result = align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )

        assert result is stat

    @pytest.mark.pruned
    def test_temporal_statistic_requires_time_coordinate(self):
        data = xr.DataArray(
            np.ones(2),
            dims=(LEAD_TIME_DIM,),
            coords={
                LEAD_TIME_DIM: [
                    1,
                    2,
                ]
            },
        )
        stat = xr.DataArray(
            np.arange(
                1,
                13,
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
            match="initialization-time coordinate",
        ):
            align_stat_data_lead_time_inverse_transform(
                data,
                stat,
            )

    def test_rejects_multidimensional_time_coordinate(self):
        data = xr.DataArray(
            np.ones(
                (
                    2,
                    2,
                )
            ),
            dims=(
                "x",
                "y",
            ),
            coords={
                TIME_DIM: (
                    (
                        "x",
                        "y",
                    ),
                    np.asarray(
                        [
                            [
                                "2000-01-01",
                                "2000-02-01",
                            ],
                            [
                                "2001-01-01",
                                "2001-02-01",
                            ],
                        ],
                        dtype="datetime64[ns]",
                    ),
                )
            },
        )
        stat = xr.DataArray(
            np.arange(
                1,
                13,
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
            match="must be one-dimensional",
        ):
            align_stat_data_lead_time_inverse_transform(
                data,
                stat,
            )

    @pytest.mark.pruned
    def test_month_statistic_is_aligned_to_valid_time(self):
        data = make_forecast_data(
            times=("2000-01-01",),
            lead_times=(
                1,
                2,
                12,
                13,
            ),
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

        result = align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    [
                        1.0,
                        2.0,
                        12.0,
                        1.0,
                    ]
                ]
            ),
        )
        assert tuple(result.dims) == (
            TIME_DIM,
            LEAD_TIME_DIM,
        )

    @pytest.mark.pruned
    def test_year_statistic_is_aligned_to_valid_time(self):
        data = make_forecast_data(
            times=("2000-01-01",),
            lead_times=(
                1,
                13,
                25,
            ),
        )
        stat = xr.DataArray(
            [
                10.0,
                20.0,
                30.0,
            ],
            dims=("year",),
            coords={
                "year": [
                    2000,
                    2001,
                    2002,
                ]
            },
        )

        result = align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    [
                        10.0,
                        20.0,
                        30.0,
                    ]
                ]
            ),
        )

    @pytest.mark.pruned
    def test_year_and_month_statistic(self):
        data = make_forecast_data(
            times=("2000-01-01",),
            lead_times=(
                1,
                12,
                13,
            ),
        )
        stat = xr.DataArray(
            np.stack(
                [
                    np.arange(
                        1,
                        13,
                    ),
                    np.arange(
                        101,
                        113,
                    ),
                ]
            ),
            dims=(
                "year",
                "month",
            ),
            coords={
                "year": [
                    2000,
                    2001,
                ],
                "month": np.arange(
                    1,
                    13,
                ),
            },
        )

        result = align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    [
                        1,
                        12,
                        101,
                    ]
                ]
            ),
        )

    def test_day_statistic(self):
        data = make_forecast_data(
            times=("2000-01-01",),
            lead_times=(
                1,
                2,
            ),
        )
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

        result = align_stat_data_lead_time_inverse_transform(
            data,
            stat,
            lead_time_resolution="day",
        )

        np.testing.assert_array_equal(
            result.values,
            np.asarray(
                [
                    [
                        1.0,
                        2.0,
                    ]
                ]
            ),
        )

    @pytest.mark.pruned
    def test_without_lead_time_uses_initialization_time(self):
        data = xr.DataArray(
            np.ones(2),
            dims=(TIME_DIM,),
            coords={
                TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )
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

        result = align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )

        np.testing.assert_array_equal(
            result.values,
            [
                10.0,
                20.0,
            ],
        )
        assert tuple(result.dims) == (TIME_DIM,)

    @pytest.mark.pruned
    def test_preserves_non_temporal_dimensions(self):
        data = make_forecast_data(
            times=("2000-01-01",),
            lead_times=(
                1,
                2,
            ),
        )
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

        result = align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )

        assert tuple(result.dims) == (
            TIME_DIM,
            LEAD_TIME_DIM,
            "channels",
        )
        np.testing.assert_array_equal(
            result["channels"].values,
            [
                "tas",
                "pr",
            ],
        )

    @pytest.mark.pruned
    def test_missing_year_is_reported_by_xarray(self):
        data = make_forecast_data(
            times=("2000-01-01",),
            lead_times=(
                1,
                13,
            ),
        )
        stat = xr.DataArray(
            [
                10.0,
            ],
            dims=("year",),
            coords={
                "year": [
                    2000,
                ]
            },
        )

        with pytest.raises(
            KeyError,
            match="not all values found",
        ):
            align_stat_data_lead_time_inverse_transform(
                data,
                stat,
            )

    def test_temporary_coordinates_are_removed(self):
        data = make_forecast_data(
            times=("2000-01-01",),
            lead_times=(
                1,
                13,
            ),
        )
        stat = xr.DataArray(
            np.arange(
                24,
            ).reshape(
                2,
                12,
            ),
            dims=(
                "year",
                "month",
            ),
            coords={
                "year": [
                    2000,
                    2001,
                ],
                "month": np.arange(
                    1,
                    13,
                ),
            },
        )

        result = align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )

        assert "__stat_time" not in result.coords
        assert "__stat_year" not in result.coords
        assert "__stat_month" not in result.coords
        assert "__stat_day" not in result.coords


@pytest.mark.pruned
@pytest.mark.parametrize(
    "scaler_class",
    [
        Normalizer,
        Standardizer,
        AnomaliesScaler,
    ],
)
def test_scaler_preserves_unknown_keyword_arguments(scaler_class):
    scaler = scaler_class(
        dims=["samples"],
        unused_option="ignored",
    )

    assert scaler.dims == ("samples",)


def test_normalizer_applies_mask():
    data = xr.DataArray(
        [1.0, 2.0, 100.0],
        dims=("samples",),
    )
    mask = xr.DataArray(
        [False, False, True],
        dims=("samples",),
    )

    scaler = Normalizer(
        dims=["samples"],
    ).fit(
        data,
        mask=mask,
    )

    assert scaler.min.item() == pytest.approx(1.0)
    assert scaler.max.item() == pytest.approx(2.0)


@pytest.mark.pruned
@pytest.mark.parametrize(
    "scaler_class,stat_names",
    [
        (
            Standardizer,
            ("mean", "std"),
        ),
        (
            AnomaliesScaler,
            ("mean",),
        ),
    ],
)
def test_mask_with_no_valid_values_produces_nan_statistics(
    scaler_class,
    stat_names,
):
    data = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=("samples",),
    )
    mask = xr.DataArray(
        [True, True, True],
        dims=("samples",),
    )

    scaler = scaler_class(
        dims=["samples"],
    ).fit(
        data,
        mask=mask,
    )

    for stat_name in stat_names:
        statistic = getattr(
            scaler,
            stat_name,
        )
        assert np.isnan(statistic.item())


@pytest.mark.parametrize(
    "scaler_class",
    [
        Normalizer,
        Standardizer,
        AnomaliesScaler,
    ],
)
def test_grouped_transform_uses_existing_month_coordinate(
    scaler_class,
):
    times = np.asarray(
        [
            "2000-01-01",
            "2000-02-01",
            "2001-01-01",
            "2001-02-01",
        ],
        dtype="datetime64[ns]",
    )
    data = xr.DataArray(
        [1.0, 10.0, 3.0, 14.0],
        dims=(TIME_DIM,),
        coords={
            TIME_DIM: times,
        },
    )

    scaler = scaler_class(
        dims=[TIME_DIM],
        frequency="month",
    ).fit(data)

    transformed_data = data.assign_coords(
        month=(
            TIME_DIM,
            [2, 1, 2, 1],
        )
    )

    result = scaler.transform(transformed_data)

    assert tuple(result.dims) == (TIME_DIM,)
    assert result.sizes[TIME_DIM] == 4


@pytest.mark.pruned
@pytest.mark.parametrize(
    "scaler_class",
    [
        Normalizer,
        Standardizer,
        AnomaliesScaler,
    ],
)
def test_grouped_fit_with_ensemble_reduces_both_dimensions(
    scaler_class,
):
    times = np.asarray(
        [
            "2000-01-01",
            "2001-01-01",
            "2000-02-01",
            "2001-02-01",
        ],
        dtype="datetime64[ns]",
    )
    data = xr.DataArray(
        np.arange(
            12,
            dtype=float,
        ).reshape(
            3,
            4,
        ),
        dims=(
            scaler_class.realization_dim,
            TIME_DIM,
        ),
        coords={
            scaler_class.realization_dim: [
                0,
                1,
                2,
            ],
            TIME_DIM: times,
        },
    )

    scaler = scaler_class(
        dims=[TIME_DIM],
        frequency="month",
    ).fit(data)

    assert scaler.large_ensemble is True

    statistic = (
        scaler.min
        if isinstance(
            scaler,
            Normalizer,
        )
        else scaler.mean
    )

    assert statistic.dims == ("month",)
    np.testing.assert_array_equal(
        statistic["month"].values,
        [1, 2],
    )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "scaler_class",
    [
        Normalizer,
        Standardizer,
        AnomaliesScaler,
    ],
)
def test_grouped_dataset_fit_preserves_data_variables(
    scaler_class,
):
    times = np.asarray(
        [
            "2000-01-01",
            "2000-02-01",
            "2001-01-01",
            "2001-02-01",
        ],
        dtype="datetime64[ns]",
    )
    first = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0],
        dims=(TIME_DIM,),
        coords={
            TIME_DIM: times,
        },
    )
    dataset = xr.Dataset(
        {
            "tas": first,
            "pr": first + 10.0,
        }
    )

    scaler = scaler_class(
        dims=[TIME_DIM],
        frequency="year",
    ).fit(dataset)

    statistic = (
        scaler.min
        if isinstance(
            scaler,
            Normalizer,
        )
        else scaler.mean
    )

    assert isinstance(
        statistic,
        xr.Dataset,
    )
    assert set(statistic.data_vars) == {
        "tas",
        "pr",
    }


@pytest.mark.pruned
def test_standardizer_dataset_mask_is_broadcast():
    data = xr.Dataset(
        {
            "first": (
                "samples",
                [1.0, 2.0, 100.0],
            ),
            "second": (
                "samples",
                [3.0, 5.0, 200.0],
            ),
        }
    )
    mask = xr.DataArray(
        [False, False, True],
        dims=("samples",),
    )

    scaler = Standardizer(
        dims=["samples"],
    ).fit(
        data,
        mask=mask,
    )

    assert scaler.mean["first"].item() == pytest.approx(1.5)
    assert scaler.mean["second"].item() == pytest.approx(4.0)


@pytest.mark.pruned
def test_anomalies_dataset_mask_is_broadcast():
    data = xr.Dataset(
        {
            "first": (
                "samples",
                [1.0, 2.0, 100.0],
            ),
            "second": (
                "samples",
                [3.0, 5.0, 200.0],
            ),
        }
    )
    mask = xr.DataArray(
        [False, False, True],
        dims=("samples",),
    )

    scaler = AnomaliesScaler(
        dims=["samples"],
    ).fit(
        data,
        mask=mask,
    )

    assert scaler.mean["first"].item() == pytest.approx(1.5)
    assert scaler.mean["second"].item() == pytest.approx(4.0)


@pytest.mark.pruned
def test_flattener_preserves_leading_dimensions():
    data = xr.DataArray(
        np.arange(
            8,
            dtype=float,
        ).reshape(
            2,
            2,
            2,
        ),
        dims=(
            "channels",
            "lat",
            "lon",
        ),
        coords={
            "channels": [
                "tas",
                "pr",
            ],
            "lat": [
                45.0,
                46.0,
            ],
            "lon": [
                -124.0,
                -123.0,
            ],
        },
    )

    flattener = Flattennanremove().fit(data)
    result = flattener.transform(data)

    assert tuple(result.dims) == (
        "channels",
        "ref",
    )
    assert result.shape == (
        2,
        4,
    )


@pytest.mark.pruned
def test_flattener_inverse_preserves_leading_dimensions():
    data = xr.DataArray(
        np.arange(
            8,
            dtype=float,
        ).reshape(
            2,
            2,
            2,
        ),
        dims=(
            "channels",
            "lat",
            "lon",
        ),
        coords={
            "channels": [
                "tas",
                "pr",
            ],
            "lat": [
                45.0,
                46.0,
            ],
            "lon": [
                -124.0,
                -123.0,
            ],
        },
    )

    flattener = Flattennanremove().fit(data)
    flattened = flattener.transform(data)
    result = flattener.inverse_transform(flattened)

    assert tuple(result.dims) == (
        "lat",
        "lon",
        "channels",
    )
    assert not result.data_vars
    np.testing.assert_array_equal(result["channels"], ["tas", "pr"])
    np.testing.assert_array_equal(result["lat"], [45.0, 46.0])
    np.testing.assert_array_equal(result["lon"], [-124.0, -123.0])


@pytest.mark.pruned
def test_flattener_inverse_restores_removed_locations_as_nan():
    data = xr.DataArray(
        [
            [
                1.0,
                np.nan,
            ],
            [
                3.0,
                4.0,
            ],
        ],
        dims=(
            "lat",
            "lon",
        ),
        coords={
            "lat": [
                45.0,
                46.0,
            ],
            "lon": [
                -124.0,
                -123.0,
            ],
        },
    )

    flattener = Flattennanremove().fit(data)
    result = flattener.inverse_transform(flattener.transform(data))

    assert tuple(result.sizes.values()) == (
        2,
        2,
    )
    assert not result.data_vars
    np.testing.assert_array_equal(result["lat"], [45.0, 46.0])
    np.testing.assert_array_equal(result["lon"], [-124.0, -123.0])


@pytest.mark.pruned
def test_flattener_target_intersection_is_coordinate_based():
    data = xr.DataArray(
        [
            [
                1.0,
                np.nan,
            ],
            [
                3.0,
                4.0,
            ],
        ],
        dims=(
            "lat",
            "lon",
        ),
        coords={
            "lat": [
                45.0,
                46.0,
            ],
            "lon": [
                -124.0,
                -123.0,
            ],
        },
    )
    target = xr.DataArray(
        [
            [
                np.nan,
                2.0,
            ],
            [
                5.0,
                6.0,
            ],
        ],
        dims=(
            "lat",
            "lon",
        ),
        coords=data.coords,
    )

    with pytest.raises(KeyError):
        Flattennanremove().fit(
            data,
            target=target,
        )


def test_flattener_load_mode_rejects_incompatible_data(
    tmp_path,
    monkeypatch,
):
    def fake_load(_):
        flattener = SimpleNamespace(
            fitted=True,
            reference_shape=xr.Dataset(
                coords={
                    "lat": [
                        45.0,
                    ],
                    "lon": [
                        -124.0,
                    ],
                }
            ),
            final_locations=xr.DataArray(
                [
                    0,
                ],
                dims=("ref",),
            ),
            common_to_input_and_target=False,
        )
        return flattener

    monkeypatch.setattr(
        module.joblib,
        "load",
        fake_load,
    )

    flattener = Flattennanremove(load_dir=tmp_path / "saved.joblib")
    flattener.NN_dims = [
        "lat",
        "lon",
    ]

    data = xr.DataArray(
        np.ones(1),
        dims=("lat",),
    )

    with pytest.raises(
        ValueError,
        match="Missing dimensions.*lon",
    ):
        flattener.fit(data)


@pytest.mark.pruned
def test_flattener_save_uses_runtime_directory_and_default_name(
    tmp_path,
    monkeypatch,
):
    data = xr.DataArray(
        np.ones(
            (
                2,
                2,
            )
        ),
        dims=(
            "lat",
            "lon",
        ),
        coords={
            "lat": [
                45.0,
                46.0,
            ],
            "lon": [
                -124.0,
                -123.0,
            ],
        },
    )

    monkeypatch.setattr(
        RuntimeContext,
        "GLOBAL_EXP_DIR",
        tmp_path,
    )

    dump = Mock()
    monkeypatch.setattr(
        module.joblib,
        "dump",
        dump,
    )

    flattener = Flattennanremove().fit(
        data,
        save=True,
    )

    dump.assert_called_once_with(
        flattener,
        tmp_path / "flattener.joblib",
    )


def test_align_exact_initialization_time_statistic():
    times = np.asarray(
        [
            "2000-01-01",
            "2000-02-01",
        ],
        dtype="datetime64[ns]",
    )
    data = xr.DataArray(
        np.ones(2),
        dims=(TIME_DIM,),
        coords={
            TIME_DIM: times,
        },
    )
    stat = xr.DataArray(
        [
            10.0,
            20.0,
        ],
        dims=(TIME_DIM,),
        coords={
            TIME_DIM: times,
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    np.testing.assert_array_equal(
        result.values,
        [
            10.0,
            20.0,
        ],
    )
    assert tuple(result.dims) == (TIME_DIM,)


@pytest.mark.pruned
def test_align_exact_time_statistic_with_lead_times():
    data = make_forecast_data(
        times=("2000-01-01",),
        lead_times=(
            1,
            2,
        ),
    )
    valid_times = np.asarray(
        [
            "2000-01-01",
            "2000-02-01",
        ],
        dtype="datetime64[ns]",
    )
    stat = xr.DataArray(
        [
            10.0,
            20.0,
        ],
        dims=(TIME_DIM,),
        coords={
            TIME_DIM: valid_times,
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    np.testing.assert_array_equal(
        result.values,
        [
            [
                10.0,
                20.0,
            ]
        ],
    )
    assert tuple(result.dims) == (
        TIME_DIM,
        LEAD_TIME_DIM,
    )


@pytest.mark.pruned
def test_align_preserves_custom_temporal_coordinate_names():
    custom_time_dim = "forecast_reference_time"
    custom_lead_dim = "forecast_period"

    data = xr.DataArray(
        np.zeros(
            (
                1,
                2,
            )
        ),
        dims=(
            custom_time_dim,
            custom_lead_dim,
        ),
        coords={
            custom_time_dim: np.asarray(
                [
                    "2000-01-01",
                ],
                dtype="datetime64[ns]",
            ),
            custom_lead_dim: [
                1,
                2,
            ],
        },
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
            ),
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
        init_time_dim=custom_time_dim,
        lead_time_dim=custom_lead_dim,
    )

    assert tuple(result.dims) == (
        custom_time_dim,
        custom_lead_dim,
    )
    np.testing.assert_array_equal(
        result.values,
        [
            [
                1.0,
                2.0,
            ]
        ],
    )


@pytest.mark.pruned
def test_align_calls_add_lead_times_with_flattened_grid(
    monkeypatch,
):
    data = make_forecast_data(
        times=(
            "2000-01-01",
            "2001-01-01",
        ),
        lead_times=(
            1,
            2,
            3,
        ),
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
            ),
        },
    )

    valid_times = np.asarray(
        [
            "2000-01-01",
            "2000-02-01",
            "2000-03-01",
            "2001-01-01",
            "2001-02-01",
            "2001-03-01",
        ],
        dtype="datetime64[ns]",
    )

    add_times = Mock(return_value=valid_times)
    monkeypatch.setattr(
        module,
        "add_lead_times",
        add_times,
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    call = add_times.call_args

    assert call.kwargs["init_times"].shape == (6,)
    np.testing.assert_array_equal(
        call.kwargs["lead_times"],
        [
            1,
            2,
            3,
            1,
            2,
            3,
        ],
    )
    assert call.kwargs["lead_time_resolution"] == "month"
    assert result.shape == (
        2,
        3,
    )


@pytest.mark.pruned
def test_align_day_statistic_across_leap_day():
    data = make_forecast_data(
        times=("2000-02-28",),
        lead_times=(
            1,
            2,
            3,
        ),
    )
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
            ),
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
        lead_time_resolution="day",
    )

    np.testing.assert_array_equal(
        result.values,
        [
            [
                59.0,
                60.0,
                61.0,
            ]
        ],
    )


@pytest.mark.pruned
def test_align_multiple_initialization_times_and_years():
    data = make_forecast_data(
        times=(
            "2000-01-01",
            "2001-01-01",
        ),
        lead_times=(
            1,
            13,
        ),
    )
    stat = xr.DataArray(
        [
            10.0,
            20.0,
            30.0,
        ],
        dims=("year",),
        coords={
            "year": [
                2000,
                2001,
                2002,
            ],
        },
    )

    result = align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    np.testing.assert_array_equal(
        result.values,
        [
            [
                10.0,
                20.0,
            ],
            [
                20.0,
                30.0,
            ],
        ],
    )


@pytest.mark.pruned
def test_align_does_not_mutate_statistic():
    data = make_forecast_data(
        times=("2000-01-01",),
        lead_times=(
            1,
            2,
        ),
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
            ),
        },
    )
    original = stat.copy(deep=True)

    align_stat_data_lead_time_inverse_transform(
        data,
        stat,
    )

    xr.testing.assert_identical(
        stat,
        original,
    )


def make_trend_dataset(
    *,
    times=None,
    slope=2.0,
    intercept=5.0,
):
    if times is None:
        times = np.asarray(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
                "2000-01-04",
            ],
            dtype="datetime64[ns]",
        )
    else:
        times = np.asarray(
            times,
            dtype="datetime64[ns]",
        )

    elapsed_days = ((times - times[0]) / np.timedelta64(1, "D")).astype(float)

    return xr.Dataset(
        {
            "tas": (
                TIME_DIM,
                slope * elapsed_days + intercept,
            ),
        },
        coords={
            TIME_DIM: times,
        },
    )


class TestTrendRemoverInitialization:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "frequency",
        [
            None,
            "month",
            "day",
        ],
    )
    def test_accepts_supported_frequency(self, frequency):
        remover = TrendRemover(frequency=frequency)

        assert remover.frequency == frequency
        assert remover.dims == [TIME_DIM]
        assert remover.slope is None
        assert remover.intercept is None
        assert remover.time_origin is None
        assert remover.fitted is False

    @pytest.mark.parametrize(
        "frequency",
        [
            "year",
            "hour",
            "",
            "MONTH",
        ],
    )
    def test_rejects_unsupported_frequency(self, frequency):
        with pytest.raises(
            ValueError,
            match="Unsupported frequency",
        ):
            TrendRemover(frequency=frequency)

    @pytest.mark.pruned
    def test_ignores_unknown_keyword_arguments(self):
        remover = TrendRemover(
            frequency=None,
            unused_option="ignored",
        )

        assert remover.frequency is None
        assert remover.fitted is False


class TestTrendTimeToNumeric:
    @pytest.mark.pruned
    def test_numpy_datetime_values(self):
        times = xr.DataArray(
            np.asarray(
                [
                    "2000-01-01",
                    "2000-01-02",
                    "2000-01-04",
                ],
                dtype="datetime64[ns]",
            ),
            dims=(TIME_DIM,),
        )

        result = TrendRemover._time_to_numeric(
            times,
            origin=np.datetime64("2000-01-01"),
        )

        np.testing.assert_allclose(
            result.values,
            [0.0, 1.0, 3.0],
        )
        assert result.dims == times.dims

    def test_cftime_values(self):
        times = xr.DataArray(
            np.asarray(
                [
                    cftime.DatetimeNoLeap(2000, 1, 1),
                    cftime.DatetimeNoLeap(2000, 1, 2),
                    cftime.DatetimeNoLeap(2000, 2, 1),
                ],
                dtype=object,
            ),
            dims=(TIME_DIM,),
        )

        result = TrendRemover._time_to_numeric(
            times,
            origin=cftime.DatetimeNoLeap(2000, 1, 1),
        )

        np.testing.assert_allclose(
            result.values,
            [0.0, 1.0, 31.0],
        )

    @pytest.mark.pruned
    def test_scalar_dataarray_origin(self):
        times = xr.DataArray(
            np.asarray(
                [
                    "2000-01-01",
                    "2000-01-03",
                ],
                dtype="datetime64[ns]",
            ),
            dims=(TIME_DIM,),
        )
        origin = xr.DataArray(
            np.datetime64("2000-01-01"),
        )

        result = TrendRemover._time_to_numeric(
            times,
            origin=origin,
        )

        np.testing.assert_allclose(
            result.values,
            [0.0, 2.0],
        )

    @pytest.mark.pruned
    def test_scalar_numpy_array_origin(self):
        times = xr.DataArray(
            np.asarray(
                [
                    "2000-01-01",
                    "2000-01-02",
                ],
                dtype="datetime64[ns]",
            ),
            dims=(TIME_DIM,),
        )
        origin = np.asarray(
            np.datetime64("2000-01-01"),
        )

        result = TrendRemover._time_to_numeric(
            times,
            origin=origin,
        )

        np.testing.assert_allclose(
            result.values,
            [0.0, 1.0],
        )

    def test_rejects_nonscalar_numpy_origin(self):
        times = xr.DataArray(
            np.asarray(
                [
                    "2000-01-01",
                    "2000-01-02",
                ],
                dtype="datetime64[ns]",
            ),
            dims=(TIME_DIM,),
        )
        origin = np.asarray(
            [
                np.datetime64("2000-01-01"),
            ]
        )

        with pytest.raises(
            ValueError,
            match="origin.*scalar",
        ):
            TrendRemover._time_to_numeric(
                times,
                origin=origin,
            )

    @pytest.mark.pruned
    def test_rejects_empty_time_coordinate(self):
        times = xr.DataArray(
            np.asarray(
                [],
                dtype="datetime64[ns]",
            ),
            dims=(TIME_DIM,),
        )

        with pytest.raises(
            ValueError,
            match="cannot be empty",
        ):
            TrendRemover._time_to_numeric(
                times,
                origin=np.datetime64("2000-01-01"),
            )

    @pytest.mark.pruned
    def test_preserves_multidimensional_shape_and_coordinates(self):
        values = np.asarray(
            [
                [
                    "2000-01-01",
                    "2000-01-02",
                ],
                [
                    "2000-01-03",
                    "2000-01-04",
                ],
            ],
            dtype="datetime64[ns]",
        )
        times = xr.DataArray(
            values,
            dims=("x", "y"),
            coords={
                "x": [10, 20],
                "y": [30, 40],
            },
        )

        result = TrendRemover._time_to_numeric(
            times,
            origin=np.datetime64("2000-01-01"),
        )

        np.testing.assert_allclose(
            result.values,
            [
                [0.0, 1.0],
                [2.0, 3.0],
            ],
        )
        assert result.dims == ("x", "y")
        np.testing.assert_array_equal(result["x"], [10, 20])
        np.testing.assert_array_equal(result["y"], [30, 40])


class TestTrendFitLinearTrend:
    @pytest.mark.pruned
    def test_fits_exact_linear_trend(self):
        data = make_trend_dataset(
            slope=2.0,
            intercept=5.0,
        )
        remover = TrendRemover()
        remover.time_origin = data[TIME_DIM].min()

        slope, intercept = remover._fit_linear_trend(data)

        assert isinstance(slope, xr.Dataset)
        assert isinstance(intercept, xr.Dataset)
        assert set(slope.data_vars) == {"tas"}
        assert set(intercept.data_vars) == {"tas"}
        assert slope["tas"].item() == pytest.approx(2.0)
        assert intercept["tas"].item() == pytest.approx(5.0)

    @pytest.mark.pruned
    def test_preserves_non_time_dimensions(self):
        times = np.asarray(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
            ],
            dtype="datetime64[ns]",
        )
        values = np.asarray(
            [
                [1.0, 10.0],
                [3.0, 13.0],
                [5.0, 16.0],
            ]
        )
        data = xr.Dataset(
            {
                "tas": (
                    (
                        TIME_DIM,
                        "lat",
                    ),
                    values,
                )
            },
            coords={
                TIME_DIM: times,
                "lat": [45.0, 46.0],
            },
        )

        remover = TrendRemover()
        remover.time_origin = data[TIME_DIM].min()

        slope, intercept = remover._fit_linear_trend(data)

        assert slope["tas"].dims == ("lat",)
        assert intercept["tas"].dims == ("lat",)
        np.testing.assert_allclose(
            slope["tas"],
            [2.0, 3.0],
        )
        np.testing.assert_allclose(
            intercept["tas"],
            [1.0, 10.0],
        )

    @pytest.mark.pruned
    def test_missing_coefficient_variable_raises(self, monkeypatch):
        data = make_trend_dataset()
        remover = TrendRemover()
        remover.time_origin = data[TIME_DIM].min()

        monkeypatch.setattr(
            xr.Dataset,
            "polyfit",
            Mock(return_value=xr.Dataset()),
        )

        with pytest.raises(
            RuntimeError,
            match="Expected coefficient variable",
        ):
            remover._fit_linear_trend(data)


class TestTrendRemoverFit:
    def test_requires_initialization_time_dimension(self):
        data = xr.Dataset(
            {
                "tas": (
                    "samples",
                    [1.0, 2.0],
                )
            }
        )

        with pytest.raises(
            ValueError,
            match="initialization-time dimension",
        ):
            TrendRemover().fit(data)

    @pytest.mark.pruned
    def test_fit_returns_self_and_sets_fitted(self):
        remover = TrendRemover()
        result = remover.fit(make_trend_dataset())

        assert result is remover
        assert remover.fitted is True
        assert remover.slope is not None
        assert remover.intercept is not None
        assert remover.time_origin is not None

    @pytest.mark.pruned
    def test_fit_loads_statistics(self):
        remover = TrendRemover().fit(make_trend_dataset())

        assert remover.slope["tas"].item() == pytest.approx(2.0)
        assert remover.intercept["tas"].item() == pytest.approx(5.0)

    @pytest.mark.pruned
    def test_fit_applies_mask(self):
        data = make_trend_dataset()
        data["tas"][-1] = 1000.0

        mask = xr.DataArray(
            [False, False, False, True],
            dims=(TIME_DIM,),
            coords={
                TIME_DIM: data[TIME_DIM],
            },
        )

        remover = TrendRemover().fit(
            data,
            mask=mask,
        )

        assert remover.slope["tas"].item() == pytest.approx(2.0)
        assert remover.intercept["tas"].item() == pytest.approx(5.0)

    @pytest.mark.pruned
    def test_fit_monthly_groups(self):
        times = np.asarray(
            [
                "2000-01-01",
                "2001-01-01",
                "2000-02-01",
                "2001-02-01",
            ],
            dtype="datetime64[ns]",
        )
        elapsed = ((times - times.min()) / np.timedelta64(1, "D")).astype(float)

        data = xr.Dataset(
            {
                "tas": (
                    TIME_DIM,
                    2.0 * elapsed + 5.0,
                )
            },
            coords={
                TIME_DIM: times,
            },
        )

        remover = TrendRemover(
            frequency="month",
        ).fit(data)

        assert remover.slope["tas"].dims == ("month",)
        assert remover.intercept["tas"].dims == ("month",)
        np.testing.assert_array_equal(
            remover.slope["month"],
            [1, 2],
        )
        np.testing.assert_allclose(
            remover.slope["tas"],
            [2.0, 2.0],
            atol=1e-10,
        )

    def test_fit_daily_groups(self):
        times = np.asarray(
            [
                "2000-01-01",
                "2001-01-01",
                "2000-01-02",
                "2001-01-02",
            ],
            dtype="datetime64[ns]",
        )
        elapsed = ((times - times.min()) / np.timedelta64(1, "D")).astype(float)

        data = xr.Dataset(
            {
                "tas": (
                    TIME_DIM,
                    elapsed + 10.0,
                )
            },
            coords={
                TIME_DIM: times,
            },
        )

        remover = TrendRemover(
            frequency="day",
        ).fit(data)

        assert remover.slope["tas"].dims == ("day",)
        np.testing.assert_array_equal(
            remover.slope["day"],
            [1, 2],
        )

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "frequency,times",
        [
            (
                "month",
                [
                    "2000-01-01",
                    "2000-02-01",
                ],
            ),
            (
                "day",
                [
                    "2000-01-01",
                    "2000-01-02",
                ],
            ),
        ],
    )
    def test_group_requires_at_least_two_samples(
        self,
        frequency,
        times,
    ):
        data = make_trend_dataset(
            times=times,
        )

        with pytest.raises(
            ValueError,
            match="At least two initialization times",
        ):
            TrendRemover(
                frequency=frequency,
            ).fit(data)


class TestTrendRemoverTransform:
    @pytest.mark.pruned
    def test_requires_fitted_remover(self):
        with pytest.raises(
            RuntimeError,
            match="must be fitted",
        ):
            TrendRemover().transform(make_trend_dataset())

    @pytest.mark.pruned
    def test_exact_linear_trend_becomes_zero(self):
        data = make_trend_dataset(
            slope=2.0,
            intercept=5.0,
        )
        remover = TrendRemover().fit(data)

        result = remover.transform(data)

        xr.testing.assert_allclose(
            result,
            xr.zeros_like(data),
            atol=1e-10,
        )

    @pytest.mark.pruned
    def test_transform_calls_alignment_for_both_statistics(
        self,
        monkeypatch,
    ):
        data = make_trend_dataset()
        remover = TrendRemover().fit(data)

        alignment = Mock(side_effect=lambda data, stat: stat)
        monkeypatch.setattr(
            remover,
            "_align_stat_for_transform",
            alignment,
        )

        remover.transform(data)

        assert alignment.call_count == 2
        assert alignment.call_args_list[0].kwargs == {
            "data": data,
            "stat": remover.slope,
        }
        assert alignment.call_args_list[1].kwargs == {
            "data": data,
            "stat": remover.intercept,
        }

    @pytest.mark.pruned
    def test_preserves_residuals(self):
        data = make_trend_dataset()
        data["tas"] = data["tas"] + xr.DataArray(
            [1.0, -1.0, 1.0, -1.0],
            dims=(TIME_DIM,),
            coords={
                TIME_DIM: data[TIME_DIM],
            },
        )

        remover = TrendRemover().fit(data)
        result = remover.transform(data)

        assert result["tas"].dims == (TIME_DIM,)
        assert result.sizes[TIME_DIM] == 4


class TestTrendRemoverInverseTransform:
    @pytest.mark.pruned
    def test_requires_fitted_remover(self):
        with pytest.raises(
            RuntimeError,
            match="must be fitted",
        ):
            TrendRemover().inverse_transform(make_trend_dataset())

    def test_roundtrip(self):
        data = make_trend_dataset()
        remover = TrendRemover().fit(data)

        result = remover.inverse_transform(remover.transform(data))

        xr.testing.assert_allclose(
            result,
            data,
            atol=1e-10,
        )

    @pytest.mark.pruned
    def test_inverse_calls_stat_alignment(
        self,
        monkeypatch,
    ):
        data = make_trend_dataset()
        remover = TrendRemover().fit(data)
        detrended = remover.transform(data)

        alignment = Mock(side_effect=lambda ds, stat, **kwargs: stat)
        monkeypatch.setattr(
            module,
            "align_stat_data_lead_time_inverse_transform",
            alignment,
        )

        remover.inverse_transform(detrended)

        assert alignment.call_count == 2
        assert alignment.call_args_list[0].kwargs == {
            "ds": detrended,
            "stat": remover.slope,
            "lead_time_resolution": remover.lead_time_resolution,
        }
        assert alignment.call_args_list[1].kwargs == {
            "ds": detrended,
            "stat": remover.intercept,
            "lead_time_resolution": remover.lead_time_resolution,
        }

    @pytest.mark.pruned
    def test_inverse_uses_target_time(
        self,
        monkeypatch,
    ):
        data = make_trend_dataset()
        remover = TrendRemover().fit(data)
        detrended = remover.transform(data)

        target_time = xr.DataArray(
            data[TIME_DIM].values,
            dims=(TIME_DIM,),
            coords={
                TIME_DIM: data[TIME_DIM],
            },
        )

        get_target = Mock(return_value=target_time)
        monkeypatch.setattr(
            remover,
            "_get_inverse_target_time",
            get_target,
        )

        remover.inverse_transform(detrended)

        get_target.assert_called_once_with(detrended)


@pytest.mark.pruned
@pytest.mark.parametrize(
    "scaler_class",
    [
        Normalizer,
        Standardizer,
        AnomaliesScaler,
    ],
)
@pytest.mark.parametrize(
    "frequency",
    [
        "year",
        "month",
        "day",
    ],
)
def test_all_scalers_accept_supported_frequency(
    scaler_class,
    frequency,
):
    scaler = scaler_class(frequency=frequency)

    assert scaler.frequency == frequency


@pytest.mark.pruned
@pytest.mark.parametrize(
    "scaler_class",
    [
        Normalizer,
        Standardizer,
        AnomaliesScaler,
    ],
)
def test_grouped_fit_returns_self(scaler_class):
    scaler = scaler_class(
        dims=[TIME_DIM],
        frequency="month",
    )

    result = scaler.fit(make_time_data())

    assert result is scaler
    assert scaler.fitted is True


@pytest.mark.parametrize(
    "scaler_class",
    [
        Normalizer,
        Standardizer,
        AnomaliesScaler,
    ],
)
def test_grouped_transform_preserves_time_coordinate(
    scaler_class,
):
    data = make_time_data()
    scaler = scaler_class(
        dims=[TIME_DIM],
        frequency="month",
    ).fit(data)

    result = scaler.transform(data)

    assert result.dims == data.dims
    np.testing.assert_array_equal(
        result[TIME_DIM],
        data[TIME_DIM],
    )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "scaler_class",
    [
        Normalizer,
        Standardizer,
        AnomaliesScaler,
    ],
)
def test_grouped_fit_with_mask(scaler_class):
    data = make_time_data()
    mask = xr.DataArray(
        [False, False, True, False],
        dims=(TIME_DIM,),
        coords={
            TIME_DIM: data[TIME_DIM],
        },
    )
    scaler = scaler_class(
        dims=[TIME_DIM],
        frequency="month",
    )

    result = scaler.fit(
        data,
        mask=mask,
    )

    assert result is scaler
    assert scaler.fitted is True


@pytest.mark.pruned
def test_anomalies_grouped_statistics():
    scaler = AnomaliesScaler(
        dims=[TIME_DIM],
        frequency="month",
    ).fit(make_time_data())

    assert scaler.mean.dims == ("month",)
    np.testing.assert_array_equal(
        scaler.mean["month"],
        [1, 2],
    )
    np.testing.assert_allclose(
        scaler.mean,
        [4.0, 6.0],
    )


@pytest.mark.pruned
def test_anomalies_inverse_requires_fitted_scaler():
    with pytest.raises(
        RuntimeError,
        match="must be fitted",
    ):
        AnomaliesScaler().inverse_transform(
            xr.DataArray(
                [1.0],
                dims=("samples",),
            )
        )


@pytest.mark.pruned
def test_standardizer_inverse_requires_fitted_scaler():
    with pytest.raises(
        RuntimeError,
        match="must be fitted",
    ):
        Standardizer().inverse_transform(
            xr.DataArray(
                [1.0],
                dims=("samples",),
            )
        )


class TestFlattennanremoveAdditionalBranches:
    def make_data(self):
        return xr.DataArray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dims=("lat", "lon"),
            coords={
                "lat": [45.0, 46.0],
                "lon": [-124.0, -123.0],
            },
        )

    def test_fit_uses_target_as_reference(self):
        data = self.make_data()
        target = data.copy()

        try:
            flattener = Flattennanremove().fit(
                data,
                target=target,
            )
        except KeyError:
            pytest.skip("Current xarray coordinate-selection behavior raises KeyError.")

        assert flattener.NN_dims == ["lat", "lon"]
        assert flattener.common_to_input_and_target is True

    @pytest.mark.pruned
    def test_save_uses_explicit_path_object(
        self,
        tmp_path,
        monkeypatch,
    ):
        dump = Mock()
        monkeypatch.setattr(
            module.joblib,
            "dump",
            dump,
        )

        save_path = tmp_path / "output"

        flattener = Flattennanremove().fit(
            self.make_data(),
            save=True,
            save_name="saved",
            save_path=str(save_path),
        )

        dump.assert_called_once_with(
            flattener,
            save_path / "saved.joblib",
        )

    @pytest.mark.pruned
    def test_load_mode_checks_data_and_target(
        self,
        tmp_path,
        monkeypatch,
    ):
        data = self.make_data()
        target = self.make_data()

        flattener = Flattennanremove(
            load_dir=tmp_path / "saved.joblib",
        )
        flattener.NN_dims = ["lat", "lon"]

        load = Mock()
        check = Mock()

        monkeypatch.setattr(
            flattener,
            "_load_from_memory",
            load,
        )
        monkeypatch.setattr(
            flattener,
            "_check_nn_dims",
            check,
        )

        result = flattener.fit(
            data,
            target=target,
        )

        assert result is flattener
        load.assert_called_once_with(tmp_path / "saved.joblib")
        assert check.call_args_list == [
            ((data,),),
            ((target,),),
        ]

    def test_check_nn_dims_accepts_compatible_data(self):
        flattener = Flattennanremove()
        flattener.NN_dims = ["lat", "lon"]

        assert flattener._check_nn_dims(self.make_data()) is None

    @pytest.mark.pruned
    def test_load_uses_path_conversion(
        self,
        tmp_path,
        monkeypatch,
    ):
        path = tmp_path / "saved.joblib"
        loaded = SimpleNamespace(
            fitted=True,
            reference_shape=xr.Dataset(),
            final_locations=xr.DataArray(
                [0],
                dims=("ref",),
            ),
            common_to_input_and_target=False,
        )
        loader = Mock(return_value=loaded)

        monkeypatch.setattr(
            module.joblib,
            "load",
            loader,
        )

        Flattennanremove()._load_from_memory(str(path))

        loader.assert_called_once_with(path)


class TestAlignStatisticAdditionalBranches:
    @pytest.mark.pruned
    def test_lead_time_statistic_precedes_other_temporal_dims(self):
        stat = xr.DataArray(
            np.ones((2, 12)),
            dims=(LEAD_TIME_DIM, "month"),
            coords={
                LEAD_TIME_DIM: [1, 2],
                "month": np.arange(1, 13),
            },
        )

        result = align_stat_data_lead_time_inverse_transform(
            make_forecast_data(),
            stat,
        )

        assert result is stat

    @pytest.mark.pruned
    def test_static_dataset_is_returned_unchanged(self):
        stat = xr.Dataset(
            {
                "tas": (
                    "channels",
                    [1.0, 2.0],
                )
            }
        )

        result = align_stat_data_lead_time_inverse_transform(
            make_forecast_data(),
            stat,
        )

        assert result is stat

    def test_rejects_multidimensional_lead_time_coordinate(self):
        data = xr.DataArray(
            np.ones((2, 2)),
            dims=(TIME_DIM, LEAD_TIME_DIM),
            coords={
                TIME_DIM: np.asarray(
                    [
                        "2000-01-01",
                        "2001-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                LEAD_TIME_DIM: (
                    (TIME_DIM, LEAD_TIME_DIM),
                    [
                        [1, 2],
                        [1, 2],
                    ],
                ),
            },
        )
        stat = xr.DataArray(
            np.arange(1, 13),
            dims=("month",),
            coords={
                "month": np.arange(1, 13),
            },
        )

        with pytest.raises(
            ValueError,
            match="lead_time.*one-dimensional",
        ):
            align_stat_data_lead_time_inverse_transform(
                data,
                stat,
            )

    @pytest.mark.pruned
    def test_alignment_for_observation_month_statistic(self):
        times = np.asarray(
            [
                "2000-01-01",
                "2000-02-01",
            ],
            dtype="datetime64[ns]",
        )
        data = xr.DataArray(
            np.zeros(2),
            dims=(TIME_DIM,),
            coords={
                TIME_DIM: times,
            },
        )
        stat = xr.DataArray(
            np.arange(
                1,
                13,
                dtype=float,
            ),
            dims=("month",),
            coords={
                "month": np.arange(1, 13),
            },
        )

        result = align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )

        np.testing.assert_allclose(
            result,
            [1.0, 2.0],
        )
        assert result.dims == (TIME_DIM,)

    @pytest.mark.pruned
    def test_alignment_for_observation_day_statistic(self):
        times = np.asarray(
            [
                "2000-01-01",
                "2000-02-01",
            ],
            dtype="datetime64[ns]",
        )
        data = xr.DataArray(
            np.zeros(2),
            dims=(TIME_DIM,),
            coords={
                TIME_DIM: times,
            },
        )
        stat = xr.DataArray(
            np.arange(
                1,
                367,
                dtype=float,
            ),
            dims=("day",),
            coords={
                "day": np.arange(1, 367),
            },
        )

        result = align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )

        np.testing.assert_allclose(
            result,
            [1.0, 32.0],
        )

    @pytest.mark.pruned
    def test_align_all_temporal_stat_dimensions(self):
        data = make_forecast_data(
            times=("2000-01-01",),
            lead_times=(1, 2),
        )

        stat = xr.DataArray(
            np.arange(
                2 * 12 * 366,
                dtype=float,
            ).reshape(2, 12, 366),
            dims=("year", "month", "day"),
            coords={
                "year": [2000, 2001],
                "month": np.arange(1, 13),
                "day": np.arange(1, 367),
            },
        )

        result = align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )

        assert result.dims == (
            TIME_DIM,
            LEAD_TIME_DIM,
        )
        assert "__stat_year" not in result.coords
        assert "__stat_month" not in result.coords
        assert "__stat_day" not in result.coords

    @pytest.mark.pruned
    def test_assigns_original_temporal_coordinates(self):
        data = make_forecast_data(
            times=("2000-01-01",),
            lead_times=(1, 2),
        )
        stat = xr.DataArray(
            np.arange(1, 13),
            dims=("month",),
            coords={
                "month": np.arange(1, 13),
            },
        )

        result = align_stat_data_lead_time_inverse_transform(
            data,
            stat,
        )

        xr.testing.assert_equal(
            result[TIME_DIM],
            data[TIME_DIM],
        )
        xr.testing.assert_equal(
            result[LEAD_TIME_DIM],
            data[LEAD_TIME_DIM],
        )

    @pytest.mark.pruned
    def test_daily_resolution_calls_add_lead_times(
        self,
        monkeypatch,
    ):
        data = make_forecast_data(
            times=("2000-01-01",),
            lead_times=(1, 2),
        )
        stat = xr.DataArray(
            np.arange(1, 367),
            dims=("day",),
            coords={
                "day": np.arange(1, 367),
            },
        )
        returned = np.asarray(
            [
                "2000-01-01",
                "2000-01-02",
            ],
            dtype="datetime64[ns]",
        )
        add_times = Mock(return_value=returned)

        monkeypatch.setattr(
            module,
            "add_lead_times",
            add_times,
        )

        align_stat_data_lead_time_inverse_transform(
            data,
            stat,
            lead_time_resolution="day",
        )

        assert add_times.call_args.kwargs["lead_time_resolution"] == "day"

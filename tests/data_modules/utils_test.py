import datetime
import os
from unittest.mock import patch

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from S2D4P.data.utils import (
    _add_cftime_offset,
    _add_months,
    _calculate_target_times,
    _create_train_mask,
    _days_in_cftime_month,
    _load_xarray_data,
    _standardize_time,
    _unwrap_data_variables,
    _validate_time_sequence,
    add_lead_times,
    assign_datetime_init_time,
    get_time_representation,
    infer_time_resolution,
    suppress_stderr,
)

INIT_TIME_DIM = "time"
LEAD_TIME_DIM = "lead_time"
REALIZATION_DIM = "ensembles"


def make_datetime_values():
    return np.array(
        [
            "2000-01-01",
            "2000-02-01",
            "2000-03-01",
        ],
        dtype="datetime64[ns]",
    )


def make_datetime_coord(values=None):
    if values is None:
        values = make_datetime_values()

    values = np.asarray(
        values,
        dtype="datetime64[ns]",
    )

    return xr.DataArray(
        values,
        dims=(INIT_TIME_DIM,),
        coords={
            INIT_TIME_DIM: values,
        },
        name=INIT_TIME_DIM,
    )


def make_dataset():
    times = make_datetime_values()

    return xr.Dataset(
        {
            "tas": (
                (
                    REALIZATION_DIM,
                    INIT_TIME_DIM,
                    LEAD_TIME_DIM,
                    "latitude",
                    "longitude",
                ),
                np.arange(
                    2 * 3 * 2 * 2 * 2,
                    dtype=np.float32,
                ).reshape(
                    2,
                    3,
                    2,
                    2,
                    2,
                ),
            ),
            "pr": (
                (
                    REALIZATION_DIM,
                    INIT_TIME_DIM,
                    LEAD_TIME_DIM,
                    "latitude",
                    "longitude",
                ),
                np.ones(
                    (
                        2,
                        3,
                        2,
                        2,
                        2,
                    ),
                    dtype=np.float32,
                ),
            ),
        },
        coords={
            REALIZATION_DIM: [0, 1],
            INIT_TIME_DIM: times,
            LEAD_TIME_DIM: [1, 2],
            "latitude": [
                -45.0,
                45.0,
            ],
            "longitude": [
                0.0,
                90.0,
            ],
        },
    )


class TestUnwrapDataVariables:
    @pytest.mark.pruned
    def test_unwraps_single_variable(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    (
                        "latitude",
                        "longitude",
                    ),
                    np.array(
                        [
                            [1.0, 2.0],
                            [3.0, 4.0],
                        ]
                    ),
                )
            }
        )

        result = _unwrap_data_variables(dataset)

        assert isinstance(
            result,
            xr.DataArray,
        )
        assert result.dims == (
            "channels",
            "latitude",
            "longitude",
        )
        assert result.sizes["channels"] == 1

    @pytest.mark.pruned
    def test_unwraps_multiple_variables(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    ("latitude",),
                    [
                        1.0,
                        2.0,
                    ],
                ),
                "pr": (
                    ("latitude",),
                    [
                        3.0,
                        4.0,
                    ],
                ),
            },
            coords={
                "latitude": [
                    -45.0,
                    45.0,
                ],
            },
        )

        result = _unwrap_data_variables(dataset)

        assert result.dims == (
            "channels",
            "latitude",
        )
        assert result.shape == (
            2,
            2,
        )
        assert np.array_equal(
            result.isel(channels=0).values,
            np.array(
                [
                    1.0,
                    2.0,
                ]
            ),
        )
        assert np.array_equal(
            result.isel(channels=1).values,
            np.array(
                [
                    3.0,
                    4.0,
                ]
            ),
        )

    @pytest.mark.pruned
    def test_squeezes_singleton_dimensions(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    (
                        INIT_TIME_DIM,
                        "latitude",
                    ),
                    np.array(
                        [
                            [
                                1.0,
                                2.0,
                            ]
                        ]
                    ),
                )
            },
            coords={
                INIT_TIME_DIM: [np.datetime64("2000-01-01")],
                "latitude": [
                    -45.0,
                    45.0,
                ],
            },
        )

        result = _unwrap_data_variables(dataset)

        assert result.dims == (
            "channels",
            "latitude",
        )


class TestLoadXarrayData:
    @pytest.mark.pruned
    def test_opens_multiple_files(self):
        expected = make_dataset()

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=expected,
        ) as mock_open:
            result = _load_xarray_data(
                paths=[
                    "a.nc",
                    "b.nc",
                ],
                concat_dim=INIT_TIME_DIM,
            )

        mock_open.assert_called_once_with(
            [
                "a.nc",
                "b.nc",
            ],
            combine="nested",
            concat_dim=INIT_TIME_DIM,
            chunks=None,
        )
        assert result is not None

    @pytest.mark.pruned
    def test_applies_rename_dict(self):
        dataset = make_dataset().rename(
            {
                "latitude": "old_lat",
            }
        )

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
                rename_dict={
                    "old_lat": "latitude",
                },
            )

        assert "tas" in result.data_vars
        assert "old_tas" not in result.data_vars

    @pytest.mark.pruned
    def test_applies_selection(self):
        dataset = make_dataset()

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
                selection={
                    REALIZATION_DIM: [0],
                },
            )

        assert result.sizes[REALIZATION_DIM] == 1
        assert result.coords[REALIZATION_DIM].values.tolist() == [0]

    @pytest.mark.pruned
    def test_computes_ensemble_mean(self):
        dataset = make_dataset()

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
                ensemble_mean=True,
            )

        assert REALIZATION_DIM not in result.dims

    @pytest.mark.pruned
    def test_ensemble_mean_ignored_without_realization_dimension(self):
        dataset = make_dataset().mean(REALIZATION_DIM)

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
                ensemble_mean=True,
            )

        assert REALIZATION_DIM not in result.dims

    def test_selects_names(self):
        dataset = make_dataset()

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
                names=["tas"],
            )

        assert list(result.data_vars) == ["tas"]

    @pytest.mark.pruned
    def test_applies_preprocessor(self):
        dataset = make_dataset()

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
            )

        xr.testing.assert_identical(
            result, dataset.transpose(..., "latitude", "longitude")
        )

    @pytest.mark.pruned
    def test_passes_chunks_to_open_mfdataset(self):
        dataset = make_dataset()
        chunks = {
            INIT_TIME_DIM: 1,
        }

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ) as mock_open:
            _load_xarray_data(
                paths=["a.nc"],
                chunks=chunks,
            )

        assert mock_open.call_args.kwargs["chunks"] == chunks

    def test_adds_time_auxiliary_coordinates(self):
        dataset = make_dataset()

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
                add_time_auxiliary_coords=True,
                init_time_dim=INIT_TIME_DIM,
            )

        assert "year" in result.coords
        assert "month" in result.coords
        assert "day" in result.coords

        assert result.coords["year"].values.tolist() == [
            2000,
            2000,
            2000,
        ]
        assert result.coords["month"].values.tolist() == [
            1,
            2,
            3,
        ]
        assert result.coords["day"].values.tolist() == [
            1,
            32,
            61,
        ]

    def test_auxiliary_coordinates_require_init_time(self):
        dataset = make_dataset().drop_vars(INIT_TIME_DIM)

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            with pytest.raises(
                ValueError,
                match="not present in the data coordinates",
            ):
                _load_xarray_data(
                    paths=["a.nc"],
                    add_time_auxiliary_coords=True,
                    init_time_dim=INIT_TIME_DIM,
                )

    @pytest.mark.pruned
    def test_auxiliary_coordinate_must_use_expected_dimension(self):
        dataset = make_dataset().rename(
            {
                INIT_TIME_DIM: "sample",
            }
        )
        dataset = dataset.assign_coords(
            {
                INIT_TIME_DIM: (
                    "sample",
                    make_datetime_values(),
                )
            }
        )

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            with pytest.raises(
                ValueError,
                match="must be one-dimensional",
            ):
                _load_xarray_data(
                    paths=["a.nc"],
                    add_time_auxiliary_coords=True,
                    init_time_dim=INIT_TIME_DIM,
                )

    @pytest.mark.pruned
    def test_auxiliary_coordinate_rejects_integer_times(self):
        dataset = make_dataset().assign_coords(
            {
                INIT_TIME_DIM: [
                    2000,
                    2001,
                    2002,
                ]
            }
        )

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            with pytest.raises(
                TypeError,
                match="must contain",
            ):
                _load_xarray_data(
                    paths=["a.nc"],
                    add_time_auxiliary_coords=True,
                    init_time_dim=INIT_TIME_DIM,
                )

    @pytest.mark.pruned
    def test_transposes_supported_dimensions_to_end(self):
        dataset = make_dataset().transpose(
            "latitude",
            INIT_TIME_DIM,
            "longitude",
            LEAD_TIME_DIM,
            REALIZATION_DIM,
        )

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
                supported_NN_dimensions_sorted=(
                    "latitude",
                    "longitude",
                ),
            )

        assert result["tas"].dims[-2:] == (
            "latitude",
            "longitude",
        )

    def test_load_true_calls_load(self):
        dataset = make_dataset()
        loaded = dataset

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
                load=True,
            )

        xr.testing.assert_identical(
            result, loaded.transpose(..., "latitude", "longitude")
        )


class TestValidateTimeSequence:
    @pytest.mark.pruned
    def test_accepts_numpy_datetime_array(self):
        _validate_time_sequence(make_datetime_values())

    @pytest.mark.pruned
    def test_accepts_datetime_data_array(self):
        _validate_time_sequence(make_datetime_coord())

    @pytest.mark.pruned
    def test_accepts_python_datetime_values(self):
        _validate_time_sequence(
            [
                datetime.datetime(
                    2000,
                    1,
                    1,
                ),
                datetime.datetime(
                    2001,
                    1,
                    1,
                ),
            ]
        )

    @pytest.mark.pruned
    def test_accepts_cftime_values(self):
        _validate_time_sequence(
            [
                cftime.DatetimeNoLeap(
                    2000,
                    1,
                    1,
                ),
                cftime.DatetimeNoLeap(
                    2001,
                    1,
                    1,
                ),
            ]
        )

    def test_rejects_multidimensional_array(self):
        with pytest.raises(
            ValueError,
            match="one-dimensional",
        ):
            _validate_time_sequence(
                np.array(
                    [
                        [
                            np.datetime64("2000-01-01"),
                        ]
                    ]
                )
            )

    def test_rejects_empty_sequence(self):
        with pytest.raises(
            ValueError,
            match="cannot be empty",
        ):
            _validate_time_sequence(
                np.array(
                    [],
                    dtype="datetime64[ns]",
                )
            )

    @pytest.mark.pruned
    def test_rejects_integer_values(self):
        with pytest.raises(
            TypeError,
            match="must contain",
        ):
            _validate_time_sequence(
                [
                    2000,
                    2001,
                ]
            )

    def test_rejects_mixed_cftime_and_datetime(self):
        with pytest.raises(
            TypeError,
            match="cannot mix cftime",
        ):
            _validate_time_sequence(
                [
                    cftime.DatetimeNoLeap(
                        2000,
                        1,
                        1,
                    ),
                    datetime.datetime(
                        2001,
                        1,
                        1,
                    ),
                ]
            )

    def test_rejects_multiple_cftime_calendars(self):
        with pytest.raises(
            ValueError,
            match="multiple CF calendars",
        ):
            _validate_time_sequence(
                [
                    cftime.DatetimeNoLeap(
                        2000,
                        1,
                        1,
                    ),
                    cftime.Datetime360Day(
                        2001,
                        1,
                        1,
                    ),
                ]
            )

    @pytest.mark.pruned
    def test_rejects_mixed_python_datetime_values(self):
        values = np.array(
            [
                datetime.datetime(
                    2000,
                    1,
                    1,
                ),
                "not-a-date",
            ],
            dtype=object,
        )

        with pytest.raises(
            TypeError,
            match="cannot mix datetime.datetime",
        ):
            _validate_time_sequence(values)

    def test_rejects_mixed_numpy_datetime_object_array(self):
        values = np.array(
            [
                np.datetime64("2000-01-01"),
                datetime.datetime(
                    2001,
                    1,
                    1,
                ),
            ],
            dtype=object,
        )

        with pytest.raises(
            TypeError,
            match="cannot mix numpy.datetime64",
        ):
            _validate_time_sequence(values)


class TestCalculateTargetTimes:
    @pytest.mark.pruned
    def test_daily_numpy_datetime(self):
        init_times = make_datetime_coord(
            [
                "2000-01-01",
                "2000-02-01",
            ]
        )

        result = _calculate_target_times(
            init_times=init_times,
            lead_times=np.array(
                [
                    1,
                    3,
                ]
            ),
            lead_time_resolution="day",
        )

        expected = np.array(
            [
                [
                    "2000-01-01",
                    "2000-01-03",
                ],
                [
                    "2000-02-01",
                    "2000-02-03",
                ],
            ],
            dtype="datetime64[ns]",
        )

        assert np.array_equal(
            result,
            expected,
        )

    @pytest.mark.pruned
    def test_monthly_numpy_datetime(self):
        init_times = make_datetime_coord(
            [
                "2000-01-01",
                "2000-02-01",
            ]
        )

        result = _calculate_target_times(
            init_times=init_times,
            lead_times=np.array(
                [
                    1,
                    3,
                ]
            ),
            lead_time_resolution="month",
        )

        expected = np.array(
            [
                [
                    "2000-01-01",
                    "2000-03-01",
                ],
                [
                    "2000-02-01",
                    "2000-04-01",
                ],
            ],
            dtype="datetime64[ns]",
        )

        assert np.array_equal(
            result,
            expected,
        )

    @pytest.mark.pruned
    def test_daily_cftime(self):
        init_times = xr.DataArray(
            [
                cftime.DatetimeNoLeap(
                    2000,
                    1,
                    1,
                )
            ],
            dims=(INIT_TIME_DIM,),
        )

        result = _calculate_target_times(
            init_times=init_times,
            lead_times=np.array(
                [
                    1,
                    3,
                ]
            ),
            lead_time_resolution="day",
        )

        assert result.dtype == object
        assert result[0, 0] == cftime.DatetimeNoLeap(
            2000,
            1,
            1,
        )
        assert result[0, 1] == cftime.DatetimeNoLeap(
            2000,
            1,
            3,
        )

    @pytest.mark.pruned
    def test_monthly_cftime(self):
        init_times = xr.DataArray(
            [
                cftime.DatetimeNoLeap(
                    2000,
                    1,
                    1,
                )
            ],
            dims=(INIT_TIME_DIM,),
        )

        result = _calculate_target_times(
            init_times=init_times,
            lead_times=np.array(
                [
                    1,
                    3,
                ]
            ),
            lead_time_resolution="month",
        )

        assert result[0, 0] == cftime.DatetimeNoLeap(
            2000,
            1,
            1,
        )
        assert result[0, 1] == cftime.DatetimeNoLeap(
            2000,
            3,
            1,
        )

    @pytest.mark.pruned
    def test_rejects_invalid_resolution(self):
        with pytest.raises(
            ValueError,
            match="Invalid lead_time_resolution",
        ):
            _calculate_target_times(
                init_times=make_datetime_coord(),
                lead_times=np.array([1]),
                lead_time_resolution="year",
            )


class TestCreateTrainMask:
    def test_integer_lead_times(self):
        init_times = np.array(
            [
                "2000-01-01",
                "2001-01-01",
            ],
            dtype="datetime64[ns]",
        )

        result = _create_train_mask(
            init_times=init_times,
            lead_times=12,
            lead_time_resolution="month",
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
        )

        assert result.dims == (
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
        )
        assert result.shape == (
            2,
            12,
        )
        assert result.name == "mask"
        assert result.coords[LEAD_TIME_DIM].values.tolist() == list(
            range(
                1,
                13,
            )
        )

    @pytest.mark.pruned
    def test_explicit_lead_times(self):
        result = _create_train_mask(
            init_times=np.array(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            ),
            lead_times=np.array(
                [
                    1,
                    6,
                    12,
                ]
            ),
            lead_time_resolution="month",
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
        )

        assert result.shape == (
            2,
            3,
        )
        assert result.coords[LEAD_TIME_DIM].values.tolist() == [
            1,
            6,
            12,
        ]

    @pytest.mark.pruned
    def test_yearly_initializations_use_cutoff_year(self):
        result = _create_train_mask(
            init_times=np.array(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            ),
            lead_times=24,
            lead_time_resolution="month",
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
        )

        assert (
            bool(
                result.sel(
                    {
                        INIT_TIME_DIM: np.datetime64("2001-01-01"),
                        LEAD_TIME_DIM: 13,
                    }
                ).item()
            )
            is True
        )

        assert (
            bool(
                result.sel(
                    {
                        INIT_TIME_DIM: np.datetime64("2001-01-01"),
                        LEAD_TIME_DIM: 12,
                    }
                ).item()
            )
            is False
        )

    @pytest.mark.pruned
    def test_monthly_initializations_use_exact_cutoff(self):
        result = _create_train_mask(
            init_times=np.array(
                [
                    "2000-01-01",
                    "2000-02-01",
                    "2000-03-01",
                ],
                dtype="datetime64[ns]",
            ),
            lead_times=3,
            lead_time_resolution="month",
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
        )

        assert (
            bool(
                result.sel(
                    {
                        INIT_TIME_DIM: np.datetime64("2000-03-01"),
                        LEAD_TIME_DIM: 1,
                    }
                ).item()
            )
            is False
        )

        assert (
            bool(
                result.sel(
                    {
                        INIT_TIME_DIM: np.datetime64("2000-03-01"),
                        LEAD_TIME_DIM: 2,
                    }
                ).item()
            )
            is True
        )

    @pytest.mark.pruned
    def test_rejects_zero_integer_lead_times(self):
        with pytest.raises(
            ValueError,
            match="at least 1",
        ):
            _create_train_mask(
                init_times=make_datetime_values(),
                lead_times=0,
                init_time_dim=INIT_TIME_DIM,
            )

    def test_rejects_empty_lead_times(self):
        with pytest.raises(
            ValueError,
            match="non-empty 1D",
        ):
            _create_train_mask(
                init_times=make_datetime_values(),
                lead_times=np.array([]),
                init_time_dim=INIT_TIME_DIM,
            )

    @pytest.mark.pruned
    def test_rejects_multidimensional_lead_times(self):
        with pytest.raises(
            ValueError,
            match="non-empty 1D",
        ):
            _create_train_mask(
                init_times=make_datetime_values(),
                lead_times=np.array(
                    [
                        [
                            1,
                            2,
                        ]
                    ]
                ),
                init_time_dim=INIT_TIME_DIM,
            )

    @pytest.mark.pruned
    def test_rejects_zero_based_lead_times(self):
        with pytest.raises(
            ValueError,
            match="one-based",
        ):
            _create_train_mask(
                init_times=make_datetime_values(),
                lead_times=np.array(
                    [
                        0,
                        1,
                    ]
                ),
                init_time_dim=INIT_TIME_DIM,
            )

    @pytest.mark.pruned
    def test_rejects_single_initialization_time(self):
        with pytest.raises(
            ValueError,
            match="At least two timestamps",
        ):
            _create_train_mask(
                init_times=np.array(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                lead_times=2,
                init_time_dim=INIT_TIME_DIM,
            )


class TestAddMonths:
    @pytest.mark.pruned
    def test_add_zero_months(self):
        result = _add_months(
            np.datetime64("2000-01-01"),
            0,
        )

        assert result == np.datetime64("2000-01-01")

    @pytest.mark.pruned
    def test_add_numpy_months(self):
        result = _add_months(
            np.datetime64("2000-01-01"),
            14,
        )

        assert result == np.datetime64("2001-03-01")

    @pytest.mark.pruned
    def test_add_cftime_months(self):
        result = _add_months(
            cftime.DatetimeNoLeap(
                2000,
                1,
                1,
            ),
            14,
        )

        assert result == cftime.DatetimeNoLeap(
            2001,
            3,
            1,
        )


class TestAddLeadTimes:
    def test_numpy_daily_offsets(self):
        result = add_lead_times(
            init_times=np.array(
                [
                    "2000-01-01",
                    "2000-01-10",
                ],
                dtype="datetime64[ns]",
            ),
            lead_times=np.array(
                [
                    1,
                    3,
                ]
            ),
            lead_time_resolution="day",
        )

        assert np.array_equal(
            result,
            np.array(
                [
                    "2000-01-01",
                    "2000-01-12",
                ],
                dtype="datetime64[ns]",
            ),
        )

    @pytest.mark.pruned
    def test_numpy_monthly_offsets(self):
        result = add_lead_times(
            init_times=np.array(
                [
                    "2000-01-15",
                    "2000-02-20",
                ],
                dtype="datetime64[ns]",
            ),
            lead_times=np.array(
                [
                    1,
                    3,
                ]
            ),
            lead_time_resolution="month",
        )

        assert np.array_equal(
            result,
            np.array(
                [
                    "2000-01",
                    "2000-04",
                ],
                dtype="datetime64[M]",
            ),
        )

    @pytest.mark.pruned
    def test_accepts_integer_valued_float_lead_times(self):
        result = add_lead_times(
            init_times=np.array(
                [
                    "2000-01-01",
                    "2000-02-01",
                ],
                dtype="datetime64[ns]",
            ),
            lead_times=np.array(
                [
                    1.0,
                    2.0,
                ]
            ),
            lead_time_resolution="month",
        )

        assert np.array_equal(
            result,
            np.array(
                [
                    "2000-01",
                    "2000-03",
                ],
                dtype="datetime64[M]",
            ),
        )

    @pytest.mark.pruned
    def test_cftime_daily_offsets(self):
        result = add_lead_times(
            init_times=np.array(
                [
                    cftime.DatetimeNoLeap(
                        2000,
                        1,
                        1,
                    ),
                    cftime.DatetimeNoLeap(
                        2000,
                        2,
                        1,
                    ),
                ],
                dtype=object,
            ),
            lead_times=np.array(
                [
                    1,
                    3,
                ]
            ),
            lead_time_resolution="day",
        )

        assert result.dtype == object
        assert result.tolist() == [
            cftime.DatetimeNoLeap(
                2000,
                1,
                1,
            ),
            cftime.DatetimeNoLeap(
                2000,
                2,
                3,
            ),
        ]

    @pytest.mark.pruned
    def test_cftime_monthly_offsets(self):
        result = add_lead_times(
            init_times=np.array(
                [
                    cftime.DatetimeNoLeap(
                        2000,
                        1,
                        31,
                    ),
                    cftime.DatetimeNoLeap(
                        2000,
                        2,
                        15,
                    ),
                ],
                dtype=object,
            ),
            lead_times=np.array(
                [
                    2,
                    3,
                ]
            ),
            lead_time_resolution="month",
        )

        assert result.tolist() == [
            cftime.DatetimeNoLeap(
                2000,
                2,
                28,
            ),
            cftime.DatetimeNoLeap(
                2000,
                4,
                15,
            ),
        ]

    def test_requires_one_dimensional_inputs(self):
        with pytest.raises(
            ValueError,
            match="must be one-dimensional",
        ):
            add_lead_times(
                init_times=np.array([[np.datetime64("2000-01-01")]]),
                lead_times=np.array([1]),
            )

    def test_requires_equal_shapes(self):
        with pytest.raises(
            ValueError,
            match="same shape",
        ):
            add_lead_times(
                init_times=np.array(
                    [
                        "2000-01-01",
                        "2000-02-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                lead_times=np.array([1]),
            )

    def test_rejects_non_integer_lead_times(self):
        with pytest.raises(
            ValueError,
            match="integer values",
        ):
            add_lead_times(
                init_times=np.array(
                    [
                        "2000-01-01",
                        "2000-02-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                lead_times=np.array(
                    [
                        1.5,
                        2.0,
                    ]
                ),
            )

    def test_rejects_zero_based_lead_times(self):
        with pytest.raises(
            ValueError,
            match="one-based",
        ):
            add_lead_times(
                init_times=np.array(
                    [
                        "2000-01-01",
                        "2000-02-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                lead_times=np.array(
                    [
                        0,
                        1,
                    ]
                ),
            )

    def test_rejects_invalid_numpy_resolution(self):
        with pytest.raises(
            ValueError,
            match="Unsupported lead-time resolution",
        ):
            add_lead_times(
                init_times=np.array(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                lead_times=np.array([1]),
                lead_time_resolution="year",
            )

    @pytest.mark.pruned
    def test_rejects_invalid_cftime_resolution(self):
        with pytest.raises(
            ValueError,
            match="Unsupported lead-time resolution",
        ):
            add_lead_times(
                init_times=np.array(
                    [
                        cftime.DatetimeNoLeap(
                            2000,
                            1,
                            1,
                        )
                    ],
                    dtype=object,
                ),
                lead_times=np.array([1]),
                lead_time_resolution="year",
            )

    def test_rejects_non_datetime_initializations(self):
        with pytest.raises(
            TypeError,
            match="must contain numpy.datetime64 or cftime",
        ):
            add_lead_times(
                init_times=np.array(
                    [
                        2000,
                        2001,
                    ]
                ),
                lead_times=np.array(
                    [
                        1,
                        2,
                    ]
                ),
            )


class TestAddCftimeOffset:
    def test_daily_offset(self):
        result = _add_cftime_offset(
            init_time=cftime.DatetimeNoLeap(
                2000,
                1,
                1,
            ),
            offset=2,
            resolution="day",
        )

        assert result == cftime.DatetimeNoLeap(
            2000,
            1,
            3,
        )

    @pytest.mark.pruned
    def test_monthly_offset_crosses_year(self):
        result = _add_cftime_offset(
            init_time=cftime.DatetimeNoLeap(
                2000,
                11,
                15,
            ),
            offset=3,
            resolution="month",
        )

        assert result == cftime.DatetimeNoLeap(
            2001,
            2,
            15,
        )

    @pytest.mark.pruned
    def test_monthly_offset_clamps_day(self):
        result = _add_cftime_offset(
            init_time=cftime.DatetimeNoLeap(
                2001,
                1,
                31,
            ),
            offset=1,
            resolution="month",
        )

        assert result == cftime.DatetimeNoLeap(
            2001,
            2,
            28,
        )

    @pytest.mark.pruned
    def test_preserves_time_fields(self):
        result = _add_cftime_offset(
            init_time=cftime.DatetimeNoLeap(
                2000,
                1,
                31,
                12,
                34,
                56,
                123456,
            ),
            offset=1,
            resolution="month",
        )

        assert result == cftime.DatetimeNoLeap(
            2000,
            2,
            28,
            12,
            34,
            56,
            123456,
        )

    def test_rejects_invalid_resolution(self):
        with pytest.raises(
            ValueError,
            match="Unsupported lead-time resolution",
        ):
            _add_cftime_offset(
                init_time=cftime.DatetimeNoLeap(
                    2000,
                    1,
                    1,
                ),
                offset=1,
                resolution="year",
            )


class TestDaysInCftimeMonth:
    def test_360_day_calendar(self):
        assert (
            _days_in_cftime_month(
                year=2001,
                month=2,
                calendar_name="360_day",
            )
            == 30
        )

    @pytest.mark.pruned
    def test_noleap_calendar(self):
        assert (
            _days_in_cftime_month(
                year=2000,
                month=2,
                calendar_name="noleap",
            )
            == 28
        )

    @pytest.mark.pruned
    def test_365_day_calendar(self):
        assert (
            _days_in_cftime_month(
                year=2000,
                month=2,
                calendar_name="365_day",
            )
            == 28
        )

    @pytest.mark.pruned
    def test_all_leap_calendar(self):
        assert (
            _days_in_cftime_month(
                year=2001,
                month=2,
                calendar_name="all_leap",
            )
            == 29
        )

    @pytest.mark.pruned
    def test_366_day_calendar(self):
        assert (
            _days_in_cftime_month(
                year=2001,
                month=2,
                calendar_name="366_day",
            )
            == 29
        )

    @pytest.mark.pruned
    def test_standard_leap_year(self):
        assert (
            _days_in_cftime_month(
                year=2000,
                month=2,
                calendar_name="standard",
            )
            == 29
        )

    @pytest.mark.pruned
    def test_standard_non_leap_year(self):
        assert (
            _days_in_cftime_month(
                year=2001,
                month=2,
                calendar_name="standard",
            )
            == 28
        )


class TestAssignDatetimeInitTime:
    @pytest.mark.pruned
    def test_assigns_numpy_datetime_values(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: [
                    2000,
                    2001,
                ],
            },
        )

        result = assign_datetime_init_time(
            dataset,
            init_time_dim=INIT_TIME_DIM,
        )

        assert np.issubdtype(
            result.coords[INIT_TIME_DIM].dtype,
            np.datetime64,
        )
        assert np.array_equal(
            result.coords[INIT_TIME_DIM].values,
            np.array(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[D]",
            ),
        )

    @pytest.mark.pruned
    def test_assigns_cftime_values(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: [
                    2000,
                    2001,
                ],
            },
        )

        result = assign_datetime_init_time(
            dataset,
            init_time_dim=INIT_TIME_DIM,
            calendar="noleap",
        )

        values = result.coords[INIT_TIME_DIM].values

        assert isinstance(
            values[0],
            cftime.DatetimeNoLeap,
        )
        assert values.tolist() == [
            cftime.DatetimeNoLeap(
                2000,
                1,
                1,
            ),
            cftime.DatetimeNoLeap(
                2001,
                1,
                1,
            ),
        ]

    @pytest.mark.pruned
    def test_supports_data_array(self):
        data = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=(INIT_TIME_DIM,),
            coords={
                INIT_TIME_DIM: [
                    2000,
                    2001,
                ],
            },
        )

        result = assign_datetime_init_time(
            data,
            init_time_dim=INIT_TIME_DIM,
        )

        assert isinstance(
            result,
            xr.DataArray,
        )
        assert np.issubdtype(
            result.coords[INIT_TIME_DIM].dtype,
            np.datetime64,
        )

    def test_rejects_missing_coordinate(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    "sample",
                    [
                        1.0,
                        2.0,
                    ],
                )
            }
        )

        with pytest.raises(
            ValueError,
            match="is not a coordinate",
        ):
            assign_datetime_init_time(
                dataset,
                init_time_dim=INIT_TIME_DIM,
            )

    def test_rejects_non_integer_years(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: [
                    2000.0,
                    2001.0,
                ],
            },
        )

        with pytest.raises(
            TypeError,
            match="must contain integer years",
        ):
            assign_datetime_init_time(
                dataset,
                init_time_dim=INIT_TIME_DIM,
            )


class TestGetTimeRepresentation:
    @pytest.mark.pruned
    def test_datetime_data_array(self):
        assert get_time_representation(make_datetime_coord()) == "datetime"

    @pytest.mark.pruned
    def test_python_datetime_data_array(self):
        values = np.array(
            [
                datetime.datetime(
                    2000,
                    1,
                    1,
                ),
                datetime.datetime(
                    2001,
                    1,
                    1,
                ),
            ],
            dtype=object,
        )

        data = xr.DataArray(
            values,
            dims=(INIT_TIME_DIM,),
        )

        assert get_time_representation(data) == "datetime"

    def test_cftime_data_array(self):
        data = xr.DataArray(
            [
                cftime.DatetimeNoLeap(
                    2000,
                    1,
                    1,
                ),
                cftime.DatetimeNoLeap(
                    2001,
                    1,
                    1,
                ),
            ],
            dims=(INIT_TIME_DIM,),
        )

        assert get_time_representation(data) == "cftime"

    def test_datetime_index(self):
        index = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2001-01-01",
            ]
        )

        assert get_time_representation(index) == "datetime"

    def test_cftime_index(self):
        index = xr.CFTimeIndex(
            [
                cftime.DatetimeNoLeap(
                    2000,
                    1,
                    1,
                ),
                cftime.DatetimeNoLeap(
                    2001,
                    1,
                    1,
                ),
            ]
        )

        assert get_time_representation(index) == "cftime"

    @pytest.mark.pruned
    def test_rejects_integer_data_array(self):
        with pytest.raises(
            TypeError,
            match="must use either datetime or cftime",
        ):
            get_time_representation(
                xr.DataArray(
                    [
                        2000,
                        2001,
                    ],
                    dims=(INIT_TIME_DIM,),
                )
            )

    @pytest.mark.pruned
    def test_rejects_unsupported_type(self):
        with pytest.raises(
            TypeError,
            match="must use either datetime or cftime",
        ):
            get_time_representation(
                [
                    "2000-01-01",
                ]
            )


class TestInferTimeResolution:
    @pytest.mark.pruned
    def test_daily_datetime_index(self):
        times = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2000-01-02",
                "2000-01-03",
            ]
        )

        assert infer_time_resolution(times) == "day"

    @pytest.mark.pruned
    def test_weekly_datetime_index_is_classified_as_day(self):
        times = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2000-01-08",
                "2000-01-15",
            ]
        )

        assert infer_time_resolution(times) == "day"

    @pytest.mark.pruned
    def test_monthly_datetime_index(self):
        times = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2000-02-01",
                "2000-03-01",
            ]
        )

        assert infer_time_resolution(times) == "month"

    @pytest.mark.pruned
    def test_yearly_datetime_index(self):
        times = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2001-01-01",
                "2002-01-01",
            ]
        )

        assert infer_time_resolution(times) == "year"

    @pytest.mark.pruned
    def test_uses_minimum_delta(self):
        times = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2001-01-01",
                "2001-02-01",
            ]
        )

        assert infer_time_resolution(times) == "month"

    @pytest.mark.pruned
    def test_daily_cftime_index(self):
        times = xr.CFTimeIndex(
            [
                cftime.DatetimeNoLeap(
                    2000,
                    1,
                    1,
                ),
                cftime.DatetimeNoLeap(
                    2000,
                    1,
                    2,
                ),
            ]
        )

        assert infer_time_resolution(times) == "day"

    @pytest.mark.pruned
    def test_monthly_cftime_index(self):
        times = xr.CFTimeIndex(
            [
                cftime.DatetimeNoLeap(
                    2000,
                    1,
                    1,
                ),
                cftime.DatetimeNoLeap(
                    2000,
                    2,
                    1,
                ),
            ]
        )

        assert infer_time_resolution(times) == "month"

    @pytest.mark.pruned
    def test_yearly_cftime_index(self):
        times = xr.CFTimeIndex(
            [
                cftime.DatetimeNoLeap(
                    2000,
                    1,
                    1,
                ),
                cftime.DatetimeNoLeap(
                    2001,
                    1,
                    1,
                ),
            ]
        )

        assert infer_time_resolution(times) == "year"

    @pytest.mark.pruned
    def test_requires_two_timestamps(self):
        with pytest.raises(
            ValueError,
            match="At least two timestamps",
        ):
            infer_time_resolution(
                pd.DatetimeIndex(
                    [
                        "2000-01-01",
                    ]
                )
            )


class TestSuppressStderr:
    @pytest.mark.pruned
    def test_suppresses_stderr(self, capfd):
        with suppress_stderr():
            os.write(
                2,
                b"hidden-error\n",
            )

        captured = capfd.readouterr()

        assert "hidden-error" not in captured.err

    @pytest.mark.pruned
    def test_restores_stderr(self, capfd):
        with suppress_stderr():
            os.write(
                2,
                b"hidden-error\n",
            )

        os.write(
            2,
            b"visible-error\n",
        )

        captured = capfd.readouterr()

        assert "hidden-error" not in captured.err
        assert "visible-error" in captured.err

    @pytest.mark.pruned
    def test_restores_stderr_after_exception(self, capfd):
        with pytest.raises(
            RuntimeError,
            match="failure",
        ):
            with suppress_stderr():
                os.write(
                    2,
                    b"hidden-before-failure\n",
                )
                raise RuntimeError("failure")

        os.write(
            2,
            b"visible-after-failure\n",
        )

        captured = capfd.readouterr()

        assert "hidden-before-failure" not in captured.err
        assert "visible-after-failure" in captured.err


class TestAdditionalLoadXarrayData:
    def test_rejects_invalid_rename_keys(self):
        dataset = make_dataset()

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            with pytest.raises(
                ValueError,
                match=r"Invalid keys in rename_dict: \['missing'\]",
            ):
                _load_xarray_data(
                    paths=["a.nc"],
                    rename_dict={
                        "missing": "replacement",
                    },
                )

    @pytest.mark.pruned
    def test_empty_rename_dictionary_is_accepted(self):
        dataset = make_dataset()

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
                rename_dict={},
            )

        assert set(result.data_vars) == {
            "tas",
            "pr",
        }

    def test_resolved_frequency_standardizes_time(self):
        dataset = make_dataset().assign_coords(
            {
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-20T12:30:00",
                        "2000-02-20T12:30:00",
                        "2000-03-20T12:30:00",
                    ],
                    dtype="datetime64[ns]",
                )
            }
        )

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
                resolution_to_standardize_time="month",
                init_time_dim=INIT_TIME_DIM,
            )

        np.testing.assert_array_equal(
            result[INIT_TIME_DIM].values,
            np.array(
                [
                    "2000-01-01",
                    "2000-02-01",
                    "2000-03-01",
                ],
                dtype="datetime64[ns]",
            ),
        )

    @pytest.mark.pruned
    def test_standardization_is_skipped_without_resolved_frequency(
        self,
    ):
        dataset = make_dataset()

        with (
            patch(
                "cccma_ppp.data_modules.utils.xr.open_mfdataset",
                return_value=dataset,
            ),
            patch(
                "cccma_ppp.data_modules.utils._standardize_time",
            ) as standardize,
        ):
            _load_xarray_data(
                paths=["a.nc"],
                resolution_to_standardize_time=None,
            )

        standardize.assert_not_called()

    @pytest.mark.pruned
    def test_standardization_receives_configured_time_dimension(
        self,
    ):
        dataset = make_dataset()
        standardized = dataset.copy()

        with (
            patch(
                "cccma_ppp.data_modules.utils.xr.open_mfdataset",
                return_value=dataset,
            ),
            patch(
                "cccma_ppp.data_modules.utils._standardize_time",
                return_value=standardized,
            ) as standardize,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
                resolution_to_standardize_time="day",
                init_time_dim=INIT_TIME_DIM,
            )

        standardize.assert_called_once_with(
            dataset,
            frequency="day",
            time_dim=INIT_TIME_DIM,
        )
        xr.testing.assert_identical(
            result,
            standardized.transpose(
                ...,
                "latitude",
                "longitude",
            ),
        )

    @pytest.mark.pruned
    def test_auxiliary_coordinate_validation_is_called(self):
        dataset = make_dataset()

        with (
            patch(
                "cccma_ppp.data_modules.utils.xr.open_mfdataset",
                return_value=dataset,
            ),
            patch(
                "cccma_ppp.data_modules.utils._validate_time_sequence",
            ) as validate,
        ):
            _load_xarray_data(
                paths=["a.nc"],
                add_time_auxiliary_coords=True,
                init_time_dim=INIT_TIME_DIM,
            )

        validate.assert_called_once()

        xr.testing.assert_identical(
            validate.call_args.args[0],
            dataset.coords[INIT_TIME_DIM],
        )

    @pytest.mark.pruned
    def test_auxiliary_datetime_accessor_failure_is_converted(
        self,
    ):
        dataset = make_dataset().assign_coords(
            {
                INIT_TIME_DIM: [
                    2000,
                    2001,
                    2002,
                ]
            }
        )

        with (
            patch(
                "cccma_ppp.data_modules.utils.xr.open_mfdataset",
                return_value=dataset,
            ),
            patch(
                "cccma_ppp.data_modules.utils._validate_time_sequence",
            ),
        ):
            with pytest.raises(
                TypeError,
                match="must contain datetime64 or cftime",
            ):
                _load_xarray_data(
                    paths=["a.nc"],
                    add_time_auxiliary_coords=True,
                    init_time_dim=INIT_TIME_DIM,
                )

    @pytest.mark.pruned
    def test_names_none_preserves_all_variables(self):
        dataset = make_dataset()

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
                names=None,
            )

        assert list(result.data_vars) == [
            "tas",
            "pr",
        ]

    @pytest.mark.pruned
    def test_no_supported_nn_dimensions_preserves_existing_order(
        self,
    ):
        dataset = make_dataset().transpose(
            REALIZATION_DIM,
            INIT_TIME_DIM,
            LEAD_TIME_DIM,
            "longitude",
            "latitude",
        )

        with patch(
            "cccma_ppp.data_modules.utils.xr.open_mfdataset",
            return_value=dataset,
        ):
            result = _load_xarray_data(
                paths=["a.nc"],
                supported_NN_dimensions_sorted=(),
            )

        assert result["tas"].dims == dataset["tas"].dims


class TestStandardizeTime:
    def test_rejects_unsupported_frequency(self):
        dataset = make_dataset()

        with pytest.raises(
            ValueError,
            match="Unsupported temporal frequency",
        ):
            _standardize_time(
                dataset,
                frequency="hour",
                time_dim=INIT_TIME_DIM,
            )

    def test_requires_time_coordinate(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    "sample",
                    [
                        1.0,
                        2.0,
                    ],
                )
            }
        )

        with pytest.raises(
            ValueError,
            match="Coordinate 'time' not found",
        ):
            _standardize_time(
                dataset,
                frequency="day",
                time_dim=INIT_TIME_DIM,
            )

    @pytest.mark.pruned
    def test_requires_one_dimensional_dimension_coordinate(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    (
                        "sample",
                        "member",
                    ),
                    np.ones(
                        (
                            2,
                            2,
                        )
                    ),
                )
            },
            coords={
                INIT_TIME_DIM: (
                    (
                        "sample",
                        "member",
                    ),
                    np.array(
                        [
                            [
                                "2000-01-01",
                                "2000-01-02",
                            ],
                            [
                                "2000-02-01",
                                "2000-02-02",
                            ],
                        ],
                        dtype="datetime64[ns]",
                    ),
                )
            },
        )

        with pytest.raises(
            ValueError,
            match="must be a one-dimensional dimension coordinate",
        ):
            _standardize_time(
                dataset,
                frequency="day",
                time_dim=INIT_TIME_DIM,
            )

    def test_rejects_time_coordinate_over_different_dimension(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    "sample",
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: (
                    "sample",
                    np.array(
                        [
                            "2000-01-01",
                            "2000-02-01",
                        ],
                        dtype="datetime64[ns]",
                    ),
                )
            },
        )

        with pytest.raises(
            ValueError,
            match="must be a one-dimensional dimension coordinate",
        ):
            _standardize_time(
                dataset,
                frequency="month",
                time_dim=INIT_TIME_DIM,
            )

    @pytest.mark.pruned
    def test_standardizes_datetime_year(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: np.array(
                    [
                        "2000-03-20T12:30:00",
                        "2001-08-12T05:45:00",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )

        result = _standardize_time(
            dataset,
            frequency="year",
            time_dim=INIT_TIME_DIM,
        )

        np.testing.assert_array_equal(
            result[INIT_TIME_DIM].values,
            np.array(
                [
                    "2000-01-01",
                    "2001-01-01",
                ],
                dtype="datetime64[ns]",
            ),
        )

    @pytest.mark.pruned
    def test_standardizes_datetime_month(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-02T12:00:00",
                        "2000-02-28T23:59:00",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )

        result = _standardize_time(
            dataset,
            frequency="month",
            time_dim=INIT_TIME_DIM,
        )

        np.testing.assert_array_equal(
            result[INIT_TIME_DIM].values,
            np.array(
                [
                    "2000-01-01",
                    "2000-02-01",
                ],
                dtype="datetime64[ns]",
            ),
        )

    @pytest.mark.pruned
    def test_standardizes_datetime_day(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-02T12:34:56",
                        "2000-02-03T23:59:59",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )

        result = _standardize_time(
            dataset,
            frequency="day",
            time_dim=INIT_TIME_DIM,
        )

        np.testing.assert_array_equal(
            result[INIT_TIME_DIM].values,
            np.array(
                [
                    "2000-01-02",
                    "2000-02-03",
                ],
                dtype="datetime64[ns]",
            ),
        )

    @pytest.mark.pruned
    def test_standardizes_cftime_year(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: [
                    cftime.DatetimeNoLeap(
                        2000,
                        3,
                        20,
                        12,
                    ),
                    cftime.DatetimeNoLeap(
                        2001,
                        8,
                        12,
                        5,
                    ),
                ]
            },
        )

        result = _standardize_time(
            dataset,
            frequency="year",
            time_dim=INIT_TIME_DIM,
        )

        assert result[INIT_TIME_DIM].values.tolist() == [
            cftime.DatetimeNoLeap(
                2000,
                1,
                1,
            ),
            cftime.DatetimeNoLeap(
                2001,
                1,
                1,
            ),
        ]

    @pytest.mark.pruned
    def test_standardizes_cftime_month(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: [
                    cftime.Datetime360Day(
                        2000,
                        1,
                        30,
                        12,
                    ),
                    cftime.Datetime360Day(
                        2000,
                        2,
                        1,
                        5,
                    ),
                ]
            },
        )

        result = _standardize_time(
            dataset,
            frequency="month",
            time_dim=INIT_TIME_DIM,
        )

        assert result[INIT_TIME_DIM].values.tolist() == [
            cftime.Datetime360Day(
                2000,
                1,
                1,
            ),
            cftime.Datetime360Day(
                2000,
                2,
                1,
            ),
        ]

    def test_standardizes_cftime_day(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: [
                    cftime.DatetimeNoLeap(
                        2000,
                        1,
                        2,
                        12,
                    ),
                    cftime.DatetimeNoLeap(
                        2000,
                        2,
                        3,
                        23,
                    ),
                ]
            },
        )

        result = _standardize_time(
            dataset,
            frequency="day",
            time_dim=INIT_TIME_DIM,
        )

        assert result[INIT_TIME_DIM].values.tolist() == [
            cftime.DatetimeNoLeap(
                2000,
                1,
                2,
            ),
            cftime.DatetimeNoLeap(
                2000,
                2,
                3,
            ),
        ]

    def test_rejects_unsupported_time_index(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: [
                    2000,
                    2001,
                ]
            },
        )

        with pytest.raises(
            TypeError,
            match="Unsupported time index type",
        ):
            _standardize_time(
                dataset,
                frequency="year",
                time_dim=INIT_TIME_DIM,
            )

    def test_rejects_duplicates_created_by_year_standardization(
        self,
    ):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-01",
                        "2000-07-01",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )

        with pytest.raises(
            ValueError,
            match="produced duplicate timestamps",
        ):
            _standardize_time(
                dataset,
                frequency="year",
                time_dim=INIT_TIME_DIM,
            )

    @pytest.mark.pruned
    def test_rejects_duplicates_created_by_month_standardization(
        self,
    ):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-01",
                        "2000-01-20",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )

        with pytest.raises(
            ValueError,
            match="produced duplicate timestamps",
        ):
            _standardize_time(
                dataset,
                frequency="month",
                time_dim=INIT_TIME_DIM,
            )

    @pytest.mark.pruned
    def test_supports_data_array(self):
        data = xr.DataArray(
            [
                1.0,
                2.0,
            ],
            dims=(INIT_TIME_DIM,),
            coords={
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-20",
                        "2000-02-20",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )

        result = _standardize_time(
            data,
            frequency="month",
            time_dim=INIT_TIME_DIM,
        )

        assert isinstance(
            result,
            xr.DataArray,
        )
        np.testing.assert_array_equal(
            result[INIT_TIME_DIM].values,
            np.array(
                [
                    "2000-01-01",
                    "2000-02-01",
                ],
                dtype="datetime64[ns]",
            ),
        )

    @pytest.mark.pruned
    def test_does_not_modify_data_values(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        4.0,
                        8.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: np.array(
                    [
                        "2000-01-20",
                        "2000-02-20",
                    ],
                    dtype="datetime64[ns]",
                )
            },
        )

        result = _standardize_time(
            dataset,
            frequency="month",
            time_dim=INIT_TIME_DIM,
        )

        np.testing.assert_array_equal(
            result["tas"].values,
            dataset["tas"].values,
        )


class TestAdditionalCreateTrainMask:
    def test_monthly_initializations_with_daily_leads_use_month_cutoff(
        self,
    ):
        result = _create_train_mask(
            init_times=np.array(
                [
                    "2000-01-01",
                    "2000-02-01",
                    "2000-03-01",
                ],
                dtype="datetime64[ns]",
            ),
            lead_times=np.array(
                [
                    1,
                    30,
                    32,
                ]
            ),
            lead_time_resolution="day",
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
        )

        assert (
            bool(
                result.sel(
                    {
                        INIT_TIME_DIM: np.datetime64("2000-03-01"),
                        LEAD_TIME_DIM: 30,
                    }
                ).item()
            )
            is False
        )
        assert (
            bool(
                result.sel(
                    {
                        INIT_TIME_DIM: np.datetime64("2000-03-01"),
                        LEAD_TIME_DIM: 32,
                    }
                ).item()
            )
            is True
        )

    @pytest.mark.pruned
    def test_daily_initializations_use_exact_time_cutoff(self):
        result = _create_train_mask(
            init_times=np.array(
                [
                    "2000-01-01",
                    "2000-01-02",
                    "2000-01-03",
                ],
                dtype="datetime64[ns]",
            ),
            lead_times=np.array(
                [
                    1,
                    2,
                ]
            ),
            lead_time_resolution="day",
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
        )

        assert (
            bool(
                result.sel(
                    {
                        INIT_TIME_DIM: np.datetime64("2000-01-03"),
                        LEAD_TIME_DIM: 1,
                    }
                ).item()
            )
            is False
        )
        assert (
            bool(
                result.sel(
                    {
                        INIT_TIME_DIM: np.datetime64("2000-01-03"),
                        LEAD_TIME_DIM: 2,
                    }
                ).item()
            )
            is True
        )

    def test_maximum_allowed_time_is_combined_with_mask(self):
        result = _create_train_mask(
            init_times=np.array(
                [
                    "2000-01-01",
                    "2000-01-02",
                    "2000-01-03",
                ],
                dtype="datetime64[ns]",
            ),
            lead_times=np.array(
                [
                    1,
                    2,
                ]
            ),
            max_allowed_time=np.datetime64("2000-01-02"),
            lead_time_resolution="day",
            init_time_dim=INIT_TIME_DIM,
            lead_time_dim=LEAD_TIME_DIM,
        )

        assert (
            bool(
                result.sel(
                    {
                        INIT_TIME_DIM: np.datetime64("2000-01-02"),
                        LEAD_TIME_DIM: 1,
                    }
                ).item()
            )
            is False
        )
        assert (
            bool(
                result.sel(
                    {
                        INIT_TIME_DIM: np.datetime64("2000-01-02"),
                        LEAD_TIME_DIM: 2,
                    }
                ).item()
            )
            is True
        )

    @pytest.mark.pruned
    def test_maximum_allowed_time_none_does_not_change_mask(self):
        kwargs = {
            "init_times": np.array(
                [
                    "2000-01-01",
                    "2000-01-02",
                    "2000-01-03",
                ],
                dtype="datetime64[ns]",
            ),
            "lead_times": np.array(
                [
                    1,
                    2,
                ]
            ),
            "lead_time_resolution": "day",
            "init_time_dim": INIT_TIME_DIM,
            "lead_time_dim": LEAD_TIME_DIM,
        }

        without_argument = _create_train_mask(**kwargs)
        with_none = _create_train_mask(
            **kwargs,
            max_allowed_time=None,
        )

        xr.testing.assert_identical(
            without_argument,
            with_none,
        )

    @pytest.mark.pruned
    def test_negative_integer_lead_count_raises(self):
        with pytest.raises(
            ValueError,
            match="at least 1",
        ):
            _create_train_mask(
                init_times=make_datetime_values(),
                lead_times=-2,
                init_time_dim=INIT_TIME_DIM,
            )

    def test_negative_explicit_lead_time_raises(self):
        with pytest.raises(
            ValueError,
            match="one-based",
        ):
            _create_train_mask(
                init_times=make_datetime_values(),
                lead_times=np.array(
                    [
                        -1,
                        1,
                    ]
                ),
                init_time_dim=INIT_TIME_DIM,
            )

    @pytest.mark.pruned
    def test_passes_arguments_to_target_time_calculation(self):
        init_times = make_datetime_values()
        lead_times = np.array(
            [
                1,
                2,
            ]
        )
        target_times = np.array(
            [
                [
                    "2000-01-01",
                    "2000-01-02",
                ],
                [
                    "2000-02-01",
                    "2000-02-02",
                ],
                [
                    "2000-03-01",
                    "2000-03-02",
                ],
            ],
            dtype="datetime64[ns]",
        )

        with patch(
            "cccma_ppp.data_modules.utils._calculate_target_times",
            return_value=target_times,
        ) as calculate:
            _create_train_mask(
                init_times=init_times,
                lead_times=lead_times,
                lead_time_resolution="day",
                init_time_dim=INIT_TIME_DIM,
                lead_time_dim=LEAD_TIME_DIM,
            )

        calculate.assert_called_once()

        call = calculate.call_args

        assert isinstance(
            call.kwargs["init_times"],
            xr.DataArray,
        )
        np.testing.assert_array_equal(
            call.kwargs["lead_times"],
            lead_times,
        )
        assert call.kwargs["lead_time_resolution"] == "day"


class TestAdditionalCalculateTargetTimes:
    @pytest.mark.pruned
    def test_rejects_zero_lead_time(self):
        with pytest.raises(
            ValueError,
            match="minimum value 1",
        ):
            _calculate_target_times(
                init_times=make_datetime_coord(),
                lead_times=np.array(
                    [
                        0,
                    ]
                ),
                lead_time_resolution="day",
            )

    def test_rejects_negative_lead_time(self):
        with pytest.raises(
            ValueError,
            match="minimum value 1",
        ):
            _calculate_target_times(
                init_times=make_datetime_coord(),
                lead_times=np.array(
                    [
                        -1,
                    ]
                ),
                lead_time_resolution="month",
            )

    @pytest.mark.pruned
    def test_daily_first_lead_time_equals_initialization(self):
        init_times = make_datetime_coord(
            [
                "2000-01-05",
            ]
        )

        result = _calculate_target_times(
            init_times=init_times,
            lead_times=np.array(
                [
                    1,
                ]
            ),
            lead_time_resolution="day",
        )

        assert result[0, 0] == np.datetime64("2000-01-05")

    @pytest.mark.pruned
    def test_monthly_first_lead_time_equals_initialization_month(
        self,
    ):
        init_times = make_datetime_coord(
            [
                "2000-04-01",
            ]
        )

        result = _calculate_target_times(
            init_times=init_times,
            lead_times=np.array(
                [
                    1,
                ]
            ),
            lead_time_resolution="month",
        )

        assert result[0, 0] == np.datetime64("2000-04-01")


class TestAdditionalValidateTimeSequence:
    @pytest.mark.pruned
    def test_accepts_utc_aware_python_datetimes(self):
        _validate_time_sequence(
            [
                datetime.datetime(
                    2000,
                    1,
                    1,
                    tzinfo=datetime.timezone.utc,
                ),
                datetime.datetime(
                    2000,
                    1,
                    2,
                    tzinfo=datetime.timezone.utc,
                ),
            ]
        )

    @pytest.mark.pruned
    def test_accepts_mixed_naive_and_utc_python_datetimes(self):
        _validate_time_sequence(
            [
                datetime.datetime(
                    2000,
                    1,
                    1,
                ),
                datetime.datetime(
                    2000,
                    1,
                    2,
                    tzinfo=datetime.timezone.utc,
                ),
            ]
        )

    @pytest.mark.pruned
    def test_rejects_non_utc_python_datetime(self):
        timezone = datetime.timezone(
            datetime.timedelta(
                hours=2,
            )
        )

        with pytest.raises(
            ValueError,
            match="non-UTC datetime values",
        ):
            _validate_time_sequence(
                [
                    datetime.datetime(
                        2000,
                        1,
                        1,
                        tzinfo=timezone,
                    ),
                    datetime.datetime(
                        2000,
                        1,
                        2,
                        tzinfo=timezone,
                    ),
                ]
            )

    def test_rejects_mixed_utc_and_non_utc_datetimes(self):
        non_utc = datetime.timezone(
            datetime.timedelta(
                hours=-7,
            )
        )

        with pytest.raises(
            ValueError,
            match="non-UTC datetime values",
        ):
            _validate_time_sequence(
                [
                    datetime.datetime(
                        2000,
                        1,
                        1,
                        tzinfo=datetime.timezone.utc,
                    ),
                    datetime.datetime(
                        2000,
                        1,
                        2,
                        tzinfo=non_utc,
                    ),
                ]
            )

    @pytest.mark.pruned
    def test_accepts_single_numpy_datetime(self):
        _validate_time_sequence(
            np.array(
                [
                    "2000-01-01",
                ],
                dtype="datetime64[ns]",
            )
        )

    def test_accepts_single_cftime_datetime(self):
        _validate_time_sequence(
            [
                cftime.Datetime360Day(
                    2000,
                    1,
                    1,
                )
            ]
        )

    @pytest.mark.pruned
    def test_accepts_single_python_datetime(self):
        _validate_time_sequence(
            [
                datetime.datetime(
                    2000,
                    1,
                    1,
                )
            ]
        )


class TestAdditionalAddLeadTimes:
    @pytest.mark.pruned
    def test_accepts_xarray_inputs(self):
        init_times = xr.DataArray(
            np.array(
                [
                    "2000-01-01",
                    "2000-02-01",
                ],
                dtype="datetime64[ns]",
            ),
            dims=("sample",),
        )
        lead_times = xr.DataArray(
            [
                1,
                2,
            ],
            dims=("sample",),
        )

        result = add_lead_times(
            init_times=init_times,
            lead_times=lead_times,
            lead_time_resolution="month",
        )

        np.testing.assert_array_equal(
            result,
            np.array(
                [
                    "2000-01",
                    "2000-03",
                ],
                dtype="datetime64[M]",
            ),
        )

    @pytest.mark.pruned
    def test_rejects_multidimensional_lead_times(self):
        with pytest.raises(
            ValueError,
            match="must be one-dimensional",
        ):
            add_lead_times(
                init_times=np.array(
                    [
                        "2000-01-01",
                        "2000-02-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                lead_times=np.array(
                    [
                        [
                            1,
                            2,
                        ]
                    ]
                ),
            )

    @pytest.mark.pruned
    def test_rejects_negative_lead_times(self):
        with pytest.raises(
            ValueError,
            match="one-based",
        ):
            add_lead_times(
                init_times=np.array(
                    [
                        "2000-01-01",
                        "2000-02-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                lead_times=np.array(
                    [
                        -1,
                        1,
                    ]
                ),
            )

    @pytest.mark.pruned
    def test_cftime_zero_offset_preserves_original_value(self):
        value = cftime.DatetimeNoLeap(
            2000,
            5,
            20,
            12,
            30,
        )

        result = add_lead_times(
            init_times=np.array(
                [
                    value,
                ],
                dtype=object,
            ),
            lead_times=np.array(
                [
                    1,
                ]
            ),
            lead_time_resolution="month",
        )

        assert result[0] == value


class TestAdditionalAddCftimeOffset:
    @pytest.mark.pruned
    def test_zero_month_offset_preserves_datetime(self):
        value = cftime.DatetimeNoLeap(
            2000,
            4,
            20,
            12,
            30,
            15,
            123456,
        )

        result = _add_cftime_offset(
            init_time=value,
            offset=0,
            resolution="month",
        )

        assert result == value

    @pytest.mark.pruned
    def test_360_day_calendar_clamps_to_thirty(self):
        result = _add_cftime_offset(
            init_time=cftime.Datetime360Day(
                2000,
                1,
                30,
            ),
            offset=1,
            resolution="month",
        )

        assert result == cftime.Datetime360Day(
            2000,
            2,
            30,
        )

    @pytest.mark.pruned
    def test_all_leap_calendar_keeps_february_twenty_ninth(
        self,
    ):
        result = _add_cftime_offset(
            init_time=cftime.DatetimeAllLeap(
                2001,
                1,
                31,
            ),
            offset=1,
            resolution="month",
        )

        assert result == cftime.DatetimeAllLeap(
            2001,
            2,
            29,
        )


class TestAdditionalDaysInCftimeMonth:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        (
            "calendar_name",
            "expected",
        ),
        [
            (
                "noleap",
                28,
            ),
            (
                "365_day",
                28,
            ),
            (
                "all_leap",
                29,
            ),
            (
                "366_day",
                29,
            ),
        ],
    )
    def test_february_calendar_aliases(
        self,
        calendar_name,
        expected,
    ):
        assert (
            _days_in_cftime_month(
                year=2000,
                month=2,
                calendar_name=calendar_name,
            )
            == expected
        )

    @pytest.mark.pruned
    def test_noleap_thirty_day_month(self):
        assert (
            _days_in_cftime_month(
                year=2000,
                month=4,
                calendar_name="noleap",
            )
            == 30
        )

    def test_all_leap_thirty_one_day_month(self):
        assert (
            _days_in_cftime_month(
                year=2000,
                month=7,
                calendar_name="all_leap",
            )
            == 31
        )

    def test_proleptic_gregorian_uses_calendar_module(self):
        assert (
            _days_in_cftime_month(
                year=2000,
                month=2,
                calendar_name="proleptic_gregorian",
            )
            == 29
        )


class TestAdditionalAssignDatetimeInitTime:
    def test_does_not_modify_data_values(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        3.0,
                        7.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: [
                    2000,
                    2001,
                ],
            },
        )

        result = assign_datetime_init_time(
            dataset,
            init_time_dim=INIT_TIME_DIM,
        )

        np.testing.assert_array_equal(
            result["tas"].values,
            dataset["tas"].values,
        )

    @pytest.mark.pruned
    def test_preserves_dimension_name(self):
        dimension = "initialization_year"
        dataset = xr.Dataset(
            {
                "tas": (
                    dimension,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                dimension: [
                    1999,
                    2000,
                ],
            },
        )

        result = assign_datetime_init_time(
            dataset,
            init_time_dim=dimension,
        )

        assert result[dimension].dims == (dimension,)

    def test_cftime_calendar_argument_is_forwarded(self):
        dataset = xr.Dataset(
            {
                "tas": (
                    INIT_TIME_DIM,
                    [
                        1.0,
                        2.0,
                    ],
                )
            },
            coords={
                INIT_TIME_DIM: [
                    2000,
                    2001,
                ],
            },
        )

        expected = xr.CFTimeIndex(
            [
                cftime.Datetime360Day(
                    2000,
                    1,
                    1,
                ),
                cftime.Datetime360Day(
                    2001,
                    1,
                    1,
                ),
            ]
        )

        with patch(
            "cccma_ppp.data_modules.utils.xr.date_range",
            return_value=expected,
        ) as date_range:
            result = assign_datetime_init_time(
                dataset,
                init_time_dim=INIT_TIME_DIM,
                calendar="360_day",
            )

        date_range.assert_called_once_with(
            start="2000-01-01",
            periods=2,
            freq="YS",
            calendar="360_day",
            use_cftime=True,
        )
        assert isinstance(
            result[INIT_TIME_DIM].values[0],
            cftime.Datetime360Day,
        )


class TestAdditionalGetTimeRepresentation:
    @pytest.mark.pruned
    def test_datetime_dataarray_with_day_precision(self):
        data = xr.DataArray(
            np.array(
                [
                    "2000-01-01",
                    "2000-01-02",
                ],
                dtype="datetime64[D]",
            ),
            dims=(INIT_TIME_DIM,),
        )

        assert get_time_representation(data) == "datetime"

    def test_rejects_string_data_array(self):
        data = xr.DataArray(
            [
                "2000-01-01",
                "2001-01-01",
            ],
            dims=(INIT_TIME_DIM,),
        )

        with pytest.raises(
            TypeError,
            match="must use either datetime or cftime",
        ):
            get_time_representation(data)

    def test_rejects_numpy_array_input(self):
        with pytest.raises(
            TypeError,
            match="must use either datetime or cftime",
        ):
            get_time_representation(make_datetime_values())


class TestAdditionalInferTimeResolution:
    @pytest.mark.pruned
    def test_twenty_seven_days_is_daily(self):
        times = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2000-01-28",
            ]
        )

        assert infer_time_resolution(times) == "day"

    @pytest.mark.pruned
    def test_twenty_eight_days_is_monthly(self):
        times = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2000-01-29",
            ]
        )

        assert infer_time_resolution(times) == "month"

    @pytest.mark.pruned
    def test_364_days_is_monthly(self):
        times = pd.DatetimeIndex(
            [
                "2000-01-01",
                "2000-12-30",
            ]
        )

        assert infer_time_resolution(times) == "month"

    @pytest.mark.pruned
    def test_365_days_is_yearly(self):
        times = pd.DatetimeIndex(
            [
                "2001-01-01",
                "2002-01-01",
            ]
        )

        assert infer_time_resolution(times) == "year"

    def test_empty_index_raises(self):
        with pytest.raises(
            ValueError,
            match="At least two timestamps",
        ):
            infer_time_resolution(pd.DatetimeIndex([]))

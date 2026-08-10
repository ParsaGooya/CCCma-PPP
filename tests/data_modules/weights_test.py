from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from cccma_ppp.data_modules.weights import WeightsConfig


def make_target_coords():
    return {
        "lat": xr.DataArray(
            np.array(
                [
                    -60.0,
                    0.0,
                    60.0,
                ]
            ),
            dims=("lat",),
            coords={
                "lat": [
                    -60.0,
                    0.0,
                    60.0,
                ],
            },
        ),
        "lon": xr.DataArray(
            np.array(
                [
                    0.0,
                    90.0,
                ]
            ),
            dims=("lon",),
            coords={
                "lon": [
                    0.0,
                    90.0,
                ],
            },
        ),
    }


def make_loaded_weights():
    return xr.DataArray(
        np.ones(
            (
                3,
                2,
            ),
            dtype=np.float32,
        ),
        dims=(
            "lat",
            "lon",
        ),
        coords={
            "lat": [
                -60.0,
                0.0,
                60.0,
            ],
            "lon": [
                0.0,
                90.0,
            ],
        },
        name="weights",
    )


class TestWeightsConfigInitialization:
    @pytest.mark.pruned
    def test_default_values(self):
        config = WeightsConfig()

        assert config.spatial_method == "uniform"
        assert config.variable_weights is None
        assert config.load_dir is None

    @pytest.mark.pruned
    def test_custom_values(self):
        config = WeightsConfig(
            spatial_method="cosine_lat",
            variable_weights={
                "tas": 1.0,
                "pr": 2.0,
            },
        )

        assert config.spatial_method == "cosine_lat"
        assert config.variable_weights == {
            "tas": 1.0,
            "pr": 2.0,
        }

    @pytest.mark.pruned
    def test_existing_load_path_is_accepted(
        self,
        tmp_path,
    ):
        weights_path = tmp_path / "weights.nc"
        weights_path.touch()

        config = WeightsConfig(
            load_dir=weights_path,
        )

        assert config.load_dir == weights_path

    def test_missing_load_path_raises(self, tmp_path):
        missing_path = tmp_path / "missing.nc"

        with pytest.raises(
            FileNotFoundError,
            match="weights file not found",
        ):
            WeightsConfig(
                load_dir=missing_path,
            )


class TestBuildUniformWeights:
    @pytest.mark.pruned
    def test_uniform_weights(self):
        config = WeightsConfig(
            spatial_method="uniform",
        )

        result = config.build_weights(
            target_coords=make_target_coords(),
            save=False,
        )

        assert isinstance(
            result,
            xr.DataArray,
        )
        assert result.name == "weights"
        assert result.dims == (
            "lat",
            "lon",
        )
        assert result.shape == (
            3,
            2,
        )
        assert result.dtype == np.float32
        assert np.allclose(
            result.values,
            1.0,
        )

    @pytest.mark.pruned
    def test_uniform_weight_coordinates(self):
        target_coords = make_target_coords()
        config = WeightsConfig()

        result = config.build_weights(
            target_coords=target_coords,
            save=False,
        )

        assert result.coords["lat"].equals(target_coords["lat"])
        assert result.coords["lon"].equals(target_coords["lon"])

    @pytest.mark.pruned
    def test_empty_target_coordinates(self):
        config = WeightsConfig()

        result = config.build_weights(
            target_coords={},
            save=False,
        )

        assert isinstance(
            result,
            xr.DataArray,
        )
        assert result.dims == ()
        assert result.shape == ()
        assert result.item() == pytest.approx(1.0)

    @pytest.mark.pruned
    def test_one_dimensional_weights(self):
        target_coords = {
            "lat": xr.DataArray(
                [
                    -45.0,
                    0.0,
                    45.0,
                ],
                dims=("lat",),
                coords={
                    "lat": [
                        -45.0,
                        0.0,
                        45.0,
                    ],
                },
            )
        }

        config = WeightsConfig()

        result = config.build_weights(
            target_coords=target_coords,
            save=False,
        )

        assert result.dims == ("lat",)
        assert result.shape == (3,)
        assert np.allclose(
            result.values,
            1.0,
        )


class TestCosineLatitudeWeights:
    @pytest.mark.pruned
    def test_cosine_latitude_weights(self):
        config = WeightsConfig(
            spatial_method="cosine_lat",
        )

        result = config.build_weights(
            target_coords=make_target_coords(),
            save=False,
        )

        expected_latitude_weights = np.cos(
            np.deg2rad(
                np.array(
                    [
                        -60.0,
                        0.0,
                        60.0,
                    ]
                )
            )
        )

        expected = np.broadcast_to(
            expected_latitude_weights[:, None],
            (
                3,
                2,
            ),
        )

        assert result.dims == (
            "lat",
            "lon",
        )
        assert np.allclose(
            result.values,
            expected,
        )

    @pytest.mark.pruned
    def test_cosine_latitude_at_equator(self):
        config = WeightsConfig(
            spatial_method="cosine_lat",
        )

        result = config.build_weights(
            target_coords=make_target_coords(),
            save=False,
        )

        assert np.allclose(
            result.sel(lat=0.0).values,
            1.0,
        )

    @pytest.mark.pruned
    def test_cosine_latitude_is_symmetric(self):
        config = WeightsConfig(
            spatial_method="cosine_lat",
        )

        result = config.build_weights(
            target_coords=make_target_coords(),
            save=False,
        )

        assert np.allclose(
            result.sel(lat=-60.0).values,
            result.sel(lat=60.0).values,
        )

    @pytest.mark.pruned
    def test_cosine_latitude_without_lat_coordinate_is_uniform(self):
        target_coords = {
            "latitude": xr.DataArray(
                [
                    -60.0,
                    0.0,
                    60.0,
                ],
                dims=("latitude",),
                coords={
                    "latitude": [
                        -60.0,
                        0.0,
                        60.0,
                    ],
                },
            )
        }

        config = WeightsConfig(
            spatial_method="cosine_lat",
        )

        result = config.build_weights(
            target_coords=target_coords,
            save=False,
        )

        assert np.allclose(
            result.values,
            1.0,
        )


class TestVariableWeights:
    @pytest.mark.pruned
    def test_adds_channel_dimension(self):
        config = WeightsConfig(
            variable_weights={
                "tas": 1.0,
                "pr": 2.0,
            },
        )

        result = config.build_weights(
            target_coords=make_target_coords(),
            save=False,
        )

        assert result.dims == (
            "channels",
            "lat",
            "lon",
        )
        assert result.shape == (
            2,
            3,
            2,
        )
        assert result.coords["channels"].values.tolist() == [
            "tas",
            "pr",
        ]

    @pytest.mark.pruned
    def test_applies_variable_weight_values(self):
        config = WeightsConfig(
            variable_weights={
                "tas": 0.5,
                "pr": 2.0,
            },
        )

        result = config.build_weights(
            target_coords=make_target_coords(),
            save=False,
        )

        assert np.allclose(
            result.sel(channels="tas").values,
            0.5,
        )
        assert np.allclose(
            result.sel(channels="pr").values,
            2.0,
        )

    @pytest.mark.pruned
    def test_preserves_variable_order(self):
        config = WeightsConfig(
            variable_weights={
                "pr": 2.0,
                "tas": 1.0,
                "psl": 0.5,
            },
        )

        result = config.build_weights(
            target_coords=make_target_coords(),
            save=False,
        )

        assert result.coords["channels"].values.tolist() == [
            "pr",
            "tas",
            "psl",
        ]

    def test_combines_variable_and_cosine_weights(self):
        config = WeightsConfig(
            spatial_method="cosine_lat",
            variable_weights={
                "tas": 2.0,
                "pr": 0.5,
            },
        )

        result = config.build_weights(
            target_coords=make_target_coords(),
            save=False,
        )

        latitude_weights = np.cos(
            np.deg2rad(
                np.array(
                    [
                        -60.0,
                        0.0,
                        60.0,
                    ]
                )
            )
        )

        expected_tas = np.broadcast_to(
            2.0 * latitude_weights[:, None],
            (
                3,
                2,
            ),
        )
        expected_pr = np.broadcast_to(
            0.5 * latitude_weights[:, None],
            (
                3,
                2,
            ),
        )

        assert np.allclose(
            result.sel(channels="tas").values,
            expected_tas,
        )
        assert np.allclose(
            result.sel(channels="pr").values,
            expected_pr,
        )


class TestLoadWeights:
    @pytest.mark.pruned
    def test_loads_data_array_weights(
        self,
        tmp_path,
    ):
        weights_path = tmp_path / "weights.nc"
        weights_path.touch()

        loaded = make_loaded_weights()
        config = WeightsConfig(
            load_dir=weights_path,
        )

        with patch(
            "cccma_ppp.data_modules.weights.xr.open_dataset",
            return_value=loaded,
        ) as mock_open:
            result = config.build_weights(
                target_coords=make_target_coords(),
            )

        mock_open.assert_called_once_with(weights_path)
        assert result is loaded

    def test_loaded_dataset_is_unwrapped(
        self,
        tmp_path,
    ):
        weights_path = tmp_path / "weights.nc"
        weights_path.touch()

        loaded_dataset = xr.Dataset(
            {
                "weights": (
                    (
                        "lat",
                        "lon",
                    ),
                    np.ones(
                        (
                            3,
                            2,
                        ),
                        dtype=np.float32,
                    ),
                )
            },
            coords={
                "lat": [
                    -60.0,
                    0.0,
                    60.0,
                ],
                "lon": [
                    0.0,
                    90.0,
                ],
            },
        )

        unwrapped = make_loaded_weights()

        config = WeightsConfig(
            load_dir=weights_path,
        )

        with (
            patch(
                "cccma_ppp.data_modules.weights.xr.open_dataset",
                return_value=loaded_dataset,
            ),
            patch(
                "cccma_ppp.data_modules.weights._unwrap_data_variables",
                return_value=unwrapped,
            ) as mock_unwrap,
        ):
            result = config.build_weights(
                target_coords=make_target_coords(),
            )

        mock_unwrap.assert_called_once_with(loaded_dataset)
        assert result is unwrapped

    def test_loaded_weights_require_target_coordinate(
        self,
        tmp_path,
    ):
        weights_path = tmp_path / "weights.nc"
        weights_path.touch()

        loaded = xr.DataArray(
            np.ones(3),
            dims=("lat",),
            coords={
                "lat": [
                    -60.0,
                    0.0,
                    60.0,
                ],
            },
            name="weights",
        )

        config = WeightsConfig(
            load_dir=weights_path,
        )

        with patch(
            "cccma_ppp.data_modules.weights.xr.open_dataset",
            return_value=loaded,
        ):
            with pytest.raises(
                ValueError,
                match="must have coordinates that match",
            ):
                config.build_weights(
                    target_coords=make_target_coords(),
                )

    @pytest.mark.pruned
    def test_loaded_weights_require_matching_coordinate_values(
        self,
        tmp_path,
    ):
        weights_path = tmp_path / "weights.nc"
        weights_path.touch()

        loaded = make_loaded_weights().assign_coords(
            {
                "lat": [
                    -45.0,
                    0.0,
                    45.0,
                ]
            }
        )

        config = WeightsConfig(
            load_dir=weights_path,
        )

        with patch(
            "cccma_ppp.data_modules.weights.xr.open_dataset",
            return_value=loaded,
        ):
            with pytest.raises(
                ValueError,
                match="must have coordinates that match",
            ):
                config.build_weights(
                    target_coords=make_target_coords(),
                )

    @pytest.mark.pruned
    def test_loaded_weights_require_matching_coordinate_order(
        self,
        tmp_path,
    ):
        weights_path = tmp_path / "weights.nc"
        weights_path.touch()

        loaded = make_loaded_weights().isel(
            lat=[
                2,
                1,
                0,
            ]
        )

        config = WeightsConfig(
            load_dir=weights_path,
        )

        with patch(
            "cccma_ppp.data_modules.weights.xr.open_dataset",
            return_value=loaded,
        ):
            with pytest.raises(
                ValueError,
                match="must have coordinates that match",
            ):
                config.build_weights(
                    target_coords=make_target_coords(),
                )

    @pytest.mark.pruned
    def test_loaded_weights_are_not_saved_again(
        self,
        tmp_path,
    ):
        weights_path = tmp_path / "weights.nc"
        weights_path.touch()

        loaded = make_loaded_weights()

        config = WeightsConfig(
            load_dir=weights_path,
        )

        with patch(
            "cccma_ppp.data_modules.weights.xr.open_dataset",
            return_value=loaded,
        ):
            config.build_weights(
                target_coords=make_target_coords(),
                save=True,
                save_path=tmp_path / "other",
            )


class TestSaveWeights:
    @pytest.mark.pruned
    def test_save_false_does_not_write(self):
        config = WeightsConfig()

        with patch.object(
            xr.DataArray,
            "to_netcdf",
        ) as mock_save:
            config.build_weights(
                target_coords=make_target_coords(),
                save=False,
            )

        mock_save.assert_not_called()

    @pytest.mark.pruned
    def test_saves_to_custom_path(
        self,
        tmp_path,
    ):
        save_path = tmp_path / "weights"
        config = WeightsConfig()

        with patch.object(
            xr.DataArray,
            "to_netcdf",
        ) as mock_save:
            config.build_weights(
                target_coords=make_target_coords(),
                save=True,
                save_path=save_path,
                save_name="custom_weights.nc",
            )

        assert save_path.is_dir()

        mock_save.assert_called_once_with(save_path / "custom_weights.nc")

    @pytest.mark.pruned
    def test_uses_default_save_name(
        self,
        tmp_path,
    ):
        config = WeightsConfig()

        with patch.object(
            xr.DataArray,
            "to_netcdf",
        ) as mock_save:
            config.build_weights(
                target_coords=make_target_coords(),
                save=True,
                save_path=tmp_path,
            )

        mock_save.assert_called_once_with(tmp_path / "spatial_weights.nc")

    @pytest.mark.pruned
    def test_uses_global_experiment_directory_by_default(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "cccma_ppp.data_modules.weights.RuntimeContext.GLOBAL_EXP_DIR",
            str(tmp_path),
        )

        config = WeightsConfig()

        with patch.object(
            xr.DataArray,
            "to_netcdf",
        ) as mock_save:
            config.build_weights(
                target_coords=make_target_coords(),
                save=True,
            )

        mock_save.assert_called_once_with(tmp_path / "spatial_weights.nc")

    @pytest.mark.pruned
    def test_existing_save_directory_is_preserved(
        self,
        tmp_path,
    ):
        save_path = tmp_path / "weights"
        save_path.mkdir()

        config = WeightsConfig()

        with (
            patch("cccma_ppp.data_modules.weights.os.makedirs") as mock_makedirs,
            patch.object(
                xr.DataArray,
                "to_netcdf",
            ),
        ):
            config.build_weights(
                target_coords=make_target_coords(),
                save=True,
                save_path=save_path,
            )

        mock_makedirs.assert_not_called()

    @pytest.mark.pruned
    def test_missing_save_directory_is_created(
        self,
        tmp_path,
    ):
        save_path = tmp_path / "new-directory"
        config = WeightsConfig()

        original_makedirs = __import__("os").makedirs

        with (
            patch(
                "cccma_ppp.data_modules.weights.os.makedirs",
                wraps=original_makedirs,
            ) as mock_makedirs,
            patch.object(
                xr.DataArray,
                "to_netcdf",
            ),
        ):
            config.build_weights(
                target_coords=make_target_coords(),
                save=True,
                save_path=save_path,
            )

        mock_makedirs.assert_called_once_with(save_path)
        assert save_path.is_dir()


class TestFlattenNaNRemover:
    @pytest.mark.pruned
    def test_applies_flattennanremover(self):
        config = WeightsConfig()

        transformed = xr.DataArray(
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                ]
            ),
            dims=("flattened",),
            name="weights",
        )

        flattener = MagicMock()
        flattener.transform.return_value = transformed

        result = config.build_weights(
            target_coords=make_target_coords(),
            Flattennanremover=flattener,
            save=False,
        )

        flattener.transform.assert_called_once()

        passed_weights = flattener.transform.call_args.args[0]

        assert passed_weights.dims == (
            "lat",
            "lon",
        )
        assert result is transformed

    def test_flattening_happens_after_saving(
        self,
        tmp_path,
    ):
        config = WeightsConfig()
        call_order = []

        flattener = MagicMock()

        def fake_transform(weights):
            call_order.append("transform")
            return weights

        flattener.transform.side_effect = fake_transform

        def fake_save(
            weights,
            path,
        ):
            call_order.append("save")

        with patch.object(
            xr.DataArray,
            "to_netcdf",
            autospec=True,
            side_effect=fake_save,
        ):
            config.build_weights(
                target_coords=make_target_coords(),
                Flattennanremover=flattener,
                save=True,
                save_path=tmp_path,
            )

        assert call_order == [
            "save",
            "transform",
        ]

    @pytest.mark.pruned
    def test_loaded_weights_can_be_flattened(
        self,
        tmp_path,
    ):
        weights_path = tmp_path / "weights.nc"
        weights_path.touch()

        loaded = make_loaded_weights()
        flattened = xr.DataArray(
            np.ones(6),
            dims=("flattened",),
        )

        flattener = MagicMock()
        flattener.transform.return_value = flattened

        config = WeightsConfig(
            load_dir=weights_path,
        )

        with patch(
            "cccma_ppp.data_modules.weights.xr.open_dataset",
            return_value=loaded,
        ):
            result = config.build_weights(
                target_coords=make_target_coords(),
                Flattennanremover=flattener,
            )

        flattener.transform.assert_called_once_with(loaded)
        assert result is flattened

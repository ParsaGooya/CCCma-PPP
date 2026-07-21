from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from cccma_ppp.inference.predictor_abc import (
    PredictorABC,
    save_batch_to_netcdf,
)


class DummyOutput:
    def __init__(self, output):
        self.output = output


class DummyPredictor(PredictorABC):
    def __init__(
        self,
        output_dir,
        device="cpu",
        extract_training_vars=True,
    ):
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self._extract_training_vars = extract_training_vars
        self._stats = {"residual": object()}
        self.output_sampler = None
        self.module = object()

    @property
    def extract_training_vars(self):
        return self._extract_training_vars

    def _infer_on_batch(
        self,
        batch,
        _getting_train_stats=False,
    ):
        return batch

    def _batch_to_netcdf(
        self,
        output,
        metadata,
    ):
        return output, metadata

    def _update_train_stats(
        self,
        output,
        batch,
    ):
        return output, batch


@pytest.fixture
def predictor(tmp_path):
    return DummyPredictor(tmp_path)


@pytest.mark.pruned
                                
def test_temp_save_dir(predictor):
    assert predictor.temp_save_dir == predictor.output_dir / "_temp"


@pytest.mark.pruned
def test_stats_available_when_extracting(predictor):
    assert predictor.stats == predictor._stats


def test_stats_none_when_not_extracting(tmp_path):
    predictor = DummyPredictor(
        tmp_path,
        extract_training_vars=False,
    )

    assert predictor.stats is None


@pytest.mark.pruned
def test_raw_module_returns_plain_module(predictor):
    module = object()
    predictor.module = module

    assert predictor.raw_module is module


def test_raw_module_unwraps_ddp(
    monkeypatch,
    predictor,
):
    wrapped_module = object()

    class FakeDDP:
        def __init__(self, module):
            self.module = module

    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        FakeDDP,
    )

    predictor.module = FakeDDP(wrapped_module)

    assert predictor.raw_module is wrapped_module


@pytest.mark.pruned
def test_add_decoder_noise_uses_existing_sampler(
    predictor,
):
    def sampler(size):
        return torch.ones(
            *size,
            2,
        )

    predictor.output_sampler = sampler

    output = DummyOutput(
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
    )

    result = predictor.add_decoder_noise(
        output=output,
        num_output_samples=3,
        sample_size=(2,),
        reshape_size=(2,),
    )

    assert result is output
    assert result.output.shape == (3, 2, 2)

    torch.testing.assert_close(
        result.output[0],
        torch.tensor(
            [
                [2.0, 3.0],
                [4.0, 5.0],
            ]
        ),
    )


def test_add_decoder_noise_builds_sampler(
    monkeypatch,
    predictor,
):
    calls = []

    def fake_build():
        calls.append(True)

        return lambda size: torch.zeros(
            *size,
            2,
        )

    monkeypatch.setattr(
        predictor,
        "build_output_sampler",
        fake_build,
    )

    output = DummyOutput(torch.tensor([[1.0, 2.0]]))

    result = predictor.add_decoder_noise(
        output=output,
        num_output_samples=2,
        sample_size=(1,),
        reshape_size=(2,),
    )

    assert calls == [True]
    assert predictor.output_sampler is not None
    assert result.output.shape == (2, 1, 2)

    torch.testing.assert_close(
        result.output,
        torch.tensor(
            [
                [[1.0, 2.0]],
                [[1.0, 2.0]],
            ]
        ),
    )


def test_add_decoder_noise_preserves_dtype_and_device(
    predictor,
):
    recorded = {}

    def sampler(size):
        recorded["size"] = size

        return torch.ones(
            *size,
            2,
            dtype=torch.float64,
        )

    predictor.output_sampler = sampler

    output = DummyOutput(
        torch.tensor(
            [[1.0, 2.0]],
            dtype=torch.float32,
        )
    )

    result = predictor.add_decoder_noise(
        output=output,
        num_output_samples=2,
        sample_size=(1,),
        reshape_size=(2,),
    )

    assert recorded["size"] == (2, 1)
    assert result.output.shape == (2, 1, 2)
    assert result.output.dtype == torch.float32
    assert result.output.device.type == "cpu"


@pytest.mark.pruned
def test_add_decoder_noise_multiple_sample_dimensions(
    predictor,
):
    def sampler(size):
        return torch.zeros(
            *size,
            4,
        )

    predictor.output_sampler = sampler

    output = DummyOutput(torch.ones(2, 3, 4))

    result = predictor.add_decoder_noise(
        output=output,
        num_output_samples=5,
        sample_size=(2, 3),
        reshape_size=(4,),
    )

    assert result.output.shape == (
        5,
        2,
        3,
        4,
    )

    torch.testing.assert_close(
        result.output,
        torch.ones(5, 2, 3, 4),
    )


def test_build_output_sampler_missing_stats_raises(
    predictor,
):
    with pytest.raises(
        ValueError,
        match="Training statistics",
    ):
        predictor.build_output_sampler()


def test_build_output_sampler_loads_stats(
    monkeypatch,
    predictor,
):
    stats_path = predictor.output_dir / "training_variable_stats.pt"
    stats_path.touch()

    stats = {
        "residual_mean": torch.tensor([2.0, 3.0]),
        "residual_cov": torch.eye(2),
    }
    captured = {}

    def fake_load(
        path,
        map_location=None,
        **kwargs,
    ):
        captured["path"] = path
        captured["map_location"] = map_location
        return stats

    monkeypatch.setattr(
        torch,
        "load",
        fake_load,
    )

    def fake_sample(
        mu,
        cov,
        sample_size,
        std=1.0,
    ):
        captured["mu"] = mu
        captured["cov"] = cov
        captured["sample_size"] = sample_size
        captured["std"] = std

        return torch.zeros(
            *sample_size,
            2,
        )

    monkeypatch.setattr(
        predictor,
        "_sample",
        fake_sample,
    )

    sampler = predictor.build_output_sampler()
    samples = sampler((3, 4))

    assert captured["path"] == stats_path
    assert captured["map_location"] == predictor.device

    torch.testing.assert_close(
        captured["mu"],
        torch.zeros_like(stats["residual_mean"]),
    )
    torch.testing.assert_close(
        captured["cov"],
        stats["residual_cov"],
    )

    assert captured["sample_size"] == (3, 4)
    assert captured["std"] == 1.0
    assert samples.shape == (3, 4, 2)


@pytest.mark.pruned
def test_get_multinormal_valid_covariance(
    predictor,
):
    distribution = predictor._get_multinormal(
        mu=torch.tensor([0.0, 0.0]),
        cov=torch.eye(2),
    )

    assert isinstance(
        distribution,
        torch.distributions.MultivariateNormal,
    )
    assert distribution.loc.device.type == "cpu"


@pytest.mark.parametrize(
    "std",
    [
        0.0,
        -1.0,
        -100.0,
    ],
)
def test_get_multinormal_invalid_std_raises(
    predictor,
    std,
):
    with pytest.raises(
        ValueError,
        match="std must be positive",
    ):
        predictor._get_multinormal(
            mu=torch.zeros(2),
            cov=torch.eye(2),
            std=std,
        )


@pytest.mark.pruned
def test_get_multinormal_scales_covariance(
    monkeypatch,
    predictor,
):
    captured = {}

    class FakeDistribution:
        def __init__(
            self,
            loc,
            covariance_matrix,
        ):
            captured["loc"] = loc
            captured["covariance_matrix"] = covariance_matrix

    monkeypatch.setattr(
        torch.distributions,
        "MultivariateNormal",
        FakeDistribution,
    )

    predictor._get_multinormal(
        mu=torch.tensor([1.0, 2.0]),
        cov=torch.eye(2),
        std=2.0,
    )

    expected = torch.eye(2) * 4.0 + torch.eye(2) * 1e-6

    torch.testing.assert_close(
        captured["loc"],
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        captured["covariance_matrix"],
        expected,
    )


@pytest.mark.pruned
def test_get_multinormal_converts_dtype(
    predictor,
):
    distribution = predictor._get_multinormal(
        mu=torch.tensor(
            [0.0, 0.0],
            dtype=torch.float64,
        ),
        cov=torch.eye(
            2,
            dtype=torch.float64,
        ),
    )

    assert distribution.loc.dtype == torch.float32
    assert distribution.covariance_matrix.dtype == torch.float32


@pytest.mark.pruned
def test_get_multinormal_retries_after_value_error(
    monkeypatch,
    predictor,
):
    calls = []

    class FakeDistribution:
        def __init__(
            self,
            loc,
            covariance_matrix,
        ):
            calls.append(covariance_matrix.clone())

            if len(calls) < 3:
                raise ValueError("invalid covariance")

            self.loc = loc
            self.covariance_matrix = covariance_matrix

    monkeypatch.setattr(
        torch.distributions,
        "MultivariateNormal",
        FakeDistribution,
    )

    result = predictor._get_multinormal(
        mu=torch.zeros(2),
        cov=torch.eye(2),
    )

    assert len(calls) == 3
    assert isinstance(
        result,
        FakeDistribution,
    )

    torch.testing.assert_close(
        calls[0],
        torch.eye(2) + torch.eye(2) * 1e-6,
    )
    torch.testing.assert_close(
        calls[1],
        torch.eye(2) + torch.eye(2) * 1e-5,
    )
    torch.testing.assert_close(
        calls[2],
        torch.eye(2) + torch.eye(2) * 1e-4,
    )


def test_get_multinormal_all_retries_fail(
    monkeypatch,
    predictor,
):
    calls = []

    class FakeDistribution:
        def __init__(
            self,
            loc,
            covariance_matrix,
        ):
            calls.append(covariance_matrix.clone())
            raise ValueError("invalid covariance")

    monkeypatch.setattr(
        torch.distributions,
        "MultivariateNormal",
        FakeDistribution,
    )

    with pytest.raises(
        RuntimeError,
        match=("Could not construct MultivariateNormal"),
    ):
        predictor._get_multinormal(
            mu=torch.zeros(2),
            cov=torch.eye(2),
        )

    assert len(calls) == 5


def test_sample_integer_size(
    monkeypatch,
    predictor,
):
    captured = {}

    class FakeDistribution:
        def sample(self, size):
            captured["size"] = size

            return torch.ones(
                *size,
                2,
            )

    monkeypatch.setattr(
        predictor,
        "_get_multinormal",
        lambda **kwargs: FakeDistribution(),
    )

    samples = predictor._sample(
        mu=torch.zeros(2),
        cov=torch.eye(2),
        sample_size=3,
    )

    assert captured["size"] == (3,)
    assert samples.shape == (3, 2)
    assert samples.device.type == "cpu"


def test_sample_tuple_size(
    monkeypatch,
    predictor,
):
    captured = {}

    class FakeDistribution:
        def sample(self, size):
            captured["size"] = size

            return torch.ones(
                *size,
                2,
            )

    monkeypatch.setattr(
        predictor,
        "_get_multinormal",
        lambda **kwargs: FakeDistribution(),
    )

    samples = predictor._sample(
        mu=torch.zeros(2),
        cov=torch.eye(2),
        sample_size=(3, 4),
        std=2.0,
    )

    assert captured["size"] == (3, 4)
    assert samples.shape == (3, 4, 2)


@pytest.mark.pruned
def test_sample_passes_arguments_to_distribution(
    monkeypatch,
    predictor,
):
    captured = {}

    class FakeDistribution:
        def sample(self, size):
            return torch.ones(
                *size,
                2,
            )

    def fake_get_multinormal(
        mu,
        cov,
        std,
    ):
        captured["mu"] = mu
        captured["cov"] = cov
        captured["std"] = std

        return FakeDistribution()

    monkeypatch.setattr(
        predictor,
        "_get_multinormal",
        fake_get_multinormal,
    )

    mu = torch.tensor([1.0, 2.0])
    cov = torch.eye(2)

    predictor._sample(
        mu=mu,
        cov=cov,
        sample_size=2,
        std=3.0,
    )

    assert captured["mu"] is mu
    assert captured["cov"] is cov
    assert captured["std"] == 3.0


@pytest.mark.pruned
def test_save_batch_to_netcdf_basic(
    tmp_path,
):
    prediction = torch.arange(
        2 * 1 * 3,
        dtype=torch.float32,
    ).reshape(2, 1, 3)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {"year": 2000},
            {"year": 2001},
        ],
        num_output_dims=1,
        save_name="basic.nc",
        save_dir=tmp_path,
    )

    save_path = tmp_path / "basic.nc"

    assert save_path.exists()

    with xr.open_dataarray(save_path) as data:
        assert data.name == "prediction"
        assert "year" in data.dims
        assert "channels" in data.dims
        assert "output_dim_0" in data.dims
        assert list(data.coords["year"].values) == [2000, 2001]


@pytest.mark.pruned
def test_save_batch_to_netcdf_multiple_metadata_keys(
    tmp_path,
):
    prediction = torch.arange(
        2 * 1 * 2,
        dtype=torch.float32,
    ).reshape(2, 1, 2)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {
                "year": 2000,
                "lead_time": 1,
            },
            {
                "year": 2001,
                "lead_time": 2,
            },
        ],
        num_output_dims=1,
        save_name="metadata.nc",
        save_dir=tmp_path,
    )

    with xr.open_dataarray(tmp_path / "metadata.nc") as data:
        assert "year" in data.dims
        assert "lead_time" in data.dims
        assert "channels" in data.dims


def test_save_batch_to_netcdf_with_extra_dimension(
    tmp_path,
):
    prediction = torch.arange(
        2 * 3 * 1 * 4,
        dtype=torch.float32,
    ).reshape(2, 3, 1, 4)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {"year": 2000},
            {"year": 2001},
            {"year": 2002},
        ],
        num_output_dims=1,
        save_name="samples.nc",
        save_dir=tmp_path,
        extra_dims_sorted=["sample"],
    )

    with xr.open_dataarray(tmp_path / "samples.nc") as data:
        assert "sample" in data.dims

        np.testing.assert_array_equal(
            data.coords["sample"].values,
            np.array([1, 2]),
        )


@pytest.mark.pruned
def test_save_batch_to_netcdf_multiple_extra_dimensions(
    tmp_path,
):
    prediction = torch.arange(
        2 * 3 * 1 * 1 * 4,
        dtype=torch.float32,
    ).reshape(2, 3, 1, 1, 4)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {"year": 2000},
        ],
        num_output_dims=1,
        save_name="extra.nc",
        save_dir=tmp_path,
        extra_dims_sorted=[
            "sample",
            "member",
        ],
    )

    with xr.open_dataarray(tmp_path / "extra.nc") as data:
        assert "sample" in data.dims
        assert "member" in data.dims

        np.testing.assert_array_equal(
            data.coords["sample"].values,
            np.array([1, 2]),
        )
        np.testing.assert_array_equal(
            data.coords["member"].values,
            np.array([1, 2, 3]),
        )


@pytest.mark.pruned
def test_save_batch_to_netcdf_two_output_dimensions(
    tmp_path,
):
    prediction = torch.arange(
        2 * 1 * 3 * 4,
        dtype=torch.float32,
    ).reshape(2, 1, 3, 4)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {"year": 2000},
            {"year": 2001},
        ],
        num_output_dims=2,
        save_name="spatial.nc",
        save_dir=tmp_path,
    )

    with xr.open_dataarray(tmp_path / "spatial.nc") as data:
        assert "output_dim_0" in data.dims
        assert "output_dim_1" in data.dims
        assert data.sizes["output_dim_0"] == 3
        assert data.sizes["output_dim_1"] == 4


def test_save_batch_to_netcdf_dimension_mismatch(
    tmp_path,
):
    prediction = torch.ones(2, 1, 3)

    with pytest.raises(
        ValueError,
        match=("Expected prediction with 4 dimensions"),
    ):
        save_batch_to_netcdf(
            prediction=prediction,
            metadata=[
                {"year": 2000},
                {"year": 2001},
            ],
            num_output_dims=2,
            save_name="bad.nc",
            save_dir=tmp_path,
        )


@pytest.mark.pruned
def test_save_batch_to_netcdf_metadata_length_mismatch(
    tmp_path,
):
    prediction = torch.ones(2, 1, 3)

    with pytest.raises(
        ValueError,
        match="metadata length",
    ):
        save_batch_to_netcdf(
            prediction=prediction,
            metadata=[
                {"year": 2000},
            ],
            num_output_dims=1,
            save_name="bad.nc",
            save_dir=tmp_path,
        )


@pytest.mark.pruned
def test_save_batch_to_netcdf_assign_coords(
    tmp_path,
):
    prediction = torch.ones(2, 1, 3)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {"year": 2000},
            {"year": 2001},
        ],
        num_output_dims=1,
        save_name="coords.nc",
        save_dir=tmp_path,
        assign_coords={
            "output_dim_0": [
                10.0,
                20.0,
                30.0,
            ]
        },
    )

    with xr.open_dataarray(tmp_path / "coords.nc") as data:
        np.testing.assert_array_equal(
            data.coords["output_dim_0"].values,
            np.array(
                [
                    10.0,
                    20.0,
                    30.0,
                ]
            ),
        )


def test_save_batch_to_netcdf_attrs(
    tmp_path,
):
    prediction = torch.ones(2, 1, 3)

    attrs = {
        "units": "K",
        "description": "test prediction",
    }

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {"year": 2000},
            {"year": 2001},
        ],
        num_output_dims=1,
        save_name="attrs.nc",
        save_dir=tmp_path,
        attrs=attrs,
    )

    with xr.open_dataarray(tmp_path / "attrs.nc") as data:
        assert data.attrs["units"] == "K"
        assert data.attrs["description"] == "test prediction"


@pytest.mark.pruned
def test_save_batch_to_netcdf_channel_coordinates(
    tmp_path,
):
    prediction = torch.ones(2, 3, 4)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {"year": 2000},
            {"year": 2001},
        ],
        num_output_dims=1,
        save_name="channels.nc",
        save_dir=tmp_path,
    )

    with xr.open_dataarray(tmp_path / "channels.nc") as data:
        np.testing.assert_array_equal(
            data.coords["channels"].values,
            np.array([1, 2, 3]),
        )


@pytest.mark.pruned
def test_save_batch_to_netcdf_zero_output_dimensions(
    tmp_path,
):
    prediction = torch.ones(2, 3)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {"year": 2000},
            {"year": 2001},
        ],
        num_output_dims=0,
        save_name="scalar.nc",
        save_dir=tmp_path,
    )

    with xr.open_dataarray(tmp_path / "scalar.nc") as data:
        assert "year" in data.dims
        assert "channels" in data.dims


@pytest.mark.pruned
def test_save_batch_to_netcdf_returns_none(
    tmp_path,
):
    result = save_batch_to_netcdf(
        prediction=torch.ones(1, 1, 2),
        metadata=[
            {"year": 2000},
        ],
        num_output_dims=1,
        save_name="return.nc",
        save_dir=tmp_path,
    )

    assert result is None


@pytest.mark.pruned
def test_save_batch_to_netcdf_preserves_float_values(
    tmp_path,
):
    prediction = torch.tensor(
        [
            [
                [
                    [1.25, 2.5],
                ]
            ],
            [
                [
                    [3.75, 5.0],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    assert prediction.shape == (
        2,
        1,
        1,
        2,
    )

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {"year": 2000},
            {"year": 2001},
        ],
        num_output_dims=2,
        save_name="values.nc",
        save_dir=tmp_path,
    )

    with xr.open_dataarray(tmp_path / "values.nc") as data:
        selected = data.sel(
            year=2000,
            channels=1,
        )

        assert np.isclose(
            selected.values[0, 0],
            1.25,
        )
        assert np.isclose(
            selected.values[0, 1],
            2.5,
        )


@pytest.mark.pruned
def test_save_batch_to_netcdf_overwrites_existing_file(
    tmp_path,
):
    save_path = tmp_path / "overwrite.nc"
    save_path.write_bytes(b"old")

    save_batch_to_netcdf(
        prediction=torch.ones(1, 1, 2),
        metadata=[
            {"year": 2000},
        ],
        num_output_dims=1,
        save_name="overwrite.nc",
        save_dir=tmp_path,
    )

    with xr.open_dataarray(save_path) as data:
        assert data.name == "prediction"
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import xarray as xr

import cccma_ppp.inference.predictors.predictor_abc as module
from cccma_ppp.inference.predictors.predictor_abc import (
    PredictorABC,
    save_batch_to_netcdf,
)


class ConcretePredictor(PredictorABC):
    def __init__(
        self,
        output_dir,
        *,
        extract_training_vars=False,
    ):
        self.output_dir = Path(output_dir)
        self.device = torch.device("cpu")
        self.module = Mock()
        self.output_sampler = None
        self.num_output_covariance_sampling = 0
        self._extract_training_vars = extract_training_vars
        self._stats = {"residual": object()}

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


def make_output(output=None):
    return SimpleNamespace(output=(torch.zeros(2, 3) if output is None else output))


@pytest.mark.pruned
def test_stats_returns_none_when_training_variables_disabled(
    tmp_path,
):
    predictor = ConcretePredictor(
        tmp_path,
        extract_training_vars=False,
    )

    assert predictor.stats is None


@pytest.mark.pruned
def test_stats_returns_internal_statistics_when_enabled(
    tmp_path,
):
    predictor = ConcretePredictor(
        tmp_path,
        extract_training_vars=True,
    )

    assert predictor.stats is predictor._stats
    assert set(predictor.stats) == {"residual"}


@pytest.mark.pruned
def test_raw_module_returns_plain_module(tmp_path):
    predictor = ConcretePredictor(tmp_path)
    raw = object()
    predictor.module = raw

    assert predictor.raw_module is raw


def test_raw_module_unwraps_distributed_module(
    tmp_path,
    monkeypatch,
):
    class FakeDDP:
        def __init__(self, wrapped):
            self.module = wrapped

    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        FakeDDP,
    )

    raw = object()
    predictor = ConcretePredictor(tmp_path)
    predictor.module = FakeDDP(raw)

    assert predictor.raw_module is raw


def test_add_decoder_noise_builds_sampler_when_missing(
    tmp_path,
):
    predictor = ConcretePredictor(tmp_path)

    sampler = Mock(
        return_value=torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                [
                    [5.0, 6.0],
                    [7.0, 8.0],
                ],
            ]
        )
    )
    predictor.build_output_sampler = Mock(
        return_value=sampler,
    )

    output = make_output(
        torch.tensor(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ],
            dtype=torch.float32,
        )
    )

    result = predictor.add_decoder_noise(
        output=output,
        num_output_samples=2,
        sample_size=(2,),
        reshape_size=(2,),
    )

    predictor.build_output_sampler.assert_called_once()
    sampler.assert_called_once_with((2, 2))

    expected = torch.tensor(
        [
            [
                [11.0, 22.0],
                [33.0, 44.0],
            ],
            [
                [15.0, 26.0],
                [37.0, 48.0],
            ],
        ]
    )

    assert result is output
    torch.testing.assert_close(
        output.output,
        expected,
    )


@pytest.mark.pruned
def test_add_decoder_noise_reuses_existing_sampler(
    tmp_path,
):
    predictor = ConcretePredictor(tmp_path)

    sampler = Mock(
        return_value=torch.zeros(2, 3, 4),
    )
    predictor.output_sampler = sampler
    predictor.build_output_sampler = Mock()

    output = make_output(
        torch.ones(3, 4),
    )

    result = predictor.add_decoder_noise(
        output=output,
        num_output_samples=2,
        sample_size=(3,),
        reshape_size=(4,),
    )

    predictor.build_output_sampler.assert_not_called()
    sampler.assert_called_once_with((2, 3))

    assert result.output.shape == (2, 3, 4)

    torch.testing.assert_close(
        result.output,
        torch.ones(2, 3, 4),
    )


def test_add_decoder_noise_preserves_prediction_dtype(
    tmp_path,
):
    predictor = ConcretePredictor(tmp_path)

    predictor.output_sampler = Mock(
        return_value=torch.ones(
            2,
            3,
            4,
            dtype=torch.float64,
        )
    )

    output = make_output(
        torch.zeros(
            3,
            4,
            dtype=torch.float32,
        )
    )

    predictor.add_decoder_noise(
        output=output,
        num_output_samples=2,
        sample_size=(3,),
        reshape_size=(4,),
    )

    assert output.output.dtype == torch.float32


@pytest.mark.pruned
def test_add_decoder_noise_reshapes_flat_noise(
    tmp_path,
):
    predictor = ConcretePredictor(tmp_path)

    predictor.output_sampler = Mock(
        return_value=torch.arange(
            48,
            dtype=torch.float32,
        )
    )

    output = make_output(
        torch.zeros(2, 3, 4),
    )

    result = predictor.add_decoder_noise(
        output=output,
        num_output_samples=2,
        sample_size=(2,),
        reshape_size=(3, 4),
    )

    assert result.output.shape == (2, 2, 3, 4)


def test_build_output_sampler_requires_statistics_file(
    tmp_path,
):
    predictor = ConcretePredictor(tmp_path)

    with pytest.raises(
        ValueError,
        match="Training statistics.*must be saved",
    ):
        predictor.build_output_sampler()


def test_build_output_sampler_loads_statistics(
    tmp_path,
    monkeypatch,
):
    predictor = ConcretePredictor(tmp_path)

    residual_mean = torch.tensor([4.0, 5.0])
    residual_cov = torch.eye(2)

    stats_path = tmp_path / "training_variable_stats.pt"
    stats_path.touch()

    load_mock = Mock(
        return_value={
            "residual_mean": residual_mean,
            "residual_cov": residual_cov,
        }
    )
    monkeypatch.setattr(
        module.torch,
        "load",
        load_mock,
    )

    expected = torch.ones(3, 2)
    predictor._sample = Mock(
        return_value=expected,
    )

    sampler = predictor.build_output_sampler()
    result = sampler((3,))

    load_mock.assert_called_once_with(
        stats_path,
        map_location=predictor.device,
    )

    args = predictor._sample.call_args.args

    torch.testing.assert_close(
        args[0],
        torch.zeros_like(residual_mean),
    )
    torch.testing.assert_close(
        args[1],
        residual_cov,
    )
    assert args[2] == (3,)
    assert result is expected


@pytest.mark.pruned
def test_output_sampler_accepts_integer_sample_size(
    tmp_path,
    monkeypatch,
):
    predictor = ConcretePredictor(tmp_path)

    stats_path = tmp_path / "training_variable_stats.pt"
    stats_path.touch()

    monkeypatch.setattr(
        module.torch,
        "load",
        Mock(
            return_value={
                "residual_mean": torch.ones(2),
                "residual_cov": torch.eye(2),
            }
        ),
    )

    predictor._sample = Mock(
        return_value=torch.ones(4, 2),
    )

    sampler = predictor.build_output_sampler()
    sampler(4)

    assert predictor._sample.call_args.args[2] == 4


@pytest.mark.parametrize(
    "std",
    [
        0.0,
        -0.1,
        -1.0,
    ],
)
def test_get_multinormal_rejects_nonpositive_std(
    tmp_path,
    std,
):
    predictor = ConcretePredictor(tmp_path)

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
def test_get_multinormal_returns_distribution(tmp_path):
    predictor = ConcretePredictor(tmp_path)

    distribution = predictor._get_multinormal(
        mu=torch.tensor([1.0, 2.0]),
        cov=torch.eye(2),
    )

    assert isinstance(
        distribution,
        torch.distributions.MultivariateNormal,
    )

    torch.testing.assert_close(
        distribution.loc,
        torch.tensor([1.0, 2.0]),
    )


@pytest.mark.pruned
def test_get_multinormal_converts_input_to_float(
    tmp_path,
):
    predictor = ConcretePredictor(tmp_path)

    distribution = predictor._get_multinormal(
        mu=torch.tensor([1, 2]),
        cov=torch.eye(2, dtype=torch.float64),
    )

    assert distribution.loc.dtype == torch.float32
    assert distribution.covariance_matrix.dtype == torch.float32


@pytest.mark.pruned
def test_get_multinormal_scales_covariance_by_std_squared(
    tmp_path,
):
    predictor = ConcretePredictor(tmp_path)

    distribution = predictor._get_multinormal(
        mu=torch.zeros(2),
        cov=torch.eye(2),
        std=3.0,
    )

    expected = 9.0 * torch.eye(2)
    actual = distribution.covariance_matrix

    torch.testing.assert_close(
        actual,
        expected,
        atol=2e-6,
        rtol=0,
    )


@pytest.mark.pruned
def test_get_multinormal_retries_after_value_error(
    tmp_path,
    monkeypatch,
):
    predictor = ConcretePredictor(tmp_path)

    expected = object()
    constructor = Mock(
        side_effect=[
            ValueError("first"),
            ValueError("second"),
            expected,
        ]
    )

    monkeypatch.setattr(
        module.torch.distributions,
        "MultivariateNormal",
        constructor,
    )

    result = predictor._get_multinormal(
        mu=torch.zeros(2),
        cov=torch.eye(2),
    )

    assert result is expected
    assert constructor.call_count == 3


def test_get_multinormal_raises_after_all_retries(
    tmp_path,
    monkeypatch,
):
    predictor = ConcretePredictor(tmp_path)

    constructor = Mock(
        side_effect=ValueError("invalid covariance"),
    )

    monkeypatch.setattr(
        module.torch.distributions,
        "MultivariateNormal",
        constructor,
    )

    with pytest.raises(
        RuntimeError,
        match="Could not construct MultivariateNormal",
    ):
        predictor._get_multinormal(
            mu=torch.zeros(2),
            cov=torch.eye(2),
        )

    assert constructor.call_count == 5


@pytest.mark.pruned
def test_get_multinormal_increases_jitter_between_retries(
    tmp_path,
    monkeypatch,
):
    predictor = ConcretePredictor(tmp_path)

    covariances = []

    def constructor(*, loc, covariance_matrix):
        covariances.append(covariance_matrix.detach().clone())

        if len(covariances) < 3:
            raise ValueError("retry")

        return object()

    monkeypatch.setattr(
        module.torch.distributions,
        "MultivariateNormal",
        constructor,
    )

    predictor._get_multinormal(
        mu=torch.zeros(2),
        cov=torch.eye(2),
    )

    first_jitter = covariances[0][0, 0] - 1
    second_jitter = covariances[1][0, 0] - 1
    third_jitter = covariances[2][0, 0] - 1

    assert second_jitter > first_jitter
    assert third_jitter > second_jitter


def test_sample_converts_integer_size_to_tuple(
    tmp_path,
):
    predictor = ConcretePredictor(tmp_path)

    distribution = Mock()
    distribution.sample.return_value = torch.ones(3, 2)

    predictor._get_multinormal = Mock(
        return_value=distribution,
    )

    result = predictor._sample(
        mu=torch.zeros(2),
        cov=torch.eye(2),
        sample_size=3,
        std=2.0,
    )

    predictor._get_multinormal.assert_called_once()

    kwargs = predictor._get_multinormal.call_args.kwargs

    torch.testing.assert_close(
        kwargs["mu"],
        torch.zeros(2),
    )
    torch.testing.assert_close(
        kwargs["cov"],
        torch.eye(2),
    )
    assert kwargs["std"] == 2.0

    distribution.sample.assert_called_once_with((3,))
    assert result.device == predictor.device


def test_sample_preserves_tuple_size(tmp_path):
    predictor = ConcretePredictor(tmp_path)

    distribution = Mock()
    distribution.sample.return_value = torch.ones(2, 3, 4)

    predictor._get_multinormal = Mock(
        return_value=distribution,
    )

    result = predictor._sample(
        mu=torch.zeros(4),
        cov=torch.eye(4),
        sample_size=(2, 3),
    )

    distribution.sample.assert_called_once_with((2, 3))
    assert result.shape == (2, 3, 4)


@pytest.mark.pruned
def test_sample_default_size_is_one(tmp_path):
    predictor = ConcretePredictor(tmp_path)

    distribution = Mock()
    distribution.sample.return_value = torch.ones(1, 2)

    predictor._get_multinormal = Mock(
        return_value=distribution,
    )

    predictor._sample(
        mu=torch.zeros(2),
        cov=torch.eye(2),
    )

    distribution.sample.assert_called_once_with((1,))


@pytest.mark.pruned
def test_save_batch_to_netcdf_rejects_wrong_rank(
    tmp_path,
):
    prediction = torch.zeros(2, 3)

    with pytest.raises(
        ValueError,
        match="Expected prediction with 3 dimensions",
    ):
        save_batch_to_netcdf(
            prediction=prediction,
            metadata=[
                {"year": 2000},
                {"year": 2001},
            ],
            num_output_dims=2,
            save_name="prediction.nc",
            save_dir=tmp_path,
        )


def test_save_batch_to_netcdf_rejects_metadata_mismatch(
    tmp_path,
):
    prediction = torch.zeros(2, 1, 3)

    with pytest.raises(
        ValueError,
        match="metadata length.*does not match batch size",
    ):
        save_batch_to_netcdf(
            prediction=prediction,
            metadata=[
                {"year": 2000},
            ],
            num_output_dims=2,
            save_name="prediction.nc",
            save_dir=tmp_path,
        )


@pytest.mark.pruned
def test_save_batch_to_netcdf_basic_output(tmp_path):
    prediction = torch.arange(
        12,
        dtype=torch.float32,
    ).reshape(2, 2, 3)

    metadata = [
        {
            "year": 2000,
            "month": 1,
        },
        {
            "year": 2001,
            "month": 2,
        },
    ]

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=metadata,
        num_output_dims=2,
        save_name="prediction.nc",
        save_dir=tmp_path,
    )

    save_path = tmp_path / "prediction.nc"

    assert save_path.exists()

    with xr.open_dataarray(save_path) as result:
        assert result.name == "prediction"
        assert result.sizes["channels"] == 2
        assert result.sizes["output_dim_0"] == 3
        assert "year" in result.dims
        assert "month" in result.dims


def test_save_batch_to_netcdf_with_extra_dimensions(
    tmp_path,
):
    prediction = torch.arange(
        24,
        dtype=torch.float32,
    ).reshape(2, 2, 2, 3)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {
                "year": 2000,
                "month": 1,
            },
            {
                "year": 2001,
                "month": 2,
            },
        ],
        num_output_dims=2,
        save_name="extra.nc",
        save_dir=tmp_path,
        extra_dims_sorted=["samples"],
    )

    with xr.open_dataarray(tmp_path / "extra.nc") as result:
        assert result.dims[0] == "samples"
        assert result.sizes["samples"] == 2
        assert result.sizes["channels"] == 2
        assert result.sizes["output_dim_0"] == 3

        np.testing.assert_array_equal(
            result.coords["samples"].values,
            np.asarray([1, 2]),
        )


@pytest.mark.pruned
def test_save_batch_to_netcdf_assigns_coordinates(
    tmp_path,
):
    prediction = torch.ones(2, 2, 3)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {
                "year": 2000,
            },
            {
                "year": 2001,
            },
        ],
        num_output_dims=2,
        save_name="coords.nc",
        save_dir=tmp_path,
        assign_coords={
            "channels": [
                "tas",
                "pr",
            ],
        },
    )

    with xr.open_dataarray(tmp_path / "coords.nc") as result:
        assert list(result.coords["channels"].values) == [
            "tas",
            "pr",
        ]


def test_save_batch_to_netcdf_assigns_attributes(
    tmp_path,
):
    prediction = torch.ones(2, 1, 3)

    attrs = {
        "model": "cvae",
        "training_statistics": 1,
    }

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {
                "year": 2000,
            },
            {
                "year": 2001,
            },
        ],
        num_output_dims=2,
        save_name="attrs.nc",
        save_dir=tmp_path,
        attrs=attrs,
    )

    with xr.open_dataarray(tmp_path / "attrs.nc") as result:
        assert result.attrs["model"] == "cvae"
        assert result.attrs["training_statistics"] in {
            True,
            1,
            np.int8(1),
        }


@pytest.mark.pruned
def test_save_batch_to_netcdf_defaults_extra_dimensions(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_to_netcdf(self, path):
        captured["data_array"] = self
        captured["path"] = path

    monkeypatch.setattr(
        xr.DataArray,
        "to_netcdf",
        fake_to_netcdf,
    )

    prediction = torch.ones(2, 1, 3)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {
                "year": 2000,
            },
            {
                "year": 2001,
            },
        ],
        num_output_dims=2,
        save_name="default.nc",
        save_dir=tmp_path,
        extra_dims_sorted=None,
    )

    result = captured["data_array"]

    assert "batch" not in result.dims
    assert "year" in result.dims
    assert "channels" in result.dims
    assert "output_dim_0" in result.dims
    assert captured["path"] == tmp_path / "default.nc"


@pytest.mark.pruned
def test_save_batch_to_netcdf_creates_one_based_channel_coordinates(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_to_netcdf(self, path):
        captured["data_array"] = self

    monkeypatch.setattr(
        xr.DataArray,
        "to_netcdf",
        fake_to_netcdf,
    )

    prediction = torch.ones(2, 3, 4)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {
                "year": 2000,
            },
            {
                "year": 2001,
            },
        ],
        num_output_dims=2,
        save_name="channels.nc",
        save_dir=tmp_path,
    )

    result = captured["data_array"]

    np.testing.assert_array_equal(
        result.coords["channels"].values,
        np.asarray([1, 2, 3]),
    )


@pytest.mark.pruned
def test_save_batch_to_netcdf_creates_spatial_coordinates(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_to_netcdf(self, path):
        captured["data_array"] = self

    monkeypatch.setattr(
        xr.DataArray,
        "to_netcdf",
        fake_to_netcdf,
    )

    prediction = torch.ones(2, 1, 3, 4)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {
                "year": 2000,
            },
            {
                "year": 2001,
            },
        ],
        num_output_dims=3,
        save_name="spatial.nc",
        save_dir=tmp_path,
    )

    result = captured["data_array"]

    assert result.sizes["output_dim_0"] == 3
    assert result.sizes["output_dim_1"] == 4

    np.testing.assert_array_equal(
        result.coords["output_dim_0"].values,
        np.arange(3),
    )
    np.testing.assert_array_equal(
        result.coords["output_dim_1"].values,
        np.arange(4),
    )


@pytest.mark.pruned
def test_save_batch_to_netcdf_preserves_metadata_index_order(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_to_netcdf(self, path):
        captured["data_array"] = self

    monkeypatch.setattr(
        xr.DataArray,
        "to_netcdf",
        fake_to_netcdf,
    )

    prediction = torch.ones(2, 1, 3)

    save_batch_to_netcdf(
        prediction=prediction,
        metadata=[
            {
                "year": 2000,
                "month": 1,
            },
            {
                "year": 2001,
                "month": 2,
            },
        ],
        num_output_dims=2,
        save_name="metadata.nc",
        save_dir=tmp_path,
    )

    result = captured["data_array"]

    assert "year" in result.dims
    assert "month" in result.dims

    np.testing.assert_array_equal(
        result.coords["year"].values,
        np.asarray([2000, 2001]),
    )
    np.testing.assert_array_equal(
        result.coords["month"].values,
        np.asarray([1, 2]),
    )

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import cccma_ppp.inference.predictors_lib.deterministic as module
from cccma_ppp.inference.predictors_lib.deterministic import (
    DeterministicPredictorConfig,
    DetermninisticPredictor,
)


class DummyDistributed:
    def __init__(
        self,
        device: torch.device | None = None,
        rank: int = 0,
    ):
        self.device = device or torch.device("cpu")
        self.rank = rank


class DummyRunningCovariance:
    def __init__(self, distributed):
        self.distributed = distributed
        self.values = []

    def update(self, value):
        self.values.append(value)


class DummyModule:
    def __init__(
        self,
        *,
        generator=True,
        num_output_dims=1,
    ):
        self.model_config = SimpleNamespace(
            GENERATOR=generator,
            NUM_OUTPUT_DIMS=num_output_dims,
        )

        self.eval_called = False
        self.forward_output = None
        self.predict_output = None
        self.forward_kwargs = None
        self.predict_kwargs = None

    def eval(self):
        self.eval_called = True
        return self

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return self.forward_output

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return self.predict_output


class DummyBatch:
    def __init__(
        self,
        input=None,
        target=None,
        metadata=None,
    ):
        self.input = torch.ones(2, 3) if input is None else input
        self.target = target
        self.metadata = (
            [
                {"year": 2000},
                {"year": 2001},
            ]
            if metadata is None
            else metadata
        )


def make_output(output=None):
    return SimpleNamespace(output=(torch.zeros(2, 3) if output is None else output))


def make_predictor(
    tmp_path,
    *,
    generator=True,
    num_output_sampling=1,
    num_output_dims=1,
    rank=0,
):
    config = DeterministicPredictorConfig()
    model = DummyModule(
        generator=generator,
        num_output_dims=num_output_dims,
    )
    distributed = DummyDistributed(rank=rank)

    predictor = DetermninisticPredictor(
        config=config,
        module=model,
        distributed=distributed,
        output_dir=tmp_path,
        num_output_sampling=num_output_sampling,
    )

    return predictor, model


@pytest.fixture
def running_covariance_stub(monkeypatch):
    monkeypatch.setattr(
        module,
        "RunningCovariance",
        DummyRunningCovariance,
    )


@pytest.fixture
def clear_memory_mock(monkeypatch):
    clear_mock = Mock()

    monkeypatch.setattr(
        module,
        "clear_memory",
        clear_mock,
    )

    return clear_mock


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_type_is_deterministic():
    config = DeterministicPredictorConfig()

    assert config._type == "deterministic"


def test_config_build_returns_predictor(tmp_path):
    config = DeterministicPredictorConfig()
    model = DummyModule(generator=True)
    distributed = DummyDistributed()

    predictor = config.build(
        module=model,
        distributed=distributed,
        output_dir=tmp_path,
        num_output_sampling=3,
    )

    assert isinstance(
        predictor,
        DetermninisticPredictor,
    )
    assert predictor.config is config
    assert predictor.module is model
    assert predictor.distributed is distributed
    assert predictor.output_dir == Path(tmp_path)
    assert predictor.num_output_sampling == 3


# ---------------------------------------------------------------------------
# Initialization and properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_output_sampling",
    [
        0,
        -1,
        -10,
    ],
)
def test_predictor_rejects_nonpositive_output_sampling(
    tmp_path,
    num_output_sampling,
):
    with pytest.raises(
        ValueError,
        match="num_output_sampling must be larger than 1",
    ):
        DetermninisticPredictor(
            config=DeterministicPredictorConfig(),
            module=DummyModule(),
            distributed=DummyDistributed(),
            output_dir=tmp_path,
            num_output_sampling=num_output_sampling,
        )


def test_predictor_initial_state(tmp_path):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
        num_output_sampling=2,
    )

    assert predictor.module is model
    assert predictor.output_dir == Path(tmp_path)
    assert predictor.device == torch.device("cpu")
    assert predictor.output_sampler is None
    assert predictor._batch_counter == 0


def test_generator_disables_training_variable_extraction(
    tmp_path,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        num_output_sampling=4,
    )

    assert predictor.num_output_covariance_sampling == 0
    assert predictor.extract_training_vars is False
    assert predictor.stats is None


def test_non_generator_enables_training_variable_extraction(
    tmp_path,
    running_covariance_stub,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=None,
        num_output_sampling=4,
    )

    assert predictor.num_output_covariance_sampling == 4
    assert predictor.extract_training_vars is True
    assert set(predictor.stats) == {"residual"}
    assert isinstance(
        predictor.stats["residual"],
        DummyRunningCovariance,
    )
    assert predictor.stats["residual"].distributed is predictor.distributed


# ---------------------------------------------------------------------------
# Training statistics
# ---------------------------------------------------------------------------


def test_update_train_stats_updates_residual_covariance(
    tmp_path,
    running_covariance_stub,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=None,
        num_output_sampling=2,
    )

    prediction = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    target = torch.tensor(
        [
            [2.0, 4.0],
            [6.0, 8.0],
        ]
    )

    result = predictor._update_train_stats(
        make_output(output=prediction),
        DummyBatch(target=target),
    )

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    assert result is predictor.stats
    assert len(result["residual"].values) == 1

    torch.testing.assert_close(
        result["residual"].values[0],
        expected,
    )


def test_update_train_stats_flattens_nonbatch_dimensions(
    tmp_path,
    running_covariance_stub,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=None,
        num_output_sampling=2,
    )

    prediction = torch.zeros(2, 3, 4)
    target = torch.ones(2, 3, 4)

    predictor._update_train_stats(
        make_output(output=prediction),
        DummyBatch(target=target),
    )

    residual = predictor.stats["residual"].values[0]

    assert residual.shape == (2, 12)

    torch.testing.assert_close(
        residual,
        torch.ones(2, 12),
    )


def test_update_train_stats_preserves_batch_dimension(
    tmp_path,
    running_covariance_stub,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=None,
        num_output_sampling=2,
    )

    prediction = torch.zeros(4, 2, 3)
    target = torch.ones(4, 2, 3)

    predictor._update_train_stats(
        make_output(output=prediction),
        DummyBatch(target=target),
    )

    residual = predictor.stats["residual"].values[0]

    assert residual.shape == (4, 6)


def test_update_train_stats_uses_target_minus_prediction(
    tmp_path,
    running_covariance_stub,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=None,
        num_output_sampling=2,
    )

    prediction = torch.tensor(
        [
            [5.0, 3.0],
            [1.0, 7.0],
        ]
    )
    target = torch.tensor(
        [
            [8.0, 2.0],
            [4.0, 10.0],
        ]
    )

    predictor._update_train_stats(
        make_output(output=prediction),
        DummyBatch(target=target),
    )

    expected = torch.tensor(
        [
            [3.0, -1.0],
            [3.0, 3.0],
        ]
    )

    torch.testing.assert_close(
        predictor.stats["residual"].values[0],
        expected,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_infer_on_batch_requires_target_for_training_stats(
    tmp_path,
    clear_memory_mock,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
    )

    with pytest.raises(
        RuntimeError,
        match="dataset must contain the target prediction",
    ):
        predictor._infer_on_batch(
            DummyBatch(target=None),
            _getting_train_stats=True,
        )

    clear_memory_mock.assert_called_once()
    assert model.eval_called is True
    assert model.forward_kwargs is None
    assert model.predict_kwargs is None


def test_infer_on_batch_returns_training_stats(
    tmp_path,
    clear_memory_mock,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
    )

    forward_output = make_output()
    model.forward_output = forward_output

    expected = {
        "residual": object(),
    }
    predictor._update_train_stats = Mock(
        return_value=expected,
    )

    batch = DummyBatch(
        target=torch.ones(2, 3),
    )

    result = predictor._infer_on_batch(
        batch,
        _getting_train_stats=True,
    )

    assert result is expected
    assert model.eval_called is True
    assert model.forward_kwargs == {
        "data": batch,
    }
    assert model.predict_kwargs is None

    predictor._update_train_stats.assert_called_once_with(
        forward_output,
        batch,
    )


def test_infer_on_batch_predicts_and_saves(
    tmp_path,
    clear_memory_mock,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
        num_output_sampling=3,
    )

    output = make_output(
        output=torch.zeros(2, 3),
    )
    model.predict_output = output

    predictor._batch_to_netcdf = Mock()

    batch = DummyBatch()

    result = predictor._infer_on_batch(batch)

    assert result is output
    assert model.eval_called is True
    assert model.forward_kwargs is None
    assert model.predict_kwargs == {
        "data": batch,
        "output_sample_size": 3,
    }

    predictor._batch_to_netcdf.assert_called_once_with(
        output,
        batch.metadata,
    )
    clear_memory_mock.assert_called_once()


def test_infer_on_batch_skips_noise_for_generator(
    tmp_path,
    clear_memory_mock,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
        num_output_sampling=2,
    )

    output = make_output(
        output=torch.zeros(3, 4),
    )
    model.predict_output = output

    predictor.add_decoder_noise = Mock()
    predictor._batch_to_netcdf = Mock()

    result = predictor._infer_on_batch(
        DummyBatch(),
    )

    assert result is output
    predictor.add_decoder_noise.assert_not_called()


def test_infer_on_batch_adds_decoder_noise_for_non_generator(
    tmp_path,
    running_covariance_stub,
    clear_memory_mock,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=None,
        num_output_sampling=2,
    )

    original = make_output(
        output=torch.zeros(3, 4, 5),
    )
    noisy = make_output(
        output=torch.ones(2, 3, 4, 5),
    )
    model.predict_output = original

    predictor.add_decoder_noise = Mock(
        return_value=noisy,
    )
    predictor._batch_to_netcdf = Mock()

    batch = DummyBatch()

    result = predictor._infer_on_batch(batch)

    assert result is noisy

    predictor.add_decoder_noise.assert_called_once_with(
        original,
        2,
        (3,),
        torch.Size([4, 5]),
    )

    predictor._batch_to_netcdf.assert_called_once_with(
        noisy,
        batch.metadata,
    )


def test_infer_on_batch_uses_prediction_batch_as_sample_size(
    tmp_path,
    running_covariance_stub,
    clear_memory_mock,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=None,
        num_output_sampling=5,
    )

    output = make_output(
        output=torch.zeros(7, 2, 3),
    )
    model.predict_output = output

    predictor.add_decoder_noise = Mock(
        return_value=output,
    )
    predictor._batch_to_netcdf = Mock()

    predictor._infer_on_batch(
        DummyBatch(),
    )

    predictor.add_decoder_noise.assert_called_once_with(
        output,
        5,
        (7,),
        torch.Size([2, 3]),
    )


# ---------------------------------------------------------------------------
# NetCDF delegation
# ---------------------------------------------------------------------------


def test_batch_to_netcdf_delegates_prediction(
    tmp_path,
    monkeypatch,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
        num_output_sampling=2,
        num_output_dims=2,
    )

    predictor.temp_save_dir.mkdir(exist_ok=True)

    save_mock = Mock()

    monkeypatch.setattr(
        module,
        "save_batch_to_netcdf",
        save_mock,
    )

    prediction = torch.ones(2, 3, 4)
    output = make_output(output=prediction)
    metadata = [{"year": 2000}]

    predictor._batch_to_netcdf(
        output,
        metadata,
    )

    save_mock.assert_called_once()
    args = save_mock.call_args.args
    torch.testing.assert_close(args[0], prediction)
    assert args[1] == metadata
    assert args[2] == model.model_config.NUM_OUTPUT_DIMS
    assert args[3] == "prediction_rank0_00000000.nc"
    assert args[4] == predictor.temp_save_dir
    assert args[5] == ["output_samples"]
    assert args[6] is None

    assert predictor._batch_counter == 1


def test_batch_to_netcdf_detaches_prediction(
    tmp_path,
    monkeypatch,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
    )

    predictor.temp_save_dir.mkdir(exist_ok=True)

    save_mock = Mock()

    monkeypatch.setattr(
        module,
        "save_batch_to_netcdf",
        save_mock,
    )

    prediction = torch.ones(
        2,
        3,
        requires_grad=True,
    )

    predictor._batch_to_netcdf(
        make_output(output=prediction),
        [{"year": 2000}],
    )

    saved_prediction = save_mock.call_args.args[0]

    assert saved_prediction.device.type == "cpu"
    assert saved_prediction.requires_grad is False

    torch.testing.assert_close(
        saved_prediction,
        prediction.detach().cpu(),
    )


def test_batch_to_netcdf_uses_model_output_dimensions(
    tmp_path,
    monkeypatch,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        num_output_dims=4,
    )

    predictor.temp_save_dir.mkdir(exist_ok=True)

    save_mock = Mock()

    monkeypatch.setattr(
        module,
        "save_batch_to_netcdf",
        save_mock,
    )

    predictor._batch_to_netcdf(
        make_output(
            output=torch.ones(2, 3, 4),
        ),
        [{"year": 2000}],
    )

    assert save_mock.call_args.args[2] == 4


def test_batch_to_netcdf_uses_distributed_rank(
    tmp_path,
    monkeypatch,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        rank=7,
    )

    predictor.temp_save_dir.mkdir(exist_ok=True)

    save_mock = Mock()

    monkeypatch.setattr(
        module,
        "save_batch_to_netcdf",
        save_mock,
    )

    predictor._batch_to_netcdf(
        make_output(),
        [{"year": 2000}],
    )

    assert save_mock.call_args.args[3] == "prediction_rank7_00000000.nc"


def test_batch_to_netcdf_uses_zero_padded_counter(
    tmp_path,
    monkeypatch,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        rank=3,
    )

    predictor._batch_counter = 42
    predictor.temp_save_dir.mkdir(exist_ok=True)

    save_mock = Mock()

    monkeypatch.setattr(
        module,
        "save_batch_to_netcdf",
        save_mock,
    )

    predictor._batch_to_netcdf(
        make_output(),
        [{"year": 2000}],
    )

    assert save_mock.call_args.args[3] == "prediction_rank3_00000042.nc"
    assert predictor._batch_counter == 43


def test_batch_to_netcdf_increments_counter_each_call(
    tmp_path,
    monkeypatch,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
    )

    predictor.temp_save_dir.mkdir(exist_ok=True)

    save_mock = Mock()

    monkeypatch.setattr(
        module,
        "save_batch_to_netcdf",
        save_mock,
    )

    output = make_output()

    predictor._batch_to_netcdf(
        output,
        [{"year": 2000}],
    )
    predictor._batch_to_netcdf(
        output,
        [{"year": 2001}],
    )

    assert predictor._batch_counter == 2
    assert save_mock.call_count == 2

    first_name = save_mock.call_args_list[0].args[3]
    second_name = save_mock.call_args_list[1].args[3]

    assert first_name == "prediction_rank0_00000000.nc"
    assert second_name == "prediction_rank0_00000001.nc"

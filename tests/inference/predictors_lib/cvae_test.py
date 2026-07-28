from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import cccma_ppp.inference.predictors_lib.cvae as module
from cccma_ppp.inference.predictors_lib.cvae import (
    cVAEPredictor,
    cVAEPredictorConfig,
)


class DummyDistributed:
    def __init__(self):
        self.device = torch.device("cpu")
        self.rank = 0


class DummyRunningCovariance:
    def __init__(self, distributed):
        self.distributed = distributed
        self.values = []

    def update(self, value):
        self.values.append(value)


class DummyModule:
    def __init__(
        self,
        generator=None,
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
            [{"year": 2000}, {"year": 2001}] if metadata is None else metadata
        )


def make_output(
    output=None,
    samples=None,
    mu=None,
    log_var=None,
    cond_mu=None,
    cond_log_var=None,
):
    return SimpleNamespace(
        output=(torch.zeros(1, 2, 3) if output is None else output),
        samples=samples,
        mu=mu,
        log_var=log_var,
        cond_mu=cond_mu,
        cond_log_var=cond_log_var,
    )


def make_predictor(
    tmp_path,
    *,
    generator=True,
    num_output_sampling=1,
    num_latent_samples=2,
    infer_latent_samples_from_training=False,
    save_latent=False,
):
    config = cVAEPredictorConfig(
        num_latent_samples=num_latent_samples,
        infer_latent_samples_from_training=(infer_latent_samples_from_training),
        save_latent=save_latent,
    )
    model = DummyModule(generator=generator)

    predictor = cVAEPredictor(
        config=config,
        module=model,
        distributed=DummyDistributed(),
        output_dir=tmp_path,
        num_output_sampling=num_output_sampling,
    )
    return predictor, model


def test_config_rejects_zero_latent_samples():
    with pytest.raises(
        ValueError,
        match="num_latent_samples must be at least 1",
    ):
        cVAEPredictorConfig(num_latent_samples=0)


@pytest.mark.pruned
def test_config_rejects_negative_latent_samples():
    with pytest.raises(
        ValueError,
        match="num_latent_samples must be at least 1",
    ):
        cVAEPredictorConfig(num_latent_samples=-1)


@pytest.mark.parametrize("nstds", [0.0, -0.1, -10.0])
def test_config_rejects_nonpositive_nstds(nstds):
    with pytest.raises(
        ValueError,
        match="nstds must be positive",
    ):
        cVAEPredictorConfig(
            num_latent_samples=1,
            nstds=nstds,
        )


@pytest.mark.pruned
def test_config_warns_when_saving_latent():
    with pytest.warns(
        UserWarning,
        match="No predictions will be saved",
    ):
        config = cVAEPredictorConfig(
            num_latent_samples=1,
            save_latent=True,
        )

    assert config.save_latent is True


@pytest.mark.pruned
def test_config_type_is_cvae():
    config = cVAEPredictorConfig(
        num_latent_samples=1,
    )

    assert config._type == "cvae"


@pytest.mark.pruned
def test_config_build_returns_predictor(tmp_path):
    config = cVAEPredictorConfig(
        num_latent_samples=2,
    )
    model = DummyModule(generator=True)
    distributed = DummyDistributed()

    predictor = config.build(
        module=model,
        distributed=distributed,
        output_dir=tmp_path,
        num_output_sampling=3,
    )

    assert isinstance(predictor, cVAEPredictor)
    assert predictor.module is model
    assert predictor.num_output_sampling == 3


@pytest.mark.parametrize("num_output_sampling", [0, -1])
def test_predictor_rejects_nonpositive_output_sampling(
    tmp_path,
    num_output_sampling,
):
    config = cVAEPredictorConfig(
        num_latent_samples=1,
    )

    with pytest.raises(
        ValueError,
        match="num_output_sampling must be larger than 1",
    ):
        cVAEPredictor(
            config=config,
            module=DummyModule(),
            distributed=DummyDistributed(),
            output_dir=tmp_path,
            num_output_sampling=num_output_sampling,
        )


@pytest.mark.pruned
def test_non_generator_enables_output_covariance_sampling(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "RunningCovariance",
        DummyRunningCovariance,
    )

    predictor, _ = make_predictor(
        tmp_path,
        generator=None,
        num_output_sampling=4,
    )

    assert predictor.num_output_covariance_sampling == 4
    assert predictor.extract_training_residuals is True
    assert "residual" in predictor.stats


@pytest.mark.pruned
def test_generator_disables_output_covariance_sampling(
    tmp_path,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        num_output_sampling=4,
    )

    assert predictor.num_output_covariance_sampling == 0
    assert predictor.extract_training_residuals is False


@pytest.mark.pruned
def test_infer_training_samples_enables_posterior_stats(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "RunningCovariance",
        DummyRunningCovariance,
    )

    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        infer_latent_samples_from_training=True,
    )

    assert predictor.extract_posterior_samples is True
    assert predictor.extract_training_vars is True
    assert "samples" in predictor.stats


@pytest.mark.pruned
def test_predictor_initial_state(tmp_path):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
    )

    assert predictor.latent_sampler is None
    assert predictor.output_sampler is None
    assert predictor._batch_counter == 0
    assert predictor.output_dir == Path(tmp_path)
    assert predictor.device == torch.device("cpu")


@pytest.mark.pruned
def test_update_train_stats_updates_posterior_samples(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "RunningCovariance",
        DummyRunningCovariance,
    )

    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        infer_latent_samples_from_training=True,
    )

    samples = torch.arange(
        24,
        dtype=torch.float32,
    ).reshape(2, 3, 4)

    output = make_output(samples=samples)
    batch = DummyBatch()

    result = predictor._update_train_stats(
        output,
        batch,
    )

    stored = result["samples"].values[0]

    assert stored.shape == (6, 4)
    torch.testing.assert_close(
        stored,
        samples.reshape(-1, 4),
    )


def test_update_train_stats_requires_samples(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "RunningCovariance",
        DummyRunningCovariance,
    )

    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        infer_latent_samples_from_training=True,
    )

    with pytest.raises(
        RuntimeError,
        match="cVAEOutput.samples is required",
    ):
        predictor._update_train_stats(
            make_output(samples=None),
            DummyBatch(),
        )


@pytest.mark.pruned
def test_update_train_stats_updates_residuals(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "RunningCovariance",
        DummyRunningCovariance,
    )

    predictor, _ = make_predictor(
        tmp_path,
        generator=None,
        num_output_sampling=2,
    )

    prediction = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )
    target = torch.tensor(
        [
            [2.0, 4.0],
            [6.0, 8.0],
        ]
    )

    predictor._update_train_stats(
        make_output(output=prediction),
        DummyBatch(target=target),
    )

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    torch.testing.assert_close(
        predictor.stats["residual"].values[0],
        expected,
    )


def test_get_latent_samples_builds_sampler_once(
    tmp_path,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        num_latent_samples=3,
        infer_latent_samples_from_training=True,
    )

    sampler = Mock(
        return_value=torch.ones(3, 2, 4),
    )
    predictor.build_latent_sampler = Mock(
        return_value=sampler,
    )

    batch = DummyBatch(
        input=torch.ones(2, 5),
    )

    first = predictor._get_latent_samples_based_on_train(
        batch,
    )
    second = predictor._get_latent_samples_based_on_train(
        batch,
    )

    predictor.build_latent_sampler.assert_called_once()
    assert sampler.call_count == 2

    sampler.assert_called_with(
        (3, 2),
        predictor.nstds,
    )

    torch.testing.assert_close(
        first,
        torch.ones(3, 2, 4),
    )
    torch.testing.assert_close(
        second,
        torch.ones(3, 2, 4),
    )


def test_build_latent_sampler_requires_stats_file(
    tmp_path,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
    )

    with pytest.raises(
        ValueError,
        match="Training statistics.*must be saved",
    ):
        predictor.build_latent_sampler()


@pytest.mark.parametrize(
    "stats",
    [
        {},
        {"samples_mean": torch.zeros(2)},
        {"samples_cov": torch.eye(2)},
        {
            "samples_mean": None,
            "samples_cov": torch.eye(2),
        },
        {
            "samples_mean": torch.zeros(2),
            "samples_cov": None,
        },
    ],
)
def test_build_latent_sampler_rejects_incomplete_stats(
    tmp_path,
    stats,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
    )

    torch.save(
        stats,
        tmp_path / "training_variable_stats.pt",
    )

    with pytest.raises(
        ValueError,
        match="not for a cVAE model",
    ):
        predictor.build_latent_sampler()


def test_build_latent_sampler_uses_loaded_statistics(
    tmp_path,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
    )

    mean = torch.tensor([1.0, 2.0])
    covariance = torch.eye(2)

    torch.save(
        {
            "samples_mean": mean,
            "samples_cov": covariance,
        },
        tmp_path / "training_variable_stats.pt",
    )

    expected = torch.ones(3, 2)
    predictor._sample = Mock(
        return_value=expected,
    )

    sampler = predictor.build_latent_sampler()
    result = sampler(
        sample_size=(3,),
        std=2.0,
    )

    args = predictor._sample.call_args.args
    torch.testing.assert_close(args[0], mean)
    torch.testing.assert_close(args[1], covariance)
    assert args[2] == (3,)
    assert args[3] == 2.0
    assert result is expected


@pytest.mark.pruned
def test_infer_on_batch_requires_target_for_stats(
    tmp_path,
    monkeypatch,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
        infer_latent_samples_from_training=True,
    )

    monkeypatch.setattr(
        module,
        "clear_memory",
        Mock(),
    )

    with pytest.raises(
        RuntimeError,
        match="dataset must contain the target",
    ):
        predictor._infer_on_batch(
            DummyBatch(target=None),
            _getting_train_stats=True,
        )

    assert model.eval_called is True


def test_infer_on_batch_returns_training_stats(
    tmp_path,
    monkeypatch,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
    )

    monkeypatch.setattr(
        module,
        "clear_memory",
        Mock(),
    )

    forward_output = make_output()
    model.forward_output = forward_output

    expected = {"samples": object()}
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
    model.forward_kwargs == {
        "data": batch,
        "sample_size": 1,
    }
    predictor._update_train_stats.assert_called_once_with(
        forward_output,
        batch,
    )


@pytest.mark.pruned
def test_infer_on_batch_predicts_and_saves(
    tmp_path,
    monkeypatch,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
        num_output_sampling=2,
        num_latent_samples=4,
    )

    monkeypatch.setattr(
        module,
        "clear_memory",
        Mock(),
    )

    output = make_output(
        output=torch.zeros(4, 2, 3),
    )
    model.predict_output = output

    predictor._batch_to_netcdf = Mock()

    batch = DummyBatch()

    result = predictor._infer_on_batch(batch)

    assert result is output
    assert model.eval_called is True

    assert model.predict_kwargs == {
        "data": batch,
        "sample_size": 4,
        "nstds": predictor.nstds,
        "latent_samples": None,
        "output_sample_size": 2,
    }

    predictor._batch_to_netcdf.assert_called_once_with(
        output,
        batch.metadata,
    )


def test_infer_on_batch_uses_training_latent_samples(
    tmp_path,
    monkeypatch,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
        infer_latent_samples_from_training=True,
    )

    monkeypatch.setattr(
        module,
        "clear_memory",
        Mock(),
    )

    latent_samples = torch.ones(2, 2, 3)
    predictor._get_latent_samples_based_on_train = Mock(
        return_value=latent_samples,
    )
    predictor._batch_to_netcdf = Mock()

    output = make_output()
    model.predict_output = output

    batch = DummyBatch(
        target=torch.ones(2, 3),
    )

    predictor._infer_on_batch(batch)

    predictor._get_latent_samples_based_on_train.assert_called_once_with(
        data=batch,
    )
    assert model.predict_kwargs["latent_samples"] is latent_samples


def test_infer_on_batch_adds_decoder_noise(
    tmp_path,
    monkeypatch,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=None,
        num_output_sampling=2,
    )

    monkeypatch.setattr(
        module,
        "clear_memory",
        Mock(),
    )

    original = make_output(
        output=torch.zeros(3, 2, 4, 5),
    )
    noisy = make_output(
        output=torch.ones(3, 2, 4, 5),
    )

    model.predict_output = original
    predictor.add_decoder_noise = Mock(
        return_value=noisy,
    )
    predictor._batch_to_netcdf = Mock()

    result = predictor._infer_on_batch(
        DummyBatch(),
    )

    assert result is noisy
    predictor.add_decoder_noise.assert_called_once_with(
        original,
        2,
        torch.Size([3, 2]),
        torch.Size([4, 5]),
    )


def test_batch_to_netcdf_prediction(
    tmp_path,
    monkeypatch,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
        num_output_sampling=2,
    )

    predictor.output_dir = tmp_path

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
    assert args[5] == ["output_samples", "latent_samples"]
    assert args[6] is None
    assert args[7] is None

    assert predictor._batch_counter == 1


def test_batch_to_netcdf_latent_requires_variables(
    tmp_path,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        save_latent=True,
    )

    predictor.output_dir = tmp_path
    predictor.temp_save_dir.mkdir(exist_ok=True)

    with pytest.raises(
        RuntimeError,
        match="No latent variables are available",
    ):
        predictor._batch_to_netcdf(
            make_output(
                mu=None,
                log_var=None,
                samples=None,
                cond_mu=None,
                cond_log_var=None,
            ),
            [{"year": 2000}],
        )


@pytest.mark.pruned
def test_batch_to_netcdf_latent_pads_variables(
    tmp_path,
    monkeypatch,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        save_latent=True,
        infer_latent_samples_from_training=True,
    )

    predictor.output_dir = tmp_path

    save_mock = Mock()
    monkeypatch.setattr(
        module,
        "save_batch_to_netcdf",
        save_mock,
    )

    output = make_output(
        mu=torch.ones(2, 2),
        log_var=torch.ones(2, 3),
        samples=torch.ones(2, 4),
        cond_mu=None,
        cond_log_var=None,
    )

    predictor._batch_to_netcdf(
        output,
        [{"year": 2000}],
    )

    args = save_mock.call_args.args
    prediction = args[0]

    assert prediction.shape[-2:] == (3, 4)
    assert args[2] == 1
    assert args[3] == "latent_rank0_00000000.nc"
    assert args[5] == []
    assert args[6] == {
        "channels": [
            "mu",
            "log_var",
            "samples",
        ]
    }
    assert args[7] == {"infer_latent_samples_from_training": True}
    assert predictor._batch_counter == 1


@pytest.mark.pruned
def test_extract_training_vars_false_when_no_stats_requested(
    tmp_path,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        infer_latent_samples_from_training=False,
    )

    assert predictor.extract_posterior_samples is False
    assert predictor.extract_training_residuals is False
    assert predictor.extract_training_vars is False
    assert predictor.stats is None


@pytest.mark.pruned
def test_extract_training_vars_contains_both_statistics(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "RunningCovariance",
        DummyRunningCovariance,
    )

    predictor, _ = make_predictor(
        tmp_path,
        generator=None,
        num_output_sampling=2,
        infer_latent_samples_from_training=True,
    )

    assert predictor.extract_training_vars is True
    assert predictor.extract_posterior_samples is True
    assert predictor.extract_training_residuals is True
    assert set(predictor.stats) == {
        "samples",
        "residual",
    }


def test_update_train_stats_updates_samples_and_residuals(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "RunningCovariance",
        DummyRunningCovariance,
    )

    predictor, _ = make_predictor(
        tmp_path,
        generator=None,
        num_output_sampling=2,
        infer_latent_samples_from_training=True,
    )

    samples = torch.arange(
        24,
        dtype=torch.float32,
    ).reshape(2, 3, 4)

    prediction = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )
    target = torch.tensor(
        [
            [2.0, 4.0],
            [6.0, 8.0],
        ]
    )

    result = predictor._update_train_stats(
        make_output(
            output=prediction,
            samples=samples,
        ),
        DummyBatch(target=target),
    )

    assert result is predictor.stats
    assert len(result["samples"].values) == 1
    assert len(result["residual"].values) == 1

    torch.testing.assert_close(
        result["samples"].values[0],
        samples.reshape(-1, 4),
    )
    torch.testing.assert_close(
        result["residual"].values[0],
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
    )


def test_update_train_stats_returns_empty_stats_when_disabled(
    tmp_path,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        infer_latent_samples_from_training=False,
    )

    predictor._stats = {}

    result = predictor._update_train_stats(
        make_output(),
        DummyBatch(),
    )

    assert result is None


@pytest.mark.pruned
def test_get_latent_samples_uses_existing_sampler(
    tmp_path,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        num_latent_samples=4,
        infer_latent_samples_from_training=True,
    )

    existing_sampler = Mock(
        return_value=torch.ones(4, 3, 5),
    )
    predictor.latent_sampler = existing_sampler
    predictor.build_latent_sampler = Mock()

    batch = DummyBatch(
        input=torch.ones(3, 7),
    )

    result = predictor._get_latent_samples_based_on_train(
        batch,
    )

    predictor.build_latent_sampler.assert_not_called()
    existing_sampler.assert_called_once_with(
        (4, 3),
        predictor.nstds,
    )

    assert result.device == predictor.device
    assert result.shape == (4, 3, 5)


@pytest.mark.pruned
def test_infer_on_batch_save_latent_requires_target(
    tmp_path,
    monkeypatch,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
        save_latent=True,
    )

    clear_mock = Mock()
    monkeypatch.setattr(
        module,
        "clear_memory",
        clear_mock,
    )

    with pytest.raises(
        RuntimeError,
        match="dataset must contain the target",
    ):
        predictor._infer_on_batch(
            DummyBatch(target=None),
        )

    clear_mock.assert_called_once()
    assert model.eval_called is True
    assert model.forward_kwargs is None
    assert model.predict_kwargs is None


@pytest.mark.pruned
def test_infer_on_batch_save_latent_forwards_and_returns(
    tmp_path,
    monkeypatch,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
        save_latent=True,
    )

    monkeypatch.setattr(
        module,
        "clear_memory",
        Mock(),
    )

    output = make_output(
        mu=torch.ones(2, 3),
        log_var=torch.zeros(2, 3),
        samples=torch.ones(2, 3),
    )
    model.forward_output = output

    predictor._batch_to_netcdf = Mock()

    batch = DummyBatch(
        target=torch.ones(2, 3),
    )

    result = predictor._infer_on_batch(batch)

    assert result is output
    assert model.forward_kwargs == {
        "data": batch,
        "sample_size": 1,
    }
    assert model.predict_kwargs is None

    predictor._batch_to_netcdf.assert_called_once_with(
        output,
        batch.metadata,
    )


@pytest.mark.pruned
def test_infer_on_batch_training_stats_takes_precedence_over_save_latent(
    tmp_path,
    monkeypatch,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
        save_latent=True,
        infer_latent_samples_from_training=True,
    )

    monkeypatch.setattr(
        module,
        "clear_memory",
        Mock(),
    )

    output = make_output(
        samples=torch.ones(1, 2, 3),
    )
    model.forward_output = output

    expected_stats = {
        "samples": object(),
    }
    predictor._update_train_stats = Mock(
        return_value=expected_stats,
    )
    predictor._batch_to_netcdf = Mock()

    batch = DummyBatch(
        target=torch.ones(2, 3),
    )

    result = predictor._infer_on_batch(
        batch,
        _getting_train_stats=True,
    )

    assert result is expected_stats
    predictor._update_train_stats.assert_called_once_with(
        output,
        batch,
    )
    predictor._batch_to_netcdf.assert_not_called()
    assert model.predict_kwargs is None


@pytest.mark.pruned
def test_infer_on_batch_calls_clear_memory(
    tmp_path,
    monkeypatch,
):
    predictor, model = make_predictor(
        tmp_path,
        generator=True,
    )

    clear_mock = Mock()
    monkeypatch.setattr(
        module,
        "clear_memory",
        clear_mock,
    )

    model.predict_output = make_output()
    predictor._batch_to_netcdf = Mock()

    predictor._infer_on_batch(
        DummyBatch(),
    )

    clear_mock.assert_called_once()


@pytest.mark.pruned
def test_batch_to_netcdf_latent_without_training_sampler_has_no_attrs(
    tmp_path,
    monkeypatch,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        save_latent=True,
        infer_latent_samples_from_training=False,
    )

    predictor.temp_save_dir.mkdir(exist_ok=True)

    save_mock = Mock()
    monkeypatch.setattr(
        module,
        "save_batch_to_netcdf",
        save_mock,
    )

    output = make_output(
        mu=torch.ones(2, 3),
        log_var=torch.zeros(2, 3),
    )

    predictor._batch_to_netcdf(
        output,
        [{"year": 2000}, {"year": 2001}],
    )

    args = save_mock.call_args.args

    assert args[3] == "latent_rank0_00000000.nc"
    assert args[5] == []
    assert args[6] == {
        "channels": [
            "mu",
            "log_var",
        ]
    }
    assert args[7] is None


@pytest.mark.pruned
def test_batch_to_netcdf_latent_equal_sizes_need_no_padding(
    tmp_path,
    monkeypatch,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        save_latent=True,
    )

    predictor.temp_save_dir.mkdir(exist_ok=True)

    save_mock = Mock()
    monkeypatch.setattr(
        module,
        "save_batch_to_netcdf",
        save_mock,
    )

    output = make_output(
        mu=torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
        log_var=torch.tensor(
            [
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        ),
    )

    predictor._batch_to_netcdf(
        output,
        [{"year": 2000}, {"year": 2001}],
    )

    prediction = save_mock.call_args.args[0]

    assert prediction.shape[-2:] == (2, 3)
    assert torch.isneginf(prediction).sum() == 0


def test_batch_to_netcdf_latent_padding_uses_negative_infinity(
    tmp_path,
    monkeypatch,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        save_latent=True,
    )

    predictor.temp_save_dir.mkdir(exist_ok=True)

    save_mock = Mock()
    monkeypatch.setattr(
        module,
        "save_batch_to_netcdf",
        save_mock,
    )

    output = make_output(
        mu=torch.ones(2, 2),
        samples=torch.ones(2, 4),
    )

    predictor._batch_to_netcdf(
        output,
        [{"year": 2000}, {"year": 2001}],
    )

    prediction = save_mock.call_args.args[0]

    assert prediction.shape[-2:] == (2, 4)
    assert torch.isneginf(prediction[..., 0, 2:]).all()
    assert not torch.isneginf(prediction[..., 1, :]).any()


@pytest.mark.pruned
def test_batch_to_netcdf_latent_filters_none_variables(
    tmp_path,
    monkeypatch,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        save_latent=True,
    )

    predictor.temp_save_dir.mkdir(exist_ok=True)

    save_mock = Mock()
    monkeypatch.setattr(
        module,
        "save_batch_to_netcdf",
        save_mock,
    )

    output = make_output(
        mu=torch.ones(2, 3),
        log_var=None,
        samples=None,
        cond_mu=torch.ones(2, 3),
        cond_log_var=None,
    )

    predictor._batch_to_netcdf(
        output,
        [{"year": 2000}, {"year": 2001}],
    )

    args = save_mock.call_args.args

    assert args[6] == {
        "channels": [
            "mu",
            "cond_mu",
        ]
    }
    assert args[0].shape[-2] == 2


@pytest.mark.pruned
def test_batch_to_netcdf_latent_uses_distributed_rank(
    tmp_path,
    monkeypatch,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        save_latent=True,
    )

    predictor.distributed.rank = 6
    predictor._batch_counter = 12
    predictor.temp_save_dir.mkdir(exist_ok=True)

    save_mock = Mock()
    monkeypatch.setattr(
        module,
        "save_batch_to_netcdf",
        save_mock,
    )

    predictor._batch_to_netcdf(
        make_output(
            mu=torch.ones(2, 3),
        ),
        [{"year": 2000}, {"year": 2001}],
    )

    assert save_mock.call_args.args[3] == "latent_rank6_00000012.nc"
    assert predictor._batch_counter == 13


@pytest.mark.pruned
def test_batch_to_netcdf_prediction_detaches_tensor(
    tmp_path,
    monkeypatch,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        num_output_sampling=2,
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
        4,
        requires_grad=True,
    )

    predictor._batch_to_netcdf(
        make_output(output=prediction),
        [{"year": 2000}],
    )

    saved = save_mock.call_args.args[0]

    assert saved.device.type == "cpu"
    assert saved.requires_grad is False
    torch.testing.assert_close(
        saved,
        prediction.detach().cpu(),
    )


@pytest.mark.pruned
def test_batch_to_netcdf_prediction_uses_rank_and_counter(
    tmp_path,
    monkeypatch,
):
    predictor, _ = make_predictor(
        tmp_path,
        generator=True,
        num_output_sampling=2,
    )

    predictor.distributed.rank = 3
    predictor._batch_counter = 9
    predictor.temp_save_dir.mkdir(exist_ok=True)

    save_mock = Mock()
    monkeypatch.setattr(
        module,
        "save_batch_to_netcdf",
        save_mock,
    )

    predictor._batch_to_netcdf(
        make_output(
            output=torch.ones(2, 1, 3),
        ),
        [{"year": 2000}],
    )

    assert save_mock.call_args.args[3] == "prediction_rank3_00000009.nc"
    assert predictor._batch_counter == 10
from types import SimpleNamespace

import pytest
import torch
import warnings

from cccma_ppp.inference.predictors import (
    cVAEPredictorConfig,
    cVAEPredictor,
    DeterministicPredictorConfig,
    DetermninisticPredictor,
)


class DummyDistributed:
    def __init__(
        self,
        rank=0,
        device="cpu",
    ):
        self.rank = rank
        self.device = torch.device(device)


class DummyModelConfig:
    NUM_OUTPUT_DIMS = 1


class DummyModule:
    def __init__(self):
        self.eval_called = False
        self.forward_called = False
        self.predict_called = False
        self.forward_kwargs = None
        self.predict_kwargs = None
        self.model_config = DummyModelConfig()
        self.config = DummyModelConfig()
        self.forward_output = None
        self.predict_output = None

    def eval(self):
        self.eval_called = True
        return self

    def forward(self, **kwargs):
        self.forward_called = True
        self.forward_kwargs = kwargs
        return self.forward_output

    def predict(self, **kwargs):
        self.predict_called = True
        self.predict_kwargs = kwargs
        return self.predict_output


class DummyBatch:
    def __init__(
        self,
        input=None,
        target=None,
        input_mask=None,
        target_mask=None,
        added_features=None,
        metadata=None,
    ):
        self.input = torch.ones(2, 3) if input is None else input
        self.target = target
        self.input_mask = input_mask
        self.target_mask = target_mask
        self.added_features = added_features
        self.metadata = (
            [
                {"year": 2000},
                {"year": 2001},
            ]
            if metadata is None
            else metadata
        )


class DummyRunningCovariance:
    def __init__(self):
        self.updates = []

    def update(self, value):
        self.updates.append(value.clone())


def make_cvae_output(
    output=None,
    samples=None,
    mu=None,
    log_var=None,
    cond_mu=None,
    cond_log_var=None,
):
    return SimpleNamespace(
        output=(torch.ones(1, 2, 1, 3) if output is None else output),
        samples=samples,
        mu=mu,
        log_var=log_var,
        cond_mu=cond_mu,
        cond_log_var=cond_log_var,
    )


def make_deterministic_output(output=None):
    return SimpleNamespace(output=(torch.ones(2, 1, 3) if output is None else output))


def make_cvae_predictor(
    tmp_path,
    num_latent_samples=1,
    nstds=1.0,
    infer_latent_samples_from_training=False,
    save_latent=False,
    covariance_samples=0,
):
    config = cVAEPredictorConfig(
        num_latent_samples=num_latent_samples,
        nstds=nstds,
        infer_latent_samples_from_training=(infer_latent_samples_from_training),
        save_latent=save_latent,
    )
    module = DummyModule()
    distributed = DummyDistributed()

    predictor = cVAEPredictor(
        config=config,
        module=module,
        distributed=distributed,
        output_dir=tmp_path,
        num_output_covariance_sampling=covariance_samples,
    )

    return predictor, module


def make_deterministic_predictor(
    tmp_path,
    covariance_samples=0,
):
    config = DeterministicPredictorConfig()
    module = DummyModule()
    distributed = DummyDistributed()

    predictor = DetermninisticPredictor(
        config=config,
        module=module,
        distributed=distributed,
        output_dir=tmp_path,
        num_output_covariance_sampling=covariance_samples,
    )

    return predictor, module


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -10,
    ],
)
def test_cvae_config_rejects_invalid_latent_samples(value):
    with pytest.raises(
        ValueError,
        match="num_latent_samples must be at least 1",
    ):
        cVAEPredictorConfig(
            num_latent_samples=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -1.0,
        -10.0,
    ],
)
def test_cvae_config_rejects_invalid_nstds(value):
    with pytest.raises(
        ValueError,
        match="nstds must be positive",
    ):
        cVAEPredictorConfig(
            nstds=value,
        )


@pytest.mark.pruned
def test_cvae_config_defaults():
    config = cVAEPredictorConfig()

    assert config.num_latent_samples == 1
    assert config.nstds == 1.0
    assert config.infer_latent_samples_from_training is False
    assert config.save_latent is False


@pytest.mark.pruned
def test_cvae_config_save_latent_warns():
    with pytest.warns(
        UserWarning,
        match="No predictions will be saved",
    ):
        config = cVAEPredictorConfig(
            save_latent=True,
        )

    assert config.save_latent is True


@pytest.mark.pruned
def test_cvae_config_without_save_latent_no_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        cVAEPredictorConfig(
            save_latent=False,
        )

    assert captured == []


@pytest.mark.pruned
def test_cvae_config_build(tmp_path):
    config = cVAEPredictorConfig(
        num_latent_samples=3,
        nstds=2.0,
    )
    module = DummyModule()
    distributed = DummyDistributed(rank=2)

    predictor = config.build(
        module=module,
        distributed=distributed,
        output_dir=tmp_path,
        num_output_covariance_sampling=4,
    )

    assert isinstance(predictor, cVAEPredictor)
    assert predictor.module is module
    assert predictor.distributed is distributed
    assert predictor.output_dir == tmp_path
    assert predictor.num_latent_samples == 3
    assert predictor.nstds == 2.0
    assert predictor.num_output_covariance_sampling == 4


@pytest.mark.pruned
def test_cvae_predictor_initial_state(tmp_path):
    predictor, module = make_cvae_predictor(tmp_path)

    assert predictor.module is module
    assert predictor.device.type == "cpu"
    assert predictor.output_sampler is None
    assert predictor.latent_sampler is None
    assert predictor._batch_counter == 0
    assert predictor.extract_training_vars is False


@pytest.mark.parametrize(
    "infer_latent,covariance_samples,expected",
    [
        (False, 0, False),
        (True, 0, True),
        (False, 1, True),
        (True, 1, True),
    ],
)
def test_cvae_extract_training_vars_matrix(
    tmp_path,
    infer_latent,
    covariance_samples,
    expected,
):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        infer_latent_samples_from_training=infer_latent,
        covariance_samples=covariance_samples,
    )

    assert predictor.extract_training_vars is expected


@pytest.mark.pruned
def test_cvae_initializes_training_stats_for_latent_sampling(tmp_path):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        infer_latent_samples_from_training=True,
    )

    assert set(predictor.stats) == {
        "samples",
        "residual",
    }


@pytest.mark.pruned
def test_cvae_initializes_training_stats_for_output_noise(tmp_path):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        covariance_samples=2,
    )

    assert set(predictor.stats) == {
        "samples",
        "residual",
    }


def test_cvae_infer_training_stats_requires_target(
    monkeypatch,
    tmp_path,
):
    predictor, module = make_cvae_predictor(
        tmp_path,
        covariance_samples=1,
    )
    batch = DummyBatch(target=None)

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.clear_memory",
        lambda: None,
    )

    with pytest.raises(
        RuntimeError,
        match="dataset must contain the target",
    ):
        predictor._infer_on_batch(
            batch,
            _getting_train_stats=True,
        )

    assert module.eval_called


@pytest.mark.pruned
def test_cvae_save_latent_requires_target(
    monkeypatch,
    tmp_path,
):
    predictor, module = make_cvae_predictor(
        tmp_path,
        save_latent=True,
    )
    batch = DummyBatch(target=None)

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.clear_memory",
        lambda: None,
    )

    with pytest.raises(
        RuntimeError,
        match="dataset must contain the target",
    ):
        predictor._infer_on_batch(batch)


def test_cvae_infer_training_stats_branch(
    monkeypatch,
    tmp_path,
):
    predictor, module = make_cvae_predictor(
        tmp_path,
        infer_latent_samples_from_training=True,
    )
    batch = DummyBatch(
        target=torch.ones(2, 1, 3),
    )
    output = make_cvae_output(
        output=torch.zeros(1, 2, 1, 3),
        samples=torch.ones(1, 2, 4),
    )
    module.forward_output = output
    expected = {"stats": True}

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.clear_memory",
        lambda: None,
    )
    monkeypatch.setattr(
        predictor,
        "_update_train_stats",
        lambda output_value, batch_value: expected,
    )

    result = predictor._infer_on_batch(
        batch,
        _getting_train_stats=True,
    )

    assert result is expected
    assert module.forward_called
    assert module.forward_kwargs["data"] is batch
    assert module.forward_kwargs["sample_size"] == 1
    assert module.predict_called is False


def test_cvae_save_latent_branch(
    monkeypatch,
    tmp_path,
):
    predictor, module = make_cvae_predictor(
        tmp_path,
        save_latent=True,
    )
    batch = DummyBatch(
        target=torch.ones(2, 1, 3),
    )
    output = make_cvae_output(
        samples=torch.ones(1, 2, 4),
    )
    module.forward_output = output
    calls = []

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.clear_memory",
        lambda: None,
    )
    monkeypatch.setattr(
        predictor,
        "_batch_to_netcdf",
        lambda output_value, metadata: calls.append((output_value, metadata)),
    )

    result = predictor._infer_on_batch(batch)

    assert result is output
    assert module.forward_called
    assert module.predict_called is False
    assert calls == [(output, batch.metadata)]


@pytest.mark.pruned
def test_cvae_standard_prediction_branch(
    monkeypatch,
    tmp_path,
):
    predictor, module = make_cvae_predictor(
        tmp_path,
        num_latent_samples=4,
        nstds=1.5,
    )
    batch = DummyBatch()
    output = make_cvae_output()
    module.predict_output = output
    calls = []

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.clear_memory",
        lambda: None,
    )
    monkeypatch.setattr(
        predictor,
        "_batch_to_netcdf",
        lambda output_value, metadata: calls.append((output_value, metadata)),
    )

    result = predictor._infer_on_batch(batch)

    assert result is output
    assert module.predict_called
    assert module.predict_kwargs["data"] is batch
    assert module.predict_kwargs["sample_size"] == 4
    assert module.predict_kwargs["nstds"] == 1.5
    assert module.predict_kwargs["latent_samples"] is None
    assert calls == [(output, batch.metadata)]


def test_cvae_prediction_uses_training_latent_samples(
    monkeypatch,
    tmp_path,
):
    predictor, module = make_cvae_predictor(
        tmp_path,
        infer_latent_samples_from_training=True,
    )
    batch = DummyBatch()
    output = make_cvae_output()
    module.predict_output = output
    latent = torch.ones(1, 2, 4)

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.clear_memory",
        lambda: None,
    )
    monkeypatch.setattr(
        predictor,
        "_get_latent_samples_based_on_train",
        lambda data: latent,
    )
    monkeypatch.setattr(
        predictor,
        "_batch_to_netcdf",
        lambda *args: None,
    )

    predictor._infer_on_batch(batch)

    assert module.predict_kwargs["latent_samples"] is latent


def test_cvae_prediction_adds_decoder_noise(
    monkeypatch,
    tmp_path,
):
    predictor, module = make_cvae_predictor(
        tmp_path,
        covariance_samples=3,
    )
    batch = DummyBatch()
    output = make_cvae_output(
        output=torch.ones(4, 2, 1, 3),
    )
    noisy_output = make_cvae_output(
        output=torch.ones(3, 4, 2, 1, 3),
    )
    module.predict_output = output
    noise_calls = []
    save_calls = []

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.clear_memory",
        lambda: None,
    )
    monkeypatch.setattr(
        predictor,
        "add_decoder_noise",
        lambda value, count, sample_size, reshape_size: (
            noise_calls.append(
                (
                    value,
                    count,
                    sample_size,
                    reshape_size,
                )
            )
            or noisy_output
        ),
    )
    monkeypatch.setattr(
        predictor,
        "_batch_to_netcdf",
        lambda value, metadata: save_calls.append((value, metadata)),
    )

    result = predictor._infer_on_batch(batch)

    assert result is noisy_output
    assert noise_calls == [
        (
            output,
            3,
            torch.Size([4, 2]),
            torch.Size([1, 3]),
        )
    ]
    assert save_calls == [
        (
            noisy_output,
            batch.metadata,
        )
    ]


@pytest.mark.pruned
def test_cvae_update_stats_requires_samples(tmp_path):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        covariance_samples=1,
    )
    output = make_cvae_output(samples=None)
    batch = DummyBatch(
        target=torch.ones(2, 1, 3),
    )

    with pytest.raises(
        RuntimeError,
        match="samples is required",
    ):
        predictor._update_train_stats(
            output,
            batch,
        )


def test_cvae_update_train_stats(
    tmp_path,
):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        covariance_samples=1,
    )
    sample_stats = DummyRunningCovariance()
    residual_stats = DummyRunningCovariance()
    predictor._stats = {
        "samples": sample_stats,
        "residual": residual_stats,
    }

    output = make_cvae_output(
        output=torch.zeros(1, 2, 1, 3),
        samples=torch.arange(
            1 * 2 * 4,
            dtype=torch.float32,
        ).reshape(1, 2, 4),
    )
    batch = DummyBatch(
        target=torch.ones(2, 1, 3),
    )

    result = predictor._update_train_stats(
        output,
        batch,
    )

    assert result is predictor.stats
    assert sample_stats.updates[0].shape == (2, 4)
    assert residual_stats.updates[0].shape == (2, 3)
    assert torch.equal(
        residual_stats.updates[0],
        torch.ones(2, 3),
    )


@pytest.mark.pruned
def test_cvae_update_train_stats_with_target_mask(
    tmp_path,
):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        covariance_samples=1,
    )
    predictor._stats = {
        "samples": DummyRunningCovariance(),
        "residual": DummyRunningCovariance(),
    }

    output = make_cvae_output(
        output=torch.zeros(1, 2, 1, 3),
        samples=torch.ones(1, 2, 4),
    )
    target = torch.ones(2, 1, 3)

    batch = DummyBatch(
        target=target,
        target_mask=torch.ones_like(
            target,
            dtype=torch.bool,
        ),
    )

    result = predictor._update_train_stats(
        output,
        batch,
    )

    residual = predictor.stats["residual"].updates[0]

    assert result is predictor.stats
    assert residual.shape == (2, 3)

    torch.testing.assert_close(
        residual,
        torch.ones(2, 3),
    )


@pytest.mark.pruned
def test_get_latent_samples_builds_sampler(
    monkeypatch,
    tmp_path,
):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        num_latent_samples=3,
        nstds=2.0,
        infer_latent_samples_from_training=True,
    )
    calls = []

    def sampler(sample_size, std):
        calls.append(
            (
                sample_size,
                std,
            )
        )
        return torch.ones(
            *sample_size,
            4,
        )

    monkeypatch.setattr(
        predictor,
        "build_latent_sampler",
        lambda: sampler,
    )

    batch = DummyBatch(
        input=torch.ones(2, 5),
    )

    result = predictor._get_latent_samples_based_on_train(batch)

    assert calls == [
        (
            (3, 2),
            2.0,
        )
    ]
    assert predictor.latent_sampler is sampler
    assert result.shape == (3, 2, 4)
    assert result.device.type == "cpu"


def test_get_latent_samples_reuses_sampler(tmp_path):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        num_latent_samples=2,
        infer_latent_samples_from_training=True,
    )
    calls = []

    def sampler(sample_size, std):
        calls.append(
            (
                sample_size,
                std,
            )
        )
        return torch.zeros(
            *sample_size,
            3,
        )

    predictor.latent_sampler = sampler
    batch = DummyBatch(
        input=torch.ones(4, 5),
    )

    result = predictor._get_latent_samples_based_on_train(batch)

    assert calls == [
        (
            (2, 4),
            1.0,
        )
    ]
    assert result.shape == (2, 4, 3)


@pytest.mark.pruned
def test_get_latent_samples_with_input_mask(
    tmp_path,
):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        num_latent_samples=2,
        infer_latent_samples_from_training=True,
    )

    predictor.latent_sampler = lambda sample_size, std: torch.zeros(
        *sample_size,
        3,
    )

    input_tensor = torch.ones(5, 3)

    batch = DummyBatch(
        input=input_tensor,
        input_mask=torch.ones_like(
            input_tensor,
            dtype=torch.bool,
        ),
    )

    result = predictor._get_latent_samples_based_on_train(batch)

    assert result.shape == (2, 5, 3)
    assert result.device.type == "cpu"


def test_build_latent_sampler_missing_file(tmp_path):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        infer_latent_samples_from_training=True,
    )

    with pytest.raises(
        ValueError,
        match="Training statistics",
    ):
        predictor.build_latent_sampler()


@pytest.mark.parametrize(
    "stats",
    [
        {},
        {
            "samples_mean": torch.zeros(2),
        },
        {
            "samples_cov": torch.eye(2),
        },
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
def test_build_latent_sampler_invalid_stats(
    monkeypatch,
    tmp_path,
    stats,
):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        infer_latent_samples_from_training=True,
    )
    stats_path = tmp_path / "training_variable_stats.pt"
    stats_path.touch()

    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: stats,
    )

    with pytest.raises(
        ValueError,
        match="not for a cVAE model",
    ):
        predictor.build_latent_sampler()


@pytest.mark.pruned
def test_build_latent_sampler_valid_stats(
    monkeypatch,
    tmp_path,
):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        infer_latent_samples_from_training=True,
    )
    stats_path = tmp_path / "training_variable_stats.pt"
    stats_path.touch()
    stats = {
        "samples_mean": torch.tensor([1.0, 2.0]),
        "samples_cov": torch.eye(2),
    }
    captured = {}

    def fake_load(
        path,
        map_location,
        weights_only,
    ):
        captured["path"] = path
        captured["map_location"] = map_location
        captured["weights_only"] = weights_only
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
        std,
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

    sampler = predictor.build_latent_sampler()
    result = sampler(
        (3, 4),
        2.5,
    )

    assert captured["path"] == stats_path
    assert captured["map_location"] == "cpu"
    assert captured["weights_only"] is True
    assert captured["mu"] is stats["samples_mean"]
    assert captured["cov"] is stats["samples_cov"]
    assert captured["sample_size"] == (3, 4)
    assert captured["std"] == 2.5
    assert result.shape == (3, 4, 2)


def test_cvae_batch_to_netcdf_predictions_no_noise(
    monkeypatch,
    tmp_path,
):
    predictor, module = make_cvae_predictor(
        tmp_path,
        covariance_samples=0,
    )
    module.model_config.NUM_OUTPUT_DIMS = 1
    output = make_cvae_output(
        output=torch.ones(2, 2, 1, 3),
    )
    metadata = [
        {"year": 2000},
        {"year": 2001},
    ]
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.save_batch_to_netcdf",
        lambda *args: captured.update(
            {
                "args": args,
            }
        ),
    )

    predictor._batch_to_netcdf(
        output,
        metadata,
    )

    args = captured["args"]

    assert args[0].shape == (1, 2, 2, 1, 3)
    assert args[1] is metadata
    assert args[2] == 1
    assert args[3] == "prediction_rank0_00000000.nc"
    assert args[4] == tmp_path / "_temp"
    assert args[5] == [
        "output_samples",
        "latent_samples",
    ]
    assert args[6] is None
    assert predictor._batch_counter == 1


def test_cvae_batch_to_netcdf_predictions_with_noise(
    monkeypatch,
    tmp_path,
):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        covariance_samples=3,
    )
    output = make_cvae_output(
        output=torch.ones(3, 2, 2, 1, 3),
    )
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.save_batch_to_netcdf",
        lambda *args: captured.update(
            {
                "prediction": args[0],
                "name": args[3],
            }
        ),
    )

    predictor._batch_to_netcdf(
        output,
        [
            {"year": 2000},
            {"year": 2001},
        ],
    )

    assert captured["prediction"].shape == (
        3,
        2,
        2,
        1,
        3,
    )
    assert captured["name"] == ("prediction_rank0_00000000.nc")


@pytest.mark.pruned
def test_cvae_batch_to_netcdf_prediction_counter(
    monkeypatch,
    tmp_path,
):
    predictor, _ = make_cvae_predictor(
        tmp_path,
    )
    output = make_cvae_output()
    names = []

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.save_batch_to_netcdf",
        lambda *args: names.append(args[3]),
    )

    predictor._batch_to_netcdf(
        output,
        [
            {"year": 2000},
            {"year": 2001},
        ],
    )
    predictor._batch_to_netcdf(
        output,
        [
            {"year": 2000},
            {"year": 2001},
        ],
    )

    assert names == [
        "prediction_rank0_00000000.nc",
        "prediction_rank0_00000001.nc",
    ]
    assert predictor._batch_counter == 2


@pytest.mark.pruned
def test_cvae_batch_to_netcdf_no_latent_variables(
    tmp_path,
):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        save_latent=True,
    )
    output = make_cvae_output(
        mu=None,
        log_var=None,
        samples=None,
        cond_mu=None,
        cond_log_var=None,
    )

    with pytest.raises(
        RuntimeError,
        match="No latent variables",
    ):
        predictor._batch_to_netcdf(
            output,
            [{"year": 2000}],
        )


@pytest.mark.pruned
def test_cvae_batch_to_netcdf_latent_single_variable(
    monkeypatch,
    tmp_path,
):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        save_latent=True,
    )
    output = make_cvae_output(
        mu=torch.ones(2, 4),
    )
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.save_batch_to_netcdf",
        lambda *args: captured.update(
            {
                "args": args,
            }
        ),
    )

    predictor._batch_to_netcdf(
        output,
        [
            {"year": 2000},
            {"year": 2001},
        ],
    )

    args = captured["args"]

    assert args[0].shape == (2, 1, 4)
    assert args[2] == 1
    assert args[3] == "latent_rank0_00000000.nc"
    assert args[5] == []
    assert args[6] == {
        "channels": ["mu"],
    }
    assert args[7] is None


def test_cvae_batch_to_netcdf_latent_multiple_variables_padding(
    monkeypatch,
    tmp_path,
):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        save_latent=True,
    )
    output = make_cvae_output(
        mu=torch.ones(2, 4),
        log_var=torch.ones(2, 2),
        samples=torch.ones(2, 3),
    )
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.save_batch_to_netcdf",
        lambda *args: captured.update(
            {
                "prediction": args[0],
                "coords": args[6],
            }
        ),
    )

    predictor._batch_to_netcdf(
        output,
        [
            {"year": 2000},
            {"year": 2001},
        ],
    )

    prediction = captured["prediction"]

    assert prediction.shape == (2, 3, 4)
    assert captured["coords"] == {
        "channels": [
            "mu",
            "log_var",
            "samples",
        ]
    }
    assert torch.isneginf(prediction[:, 1, 2:]).all()
    assert torch.isneginf(prediction[:, 2, 3:]).all()


@pytest.mark.pruned
def test_cvae_batch_to_netcdf_latent_training_sampler_attr(
    monkeypatch,
    tmp_path,
):
    predictor, _ = make_cvae_predictor(
        tmp_path,
        infer_latent_samples_from_training=True,
        save_latent=True,
    )
    output = make_cvae_output(
        mu=torch.ones(2, 4),
    )
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.save_batch_to_netcdf",
        lambda *args: captured.update(
            {
                "attrs": args[7],
            }
        ),
    )

    predictor._batch_to_netcdf(
        output,
        [
            {"year": 2000},
            {"year": 2001},
        ],
    )

    assert captured["attrs"] == {
        "infer_latent_samples_from_training": True,
    }


@pytest.mark.pruned
def test_cvae_latent_filename_uses_rank(
    monkeypatch,
    tmp_path,
):
    config = cVAEPredictorConfig(
        save_latent=True,
    )
    predictor = cVAEPredictor(
        config=config,
        module=DummyModule(),
        distributed=DummyDistributed(rank=7),
        output_dir=tmp_path,
    )
    output = make_cvae_output(
        mu=torch.ones(2, 4),
    )
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.save_batch_to_netcdf",
        lambda *args: captured.update(
            {
                "name": args[3],
            }
        ),
    )

    predictor._batch_to_netcdf(
        output,
        [
            {"year": 2000},
            {"year": 2001},
        ],
    )

    assert captured["name"] == ("latent_rank7_00000000.nc")


@pytest.mark.pruned
def test_deterministic_config_build(tmp_path):
    config = DeterministicPredictorConfig()
    module = DummyModule()
    distributed = DummyDistributed(rank=3)

    predictor = config.build(
        module=module,
        distributed=distributed,
        output_dir=tmp_path,
        num_output_covariance_sampling=4,
    )

    assert isinstance(
        predictor,
        DetermninisticPredictor,
    )
    assert predictor.module is module
    assert predictor.distributed is distributed
    assert predictor.output_dir == tmp_path
    assert predictor.num_output_covariance_sampling == 4


@pytest.mark.parametrize(
    "samples,expected",
    [
        (0, False),
        (1, True),
        (5, True),
    ],
)
def test_deterministic_extract_training_vars(
    tmp_path,
    samples,
    expected,
):
    predictor, _ = make_deterministic_predictor(
        tmp_path,
        covariance_samples=samples,
    )

    assert predictor.extract_training_vars is expected


@pytest.mark.pruned
def test_deterministic_initial_state_without_covariance(
    tmp_path,
):
    predictor, module = make_deterministic_predictor(
        tmp_path,
        covariance_samples=0,
    )

    assert predictor.module is module
    assert predictor.output_sampler is None
    assert predictor._batch_counter == 0
    assert predictor.extract_training_vars is False


@pytest.mark.pruned
def test_deterministic_initializes_residual_stats(
    tmp_path,
):
    predictor, _ = make_deterministic_predictor(
        tmp_path,
        covariance_samples=1,
    )

    assert set(predictor.stats) == {
        "residual",
    }


def test_deterministic_training_stats_requires_target(
    monkeypatch,
    tmp_path,
):
    predictor, module = make_deterministic_predictor(
        tmp_path,
        covariance_samples=1,
    )
    batch = DummyBatch(target=None)

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.clear_memory",
        lambda: None,
    )

    with pytest.raises(
        RuntimeError,
        match="dataset must contain the target",
    ):
        predictor._infer_on_batch(
            batch,
            _getting_train_stats=True,
        )

    assert module.eval_called


def test_deterministic_training_stats_branch(
    monkeypatch,
    tmp_path,
):
    predictor, module = make_deterministic_predictor(
        tmp_path,
        covariance_samples=1,
    )
    batch = DummyBatch(
        target=torch.ones(2, 1, 3),
    )
    output = make_deterministic_output(torch.zeros(2, 1, 3))
    module.forward_output = output
    expected = {"stats": True}

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.clear_memory",
        lambda: None,
    )
    monkeypatch.setattr(
        predictor,
        "_update_train_stats",
        lambda output_value, batch_value: expected,
    )

    result = predictor._infer_on_batch(
        batch,
        _getting_train_stats=True,
    )

    assert result is expected
    assert module.forward_called
    assert module.forward_kwargs["data"] is batch
    assert module.predict_called is False


def test_deterministic_prediction_branch(
    monkeypatch,
    tmp_path,
):
    predictor, module = make_deterministic_predictor(
        tmp_path,
        covariance_samples=0,
    )
    batch = DummyBatch()
    output = make_deterministic_output()
    module.predict_output = output
    save_calls = []

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.clear_memory",
        lambda: None,
    )
    monkeypatch.setattr(
        predictor,
        "_batch_to_netcdf",
        lambda value, metadata: save_calls.append((value, metadata)),
    )

    result = predictor._infer_on_batch(batch)

    assert result is output
    assert module.predict_called
    assert module.predict_kwargs["data"] is batch
    assert save_calls == [
        (
            output,
            batch.metadata,
        )
    ]


def test_deterministic_prediction_adds_decoder_noise(
    monkeypatch,
    tmp_path,
):
    predictor, module = make_deterministic_predictor(
        tmp_path,
        covariance_samples=3,
    )
    batch = DummyBatch()
    output = make_deterministic_output(torch.ones(2, 1, 3))
    noisy_output = make_deterministic_output(torch.ones(3, 2, 1, 3))
    module.predict_output = output
    noise_calls = []

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.clear_memory",
        lambda: None,
    )
    monkeypatch.setattr(
        predictor,
        "add_decoder_noise",
        lambda value, count, sample_size, reshape_size: (
            noise_calls.append(
                (
                    value,
                    count,
                    sample_size,
                    reshape_size,
                )
            )
            or noisy_output
        ),
    )
    monkeypatch.setattr(
        predictor,
        "_batch_to_netcdf",
        lambda *args: None,
    )

    result = predictor._infer_on_batch(batch)

    assert result is noisy_output
    assert noise_calls == [
        (
            output,
            3,
            (2,),
            torch.Size([1, 3]),
        )
    ]


@pytest.mark.pruned
def test_deterministic_update_train_stats(
    tmp_path,
):
    predictor, _ = make_deterministic_predictor(
        tmp_path,
        covariance_samples=1,
    )
    residual_stats = DummyRunningCovariance()
    predictor._stats = {
        "residual": residual_stats,
    }
    output = make_deterministic_output(torch.zeros(2, 1, 3))
    batch = DummyBatch(
        target=torch.ones(2, 1, 3),
    )

    result = predictor._update_train_stats(
        output,
        batch,
    )

    assert result is predictor.stats
    assert residual_stats.updates[0].shape == (
        2,
        3,
    )
    assert torch.equal(
        residual_stats.updates[0],
        torch.ones(2, 3),
    )


@pytest.mark.pruned
def test_deterministic_update_train_stats_with_target_mask(
    tmp_path,
):
    predictor, _ = make_deterministic_predictor(
        tmp_path,
        covariance_samples=1,
    )
    residual_stats = DummyRunningCovariance()
    predictor._stats = {
        "residual": residual_stats,
    }

    output = make_deterministic_output(torch.zeros(2, 1, 3))
    target = torch.ones(2, 1, 3)

    batch = DummyBatch(
        target=target,
        target_mask=torch.zeros_like(
            target,
            dtype=torch.bool,
        ),
    )

    result = predictor._update_train_stats(
        output,
        batch,
    )

    assert result is predictor.stats

    torch.testing.assert_close(
        residual_stats.updates[0],
        torch.ones(2, 3),
    )


def test_deterministic_batch_to_netcdf_without_noise(
    monkeypatch,
    tmp_path,
):
    predictor, module = make_deterministic_predictor(
        tmp_path,
        covariance_samples=0,
    )
    module.config.NUM_OUTPUT_DIMS = 1
    output = make_deterministic_output(torch.ones(2, 1, 3))
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.save_batch_to_netcdf",
        lambda *args: captured.update(
            {
                "args": args,
            }
        ),
    )

    predictor._batch_to_netcdf(
        output,
        [
            {"year": 2000},
            {"year": 2001},
        ],
    )

    args = captured["args"]

    assert args[0].shape == (1, 2, 1, 3)
    assert args[2] == 1
    assert args[3] == "prediction_rank0_00000000.nc"
    assert args[4] == tmp_path / "_temp"
    assert args[5] == [
        "output_samples",
        "latent_samples",
    ]
    assert args[6] is None
    assert predictor._batch_counter == 1


def test_deterministic_batch_to_netcdf_with_noise(
    monkeypatch,
    tmp_path,
):
    predictor, _ = make_deterministic_predictor(
        tmp_path,
        covariance_samples=3,
    )
    output = make_deterministic_output(torch.ones(3, 2, 1, 3))
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.save_batch_to_netcdf",
        lambda *args: captured.update(
            {
                "prediction": args[0],
            }
        ),
    )

    predictor._batch_to_netcdf(
        output,
        [
            {"year": 2000},
            {"year": 2001},
        ],
    )

    assert captured["prediction"].shape == (
        3,
        2,
        1,
        3,
    )


@pytest.mark.pruned
def test_deterministic_batch_to_netcdf_uses_rank_and_counter(
    monkeypatch,
    tmp_path,
):
    config = DeterministicPredictorConfig()
    predictor = DetermninisticPredictor(
        config=config,
        module=DummyModule(),
        distributed=DummyDistributed(rank=9),
        output_dir=tmp_path,
    )
    output = make_deterministic_output()
    names = []

    monkeypatch.setattr(
        "cccma_ppp.inference.predictors.save_batch_to_netcdf",
        lambda *args: names.append(args[3]),
    )

    predictor._batch_to_netcdf(
        output,
        [
            {"year": 2000},
            {"year": 2001},
        ],
    )
    predictor._batch_to_netcdf(
        output,
        [
            {"year": 2000},
            {"year": 2001},
        ],
    )

    assert names == [
        "prediction_rank9_00000000.nc",
        "prediction_rank9_00000001.nc",
    ]
    assert predictor._batch_counter == 2
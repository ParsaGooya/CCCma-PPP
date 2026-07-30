import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import xarray as xr

from cccma_ppp.core.writer import (
    Writer,
    WriterConfig,
    aggregate_predictions,
)


class DummyBatch:
    def to_device(self, device):
        self.device = device
        return self


class DummyLoader:
    def __iter__(self):
        yield DummyBatch()


class DummyModuleConfig:
    _type = "dummy"


class DummyModule:
    def __init__(self):
        self.config = DummyModuleConfig()

    def _get_device(self):
        return torch.device("cpu")


class DummyPredictor:
    def __init__(
        self,
        extract_training_vars=False,
        save_latent=False,
    ):
        self.extract_training_vars = extract_training_vars
        self.save_latent = save_latent
        self.stats = {}

        self.infer_calls = []

    def _infer_on_batch(
        self,
        batch,
        _getting_train_stats=False,
    ):
        self.infer_calls.append(_getting_train_stats)
        return self.stats


class DummyPredictorConfig:
    _type = "dummy"

    def __init__(self, predictor):
        self.predictor = predictor
        self.build_args = None

    def build(
        self,
        module,
        distributed,
        output_dir,
        num_output_sampling,
    ):
        self.build_args = {
            "module": module,
            "distributed": distributed,
            "output_dir": output_dir,
            "num_output_sampling": num_output_sampling,
        }

        return self.predictor


class DummyDistributed:
    def __init__(
        self,
        distributed=False,
        root=True,
    ):
        self.device = torch.device("cpu")
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1

        self.distributed = distributed
        self._root = root

        self.barrier_called = 0

    def is_root(self):
        return self._root

    def barrier(self):
        self.barrier_called += 1


class DummyTrainLoaderConfig:
    def __init__(self):
        self.setup_called = False
        self.train_called = False
        self.validation_called = False

    def setup_distributed(self, *args, **kwargs):
        self.setup_called = True

    def build_train_loader(self, **kwargs):
        self.train_called = True
        return "TRAIN"

    def build_validation_loader(self, **kwargs):
        self.validation_called = True
        return "VALIDATION"


class DummyLogger:
    def __init__(self):
        self.messages = []

    def log(self, level, msg, *args):
        self.messages.append((level, msg))


class DummyStat:
    def __init__(self, active=True):
        self.sum_x = torch.tensor([1.0]) if active else None
        self.reduced = False

    def distributed_reduce(self):
        self.reduced = True

    def finalize(self):
        return torch.tensor([1.0]), torch.tensor([[1.0]])


def test_writer_config_negative_sampling():
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        WriterConfig(
            predictor=object(),
            num_output_sampling=-1,
        )


def test_writer_config_build(tmp_path):
    predictor = DummyPredictor()

    cfg = WriterConfig(
        predictor=DummyPredictorConfig(predictor),
        num_output_sampling=0,
    )

    writer = cfg.build(
        inference_data_loader=DummyLoader(),
        train_dataloader_config=DummyTrainLoaderConfig(),
        module=DummyModule(),
        post_processor=object(),
        output_dir=tmp_path,
    )

    assert isinstance(writer, Writer)


@pytest.mark.pruned
def test_log_root_logger_branch():
    writer = object.__new__(Writer)

    writer.is_on_root = True
    writer.logger = DummyLogger()

    writer.log_root(logging.INFO, "hello")

    assert len(writer.logger.messages) == 1


@pytest.mark.pruned
def test_log_root_print_branch(capsys):
    writer = object.__new__(Writer)

    writer.is_on_root = True
    writer.logger = None

    writer.log_root(logging.INFO, "hello")

    assert "hello" in capsys.readouterr().out


@pytest.mark.pruned
def test_log_root_non_root_noop(capsys):
    writer = object.__new__(Writer)

    writer.is_on_root = False
    writer.logger = None

    writer.log_root(logging.INFO, "hello")

    assert capsys.readouterr().out == ""


@pytest.mark.pruned
def test_raw_module_normal():
    writer = object.__new__(Writer)

    module = object()

    writer.module = module

    assert writer.raw_module is module


@pytest.mark.pruned
def test_setup_distributed_success(tmp_path):
    predictor = DummyPredictor()

    cfg = WriterConfig(
        predictor=DummyPredictorConfig(predictor),
        num_output_sampling=0,
    )

    writer = Writer(
        cfg,
        DummyLoader(),
        DummyTrainLoaderConfig(),
        DummyModule(),
        object(),
        tmp_path,
    )

    writer.setup_distributed(
        DummyDistributed(),
        DummyLogger(),
    )

    assert writer._setup is True
    assert writer.predictor is predictor


def test_setup_distributed_logger_none(tmp_path):
    predictor = DummyPredictor()

    cfg = WriterConfig(
        predictor=DummyPredictorConfig(predictor),
        num_output_sampling=0,
    )

    writer = Writer(
        cfg,
        DummyLoader(),
        DummyTrainLoaderConfig(),
        DummyModule(),
        object(),
        tmp_path,
    )

    writer.setup_distributed(
        DummyDistributed(),
        None,
    )

    assert writer._setup


def test_setup_distributed_device_mismatch(tmp_path):
    predictor = DummyPredictor()

    module = DummyModule()

    module._get_device = lambda: torch.device("cuda")

    cfg = WriterConfig(
        predictor=DummyPredictorConfig(predictor),
        num_output_sampling=0,
    )

    writer = Writer(
        cfg,
        DummyLoader(),
        DummyTrainLoaderConfig(),
        module,
        object(),
        tmp_path,
    )

    with pytest.raises(
        RuntimeError,
        match="Module is on",
    ):
        writer.setup_distributed(
            DummyDistributed(),
            DummyLogger(),
        )


@pytest.mark.pruned
def test_setup_distributed_barrier_called(tmp_path):
    predictor = DummyPredictor()

    cfg = WriterConfig(
        predictor=DummyPredictorConfig(predictor),
        num_output_sampling=0,
    )

    dist = DummyDistributed(
        distributed=True,
    )

    writer = Writer(
        cfg,
        DummyLoader(),
        DummyTrainLoaderConfig(),
        DummyModule(),
        object(),
        tmp_path,
    )

    writer.setup_distributed(
        dist,
        DummyLogger(),
    )

    assert dist.barrier_called == 1


def test_predict_requires_setup():
    writer = object.__new__(Writer)

    writer._setup = False

    with pytest.raises(
        RuntimeError,
        match="setup_distributed",
    ):
        writer.predict()


def test_predict_runs(
    monkeypatch,
):
    writer = object.__new__(Writer)

    called = {
        "predict": False,
        "clear": False,
    }

    monkeypatch.setattr(
        "cccma_ppp.core.writer.clear_memory",
        lambda: called.__setitem__("clear", True),
    )

    writer._setup = True

    writer.log_root = lambda *a, **k: None

    def fake_predict():
        called["predict"] = True

    writer._predict = fake_predict

    writer.predict()

    assert called["clear"]
    assert called["predict"]


def test_build_train_loader_train(tmp_path):
    from cccma_ppp.generic.runtime import RuntimeContext

    RuntimeContext.GLOBAL_EXP_DIR = tmp_path
    writer = object.__new__(Writer)

    cfg = DummyTrainLoaderConfig()

    writer.TrainLoaderConfig = cfg
    writer.distributed = object()

    result = writer.build_train_loader()

    assert result == "TRAIN"
    assert cfg.train_called


def test_setup_distributed_extract_training_vars(
    tmp_path,
    monkeypatch,
):
    predictor = DummyPredictor(extract_training_vars=True)

    called = {"save": False}

    monkeypatch.setattr(
        Writer,
        "_save_train_stats",
        lambda self: called.__setitem__("save", True),
    )

    cfg = WriterConfig(
        predictor=DummyPredictorConfig(predictor),
        num_output_sampling=0,
    )

    writer = Writer(
        cfg,
        DummyLoader(),
        DummyTrainLoaderConfig(),
        DummyModule(),
        object(),
        tmp_path,
    )

    writer.setup_distributed(
        DummyDistributed(),
        DummyLogger(),
    )

    assert called["save"]


def test_raw_module_ddp_branch(monkeypatch):
    class FakeModule:
        pass

    class FakeDDP:
        def __init__(self):
            self.module = FakeModule()

    monkeypatch.setattr(
        torch.nn.parallel,
        "DistributedDataParallel",
        FakeDDP,
    )

    writer = object.__new__(Writer)
    writer.module = FakeDDP()

    assert writer.raw_module is writer.module.module


def test_save_train_stats_file_exists_skips_loader(
    tmp_path,
):
    writer = object.__new__(Writer)

    writer.output_dir = tmp_path

    writer.is_distributed = False
    writer.is_on_root = True

    writer.device = torch.device("cpu")

    writer.predictor = DummyPredictor()

    torch.save(
        {},
        writer.train_stats_save_dir,
    )

    called = {
        "loader": False,
    }

    writer.build_train_loader = lambda **k: called.__setitem__(
        "loader",
        True,
    )

    writer._save_train_stats()

    assert not called["loader"]


def test_save_train_stats_barrier(tmp_path):
    writer = object.__new__(Writer)

    writer.output_dir = tmp_path

    writer.is_distributed = True
    writer.is_on_root = False

    dist = DummyDistributed(distributed=True)
    writer.distributed = dist

    writer.device = torch.device("cpu")

    writer.config = SimpleNamespace(get_trained_model_stats_from_validation=False)

    writer.predictor = DummyPredictor()

    writer.build_train_loader = lambda **k: []

    writer.aggregate_train_stats = lambda stats: None

    writer._save_train_stats()

    assert dist.barrier_called == 1


@pytest.mark.pruned
def test_save_train_stats_validation_branch(
    tmp_path,
):
    called = {}

    writer = object.__new__(Writer)

    writer.output_dir = tmp_path

    writer.device = torch.device("cpu")
    writer.is_on_root = True
    writer.is_distributed = False

    writer.config = SimpleNamespace(get_trained_model_stats_from_validation=True)

    writer.predictor = DummyPredictor()

    writer.build_train_loader = lambda **k: (
        called.setdefault(
            "validation",
            k["from_validation"],
        ),
        [],
    )[1]

    writer.aggregate_train_stats = lambda stats: None

    writer._save_train_stats()

    assert called["validation"] is True


@pytest.mark.pruned
def test_aggregate_predictions_to_netcdf_root(
    monkeypatch,
):
    called = {}

    monkeypatch.setattr(
        "cccma_ppp.core.writer.aggregate_predictions",
        lambda *args: called.setdefault(
            "called",
            True,
        ),
    )

    writer = object.__new__(Writer)

    writer.is_distributed = False
    writer.is_on_root = True

    writer.post_processor = object()
    writer.predictor = DummyPredictor()

    writer.output_dir = Path("/tmp")

    writer.aggregate_predictions_to_netcdf()

    assert called["called"]


def test_aggregate_predictions_to_netcdf_non_root(
    monkeypatch,
):
    called = {}

    monkeypatch.setattr(
        "cccma_ppp.core.writer.aggregate_predictions",
        lambda *args: called.setdefault(
            "called",
            True,
        ),
    )

    writer = object.__new__(Writer)

    writer.is_distributed = False
    writer.is_on_root = False

    writer.post_processor = object()
    writer.predictor = DummyPredictor()

    writer.output_dir = Path("/tmp")

    writer.aggregate_predictions_to_netcdf()

    assert "called" not in called


def test_aggregate_predictions_to_netcdf_no_postprocess(
    monkeypatch,
):
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.core.writer.aggregate_predictions",
        lambda post_processor, *args: captured.setdefault(
            "pp",
            post_processor,
        ),
    )

    writer = object.__new__(Writer)

    writer.is_distributed = False
    writer.is_on_root = True

    writer.post_processor = object()
    writer.output_dir = Path("/tmp")

    writer.predictor = DummyPredictor()

    writer.aggregate_predictions_to_netcdf(do_post_process=False)

    assert captured["pp"] is None


def test_aggregate_predictions_to_netcdf_latent(
    monkeypatch,
):
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.core.writer.aggregate_predictions",
        lambda pp, out, name, *_: captured.setdefault(
            "name",
            name,
        ),
    )

    writer = object.__new__(Writer)

    writer.is_distributed = False
    writer.is_on_root = True

    writer.post_processor = object()
    writer.output_dir = Path("/tmp")

    writer.predictor = DummyPredictor(save_latent=True)

    writer.aggregate_predictions_to_netcdf()

    assert captured["name"] == "latent"


@pytest.mark.pruned
def test_aggregate_predictions_cleanup_false(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    xr.DataArray(
        [[1.0]],
        dims=("year", "channels"),
        coords={
            "year": [2000],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

    aggregate_predictions(
        None,
        tmp_path,
        cleanup_temp=False,
    )

    assert temp_dir.exists()


def test_aggregate_predictions_year_not_present(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    xr.DataArray(
        [[1.0]],
        dims=("year", "channels"),
        coords={
            "year": [2000],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

    xr.DataArray(
        [[2.0]],
        dims=("year", "channels"),
        coords={
            "year": [2001],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp_dir / "prediction_rank0_1.nc")

    aggregate_predictions(
        None,
        tmp_path,
        cleanup_temp=False,
    )

    assert (tmp_path / "prediction_2000.nc").exists()

    assert (tmp_path / "prediction_2001.nc").exists()


@pytest.mark.pruned
def test_aggregate_predictions_postprocessor_branch(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    xr.DataArray(
        [[1.0]],
        dims=("year", "channels"),
        coords={
            "year": [2000],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

    class FakePP:
        def to_dataset(
            self,
            ds,
        ):
            return ds.to_dataset(dim="channels")

        def inverse_transform(
            self,
            ds,
        ):
            return ds

    aggregate_predictions(
        FakePP(),
        tmp_path,
        cleanup_temp=False,
    )

    assert (tmp_path / "prediction_2000.nc").exists()


def test_aggregate_predictions_auxiliary_year_coord(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    da = xr.DataArray(
        [[1.0]],
        dims=("sample", "channels"),
        coords={
            "channels": ["prediction"],
            "year": ("sample", [2000]),
        },
        name="prediction",
    )

    da.to_netcdf(temp_dir / "prediction_rank0_0.nc")

    aggregate_predictions(
        None,
        tmp_path,
        cleanup_temp=False,
    )

    assert (tmp_path / "prediction_2000.nc").exists()


@pytest.mark.pruned
def test_aggregate_train_stats_root_skip_none_stat(
    tmp_path,
):
    writer = object.__new__(Writer)

    writer.output_dir = tmp_path
    writer.is_on_root = True
    writer.is_distributed = False

    class EmptyStat:
        sum_x = None

    writer.aggregate_train_stats({"empty": EmptyStat()})

    saved = torch.load(writer.train_stats_save_dir)

    assert saved == {}


def test_aggregate_predictions_missing_year_coord(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    xr.DataArray(
        [1.0],
        dims=("x",),
        coords={"x": [0]},
        name="prediction",
    ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

    with pytest.raises(
        RuntimeError,
        match="year",
    ):
        aggregate_predictions(
            None,
            tmp_path,
        )


def test_aggregate_predictions_no_temp_files(
    tmp_path,
):
    with pytest.raises(
        RuntimeError,
        match="No temporary prediction files",
    ):
        aggregate_predictions(
            None,
            tmp_path,
        )


def test_aggregate_predictions_logger_called(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    messages = []

    xr.DataArray(
        [[1.0]],
        dims=("year", "channels"),
        coords={
            "year": [2000],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

    aggregate_predictions(
        None,
        tmp_path,
        cleanup_temp=False,
        logger_function=lambda lvl, msg: messages.append(msg),
    )

    assert len(messages) >= 2


@pytest.mark.pruned
def test_aggregate_train_stats_root_empty_stats(
    tmp_path,
):
    writer = object.__new__(Writer)

    writer.output_dir = tmp_path
    writer.is_distributed = False
    writer.is_on_root = True

    writer.aggregate_train_stats({})

    saved = torch.load(writer.train_stats_save_dir)

    assert saved == {}


@pytest.mark.pruned
def test_aggregate_train_stats_skip_sum_x_none(
    tmp_path,
):
    class EmptyStat:
        sum_x = None

    writer = object.__new__(Writer)

    writer.output_dir = tmp_path
    writer.is_distributed = False
    writer.is_on_root = True

    writer.aggregate_train_stats(
        {
            "empty": EmptyStat(),
        }
    )

    saved = torch.load(writer.train_stats_save_dir)

    assert saved == {}


@pytest.mark.pruned
def test_aggregate_train_stats_finalize_branch(
    tmp_path,
):
    class Stat:
        sum_x = torch.tensor([1.0])

        def distributed_reduce(self):
            pass

        def finalize(self):
            return (
                torch.tensor([1.0]),
                torch.tensor([[1.0]]),
            )

    writer = object.__new__(Writer)

    writer.output_dir = tmp_path
    writer.is_distributed = False
    writer.is_on_root = True

    writer.aggregate_train_stats(
        {
            "test": Stat(),
        }
    )

    saved = torch.load(writer.train_stats_save_dir)

    assert "test_mean" in saved
    assert "test_cov" in saved


@pytest.mark.pruned
def test_aggregate_train_stats_distributed_reduce_called():
    called = {"reduce": False}

    class Stat:
        sum_x = torch.tensor([1.0])

        def distributed_reduce(self):
            called["reduce"] = True

        def finalize(self):
            return (
                torch.tensor([1.0]),
                torch.tensor([[1.0]]),
            )

    writer = object.__new__(Writer)

    writer.is_on_root = False
    writer.is_distributed = False

    writer.aggregate_train_stats(
        {
            "x": Stat(),
        }
    )

    assert called["reduce"]


@pytest.mark.pruned
def test_save_train_stats_existing_file_skips_loader(
    tmp_path,
):
    writer = object.__new__(Writer)

    writer.output_dir = tmp_path

    writer.is_distributed = False
    writer.is_on_root = True

    torch.save(
        {},
        writer.train_stats_save_dir,
    )

    called = {"loader": False}

    writer.build_train_loader = lambda **kwargs: called.__setitem__(
        "loader",
        True,
    )

    writer._save_train_stats()

    assert called["loader"] is False


def test_save_train_stats_calls_aggregate(
    tmp_path,
):
    class Batch:
        def to_device(self, device):
            return self

    class Predictor:
        stats = {"x": object()}

        def _infer_on_batch(
            self,
            batch,
            _getting_train_stats=False,
        ):
            return self.stats

    writer = object.__new__(Writer)

    writer.output_dir = tmp_path
    writer.is_on_root = True
    writer.is_distributed = False
    writer.device = torch.device("cpu")

    writer.predictor = Predictor()

    writer.config = SimpleNamespace(get_trained_model_stats_from_validation=False)

    writer.build_train_loader = lambda **kwargs: [Batch()]

    called = {"aggregate": False}

    writer.aggregate_train_stats = lambda stats: called.__setitem__(
        "aggregate",
        True,
    )

    writer._save_train_stats()

    assert called["aggregate"]


@pytest.mark.pruned
def test_save_train_stats_validation_loader_branch(
    tmp_path,
):
    writer = object.__new__(Writer)

    writer.output_dir = tmp_path

    writer.device = torch.device("cpu")
    writer.is_on_root = True
    writer.is_distributed = False

    writer.predictor = SimpleNamespace(
        stats={},
        _infer_on_batch=lambda *a, **k: {},
    )

    writer.config = SimpleNamespace(get_trained_model_stats_from_validation=True)

    called = {}

    writer.build_train_loader = lambda **kwargs: (
        called.setdefault(
            "from_validation",
            kwargs["from_validation"],
        ),
        [],
    )[1]

    writer.aggregate_train_stats = lambda stats: None

    writer._save_train_stats()

    assert called["from_validation"] is True


def test_aggregate_predictions_to_netcdf_distributed_barriers(
    monkeypatch,
):
    calls = {"aggregate": 0}

    monkeypatch.setattr(
        "cccma_ppp.core.writer.aggregate_predictions",
        lambda *args: calls.__setitem__(
            "aggregate",
            calls["aggregate"] + 1,
        ),
    )

    class Dist:
        count = 0

        def barrier(self):
            self.count += 1

    writer = object.__new__(Writer)

    writer.distributed = Dist()

    writer.is_distributed = True
    writer.is_on_root = True

    writer.output_dir = Path("/tmp")
    writer.post_processor = object()

    writer.predictor = SimpleNamespace(save_latent=False)

    writer.aggregate_predictions_to_netcdf()

    assert writer.distributed.count == 2
    assert calls["aggregate"] == 1


@pytest.mark.pruned
def test_aggregate_predictions_to_netcdf_latent_name(
    monkeypatch,
):
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.core.writer.aggregate_predictions",
        lambda pp, out, name, log: captured.setdefault(
            "name",
            name,
        ),
    )

    writer = object.__new__(Writer)

    writer.is_distributed = False
    writer.is_on_root = True

    writer.output_dir = Path("/tmp")
    writer.post_processor = object()

    writer.predictor = SimpleNamespace(save_latent=True)

    writer.aggregate_predictions_to_netcdf()

    assert captured["name"] == "latent"


@pytest.mark.pruned
def test_aggregate_predictions_no_files(
    tmp_path,
):
    with pytest.raises(
        RuntimeError,
        match="No temporary prediction files",
    ):
        aggregate_predictions(
            None,
            tmp_path,
        )


@pytest.mark.pruned
def test_aggregate_predictions_missing_year_coordinate(
    tmp_path,
):
    temp = tmp_path / "_temp"
    temp.mkdir()

    xr.DataArray(
        [1.0],
        dims=("x",),
        coords={"x": [0]},
        name="prediction",
    ).to_netcdf(temp / "prediction_rank0_0.nc")

    with pytest.raises(
        RuntimeError,
        match="year",
    ):
        aggregate_predictions(
            None,
            tmp_path,
        )


@pytest.mark.pruned
def test_aggregate_predictions_multiple_years(
    tmp_path,
):
    temp = tmp_path / "_temp"
    temp.mkdir()

    xr.DataArray(
        [[1.0]],
        dims=("year", "channels"),
        coords={
            "year": [2000],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp / "prediction_rank0_0.nc")

    xr.DataArray(
        [[2.0]],
        dims=("year", "channels"),
        coords={
            "year": [2001],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp / "prediction_rank0_1.nc")

    aggregate_predictions(
        None,
        tmp_path,
        cleanup_temp=False,
    )

    assert (tmp_path / "prediction_2000.nc").exists()

    assert (tmp_path / "prediction_2001.nc").exists()


@pytest.mark.pruned
def test_aggregate_predictions_year_missing_from_one_file(
    tmp_path,
):
    temp = tmp_path / "_temp"
    temp.mkdir()

    xr.DataArray(
        [[1.0]],
        dims=("year", "channels"),
        coords={
            "year": [2000],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp / "prediction_rank0_0.nc")

    xr.DataArray(
        [[2.0]],
        dims=("year", "channels"),
        coords={
            "year": [2001],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp / "prediction_rank0_1.nc")

    aggregate_predictions(
        None,
        tmp_path,
        cleanup_temp=False,
    )

    assert (tmp_path / "prediction_2000.nc").exists()

    assert (tmp_path / "prediction_2001.nc").exists()


def test_writer_config_build_rejects_predictor_module_type_mismatch(
    tmp_path,
):
    predictor = DummyPredictor()

    config = WriterConfig(
        predictor=DummyPredictorConfig(predictor),
        num_output_sampling=0,
    )
    config.predictor._type = "cvae"

    with pytest.raises(
        RuntimeError,
        match="provided selector config",
    ):
        config.build(
            inference_data_loader=DummyLoader(),
            train_dataloader_config=DummyTrainLoaderConfig(),
            module=DummyModule(),
            post_processor=object(),
            output_dir=tmp_path,
        )


@pytest.mark.pruned
def test_writer_config_build_is_case_insensitive_for_module_type(
    tmp_path,
):
    predictor = DummyPredictor()
    predictor_config = DummyPredictorConfig(predictor)
    predictor_config._type = "dummy"

    module = DummyModule()
    module.config._type = "DUMMY"

    config = WriterConfig(
        predictor=predictor_config,
        num_output_sampling=0,
    )

    writer = config.build(
        inference_data_loader=DummyLoader(),
        train_dataloader_config=DummyTrainLoaderConfig(),
        module=module,
        post_processor=object(),
        output_dir=tmp_path,
    )

    assert isinstance(writer, Writer)


@pytest.mark.pruned
def test_writer_initialization_converts_output_dir_to_path(
    tmp_path,
):
    writer = Writer(
        WriterConfig(
            predictor=DummyPredictorConfig(DummyPredictor()),
            num_output_sampling=0,
        ),
        DummyLoader(),
        DummyTrainLoaderConfig(),
        DummyModule(),
        object(),
        str(tmp_path),
    )

    assert writer.output_dir == tmp_path
    assert isinstance(writer.output_dir, Path)
    assert writer._setup is False


@pytest.mark.pruned
def test_setup_distributed_passes_build_arguments(
    tmp_path,
):
    predictor = DummyPredictor()
    predictor_config = DummyPredictorConfig(predictor)

    config = WriterConfig(
        predictor=predictor_config,
        num_output_sampling=12,
    )
    module = DummyModule()
    distributed = DummyDistributed()

    writer = Writer(
        config,
        DummyLoader(),
        DummyTrainLoaderConfig(),
        module,
        object(),
        tmp_path,
    )

    writer.setup_distributed(
        distributed,
        DummyLogger(),
    )

    assert predictor_config.build_args == {
        "module": module,
        "distributed": distributed,
        "output_dir": tmp_path,
        "num_output_sampling": 12,
    }


@pytest.mark.pruned
def test_setup_distributed_creates_temp_directory_on_root(
    tmp_path,
):
    writer = Writer(
        WriterConfig(
            predictor=DummyPredictorConfig(DummyPredictor()),
            num_output_sampling=0,
        ),
        DummyLoader(),
        DummyTrainLoaderConfig(),
        DummyModule(),
        object(),
        tmp_path,
    )

    writer.setup_distributed(
        DummyDistributed(root=True),
        DummyLogger(),
    )

    assert writer.temp_save_dir == tmp_path / "_temp"
    assert writer.temp_save_dir.is_dir()


@pytest.mark.pruned
def test_setup_distributed_non_root_does_not_create_temp_directory(
    tmp_path,
):
    writer = Writer(
        WriterConfig(
            predictor=DummyPredictorConfig(DummyPredictor()),
            num_output_sampling=0,
        ),
        DummyLoader(),
        DummyTrainLoaderConfig(),
        DummyModule(),
        object(),
        tmp_path,
    )

    writer.setup_distributed(
        DummyDistributed(root=False),
        DummyLogger(),
    )

    assert writer.temp_save_dir == tmp_path / "_temp"
    assert not writer.temp_save_dir.exists()


def test_setup_distributed_records_distributed_properties(
    tmp_path,
):
    distributed = DummyDistributed(
        distributed=True,
        root=False,
    )
    distributed.rank = 2
    distributed.local_rank = 1
    distributed.world_size = 4

    writer = Writer(
        WriterConfig(
            predictor=DummyPredictorConfig(DummyPredictor()),
            num_output_sampling=0,
        ),
        DummyLoader(),
        DummyTrainLoaderConfig(),
        DummyModule(),
        object(),
        tmp_path,
    )

    writer.setup_distributed(
        distributed,
        DummyLogger(),
    )

    assert writer.device == torch.device("cpu")
    assert writer.rank == 2
    assert writer.local_rank == 1
    assert writer.world_size == 4
    assert writer.is_distributed is True
    assert writer.is_on_root is False


@pytest.mark.pruned
def test_predict_logs_elapsed_time(
    monkeypatch,
):
    writer = object.__new__(Writer)
    writer._setup = True

    messages = []
    writer.log_root = lambda level, message: messages.append(message)
    writer._predict = lambda: None

    times = iter([100.0, 102.5])

    monkeypatch.setattr(
        "cccma_ppp.core.writer.time.time",
        lambda: next(times),
    )
    monkeypatch.setattr(
        "cccma_ppp.core.writer.clear_memory",
        lambda: None,
    )

    writer.predict()

    assert messages[0] == "Starting Inference Loop..."
    assert messages[-1] == "Inference finished in 2.50s"


@pytest.mark.pruned
def test_save_train_stats_moves_batch_and_requests_training_stats(
    tmp_path,
):
    batch = DummyBatch()

    class RecordingPredictor:
        def __init__(self):
            self.stats = {}
            self.calls = []

        def _infer_on_batch(
            self,
            received_batch,
            _getting_train_stats=False,
        ):
            self.calls.append(
                {
                    "batch": received_batch,
                    "getting_stats": _getting_train_stats,
                }
            )
            return self.stats

    predictor = RecordingPredictor()
    writer = object.__new__(Writer)

    writer.output_dir = tmp_path
    writer.device = torch.device("cpu")
    writer.is_on_root = True
    writer.is_distributed = False
    writer.predictor = predictor
    writer.config = SimpleNamespace(
        get_trained_model_stats_from_validation=False,
    )
    writer.build_train_loader = lambda **kwargs: [batch]

    captured = {}
    writer.aggregate_train_stats = lambda stats: captured.setdefault(
        "stats",
        stats,
    )

    writer._save_train_stats()

    assert batch.device == torch.device("cpu")
    assert predictor.calls == [
        {
            "batch": batch,
            "getting_stats": True,
        }
    ]
    assert captured["stats"] is predictor.stats


@pytest.mark.pruned
def test_save_train_stats_calls_gc_collect(
    tmp_path,
    monkeypatch,
):
    writer = object.__new__(Writer)

    writer.output_dir = tmp_path
    writer.device = torch.device("cpu")
    writer.is_on_root = True
    writer.is_distributed = False
    writer.predictor = DummyPredictor()
    writer.config = SimpleNamespace(
        get_trained_model_stats_from_validation=False,
    )
    writer.build_train_loader = lambda **kwargs: []
    writer.aggregate_train_stats = lambda stats: None

    calls = []

    monkeypatch.setattr(
        "cccma_ppp.core.writer.gc.collect",
        lambda: calls.append(True),
    )

    writer._save_train_stats()

    assert calls == [True]


def test_aggregate_train_stats_saves_expected_values(
    tmp_path,
):
    active = DummyStat(active=True)
    inactive = DummyStat(active=False)

    writer = object.__new__(Writer)
    writer.output_dir = tmp_path
    writer.is_on_root = True
    writer.is_distributed = False

    writer.aggregate_train_stats(
        {
            "active": active,
            "inactive": inactive,
        }
    )

    saved = torch.load(
        writer.train_stats_save_dir,
        weights_only=True,
    )

    assert active.reduced is True
    assert inactive.reduced is False
    assert torch.equal(
        saved["active_mean"],
        torch.tensor([1.0]),
    )
    assert torch.equal(
        saved["active_cov"],
        torch.tensor([[1.0]]),
    )
    assert "inactive_mean" not in saved
    assert "inactive_cov" not in saved


@pytest.mark.pruned
def test_aggregate_train_stats_non_root_does_not_save(
    tmp_path,
):
    writer = object.__new__(Writer)
    writer.output_dir = tmp_path
    writer.is_on_root = False
    writer.is_distributed = False

    stat = DummyStat(active=True)

    writer.aggregate_train_stats(
        {"value": stat},
    )

    assert stat.reduced is True
    assert not writer.train_stats_save_dir.exists()


def test_aggregate_train_stats_distributed_barrier(
    tmp_path,
):
    distributed = DummyDistributed(
        distributed=True,
    )

    writer = object.__new__(Writer)
    writer.output_dir = tmp_path
    writer.is_on_root = False
    writer.is_distributed = True
    writer.distributed = distributed

    writer.aggregate_train_stats({})

    assert distributed.barrier_called == 1


@pytest.mark.pruned
def test_log_root_passes_formatting_arguments():
    writer = object.__new__(Writer)
    writer.is_on_root = True

    class RecordingLogger:
        def __init__(self):
            self.call = None

        def log(self, level, msg, *args):
            self.call = (level, msg, args)

    writer.logger = RecordingLogger()

    writer.log_root(
        logging.INFO,
        "value=%s",
        42,
    )

    assert writer.logger.call == (
        logging.INFO,
        "value=%s",
        (42,),
    )


def test_aggregate_predictions_cleanup_removes_temp_files_and_directory(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    temp_file = temp_dir / "prediction_rank0_0.nc"

    xr.DataArray(
        [[1.0]],
        dims=("year", "channels"),
        coords={
            "year": [2000],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp_file)

    aggregate_predictions(
        None,
        tmp_path,
        cleanup_temp=True,
    )

    assert (tmp_path / "prediction_2000.nc").exists()
    assert not temp_file.exists()
    assert not temp_dir.exists()


@pytest.mark.pruned
def test_aggregate_predictions_custom_naming_convention(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    xr.DataArray(
        [[1.0]],
        dims=("year", "channels"),
        coords={
            "year": [2000],
            "channels": ["latent"],
        },
        name="latent",
    ).to_netcdf(
        temp_dir / "latent_rank0_0.nc",
    )

    aggregate_predictions(
        None,
        tmp_path,
        naming_convention="latent",
        cleanup_temp=False,
    )

    assert (tmp_path / "latent_2000.nc").exists()


@pytest.mark.pruned
def test_aggregate_predictions_ignores_unrelated_temp_files(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    xr.DataArray(
        [[1.0]],
        dims=("year", "channels"),
        coords={
            "year": [2000],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(
        temp_dir / "prediction_rank0_0.nc",
    )

    unrelated = temp_dir / "other_rank0_0.nc"

    xr.DataArray(
        [[2.0]],
        dims=("year", "channels"),
        coords={
            "year": [2001],
            "channels": ["other"],
        },
        name="other",
    ).to_netcdf(unrelated)

    aggregate_predictions(
        None,
        tmp_path,
        cleanup_temp=False,
    )

    assert (tmp_path / "prediction_2000.nc").exists()
    assert not (tmp_path / "prediction_2001.nc").exists()
    assert unrelated.exists()


@pytest.mark.pruned
def test_aggregate_predictions_logger_receives_expected_messages(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    xr.DataArray(
        [[1.0]],
        dims=("year", "channels"),
        coords={
            "year": [2000],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(
        temp_dir / "prediction_rank0_0.nc",
    )

    calls = []

    aggregate_predictions(
        None,
        tmp_path,
        logger_function=lambda level, message: calls.append((level, message)),
        cleanup_temp=False,
    )

    assert calls[0][0] == logging.INFO
    assert "Aggregating temporary prediction files" in calls[0][1]
    assert calls[-1][0] == logging.INFO
    assert "Saved aggregated predictions for year 2000" in calls[-1][1]

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import xarray as xr

from cccma_ppp.core.writer import (
    Writer,
    WriterConfig,
    aggregate_predictions,
)
from cccma_ppp.generic.runtime import RuntimeContext


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


def test_writer_config_build(tmp_path):
    predictor = DummyPredictor()

    cfg = WriterConfig(
        predictor=DummyPredictorConfig(predictor),
        num_output_sampling=1,
    )

    writer = cfg.build(
        inference_data_loader=DummyLoader(),
        train_dataloader_config=DummyTrainLoaderConfig(),
        module=DummyModule(),
        post_processor=object(),
        output_dir=tmp_path,
    )

    assert isinstance(writer, Writer)


def test_log_root_logger_branch():
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

    writer.is_on_root = True
    writer.logger = DummyLogger()

    writer.log_root(logging.INFO, "hello")

    assert len(writer.logger.messages) == 1


def test_log_root_print_branch(capsys):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

    writer.is_on_root = True
    writer.logger = None

    writer.log_root(logging.INFO, "hello")

    assert "hello" in capsys.readouterr().out


def test_log_root_non_root_noop(capsys):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

    writer.is_on_root = False
    writer.logger = None

    writer.log_root(logging.INFO, "hello")

    assert capsys.readouterr().out == ""


def test_raw_module_normal():
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

    module = object()

    writer.module = module

    assert writer.raw_module is module


def test_setup_distributed_logger_none(tmp_path):
    predictor = DummyPredictor()

    cfg = WriterConfig(
        predictor=DummyPredictorConfig(predictor),
        num_output_sampling=1,
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
        num_output_sampling=1,
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


def test_setup_distributed_barrier_called(tmp_path):
    predictor = DummyPredictor()

    cfg = WriterConfig(
        predictor=DummyPredictorConfig(predictor),
        num_output_sampling=1,
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
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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
        num_output_sampling=1,
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
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)
    writer.module = FakeDDP()

    assert writer.raw_module is writer.module.module


def test_save_train_stats_file_exists_skips_loader(
    tmp_path,
):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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


def test_save_train_stats_validation_branch(
    tmp_path,
):
    called = {}

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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


def test_aggregate_predictions_to_netcdf_root(
    monkeypatch,
):
    called = {}

    monkeypatch.setattr(
        "cccma_ppp.core.writer.aggregate_predictions",
        lambda **kwargs: called.setdefault(
            "called",
            True,
        ),
    )

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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
        lambda **kwargs: called.setdefault(
            "called",
            True,
        ),
    )

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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
        lambda **kwargs: captured.setdefault(
            "pp",
            kwargs["post_processor"],
        ),
    )

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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
        lambda **kwargs: captured.setdefault(
            "name",
            "latent",
        ),
    )

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

    writer.is_distributed = False
    writer.is_on_root = True

    writer.post_processor = object()
    writer.output_dir = Path("/tmp")

    writer.predictor = DummyPredictor(save_latent=True)

    writer.aggregate_predictions_to_netcdf()

    assert captured["name"] == "latent"


def test_aggregate_predictions_cleanup_false(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    xr.DataArray(
        [[[1.0]]],
        dims=("time", "lead_time", "channels"),
        coords={
            "time": np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
            "lead_time": [1],
            "channels": ["prediction"],
        },
        name="prediction",
        attrs={"source": "prediction", "nstds": "NA"},
    ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

    aggregate_predictions(
        None,
        tmp_path,
        cleanup_temp=False,
    )

    assert temp_dir.exists()


def test_aggregate_train_stats_root_skip_none_stat(
    tmp_path,
):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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
        attrs={"source": "prediction", "nstds": "NA"},
    ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

    with pytest.raises(
        RuntimeError,
        match="time",
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
        [[[1.0]]],
        dims=("time", "lead_time", "channels"),
        coords={
            "time": np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
            "lead_time": [1],
            "channels": ["prediction"],
        },
        name="prediction",
        attrs={"source": "prediction", "nstds": "NA"},
    ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

    aggregate_predictions(
        None,
        tmp_path,
        cleanup_temp=False,
        logger_function=lambda lvl, msg: messages.append(msg),
    )

    assert len(messages) >= 2


def test_aggregate_train_stats_root_empty_stats(
    tmp_path,
):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

    writer.output_dir = tmp_path
    writer.is_distributed = False
    writer.is_on_root = True

    writer.aggregate_train_stats({})

    saved = torch.load(writer.train_stats_save_dir)

    assert saved == {}


def test_aggregate_train_stats_skip_sum_x_none(
    tmp_path,
):
    class EmptyStat:
        sum_x = None

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

    writer.is_on_root = False
    writer.is_distributed = False

    writer.aggregate_train_stats(
        {
            "x": Stat(),
        }
    )

    assert called["reduce"]


def test_save_train_stats_existing_file_skips_loader(
    tmp_path,
):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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


def test_save_train_stats_validation_loader_branch(
    tmp_path,
):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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
        lambda **kwargs: calls.__setitem__(
            "aggregate",
            calls["aggregate"] + 1,
        ),
    )

    class Dist:
        count = 0

        def barrier(self):
            self.count += 1

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

    writer.distributed = Dist()

    writer.is_distributed = True
    writer.is_on_root = True

    writer.output_dir = Path("/tmp")
    writer.post_processor = object()

    writer.predictor = SimpleNamespace(save_latent=False)

    writer.aggregate_predictions_to_netcdf()

    assert writer.distributed.count == 2
    assert calls["aggregate"] == 1


def test_aggregate_predictions_to_netcdf_latent_name(
    monkeypatch,
):
    captured = {}

    monkeypatch.setattr(
        "cccma_ppp.core.writer.aggregate_predictions",
        lambda **kwargs: captured.setdefault(
            "name",
            "latent",
        ),
    )

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

    writer.is_distributed = False
    writer.is_on_root = True

    writer.output_dir = Path("/tmp")
    writer.post_processor = object()

    writer.predictor = SimpleNamespace(save_latent=True)

    writer.aggregate_predictions_to_netcdf()

    assert captured["name"] == "latent"


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


def test_writer_config_build_rejects_predictor_module_type_mismatch(
    tmp_path,
):
    predictor = DummyPredictor()

    config = WriterConfig(
        predictor=DummyPredictorConfig(predictor),
        num_output_sampling=1,
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
        num_output_sampling=1,
    )

    writer = config.build(
        inference_data_loader=DummyLoader(),
        train_dataloader_config=DummyTrainLoaderConfig(),
        module=module,
        post_processor=object(),
        output_dir=tmp_path,
    )

    assert isinstance(writer, Writer)


def test_writer_initialization_converts_output_dir_to_path(
    tmp_path,
):
    writer = Writer(
        WriterConfig(
            predictor=DummyPredictorConfig(DummyPredictor()),
            num_output_sampling=1,
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


def test_setup_distributed_creates_temp_directory_on_root(
    tmp_path,
):
    writer = Writer(
        WriterConfig(
            predictor=DummyPredictorConfig(DummyPredictor()),
            num_output_sampling=1,
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


def test_setup_distributed_non_root_does_not_create_temp_directory(
    tmp_path,
):
    writer = Writer(
        WriterConfig(
            predictor=DummyPredictorConfig(DummyPredictor()),
            num_output_sampling=1,
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
            num_output_sampling=1,
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


def test_predict_logs_elapsed_time(
    monkeypatch,
):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)
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
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

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


def test_aggregate_train_stats_saves_expected_values(
    tmp_path,
):
    active = DummyStat(active=True)
    inactive = DummyStat(active=False)

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)
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


def test_aggregate_train_stats_non_root_does_not_save(
    tmp_path,
):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)
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
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)
    writer.output_dir = tmp_path
    writer.is_on_root = False
    writer.is_distributed = True
    writer.distributed = distributed

    writer.aggregate_train_stats({})

    assert distributed.barrier_called == 1


def test_log_root_passes_formatting_arguments():
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)
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


class RecordingLoader:
    def __init__(self, batches):
        self.batches = batches
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        yield from self.batches


class EvalRecordingModule(DummyModule):
    def __init__(self):
        super().__init__()
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


def make_predict_writer(
    *,
    loader=None,
    predictor=None,
    is_on_root=True,
):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)

    writer.module = EvalRecordingModule()
    writer.InferenceLoader = loader or RecordingLoader(
        [
            DummyBatch(),
        ]
    )
    writer.predictor = predictor or DummyPredictor()
    writer.device = torch.device("cpu")
    writer.is_on_root = is_on_root
    writer.is_distributed = False
    writer.output_dir = Path("/tmp")

    return writer


def test_writer_config_accepts_positive_output_sampling():
    config = WriterConfig(
        predictor=object(),
        num_output_sampling=1,
    )

    assert config.num_output_sampling == 1


def test_build_train_loader_forwards_training_arguments(
    tmp_path,
):
    RuntimeContext.GLOBAL_EXP_DIR = str(tmp_path)

    captured = {}

    class Config:
        def setup_distributed(
            self,
            distributed,
            **kwargs,
        ):
            captured["setup"] = (
                distributed,
                kwargs,
            )

        def build_train_loader(
            self,
            **kwargs,
        ):
            captured["build"] = kwargs
            return "loader"

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)
    writer.TrainLoaderConfig = Config()
    writer.distributed = object()

    result = writer.build_train_loader(
        return_metadata=True,
        shuffle=False,
    )

    assert result == "loader"
    assert captured["setup"] == (
        writer.distributed,
        {
            "load_path": (tmp_path / "preprocessing_pipeline"),
        },
    )
    assert captured["build"] == {
        "return_metadata": True,
        "shuffle": False,
    }


def test_build_train_loader_forwards_validation_arguments(
    tmp_path,
):
    RuntimeContext.GLOBAL_EXP_DIR = str(tmp_path)

    captured = {}

    class Config:
        def setup_distributed(
            self,
            distributed,
            **kwargs,
        ):
            captured["setup"] = (
                distributed,
                kwargs,
            )

        def build_validation_loader(
            self,
            **kwargs,
        ):
            captured["build"] = kwargs
            return "validation"

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)
    writer.TrainLoaderConfig = Config()
    writer.distributed = object()

    result = writer.build_train_loader(
        from_validation=True,
        return_metadata=True,
        shuffle=True,
    )

    assert result == "validation"
    assert captured["setup"][0] is writer.distributed
    assert captured["setup"][1]["load_path"] == (tmp_path / "preprocessing_pipeline")
    assert captured["build"] == {
        "supress_error": False,
        "return_metadata": True,
        "shuffle": True,
    }


def test_predict_loop_propagates_predictor_error():
    class FailingPredictor(DummyPredictor):
        def _infer_on_batch(
            self,
            batch,
            _getting_train_stats=False,
        ):
            raise RuntimeError("inference failed")

    writer = make_predict_writer(
        predictor=FailingPredictor(),
    )
    writer.aggregate_predictions_to_netcdf = lambda value: pytest.fail(
        "Aggregation must not run after failure."
    )

    with pytest.raises(
        RuntimeError,
        match="inference failed",
    ):
        writer._predict()


def test_setup_distributed_does_not_mark_setup_after_predictor_error(
    tmp_path,
):
    class FailingPredictorConfig(DummyPredictorConfig):
        def build(
            self,
            module,
            distributed,
            output_dir,
            num_output_sampling,
        ):
            raise RuntimeError("predictor failed")

    config = WriterConfig(
        predictor=FailingPredictorConfig(
            DummyPredictor(),
        ),
    )

    writer = Writer(
        config,
        DummyLoader(),
        DummyTrainLoaderConfig(),
        DummyModule(),
        object(),
        tmp_path,
    )

    with pytest.raises(
        RuntimeError,
        match="predictor failed",
    ):
        writer.setup_distributed(
            DummyDistributed(),
            DummyLogger(),
        )

    assert writer._setup is False


def test_setup_distributed_logs_training_statistics_message(
    tmp_path,
    monkeypatch,
):
    predictor = DummyPredictor(
        extract_training_vars=True,
    )
    logger = DummyLogger()

    monkeypatch.setattr(
        Writer,
        "_save_train_stats",
        lambda self: None,
    )

    writer = Writer(
        WriterConfig(
            predictor=DummyPredictorConfig(
                predictor,
            ),
        ),
        DummyLoader(),
        DummyTrainLoaderConfig(),
        DummyModule(),
        object(),
        tmp_path,
    )

    writer.setup_distributed(
        DummyDistributed(),
        logger,
    )

    assert any(
        "extract training statistics" in message for _, message in logger.messages
    )


def test_aggregate_train_stats_does_not_finalize_inactive_stat(
    tmp_path,
):
    class InactiveStat:
        sum_x = None

        def distributed_reduce(self):
            pytest.fail("Inactive stat must not be reduced.")

        def finalize(self):
            pytest.fail("Inactive stat must not be finalized.")

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)
    writer.output_dir = tmp_path
    writer.is_on_root = True
    writer.is_distributed = False

    writer.aggregate_train_stats(
        {
            "inactive": InactiveStat(),
        }
    )

    assert (
        torch.load(
            writer.train_stats_save_dir,
            weights_only=True,
        )
        == {}
    )


def test_aggregate_predictions_postprocessor_call_order(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    xr.DataArray(
        [
            [
                [1.0],
            ]
        ],
        dims=(
            "time",
            "lead_time",
            "channels",
        ),
        coords={
            "time": np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
            "lead_time": [1],
            "channels": ["prediction"],
        },
        name="prediction",
        attrs={"source": "prediction", "nstds": "NA"},
    ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

    calls = []

    class RecordingPostprocessor:
        def to_dataset(
            self,
            value,
        ):
            calls.append("to_dataset")
            return value.to_dataset(dim="channels")

        def inverse_transform(
            self,
            value,
        ):
            calls.append("inverse_transform")
            return value

        def inverse_rename(self, value):
            calls.append("inverse_rename")
            return value

    aggregate_predictions(
        RecordingPostprocessor(),
        tmp_path,
        cleanup_temp=False,
    )

    assert calls == [
        "to_dataset",
        "inverse_transform",
        "inverse_rename",
    ]


def test_aggregate_predictions_closes_loaded_year_parts(
    tmp_path,
    monkeypatch,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    xr.DataArray(
        [
            [
                [1.0],
            ]
        ],
        dims=(
            "time",
            "lead_time",
            "channels",
        ),
        coords={
            "time": np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
            "lead_time": [1],
            "channels": ["prediction"],
        },
        name="prediction",
        attrs={"source": "prediction", "nstds": "NA"},
    ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

    close_calls = []
    original_close = xr.DataArray.close

    def recording_close(self):
        close_calls.append(True)
        return original_close(self)

    monkeypatch.setattr(
        xr.DataArray,
        "close",
        recording_close,
    )

    aggregate_predictions(
        None,
        tmp_path,
        cleanup_temp=False,
    )

    assert close_calls


def test_aggregate_predictions_cleanup_only_matching_files(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    xr.DataArray(
        [
            [
                [1.0],
            ]
        ],
        dims=(
            "time",
            "lead_time",
            "channels",
        ),
        coords={
            "time": np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
            "lead_time": [1],
            "channels": ["prediction"],
        },
        name="prediction",
        attrs={"source": "prediction", "nstds": "NA"},
    ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

    unrelated = temp_dir / "keep.txt"
    unrelated.write_text("keep")

    with pytest.raises(OSError):
        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=True,
        )

    assert unrelated.exists()
    assert not (temp_dir / "prediction_rank0_0.nc").exists()


def test_aggregate_predictions_to_netcdf_nonroot_distributed_barriers(
    monkeypatch,
):
    aggregate_mock = []

    monkeypatch.setattr(
        "cccma_ppp.core.writer.aggregate_predictions",
        lambda **kwargs: aggregate_mock.append(kwargs),
    )

    distributed = DummyDistributed(
        distributed=True,
        root=False,
    )

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=1, aggregate_only=False)
    writer.distributed = distributed
    writer.is_distributed = True
    writer.is_on_root = False
    writer.output_dir = Path("/tmp")
    writer.post_processor = object()
    writer.predictor = DummyPredictor()

    writer.aggregate_predictions_to_netcdf()

    assert distributed.barrier_called == 2
    assert aggregate_mock == []


def make_writer_config_namespace(
    *,
    num_output_sampling=1,
    aggregate_only=False,
    predict_only=False,
):
    return SimpleNamespace(
        num_output_sampling=num_output_sampling,
        aggregate_only=aggregate_only,
        predict_only=predict_only,
    )


def write_prediction_file(
    path,
    *,
    values=None,
    times=None,
    lead_times=None,
    channels=None,
    source="prediction",
    nstds="NA",
    extra_dims=None,
    extra_coords=None,
):
    if times is None:
        times = ["2000-01-01"]

    if lead_times is None:
        lead_times = [1]

    if channels is None:
        channels = ["prediction"]

    dims = [
        "time",
        "lead_time",
        "channels",
    ]
    coords = {
        "time": np.asarray(
            times,
            dtype="datetime64[ns]",
        ),
        "lead_time": lead_times,
        "channels": channels,
    }

    shape = [
        len(times),
        len(lead_times),
        len(channels),
    ]

    if extra_dims is not None:
        for dim, size in extra_dims.items():
            dims.append(dim)
            shape.append(size)

    if extra_coords is not None:
        coords.update(extra_coords)

    if values is None:
        values = np.ones(
            shape,
            dtype=np.float32,
        )

    attrs = {
        "source": source,
    }

    if nstds is not None:
        attrs["nstds"] = nstds

    xr.DataArray(
        values,
        dims=tuple(dims),
        coords=coords,
        name=source,
        attrs=attrs,
    ).to_netcdf(path)


class TestAggregatePredictionFiles:
    def test_ignores_files_not_matching_rank_pattern(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            times=[
                "2000-01-01",
            ],
            source="prediction",
            nstds=None,
        )
        write_prediction_file(
            temp_dir / "other_rank0_0.nc",
            times=[
                "2001-01-01",
            ],
            source="other",
            channels=[
                "other",
            ],
            nstds=None,
        )

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=False,
        )

        assert (tmp_path / "prediction_2000.nc").exists()
        assert not (tmp_path / "prediction_2001.nc").exists()
        assert (temp_dir / "other_rank0_0.nc").exists()

    def test_creates_one_file_per_year(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            times=[
                "1999-12-31",
                "2000-01-01",
                "2001-01-01",
            ],
            source="prediction",
            nstds=None,
        )

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=False,
        )

        assert (tmp_path / "prediction_1999.nc").exists()
        assert (tmp_path / "prediction_2000.nc").exists()
        assert (tmp_path / "prediction_2001.nc").exists()

    def test_initialization_times_are_sorted(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            times=[
                "2000-03-01",
                "2000-01-01",
                "2000-02-01",
            ],
            source="prediction",
            nstds=None,
        )

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=False,
        )

        with xr.open_dataset(tmp_path / "prediction_2000.nc") as dataset:
            actual = dataset["time"].values

        expected = np.asarray(
            [
                "2000-01-01",
                "2000-02-01",
                "2000-03-01",
            ],
            dtype="datetime64[ns]",
        )

        np.testing.assert_array_equal(
            actual,
            expected,
        )

    def test_lead_times_are_sorted(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            lead_times=[
                3,
                1,
                2,
            ],
            source="prediction",
            nstds=None,
        )

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=False,
        )

        with xr.open_dataset(tmp_path / "prediction_2000.nc") as dataset:
            np.testing.assert_array_equal(
                dataset["lead_time"].values,
                np.asarray(
                    [
                        1,
                        2,
                        3,
                    ]
                ),
            )

    def test_duplicate_lead_times_are_removed(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            lead_times=[
                1,
            ],
            source="prediction",
            nstds=None,
        )
        write_prediction_file(
            temp_dir / "prediction_rank1_0.nc",
            lead_times=[
                1,
            ],
            source="prediction",
            nstds=None,
        )

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=False,
        )

        with xr.open_dataset(tmp_path / "prediction_2000.nc") as dataset:
            assert dataset.sizes["lead_time"] == 1
            assert dataset["lead_time"].item() == 1

    def test_cleanup_removes_all_matching_temp_files(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            source="prediction",
            nstds=None,
        )
        write_prediction_file(
            temp_dir / "prediction_rank1_0.nc",
            source="prediction",
            nstds=None,
        )

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=True,
        )

        assert not temp_dir.exists()
        assert (tmp_path / "prediction_2000.nc").exists()

    def test_cleanup_false_preserves_matching_temp_files(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()
        first = temp_dir / "prediction_rank0_0.nc"
        second = temp_dir / "prediction_rank1_0.nc"

        write_prediction_file(
            first,
            source="prediction",
            nstds=None,
        )
        write_prediction_file(
            second,
            source="prediction",
            nstds=None,
        )

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=False,
        )

        assert first.exists()
        assert second.exists()
        assert temp_dir.exists()

    def test_empty_initialization_times_raise_runtime_error(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        xr.DataArray(
            np.empty(
                (
                    0,
                    1,
                    1,
                )
            ),
            dims=(
                "time",
                "lead_time",
                "channels",
            ),
            coords={
                "time": np.asarray(
                    [],
                    dtype="datetime64[ns]",
                ),
                "lead_time": [1],
                "channels": [
                    "prediction",
                ],
            },
            name="prediction",
            attrs={
                "source": "prediction",
                "nstds": "NA",
            },
        ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

        with pytest.raises(
            RuntimeError,
            match="no initialization times",
        ):
            aggregate_predictions(
                None,
                tmp_path,
                cleanup_temp=False,
            )


class TestAggregatePredictionPostprocessing:
    def test_without_postprocessor_converts_channels_to_variables(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        values = np.asarray(
            [
                [
                    [
                        1.0,
                        2.0,
                    ]
                ]
            ]
        )

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            values=values,
            channels=[
                "temperature",
                "precipitation",
            ],
            source="prediction",
            nstds=None,
        )

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=False,
        )

        with xr.open_dataset(tmp_path / "prediction_2000.nc") as dataset:
            assert set(dataset.data_vars) == {
                "temperature",
                "precipitation",
            }
            assert "channels" not in dataset.dims

    def test_postprocessor_receives_each_initialization_time(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            times=[
                "2000-01-01",
                "2000-02-01",
            ],
            source="prediction",
            nstds=None,
        )

        calls = []

        class Postprocessor:
            def to_dataset(self, value):
                calls.append(
                    (
                        "to_dataset",
                        value.sizes["time"],
                    )
                )
                return value.to_dataset(dim="channels")

            def inverse_transform(self, value):
                calls.append(
                    (
                        "inverse_transform",
                        value.sizes["time"],
                    )
                )
                return value

            def inverse_rename(self, value):
                calls.append(
                    (
                        "inverse_rename",
                        value.sizes["time"],
                    )
                )
                return value

        aggregate_predictions(
            Postprocessor(),
            tmp_path,
            cleanup_temp=False,
        )

        assert calls == [
            (
                "to_dataset",
                1,
            ),
            (
                "inverse_transform",
                1,
            ),
            (
                "to_dataset",
                1,
            ),
            (
                "inverse_transform",
                1,
            ),
            (
                "inverse_rename",
                2,
            ),
        ]

    def test_inverse_rename_can_change_output_variables(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            source="prediction",
            nstds=None,
        )

        class Postprocessor:
            def to_dataset(self, value):
                return value.to_dataset(dim="channels")

            def inverse_transform(self, value):
                return value

            def inverse_rename(self, value):
                return value.rename(
                    {
                        "prediction": "renamed",
                    }
                )

        aggregate_predictions(
            Postprocessor(),
            tmp_path,
            cleanup_temp=False,
        )

        with xr.open_dataset(tmp_path / "prediction_2000.nc") as dataset:
            assert "renamed" in dataset
            assert "prediction" not in dataset


class TestAggregatePredictionOptionalDimensions:
    def test_optional_ensemble_coordinate_is_sorted(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        values = np.asarray(
            [
                [
                    [
                        [
                            30.0,
                            10.0,
                            20.0,
                        ]
                    ]
                ]
            ]
        )

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            values=values,
            source="prediction",
            nstds=None,
            extra_dims={
                "ensembles": 3,
            },
            extra_coords={
                "ensembles": [
                    3,
                    1,
                    2,
                ],
            },
        )

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=False,
            optional_sample_dimensions=("ensembles",),
        )

        with xr.open_dataset(tmp_path / "prediction_2000.nc") as dataset:
            np.testing.assert_array_equal(
                dataset["ensembles"].values,
                np.asarray(
                    [
                        1,
                        2,
                        3,
                    ]
                ),
            )

    def test_all_nan_optional_samples_are_dropped(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        values = np.asarray(
            [
                [
                    [
                        [
                            1.0,
                            np.nan,
                            3.0,
                        ]
                    ]
                ]
            ]
        )

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            values=values,
            source="prediction",
            nstds=None,
            extra_dims={
                "ensembles": 3,
            },
            extra_coords={
                "ensembles": [
                    1,
                    2,
                    3,
                ],
            },
        )

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=False,
            optional_sample_dimensions=("ensembles",),
        )

        with xr.open_dataset(tmp_path / "prediction_2000.nc") as dataset:
            np.testing.assert_array_equal(
                dataset["ensembles"].values,
                np.asarray(
                    [
                        1,
                        3,
                    ]
                ),
            )


class TestWriterSetupCompletion:
    def test_setup_logs_start_and_completion(
        self,
        tmp_path,
    ):
        logger = DummyLogger()
        writer = Writer(
            WriterConfig(predictor=DummyPredictorConfig(DummyPredictor())),
            DummyLoader(),
            DummyTrainLoaderConfig(),
            DummyModule(),
            object(),
            tmp_path,
        )

        writer.setup_distributed(
            DummyDistributed(),
            logger,
        )

        messages = [message for _, message in logger.messages]

        assert "Setting up writer." in messages
        assert "Writer setup complete." in messages

    def test_setup_logs_distributed_metadata(
        self,
        tmp_path,
    ):
        logger = DummyLogger()
        distributed = DummyDistributed(
            distributed=True,
        )
        distributed.rank = 2
        distributed.local_rank = 1
        distributed.world_size = 4

        writer = Writer(
            WriterConfig(predictor=DummyPredictorConfig(DummyPredictor())),
            DummyLoader(),
            DummyTrainLoaderConfig(),
            DummyModule(),
            object(),
            tmp_path,
        )

        writer.setup_distributed(
            distributed,
            logger,
        )

        assert any(
            "rank=2" in message
            and "local_rank=1" in message
            and "world_size=4" in message
            for _, message in logger.messages
        )

    def test_training_stats_failure_prevents_setup_completion(
        self,
        tmp_path,
        monkeypatch,
    ):
        predictor = DummyPredictor(
            extract_training_vars=True,
        )

        monkeypatch.setattr(
            Writer,
            "_save_train_stats",
            Mock(side_effect=RuntimeError("statistics failed")),
        )

        writer = Writer(
            WriterConfig(predictor=DummyPredictorConfig(predictor)),
            DummyLoader(),
            DummyTrainLoaderConfig(),
            DummyModule(),
            object(),
            tmp_path,
        )

        with pytest.raises(
            RuntimeError,
            match="statistics failed",
        ):
            writer.setup_distributed(
                DummyDistributed(),
                DummyLogger(),
            )

        assert writer._setup is False


class TestAdditionalWriterConfig:
    @pytest.mark.parametrize(
        "num_output_sampling",
        [
            -1,
            -10,
        ],
    )
    def test_negative_output_sampling_is_rejected(
        self,
        num_output_sampling,
    ):
        with pytest.raises(
            ValueError,
            match="cannot be negative",
        ):
            WriterConfig(
                predictor=DummyPredictorConfig(
                    DummyPredictor(),
                ),
                num_output_sampling=num_output_sampling,
            )

    def test_zero_output_sampling_is_accepted(self):
        config = WriterConfig(
            predictor=DummyPredictorConfig(
                DummyPredictor(),
            ),
            num_output_sampling=0,
        )

        assert config.num_output_sampling == 0

    def test_training_statistics_default_is_false(self):
        config = WriterConfig(
            predictor=DummyPredictorConfig(
                DummyPredictor(),
            ),
        )

        assert config.get_trained_model_stats_from_validation is False

    def test_build_forwards_exact_writer_arguments(
        self,
        tmp_path,
        monkeypatch,
    ):
        config = WriterConfig(
            predictor=DummyPredictorConfig(
                DummyPredictor(),
            ),
        )
        loader = DummyLoader()
        train_config = DummyTrainLoaderConfig()
        module = DummyModule()
        postprocessor = object()
        built = object()

        constructor = Mock(
            return_value=built,
        )
        monkeypatch.setattr(
            "cccma_ppp.core.writer.Writer",
            constructor,
        )

        result = config.build(
            inference_data_loader=loader,
            train_dataloader_config=train_config,
            module=module,
            post_processor=postprocessor,
            output_dir=str(tmp_path),
        )

        assert result is built
        constructor.assert_called_once_with(
            config=config,
            inference_data_loader=loader,
            train_dataloader_config=train_config,
            module=module,
            post_processor=postprocessor,
            output_dir=str(tmp_path),
        )


class TestAdditionalWriterPredictionLoop:
    def test_predict_loop_processes_every_batch(
        self,
        monkeypatch,
    ):
        batches = [
            DummyBatch(),
            DummyBatch(),
            DummyBatch(),
        ]
        loader = RecordingLoader(batches)
        predictor = DummyPredictor()

        writer = make_predict_writer(
            loader=loader,
            predictor=predictor,
        )
        writer.aggregate_predictions_to_netcdf = Mock()

        clear = Mock()
        monkeypatch.setattr(
            "cccma_ppp.core.writer.clear_memory",
            clear,
        )

        writer._predict()

        assert writer.module.eval_called is True
        assert loader.iterations == 1
        assert predictor.infer_calls == [
            False,
            False,
            False,
        ]

        for batch in batches:
            assert batch.device == torch.device("cpu")

        clear.assert_called_once_with()
        writer.aggregate_predictions_to_netcdf.assert_called_once_with(
            True,
        )

    def test_predict_loop_uses_training_loader_for_latent_output(
        self,
        monkeypatch,
    ):
        inference_loader = RecordingLoader(
            [
                DummyBatch(),
            ]
        )
        training_batches = [
            DummyBatch(),
            DummyBatch(),
        ]
        training_loader = RecordingLoader(
            training_batches,
        )
        predictor = DummyPredictor(
            save_latent=True,
        )

        writer = make_predict_writer(
            loader=inference_loader,
            predictor=predictor,
        )
        writer.build_train_loader = Mock(
            return_value=training_loader,
        )
        writer.aggregate_predictions_to_netcdf = Mock()

        monkeypatch.setattr(
            "cccma_ppp.core.writer.clear_memory",
            Mock(),
        )

        writer._predict()

        writer.build_train_loader.assert_called_once_with(
            return_metadata=True,
            shuffle=False,
        )
        assert inference_loader.iterations == 0
        assert training_loader.iterations == 1
        assert predictor.infer_calls == [
            False,
            False,
        ]
        writer.aggregate_predictions_to_netcdf.assert_called_once_with(
            False,
        )

    def test_predict_loop_disables_progress_bar_on_nonroot(
        self,
        monkeypatch,
    ):
        writer = make_predict_writer(
            is_on_root=False,
        )
        writer.aggregate_predictions_to_netcdf = Mock()

        tqdm_mock = Mock(
            side_effect=lambda iterable, **kwargs: iterable,
        )
        monkeypatch.setattr(
            "cccma_ppp.core.writer.tqdm",
            tqdm_mock,
        )
        monkeypatch.setattr(
            "cccma_ppp.core.writer.clear_memory",
            Mock(),
        )

        writer._predict()

        assert tqdm_mock.call_args.kwargs["disable"] is True
        assert tqdm_mock.call_args.kwargs["desc"] == "Inference"

    def test_predict_loop_enables_progress_bar_on_root(
        self,
        monkeypatch,
    ):
        writer = make_predict_writer(
            is_on_root=True,
        )
        writer.aggregate_predictions_to_netcdf = Mock()

        tqdm_mock = Mock(
            side_effect=lambda iterable, **kwargs: iterable,
        )
        monkeypatch.setattr(
            "cccma_ppp.core.writer.tqdm",
            tqdm_mock,
        )
        monkeypatch.setattr(
            "cccma_ppp.core.writer.clear_memory",
            Mock(),
        )

        writer._predict()

        assert tqdm_mock.call_args.kwargs["disable"] is False
        assert tqdm_mock.call_args.kwargs["desc"] == "Inference"

    def test_predict_does_not_log_completion_after_failure(
        self,
        monkeypatch,
    ):
        writer = object.__new__(Writer)
        writer._setup = True
        writer._predict = Mock(
            side_effect=RuntimeError("prediction failed"),
        )

        messages = []
        writer.log_root = lambda level, message: messages.append(message)

        monkeypatch.setattr(
            "cccma_ppp.core.writer.clear_memory",
            Mock(),
        )
        monkeypatch.setattr(
            "cccma_ppp.core.writer.time.time",
            Mock(return_value=10.0),
        )

        with pytest.raises(
            RuntimeError,
            match="prediction failed",
        ):
            writer.predict()

        assert messages == [
            "Starting Inference Loop...",
        ]


class TestAdditionalWriterSetup:
    def test_setup_does_not_extract_statistics_when_disabled(
        self,
        tmp_path,
        monkeypatch,
    ):
        predictor = DummyPredictor(
            extract_training_vars=False,
        )
        save_stats = Mock()

        monkeypatch.setattr(
            Writer,
            "_save_train_stats",
            save_stats,
        )

        writer = Writer(
            WriterConfig(
                predictor=DummyPredictorConfig(
                    predictor,
                ),
            ),
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

        save_stats.assert_not_called()
        assert writer._setup is True

    def test_setup_root_creates_existing_temp_directory_safely(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()
        marker = temp_dir / "existing.txt"
        marker.write_text("keep")

        writer = Writer(
            WriterConfig(
                predictor=DummyPredictorConfig(
                    DummyPredictor(),
                ),
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

        assert temp_dir.is_dir()
        assert marker.read_text() == "keep"

    def test_setup_distributed_barrier_precedes_predictor_build(
        self,
        tmp_path,
    ):
        events = []

        class DistributedWithEvents(DummyDistributed):
            def barrier(self):
                events.append("barrier")
                super().barrier()

        class PredictorConfigWithEvents(DummyPredictorConfig):
            def build(
                self,
                module,
                distributed,
                output_dir,
                num_output_sampling,
            ):
                events.append("build")
                return super().build(
                    module,
                    distributed,
                    output_dir,
                    num_output_sampling,
                )

        distributed = DistributedWithEvents(
            distributed=True,
        )
        predictor_config = PredictorConfigWithEvents(
            DummyPredictor(),
        )

        writer = Writer(
            WriterConfig(
                predictor=predictor_config,
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

        assert events == [
            "barrier",
            "build",
        ]


class TestAdditionalTrainStatistics:
    def test_save_train_stats_uses_initial_predictor_stats_for_empty_loader(
        self,
        tmp_path,
        monkeypatch,
    ):
        initial_stats = {
            "initial": object(),
        }
        predictor = DummyPredictor()
        predictor.stats = initial_stats

        writer = object.__new__(Writer)
        writer.output_dir = tmp_path
        writer.device = torch.device("cpu")
        writer.is_on_root = True
        writer.is_distributed = False
        writer.predictor = predictor
        writer.config = SimpleNamespace(
            get_trained_model_stats_from_validation=False,
        )
        writer.build_train_loader = Mock(
            return_value=[],
        )
        writer.aggregate_train_stats = Mock()

        clear = Mock()
        monkeypatch.setattr(
            "cccma_ppp.core.writer.clear_memory",
            clear,
        )

        writer._save_train_stats()

        writer.aggregate_train_stats.assert_called_once_with(
            initial_stats,
        )
        clear.assert_called_once_with()

    def test_save_train_stats_uses_stats_from_last_batch(
        self,
        tmp_path,
        monkeypatch,
    ):
        first_stats = {
            "first": object(),
        }
        second_stats = {
            "second": object(),
        }

        predictor = DummyPredictor()
        predictor._infer_on_batch = Mock(
            side_effect=[
                first_stats,
                second_stats,
            ]
        )

        batches = [
            DummyBatch(),
            DummyBatch(),
        ]

        writer = object.__new__(Writer)
        writer.output_dir = tmp_path
        writer.device = torch.device("cpu")
        writer.is_on_root = True
        writer.is_distributed = False
        writer.predictor = predictor
        writer.config = SimpleNamespace(
            get_trained_model_stats_from_validation=False,
        )
        writer.build_train_loader = Mock(
            return_value=batches,
        )
        writer.aggregate_train_stats = Mock()

        monkeypatch.setattr(
            "cccma_ppp.core.writer.clear_memory",
            Mock(),
        )

        writer._save_train_stats()

        assert predictor._infer_on_batch.call_count == 2
        writer.aggregate_train_stats.assert_called_once_with(
            second_stats,
        )

    def test_save_train_stats_existing_file_still_barriers(
        self,
        tmp_path,
    ):
        distributed = DummyDistributed(
            distributed=True,
        )

        writer = object.__new__(Writer)
        writer.output_dir = tmp_path
        writer.is_distributed = True
        writer.is_on_root = False
        writer.distributed = distributed
        writer.build_train_loader = Mock()

        torch.save(
            {},
            writer.train_stats_save_dir,
        )

        writer._save_train_stats()

        writer.build_train_loader.assert_not_called()
        assert distributed.barrier_called == 1

    def test_aggregate_train_stats_propagates_finalize_error(
        self,
        tmp_path,
    ):
        class FailingStat:
            sum_x = torch.tensor(
                [
                    1.0,
                ]
            )

            def distributed_reduce(self):
                pass

            def finalize(self):
                raise RuntimeError("finalize failed")

        writer = object.__new__(Writer)
        writer.output_dir = tmp_path
        writer.is_on_root = True
        writer.is_distributed = False

        with pytest.raises(
            RuntimeError,
            match="finalize failed",
        ):
            writer.aggregate_train_stats(
                {
                    "failing": FailingStat(),
                }
            )

        assert not writer.train_stats_save_dir.exists()


class TestAdditionalAggregationDispatch:
    def make_writer(
        self,
        *,
        predictor,
        num_output_sampling,
        post_processor=None,
        distributed=False,
        root=True,
    ):
        writer = object.__new__(Writer)
        writer.config = SimpleNamespace(
            num_output_sampling=num_output_sampling,
        )
        writer.predictor = predictor
        writer.post_processor = post_processor
        writer.output_dir = Path("/tmp/output")
        writer.is_distributed = distributed
        writer.is_on_root = root
        writer.log_root = Mock()

        if distributed:
            writer.distributed = DummyDistributed(
                distributed=True,
                root=root,
            )

        return writer

    def test_prediction_dispatch_forwards_exact_arguments(
        self,
        monkeypatch,
    ):
        predictor = SimpleNamespace(
            save_latent=False,
        )
        postprocessor = object()
        writer = self.make_writer(
            predictor=predictor,
            num_output_sampling=0,
            post_processor=postprocessor,
        )

        aggregate = Mock()
        monkeypatch.setattr(
            "cccma_ppp.core.writer.aggregate_predictions",
            aggregate,
        )

        writer.aggregate_predictions_to_netcdf()

        aggregate.assert_called_once_with(
            post_processor=postprocessor,
            output_dir=Path("/tmp/output"),
            load_naming_convention="prediction",
            save_naming_convention="prediction",
            logger_function=writer.log_root,
        )

    def test_output_ensemble_suffix_is_added(
        self,
        monkeypatch,
    ):
        predictor = SimpleNamespace(
            save_latent=False,
        )
        writer = self.make_writer(
            predictor=predictor,
            num_output_sampling=3,
        )

        aggregate = Mock()
        monkeypatch.setattr(
            "cccma_ppp.core.writer.aggregate_predictions",
            aggregate,
        )

        writer.aggregate_predictions_to_netcdf()

        assert (
            aggregate.call_args.kwargs["save_naming_convention"]
            == "prediction_output_ensemble"
        )

    def test_nstds_suffix_is_added(
        self,
        monkeypatch,
    ):
        predictor = SimpleNamespace(
            save_latent=False,
            nstds=1.5,
        )
        writer = self.make_writer(
            predictor=predictor,
            num_output_sampling=0,
        )

        aggregate = Mock()
        monkeypatch.setattr(
            "cccma_ppp.core.writer.aggregate_predictions",
            aggregate,
        )

        writer.aggregate_predictions_to_netcdf()

        assert (
            aggregate.call_args.kwargs["save_naming_convention"] == "prediction_1.5stds"
        )

    def test_nstds_and_output_ensemble_suffixes_are_combined(
        self,
        monkeypatch,
    ):
        predictor = SimpleNamespace(
            save_latent=False,
            nstds=2.0,
        )
        writer = self.make_writer(
            predictor=predictor,
            num_output_sampling=4,
        )

        aggregate = Mock()
        monkeypatch.setattr(
            "cccma_ppp.core.writer.aggregate_predictions",
            aggregate,
        )

        writer.aggregate_predictions_to_netcdf()

        assert (
            aggregate.call_args.kwargs["save_naming_convention"]
            == "prediction_2.0stds_output_ensemble"
        )

    def test_latent_dispatch_uses_latent_load_name(
        self,
        monkeypatch,
    ):
        predictor = SimpleNamespace(
            save_latent=True,
        )
        writer = self.make_writer(
            predictor=predictor,
            num_output_sampling=0,
        )

        aggregate = Mock()
        monkeypatch.setattr(
            "cccma_ppp.core.writer.aggregate_predictions",
            aggregate,
        )

        writer.aggregate_predictions_to_netcdf(
            do_post_process=False,
        )

        assert aggregate.call_args.kwargs["load_naming_convention"] == "latent"
        assert aggregate.call_args.kwargs["save_naming_convention"] == "latent"
        assert aggregate.call_args.kwargs["post_processor"] is None

    def test_nonroot_distributed_process_does_not_aggregate(
        self,
        monkeypatch,
    ):
        predictor = SimpleNamespace(
            save_latent=False,
        )
        writer = self.make_writer(
            predictor=predictor,
            num_output_sampling=0,
            distributed=True,
            root=False,
        )

        aggregate = Mock()
        monkeypatch.setattr(
            "cccma_ppp.core.writer.aggregate_predictions",
            aggregate,
        )

        writer.aggregate_predictions_to_netcdf()

        aggregate.assert_not_called()
        assert writer.distributed.barrier_called == 2


class TestAdditionalAggregatePredictions:
    def test_custom_save_naming_convention_is_used(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            source="prediction",
            nstds=None,
        )

        aggregate_predictions(
            None,
            tmp_path,
            save_naming_convention="custom",
            cleanup_temp=False,
        )

        assert (tmp_path / "custom_2000.nc").exists()
        assert not (tmp_path / "prediction_2000.nc").exists()

    def test_custom_load_naming_convention_selects_matching_files(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "latent_rank0_0.nc",
            source="latent",
            channels=[
                "latent",
            ],
            nstds=None,
        )
        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            times=[
                "2001-01-01",
            ],
            source="prediction",
            nstds=None,
        )

        aggregate_predictions(
            None,
            tmp_path,
            load_naming_convention="latent",
            cleanup_temp=False,
        )

        assert (tmp_path / "latent_2000.nc").exists()
        assert not (tmp_path / "latent_2001.nc").exists()

    def test_duplicate_initialization_times_are_merged(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            times=[
                "2000-01-01",
            ],
            lead_times=[
                1,
            ],
            source="prediction",
            nstds=None,
        )
        write_prediction_file(
            temp_dir / "prediction_rank1_0.nc",
            times=[
                "2000-01-01",
            ],
            lead_times=[
                2,
            ],
            source="prediction",
            nstds=None,
        )

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=False,
        )

        with xr.open_dataset(tmp_path / "prediction_2000.nc") as dataset:
            assert dataset.sizes["time"] == 1
            np.testing.assert_array_equal(
                dataset["lead_time"].values,
                np.asarray(
                    [
                        1,
                        2,
                    ]
                ),
            )

    def test_cleanup_removes_empty_temp_directory(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        temporary_file = temp_dir / "prediction_rank0_0.nc"
        write_prediction_file(
            temporary_file,
            source="prediction",
            nstds=None,
        )

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=True,
        )

        assert not temporary_file.exists()
        assert not temp_dir.exists()

    def test_postprocessor_failure_is_propagated(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            source="prediction",
            nstds=None,
        )

        class FailingPostprocessor:
            def to_dataset(self, value):
                raise RuntimeError("postprocessing failed")

        with pytest.raises(
            RuntimeError,
            match="postprocessing failed",
        ):
            aggregate_predictions(
                FailingPostprocessor(),
                tmp_path,
                cleanup_temp=False,
            )

    def test_saved_logger_message_contains_output_path(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            source="prediction",
            nstds=None,
        )

        calls = []

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=False,
            logger_function=lambda level, message: calls.append(
                (
                    level,
                    message,
                )
            ),
        )

        output_path = tmp_path / "prediction_2000.nc"

        assert calls[0] == (
            logging.INFO,
            "Aggregating temporary prediction files year-by-year.",
        )
        assert calls[-1][0] == logging.INFO
        assert str(output_path) in calls[-1][1]

    def test_multiple_years_are_logged_individually(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        write_prediction_file(
            temp_dir / "prediction_rank0_0.nc",
            times=[
                "1999-01-01",
                "2000-01-01",
                "2001-01-01",
            ],
            source="prediction",
            nstds=None,
        )

        messages = []

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=False,
            logger_function=lambda level, message: messages.append(message),
        )

        saved_messages = [
            message
            for message in messages
            if message.startswith("Saved aggregated predictions")
        ]

        assert len(saved_messages) == 3
        assert any("1999" in message for message in saved_messages)
        assert any("2000" in message for message in saved_messages)
        assert any("2001" in message for message in saved_messages)

    def test_custom_time_and_lead_dimension_names(
        self,
        tmp_path,
    ):
        temp_dir = tmp_path / "_temp"
        temp_dir.mkdir()

        xr.DataArray(
            np.asarray(
                [
                    [
                        [
                            3.0,
                        ],
                        [
                            1.0,
                        ],
                        [
                            2.0,
                        ],
                    ]
                ]
            ),
            dims=(
                "initialization",
                "forecast_step",
                "channels",
            ),
            coords={
                "initialization": np.asarray(
                    [
                        "2000-01-01",
                    ],
                    dtype="datetime64[ns]",
                ),
                "forecast_step": [
                    3,
                    1,
                    2,
                ],
                "channels": [
                    "prediction",
                ],
            },
            name="prediction",
        ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

        aggregate_predictions(
            None,
            tmp_path,
            cleanup_temp=False,
            init_time_dim="initialization",
            lead_time_dim="forecast_step",
            optional_sample_dimensions=(),
        )

        with xr.open_dataset(tmp_path / "prediction_2000.nc") as dataset:
            np.testing.assert_array_equal(
                dataset["forecast_step"].values,
                np.asarray(
                    [
                        1,
                        2,
                        3,
                    ]
                ),
            )
            assert "initialization" in dataset.dims

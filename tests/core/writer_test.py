import logging
from pathlib import Path
from types import SimpleNamespace

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


def test_log_root_logger_branch():
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=0)

    writer.is_on_root = True
    writer.logger = DummyLogger()

    writer.log_root(logging.INFO, "hello")

    assert len(writer.logger.messages) == 1


def test_log_root_print_branch(capsys):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=0)

    writer.is_on_root = True
    writer.logger = None

    writer.log_root(logging.INFO, "hello")

    assert "hello" in capsys.readouterr().out


def test_log_root_non_root_noop(capsys):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=0)

    writer.is_on_root = False
    writer.logger = None

    writer.log_root(logging.INFO, "hello")

    assert capsys.readouterr().out == ""


def test_raw_module_normal():
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=0)

    module = object()

    writer.module = module

    assert writer.raw_module is module


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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)
    writer.module = FakeDDP()

    assert writer.raw_module is writer.module.module


def test_save_train_stats_file_exists_skips_loader(
    tmp_path,
):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
            kwargs["naming_convention"],
        ),
    )

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=0)

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
        [[[1.0]]],
        dims=("time", "lead_time", "channels"),
        coords={
            "time": np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
            "lead_time": [1],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

    xr.DataArray(
        [[[2.0]]],
        dims=("time", "lead_time", "channels"),
        coords={
            "time": np.asarray(["2001-01-01"], dtype="datetime64[ns]"),
            "lead_time": [1],
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


def test_aggregate_predictions_postprocessor_branch(
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


def test_aggregate_train_stats_root_skip_none_stat(
    tmp_path,
):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)

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
            kwargs["naming_convention"],
        ),
    )

    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=0)

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
        match="time",
    ):
        aggregate_predictions(
            None,
            tmp_path,
        )


def test_aggregate_predictions_multiple_years(
    tmp_path,
):
    temp = tmp_path / "_temp"
    temp.mkdir()

    xr.DataArray(
        [[[1.0]]],
        dims=("time", "lead_time", "channels"),
        coords={
            "time": np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
            "lead_time": [1],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp / "prediction_rank0_0.nc")

    xr.DataArray(
        [[[2.0]]],
        dims=("time", "lead_time", "channels"),
        coords={
            "time": np.asarray(["2001-01-01"], dtype="datetime64[ns]"),
            "lead_time": [1],
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


def test_aggregate_predictions_year_missing_from_one_file(
    tmp_path,
):
    temp = tmp_path / "_temp"
    temp.mkdir()

    xr.DataArray(
        [[[1.0]]],
        dims=("time", "lead_time", "channels"),
        coords={
            "time": np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
            "lead_time": [1],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp / "prediction_rank0_0.nc")

    xr.DataArray(
        [[[2.0]]],
        dims=("time", "lead_time", "channels"),
        coords={
            "time": np.asarray(["2001-01-01"], dtype="datetime64[ns]"),
            "lead_time": [1],
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


def test_predict_logs_elapsed_time(
    monkeypatch,
):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=0)
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
    writer.config = SimpleNamespace(num_output_sampling=0)

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


def test_save_train_stats_calls_gc_collect(
    tmp_path,
    monkeypatch,
):
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=0)

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
    writer.config = SimpleNamespace(num_output_sampling=0)
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
    writer.config = SimpleNamespace(num_output_sampling=0)
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
    writer.config = SimpleNamespace(num_output_sampling=0)
    writer.output_dir = tmp_path
    writer.is_on_root = False
    writer.is_distributed = True
    writer.distributed = distributed

    writer.aggregate_train_stats({})

    assert distributed.barrier_called == 1


def test_log_root_passes_formatting_arguments():
    writer = object.__new__(Writer)
    writer.config = SimpleNamespace(num_output_sampling=0)
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
        [[[1.0]]],
        dims=("time", "lead_time", "channels"),
        coords={
            "time": np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
            "lead_time": [1],
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


def test_aggregate_predictions_custom_naming_convention(
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


def test_aggregate_predictions_ignores_unrelated_temp_files(
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
    ).to_netcdf(
        temp_dir / "prediction_rank0_0.nc",
    )

    unrelated = temp_dir / "other_rank0_0.nc"

    xr.DataArray(
        [[[2.0]]],
        dims=("time", "lead_time", "channels"),
        coords={
            "time": np.asarray(["2001-01-01"], dtype="datetime64[ns]"),
            "lead_time": [1],
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
    writer.config = SimpleNamespace(num_output_sampling=0)

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


def test_writer_config_accepts_zero_output_sampling():
    config = WriterConfig(
        predictor=object(),
        num_output_sampling=0,
    )

    assert config.num_output_sampling == 0


def test_writer_config_accepts_positive_output_sampling():
    config = WriterConfig(
        predictor=object(),
        num_output_sampling=7,
    )

    assert config.num_output_sampling == 7


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
    writer.config = SimpleNamespace(num_output_sampling=0)
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
    writer.config = SimpleNamespace(num_output_sampling=0)
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


def test_predict_loop_evaluates_module_and_moves_batch():
    batch = DummyBatch()
    predictor = DummyPredictor()
    loader = RecordingLoader([batch])

    writer = make_predict_writer(
        loader=loader,
        predictor=predictor,
    )

    aggregated = []
    writer.aggregate_predictions_to_netcdf = lambda value: aggregated.append(value)

    writer._predict()

    assert writer.module.eval_called is True
    assert batch.device == torch.device("cpu")
    assert predictor.infer_calls == [False]
    assert aggregated == [True]


def test_predict_loop_processes_every_batch():
    batches = [
        DummyBatch(),
        DummyBatch(),
        DummyBatch(),
    ]
    predictor = DummyPredictor()

    writer = make_predict_writer(
        loader=RecordingLoader(batches),
        predictor=predictor,
    )
    writer.aggregate_predictions_to_netcdf = lambda value: None

    writer._predict()

    assert predictor.infer_calls == [
        False,
        False,
        False,
    ]

    for batch in batches:
        assert batch.device == torch.device("cpu")


def test_predict_loop_latent_uses_training_loader():
    inference_loader = RecordingLoader(
        [
            DummyBatch(),
        ]
    )
    training_loader = RecordingLoader(
        [
            DummyBatch(),
            DummyBatch(),
        ]
    )
    predictor = DummyPredictor(
        save_latent=True,
    )

    writer = make_predict_writer(
        loader=inference_loader,
        predictor=predictor,
    )

    captured = {}

    def build_train_loader(**kwargs):
        captured["kwargs"] = kwargs
        return training_loader

    writer.build_train_loader = build_train_loader
    writer.aggregate_predictions_to_netcdf = lambda value: captured.setdefault(
        "postprocess",
        value,
    )

    writer._predict()

    assert captured["kwargs"] == {
        "return_metadata": True,
        "shuffle": False,
    }
    assert captured["postprocess"] is False
    assert inference_loader.iterations == 0
    assert training_loader.iterations == 1
    assert predictor.infer_calls == [
        False,
        False,
    ]


def test_predict_loop_nonlatent_uses_inference_loader():
    inference_loader = RecordingLoader(
        [
            DummyBatch(),
        ]
    )
    predictor = DummyPredictor(
        save_latent=False,
    )

    writer = make_predict_writer(
        loader=inference_loader,
        predictor=predictor,
    )

    writer.build_train_loader = lambda **kwargs: pytest.fail(
        "Training loader should not be built."
    )

    captured = []
    writer.aggregate_predictions_to_netcdf = lambda value: captured.append(value)

    writer._predict()

    assert inference_loader.iterations == 1
    assert captured == [True]


def test_predict_loop_aggregates_after_all_batches():
    events = []

    class Predictor(DummyPredictor):
        def _infer_on_batch(
            self,
            batch,
            _getting_train_stats=False,
        ):
            events.append("infer")
            return super()._infer_on_batch(
                batch,
                _getting_train_stats=_getting_train_stats,
            )

    writer = make_predict_writer(
        loader=RecordingLoader(
            [
                DummyBatch(),
                DummyBatch(),
            ]
        ),
        predictor=Predictor(),
    )
    writer.aggregate_predictions_to_netcdf = lambda value: events.append("aggregate")

    writer._predict()

    assert events == [
        "infer",
        "infer",
        "aggregate",
    ]


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
    writer.config = SimpleNamespace(num_output_sampling=0)
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


def test_aggregate_predictions_sorts_lead_times(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    xr.DataArray(
        [
            [
                [20.0],
                [10.0],
            ]
        ],
        dims=(
            "time",
            "lead_time",
            "channels",
        ),
        coords={
            "time": np.asarray(["2000-01-01"], dtype="datetime64[ns]"),
            "lead_time": [2, 1],
            "channels": ["prediction"],
        },
        name="prediction",
    ).to_netcdf(temp_dir / "prediction_rank0_0.nc")

    aggregate_predictions(
        None,
        tmp_path,
        cleanup_temp=False,
    )

    with xr.open_dataset(tmp_path / "prediction_2000.nc") as dataset:
        assert dataset["lead_time"].values.tolist() == [
            1,
            2,
        ]


def test_aggregate_predictions_removes_duplicate_lead_times(
    tmp_path,
):
    temp_dir = tmp_path / "_temp"
    temp_dir.mkdir()

    for rank, value in enumerate(
        [
            1.0,
            2.0,
        ]
    ):
        xr.DataArray(
            [
                [
                    [value],
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
        ).to_netcdf(temp_dir / f"prediction_rank{rank}_0.nc")

    aggregate_predictions(
        None,
        tmp_path,
        cleanup_temp=False,
    )

    with xr.open_dataset(tmp_path / "prediction_2000.nc") as dataset:
        assert dataset.sizes["lead_time"] == 1
        assert dataset["lead_time"].values.tolist() == [1]


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

    aggregate_predictions(
        RecordingPostprocessor(),
        tmp_path,
        cleanup_temp=False,
    )

    assert calls == [
        "to_dataset",
        "inverse_transform",
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
    writer.config = SimpleNamespace(num_output_sampling=0)
    writer.distributed = distributed
    writer.is_distributed = True
    writer.is_on_root = False
    writer.output_dir = Path("/tmp")
    writer.post_processor = object()
    writer.predictor = DummyPredictor()

    writer.aggregate_predictions_to_netcdf()

    assert distributed.barrier_called == 2
    assert aggregate_mock == []

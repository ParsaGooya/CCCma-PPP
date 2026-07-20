from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.inference.inference_configs import (
    InferenceConfig,
    build_writer,
    prepare_config,
)


class DummyDistributed:
    def __init__(
        self,
        rank=0,
        world_size=1,
        root=True,
        device="cpu",
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self._root = root
        self.barrier_called = False
        self.barrier_calls = 0

    def is_root(self):
        return self._root

    def barrier(self):
        self.barrier_called = True
        self.barrier_calls += 1


class DummyInferenceLoaderConfig:
    def __init__(
        self,
        dataset_config=None,
        input_metadata=None,
    ):
        self.dataset_config = dataset_config
        self._input_metadata = {"tas": {}} if input_metadata is None else input_metadata
        self.read_called = False
        self.read_arg = None
        self.setup_called = False
        self.setup_args = None
        self.build_called = False
        self.loader = DummyInferenceLoader()

    @property
    def input_var_metadata(self):
        return self._input_metadata

    def read_datasetConfig_from_train(self, train_dataset_config):
        self.read_called = True
        self.read_arg = train_dataset_config
        self.dataset_config = object()

    def setup_distributed(self, train_loader, distributed):
        self.setup_called = True
        self.setup_args = (train_loader, distributed)

    def build_inference_loader(self):
        self.build_called = True
        return self.loader


class DummyInferenceLoader:
    def __init__(
        self,
        input_shape=(2, 3),
        output_shape=(2, 3),
        added_features_dim=0,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.added_features_dim = added_features_dim


class DummyTrainDatasetConfig:
    pass


class DummyTrainLoaderConfig:
    def __init__(
        self,
        input_metadata=None,
    ):
        self.dataset_config = DummyTrainDatasetConfig()
        self._input_metadata = {"tas": {}} if input_metadata is None else input_metadata

    @property
    def input_var_metadata(self):
        return self._input_metadata


class DummyModule:
    def __init__(self):
        self.loaded_state = None
        self.loaded_strict = None
        self.device = None

    def load_state_dict(self, state, strict=True):
        self.loaded_state = state
        self.loaded_strict = strict
        return SimpleNamespace(
            missing_keys=[],
            unexpected_keys=[],
        )

    def to(self, device):
        self.device = device
        return self


class DummySelector:
    def __init__(self):
        self.build_kwargs = None
        self.module = DummyModule()

    def build_module(self, **kwargs):
        self.build_kwargs = kwargs
        return self.module


class DummyWriter:
    pass


class DummyWriterConfig:
    def __init__(self):
        self.build_called = False
        self.build_kwargs = None
        self.writer = DummyWriter()

    def build(self, **kwargs):
        self.build_called = True
        self.build_kwargs = kwargs
        return self.writer


def make_bare_config(
    tmp_path,
    save_path=None,
    seed=None,
    checkpoint_name=None,
    inference_dataset_config=None,
    inference_metadata=None,
    train_metadata=None,
):
    config = object.__new__(InferenceConfig)
    config.experiment_dir = Path(tmp_path)
    config.save_path = Path(save_path) if save_path is not None else None
    config.seed = seed
    config.checkpoint_name = checkpoint_name
    config.writer = DummyWriterConfig()
    config.inference_loader = DummyInferenceLoaderConfig(
        dataset_config=inference_dataset_config,
        input_metadata=inference_metadata,
    )
    config.train_loader = DummyTrainLoaderConfig(
        input_metadata=train_metadata,
    )
    config.train_config = {
        "train_loader": {
            "dataset_config": {},
        },
        "module": {
            "type": "deterministic",
        },
    }
    return config


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train_loader": {
                    "dataset_config": {
                        "model": {
                            "names": ["tas"],
                        }
                    }
                },
                "module": {
                    "type": "deterministic",
                },
            }
        )
    )
    return path


@pytest.mark.pruned
# Remove test due to no coverage
def test_prepare_config_reads_yaml(config_file):
    result = prepare_config(config_file)

    assert result["module"]["type"] == "deterministic"
    assert result["train_loader"]["dataset_config"]["model"]["names"] == ["tas"]


@pytest.mark.pruned
# Remove test due to no coverage
def test_prepare_config_accepts_string_path(config_file):
    result = prepare_config(str(config_file))

    assert isinstance(result, dict)


@pytest.mark.pruned
# Remove test due to no coverage
def test_prepare_config_empty_yaml(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")

    assert prepare_config(path) is None


def test_post_init_converts_paths_and_loads_dependencies(
    monkeypatch,
    tmp_path,
):
    writer = DummyWriterConfig()
    inference_loader = DummyInferenceLoaderConfig(
        dataset_config=object(),
    )
    train_loader = DummyTrainLoaderConfig()

    monkeypatch.setattr(
        InferenceConfig,
        "load_train_config",
        lambda self: {
            "train_loader": {
                "dataset_config": {},
            }
        },
    )
    monkeypatch.setattr(
        InferenceConfig,
        "load_train_dataloader_config",
        lambda self: train_loader,
    )
    monkeypatch.setattr(
        InferenceConfig,
        "_resolve_inference_dataset_config",
        lambda self: setattr(self, "resolved", True),
    )

    save_path = tmp_path / "predictions"

    config = InferenceConfig(
        experiment_dir=str(tmp_path),
        writer=writer,
        inference_loader=inference_loader,
        save_path=str(save_path),
    )

    assert config.experiment_dir == tmp_path
    assert config.save_path == save_path
    assert config.train_loader is train_loader
    assert config.resolved is True


def test_post_init_leaves_save_path_none(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        InferenceConfig,
        "load_train_config",
        lambda self: {
            "train_loader": {
                "dataset_config": {},
            }
        },
    )
    monkeypatch.setattr(
        InferenceConfig,
        "load_train_dataloader_config",
        lambda self: DummyTrainLoaderConfig(),
    )
    monkeypatch.setattr(
        InferenceConfig,
        "_resolve_inference_dataset_config",
        lambda self: None,
    )

    config = InferenceConfig(
        experiment_dir=str(tmp_path),
        writer=DummyWriterConfig(),
        save_path=None,
    )

    assert config.save_path is None


def test_resolve_inference_dataset_from_training(tmp_path):
    config = make_bare_config(
        tmp_path,
        inference_dataset_config=None,
    )

    config._resolve_inference_dataset_config()

    assert config.inference_loader.read_called
    assert config.inference_loader.read_arg is config.train_loader.dataset_config


def test_resolve_existing_inference_dataset_checks_metadata(
    monkeypatch,
    tmp_path,
):
    config = make_bare_config(
        tmp_path,
        inference_dataset_config=object(),
    )
    called = {"value": False}

    def fake_check():
        called["value"] = True

    monkeypatch.setattr(
        config,
        "_check_inference_dataset",
        fake_check,
    )

    config._resolve_inference_dataset_config()

    assert called["value"] is True
    assert config.inference_loader.read_called is False


def test_check_inference_dataset_matching_metadata(tmp_path):
    config = make_bare_config(
        tmp_path,
        inference_dataset_config=object(),
        inference_metadata={"tas": {"units": "K"}},
        train_metadata={"tas": {"units": "K"}},
    )

    assert config._check_inference_dataset() is None


def test_check_inference_dataset_mismatch_raises(tmp_path):
    config = make_bare_config(
        tmp_path,
        inference_dataset_config=object(),
        inference_metadata={"tas": {}},
        train_metadata={"pr": {}},
    )

    with pytest.raises(
        RuntimeError,
        match="Input variables or preprocessing steps",
    ):
        config._check_inference_dataset()


def test_output_preprocessor_dir_observation_branch(tmp_path):
    config = make_bare_config(tmp_path)
    config.train_config = {
        "train_loader": {
            "dataset_config": {
                "observation": {
                    "names": ["tas"],
                }
            }
        }
    }

    assert config.output_preprocessor_dir == (
        tmp_path
        / "preprocessing_pipeline"
        / "observation_preprocessing_pipeline.joblib"
    )


def test_output_preprocessor_dir_model_branch(tmp_path):
    config = make_bare_config(tmp_path)
    config.train_config = {
        "train_loader": {
            "dataset_config": {
                "model": {
                    "names": ["tas"],
                }
            }
        }
    }

    assert config.output_preprocessor_dir == (
        tmp_path / "preprocessing_pipeline" / "model_preprocessing_pipeline.joblib"
    )


@pytest.mark.pruned
# Remove test due to no coverage
def test_output_dir_uses_custom_save_path(tmp_path):
    save_path = tmp_path / "custom-output"
    config = make_bare_config(
        tmp_path,
        save_path=save_path,
    )

    assert config.output_dir == save_path


@pytest.mark.pruned
# Remove test due to no coverage
def test_output_dir_defaults_to_inference_directory(tmp_path):
    config = make_bare_config(
        tmp_path,
        save_path=None,
    )

    assert config.output_dir == tmp_path / "inference"


@pytest.mark.pruned
# Remove test due to no coverage
def test_log_dir(tmp_path):
    config = make_bare_config(tmp_path)

    assert config.log_dir == tmp_path / "logs"


@pytest.mark.pruned
# Remove test due to no coverage
def test_prepare_runtime_variables_default_output(tmp_path):
    config = make_bare_config(
        tmp_path,
        inference_metadata={"tas": {"units": "K"}},
    )

    config._prepare_runtime_variables()

    assert RuntimeContext.GLOBAL_EXP_DIR == str(tmp_path)
    assert RuntimeContext.GLOBAL_OUTPUT_DIR == str(tmp_path / "inference")
    assert RuntimeContext.GLOBAL_LOG_DIR == str(tmp_path / "logs")
    assert RuntimeContext.INPUT_VAR_METADATA == {
        "tas": {
            "units": "K",
        }
    }


@pytest.mark.pruned
# Remove test due to no coverage
def test_prepare_runtime_variables_custom_output(tmp_path):
    output = tmp_path / "results"
    config = make_bare_config(
        tmp_path,
        save_path=output,
    )

    config._prepare_runtime_variables()

    assert RuntimeContext.GLOBAL_OUTPUT_DIR == str(output)


@pytest.mark.parametrize(
    "root",
    [
        True,
        False,
    ],
)
def test_prepare_directory_branch_matrix(
    tmp_path,
    root,
):
    output = tmp_path / "output"
    config = make_bare_config(
        tmp_path,
        save_path=output,
    )
    distributed = DummyDistributed(root=root)

    config.prepare_directory(distributed)

    assert output.exists() is root
    assert distributed.barrier_called


@pytest.mark.pruned
def test_prepare_directory_existing_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()

    config = make_bare_config(
        tmp_path,
        save_path=output,
    )
    distributed = DummyDistributed(root=True)

    config.prepare_directory(distributed)

    assert output.is_dir()
    assert distributed.barrier_calls == 1


def test_set_random_seed_none_does_not_call_set_seed(
    monkeypatch,
    tmp_path,
):
    config = make_bare_config(
        tmp_path,
        seed=None,
    )
    called = []

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.set_seed",
        lambda seed: called.append(seed),
    )

    config.set_random_seed(rank=3)

    assert called == []


@pytest.mark.parametrize(
    "seed,rank,expected",
    [
        (0, 0, 0),
        (10, 0, 10),
        (10, 4, 14),
        (100, 2, 102),
    ],
)
def test_set_random_seed_uses_rank_offset(
    monkeypatch,
    tmp_path,
    seed,
    rank,
    expected,
):
    config = make_bare_config(
        tmp_path,
        seed=seed,
    )
    called = []

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.set_seed",
        lambda value: called.append(value),
    )

    config.set_random_seed(rank)

    assert called == [expected]


@pytest.mark.pruned
# Remove test due to no coverage
def test_load_train_config_calls_prepare_config(
    monkeypatch,
    tmp_path,
):
    config = make_bare_config(tmp_path)
    captured = {}

    def fake_prepare(path):
        captured["path"] = path
        return {"loaded": True}

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.prepare_config",
        fake_prepare,
    )

    result = config.load_train_config()

    assert result == {"loaded": True}
    assert captured["path"] == tmp_path / "config.yaml"


@pytest.mark.pruned
# Remove test due to no coverage
def test_load_train_dataloader_config_calls_dacite(
    monkeypatch,
    tmp_path,
):
    config = make_bare_config(tmp_path)
    config.train_config = {
        "train_loader": {
            "batch_size": 8,
        }
    }
    expected = object()
    captured = {}

    def fake_from_dict(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.dacite.from_dict",
        fake_from_dict,
    )

    result = config.load_train_dataloader_config()

    assert result is expected
    assert captured["data"] == {"batch_size": 8}
    assert captured["data_class"].__name__ == "TrainDataloaderConfig"


def save_checkpoint(
    path,
    input_shape=(2, 3),
    output_shape=(2, 3),
    added_features_dim=0,
    module_state=None,
    extra=None,
):
    checkpoint = {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "added_features_dim": added_features_dim,
        "module": (
            {"weight": torch.tensor([1.0])} if module_state is None else module_state
        ),
    }

    if extra is not None:
        checkpoint.update(extra)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    torch.save(checkpoint, path)

    return checkpoint


def patch_module_selector(monkeypatch):
    selector = DummySelector()

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.dacite.from_dict",
        lambda **kwargs: selector,
    )

    return selector


@pytest.mark.pruned
def test_load_module_missing_default_checkpoint(tmp_path):
    config = make_bare_config(tmp_path)

    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint not found",
    ):
        config.load_module()


def test_load_module_missing_named_checkpoint(tmp_path):
    config = make_bare_config(tmp_path)
    config.checkpoint_name = "epoch_10.pt"

    with pytest.raises(FileNotFoundError) as error:
        config.load_module()

    assert "epoch_10.pt" in str(error.value)


def test_load_module_missing_required_keys(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    torch.save(
        {
            "input_shape": (2,),
        },
        checkpoint_dir / "best.pt",
    )

    config = make_bare_config(tmp_path)

    with pytest.raises(KeyError) as error:
        config.load_module()

    message = str(error.value)

    assert "added_features_dim" in message
    assert "module" in message
    assert "output_shape" in message


@pytest.mark.parametrize(
    "loader",
    [
        DummyInferenceLoader(
            input_shape=(99,),
            output_shape=(2, 3),
            added_features_dim=0,
        ),
        DummyInferenceLoader(
            input_shape=(2, 3),
            output_shape=(99,),
            added_features_dim=0,
        ),
        DummyInferenceLoader(
            input_shape=(2, 3),
            output_shape=(2, 3),
            added_features_dim=99,
        ),
    ],
)
def test_load_module_loader_dimension_mismatch(
    monkeypatch,
    tmp_path,
    loader,
):
    save_checkpoint(
        tmp_path / "checkpoints" / "best.pt",
    )
    config = make_bare_config(tmp_path)

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.dacite.from_dict",
        lambda **kwargs: pytest.fail("selector should not be built"),
    )

    with pytest.raises(
        RuntimeError,
        match="Data and model IO dimensions do not match",
    ):
        config.load_module(loader)


@pytest.mark.pruned
def test_load_module_without_loader(
    monkeypatch,
    tmp_path,
):
    checkpoint = save_checkpoint(
        tmp_path / "checkpoints" / "best.pt",
        input_shape=(4,),
        output_shape=(2,),
        added_features_dim=3,
    )
    selector = patch_module_selector(monkeypatch)
    config = make_bare_config(tmp_path)

    module = config.load_module()

    assert module is selector.module
    assert selector.build_kwargs == {
        "input_shape": (4,),
        "output_shape": (2,),
        "added_features_dim": 3,
    }
    assert module.loaded_state == checkpoint["module"]
    assert module.loaded_strict is True


@pytest.mark.pruned
def test_load_module_with_matching_loader(
    monkeypatch,
    tmp_path,
):
    checkpoint = save_checkpoint(
        tmp_path / "checkpoints" / "best.pt",
        input_shape=(4,),
        output_shape=(2,),
        added_features_dim=3,
    )
    selector = patch_module_selector(monkeypatch)
    config = make_bare_config(tmp_path)
    loader = DummyInferenceLoader(
        input_shape=(4,),
        output_shape=(2,),
        added_features_dim=3,
    )

    module = config.load_module(loader)

    assert module is selector.module
    assert module.loaded_state == checkpoint["module"]


@pytest.mark.pruned
def test_load_module_strict_false(
    monkeypatch,
    tmp_path,
):
    save_checkpoint(
        tmp_path / "checkpoints" / "best.pt",
    )
    selector = patch_module_selector(monkeypatch)
    config = make_bare_config(tmp_path)

    module = config.load_module(
        strict=False,
    )

    assert module.loaded_strict is False


@pytest.mark.pruned
def test_load_module_named_checkpoint(
    monkeypatch,
    tmp_path,
):
    save_checkpoint(
        tmp_path / "checkpoints" / "custom.pt",
        input_shape=(7,),
        output_shape=(8,),
        added_features_dim=9,
    )
    selector = patch_module_selector(monkeypatch)
    config = make_bare_config(tmp_path)
    config.checkpoint_name = "custom.pt"

    module = config.load_module()

    assert module is selector.module
    assert selector.build_kwargs == {
        "input_shape": (7,),
        "output_shape": (8,),
        "added_features_dim": 9,
    }


@pytest.mark.pruned
def test_load_module_dacite_receives_module_config(
    monkeypatch,
    tmp_path,
):
    save_checkpoint(
        tmp_path / "checkpoints" / "best.pt",
    )
    config = make_bare_config(tmp_path)
    config.train_config["module"] = {
        "type": "custom",
        "hidden_size": 32,
    }

    selector = DummySelector()
    captured = {}

    def fake_from_dict(**kwargs):
        captured.update(kwargs)
        return selector

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.dacite.from_dict",
        fake_from_dict,
    )

    config.load_module()

    assert captured["data"] == {
        "type": "custom",
        "hidden_size": 32,
    }
    assert captured["data_class"].__name__ == "ModuleSelector"


@pytest.mark.pruned
def test_load_module_calls_gc_collect(
    monkeypatch,
    tmp_path,
):
    save_checkpoint(
        tmp_path / "checkpoints" / "best.pt",
    )
    patch_module_selector(monkeypatch)
    config = make_bare_config(tmp_path)
    calls = []

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.gc.collect",
        lambda: calls.append(True),
    )

    config.load_module()

    assert calls == [True]


def make_writer_config_fixture(tmp_path):
    inference_loader_config = DummyInferenceLoaderConfig(
        dataset_config=object(),
    )
    train_loader = DummyTrainLoaderConfig()
    writer_config = DummyWriterConfig()
    module = DummyModule()

    config = SimpleNamespace(
        inference_loader=inference_loader_config,
        train_loader=train_loader,
        writer=writer_config,
        output_preprocessor_dir=(
            tmp_path
            / "preprocessing_pipeline"
            / "observation_preprocessing_pipeline.joblib"
        ),
        output_dir=tmp_path / "inference",
        load_module=lambda loader: module,
    )

    return (
        config,
        inference_loader_config,
        train_loader,
        writer_config,
        module,
    )


@pytest.mark.parametrize(
    "root,with_logger",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_build_writer_logging_matrix(
    monkeypatch,
    tmp_path,
    capsys,
    root,
    with_logger,
):
    (
        config,
        inference_config,
        train_loader,
        writer_config,
        module,
    ) = make_writer_config_fixture(tmp_path)

    postprocessor = object()

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.PreprocessingPipeline.load_from_memory",
        lambda self, path: postprocessor,
    )

    logger_messages = []

    logger = (
        SimpleNamespace(info=lambda message, **kwargs: logger_messages.append(message))
        if with_logger
        else None
    )

    distributed = DummyDistributed(
        root=root,
    )

    writer = build_writer(
        config,
        distributed,
        logger,
    )

    assert writer is writer_config.writer
    assert inference_config.setup_called
    assert inference_config.build_called
    assert module.device == torch.device("cpu")
    assert writer_config.build_called

    if root and with_logger:
        assert logger_messages == [
            "creating data loader ...",
            "Loading saved module ...",
            "Loading postprocessor ...",
            "Creating writer ...",
        ]
        assert capsys.readouterr().out == ""
    elif root and not with_logger:
        output = capsys.readouterr().out
        assert "creating data loader ..." in output
        assert "Loading saved module ..." in output
        assert "Loading postprocessor ..." in output
        assert "Creating writer ..." in output
    else:
        assert logger_messages == []
        assert capsys.readouterr().out == ""


@pytest.mark.pruned
def test_build_writer_passes_all_writer_arguments(
    monkeypatch,
    tmp_path,
):
    (
        config,
        inference_config,
        train_loader,
        writer_config,
        module,
    ) = make_writer_config_fixture(tmp_path)

    postprocessor = object()

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.PreprocessingPipeline.load_from_memory",
        lambda self, path: postprocessor,
    )

    distributed = DummyDistributed()

    writer = build_writer(
        config,
        distributed,
    )

    assert writer is writer_config.writer
    assert writer_config.build_kwargs == {
        "inference_data_loader": inference_config.loader,
        "train_dataloader_config": train_loader,
        "module": module,
        "post_processor": postprocessor,
        "output_dir": tmp_path / "inference",
    }


@pytest.mark.pruned
def test_build_writer_loads_expected_preprocessor(
    monkeypatch,
    tmp_path,
):
    (
        config,
        _,
        _,
        _,
        _,
    ) = make_writer_config_fixture(tmp_path)
    captured = {}

    def fake_load(self, path):
        captured["path"] = path
        return object()

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.PreprocessingPipeline.load_from_memory",
        fake_load,
    )

    build_writer(
        config,
        DummyDistributed(),
    )

    assert captured["path"] == config.output_preprocessor_dir


@pytest.mark.pruned
def test_build_writer_propagates_loader_setup_error(
    monkeypatch,
    tmp_path,
):
    (
        config,
        inference_config,
        _,
        writer_config,
        _,
    ) = make_writer_config_fixture(tmp_path)

    def fail_setup(*args, **kwargs):
        raise RuntimeError("setup failed")

    inference_config.setup_distributed = fail_setup

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.PreprocessingPipeline.load_from_memory",
        lambda self, path: object(),
    )

    with pytest.raises(
        RuntimeError,
        match="setup failed",
    ):
        build_writer(
            config,
            DummyDistributed(),
        )

    assert writer_config.build_called is False


@pytest.mark.pruned
def test_build_writer_propagates_module_load_error(
    monkeypatch,
    tmp_path,
):
    (
        config,
        _,
        _,
        writer_config,
        _,
    ) = make_writer_config_fixture(tmp_path)

    def fail_load(_):
        raise FileNotFoundError("missing model")

    config.load_module = fail_load

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.PreprocessingPipeline.load_from_memory",
        lambda self, path: object(),
    )

    with pytest.raises(
        FileNotFoundError,
        match="missing model",
    ):
        build_writer(
            config,
            DummyDistributed(),
        )

    assert writer_config.build_called is False


@pytest.mark.pruned
def test_build_writer_propagates_postprocessor_error(
    monkeypatch,
    tmp_path,
):
    (
        config,
        _,
        _,
        writer_config,
        _,
    ) = make_writer_config_fixture(tmp_path)

    def fail_load(self, path):
        raise FileNotFoundError("missing preprocessor")

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.PreprocessingPipeline.load_from_memory",
        fail_load,
    )

    with pytest.raises(
        FileNotFoundError,
        match="missing preprocessor",
    ):
        build_writer(
            config,
            DummyDistributed(),
        )

    assert writer_config.build_called is False
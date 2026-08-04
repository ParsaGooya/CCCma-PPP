from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from cccma_ppp.inference.inference_configs import (
    InferenceConfig,
    build_writer,
)
from copy import deepcopy


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
        time_features=None,
    ):
        self.dataset_config = dataset_config
        self._input_metadata = {"tas": {}} if input_metadata is None else input_metadata
        self.time_features = time_features
        self.read_called = False
        self.read_arg = None
        self.setup_called = False
        self.setup_args = None
        self.build_called = False
        self.loader = DummyInferenceLoader()

    @property
    def input_var_metadata(self):
        return self._input_metadata

    def read_configs_from_train(
        self,
        train_loader_config,
    ):
        self.read_called = True
        self.read_arg = train_loader_config

        if self.dataset_config is None:
            self.dataset_config = object()

        if self.time_features is None:
            self.time_features = deepcopy(train_loader_config.time_features)

    def setup_distributed(
        self,
        train_loader,
        distributed,
    ):
        self.setup_called = True
        self.setup_args = (
            train_loader,
            distributed,
        )

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
        time_features=None,
    ):
        self.dataset_config = DummyTrainDatasetConfig()
        self._input_metadata = {"tas": {}} if input_metadata is None else input_metadata
        self.time_features = time_features

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
    inference_time_features=None,
    train_time_features=None,
):
    config = object.__new__(InferenceConfig)

    config.experiment_dir = Path(tmp_path)
    config.save_path = Path(save_path) if save_path is not None else None
    config.seed = seed
    config.checkpoint_name = checkpoint_name
    config.writer = DummyWriterConfig()

    config.inference_loader = DummyInferenceLoaderConfig(
        dataset_config=(inference_dataset_config),
        input_metadata=(inference_metadata),
        time_features=(inference_time_features),
    )

    config.train_loader = DummyTrainLoaderConfig(
        input_metadata=train_metadata,
        time_features=train_time_features,
    )

    config.train_config = {
        "train_loader": {
            "time_features": (train_time_features),
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
    assert module.loaded_strict is False


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


def test_resolve_inference_dataset_reads_train_config(
    tmp_path,
):
    config = make_bare_config(
        tmp_path,
        inference_dataset_config=None,
        inference_time_features=None,
        train_time_features=["year"],
    )

    config._resolve_inference_dataset_config()

    assert config.inference_loader.read_called is True
    assert config.inference_loader.read_arg is config.train_loader
    assert config.inference_loader.dataset_config is not None
    assert config.inference_loader.time_features == ["year"]


def test_check_inference_dataset_matching_metadata_and_features(
    tmp_path,
):
    config = make_bare_config(
        tmp_path,
        inference_dataset_config=object(),
        inference_metadata={"tas": {"units": "K"}},
        train_metadata={"tas": {"units": "K"}},
        inference_time_features=["year"],
        train_time_features=["year"],
    )

    assert config._check_inference_dataset() is None


def test_check_inference_dataset_time_features_mismatch(
    tmp_path,
):
    config = make_bare_config(
        tmp_path,
        inference_dataset_config=object(),
        inference_metadata={"tas": {"units": "K"}},
        train_metadata={"tas": {"units": "K"}},
        inference_time_features=["year"],
        train_time_features=["lead_time"],
    )

    with pytest.raises(RuntimeError):
        config._check_inference_dataset()


def test_check_inference_dataset_none_features_match(
    tmp_path,
):
    config = make_bare_config(
        tmp_path,
        inference_dataset_config=object(),
        inference_metadata={"tas": {"units": "K"}},
        train_metadata={"tas": {"units": "K"}},
        inference_time_features=None,
        train_time_features=None,
    )

    assert config._check_inference_dataset() is None


def test_check_inference_dataset_metadata_checked_before_features(
    tmp_path,
):
    config = make_bare_config(
        tmp_path,
        inference_dataset_config=object(),
        inference_metadata={"tas": {}},
        train_metadata={"pr": {}},
        inference_time_features=["year"],
        train_time_features=["lead_time"],
    )

    with pytest.raises(
        RuntimeError,
        match="Input variables",
    ):
        config._check_inference_dataset()


def test_output_preprocessor_dir_prefers_observation(
    tmp_path,
):
    config = make_bare_config(tmp_path)
    config.train_config = {
        "train_loader": {
            "dataset_config": {
                "model": {"names": ["model"]},
                "observation": {"names": ["observation"]},
            }
        }
    }

    assert config.output_preprocessor_dir == (
        tmp_path
        / "preprocessing_pipeline"
        / "observation_preprocessing_pipeline.joblib"
    )


def test_output_preprocessor_dir_uses_model_without_observation(
    tmp_path,
):
    config = make_bare_config(tmp_path)
    config.train_config = {
        "train_loader": {"dataset_config": {"model": {"names": ["model"]}}}
    }

    assert config.output_preprocessor_dir == (
        tmp_path / "preprocessing_pipeline" / "model_preprocessing_pipeline.joblib"
    )


def test_prepare_directory_root_creates_nested_directory(
    tmp_path,
):
    output = tmp_path / "nested" / "predictions"
    config = make_bare_config(
        tmp_path,
        save_path=output,
    )
    distributed = DummyDistributed(root=True)

    config.prepare_directory(distributed)

    assert output.is_dir()
    assert distributed.barrier_calls == 1


def test_prepare_directory_nonroot_does_not_create_directory(
    tmp_path,
):
    output = tmp_path / "output"
    config = make_bare_config(
        tmp_path,
        save_path=output,
    )
    distributed = DummyDistributed(root=False)

    config.prepare_directory(distributed)

    assert not output.exists()
    assert distributed.barrier_calls == 1


def test_prepare_directory_preserves_existing_contents(
    tmp_path,
):
    output = tmp_path / "output"
    output.mkdir()

    existing = output / "existing.txt"
    existing.write_text("keep")

    config = make_bare_config(
        tmp_path,
        save_path=output,
    )
    distributed = DummyDistributed(root=True)

    config.prepare_directory(distributed)

    assert existing.read_text() == "keep"
    assert distributed.barrier_calls == 1


@pytest.mark.parametrize(
    "strict",
    [True, False],
)
def test_load_module_passes_requested_strict_value(
    monkeypatch,
    tmp_path,
    strict,
):
    save_checkpoint(tmp_path / "checkpoints" / "best.pt")
    selector = patch_module_selector(monkeypatch)
    config = make_bare_config(tmp_path)

    module = config.load_module(strict=strict)

    assert module.loaded_strict is strict
    assert module is selector.module


def test_load_module_propagates_state_dict_error(
    monkeypatch,
    tmp_path,
):
    save_checkpoint(tmp_path / "checkpoints" / "best.pt")
    selector = patch_module_selector(monkeypatch)
    config = make_bare_config(tmp_path)

    def fail_load_state(
        state,
        strict=True,
    ):
        raise RuntimeError("invalid state")

    selector.module.load_state_dict = fail_load_state

    with pytest.raises(
        RuntimeError,
        match="invalid state",
    ):
        config.load_module()


def test_load_module_propagates_selector_build_error(
    monkeypatch,
    tmp_path,
):
    save_checkpoint(tmp_path / "checkpoints" / "best.pt")
    config = make_bare_config(tmp_path)

    class FailingSelector:
        def build_module(self, **kwargs):
            raise RuntimeError("build failed")

    monkeypatch.setattr(
        "cccma_ppp.inference.inference_configs.dacite.from_dict",
        lambda **kwargs: FailingSelector(),
    )

    with pytest.raises(
        RuntimeError,
        match="build failed",
    ):
        config.load_module()


def test_load_module_uses_named_checkpoint_path(
    monkeypatch,
    tmp_path,
):
    checkpoint_path = tmp_path / "checkpoints" / "epoch_20.pt"
    save_checkpoint(
        checkpoint_path,
        input_shape=(5,),
        output_shape=(6,),
        added_features_dim=1,
    )

    selector = patch_module_selector(monkeypatch)
    config = make_bare_config(
        tmp_path,
        checkpoint_name="epoch_20.pt",
    )

    module = config.load_module()

    assert module is selector.module
    assert selector.build_kwargs == {
        "input_shape": (5,),
        "output_shape": (6,),
        "added_features_dim": 1,
    }


@pytest.mark.parametrize(
    ("root", "logger_present"),
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_build_writer_setup_arguments(
    monkeypatch,
    tmp_path,
    root,
    logger_present,
):
    (
        config,
        inference_config,
        train_loader,
        _,
        _,
    ) = make_writer_config_fixture(tmp_path)

    monkeypatch.setattr(
        (
            "cccma_ppp.inference.inference_configs."
            "PreprocessingPipeline.load_from_memory"
        ),
        lambda self, path: object(),
    )

    logger = (
        SimpleNamespace(info=lambda *args, **kwargs: None) if logger_present else None
    )
    distributed = DummyDistributed(root=root)

    build_writer(
        config,
        distributed,
        logger,
    )

    assert inference_config.setup_args == (
        train_loader,
        distributed,
    )


def test_build_writer_module_receives_device(
    monkeypatch,
    tmp_path,
):
    (
        config,
        _,
        _,
        _,
        module,
    ) = make_writer_config_fixture(tmp_path)

    monkeypatch.setattr(
        (
            "cccma_ppp.inference.inference_configs."
            "PreprocessingPipeline.load_from_memory"
        ),
        lambda self, path: object(),
    )

    distributed = DummyDistributed(device="cpu")

    build_writer(
        config,
        distributed,
    )

    assert module.device == distributed.device


def test_build_writer_propagates_loader_build_error(
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

    def fail_build_loader():
        raise RuntimeError("loader failed")

    inference_config.build_inference_loader = fail_build_loader

    monkeypatch.setattr(
        (
            "cccma_ppp.inference.inference_configs."
            "PreprocessingPipeline.load_from_memory"
        ),
        lambda self, path: object(),
    )

    with pytest.raises(
        RuntimeError,
        match="loader failed",
    ):
        build_writer(
            config,
            DummyDistributed(),
        )

    assert writer_config.build_called is False


def test_build_writer_propagates_writer_build_error(
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

    monkeypatch.setattr(
        (
            "cccma_ppp.inference.inference_configs."
            "PreprocessingPipeline.load_from_memory"
        ),
        lambda self, path: object(),
    )

    def fail_build(**kwargs):
        raise RuntimeError("writer failed")

    writer_config.build = fail_build

    with pytest.raises(
        RuntimeError,
        match="writer failed",
    ):
        build_writer(
            config,
            DummyDistributed(),
        )

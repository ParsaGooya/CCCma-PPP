from pathlib import Path
import uuid

import numpy as np
import pytest
import torch

import cccma_ppp.core.selectors as selectors_mod
from cccma_ppp.core.selectors import (
    FlowSelector,
    ModelSelector,
    ModuleSelector,
    cVAEModelSelector,
    deterministicModelSelector,
    _load_config_from_checkpoint,
)


def unique_name(prefix="test"):
    return f"{prefix}_{uuid.uuid4().hex}"


class DummyBuiltModule:
    def __init__(self, input_shape=None, output_shape=None, added_features_dim=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.added_features_dim = added_features_dim
        self.built = True


class DummyModuleConfig:
    def __init__(self, value=1, **kwargs):
        self.value = value
        self.kwargs = kwargs
        self.was_built = False

    def build(self, input_shape, output_shape=None, added_features_dim=None):
        self.was_built = True
        return DummyBuiltModule(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


class DummyModelConfig:
    def __init__(self, value=1, **kwargs):
        self.value = value
        self.kwargs = kwargs
        self.checkpoint_config = None

    def _add_checkpoint_config(self, checkpoint_config):
        self.checkpoint_config = checkpoint_config
        return self

    def build(self):
        return object()


class DummyFlow:
    def __init__(self, scale=1, **kwargs):
        self.scale = scale
        self.kwargs = kwargs


class LocalModelSelector(ModelSelector):
    registery = type(cVAEModelSelector.registery)()

    pass


def make_checkpoint(
    path: Path,
    model_type: str = "dummy",
    config: dict | None = None,
    input_shape=None,
    output_shape=None,
):
    if config is None:
        config = {"value": 10}

    torch.save(
        {
            "module_config": {
                "ModelConfig": {
                    "type": model_type,
                    "config": config,
                }
            },
            "model_input_shape": (
                input_shape if input_shape is not None else np.array([1])
            ),
            "model_output_shape": (
                output_shape if output_shape is not None else np.array([2])
            ),
        },
        path,
    )


@pytest.mark.pruned
def test_module_selector_build_without_output_shape():
    name = unique_name("module_no_output")

    ModuleSelector.register(name)(DummyModuleConfig)

    selector = ModuleSelector(type=name, config={"value": 1})
    built = selector.build_module(input_shape=np.array([5]))

    assert isinstance(built, DummyBuiltModule)
    assert np.array_equal(built.input_shape, np.array([5]))
    assert built.output_shape is None


@pytest.mark.pruned
def test_module_selector_unregistered_type_raises():
    with pytest.raises(Exception):
        ModuleSelector(type=unique_name("missing_module"), config={})


@pytest.mark.pruned
def test_module_selector_register_lowercases_name():
    name = unique_name("mixed_module")

    ModuleSelector.register(name.upper())(DummyModuleConfig)

    assert name.lower() in ModuleSelector.available()

    selector = ModuleSelector(type=name.upper(), config={})

    assert isinstance(selector._module_config, DummyModuleConfig)


@pytest.mark.pruned
def test_module_selector_register_available_and_build():
    name = unique_name("module")

    ModuleSelector.register(name)(DummyModuleConfig)

    assert name in ModuleSelector.available()

    selector = ModuleSelector(type=name.upper(), config={"value": 42})

    built = selector.build_module(
        input_shape=np.array([1]),
        output_shape=np.array([2]),
        added_features_dim=3,
    )

    assert isinstance(built, DummyBuiltModule)
    assert np.array_equal(built.input_shape, np.array([1]))
    assert np.array_equal(built.output_shape, np.array([2]))
    assert built.added_features_dim == 3


def test_model_selector_requires_config_or_load_dir():
    with pytest.raises(RuntimeError):
        LocalModelSelector(type="anything", config=None, load_dir=None)


@pytest.mark.pruned
def test_model_selector_register_available_get_model_config():
    name = unique_name("model")

    LocalModelSelector.register(name)(DummyModelConfig)

    selector = LocalModelSelector(type=name.upper(), config={"value": 7})
    model_config = selector.get_model_config()

    assert isinstance(model_config, DummyModelConfig)
    assert model_config.value == 7
    assert model_config.checkpoint_config is None
    assert name in LocalModelSelector.available()


@pytest.mark.pruned
def test_model_selector_register_lowercase_lookup():
    name = unique_name("case_model")

    LocalModelSelector.register(name.upper())(DummyModelConfig)

    selector = LocalModelSelector(type=name.upper(), config={"value": 3})
    cfg = selector.get_model_config()

    assert isinstance(cfg, DummyModelConfig)
    assert cfg.value == 3


def test_model_selector_unregistered_raises():
    selector = LocalModelSelector(type=unique_name("missing_model"), config={})

    with pytest.raises(Exception):
        selector.get_model_config()


def test_model_selector_checkpoint_load_overwrites_config(monkeypatch, tmp_path):
    name = unique_name("checkpoint_model")
    LocalModelSelector.register(name)(DummyModelConfig)

    fake_checkpoint_config = object()

    def fake_load_config_from_checkpoint(load_dir, freeze_weights=False):
        return (
            {
                "ModelConfig": {
                    "type": name,
                    "config": {"value": 99},
                }
            },
            fake_checkpoint_config,
        )

    monkeypatch.setattr(
        selectors_mod,
        "_load_config_from_checkpoint",
        fake_load_config_from_checkpoint,
    )

    with pytest.warns(UserWarning):
        selector = LocalModelSelector(
            type=name,
            config={"value": 1},
            load_dir=tmp_path / "checkpoint.pt",
        )

    assert selector.config == {"value": 99}
    assert selector.checkpoint_config is fake_checkpoint_config

    model_config = selector.get_model_config()

    assert isinstance(model_config, DummyModelConfig)
    assert model_config.value == 99
    assert model_config.checkpoint_config is fake_checkpoint_config


@pytest.mark.pruned
def test_model_selector_load_dir_type_mismatch(monkeypatch, tmp_path):
    def fake_load_config_from_checkpoint(load_dir, freeze_weights=False):
        return (
            {
                "ModelConfig": {
                    "type": "different_type",
                    "config": {},
                }
            },
            object(),
        )

    monkeypatch.setattr(
        selectors_mod,
        "_load_config_from_checkpoint",
        fake_load_config_from_checkpoint,
    )

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        LocalModelSelector(
            type="expected_type",
            config={},
            load_dir=tmp_path / "checkpoint.pt",
        )


def test_model_selector_freeze_weights_warning(monkeypatch, tmp_path):
    name = unique_name("freeze_model")
    LocalModelSelector.register(name)(DummyModelConfig)

    def fake_load_config_from_checkpoint(load_dir, freeze_weights=False):
        return (
            {
                "ModelConfig": {
                    "type": name,
                    "config": {"value": 11},
                }
            },
            object(),
        )

    monkeypatch.setattr(
        selectors_mod,
        "_load_config_from_checkpoint",
        fake_load_config_from_checkpoint,
    )

    with pytest.warns(UserWarning) as record:
        LocalModelSelector(
            type=name,
            config=None,
            load_dir=tmp_path / "checkpoint.pt",
            freeze_weights=True,
        )

    messages = [str(w.message) for w in record]

    assert any("overwritten" in msg for msg in messages)
    assert any("freeze" in msg.lower() for msg in messages)


@pytest.mark.pruned
def test_model_selector_load_dir_without_freeze_only_one_warning(monkeypatch, tmp_path):
    name = unique_name("nofreeze_model")
    LocalModelSelector.register(name)(DummyModelConfig)

    def fake_load_config_from_checkpoint(load_dir, freeze_weights=False):
        return (
            {
                "ModelConfig": {
                    "type": name,
                    "config": {"value": 12},
                }
            },
            object(),
        )

    monkeypatch.setattr(
        selectors_mod,
        "_load_config_from_checkpoint",
        fake_load_config_from_checkpoint,
    )

    with pytest.warns(UserWarning) as record:
        LocalModelSelector(
            type=name,
            config=None,
            load_dir=tmp_path / "checkpoint.pt",
            freeze_weights=False,
        )

    assert len(record) == 1
    assert "overwritten" in str(record[0].message)


def test_cvae_model_selector_has_registry():
    name = unique_name("cvae_model")

    cVAEModelSelector.register(name)(DummyModelConfig)

    selector = cVAEModelSelector(type=name, config={"value": 5})
    cfg = selector.get_model_config()

    assert isinstance(cfg, DummyModelConfig)
    assert cfg.value == 5
    assert name in cVAEModelSelector.available()


@pytest.mark.pruned
def test_deterministic_model_selector_has_registry():
    name = unique_name("det_model")

    deterministicModelSelector.register(name)(DummyModelConfig)

    selector = deterministicModelSelector(type=name, config={"value": 6})
    cfg = selector.get_model_config()

    assert isinstance(cfg, DummyModelConfig)
    assert cfg.value == 6
    assert name in deterministicModelSelector.available()


@pytest.mark.pruned
def test_flow_selector_register_available_and_get_model():
    name = unique_name("flow")

    FlowSelector.register(name)(DummyFlow)

    selector = FlowSelector(type=name.upper(), args={"scale": 3})
    flow = selector.get_model()

    assert isinstance(flow, DummyFlow)
    assert flow.scale == 3
    assert name in FlowSelector.available()


@pytest.mark.pruned
def test_flow_selector_unregistered_raises():
    selector = FlowSelector(type=unique_name("missing_flow"), args={})

    with pytest.raises(Exception):
        selector.get_model()


@pytest.mark.pruned
def test_flow_selector_post_init_noop():
    name = unique_name("flow_noop")

    FlowSelector.register(name)(DummyFlow)

    selector = FlowSelector(type=name, args={"scale": 4})
    flow = selector.get_model()

    assert isinstance(flow, DummyFlow)
    assert flow.scale == 4


@pytest.mark.pruned
def test_flow_selector_case_insensitive_lookup():
    name = unique_name("flow_case")

    FlowSelector.register(name.upper())(DummyFlow)

    selector = FlowSelector(type=name.upper(), args={"scale": 8})
    flow = selector.get_model()

    assert isinstance(flow, DummyFlow)
    assert flow.scale == 8


def test_load_config_from_checkpoint_missing_file():
    with pytest.raises(FileNotFoundError):
        _load_config_from_checkpoint("missing_checkpoint.pt")


@pytest.mark.pruned
def test_load_config_from_checkpoint_success(tmp_path):
    path = tmp_path / "checkpoint.pt"

    make_checkpoint(
        path,
        model_type="abc",
        config={"value": 123},
        input_shape=np.array([1, 2]),
        output_shape=np.array([3, 4]),
    )

    checkpoint_module, checkpoint_config = _load_config_from_checkpoint(path)

    assert checkpoint_module["ModelConfig"]["type"] == "abc"
    assert checkpoint_module["ModelConfig"]["config"] == {"value": 123}
    assert checkpoint_config is not None
    assert checkpoint_config.load_path == path
    assert checkpoint_config.strict is True


@pytest.mark.pruned
def test_load_config_from_checkpoint_strict_false(tmp_path):
    path = tmp_path / "checkpoint.pt"

    make_checkpoint(path, model_type="abc")

    checkpoint_module, checkpoint_config = _load_config_from_checkpoint(
        path,
        strict=False,
    )

    assert checkpoint_module["ModelConfig"]["type"] == "abc"
    assert checkpoint_config is not None
    assert checkpoint_config.strict is False


@pytest.mark.pruned
def test_load_config_from_checkpoint_missing_module_config(tmp_path):
    path = tmp_path / "bad_checkpoint.pt"

    torch.save(
        {
            "model_input_shape": np.array([1]),
            "model_output_shape": np.array([1]),
        },
        path,
    )

    checkpoint_module, checkpoint_config = _load_config_from_checkpoint(path)

    assert checkpoint_module in (None, {})
    assert checkpoint_config is not None
    assert checkpoint_config.load_path == path


@pytest.mark.pruned
def test_load_config_from_checkpoint_missing_shapes_allowed(tmp_path):
    path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "module_config": {
                "ModelConfig": {
                    "type": "abc",
                    "config": {"value": 1},
                }
            },
            "model_input_shape": None,
            "model_output_shape": None,
        },
        path,
    )

    checkpoint_module, checkpoint_config = _load_config_from_checkpoint(path)

    assert checkpoint_module["ModelConfig"]["type"] == "abc"
    assert checkpoint_config is not None
    assert checkpoint_config.load_path == path

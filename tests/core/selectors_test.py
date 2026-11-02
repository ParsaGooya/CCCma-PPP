import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

import S2D4P.core.selectors as selectors_mod
from S2D4P.core.registery import Registery
from S2D4P.core.selectors import (
    ArchSelector,
    FlowSelector,
    ModuleSelector,
    _load_config_from_checkpoint,
    cVAEArchSelector,
    deterministicArchSelector,
)


class LocalArchSelector(ArchSelector):
    registery = Registery()


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


class DummyArchConfig:
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


def make_checkpoint(
    path: Path,
    architecture_type: str = "dummy",
    config: dict | None = None,
    input_shape=None,
    output_shape=None,
):
    """Save a checkpoint in the layout that `_load_config_from_checkpoint` reads."""
    torch.save(
        {
            "module_config": {
                "Architecture": {
                    "type": architecture_type,
                    "config": {"value": 10} if config is None else config,
                }
            },
            "input_shape": np.array([1]) if input_shape is None else input_shape,
            "output_shape": np.array([2]) if output_shape is None else output_shape,
            "input_var_metadata": {},
            "output_var_metadata": {},
        },
        path,
    )


def patch_checkpoint_loader(
    monkeypatch,
    architecture_type,
    config,
    checkpoint_config=None,
):
    """
    Replace `_load_config_from_checkpoint` with a stub.

    Returns the checkpoint config the stub hands back, so tests can check
    identity.
    """
    sentinel = checkpoint_config if checkpoint_config is not None else object()

    def fake_loader(load_dir, freeze_weights=False):
        return (
            {"Architecture": {"type": architecture_type, "config": config}},
            sentinel,
        )

    monkeypatch.setattr(selectors_mod, "_load_config_from_checkpoint", fake_loader)
    return sentinel


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
def test_module_selector_build_passes_all_shapes():
    name = unique_name("module")
    ModuleSelector.register(name)(DummyModuleConfig)

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


def test_arch_selector_requires_config_or_load_dir():
    with pytest.raises(RuntimeError):
        LocalArchSelector(type="anything", config=None, load_dir=None)


def test_arch_selector_freeze_without_load_dir_raises():
    with pytest.raises(ValueError):
        LocalArchSelector(type="anything", config={}, freeze_weights=True)


def test_arch_selector_register_and_get_architecture_config():
    name = unique_name("arch")
    LocalArchSelector.register(name)(DummyArchConfig)

    selector = LocalArchSelector(type=name.upper(), config={"value": 7})
    arch_config = selector.get_architecture_config()

    assert isinstance(arch_config, DummyArchConfig)
    assert arch_config.value == 7
    assert arch_config.checkpoint_config is None
    assert name in LocalArchSelector.available()


@pytest.mark.pruned
def test_arch_selector_register_lowercase_lookup():
    name = unique_name("case_arch")
    LocalArchSelector.register(name.upper())(DummyArchConfig)

    selector = LocalArchSelector(type=name.upper(), config={"value": 3})
    arch_config = selector.get_architecture_config()

    assert isinstance(arch_config, DummyArchConfig)
    assert arch_config.value == 3


@pytest.mark.pruned
def test_arch_selector_unregistered_raises():
    selector = LocalArchSelector(type=unique_name("missing_arch"), config={})

    with pytest.raises(Exception):
        selector.get_architecture_config()


def test_arch_selector_checkpoint_load_overwrites_config(monkeypatch, tmp_path):
    name = unique_name("checkpoint_arch")
    LocalArchSelector.register(name)(DummyArchConfig)
    fake_checkpoint_config = patch_checkpoint_loader(monkeypatch, name, {"value": 99})

    with pytest.warns(UserWarning):
        selector = LocalArchSelector(
            type=name,
            config={"value": 1},
            load_dir=tmp_path / "checkpoint.pt",
        )

    assert selector.config == {"value": 99}
    assert selector.checkpoint_config is fake_checkpoint_config

    arch_config = selector.get_architecture_config()

    assert isinstance(arch_config, DummyArchConfig)
    assert arch_config.value == 99
    assert arch_config.checkpoint_config is fake_checkpoint_config


@pytest.mark.pruned
def test_arch_selector_load_dir_type_mismatch_raises(monkeypatch, tmp_path):
    patch_checkpoint_loader(monkeypatch, "different_type", {})

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        LocalArchSelector(
            type="expected_type",
            config={},
            load_dir=tmp_path / "checkpoint.pt",
        )


def test_arch_selector_freeze_weights_emits_both_warnings(monkeypatch, tmp_path):
    name = unique_name("freeze_arch")
    LocalArchSelector.register(name)(DummyArchConfig)
    patch_checkpoint_loader(monkeypatch, name, {"value": 11})

    with pytest.warns(UserWarning) as record:
        LocalArchSelector(
            type=name,
            config=None,
            load_dir=tmp_path / "checkpoint.pt",
            freeze_weights=True,
        )

    messages = [str(w.message) for w in record]

    assert any("overwritten" in msg for msg in messages)
    assert any("freeze" in msg.lower() for msg in messages)


@pytest.mark.pruned
def test_arch_selector_load_dir_without_freeze_warns_once(monkeypatch, tmp_path):
    name = unique_name("nofreeze_arch")
    LocalArchSelector.register(name)(DummyArchConfig)
    patch_checkpoint_loader(monkeypatch, name, {"value": 12})

    with pytest.warns(UserWarning) as record:
        LocalArchSelector(
            type=name,
            config=None,
            load_dir=tmp_path / "checkpoint.pt",
            freeze_weights=False,
        )

    assert len(record) == 1
    assert "overwritten" in str(record[0].message)


@pytest.mark.pruned
def test_cvae_arch_selector_has_registry():
    name = unique_name("cvae_arch")
    cVAEArchSelector.register(name)(DummyArchConfig)

    selector = cVAEArchSelector(type=name, config={"value": 5})
    arch_config = selector.get_architecture_config()

    assert isinstance(arch_config, DummyArchConfig)
    assert arch_config.value == 5
    assert name in cVAEArchSelector.available()


@pytest.mark.pruned
def test_deterministic_arch_selector_has_registry():
    name = unique_name("det_arch")
    deterministicArchSelector.register(name)(DummyArchConfig)

    selector = deterministicArchSelector(type=name, config={"value": 6})
    arch_config = selector.get_architecture_config()

    assert isinstance(arch_config, DummyArchConfig)
    assert arch_config.value == 6
    assert name in deterministicArchSelector.available()


@pytest.mark.pruned
def test_flow_selector_register_available_and_get_flow():
    name = unique_name("flow")
    FlowSelector.register(name)(DummyFlow)

    selector = FlowSelector(type=name.upper(), args={"scale": 3})
    flow = selector.get_flow()

    assert isinstance(flow, DummyFlow)
    assert flow.scale == 3
    assert name in FlowSelector.available()


@pytest.mark.pruned
def test_flow_selector_unregistered_raises():
    selector = FlowSelector(type=unique_name("missing_flow"), args={})

    with pytest.raises(Exception):
        selector.get_flow()


@pytest.mark.pruned
def test_flow_selector_case_insensitive_lookup():
    name = unique_name("flow_case")
    FlowSelector.register(name.upper())(DummyFlow)

    selector = FlowSelector(type=name.upper(), args={"scale": 8})
    flow = selector.get_flow()

    assert isinstance(flow, DummyFlow)
    assert flow.scale == 8


@pytest.mark.pruned
def test_load_config_from_checkpoint_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        _load_config_from_checkpoint("missing_checkpoint.pt")


@pytest.mark.pruned
def test_load_config_from_checkpoint_returns_module_and_config(tmp_path):
    path = tmp_path / "checkpoint.pt"
    make_checkpoint(
        path,
        architecture_type="abc",
        config={"value": 123},
        input_shape=np.array([1, 2]),
        output_shape=np.array([3, 4]),
    )

    checkpoint_module, checkpoint_config = _load_config_from_checkpoint(path)

    assert checkpoint_module["Architecture"]["type"] == "abc"
    assert checkpoint_module["Architecture"]["config"] == {"value": 123}
    assert checkpoint_config is not None
    assert checkpoint_config.load_path == path
    assert checkpoint_config.strict is True


@pytest.mark.pruned
def test_load_config_from_checkpoint_forwards_strict_flag(tmp_path):
    path = tmp_path / "checkpoint.pt"
    make_checkpoint(path, architecture_type="abc")

    checkpoint_module, checkpoint_config = _load_config_from_checkpoint(
        path,
        strict=False,
    )

    assert checkpoint_module["Architecture"]["type"] == "abc"
    assert checkpoint_config.strict is False


@pytest.mark.pruned
def test_load_config_from_checkpoint_without_module_config_returns_none(tmp_path):
    path = tmp_path / "bad_checkpoint.pt"
    torch.save(
        {
            "input_shape": np.array([1]),
            "output_shape": np.array([1]),
        },
        path,
    )

    checkpoint_module, checkpoint_config = _load_config_from_checkpoint(path)

    assert checkpoint_module is None
    assert checkpoint_config.load_path == path


@pytest.mark.pruned
def test_load_config_from_checkpoint_allows_missing_shapes(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "module_config": {
                "Architecture": {
                    "type": "abc",
                    "config": {"value": 1},
                }
            },
            "input_shape": None,
            "output_shape": None,
        },
        path,
    )

    checkpoint_module, checkpoint_config = _load_config_from_checkpoint(path)

    assert checkpoint_module["Architecture"]["type"] == "abc"
    assert checkpoint_config.load_path == path

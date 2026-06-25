import pytest
import dataclasses

from cccma_ppp.loss.registery import Registery


@dataclasses.dataclass
class DummyDataClass:
    x: int
    y: int


class DummyClass:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y


def test_get_unregistered():
    reg = Registery()
    with pytest.raises(ValueError):
        reg.get("missing")


def test_get_regular_class_with_dict():
    reg = Registery()

    @reg.register("a")
    class A:
        def __init__(self, x=None):
            self.x = x

    obj = reg.get("a", {"x": 5})
    assert obj.x == 5


def test_get_regular_class_without_dict():
    reg = Registery()

    @reg.register("a")
    class A:
        def __init__(self, value):
            self.value = value

    obj = reg.get("a", 10)
    assert obj.value == 10


@pytest.mark.pruned
def test_get_dataclass_with_dict():
    reg = Registery()

    @reg.register("dc")
    @dataclasses.dataclass
    class DC:
        x: int
        y: int

    obj = reg.get("dc", {"x": 1, "y": 2})
    assert obj.x == 1
    assert obj.y == 2


@pytest.mark.pruned
def test_get_dataclass_missing_field_strict():
    reg = Registery()

    @reg.register("dc")
    @dataclasses.dataclass
    class DC:
        x: int
        y: int

    with pytest.raises(Exception):
        reg.get("dc", {"x": 1})


def test_get_dataclass_extra_field_strict():
    reg = Registery()

    @reg.register("dc")
    @dataclasses.dataclass
    class DC:
        x: int
        y: int

    with pytest.raises(Exception):
        reg.get("dc", {"x": 1, "y": 2, "z": 3})


@pytest.mark.pruned
def test_get_with_none_config_regular_class():
    reg = Registery()

    @reg.register("a")
    class A:
        def __init__(self, val=None):
            self.val = val

    obj = reg.get("a")
    assert hasattr(obj, "val")


@pytest.mark.pruned
def test_register_overwrite():
    reg = Registery()

    @reg.register("a")
    class A:
        def __init__(self, x=None):
            self.val = 1

    @reg.register("a")
    class B:
        def __init__(self, x=None):
            self.val = 2

    obj = reg.get("a")

    assert obj.val == 2


@pytest.mark.pruned
def test_get_with_empty_dict():
    reg = Registery()

    @reg.register("a")
    class A:
        def __init__(self):
            self.flag = True

    obj = reg.get("a", {})
    assert obj.flag is True


@pytest.mark.pruned
def test_get_dataclass_with_empty_dict_defaults():
    reg = Registery()

    @reg.register("dc")
    @dataclasses.dataclass
    class DC:
        x: int = 1
        y: int = 2

    obj = reg.get("dc", {})
    assert obj.x == 1 and obj.y == 2

import pytest
import dataclasses
from cccma_ppp.preprocessing.registery import Registery


def test_register_and_get_class_without_config():
    reg = Registery()

    @reg.register("A")
    class A:
        def __init__(self):
            self.value = 1

    obj = reg.get("A", {})
    assert isinstance(obj, A)
    assert obj.value == 1


def test_register_and_get_with_config():
    reg = Registery()

    @reg.register("B")
    class B:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    obj = reg.get("B", {"x": 2, "y": 3})

    assert isinstance(obj, B)
    assert obj.x == 2
    assert obj.y == 3


def test_get_unregistered_raises():
    reg = Registery()

    with pytest.raises(ValueError):
        reg.get("missing")


def test_overwrite_registration():
    reg = Registery()

    @reg.register("C")
    class C1:
        pass

    @reg.register("C")
    class C2:
        pass

    cls = reg.get("C")

    assert cls is C2


def test_get_with_empty_config_dict():
    reg = Registery()

    @reg.register("D")
    class D:
        def __init__(self):
            self.ok = True

    obj = reg.get("D", {})

    assert isinstance(obj, D)
    assert obj.ok


def test_get_with_none_config_passes_none():
    reg = Registery()

    @reg.register("E")
    class E:
        def __init__(self, value=None):
            self.value = value

    cls = reg.get("E")

    assert cls is E


def test_multiple_registrations_independent():
    reg = Registery()

    @reg.register("A")
    class A:
        pass

    @reg.register("B")
    class B:
        pass

    assert reg.get("A") is A
    assert reg.get("B") is B


def test_config_passed_as_kwargs_correctly():
    reg = Registery()

    @reg.register("F")
    class F:
        def __init__(self, a=0, b=0):
            self.a = a
            self.b = b

    obj = reg.get("F", {"a": 5})

    assert obj.a == 5
    assert obj.b == 0


def test_get_dataclass_with_config():
    reg = Registery()

    @dataclasses.dataclass
    class D:
        x: int
        y: int

    reg.register("dataclass")(D)

    obj = reg.get("dataclass", {"x": 1, "y": 2})

    assert isinstance(obj, D)
    assert obj.x == 1
    assert obj.y == 2


def test_get_returns_class_when_config_none():
    reg = Registery()

    @reg.register("A")
    class A:
        pass

    cls = reg.get("A")

    assert cls is A


def test_get_with_invalid_config_raises_typeerror():
    reg = Registery()

    @reg.register("A")
    class A:
        def __init__(self, x):
            self.x = x

    with pytest.raises(TypeError):
        reg.get("A", {})


def test_get_with_non_dict_non_none_config():
    reg = Registery()

    @reg.register("A")
    class A:
        def __init__(self, value):
            self.value = value

    with pytest.raises(TypeError):
        reg.get("A", 123)


def test_get_with_extra_kwargs_raises():
    reg = Registery()

    @reg.register("A")
    class A:
        def __init__(self, x):
            self.x = x

    with pytest.raises(TypeError):
        reg.get("A", {"x": 1, "y": 2})

import pytest

from cccma_ppp.preprocessing.registery import Registery


def test_register_and_get_class():
    registry = Registery()

    @registry.register("test")
    class Dummy:
        pass

    cls = registry.get("test")

    assert cls is Dummy


def test_get_with_config_instantiates():
    registry = Registery()

    @registry.register("test")
    class Dummy:
        def __init__(self, x):
            self.x = x

    obj = registry.get("test", config={"x": 10})

    assert isinstance(obj, Dummy)
    assert obj.x == 10


def test_get_without_config():
    registry = Registery()

    @registry.register("test")
    class Dummy:
        pass

    result = registry.get("test", config=None)

    assert result is Dummy


def test_get_unknown_name():
    registry = Registery()

    with pytest.raises(ValueError) as exc:
        registry.get("missing")

    assert "missing not registered" in str(exc.value)


def test_register_overwrites_existing():
    registry = Registery()

    @registry.register("name")
    class A:
        pass

    @registry.register("name")
    class B:
        pass

    result = registry.get("name")

    assert result is B


def test_get_invalid_config_type():
    registry = Registery()

    @registry.register("test")
    class Dummy:
        def __init__(self, x):
            self.x = x

    with pytest.raises(TypeError):
        registry.get("test", config="not_a_dict")


def test_get_missing_required_argument():
    registry = Registery()

    @registry.register("test")
    class Dummy:
        def __init__(self, x):
            self.x = x

    with pytest.raises(TypeError):
        registry.get("test", config={})

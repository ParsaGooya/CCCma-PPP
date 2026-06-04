import pytest

from cccma_ppp.preprocessing.registery import Registery


# BASIC REGISTRATION


def test_register_and_get_class():
    registry = Registery()

    @registry.register("test")
    class Dummy:
        pass

    cls = registry.get("test")

    assert cls is Dummy


# GET WITH CONFIG (INSTANTIATION)


def test_get_with_config_instantiates():
    registry = Registery()

    @registry.register("test")
    class Dummy:
        def __init__(self, x):
            self.x = x

    obj = registry.get("test", config={"x": 10})

    assert isinstance(obj, Dummy)
    assert obj.x == 10


# GET WITHOUT CONFIG RETURNS CLASS


def test_get_without_config():
    registry = Registery()

    @registry.register("test")
    class Dummy:
        pass

    result = registry.get("test", config=None)

    assert result is Dummy


# UNKNOWN NAME ERROR


def test_get_unknown_name():
    registry = Registery()

    with pytest.raises(ValueError) as exc:
        registry.get("missing")

    assert "missing not registered" in str(exc.value)


# AVAILABLE NAMES


def test_available_returns_registered_names():
    registry = Registery()

    @registry.register("a")
    class A:
        pass

    @registry.register("b")
    class B:
        pass

    names = registry.available()

    assert set(names) == {"a", "b"}


# DECORATOR OVERWRITE BEHAVIOR


def test_register_overwrites_existing():
    registry = Registery()

    @registry.register("name")
    class A:
        pass

    @registry.register("name")
    class B:
        pass

    result = registry.get("name")

    assert result is B  # overwritten


# CONFIG TYPE ERROR


def test_get_invalid_config_type():
    registry = Registery()

    @registry.register("test")
    class Dummy:
        def __init__(self, x):
            self.x = x

    # passing wrong config type
    with pytest.raises(TypeError):
        registry.get("test", config="not_a_dict")


# CONFIG MISSING REQUIRED ARG


def test_get_missing_required_argument():
    registry = Registery()

    @registry.register("test")
    class Dummy:
        def __init__(self, x):
            self.x = x

    with pytest.raises(TypeError):
        registry.get("test", config={})  # missing x


# MULTIPLE REGISTRATIONS


def test_multiple_registry_instances_isolated():
    reg1 = Registery()
    reg2 = Registery()

    @reg1.register("a")
    class A:
        pass

    @reg2.register("b")
    class B:
        pass

    assert reg1.available() == ["a"]
    assert reg2.available() == ["b"]

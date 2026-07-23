from dataclasses import dataclass

import pytest

from cccma_ppp.core.registery import Registery


def test_get_unregistered_raises():
    registry = Registery()

    with pytest.raises(ValueError, match="missing not registered"):
        registry.get("missing")


@pytest.mark.pruned
def test_get_dataclass_from_dict():
    registry = Registery()

    @dataclass
    class Config:
        host: str
        port: int

    registry.register("config")(Config)

    result = registry.get(
        "config",
        {"host": "localhost", "port": 8080},
    )

    assert isinstance(result, Config)
    assert result.host == "localhost"
    assert result.port == 8080


@pytest.mark.pruned
def test_get_regular_class_from_dict():
    registry = Registery()

    @registry.register("service")
    class Service:
        def __init__(self, host, port):
            self.host = host
            self.port = port

    result = registry.get(
        "service",
        {"host": "localhost", "port": 8080},
    )

    assert isinstance(result, Service)
    assert result.host == "localhost"
    assert result.port == 8080


def test_get_regular_class_with_non_dict_config():
    registry = Registery()

    @registry.register("wrapper")
    class Wrapper:
        def __init__(self, value):
            self.value = value

    result = registry.get("wrapper", "abc")

    assert isinstance(result, Wrapper)
    assert result.value == "abc"


@pytest.mark.pruned
def test_get_regular_class_with_none_config():
    registry = Registery()

    @registry.register("wrapper")
    class Wrapper:
        def __init__(self, value):
            self.value = value

    result = registry.get("wrapper", None)

    assert result.value is None
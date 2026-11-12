import dataclasses

import dacite
import pytest

from S2D4P.core.registery import Registery


@dataclasses.dataclass
class ExampleDataclass:
    name: str
    count: int = 1


@dataclasses.dataclass
class NestedConfig:
    enabled: bool


@dataclasses.dataclass
class DataclassWithNestedConfig:
    name: str
    nested: NestedConfig


class ExampleClass:
    def __init__(self, name=None, count=1):
        self.name = name
        self.count = count


class PositionalConfigClass:
    def __init__(self, config):
        self.config = config


class TestGetErrors:
    @pytest.mark.pruned
    def test_unregistered_name_raises_value_error(self):
        registry = Registery()

        with pytest.raises(
            ValueError,
            match="missing not registered",
        ):
            registry.get("missing")

    def test_unregistered_error_lists_available_names(self):
        registry = Registery()
        registry.register("first")(ExampleClass)
        registry.register("second")(PositionalConfigClass)

        with pytest.raises(ValueError) as exc_info:
            registry.get("missing")

        message = str(exc_info.value)

        assert "first" in message
        assert "second" in message

    @pytest.mark.pruned
    def test_unregistered_lookup_does_not_modify_registry(self):
        registry = Registery()
        registry.register("example")(ExampleClass)
        original_modules = registry._modules.copy()

        with pytest.raises(ValueError):
            registry.get("missing")

        assert registry._modules == original_modules


class TestGetDataclass:
    @pytest.mark.pruned
    def test_dictionary_builds_dataclass_instance(self):
        registry = Registery()
        registry.register("example")(ExampleDataclass)

        result = registry.get(
            "example",
            {
                "name": "test",
                "count": 3,
            },
        )

        assert isinstance(result, ExampleDataclass)
        assert result.name == "test"
        assert result.count == 3

    @pytest.mark.pruned
    def test_dictionary_uses_dataclass_defaults(self):
        registry = Registery()
        registry.register("example")(ExampleDataclass)

        result = registry.get(
            "example",
            {
                "name": "test",
            },
        )

        assert result == ExampleDataclass(
            name="test",
            count=1,
        )

    @pytest.mark.pruned
    def test_empty_dictionary_is_forwarded_to_dacite(self):
        @dataclasses.dataclass
        class DefaultDataclass:
            name: str = "default"
            count: int = 2

        registry = Registery()
        registry.register("default")(DefaultDataclass)

        result = registry.get(
            "default",
            {},
        )

        assert result == DefaultDataclass(
            name="default",
            count=2,
        )

    @pytest.mark.pruned
    def test_nested_dictionary_builds_nested_dataclass(self):
        registry = Registery()
        registry.register("nested")(DataclassWithNestedConfig)

        result = registry.get(
            "nested",
            {
                "name": "outer",
                "nested": {
                    "enabled": True,
                },
            },
        )

        assert result == DataclassWithNestedConfig(
            name="outer",
            nested=NestedConfig(enabled=True),
        )

    @pytest.mark.pruned
    def test_unknown_dataclass_field_is_rejected(self):
        registry = Registery()
        registry.register("example")(ExampleDataclass)

        with pytest.raises(dacite.UnexpectedDataError):
            registry.get(
                "example",
                {
                    "name": "test",
                    "unknown": "value",
                },
            )

    @pytest.mark.pruned
    def test_missing_required_dataclass_field_is_rejected(self):
        registry = Registery()
        registry.register("example")(ExampleDataclass)

        with pytest.raises(dacite.MissingValueError):
            registry.get(
                "example",
                {
                    "count": 3,
                },
            )

    @pytest.mark.pruned
    def test_invalid_dataclass_field_type_is_rejected(self):
        registry = Registery()
        registry.register("example")(ExampleDataclass)

        with pytest.raises(dacite.WrongTypeError):
            registry.get(
                "example",
                {
                    "name": "test",
                    "count": "not-an-integer",
                },
            )

    def test_dataclass_dictionary_uses_strict_dacite_configuration(
        self,
        monkeypatch,
    ):
        registry = Registery()
        registry.register("example")(ExampleDataclass)

        expected = ExampleDataclass(
            name="constructed",
            count=5,
        )

        calls = {}

        def fake_from_dict(*, data_class, data, config):
            calls["data_class"] = data_class
            calls["data"] = data
            calls["config"] = config
            return expected

        monkeypatch.setattr(
            dacite,
            "from_dict",
            fake_from_dict,
        )

        result = registry.get(
            "example",
            {
                "name": "test",
                "count": 3,
            },
        )

        assert result is expected
        assert calls["data_class"] is ExampleDataclass
        assert calls["data"] == {
            "name": "test",
            "count": 3,
        }
        assert isinstance(calls["config"], dacite.Config)
        assert calls["config"].strict is True


class TestGetRegularClass:
    @pytest.mark.pruned
    def test_dictionary_is_expanded_as_keyword_arguments(self):
        registry = Registery()
        registry.register("example")(ExampleClass)

        result = registry.get(
            "example",
            {
                "name": "test",
                "count": 4,
            },
        )

        assert isinstance(result, ExampleClass)
        assert result.name == "test"
        assert result.count == 4

    @pytest.mark.pruned
    def test_empty_dictionary_constructs_class_without_arguments(self):
        registry = Registery()
        registry.register("example")(ExampleClass)

        result = registry.get(
            "example",
            {},
        )

        assert result.name is None
        assert result.count == 1

    @pytest.mark.pruned
    def test_unknown_keyword_argument_is_rejected(self):
        registry = Registery()
        registry.register("example")(ExampleClass)

        with pytest.raises(
            TypeError,
            match="unexpected keyword argument",
        ):
            registry.get(
                "example",
                {
                    "unknown": "value",
                },
            )

    @pytest.mark.pruned
    def test_non_dictionary_config_is_passed_positionally(self):
        registry = Registery()
        registry.register("positional")(PositionalConfigClass)
        config = SimpleConfig(value=7)

        result = registry.get(
            "positional",
            config,
        )

        assert isinstance(result, PositionalConfigClass)
        assert result.config is config

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "config",
        [
            "value",
            4,
            [1, 2],
            ("a", "b"),
            object(),
        ],
    )
    def test_non_dictionary_values_are_preserved(self, config):
        registry = Registery()
        registry.register("positional")(PositionalConfigClass)

        result = registry.get(
            "positional",
            config,
        )

        assert result.config is config

    @pytest.mark.pruned
    def test_none_is_passed_positionally(self):
        registry = Registery()
        registry.register("positional")(PositionalConfigClass)

        result = registry.get(
            "positional",
            None,
        )

        assert result.config is None

    def test_existing_instance_is_passed_positionally(self):
        registry = Registery()
        registry.register("positional")(PositionalConfigClass)
        existing = ExampleClass(
            name="existing",
            count=6,
        )

        result = registry.get(
            "positional",
            existing,
        )

        assert result.config is existing


@dataclasses.dataclass
class SimpleConfig:
    value: int

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

import dacite
import pytest

from S2D4P.loss.registery import Registery


@dataclasses.dataclass
class ExampleDataclass:
    name: str
    count: int = 1


@dataclasses.dataclass
class NestedDataclass:
    enabled: bool


@dataclasses.dataclass
class ExampleNestedDataclass:
    name: str
    nested: NestedDataclass


@dataclasses.dataclass
class DefaultDataclass:
    name: str = "default"
    count: int = 1


class ExampleClass:
    def __init__(
        self,
        name=None,
        count=1,
    ):
        self.name = name
        self.count = count


class VariadicClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class PositionalConfigClass:
    def __init__(self, config):
        self.config = config


class NoArgumentClass:
    def __init__(self):
        self.initialized = True


class TestGetErrors:
    @pytest.mark.pruned
    def test_unregistered_name_raises_value_error(self):
        registry = Registery()

        with pytest.raises(
            ValueError,
            match="missing not registered",
        ):
            registry.get("missing")

    @pytest.mark.pruned
    def test_unregistered_error_lists_available_names(self):
        registry = Registery()
        registry.register("first")(ExampleClass)
        registry.register("second")(ExampleDataclass)

        with pytest.raises(ValueError) as exc_info:
            registry.get("missing")

        message = str(exc_info.value)

        assert "first" in message
        assert "second" in message

    def test_failed_lookup_does_not_modify_registry(self):
        registry = Registery()
        registry.register("example")(ExampleClass)
        original_modules = registry._modules.copy()

        with pytest.raises(ValueError):
            registry.get("missing")

        assert registry._modules == original_modules


class TestGetDataclass:
    @pytest.mark.pruned
    def test_dictionary_constructs_dataclass(self):
        registry = Registery()
        registry.register("example")(ExampleDataclass)

        result = registry.get(
            "example",
            {
                "name": "test",
                "count": 3,
            },
        )

        assert result == ExampleDataclass(
            name="test",
            count=3,
        )

    @pytest.mark.pruned
    def test_dictionary_uses_dataclass_default_values(self):
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
    def test_empty_dictionary_constructs_default_dataclass(self):
        registry = Registery()
        registry.register("default")(DefaultDataclass)

        result = registry.get(
            "default",
            {},
        )

        assert result == DefaultDataclass()

    @pytest.mark.pruned
    def test_nested_dictionary_constructs_nested_dataclass(self):
        registry = Registery()
        registry.register("nested")(ExampleNestedDataclass)

        result = registry.get(
            "nested",
            {
                "name": "outer",
                "nested": {
                    "enabled": True,
                },
            },
        )

        assert result == ExampleNestedDataclass(
            name="outer",
            nested=NestedDataclass(enabled=True),
        )

    @pytest.mark.pruned
    def test_unexpected_dataclass_field_is_rejected(self):
        registry = Registery()
        registry.register("example")(ExampleDataclass)

        with pytest.raises(dacite.UnexpectedDataError):
            registry.get(
                "example",
                {
                    "name": "test",
                    "unexpected": "value",
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
    def test_wrong_dataclass_field_type_is_rejected(self):
        registry = Registery()
        registry.register("example")(ExampleDataclass)

        with pytest.raises(dacite.WrongTypeError):
            registry.get(
                "example",
                {
                    "name": "test",
                    "count": "invalid",
                },
            )

    @pytest.mark.pruned
    def test_dataclass_uses_strict_dacite_config(self):
        registry = Registery()
        registry.register("example")(ExampleDataclass)
        expected = ExampleDataclass(
            name="constructed",
            count=5,
        )

        with patch.object(
            dacite,
            "from_dict",
            return_value=expected,
        ) as from_dict:
            result = registry.get(
                "example",
                {
                    "name": "test",
                    "count": 2,
                },
            )

        assert result is expected

        from_dict.assert_called_once()
        call = from_dict.call_args

        assert call.kwargs["data_class"] is ExampleDataclass
        assert call.kwargs["data"] == {
            "name": "test",
            "count": 2,
        }
        assert isinstance(
            call.kwargs["config"],
            dacite.Config,
        )
        assert call.kwargs["config"].strict is True

    def test_dataclass_input_dictionary_is_not_modified(self):
        registry = Registery()
        registry.register("example")(ExampleDataclass)

        config = {
            "name": "test",
            "count": 3,
        }
        original = config.copy()

        registry.get(
            "example",
            config,
        )

        assert config == original


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
    def test_dictionary_uses_constructor_defaults(self):
        registry = Registery()
        registry.register("example")(ExampleClass)

        result = registry.get(
            "example",
            {
                "name": "test",
            },
        )

        assert result.name == "test"
        assert result.count == 1

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
    def test_dictionary_passes_all_values_to_variadic_class(self):
        registry = Registery()
        registry.register("variadic")(VariadicClass)

        config = {
            "name": "test",
            "count": 3,
            "enabled": False,
        }

        result = registry.get(
            "variadic",
            config,
        )

        assert result.kwargs == config

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

    def test_constructor_exception_propagates(self):
        class FailingClass:
            def __init__(self, **kwargs):
                raise RuntimeError("construction failed")

        registry = Registery()
        registry.register("failing")(FailingClass)

        with pytest.raises(
            RuntimeError,
            match="construction failed",
        ):
            registry.get(
                "failing",
                {
                    "value": 1,
                },
            )

    @pytest.mark.pruned
    def test_input_dictionary_is_not_modified(self):
        registry = Registery()
        registry.register("example")(ExampleClass)

        config = {
            "name": "test",
            "count": 3,
        }
        original = config.copy()

        registry.get(
            "example",
            config,
        )

        assert config == original


class TestGetNonDictionaryConfig:
    @pytest.mark.pruned
    def test_none_is_passed_positionally(self):
        registry = Registery()
        registry.register("positional")(PositionalConfigClass)

        result = registry.get(
            "positional",
            None,
        )

        assert result.config is None

    @pytest.mark.pruned
    def test_omitted_config_passes_none_positionally(self):
        registry = Registery()
        registry.register("positional")(PositionalConfigClass)

        result = registry.get("positional")

        assert result.config is None

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "config",
        [
            "value",
            1,
            1.5,
            [1, 2],
            ("a", "b"),
            object(),
        ],
    )
    def test_non_dictionary_config_is_passed_positionally(
        self,
        config,
    ):
        registry = Registery()
        registry.register("positional")(PositionalConfigClass)

        result = registry.get(
            "positional",
            config,
        )

        assert result.config is config

    def test_dataclass_instance_is_passed_positionally(self):
        registry = Registery()
        registry.register("positional")(PositionalConfigClass)
        config = ExampleDataclass(
            name="test",
            count=2,
        )

        result = registry.get(
            "positional",
            config,
        )

        assert result.config is config

    @pytest.mark.pruned
    def test_namespace_is_passed_positionally(self):
        registry = Registery()
        registry.register("positional")(PositionalConfigClass)
        config = SimpleNamespace(
            name="test",
            count=2,
        )

        result = registry.get(
            "positional",
            config,
        )

        assert result.config is config

    @pytest.mark.pruned
    def test_none_fails_for_no_argument_constructor(self):
        registry = Registery()
        registry.register("no_argument")(NoArgumentClass)

        with pytest.raises(TypeError):
            registry.get("no_argument")

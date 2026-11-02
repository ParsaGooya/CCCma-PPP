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


class TestRegisteryInitialization:
    def test_initializes_with_empty_modules(self):
        registry = Registery()

        assert registry._modules == {}

    def test_available_is_empty_initially(self):
        registry = Registery()

        assert registry.available() == []


class TestRegister:
    def test_returns_callable_decorator(self):
        registry = Registery()

        decorator = registry.register("example")

        assert callable(decorator)

    def test_decorator_returns_original_class(self):
        registry = Registery()
        decorator = registry.register("example")

        result = decorator(ExampleClass)

        assert result is ExampleClass

    def test_register_stores_class(self):
        registry = Registery()

        registry.register("example")(ExampleClass)

        assert registry._modules["example"] is ExampleClass

    def test_register_supports_decorator_syntax(self):
        registry = Registery()

        @registry.register("decorated")
        class DecoratedClass:
            pass

        assert registry._modules["decorated"] is DecoratedClass

    def test_register_accepts_non_string_name(self):
        registry = Registery()
        name = ("example", 1)

        registry.register(name)(ExampleClass)

        assert registry._modules[name] is ExampleClass

    def test_registering_same_name_replaces_existing_class(self):
        registry = Registery()

        class FirstClass:
            pass

        class SecondClass:
            pass

        registry.register("example")(FirstClass)
        registry.register("example")(SecondClass)

        assert registry._modules["example"] is SecondClass

    def test_same_class_can_be_registered_under_multiple_names(self):
        registry = Registery()

        registry.register("first")(ExampleClass)
        registry.register("second")(ExampleClass)

        assert registry._modules["first"] is ExampleClass
        assert registry._modules["second"] is ExampleClass


class TestGetErrors:
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

    def test_empty_dictionary_constructs_default_dataclass(self):
        registry = Registery()
        registry.register("default")(DefaultDataclass)

        result = registry.get(
            "default",
            {},
        )

        assert result == DefaultDataclass()

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

    def test_empty_dictionary_constructs_class_without_arguments(self):
        registry = Registery()
        registry.register("example")(ExampleClass)

        result = registry.get(
            "example",
            {},
        )

        assert result.name is None
        assert result.count == 1

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
    def test_none_is_passed_positionally(self):
        registry = Registery()
        registry.register("positional")(PositionalConfigClass)

        result = registry.get(
            "positional",
            None,
        )

        assert result.config is None

    def test_omitted_config_passes_none_positionally(self):
        registry = Registery()
        registry.register("positional")(PositionalConfigClass)

        result = registry.get("positional")

        assert result.config is None

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

    def test_none_fails_for_no_argument_constructor(self):
        registry = Registery()
        registry.register("no_argument")(NoArgumentClass)

        with pytest.raises(TypeError):
            registry.get("no_argument")


class TestAvailable:
    def test_returns_registered_names(self):
        registry = Registery()
        registry.register("first")(ExampleClass)
        registry.register("second")(ExampleDataclass)

        assert registry.available() == [
            "first",
            "second",
        ]

    def test_preserves_registration_order(self):
        registry = Registery()

        registry.register("third")(ExampleClass)
        registry.register("first")(ExampleDataclass)
        registry.register("second")(PositionalConfigClass)

        assert registry.available() == [
            "third",
            "first",
            "second",
        ]

    def test_replacing_registration_preserves_original_order(self):
        registry = Registery()

        class Replacement:
            pass

        registry.register("first")(ExampleClass)
        registry.register("second")(ExampleDataclass)
        registry.register("first")(Replacement)

        assert registry.available() == [
            "first",
            "second",
        ]
        assert registry._modules["first"] is Replacement

    def test_returns_new_list_each_time(self):
        registry = Registery()
        registry.register("example")(ExampleClass)

        first = registry.available()
        second = registry.available()

        assert first == second
        assert first is not second

    def test_mutating_result_does_not_modify_registry(self):
        registry = Registery()
        registry.register("example")(ExampleClass)

        available = registry.available()
        available.append("fake")

        assert registry.available() == ["example"]
        assert "fake" not in registry._modules

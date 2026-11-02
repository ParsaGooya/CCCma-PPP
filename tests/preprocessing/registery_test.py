import pytest

from S2D4P.preprocessing.registery import Registery


class DefaultClass:
    def __init__(self):
        self.created = True


class ConfigurableClass:
    def __init__(
        self,
        name,
        count=1,
        enabled=True,
    ):
        self.name = name
        self.count = count
        self.enabled = enabled


class VariadicClass:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestRegister:
    @pytest.mark.pruned
    def test_returns_callable_decorator(self):
        registry = Registery()

        decorator = registry.register("example")

        assert callable(decorator)

    @pytest.mark.pruned
    def test_decorator_returns_original_class(self):
        registry = Registery()
        decorator = registry.register("example")

        result = decorator(DefaultClass)

        assert result is DefaultClass

    @pytest.mark.pruned
    def test_register_stores_class_under_name(self):
        registry = Registery()

        registry.register("example")(DefaultClass)

        assert registry._modules["example"] is DefaultClass

    @pytest.mark.pruned
    def test_register_supports_decorator_syntax(self):
        registry = Registery()

        @registry.register("decorated")
        class DecoratedClass:
            pass

        assert registry._modules["decorated"] is DecoratedClass

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "name",
        [
            None,
            1,
            1.5,
            [],
            {},
            (),
            object(),
        ],
    )
    def test_rejects_non_string_name(self, name):
        registry = Registery()

        with pytest.raises(
            TypeError,
            match="Name must be string",
        ):
            registry.register(name)

    def test_non_string_error_reports_actual_type(self):
        registry = Registery()

        with pytest.raises(TypeError) as exc_info:
            registry.register(123)

        assert str(type(123)) in str(exc_info.value)

    def test_rejects_empty_name(self):
        registry = Registery()

        with pytest.raises(
            ValueError,
            match="Name cannot be empty string",
        ):
            registry.register("")

    @pytest.mark.pruned
    def test_whitespace_name_is_allowed(self):
        registry = Registery()

        registry.register(" ")(DefaultClass)

        assert registry.available() == [" "]

    @pytest.mark.pruned
    def test_duplicate_name_is_rejected(self):
        registry = Registery()
        registry.register("example")(DefaultClass)

        with pytest.raises(
            ValueError,
            match="'example' already registered",
        ):
            registry.register("example")

    def test_duplicate_name_error_lists_available_names(self):
        registry = Registery()
        registry.register("first")(DefaultClass)
        registry.register("second")(ConfigurableClass)

        with pytest.raises(ValueError) as exc_info:
            registry.register("first")

        message = str(exc_info.value)

        assert "first" in message
        assert "second" in message

    @pytest.mark.pruned
    def test_duplicate_registration_does_not_replace_class(self):
        registry = Registery()
        registry.register("example")(DefaultClass)

        with pytest.raises(ValueError):
            registry.register("example")

        assert registry._modules["example"] is DefaultClass

    @pytest.mark.pruned
    def test_duplicate_name_is_rejected_before_decorator_is_created(self):
        registry = Registery()
        registry.register("example")(DefaultClass)

        with pytest.raises(ValueError):
            registry.register("example")

    @pytest.mark.parametrize(
        "value",
        [
            lambda: None,
            object(),
            1,
            "not-a-class",
            [],
            {},
        ],
    )
    def test_decorator_rejects_non_class_values(self, value):
        registry = Registery()
        decorator = registry.register("example")

        with pytest.raises(
            TypeError,
            match="Can only register classes",
        ):
            decorator(value)

    @pytest.mark.pruned
    def test_non_class_error_reports_value_type_and_registration_name(self):
        registry = Registery()
        decorator = registry.register("example")

        with pytest.raises(TypeError) as exc_info:
            decorator(lambda: None)

        message = str(exc_info.value)

        assert "function" in message
        assert "'example'" in message

    @pytest.mark.pruned
    def test_failed_non_class_registration_does_not_store_value(self):
        registry = Registery()
        decorator = registry.register("example")

        with pytest.raises(TypeError):
            decorator(lambda: None)

        assert registry.available() == []
        assert "example" not in registry._modules

    @pytest.mark.pruned
    def test_same_class_can_be_registered_under_multiple_names(self):
        registry = Registery()

        registry.register("first")(DefaultClass)
        registry.register("second")(DefaultClass)

        assert registry._modules["first"] is DefaultClass
        assert registry._modules["second"] is DefaultClass

    @pytest.mark.pruned
    def test_class_does_not_receive_registry_metadata(self):
        registry = Registery()

        registry.register("example")(DefaultClass)

        assert not hasattr(DefaultClass, "_type")


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
        registry.register("first")(DefaultClass)
        registry.register("second")(ConfigurableClass)

        with pytest.raises(ValueError) as exc_info:
            registry.get("missing")

        message = str(exc_info.value)

        assert "first" in message
        assert "second" in message

    @pytest.mark.pruned
    def test_unregistered_lookup_does_not_modify_registry(self):
        registry = Registery()
        registry.register("example")(DefaultClass)
        original_modules = registry._modules.copy()

        with pytest.raises(ValueError):
            registry.get("missing")

        assert registry._modules == original_modules


class TestGetWithoutConfig:
    @pytest.mark.pruned
    def test_none_constructs_class_without_arguments(self):
        registry = Registery()
        registry.register("default")(DefaultClass)

        result = registry.get(
            "default",
            None,
        )

        assert isinstance(result, DefaultClass)
        assert result.created is True

    @pytest.mark.pruned
    def test_omitted_config_constructs_class_without_arguments(self):
        registry = Registery()
        registry.register("default")(DefaultClass)

        result = registry.get("default")

        assert isinstance(result, DefaultClass)

    @pytest.mark.pruned
    def test_none_fails_when_constructor_requires_arguments(self):
        registry = Registery()
        registry.register("configurable")(ConfigurableClass)

        with pytest.raises(TypeError):
            registry.get(
                "configurable",
                None,
            )


class TestGetWithDictionary:
    @pytest.mark.pruned
    def test_dictionary_is_expanded_as_keyword_arguments(self):
        registry = Registery()
        registry.register("configurable")(ConfigurableClass)

        result = registry.get(
            "configurable",
            {
                "name": "example",
                "count": 4,
                "enabled": False,
            },
        )

        assert isinstance(result, ConfigurableClass)
        assert result.name == "example"
        assert result.count == 4
        assert result.enabled is False

    @pytest.mark.pruned
    def test_dictionary_uses_constructor_defaults(self):
        registry = Registery()
        registry.register("configurable")(ConfigurableClass)

        result = registry.get(
            "configurable",
            {
                "name": "example",
            },
        )

        assert result.name == "example"
        assert result.count == 1
        assert result.enabled is True

    @pytest.mark.pruned
    def test_empty_dictionary_constructs_no_argument_class(self):
        registry = Registery()
        registry.register("default")(DefaultClass)

        result = registry.get(
            "default",
            {},
        )

        assert isinstance(result, DefaultClass)

    @pytest.mark.pruned
    def test_empty_dictionary_fails_for_required_arguments(self):
        registry = Registery()
        registry.register("configurable")(ConfigurableClass)

        with pytest.raises(TypeError):
            registry.get(
                "configurable",
                {},
            )

    @pytest.mark.pruned
    def test_unknown_keyword_is_rejected_by_constructor(self):
        registry = Registery()
        registry.register("configurable")(ConfigurableClass)

        with pytest.raises(
            TypeError,
            match="unexpected keyword argument",
        ):
            registry.get(
                "configurable",
                {
                    "name": "example",
                    "unknown": 42,
                },
            )

    @pytest.mark.pruned
    def test_variadic_class_receives_all_dictionary_values(self):
        registry = Registery()
        registry.register("variadic")(VariadicClass)

        config = {
            "name": "example",
            "count": 3,
            "enabled": False,
        }

        result = registry.get(
            "variadic",
            config,
        )

        assert result.kwargs == config

    def test_configuration_dictionary_is_not_mutated(self):
        registry = Registery()
        registry.register("configurable")(ConfigurableClass)

        config = {
            "name": "example",
            "count": 3,
        }
        original = config.copy()

        registry.get(
            "configurable",
            config,
        )

        assert config == original

    @pytest.mark.pruned
    def test_constructor_exception_propagates(self):
        class FailingClass:
            def __init__(self, **kwargs):
                raise RuntimeError("constructor failed")

        registry = Registery()
        registry.register("failing")(FailingClass)

        with pytest.raises(
            RuntimeError,
            match="constructor failed",
        ):
            registry.get(
                "failing",
                {
                    "value": 1,
                },
            )


class TestGetInvalidConfig:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "config",
        [
            "value",
            1,
            1.5,
            [],
            (),
            set(),
            object(),
        ],
    )
    def test_rejects_non_dictionary_config(self, config):
        registry = Registery()
        registry.register("example")(DefaultClass)

        with pytest.raises(
            TypeError,
            match="Config must be dict or None",
        ):
            registry.get(
                "example",
                config,
            )

    @pytest.mark.pruned
    def test_invalid_config_error_reports_config_type(self):
        registry = Registery()
        registry.register("example")(DefaultClass)

        with pytest.raises(TypeError) as exc_info:
            registry.get(
                "example",
                [1, 2],
            )

        assert "list" in str(exc_info.value)

    @pytest.mark.pruned
    def test_invalid_config_error_reports_class_name(self):
        registry = Registery()
        registry.register("example")(DefaultClass)

        with pytest.raises(TypeError) as exc_info:
            registry.get(
                "example",
                [1, 2],
            )

        assert "DefaultClass" in str(exc_info.value)

    @pytest.mark.pruned
    def test_invalid_config_error_reports_config_value(self):
        registry = Registery()
        registry.register("example")(DefaultClass)
        config = ["invalid", "configuration"]

        with pytest.raises(TypeError) as exc_info:
            registry.get(
                "example",
                config,
            )

        assert str(config) in str(exc_info.value)


class TestAvailable:
    @pytest.mark.pruned
    def test_returns_registered_names(self):
        registry = Registery()
        registry.register("first")(DefaultClass)
        registry.register("second")(ConfigurableClass)

        assert registry.available() == [
            "first",
            "second",
        ]

    @pytest.mark.pruned
    def test_preserves_registration_order(self):
        registry = Registery()

        registry.register("third")(DefaultClass)
        registry.register("first")(ConfigurableClass)
        registry.register("second")(VariadicClass)

        assert registry.available() == [
            "third",
            "first",
            "second",
        ]

    @pytest.mark.pruned
    def test_returns_new_list_each_time(self):
        registry = Registery()
        registry.register("example")(DefaultClass)

        first = registry.available()
        second = registry.available()

        assert first == second
        assert first is not second

    @pytest.mark.pruned
    def test_mutating_result_does_not_modify_registry(self):
        registry = Registery()
        registry.register("example")(DefaultClass)

        available = registry.available()
        available.append("fake")

        assert registry.available() == ["example"]
        assert "fake" not in registry._modules

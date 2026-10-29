from __future__ import annotations

import dataclasses
from unittest.mock import Mock

import pytest

from S2D4P.preprocessing.registery import Registery
from S2D4P.preprocessing.selector import PreprocessingStepSelector


class DummyPreprocessor:
    def __init__(
        self,
        value=1,
        enabled=True,
    ):
        self.value = value
        self.enabled = enabled


@pytest.fixture
def isolated_registry(monkeypatch):
    registry = Registery()

    monkeypatch.setattr(
        PreprocessingStepSelector,
        "registery",
        registry,
    )

    return registry


class TestPreprocessingStepSelectorInitialization:
    @pytest.mark.pruned
    def test_constructs_registered_step_without_arguments(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        selector = PreprocessingStepSelector(
            name="dummy",
        )

        assert selector.name == "dummy"
        assert selector.args is None

    @pytest.mark.pruned
    def test_constructs_registered_step_with_arguments(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        args = {
            "value": 4,
            "enabled": False,
        }

        selector = PreprocessingStepSelector(
            name="dummy",
            args=args,
        )

        assert selector.name == "dummy"
        assert selector.args is args

    @pytest.mark.pruned
    def test_name_is_normalized_to_lowercase(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        selector = PreprocessingStepSelector(
            name="DuMmY",
        )

        assert selector.name == "dummy"

    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "name",
        [
            "DUMMY",
            "Dummy",
            "dummy",
            "dUmMy",
        ],
    )
    def test_registered_name_lookup_is_case_insensitive(
        self,
        isolated_registry,
        name,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        selector = PreprocessingStepSelector(
            name=name,
        )

        assert selector.name == "dummy"

    @pytest.mark.pruned
    def test_name_is_normalized_before_availability_check(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("standardizer")
        class Standardizer:
            pass

        selector = PreprocessingStepSelector(
            name="STANDARDIZER",
        )

        assert selector.name == "standardizer"

    @pytest.mark.pruned
    def test_args_dictionary_is_not_copied(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        args = {
            "value": 4,
        }

        selector = PreprocessingStepSelector(
            name="dummy",
            args=args,
        )

        assert selector.args is args

    @pytest.mark.pruned
    def test_empty_arguments_dictionary_is_preserved(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        args = {}

        selector = PreprocessingStepSelector(
            name="dummy",
            args=args,
        )

        assert selector.args is args
        assert selector.args == {}

    @pytest.mark.pruned
    def test_selector_is_a_dataclass(self):
        assert dataclasses.is_dataclass(PreprocessingStepSelector)

    @pytest.mark.pruned
    def test_registry_is_class_variable(self):
        dataclass_fields = {
            field.name for field in dataclasses.fields(PreprocessingStepSelector)
        }

        assert "registery" not in dataclass_fields


class TestPreprocessingStepSelectorArgumentValidation:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "args",
        [
            [],
            (),
            "value",
            1,
            1.5,
            object(),
        ],
    )
    def test_rejects_non_dictionary_arguments(
        self,
        isolated_registry,
        args,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        with pytest.raises(
            TypeError,
            match="Args must be dict or None",
        ):
            PreprocessingStepSelector(
                name="dummy",
                args=args,
            )

    def test_argument_error_reports_argument_type(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        with pytest.raises(
            TypeError,
            match="list",
        ):
            PreprocessingStepSelector(
                name="dummy",
                args=[
                    1,
                    2,
                ],
            )

    @pytest.mark.pruned
    def test_argument_error_reports_provided_value(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        with pytest.raises(
            TypeError,
            match=r"\[1, 2\]",
        ):
            PreprocessingStepSelector(
                name="dummy",
                args=[
                    1,
                    2,
                ],
            )

    @pytest.mark.pruned
    def test_none_arguments_are_accepted(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        selector = PreprocessingStepSelector(
            name="dummy",
            args=None,
        )

        assert selector.args is None

    @pytest.mark.pruned
    def test_dictionary_arguments_are_accepted(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        selector = PreprocessingStepSelector(
            name="dummy",
            args={
                "value": 2,
            },
        )

        assert selector.args == {
            "value": 2,
        }

    @pytest.mark.pruned
    def test_arguments_are_validated_before_name(
        self,
        isolated_registry,
    ):
        with pytest.raises(
            TypeError,
            match="Args must be dict or None",
        ):
            PreprocessingStepSelector(
                name=123,
                args=[],
            )


class TestPreprocessingStepSelectorNameValidation:
    @pytest.mark.pruned
    @pytest.mark.parametrize(
        "name",
        [
            None,
            1,
            1.5,
            [],
            {},
            object(),
        ],
    )
    def test_rejects_non_string_name(
        self,
        isolated_registry,
        name,
    ):
        with pytest.raises(
            TypeError,
            match="Step name must be string",
        ):
            PreprocessingStepSelector(
                name=name,
            )

    def test_name_type_error_reports_actual_type(
        self,
        isolated_registry,
    ):
        with pytest.raises(
            TypeError,
            match="list",
        ):
            PreprocessingStepSelector(
                name=[],
            )

    def test_rejects_empty_name(
        self,
        isolated_registry,
    ):
        with pytest.raises(
            ValueError,
            match="Step name cannot be empty",
        ):
            PreprocessingStepSelector(
                name="",
            )

    @pytest.mark.pruned
    def test_whitespace_name_is_not_considered_empty(
        self,
        isolated_registry,
    ):
        @isolated_registry.register(" ")
        class WhitespaceStep:
            pass

        selector = PreprocessingStepSelector(
            name=" ",
        )

        assert selector.name == " "

    @pytest.mark.pruned
    def test_name_is_not_stripped(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        with pytest.raises(
            ValueError,
            match="not registered",
        ):
            PreprocessingStepSelector(
                name=" dummy ",
            )


class TestPreprocessingStepSelectorRegistrationValidation:
    @pytest.mark.pruned
    def test_rejects_unregistered_step(
        self,
        isolated_registry,
    ):
        with pytest.raises(
            ValueError,
            match="is not registered",
        ):
            PreprocessingStepSelector(
                name="missing",
            )

    @pytest.mark.pruned
    def test_unregistered_error_reports_normalized_name(
        self,
        isolated_registry,
    ):
        with pytest.raises(
            ValueError,
            match="'missing'",
        ):
            PreprocessingStepSelector(
                name="MISSING",
            )

    @pytest.mark.pruned
    def test_unregistered_error_lists_available_steps(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("first")
        class First:
            pass

        @isolated_registry.register("second")
        class Second:
            pass

        with pytest.raises(
            ValueError,
        ) as error:
            PreprocessingStepSelector(
                name="missing",
            )

        message = str(error.value)

        assert "first" in message
        assert "second" in message
        assert "Available" in message

    @pytest.mark.pruned
    def test_empty_registry_is_reported(
        self,
        isolated_registry,
    ):
        with pytest.raises(
            ValueError,
            match=r"Available: \[\]",
        ):
            PreprocessingStepSelector(
                name="missing",
            )


class TestGetPreprocessor:
    @pytest.mark.pruned
    def test_builds_registered_preprocessor_without_arguments(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            def __init__(self):
                self.value = 1

        selector = PreprocessingStepSelector(
            name="dummy",
        )

        result = selector.get_preprocessor()

        assert isinstance(
            result,
            RegisteredDummy,
        )
        assert result.value == 1

    def test_builds_registered_preprocessor_with_arguments(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            def __init__(
                self,
                value,
                enabled=True,
            ):
                self.value = value
                self.enabled = enabled

        selector = PreprocessingStepSelector(
            name="dummy",
            args={
                "value": 7,
                "enabled": False,
            },
        )

        result = selector.get_preprocessor()

        assert isinstance(
            result,
            RegisteredDummy,
        )
        assert result.value == 7
        assert result.enabled is False

    @pytest.mark.pruned
    def test_get_preprocessor_delegates_to_registry(
        self,
        isolated_registry,
        monkeypatch,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        selector = PreprocessingStepSelector(
            name="DUMMY",
            args={
                "value": 3,
            },
        )

        expected = object()
        get = Mock(
            return_value=expected,
        )

        monkeypatch.setattr(
            isolated_registry,
            "get",
            get,
        )

        result = selector.get_preprocessor()

        get.assert_called_once_with(
            "dummy",
            {
                "value": 3,
            },
        )
        assert result is expected

    @pytest.mark.pruned
    def test_none_arguments_are_forwarded_as_none(
        self,
        isolated_registry,
        monkeypatch,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        selector = PreprocessingStepSelector(
            name="dummy",
            args=None,
        )

        expected = object()
        get = Mock(
            return_value=expected,
        )
        monkeypatch.setattr(
            isolated_registry,
            "get",
            get,
        )

        result = selector.get_preprocessor()

        get.assert_called_once_with(
            "dummy",
            None,
        )
        assert result is expected

    @pytest.mark.pruned
    def test_empty_arguments_are_forwarded_as_empty_dictionary(
        self,
        isolated_registry,
        monkeypatch,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        args = {}
        selector = PreprocessingStepSelector(
            name="dummy",
            args=args,
        )

        expected = object()
        get = Mock(
            return_value=expected,
        )
        monkeypatch.setattr(
            isolated_registry,
            "get",
            get,
        )

        result = selector.get_preprocessor()

        get.assert_called_once_with(
            "dummy",
            args,
        )
        assert result is expected

    @pytest.mark.pruned
    def test_constructor_type_error_is_propagated(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            def __init__(
                self,
                required,
            ):
                self.required = required

        selector = PreprocessingStepSelector(
            name="dummy",
            args={},
        )

        with pytest.raises(TypeError):
            selector.get_preprocessor()

    @pytest.mark.pruned
    def test_constructor_runtime_error_is_propagated(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            def __init__(self):
                raise RuntimeError("construction failed")

        selector = PreprocessingStepSelector(
            name="dummy",
        )

        with pytest.raises(
            RuntimeError,
            match="construction failed",
        ):
            selector.get_preprocessor()

    @pytest.mark.pruned
    def test_each_call_creates_new_preprocessor(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        selector = PreprocessingStepSelector(
            name="dummy",
        )

        first = selector.get_preprocessor()
        second = selector.get_preprocessor()

        assert first is not second


class TestClassRegister:
    @pytest.mark.pruned
    def test_register_adds_class_to_registry(
        self,
        isolated_registry,
    ):
        @PreprocessingStepSelector.register("dummy")
        class RegisteredDummy:
            pass

        assert "dummy" in (PreprocessingStepSelector.available())

    @pytest.mark.pruned
    def test_register_returns_original_class(
        self,
        isolated_registry,
    ):
        class RegisteredDummy:
            pass

        returned = PreprocessingStepSelector.register("dummy")(RegisteredDummy)

        assert returned is RegisteredDummy

    @pytest.mark.pruned
    def test_register_normalizes_name_to_lowercase(
        self,
        isolated_registry,
    ):
        @PreprocessingStepSelector.register("DuMmY")
        class RegisteredDummy:
            pass

        assert isolated_registry.available() == [
            "dummy",
        ]

        selector = PreprocessingStepSelector(
            name="DUMMY",
        )
        result = selector.get_preprocessor()

        assert isinstance(
            result,
            RegisteredDummy,
        )

    @pytest.mark.pruned
    def test_register_delegates_to_registry(
        self,
        isolated_registry,
        monkeypatch,
    ):
        decorator = object()
        register = Mock(
            return_value=decorator,
        )
        monkeypatch.setattr(
            isolated_registry,
            "register",
            register,
        )

        result = PreprocessingStepSelector.register("MiXeD")

        register.assert_called_once_with(
            "mixed",
        )
        assert result is decorator

    def test_duplicate_registration_error_is_propagated(
        self,
        isolated_registry,
    ):
        @PreprocessingStepSelector.register("dummy")
        class First:
            pass

        with pytest.raises(
            ValueError,
            match="already registered",
        ):

            @PreprocessingStepSelector.register("DUMMY")
            class Second:
                pass

    @pytest.mark.pruned
    def test_non_string_registration_name_fails_before_registry(
        self,
        isolated_registry,
    ):
        with pytest.raises(
            AttributeError,
        ):
            PreprocessingStepSelector.register(123)

    @pytest.mark.pruned
    def test_non_class_registration_error_is_propagated(
        self,
        isolated_registry,
    ):
        decorator = PreprocessingStepSelector.register("dummy")

        with pytest.raises(
            TypeError,
            match="Can only register classes",
        ):
            decorator(lambda: None)


class TestClassAvailable:
    @pytest.mark.pruned
    def test_available_returns_registry_names(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("first")
        class First:
            pass

        @isolated_registry.register("second")
        class Second:
            pass

        assert PreprocessingStepSelector.available() == [
            "first",
            "second",
        ]

    @pytest.mark.pruned
    def test_available_delegates_to_registry(
        self,
        isolated_registry,
        monkeypatch,
    ):
        available = Mock(
            return_value=[
                "one",
                "two",
            ]
        )
        monkeypatch.setattr(
            isolated_registry,
            "available",
            available,
        )

        result = PreprocessingStepSelector.available()

        available.assert_called_once_with()
        assert result == [
            "one",
            "two",
        ]

    @pytest.mark.pruned
    def test_available_is_empty_for_isolated_empty_registry(
        self,
        isolated_registry,
    ):
        assert PreprocessingStepSelector.available() == []


class TestSharedRegistryBehavior:
    @pytest.mark.pruned
    def test_instances_use_current_class_registry(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        first = PreprocessingStepSelector(
            name="dummy",
        )
        second = PreprocessingStepSelector(
            name="DUMMY",
        )

        assert first.registery is isolated_registry
        assert second.registery is isolated_registry

    @pytest.mark.pruned
    def test_registration_is_visible_to_new_selectors(
        self,
        isolated_registry,
    ):
        @PreprocessingStepSelector.register("dummy")
        class RegisteredDummy:
            pass

        selector = PreprocessingStepSelector(
            name="dummy",
        )

        assert isinstance(
            selector.get_preprocessor(),
            RegisteredDummy,
        )

    def test_replacing_registry_isolates_previous_registrations(
        self,
        isolated_registry,
        monkeypatch,
    ):
        @isolated_registry.register("old")
        class Old:
            pass

        replacement = Registery()

        @replacement.register("new")
        class New:
            pass

        monkeypatch.setattr(
            PreprocessingStepSelector,
            "registery",
            replacement,
        )

        with pytest.raises(
            ValueError,
            match="old.*not registered",
        ):
            PreprocessingStepSelector(
                name="old",
            )

        selector = PreprocessingStepSelector(
            name="new",
        )

        assert isinstance(
            selector.get_preprocessor(),
            New,
        )


class TestDataclassBehavior:
    @pytest.mark.pruned
    def test_equal_selectors_compare_equal(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        first = PreprocessingStepSelector(
            name="DUMMY",
            args={
                "value": 1,
            },
        )
        second = PreprocessingStepSelector(
            name="dummy",
            args={
                "value": 1,
            },
        )

        assert first == second

    @pytest.mark.pruned
    def test_different_names_compare_not_equal(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("first")
        class First:
            pass

        @isolated_registry.register("second")
        class Second:
            pass

        first = PreprocessingStepSelector(
            name="first",
        )
        second = PreprocessingStepSelector(
            name="second",
        )

        assert first != second

    @pytest.mark.pruned
    def test_different_arguments_compare_not_equal(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        first = PreprocessingStepSelector(
            name="dummy",
            args={
                "value": 1,
            },
        )
        second = PreprocessingStepSelector(
            name="dummy",
            args={
                "value": 2,
            },
        )

        assert first != second

    @pytest.mark.pruned
    def test_repr_contains_normalized_name(
        self,
        isolated_registry,
    ):
        @isolated_registry.register("dummy")
        class RegisteredDummy:
            pass

        selector = PreprocessingStepSelector(
            name="DUMMY",
            args={
                "value": 1,
            },
        )

        representation = repr(selector)

        assert "dummy" in representation
        assert "value" in representation

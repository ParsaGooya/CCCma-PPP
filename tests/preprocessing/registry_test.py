import dataclasses

import pytest

from cccma_ppp.preprocessing.registery import Registery


@pytest.mark.pruned
def test_register_and_get_class_without_config():
    reg = Registery()

    @reg.register("A")
    class A:
        def __init__(self):
            self.value = 1

    obj = reg.get("A", {})
    assert isinstance(obj, A)
    assert obj.value == 1


@pytest.mark.pruned
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


def legacy_overwrite_registration():
    reg = Registery()

    @reg.register("C")
    class C1:
        pass

    @reg.register("C")
    class C2:
        pass

    cls = reg.get("C")

    assert isinstance(cls, C2)


@pytest.mark.pruned
def test_get_with_empty_config_dict():
    reg = Registery()

    @reg.register("D")
    class D:
        def __init__(self):
            self.ok = True

    obj = reg.get("D", {})

    assert isinstance(obj, D)
    assert obj.ok


@pytest.mark.pruned
def test_get_with_none_config_passes_none():
    reg = Registery()

    @reg.register("E")
    class E:
        def __init__(self, value=None):
            self.value = value

    cls = reg.get("E")

    assert isinstance(cls, E)


@pytest.mark.pruned
def test_multiple_registrations_independent():
    reg = Registery()

    @reg.register("A")
    class A:
        pass

    @reg.register("B")
    class B:
        pass

    assert isinstance(reg.get("A"), A)
    assert isinstance(reg.get("B"), B)


@pytest.mark.pruned
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


@pytest.mark.pruned
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


@pytest.mark.pruned
def test_get_returns_class_when_config_none():
    reg = Registery()

    @reg.register("A")
    class A:
        pass

    cls = reg.get("A")

    assert isinstance(cls, A)


@pytest.mark.pruned
def test_get_with_invalid_config_raises_typeerror():
    reg = Registery()

    @reg.register("A")
    class A:
        def __init__(self, x):
            self.x = x

    with pytest.raises(TypeError):
        reg.get("A", {})


@pytest.mark.pruned
def test_get_with_non_dict_non_none_config():
    reg = Registery()

    @reg.register("A")
    class A:
        def __init__(self, value):
            self.value = value

    with pytest.raises(TypeError):
        reg.get("A", 123)


@pytest.mark.pruned
def test_get_with_extra_kwargs_raises():
    reg = Registery()

    @reg.register("A")
    class A:
        def __init__(self, x):
            self.x = x

    with pytest.raises(TypeError):
        reg.get("A", {"x": 1, "y": 2})


# Remove test due to no coverage
@pytest.mark.pruned
def test_available_is_empty_for_new_registry():
    reg = Registery()

    assert reg.available() == []


@pytest.mark.pruned
def test_available_returns_registered_names_in_order():
    reg = Registery()

    @reg.register("first")
    class First:
        pass

    @reg.register("second")
    class Second:
        pass

    @reg.register("third")
    class Third:
        pass

    assert reg.available() == [
        "first",
        "second",
        "third",
    ]


@pytest.mark.pruned
def test_registered_class_is_returned_by_decorator():
    reg = Registery()

    class Example:
        pass

    returned = reg.register("example")(Example)

    assert returned is Example
    assert reg._modules["example"] is Example


@pytest.mark.pruned
def test_register_decorator_preserves_class_identity():
    reg = Registery()

    @reg.register("example")
    class Example:
        pass

    assert reg._modules["example"] is Example


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
def test_register_rejects_non_string_name(name):
    reg = Registery()

    with pytest.raises(
        TypeError,
        match="Name must be string",
    ):
        reg.register(name)

    assert reg.available() == []


@pytest.mark.pruned
def test_register_type_error_reports_actual_type():
    reg = Registery()

    with pytest.raises(
        TypeError,
        match="list",
    ):
        reg.register([])


def test_register_rejects_empty_name():
    reg = Registery()

    with pytest.raises(
        ValueError,
        match="Name cannot be empty string",
    ):
        reg.register("")

    assert reg.available() == []


@pytest.mark.pruned
def test_register_accepts_whitespace_name():
    reg = Registery()

    @reg.register(" ")
    class WhitespaceName:
        pass

    assert isinstance(
        reg.get(" "),
        WhitespaceName,
    )


@pytest.mark.pruned
def test_register_accepts_case_distinct_names():
    reg = Registery()

    @reg.register("model")
    class LowercaseModel:
        pass

    @reg.register("MODEL")
    class UppercaseModel:
        pass

    assert isinstance(
        reg.get("model"),
        LowercaseModel,
    )
    assert isinstance(
        reg.get("MODEL"),
        UppercaseModel,
    )


@pytest.mark.pruned
def test_duplicate_registration_raises():
    reg = Registery()

    @reg.register("duplicate")
    class First:
        pass

    with pytest.raises(
        ValueError,
        match="already registered",
    ):

        @reg.register("duplicate")
        class Second:
            pass


@pytest.mark.pruned
def test_duplicate_registration_error_lists_available_names():
    reg = Registery()

    @reg.register("first")
    class First:
        pass

    @reg.register("second")
    class Second:
        pass

    with pytest.raises(
        ValueError,
    ) as error:

        @reg.register("first")
        class Duplicate:
            pass

    message = str(error.value)

    assert "first" in message
    assert "second" in message
    assert "Available" in message


@pytest.mark.pruned
@pytest.mark.parametrize(
    "value",
    [
        lambda: None,
        42,
        "not-a-class",
        object(),
    ],
)
def test_register_decorator_rejects_non_class(value):
    reg = Registery()
    decorator = reg.register("invalid")

    with pytest.raises(
        TypeError,
        match="Can only register classes",
    ):
        decorator(value)

    assert "invalid" not in reg.available()


def test_non_class_registration_error_reports_value_type():
    reg = Registery()

    with pytest.raises(
        TypeError,
        match="function",
    ):
        reg.register("callable")(lambda: None)


@pytest.mark.pruned
def test_failed_non_class_registration_does_not_reserve_name():
    reg = Registery()

    with pytest.raises(TypeError):
        reg.register("example")(lambda: None)

    @reg.register("example")
    class Example:
        pass

    assert isinstance(
        reg.get("example"),
        Example,
    )


@pytest.mark.pruned
def test_get_unregistered_error_lists_available_names():
    reg = Registery()

    @reg.register("available")
    class Available:
        pass

    with pytest.raises(
        ValueError,
    ) as error:
        reg.get("missing")

    message = str(error.value)

    assert "missing not registered" in message
    assert "available" in message


@pytest.mark.pruned
def test_get_is_case_sensitive():
    reg = Registery()

    @reg.register("Example")
    class Example:
        pass

    with pytest.raises(
        ValueError,
        match="example not registered",
    ):
        reg.get("example")


@pytest.mark.pruned
def test_get_none_config_calls_constructor_without_arguments():
    reg = Registery()
    calls = []

    @reg.register("example")
    class Example:
        def __init__(self):
            calls.append(())

    result = reg.get(
        "example",
        None,
    )

    assert isinstance(
        result,
        Example,
    )
    assert calls == [
        (),
    ]


@pytest.mark.pruned
def test_get_empty_config_calls_constructor_with_no_keyword_arguments():
    reg = Registery()
    calls = []

    @reg.register("example")
    class Example:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    result = reg.get(
        "example",
        {},
    )

    assert isinstance(
        result,
        Example,
    )
    assert calls == [
        {},
    ]


@pytest.mark.pruned
def test_get_passes_config_as_keyword_arguments():
    reg = Registery()
    received = {}

    @reg.register("example")
    class Example:
        def __init__(self, **kwargs):
            received.update(kwargs)

    config = {
        "alpha": 1,
        "beta": "two",
    }

    reg.get(
        "example",
        config,
    )

    assert received == config


@pytest.mark.pruned
def test_get_does_not_mutate_config_dictionary():
    reg = Registery()

    @reg.register("example")
    class Example:
        def __init__(self, value):
            self.value = value

    config = {
        "value": 3,
    }
    original = config.copy()

    reg.get(
        "example",
        config,
    )

    assert config == original


@pytest.mark.parametrize(
    "config",
    [
        [],
        (),
        "value",
        1,
        1.5,
        object(),
    ],
)
def test_get_rejects_non_dictionary_config(config):
    reg = Registery()

    @reg.register("example")
    class Example:
        pass

    with pytest.raises(
        TypeError,
        match="Config must be dict or None",
    ):
        reg.get(
            "example",
            config,
        )


@pytest.mark.pruned
def test_invalid_config_error_reports_config_type_and_class_name():
    reg = Registery()

    @reg.register("example")
    class Example:
        pass

    with pytest.raises(
        TypeError,
    ) as error:
        reg.get(
            "example",
            [
                1,
                2,
            ],
        )

    message = str(error.value)

    assert "list" in message
    assert "Example" in message


@pytest.mark.pruned
def test_constructor_missing_required_argument_error_propagates():
    reg = Registery()

    @reg.register("example")
    class Example:
        def __init__(self, required):
            self.required = required

    with pytest.raises(TypeError):
        reg.get(
            "example",
            {},
        )


@pytest.mark.pruned
def test_constructor_unexpected_argument_error_propagates():
    reg = Registery()

    @reg.register("example")
    class Example:
        def __init__(self, expected):
            self.expected = expected

    with pytest.raises(TypeError):
        reg.get(
            "example",
            {
                "expected": 1,
                "unexpected": 2,
            },
        )


@pytest.mark.pruned
def test_constructor_exception_propagates():
    reg = Registery()

    @reg.register("example")
    class Example:
        def __init__(self):
            raise RuntimeError("construction failed")

    with pytest.raises(
        RuntimeError,
        match="construction failed",
    ):
        reg.get("example")


@pytest.mark.pruned
def test_each_get_returns_new_instance():
    reg = Registery()

    @reg.register("example")
    class Example:
        pass

    first = reg.get("example")
    second = reg.get("example")

    assert first is not second
    assert isinstance(
        first,
        Example,
    )
    assert isinstance(
        second,
        Example,
    )


@pytest.mark.pruned
def test_registry_instances_are_independent():
    first_registry = Registery()
    second_registry = Registery()

    @first_registry.register("example")
    class Example:
        pass

    assert first_registry.available() == [
        "example",
    ]
    assert second_registry.available() == []

    with pytest.raises(ValueError):
        second_registry.get("example")


@pytest.mark.pruned
def test_available_returns_new_list():
    reg = Registery()

    @reg.register("example")
    class Example:
        pass

    available = reg.available()
    available.append("modified")

    assert reg.available() == [
        "example",
    ]


@pytest.mark.pruned
def test_available_reflects_later_registrations():
    reg = Registery()

    assert reg.available() == []

    @reg.register("first")
    class First:
        pass

    assert reg.available() == [
        "first",
    ]

    @reg.register("second")
    class Second:
        pass

    assert reg.available() == [
        "first",
        "second",
    ]


@pytest.mark.pruned
def test_dataclass_defaults_are_applied():
    reg = Registery()

    @reg.register("config")
    @dataclasses.dataclass
    class Config:
        required: int
        optional: int = 10

    result = reg.get(
        "config",
        {
            "required": 3,
        },
    )

    assert result.required == 3
    assert result.optional == 10


@pytest.mark.pruned
def test_registered_inheritance_is_supported():
    reg = Registery()

    class Base:
        pass

    @reg.register("child")
    class Child(Base):
        pass

    result = reg.get("child")

    assert isinstance(
        result,
        Child,
    )
    assert isinstance(
        result,
        Base,
    )


@pytest.mark.pruned
def test_class_with_variadic_keywords_is_supported():
    reg = Registery()

    @reg.register("flexible")
    class Flexible:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    result = reg.get(
        "flexible",
        {
            "one": 1,
            "two": 2,
        },
    )

    assert result.kwargs == {
        "one": 1,
        "two": 2,
    }

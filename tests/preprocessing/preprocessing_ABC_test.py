import pytest

from cccma_ppp.preprocessing.preprocessing_ABC import PreprocessModuleABC


# ABSTRACT CLASS INSTANTIATION


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        PreprocessModuleABC()


# VALID SUBCLASS


class DummyPreprocess(PreprocessModuleABC):
    def fit(self, data):
        self.mean = sum(data) / len(data)

    def transform(self, data, **kwargs):
        return [x - self.mean for x in data]

    def inverse_transform(self, data, **kwargs):
        return [x + self.mean for x in data]


def test_valid_subclass_workflow():
    proc = DummyPreprocess()

    data = [1, 2, 3]

    proc.fit(data)
    transformed = proc.transform(data)
    restored = proc.inverse_transform(transformed)

    assert transformed != data
    assert restored == data


# PARTIAL IMPLEMENTATION (SHOULD FAIL)


class IncompletePreprocess1(PreprocessModuleABC):
    def fit(self, data):
        pass

    # missing transform + inverse


class IncompletePreprocess2(PreprocessModuleABC):
    def fit(self, data):
        pass

    def transform(self, data, **kwargs):
        return data

    # missing inverse_transform


def test_incomplete_implementation_fails():
    with pytest.raises(TypeError):
        IncompletePreprocess1()

    with pytest.raises(TypeError):
        IncompletePreprocess2()


# METHOD SIGNATURE FLEXIBILITY


class FlexiblePreprocess(PreprocessModuleABC):
    def fit(self, data):
        pass

    def transform(self, data, **kwargs):
        return data

    def inverse_transform(self, data, **kwargs):
        return data


def test_kwargs_support():
    proc = FlexiblePreprocess()

    data = [1, 2, 3]

    out = proc.transform(data, extra=True)
    inv = proc.inverse_transform(out, something=123)

    assert out == data
    assert inv == data

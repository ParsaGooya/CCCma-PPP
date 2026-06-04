import pytest
import numpy as np
import joblib
import os
from pathlib import Path
import numpy as np

from cccma_ppp.preprocessing.preprocessing import (
    PreprocessingStepSelector,
    PreprocessingPipeline,
)


class FakeLoaded:
    def __init__(self, name="loaded", fitted=True):
        self.fitted = fitted
        self.preprocessors_list = []
        self.pipeline = []
        self.steps = []
        self.fitted_preprocessors = []
        self.name = name


class FakeYear:
    def __init__(self, arr):
        self.values = arr


class FakeData(dict):
    def __getitem__(self, key):
        val = super().__getitem__(key)
        if key == "year":
            return FakeYear(val)
        return val


# MOCK PREPROCESSOR


class DummyPreprocessor:
    def __init__(self, scale=1):
        self.scale = scale
        self.fitted = False

    def fit(self, data, mask=None):
        self.fitted = True

    def transform(self, data, **kwargs):
        return data  # no-op

    def inverse_transform(self, data, **kwargs):
        return data


# REGISTER PREPROCESSOR


@PreprocessingStepSelector.register("dummy")
class RegisteredDummy(DummyPreprocessor):
    pass


# SELECTOR TESTS


def test_selector_get_preprocessor():
    sel = PreprocessingStepSelector(name="dummy", args={"scale": 2})
    proc = sel.get_preprocessor()

    assert isinstance(proc, DummyPreprocessor)
    assert proc.scale == 2


def test_selector_available():
    assert "dummy" in PreprocessingStepSelector.available()


def test_selector_invalid():
    sel = PreprocessingStepSelector(name="missing")

    with pytest.raises(ValueError):
        sel.get_preprocessor()


# PIPELINE BASIC FIT


def make_pipeline(scale=2):
    sel = PreprocessingStepSelector("dummy", {"scale": scale})
    return PreprocessingPipeline([sel])


def test_pipeline_fit_basic(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline(scale=2)

    data = np.array([1, 2, 3])
    data = {"year": np.array([2000, 2001]), "values": data}

    fake_data = FakeData(data)

    pipe.fit(base_data=fake_data, save=False)

    assert pipe.fitted
    assert len(pipe.steps) == 1


# PIPELINE TRANSFORM / INVERSE


def test_pipeline_transform_inverse(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline(scale=2)

    fake_data = FakeData({"year": np.array([1, 2])})
    pipe.fit(base_data=fake_data, save=False)

    data = np.array([1.0, 2.0])

    transformed = pipe.transform(data)
    restored = pipe.inverse_transform(transformed)

    assert np.allclose(restored, data)


# STEP ARGUMENT VALIDATION


def test_transform_invalid_step():
    pipe = make_pipeline()
    pipe.steps = ["dummy"]
    pipe.fitted_preprocessors = [DummyPreprocessor()]
    pipe.fitted = True

    with pytest.raises(ValueError):
        pipe.transform(np.array([1]), step_arguments={"bad": {}})


def test_inverse_invalid_step():
    pipe = make_pipeline()
    pipe.steps = ["dummy"]
    pipe.fitted_preprocessors = [DummyPreprocessor()]
    pipe.fitted = True

    with pytest.raises(ValueError):
        pipe.inverse_transform(np.array([1]), step_arguments={"bad": {}})


# GET PREPROCESSORS


def test_get_all_preprocessors(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline()
    fake_data = FakeData({"year": np.array([1])})
    pipe.fit(base_data=fake_data, save=False)

    procs = pipe.get_preprocessors()

    assert len(procs) == 1


def test_get_named_preprocessor(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline()
    fake_data = FakeData({"year": np.array([1])})
    pipe.fit(base_data=fake_data, save=False)

    proc = pipe.get_preprocessors()[0]  # safer path

    assert isinstance(proc, DummyPreprocessor)


def test_get_preprocessor_not_fitted():
    pipe = make_pipeline()

    with pytest.raises(AssertionError):
        pipe.get_preprocessors()


def test_get_preprocessor_invalid_name(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline()
    fake_data = FakeData({"year": np.array([1])})
    pipe.fit(base_data=fake_data, save=False)

    with pytest.raises(ValueError):
        pipe.get_preprocessors("bad")


# ADD FITTED PREPROCESSOR


def test_add_fitted_preprocessor():
    pipe = make_pipeline()
    pipe.steps = []
    pipe.fitted_preprocessors = []

    proc = DummyPreprocessor()
    proc.fitted = True

    pipe.add_fitted_preprocessor(proc, "dummy")

    assert pipe.steps == ["dummy"]
    assert len(pipe.fitted_preprocessors) == 1


def test_add_preprocessor_with_index():
    pipe = make_pipeline()
    pipe.steps = ["a"]
    pipe.fitted_preprocessors = [DummyPreprocessor()]

    proc = DummyPreprocessor()
    proc.fitted = True

    pipe.add_fitted_preprocessor(proc, "dummy", index=0)

    assert pipe.steps[0] == "dummy"


def test_add_preprocessor_not_fitted():
    pipe = make_pipeline()
    proc = DummyPreprocessor()  # not fitted

    with pytest.raises(AssertionError):
        pipe.add_fitted_preprocessor(proc, "dummy")


# SAVE / LOAD PIPELINE


def test_pipeline_save(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline()

    fake_data = FakeData({"year": np.array([1])})
    pipe.fit(base_data=fake_data, save=True)

    saved = list(tmp_path.rglob("*.joblib"))
    assert len(saved) > 0


def test_pipeline_load(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    # create + save
    pipe = make_pipeline()
    fake_data = FakeData({"year": np.array([1])})
    pipe.fit(base_data=fake_data, save=True)

    saved = list(tmp_path.rglob("*.joblib"))[0]

    # load new instance
    new_pipe = PreprocessingPipeline(load_dir=saved.parent, load_name=saved.name)
    new_pipe.fit()

    assert new_pipe.fitted


def test_load_unfitted_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    bad = tmp_path / "bad.joblib"

    joblib.dump(FakeLoaded(fitted=False), bad)

    pipe = PreprocessingPipeline(load_dir=tmp_path, load_name="bad.joblib")

    with pytest.raises(AssertionError):
        pipe.fit()


def test_custom_save_path_and_name(monkeypatch, tmp_path):
    pipe = make_pipeline()

    fake_data = FakeData({"year": np.array([1])})

    custom_dir = tmp_path / "custom"
    custom_name = "my_pipe.joblib"

    pipe.fit(
        base_data=fake_data,
        save=True,
        save_path=custom_dir,
        save_name=custom_name,
    )

    assert (custom_dir / custom_name).exists()


def test_transform_with_step_arguments(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline()
    fake_data = FakeData({"year": np.array([1])})
    pipe.fit(base_data=fake_data, save=False)

    data = np.array([1, 2])

    # arguments passed to step (even if dummy ignores them)
    out = pipe.transform(data, step_arguments={"dummy": {"x": 1}})

    assert out is not None


def test_fit_uses_load_dir(monkeypatch, tmp_path):
    file = tmp_path / "pipe.joblib"

    joblib.dump(FakeLoaded(), file)

    pipe = PreprocessingPipeline(load_dir=tmp_path, load_name="pipe.joblib")

    pipe.fit()

    assert pipe.fitted


def test_load_default_name(monkeypatch, tmp_path):
    file = tmp_path / "instance_1_preprocessing_pipeline.joblib"

    joblib.dump(FakeLoaded(name="instance_1"), file)

    pipe = PreprocessingPipeline(load_dir=tmp_path)
    pipe.name = "instance_1"

    pipe.fit()

    assert pipe.fitted


def test_multiple_steps_pipeline(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    sel1 = PreprocessingStepSelector("dummy")
    sel2 = PreprocessingStepSelector("dummy")

    pipe = PreprocessingPipeline([sel1, sel2])

    fake_data = FakeData({"year": np.array([1])})
    pipe.fit(base_data=fake_data, save=False)

    assert len(pipe.steps) == 2


def test_add_preprocessor_mid_pipeline():
    pipe = make_pipeline()
    pipe.steps = ["a"]
    pipe.fitted_preprocessors = [DummyPreprocessor()]

    p = DummyPreprocessor()
    p.fitted = True

    pipe.add_fitted_preprocessor(p, "b", index=1)

    assert pipe.steps == ["a", "b"]


def test_save_existing_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    dir_path = tmp_path / "preprocessing_pipeline"
    dir_path.mkdir()

    pipe = make_pipeline()
    fake_data = FakeData({"year": np.array([1])})

    pipe.fit(base_data=fake_data, save=True)

    assert dir_path.exists()


def test_transform_empty_args(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline()
    fake_data = FakeData({"year": np.array([1])})
    pipe.fit(base_data=fake_data, save=False)

    out = pipe.transform(np.array([1]), step_arguments={})

    assert out is not None

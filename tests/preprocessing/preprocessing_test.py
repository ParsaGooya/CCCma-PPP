import pytest
import numpy as np
import joblib
import xarray as xr
from cccma_ppp.preprocessing.preprocessing import (
    PreprocessingStepSelector,
    PreprocessingPipeline,
)
from cccma_ppp.generic.runtime import RuntimeContext


class FakeLoaded:
    def __init__(self, name="loaded", fitted=True):
        self.fitted = fitted
        self.preprocessors_list = []
        self.pipeline = []
        self.steps = []
        self.fitted_preprocessors = []
        self.name = name

        self.fitted_based_year = np.array([1])
        self.reference_coords = {}
        self.reference_var = ["var"]

        self.output_coords = {}


class FakeYear:
    def __init__(self, arr):
        self.values = arr


class FakeData(dict):
    @property
    def dims(self):
        return tuple(self.keys())

    @property
    def data_vars(self):
        return self.keys()

    def __getitem__(self, key):
        value = super().__getitem__(key)

        if key == "year":
            return FakeYear(value)

        return value


class DummyPreprocessor:
    def __init__(self, scale=1):
        self.scale = scale
        self.fitted = False

    def fit(self, data, mask=None):
        self.fitted = True

    def transform(self, data, **kwargs):
        return data

    def inverse_transform(self, data, **kwargs):
        return data


@PreprocessingStepSelector.register("dummy")
class RegisteredDummy(DummyPreprocessor):
    pass


@pytest.mark.pruned
def test_selector_get_preprocessor():
    sel = PreprocessingStepSelector(name="dummy", args={"scale": 2})
    proc = sel.get_preprocessor()

    assert isinstance(proc, DummyPreprocessor)
    assert proc.scale == 2


def test_selector_invalid():
    sel = PreprocessingStepSelector(name="missing")

    with pytest.raises(ValueError):
        sel.get_preprocessor()


def make_pipeline(scale=2):
    sel = PreprocessingStepSelector("dummy", {"scale": scale})
    return PreprocessingPipeline([sel])


@pytest.mark.pruned
def test_pipeline_fit_basic(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline(scale=2)

    data = np.array([1, 2, 3])
    data = {"year": np.array([2000, 2001]), "values": data}

    fake_data = FakeData(data)

    pipe.fit(base_data=fake_data, save=False)

    assert pipe.fitted
    assert len(pipe.steps) == 1


def test_pipeline_transform_inverse(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline(scale=2)

    fake_data = FakeData({"year": np.array([1, 2])})
    pipe.fit(base_data=fake_data, save=False)

    data = np.array([1.0, 2.0])

    transformed = pipe.transform(data)
    restored = pipe.inverse_transform(transformed)

    assert np.allclose(restored, data)


def test_transform_invalid_step():
    pipe = make_pipeline()
    pipe.steps = ["dummy"]
    pipe.fitted_preprocessors = [DummyPreprocessor()]
    pipe.fitted = True

    with pytest.raises(ValueError):
        pipe.transform(np.array([1]), step_arguments={"bad": {}})


@pytest.mark.pruned
def test_inverse_invalid_step():
    pipe = make_pipeline()
    pipe.steps = ["dummy"]
    pipe.fitted_preprocessors = [DummyPreprocessor()]
    pipe.fitted = True

    with pytest.raises(ValueError):
        pipe.inverse_transform(np.array([1]), step_arguments={"bad": {}})


def test_get_all_preprocessors(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline()
    fake_data = xr.Dataset(
        {"values": ("year", np.array([1]))},
        coords={"year": np.array([1])},
    )
    pipe.fit(base_data=fake_data, save=False)

    procs = pipe.get_preprocessors()

    assert len(procs) == 1


@pytest.mark.pruned
def test_get_named_preprocessor(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline()
    fake_data = xr.Dataset(
        {"values": ("year", np.array([1]))},
        coords={"year": np.array([1])},
    )
    pipe.fit(base_data=fake_data, save=False)

    proc = pipe.get_preprocessors()[0]

    assert isinstance(proc, DummyPreprocessor)


def test_get_preprocessor_not_fitted():
    pipe = make_pipeline()

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        pipe.get_preprocessors()


def test_get_preprocessor_invalid_name(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline()
    fake_data = xr.Dataset(
        {"values": ("year", np.array([1]))},
        coords={"year": np.array([1])},
    )
    pipe.fit(base_data=fake_data, save=False)

    with pytest.raises(ValueError):
        pipe.get_preprocessors("bad")


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


@pytest.mark.pruned
def test_add_preprocessor_not_fitted():
    pipe = make_pipeline()
    proc = DummyPreprocessor()

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        pipe.add_fitted_preprocessor(proc, "dummy")


def test_pipeline_save(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline()

    fake_data = xr.Dataset(
        {"values": ("year", np.array([1]))},
        coords={"year": np.array([1])},
    )

    monkeypatch.setattr(RuntimeContext, "GLOBAL_EXP_DIR", str(tmp_path))

    pipe.fit(base_data=fake_data, save=True)

    saved = list(tmp_path.rglob("*.joblib"))
    assert len(saved) > 0


@pytest.mark.pruned
def test_custom_save_path_and_name(monkeypatch, tmp_path):
    pipe = make_pipeline()

    fake_data = xr.Dataset(
        {"values": ("year", np.array([1]))},
        coords={"year": np.array([1])},
    )

    custom_dir = tmp_path / "custom"
    custom_name = "my_pipe.joblib"

    pipe.fit(
        base_data=fake_data,
        save=True,
        save_path=custom_dir,
        save_name=custom_name,
    )

    assert (custom_dir / custom_name).exists()


@pytest.mark.pruned
def test_transform_with_step_arguments(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline()
    fake_data = xr.Dataset(
        {"values": ("year", np.array([1]))},
        coords={"year": np.array([1])},
    )
    pipe.fit(base_data=fake_data, save=False)

    data = np.array([1, 2])

    out = pipe.transform(data, step_arguments={"dummy": {"x": 1}})

    assert out is not None


@pytest.mark.pruned
def test_multiple_steps_pipeline(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    sel1 = PreprocessingStepSelector("dummy")
    sel2 = PreprocessingStepSelector("dummy")

    pipe = PreprocessingPipeline([sel1, sel2])

    fake_data = xr.Dataset(
        {"values": ("year", np.array([1]))},
        coords={"year": np.array([1])},
    )
    pipe.fit(base_data=fake_data, save=False)

    assert len(pipe.steps) == 2


@pytest.mark.pruned
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
    monkeypatch.setattr(RuntimeContext, "GLOBAL_EXP_DIR", str(tmp_path))

    dir_path = tmp_path / "preprocessing_pipeline"
    dir_path.mkdir()

    pipe = make_pipeline()
    fake_data = xr.Dataset(
        {"values": ("year", np.array([1]))},
        coords={"year": np.array([1])},
    )

    pipe.fit(base_data=fake_data, save=True)

    assert dir_path.exists()


@pytest.mark.pruned
def test_transform_empty_args(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_EXP_DIR", str(tmp_path))

    pipe = make_pipeline()
    fake_data = xr.Dataset(
        {"values": ("year", np.array([1]))},
        coords={"year": np.array([1])},
    )
    pipe.fit(base_data=fake_data, save=False)

    out = pipe.transform(np.array([1]), step_arguments={})

    assert out is not None


@pytest.mark.pruned
def test_set_name():
    pipe = make_pipeline()

    pipe.set_name("abc")

    assert pipe.name == "abc"


def test_post_init_with_load_dir(tmp_path):
    pipe = PreprocessingPipeline(load_dir=tmp_path / "x.joblib")

    assert pipe.fitted is False
    assert pipe.reference_coords is None
    assert pipe.reference_var is None


@pytest.mark.pruned
def test_fit_returns_self(monkeypatch, tmp_path):
    monkeypatch.setattr(
        RuntimeContext,
        "GLOBAL_EXP_DIR",
        str(tmp_path),
    )

    pipe = make_pipeline()

    ds = xr.Dataset(
        {"var": ("year", [1])},
        coords={"year": [2000]},
    )

    result = pipe.fit(
        base_data=ds,
        save=False,
    )

    assert result is pipe


@pytest.mark.pruned
def test_fit_with_empty_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(
        RuntimeContext,
        "GLOBAL_EXP_DIR",
        str(tmp_path),
    )

    pipe = PreprocessingPipeline([])

    ds = xr.Dataset(
        {"var": ("year", [1])},
        coords={"year": [2000]},
    )

    pipe.fit(
        base_data=ds,
        save=False,
    )

    assert pipe.fitted
    assert pipe.steps == []
    assert pipe.fitted_preprocessors == []


@pytest.mark.pruned
def test_transform_none_step_arguments():
    pipe = make_pipeline()

    pipe.steps = ["dummy"]
    pipe.fitted_preprocessors = [DummyPreprocessor()]
    pipe.fitted = True

    result = pipe.transform(
        np.array([1]),
        step_arguments=None,
    )

    assert result is not None


@pytest.mark.pruned
def test_inverse_none_step_arguments():
    pipe = make_pipeline()

    pipe.steps = ["dummy"]
    pipe.fitted_preprocessors = [DummyPreprocessor()]
    pipe.fitted = True

    result = pipe.inverse_transform(
        np.array([1]),
        step_arguments=None,
    )

    assert result is not None


@pytest.mark.pruned
def test_add_fitted_preprocessor_append():
    pipe = make_pipeline()

    pipe.steps = []
    pipe.fitted_preprocessors = []

    proc = DummyPreprocessor()
    proc.fitted = True

    pipe.add_fitted_preprocessor(
        proc,
        "x",
    )

    assert pipe.steps == ["x"]


def test_extract_output_coords_not_fitted():
    pipe = make_pipeline()

    with pytest.raises(
        ValueError,
        match="fitted pipeline",
    ):
        pipe.extract_output_coords_vars(xr.Dataset())


def test_load_from_memory_not_fitted(tmp_path):
    file = tmp_path / "bad.joblib"

    joblib.dump(
        FakeLoaded(fitted=False),
        file,
    )

    pipe = make_pipeline()

    with pytest.raises(
        ValueError,
        match="fitted first",
    ):
        pipe.load_from_memory(file)


def test_load_from_memory_success(tmp_path):
    file = tmp_path / "good.joblib"

    joblib.dump(
        FakeLoaded(
            name="loaded_pipe",
            fitted=True,
        ),
        file,
    )

    pipe = make_pipeline()

    result = pipe.load_from_memory(file)

    assert result is pipe
    assert pipe.fitted
    assert pipe.name == "loaded_pipe"


@pytest.mark.pruned
def test_fit_creates_save_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        RuntimeContext,
        "GLOBAL_EXP_DIR",
        str(tmp_path),
    )

    save_dir = tmp_path / "does_not_exist"

    pipe = make_pipeline()

    ds = xr.Dataset(
        {"var": ("year", [1])},
        coords={"year": [1]},
    )

    pipe.fit(
        base_data=ds,
        save=True,
        save_path=save_dir,
    )

    assert save_dir.exists()


def test_to_dataset_channel_mismatch():
    pipe = make_pipeline()

    pipe.fitted = True
    pipe.reference_var = ["a", "b"]
    pipe.reference_coords = {}

    arr = xr.DataArray(
        np.ones((1,)),
        dims=("channels",),
        coords={"channels": ["a"]},
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        pipe.to_dataset(arr)


def test_to_dataset_normal_branch():
    pipe = make_pipeline()

    pipe.fitted = True

    pipe.reference_var = ["var"]

    pipe.reference_coords = {"lat": xr.DataArray([0], dims="lat")}

    pipe.fitted_preprocessors = []

    arr = xr.DataArray(
        [[1.0]],
        dims=("channels", "output_dim_0"),
        coords={
            "channels": ["var"],
            "output_dim_0": [0],
        },
    )

    ds = pipe.to_dataset(arr)

    assert isinstance(
        ds,
        xr.Dataset,
    )


def test_transform_passes_step_arguments():
    class ArgProc(DummyPreprocessor):
        def transform(
            self,
            data,
            **kwargs,
        ):
            self.kwargs = kwargs
            return data

    pipe = make_pipeline()

    proc = ArgProc()
    proc.fitted = True

    pipe.steps = ["dummy"]
    pipe.fitted_preprocessors = [proc]
    pipe.fitted = True

    pipe.transform(
        np.array([1]),
        step_arguments={"dummy": {"x": 123}},
    )

    assert proc.kwargs == {"x": 123}


def test_inverse_passes_step_arguments():
    class ArgProc(DummyPreprocessor):
        def inverse_transform(
            self,
            data,
            **kwargs,
        ):
            self.kwargs = kwargs
            return data

    pipe = make_pipeline()

    proc = ArgProc()
    proc.fitted = True

    pipe.steps = ["dummy"]
    pipe.fitted_preprocessors = [proc]
    pipe.fitted = True

    pipe.inverse_transform(
        np.array([1]),
        step_arguments={"dummy": {"x": 456}},
    )

    assert proc.kwargs == {"x": 456}


@pytest.mark.pruned
def test_multiple_preprocessors_order(monkeypatch, tmp_path):
    monkeypatch.setattr(
        RuntimeContext,
        "GLOBAL_EXP_DIR",
        str(tmp_path),
    )

    pipe = PreprocessingPipeline(
        [
            PreprocessingStepSelector("dummy"),
            PreprocessingStepSelector("dummy"),
        ]
    )

    ds = xr.Dataset(
        {"var": ("year", [1])},
        coords={"year": [1]},
    )

    pipe.fit(
        base_data=ds,
        save=False,
    )

    assert pipe.steps == [
        "dummy",
        "dummy",
    ]


@pytest.mark.pruned
def test_get_preprocessors_runtime_error():
    pipe = make_pipeline()

    pipe.fitted = False

    with pytest.raises(
        RuntimeError,
        match="Pipeline needs to be fitted",
    ):
        pipe.get_preprocessors()


@pytest.mark.pruned
def test_load_from_memory_copies_attributes(tmp_path):
    loaded = FakeLoaded(
        name="abc",
        fitted=True,
    )

    loaded.steps = ["a"]
    loaded.fitted_preprocessors = [DummyPreprocessor()]

    file = tmp_path / "pipe.joblib"

    joblib.dump(
        loaded,
        file,
    )

    pipe = make_pipeline()

    pipe.load_from_memory(file)

    assert pipe.steps == ["a"]
    assert len(pipe.fitted_preprocessors) == 1

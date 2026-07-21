import numpy as np
import pytest
import torch

from cccma_ppp.data_modules.dataset.dataset_abc import AddedTimeFeatures
from cccma_ppp.generic.runtime import RuntimeContext
from cccma_ppp.inference.dataloader import (
    BatchData,
    InferenceDataloaderConfig,
    collate_batch,
)


class DummyDSOperator:
    def __init__(self):
        self.input_calls = 0
        self.target_calls = 0
        self.load_calls = []

    def get_input_var_metadata(self):
        self.input_calls += 1
        return {"tas": {}}

    def get_target_var_metadata(self):
        self.target_calls += 1
        return {"obs": {}}

    def load_fitted_preprocessors(self, load_dir=None):
        self.load_calls.append(load_dir)


class DummyPreprocessingPipeline:
    def __init__(self, name):
        self.name = name


class DummyDataSource:
    def __init__(self, name):
        self.preprocessing_pipeline = DummyPreprocessingPipeline(name)


class DummyDatasetConfig:
    def __init__(self):
        self.available_times = np.asarray([2000, 2001, 2002, 2003, 2004])
        self.get_common_time = self.available_times
        self.lead_months = np.asarray([1, 2, 3])
        self.ds_operator = DummyDSOperator()
        self.model = None
        self.condition = None
        self.load_called = False
        self.load_dir = None
        self.build_called = False
        self.years = None
        self.time_features = None
        self.return_metadata = None
        self.load = None

    def load_fitted_preprocessors(self, load_dir=None):
        self.load_called = True
        self.load_dir = load_dir

    def build_dataset(
        self,
        years,
        time_features,
        return_metadata=False,
        load=False,
    ):
        self.build_called = True
        self.years = np.asarray(years)
        self.time_features = time_features
        self.return_metadata = return_metadata
        self.load = load
        return [0, 1, 2]


class DummyTrainDatasetConfig:
    def __init__(self):
        self.fit_called = False
        self.fit_args = None
        self.train_years = [2000, 2001]
        self.ds_operator = DummyDSOperator()

    def fit_preprocessors(
        self,
        train_years,
        save=True,
        save_path=None,
        save_name=None,
    ):
        self.fit_called = True
        self.fit_args = {
            "train_years": train_years,
            "save": save,
            "save_path": save_path,
        }

        if save_name is not None:
            self.fit_args["save_name"] = save_name


class DummyTrainLoader:
    def __init__(self):
        self.dataset_config = DummyTrainDatasetConfig()
        self.time_features = ["year", "lead_time"]


class DummyDistributed:
    def __init__(
        self,
        rank=0,
        world_size=1,
        root=True,
        device="cpu",
        distributed=False,
    ):
        self.rank = rank
        self.local_rank = rank
        self.world_size = world_size
        self.device = device
        self.distributed = distributed
        self._root = root
        self.barrier_called = False
        self.barrier_calls = 0

    def is_root(self):
        return self._root

    def barrier(self):
        self.barrier_called = True
        self.barrier_calls += 1


class FakeDataloader:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        for name, value in kwargs.items():
            setattr(self, name, value)


@pytest.fixture
def dataset_config():
    return DummyDatasetConfig()


@pytest.fixture
def patched_dataloader(monkeypatch):
    monkeypatch.setattr(
        "cccma_ppp.inference.dataloader.Dataloader",
        FakeDataloader,
    )


def test_batchdata_replaces_nan():
    batch = BatchData(
        input=torch.tensor(
            [
                [1.0, float("nan")],
                [float("nan"), 4.0],
            ]
        )
    )

    assert not torch.isnan(batch.input).any()

    torch.testing.assert_close(
        batch.input,
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 4.0],
            ]
        ),
    )


def test_batchdata_replaces_all_nan():
    batch = BatchData(
        input=torch.full(
            (2, 2),
            float("nan"),
        )
    )

    torch.testing.assert_close(
        batch.input,
        torch.zeros_like(batch.input),
    )


def test_batchdata_without_mask():
    batch = BatchData(
        input=torch.ones(2, 2),
        return_spatial_mask=False,
    )

    assert batch.input_mask is None


def test_batchdata_creates_spatial_mask():
    batch = BatchData(
        input=torch.tensor([[1.0, float("nan")]]),
        return_spatial_mask=True,
        reduce_spatial_mask=False,
    )

    torch.testing.assert_close(
        batch.input,
        torch.tensor([[1.0, 0.0]]),
    )
    torch.testing.assert_close(
        batch.input_mask,
        torch.tensor([[True, False]]),
    )


def test_batchdata_reduces_spatial_mask():
    batch = BatchData(
        input=torch.tensor(
            [
                [1.0, float("nan")],
                [3.0, 4.0],
            ]
        ),
        return_spatial_mask=True,
        reduce_spatial_mask=True,
    )

    torch.testing.assert_close(
        batch.input_mask,
        torch.tensor([True, False]),
    )


@pytest.mark.parametrize(
    ("with_mask", "with_features"),
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_batchdata_to_device(
    with_mask,
    with_features,
):
    batch = BatchData(
        input=torch.ones(2, 2),
        added_features=(torch.ones(3) if with_features else None),
        return_spatial_mask=with_mask,
        reduce_spatial_mask=False,
    )

    result = batch.to_device("cpu")

    assert result is batch
    assert batch.input.device.type == "cpu"

    if with_mask:
        assert batch.input_mask.device.type == "cpu"
    else:
        assert batch.input_mask is None

    if with_features:
        assert batch.added_features.device.type == "cpu"
    else:
        assert batch.added_features is None


def test_batchdata_metadata_survives_to_device():
    metadata = [{"year": 2000}]
    batch = BatchData(
        input=torch.ones(2),
        metadata=metadata,
    )

    batch.to_device("cpu")

    assert batch.metadata is metadata


def make_collate_item(
    value,
    features=None,
    metadata=None,
):
    data = {
        "input": torch.as_tensor(
            value,
            dtype=torch.float32,
        ),
        "added_features": (
            None
            if features is None
            else torch.as_tensor(
                features,
                dtype=torch.float32,
            )
        ),
    }

    if metadata is None:
        return data

    return data, metadata


def test_collate_batch_plain_inputs():
    result = collate_batch(
        [
            make_collate_item([1.0, 1.0]),
            make_collate_item([0.0, 0.0]),
        ]
    )

    assert result.input.shape == (2, 2)
    assert result.metadata is None
    assert result.added_features is None
    assert result.input_mask is None


def test_collate_batch_single_item():
    result = collate_batch([make_collate_item([1.0, 1.0])])

    assert result.input.shape == (1, 2)


def test_collate_batch_with_metadata():
    result = collate_batch(
        [
            make_collate_item(
                [1.0, 1.0],
                metadata={"year": 2000},
            ),
            make_collate_item(
                [0.0, 0.0],
                metadata={"year": 2001},
            ),
        ]
    )

    assert result.metadata == [
        {"year": 2000},
        {"year": 2001},
    ]


def test_collate_batch_with_features():
    result = collate_batch(
        [
            make_collate_item(
                [1.0, 1.0],
                features=[1.0, 2.0],
            ),
            make_collate_item(
                [0.0, 0.0],
                features=[3.0, 4.0],
            ),
        ]
    )

    torch.testing.assert_close(
        result.added_features,
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
    )


def test_collate_batch_metadata_features_and_mask():
    result = collate_batch(
        [
            make_collate_item(
                [1.0, float("nan")],
                features=[2.0],
                metadata={"year": 2000},
            )
        ],
        return_spatial_mask=True,
        reduce_spatial_mask=False,
    )

    assert result.metadata == [{"year": 2000}]
    assert result.added_features.shape == (1, 1)

    torch.testing.assert_close(
        result.input,
        torch.tensor([[1.0, 0.0]]),
    )
    torch.testing.assert_close(
        result.input_mask,
        torch.tensor([[True, False]]),
    )


def make_loader_config(
    dataset_config=None,
    time_features=(),
    **kwargs,
):
    config = InferenceDataloaderConfig(
        dataset_config=None,
        **kwargs,
    )
    config.dataset_config = dataset_config
    config.time_features = time_features
    return config


def test_init_without_dataset_config():
    config = make_loader_config()

    assert config.dataset_config is None
    assert config._setup is False
    assert config.train_dataset_config is None


def test_init_workers_zero_removes_prefetch():
    config = make_loader_config(
        num_data_workers=0,
        prefetch_factor=8,
    )

    assert config.prefetch_factor is None


def test_init_workers_positive_preserves_prefetch(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        num_data_workers=2,
        prefetch_factor=8,
    )

    assert config.prefetch_factor == 8


def test_available_times_requires_config():
    config = make_loader_config()

    with pytest.raises(RuntimeError):
        _ = config.available_times


def test_available_times(dataset_config):
    config = make_loader_config(
        dataset_config=dataset_config,
    )

    np.testing.assert_array_equal(
        config.available_times,
        dataset_config.available_times,
    )


def test_inference_years_default(dataset_config):
    config = make_loader_config(
        dataset_config=dataset_config,
    )

    np.testing.assert_array_equal(
        config._inference_years,
        dataset_config.available_times,
    )


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ((2000, 2000), [2000]),
        ((2001, 2002), [2001, 2002]),
        (
            (2000, 2004),
            [2000, 2001, 2002, 2003, 2004],
        ),
    ],
)
def test_inference_years_valid_ranges(
    dataset_config,
    requested,
    expected,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        inference_years=requested,
    )

    np.testing.assert_array_equal(
        config._inference_years,
        expected,
    )


def test_input_metadata_requires_config():
    config = make_loader_config()

    with pytest.raises(RuntimeError):
        _ = config.input_var_metadata


def test_input_metadata(dataset_config):
    config = make_loader_config(
        dataset_config=dataset_config,
    )

    assert config.input_var_metadata == {"tas": {}}
    assert dataset_config.ds_operator.input_calls == 1


def test_target_metadata_requires_train_config(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=dataset_config,
    )

    with pytest.raises(RuntimeError):
        _ = config.target_var_metadata


def test_target_metadata_success(dataset_config):
    config = make_loader_config(
        dataset_config=dataset_config,
    )
    train_config = DummyTrainDatasetConfig()
    config.train_dataset_config = train_config

    assert config.target_var_metadata == {"obs": {}}
    assert train_config.ds_operator.target_calls == 1


@pytest.mark.parametrize(
    ("model_name", "condition_name", "existing", "expected"),
    [
        (None, None, (), True),
        ("model", None, (), False),
        ("model", None, ("model",), True),
        (None, "condition", (), False),
        (None, "condition", ("condition",), True),
        ("model", "condition", (), False),
        ("model", "condition", ("model",), False),
        ("model", "condition", ("condition",), False),
        (
            "model",
            "condition",
            ("model", "condition"),
            True,
        ),
    ],
)
def test_input_preprocessor_exists_matrix(
    dataset_config,
    tmp_path,
    model_name,
    condition_name,
    existing,
    expected,
):
    dataset_config.model = (
        DummyDataSource(model_name) if model_name is not None else None
    )
    dataset_config.condition = (
        DummyDataSource(condition_name) if condition_name is not None else None
    )

    for name in existing:
        (tmp_path / f"{name}_preprocessing_pipeline.joblib").touch()

    config = make_loader_config(
        dataset_config=dataset_config,
    )

    assert config._input_preprocessor_exists(tmp_path) is expected


def test_input_preprocessor_exists_requires_config(
    tmp_path,
):
    config = make_loader_config()

    with pytest.raises(RuntimeError):
        config._input_preprocessor_exists(tmp_path)


def test_input_preprocessor_exists_default_directory(
    dataset_config,
    monkeypatch,
    tmp_path,
):
    directory = tmp_path / "preprocessing_pipeline"
    directory.mkdir()

    dataset_config.model = DummyDataSource("model")
    dataset_config.condition = None

    (directory / "model_preprocessing_pipeline.joblib").touch()

    monkeypatch.setattr(
        RuntimeContext,
        "GLOBAL_EXP_DIR",
        str(tmp_path),
    )

    config = make_loader_config(
        dataset_config=dataset_config,
    )

    assert config._input_preprocessor_exists() is True


def test_setup_distributed_requires_config():
    config = make_loader_config()

    with pytest.raises(RuntimeError):
        config.setup_distributed(
            DummyTrainLoader(),
            DummyDistributed(),
        )


@pytest.mark.parametrize(
    ("exists", "root", "fit_expected"),
    [
        (True, True, False),
        (True, False, False),
        (False, True, True),
        (False, False, False),
    ],
)
def test_setup_distributed_branch_matrix(
    dataset_config,
    monkeypatch,
    exists,
    root,
    fit_expected,
):
    config = make_loader_config(
        dataset_config=dataset_config,
    )

    monkeypatch.setattr(
        config,
        "_input_preprocessor_exists",
        lambda *args, **kwargs: exists,
    )

    train_loader = DummyTrainLoader()
    distributed = DummyDistributed(
        rank=0 if root else 1,
        world_size=2,
        root=root,
    )

    config.setup_distributed(
        train_loader,
        distributed,
    )

    assert train_loader.dataset_config.fit_called is fit_expected
    assert dataset_config.load_called is True
    assert distributed.barrier_called is True
    assert config.rank == distributed.rank
    assert config.world_size == distributed.world_size
    assert config._setup is True


def test_setup_distributed_passes_fit_arguments(
    dataset_config,
    monkeypatch,
    tmp_path,
):
    config = make_loader_config(
        dataset_config=dataset_config,
    )

    monkeypatch.setattr(
        config,
        "_input_preprocessor_exists",
        lambda *args, **kwargs: False,
    )

    train_loader = DummyTrainLoader()
    distributed = DummyDistributed(root=True)

    config.setup_distributed(
        train_loader,
        distributed,
        load_path=tmp_path,
    )

    assert train_loader.dataset_config.fit_args == {
        "train_years": [2000, 2001],
        "save": True,
        "save_path": tmp_path,
    }
    assert dataset_config.load_dir == tmp_path


def test_setup_existing_preprocessors_skips_fit(
    dataset_config,
    monkeypatch,
):
    config = make_loader_config(
        dataset_config=dataset_config,
    )

    monkeypatch.setattr(
        config,
        "_input_preprocessor_exists",
        lambda *args, **kwargs: True,
    )

    train_loader = DummyTrainLoader()
    distributed = DummyDistributed(root=True)

    config.setup_distributed(
        train_loader,
        distributed,
    )

    assert train_loader.dataset_config.fit_called is False
    assert dataset_config.load_called is True


def test_build_loader_requires_setup(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=dataset_config,
    )

    with pytest.raises(RuntimeError):
        config.build_inference_loader()


def test_build_loader_passes_default_years(
    dataset_config,
    patched_dataloader,
):
    config = make_loader_config(
        dataset_config=dataset_config,
    )
    config.rank = 0
    config.world_size = 1
    config._setup = True

    config.build_inference_loader()

    np.testing.assert_array_equal(
        dataset_config.years,
        dataset_config.available_times,
    )


def test_build_loader_passes_explicit_years(
    dataset_config,
    patched_dataloader,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        inference_years=(2001, 2002),
    )
    config.rank = 0
    config.world_size = 1
    config._setup = True

    config.build_inference_loader()

    np.testing.assert_array_equal(
        dataset_config.years,
        [2001, 2002],
    )


@pytest.fixture(autouse=True)
def reset_shared_input_mask():
    BatchData._shared_input_mask = None

    yield

    BatchData._shared_input_mask = None


def test_check_config_rejects_missing_dataset_first():
    config = make_loader_config(
        dataset_config=None,
        time_features=None,
    )

    with pytest.raises(
        RuntimeError,
        match="dataset_config must be provided",
    ):
        config._check_config()


def test_check_config_rejects_missing_time_features(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=None,
    )

    with pytest.raises(
        RuntimeError,
        match="time_features must be read",
    ):
        config._check_config()


def test_check_config_accepts_empty_time_features(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    assert config._check_config() is None


def test_check_config_accepts_nonempty_time_features(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=("year",),
    )

    assert config._check_config() is None


def test_post_init_without_dataset_does_not_read_years():
    config = InferenceDataloaderConfig(
        dataset_config=None,
    )

    assert config.dataset_config is None
    assert config.train_dataset_config is None
    assert config._setup is False


def test_post_init_with_dataset_requires_time_features(
    dataset_config,
):
    with pytest.raises(
        RuntimeError,
        match="time_features must be read",
    ):
        InferenceDataloaderConfig(
            dataset_config=dataset_config,
        )


def test_post_init_with_dataset_and_preseeded_features(
    monkeypatch,
    dataset_config,
):
    original_post_init = InferenceDataloaderConfig.__post_init__

    def patched_post_init(self):
        self.time_features = ()
        original_post_init(self)

    monkeypatch.setattr(
        InferenceDataloaderConfig,
        "__post_init__",
        patched_post_init,
    )

    config = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert config.dataset_config is dataset_config
    assert config.time_features == ()


def test_batchdata_reduced_mask_initializes_shared_mask():
    BatchData._shared_input_mask = None

    batch = BatchData(
        input=torch.tensor(
            [
                [1.0, float("nan")],
                [2.0, 3.0],
            ]
        ),
        return_spatial_mask=True,
        reduce_spatial_mask=True,
    )

    expected = torch.tensor([True, False])

    torch.testing.assert_close(
        batch.input_mask,
        expected,
    )
    torch.testing.assert_close(
        BatchData._shared_input_mask,
        expected,
    )


def test_batchdata_reduced_mask_reuses_shared_mask():
    shared = torch.tensor([False, True])
    BatchData._shared_input_mask = shared

    batch = BatchData(
        input=torch.tensor(
            [
                [1.0, float("nan")],
                [2.0, 3.0],
            ]
        ),
        return_spatial_mask=True,
        reduce_spatial_mask=True,
    )

    assert batch.input_mask is shared


def test_batchdata_unreduced_mask_does_not_set_shared_mask():
    BatchData._shared_input_mask = None

    batch = BatchData(
        input=torch.tensor([[1.0, float("nan")]]),
        return_spatial_mask=True,
        reduce_spatial_mask=False,
    )

    assert batch.input_mask is not None
    assert BatchData._shared_input_mask is None


def test_batchdata_no_mask_does_not_set_shared_mask():
    BatchData._shared_input_mask = None

    batch = BatchData(
        input=torch.tensor([[1.0, float("nan")]]),
        return_spatial_mask=False,
    )

    assert batch.input_mask is None
    assert BatchData._shared_input_mask is None


def test_batchdata_to_device_moves_only_input():
    batch = BatchData(
        input=torch.ones(2),
        added_features=None,
        return_spatial_mask=False,
    )

    result = batch.to_device("cpu")

    assert result is batch
    assert batch.input.device.type == "cpu"
    assert batch.input_mask is None
    assert batch.added_features is None


def test_batchdata_to_device_moves_all_optional_tensors():
    batch = BatchData(
        input=torch.ones(2),
        added_features=torch.ones(1),
        return_spatial_mask=True,
        reduce_spatial_mask=False,
    )

    result = batch.to_device("cpu")

    assert result is batch
    assert batch.input.device.type == "cpu"
    assert batch.input_mask.device.type == "cpu"
    assert batch.added_features.device.type == "cpu"


def test_collate_plain_batch_no_metadata_or_features():
    result = collate_batch(
        [
            {
                "input": torch.tensor([1.0]),
                "added_features": None,
            },
            {
                "input": torch.tensor([2.0]),
                "added_features": None,
            },
        ]
    )

    torch.testing.assert_close(
        result.input,
        torch.tensor(
            [
                [1.0],
                [2.0],
            ]
        ),
    )
    assert result.metadata is None
    assert result.added_features is None


def test_collate_tuple_batch_with_metadata():
    result = collate_batch(
        [
            (
                {
                    "input": torch.tensor([1.0]),
                    "added_features": None,
                },
                {"year": 2000},
            ),
            (
                {
                    "input": torch.tensor([2.0]),
                    "added_features": None,
                },
                {"year": 2001},
            ),
        ]
    )

    assert result.metadata == [
        {"year": 2000},
        {"year": 2001},
    ]
    assert result.added_features is None


def test_collate_stacks_added_features():
    result = collate_batch(
        [
            {
                "input": torch.tensor([1.0]),
                "added_features": torch.tensor([10.0, 11.0]),
            },
            {
                "input": torch.tensor([2.0]),
                "added_features": torch.tensor([20.0, 21.0]),
            },
        ]
    )

    torch.testing.assert_close(
        result.added_features,
        torch.tensor(
            [
                [10.0, 11.0],
                [20.0, 21.0],
            ]
        ),
    )


def test_collate_passes_mask_options():
    result = collate_batch(
        [
            {
                "input": torch.tensor([1.0, float("nan")]),
                "added_features": None,
            }
        ],
        return_spatial_mask=True,
        reduce_spatial_mask=False,
    )

    torch.testing.assert_close(
        result.input_mask,
        torch.tensor([[True, False]]),
    )


def test_inference_years_none_returns_available_times(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
        inference_years=None,
    )

    result = config._inference_years

    np.testing.assert_array_equal(
        result,
        dataset_config.available_times,
    )


def test_inference_years_range_is_inclusive(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
        inference_years=(2001, 2003),
    )

    np.testing.assert_array_equal(
        config._inference_years,
        [2001, 2002, 2003],
    )


def test_inference_years_single_year(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
        inference_years=(2002, 2002),
    )

    np.testing.assert_array_equal(
        config._inference_years,
        [2002],
    )


def test_inference_years_partially_unavailable(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=None,
        time_features=(),
        inference_years=(2003, 2005),
    )
    config.dataset_config = dataset_config

    with pytest.raises(
        ValueError,
        match="requested inference years",
    ):
        _ = config._inference_years


def test_inference_years_completely_unavailable(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=None,
        time_features=(),
        inference_years=(1990, 1992),
    )
    config.dataset_config = dataset_config

    with pytest.raises(
        ValueError,
        match="requested inference years",
    ):
        _ = config._inference_years


def test_available_times_rejects_missing_features(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=None,
    )

    with pytest.raises(
        RuntimeError,
        match="time_features must be read",
    ):
        _ = config.available_times


def test_available_times_returns_dataset_values(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    np.testing.assert_array_equal(
        config.available_times,
        dataset_config.available_times,
    )


def test_read_configs_copies_time_features_when_missing(
    dataset_config,
):
    train_loader = DummyTrainLoader()

    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=None,
    )

    config.read_configs_from_train(train_loader)

    assert config.time_features == train_loader.time_features
    assert config.time_features is not train_loader.time_features


def test_read_configs_preserves_existing_time_features(
    dataset_config,
):
    train_loader = DummyTrainLoader()

    existing = ["month_sin"]
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=existing,
    )

    config.read_configs_from_train(train_loader)

    assert config.time_features is existing
    assert config.time_features == ["month_sin"]


def test_read_configs_builds_missing_dataset_config(
    monkeypatch,
):
    train_loader = DummyTrainLoader()
    converted = DummyDatasetConfig()

    calls = []

    def fake_from_train(value):
        calls.append(value)
        return converted

    monkeypatch.setattr(
        "cccma_ppp.inference.dataloader._from_train",
        fake_from_train,
    )

    config = make_loader_config(
        dataset_config=None,
        time_features=None,
    )

    config.read_configs_from_train(train_loader)

    assert calls == [train_loader.dataset_config]
    assert config.dataset_config is converted
    assert config.train_dataset_config is train_loader.dataset_config


def test_read_configs_preserves_existing_dataset_config(
    monkeypatch,
    dataset_config,
):
    train_loader = DummyTrainLoader()

    def fail_from_train(value):
        pytest.fail("_from_train should not be called")

    monkeypatch.setattr(
        "cccma_ppp.inference.dataloader._from_train",
        fail_from_train,
    )

    config = make_loader_config(
        dataset_config=None,
        time_features=None,
    )
    config.dataset_config = dataset_config

    config.read_configs_from_train(train_loader)

    assert config.dataset_config is dataset_config


def test_read_configs_always_records_train_dataset(
    dataset_config,
):
    train_loader = DummyTrainLoader()

    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    config.read_configs_from_train(train_loader)

    assert config.train_dataset_config is train_loader.dataset_config


def test_input_preprocessor_exists_no_sources(
    dataset_config,
    tmp_path,
):
    dataset_config.model = None
    dataset_config.condition = None

    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    assert config._input_preprocessor_exists(tmp_path) is True


def test_input_preprocessor_exists_model_only_true(
    dataset_config,
    tmp_path,
):
    dataset_config.model = DummyDataSource("model")
    dataset_config.condition = None

    (tmp_path / "model_preprocessing_pipeline.joblib").touch()

    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    assert config._input_preprocessor_exists(tmp_path) is True


def test_input_preprocessor_exists_model_only_false(
    dataset_config,
    tmp_path,
):
    dataset_config.model = DummyDataSource("model")
    dataset_config.condition = None

    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    assert config._input_preprocessor_exists(tmp_path) is False


def test_input_preprocessor_exists_condition_only_true(
    dataset_config,
    tmp_path,
):
    dataset_config.model = None
    dataset_config.condition = DummyDataSource("condition")

    (tmp_path / "condition_preprocessing_pipeline.joblib").touch()

    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    assert config._input_preprocessor_exists(tmp_path) is True


def test_input_preprocessor_exists_condition_only_false(
    dataset_config,
    tmp_path,
):
    dataset_config.model = None
    dataset_config.condition = DummyDataSource("condition")

    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    assert config._input_preprocessor_exists(tmp_path) is False


def test_input_preprocessor_exists_both_true(
    dataset_config,
    tmp_path,
):
    dataset_config.model = DummyDataSource("model")
    dataset_config.condition = DummyDataSource("condition")

    (tmp_path / "model_preprocessing_pipeline.joblib").touch()
    (tmp_path / "condition_preprocessing_pipeline.joblib").touch()

    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    assert config._input_preprocessor_exists(tmp_path) is True


@pytest.mark.parametrize(
    ("model_exists", "condition_exists"),
    [
        (False, False),
        (True, False),
        (False, True),
    ],
)
def test_input_preprocessor_exists_both_partial(
    dataset_config,
    tmp_path,
    model_exists,
    condition_exists,
):
    dataset_config.model = DummyDataSource("model")
    dataset_config.condition = DummyDataSource("condition")

    if model_exists:
        (tmp_path / "model_preprocessing_pipeline.joblib").touch()

    if condition_exists:
        (tmp_path / "condition_preprocessing_pipeline.joblib").touch()

    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    assert config._input_preprocessor_exists(tmp_path) is False


def test_setup_distributed_missing_preprocessors_root_fits(
    dataset_config,
    monkeypatch,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    monkeypatch.setattr(
        config,
        "_input_preprocessor_exists",
        lambda load_path: False,
    )

    train_loader = DummyTrainLoader()
    distributed = DummyDistributed(
        root=True,
        distributed=False,
    )

    config.setup_distributed(
        train_loader,
        distributed,
    )

    assert train_loader.dataset_config.fit_called is True
    assert distributed.barrier_calls == 1
    assert dataset_config.load_called is True
    assert config._setup is True


def test_setup_distributed_missing_preprocessors_nonroot_skips_fit(
    dataset_config,
    monkeypatch,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    monkeypatch.setattr(
        config,
        "_input_preprocessor_exists",
        lambda load_path: False,
    )

    train_loader = DummyTrainLoader()
    distributed = DummyDistributed(
        rank=1,
        world_size=2,
        root=False,
        distributed=True,
    )

    config.setup_distributed(
        train_loader,
        distributed,
    )

    assert train_loader.dataset_config.fit_called is False
    assert distributed.barrier_calls == 1
    assert dataset_config.load_called is True


def test_setup_distributed_existing_preprocessors_root_skips_fit(
    dataset_config,
    monkeypatch,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    monkeypatch.setattr(
        config,
        "_input_preprocessor_exists",
        lambda load_path: True,
    )

    train_loader = DummyTrainLoader()
    distributed = DummyDistributed(
        root=True,
        distributed=False,
    )

    config.setup_distributed(
        train_loader,
        distributed,
    )

    assert train_loader.dataset_config.fit_called is False


def test_setup_distributed_enables_pin_memory(
    dataset_config,
    monkeypatch,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    monkeypatch.setattr(
        config,
        "_input_preprocessor_exists",
        lambda load_path: True,
    )

    distributed = DummyDistributed(
        root=True,
        distributed=True,
    )

    config.setup_distributed(
        DummyTrainLoader(),
        distributed,
    )

    assert config.pin_memory is True


def test_setup_distributed_leaves_pin_memory_default(
    dataset_config,
    monkeypatch,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )
    original_pin_memory = config.pin_memory

    monkeypatch.setattr(
        config,
        "_input_preprocessor_exists",
        lambda load_path: True,
    )

    distributed = DummyDistributed(
        root=True,
        distributed=False,
    )

    config.setup_distributed(
        DummyTrainLoader(),
        distributed,
    )

    assert config.pin_memory is original_pin_memory


def test_setup_distributed_records_rank_and_world_size(
    dataset_config,
    monkeypatch,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    monkeypatch.setattr(
        config,
        "_input_preprocessor_exists",
        lambda load_path: True,
    )

    distributed = DummyDistributed(
        rank=3,
        world_size=8,
        root=False,
        distributed=True,
    )

    config.setup_distributed(
        DummyTrainLoader(),
        distributed,
    )

    assert config.rank == 3
    assert config.world_size == 8


def test_build_loader_passes_load_true(
    dataset_config,
    patched_dataloader,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
        load=True,
    )
    config.rank = 0
    config.world_size = 1
    config._setup = True

    config.build_inference_loader()

    assert dataset_config.load is True


def test_build_loader_passes_load_false(
    dataset_config,
    patched_dataloader,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
        load=False,
    )
    config.rank = 0
    config.world_size = 1
    config._setup = True

    config.build_inference_loader()

    assert dataset_config.load is False


def test_build_loader_always_requests_metadata(
    dataset_config,
    patched_dataloader,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )
    config.rank = 0
    config.world_size = 1
    config._setup = True

    config.build_inference_loader()

    assert dataset_config.return_metadata is True


def test_build_loader_passes_same_time_features_object(
    dataset_config,
    patched_dataloader,
):
    features = AddedTimeFeatures(
        dataset_config,
        ["year"],
    )

    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=features,
    )
    config.rank = 0
    config.world_size = 1
    config._setup = True

    config.build_inference_loader()

    assert dataset_config.time_features is features


def test_build_loader_constructs_expected_dataloader(
    dataset_config,
    patched_dataloader,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
        batch_size=7,
        drop_last=True,
    )
    config.rank = 2
    config.world_size = 4
    config._setup = True

    loader = config.build_inference_loader(
        return_spatial_mask=True,
        reduce_spatial_mask=True,
    )

    assert loader.dataset == [0, 1, 2]
    assert loader.config is config
    assert loader.collate_fn is collate_batch
    assert loader.rank == 2
    assert loader.world_size == 4
    assert loader.shuffle is False
    assert loader.return_spatial_mask is True
    assert loader.reduce_spatial_mask is True


def test_input_metadata_calls_operator_each_time(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )

    assert config.input_var_metadata == {"tas": {}}
    assert config.input_var_metadata == {"tas": {}}
    assert dataset_config.ds_operator.input_calls == 2


def test_target_metadata_calls_train_operator_each_time(
    dataset_config,
):
    config = make_loader_config(
        dataset_config=dataset_config,
        time_features=(),
    )
    train_config = DummyTrainDatasetConfig()
    config.train_dataset_config = train_config

    assert config.target_var_metadata == {"obs": {}}
    assert config.target_var_metadata == {"obs": {}}
    assert train_config.ds_operator.target_calls == 2

import numpy as np
import pytest
import torch

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

    def get_input_var_metadata(self):
        self.input_calls += 1
        return {"tas": {}}

    def get_target_var_metadata(self):
        self.target_calls += 1
        return {"obs": {}}


class DummyPreprocessingPipeline:
    def __init__(self, name):
        self.name = name


class DummyDataSource:
    def __init__(self, name):
        self.preprocessing_pipeline = DummyPreprocessingPipeline(name)


class DummyDatasetConfig:
    def __init__(self):
        self.available_times = np.array([2000, 2001, 2002, 2003, 2004])
        self.ds_operator = DummyDSOperator()
        self.model = None
        self.condition = None
        self.load_called = False
        self.load_dir = None
        self.build_called = False
        self.years = None
        self.return_metadata = None

    def _load_fitted_preprocessors(self, load_dir=None):
        self.load_called = True
        self.load_dir = load_dir

    def build_dataset(
        self,
        years,
        return_metadata=False,
        load=False,
    ):
        self.build_called = True
        self.years = np.asarray(years)
        self.return_metadata = return_metadata
        self.load = load
        return [0, 1, 2]


class DummyTrainDatasetConfig:
    def __init__(self):
        self.fit_called = False
        self.fit_args = None
        self.train_years = [2000, 2001]
        self.ds_operator = DummyDSOperator()

    def _fit_preprocessors(
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
    assert batch.input[0, 1] == 0
    assert batch.input[1, 0] == 0


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


def test_batchdata_without_mask_keeps_tensor():
    batch = BatchData(
        input=torch.ones(2, 2),
        return_spatial_mask=False,
    )

    assert isinstance(batch.input, torch.Tensor)
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

    assert isinstance(
        batch.input_mask,
        torch.Tensor,
    )
    torch.testing.assert_close(
        batch.input_mask,
        torch.tensor([True, False]),
    )


def test_batchdata_to_device_without_mask_or_features():
    batch = BatchData(input=torch.ones(2, 2))

    result = batch.to_device("cpu")

    assert result is batch
    assert batch.input.device.type == "cpu"
    assert batch.input_mask is None
    assert batch.added_features is None


def test_batchdata_to_device_with_features():
    batch = BatchData(
        input=torch.ones(2, 2),
        added_features=torch.ones(3),
    )

    result = batch.to_device("cpu")

    assert result is batch
    assert batch.input.device.type == "cpu"
    assert batch.added_features.device.type == "cpu"


def test_batchdata_to_device_with_mask():
    batch = BatchData(
        input=torch.tensor([[1.0, float("nan")]]),
        return_spatial_mask=True,
        reduce_spatial_mask=False,
    )

    result = batch.to_device("cpu")

    assert result is batch
    assert batch.input.device.type == "cpu"
    assert batch.input_mask.device.type == "cpu"


def test_batchdata_to_device_with_mask_and_features():
    batch = BatchData(
        input=torch.ones(2, 2),
        added_features=torch.ones(3),
        return_spatial_mask=True,
        reduce_spatial_mask=False,
    )

    result = batch.to_device("cpu")

    assert result is batch
    assert batch.input.device.type == "cpu"
    assert batch.input_mask.device.type == "cpu"
    assert batch.added_features.device.type == "cpu"


def test_batchdata_metadata_survives_to_device():
    metadata = [{"year": 2000}]

    batch = BatchData(
        input=torch.ones(2),
        metadata=metadata,
    )

    batch.to_device("cpu")

    assert batch.metadata is metadata


def test_collate_batch_plain_inputs():
    batch = [
        {
            "input": torch.ones(2, 2),
            "added_features": None,
        },
        {
            "input": torch.zeros(2, 2),
            "added_features": None,
        },
    ]

    result = collate_batch(batch)

    assert result.input.shape == (2, 2, 2)
    assert result.metadata is None
    assert result.added_features is None
    assert result.input_mask is None


def test_collate_batch_single_item():
    result = collate_batch(
        [
            {
                "input": torch.ones(2),
                "added_features": None,
            }
        ]
    )

    assert result.input.shape == (1, 2)
    assert result.metadata is None


def test_collate_batch_with_metadata():
    batch = [
        (
            {
                "input": torch.ones(2),
                "added_features": None,
            },
            {"year": 2000},
        ),
        (
            {
                "input": torch.zeros(2),
                "added_features": None,
            },
            {"year": 2001},
        ),
    ]

    result = collate_batch(batch)

    assert result.metadata == [
        {"year": 2000},
        {"year": 2001},
    ]
    assert result.added_features is None


def test_collate_batch_with_added_features():
    batch = [
        {
            "input": torch.ones(2),
            "added_features": torch.tensor([1.0, 2.0]),
        },
        {
            "input": torch.zeros(2),
            "added_features": torch.tensor([3.0, 4.0]),
        },
    ]

    result = collate_batch(batch)

    torch.testing.assert_close(
        result.added_features,
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ),
    )


def test_collate_batch_with_metadata_and_features():
    batch = [
        (
            {
                "input": torch.ones(2),
                "added_features": torch.tensor([1.0]),
            },
            {"year": 2000},
        ),
        (
            {
                "input": torch.zeros(2),
                "added_features": torch.tensor([2.0]),
            },
            {"year": 2001},
        ),
    ]

    result = collate_batch(batch)

    assert result.metadata == [
        {"year": 2000},
        {"year": 2001},
    ]
    assert result.added_features.shape == (2, 1)


def test_collate_batch_with_spatial_mask():
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
        result.input,
        torch.tensor([[1.0, 0.0]]),
    )
    torch.testing.assert_close(
        result.input_mask,
        torch.tensor([[True, False]]),
    )


def test_collate_batch_with_metadata_features_and_mask():
    result = collate_batch(
        [
            (
                {
                    "input": torch.tensor([1.0, float("nan")]),
                    "added_features": torch.tensor([2.0]),
                },
                {"year": 2000},
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


def test_collate_batch_reduce_mask():
    result = collate_batch(
        [
            {
                "input": torch.tensor(
                    [
                        [1.0, float("nan")],
                        [1.0, float("nan")],
                    ]
                ),
                "added_features": None,
            }
        ],
        return_spatial_mask=True,
        reduce_spatial_mask=True,
    )

    assert isinstance(
        result.input_mask,
        torch.Tensor,
    )
    assert torch.all(result.input_mask == torch.tensor([True, False]))


def test_init_without_dataset_config():
    cfg = InferenceDataloaderConfig(
        dataset_config=None,
    )

    assert cfg.dataset_config is None
    assert cfg._setup is False
    assert cfg.train_dataset_config is None


def test_init_workers_zero_removes_prefetch():
    cfg = InferenceDataloaderConfig(
        dataset_config=None,
        num_data_workers=0,
        prefetch_factor=8,
    )

    assert cfg.prefetch_factor is None


def test_init_workers_positive_preserves_prefetch(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
        num_data_workers=2,
        prefetch_factor=8,
    )

    assert cfg.prefetch_factor == 8


def test_check_dataset_config_success(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert cfg._check_dataset_config() is None


def test_check_dataset_config_raises():
    cfg = InferenceDataloaderConfig(
        dataset_config=None,
    )

    with pytest.raises(RuntimeError):
        cfg._check_dataset_config()


def test_available_times_requires_config():
    cfg = InferenceDataloaderConfig(
        dataset_config=None,
    )

    with pytest.raises(RuntimeError):
        _ = cfg.available_times


def test_available_times(dataset_config):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    np.testing.assert_array_equal(
        cfg.available_times,
        np.array([2000, 2001, 2002, 2003, 2004]),
    )


def test_inference_years_default(dataset_config):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    np.testing.assert_array_equal(
        cfg._inference_years,
        dataset_config.available_times,
    )


@pytest.mark.parametrize(
    "requested,expected",
    [
        ((2000, 2000), np.array([2000])),
        (
            (2001, 2002),
            np.array([2001, 2002]),
        ),
        (
            (2000, 2004),
            np.array([2000, 2001, 2002, 2003, 2004]),
        ),
    ],
)
def test_inference_years_valid_ranges(
    dataset_config,
    requested,
    expected,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
        inference_years=requested,
    )

    np.testing.assert_array_equal(
        cfg._inference_years,
        expected,
    )


@pytest.mark.parametrize(
    "requested",
    [
        (1999, 2000),
        (2003, 2005),
        (1990, 1991),
    ],
)
def test_inference_years_invalid_ranges(
    dataset_config,
    requested,
):
    with pytest.raises(ValueError):
        InferenceDataloaderConfig(
            dataset_config=dataset_config,
            inference_years=requested,
        )


def test_read_dataset_config_from_train_when_already_set(
    dataset_config,
    monkeypatch,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    called = {"value": False}

    def fake_from_train(value):
        called["value"] = True
        return DummyDatasetConfig()

    monkeypatch.setattr(
        "cccma_ppp.inference.dataloader._from_train",
        fake_from_train,
    )

    cfg.read_datasetConfig_from_train(DummyTrainDatasetConfig())

    assert called["value"] is False
    assert cfg.dataset_config is dataset_config
    assert cfg.train_dataset_config is None


def test_read_dataset_config_from_train_when_missing(
    monkeypatch,
):
    converted = DummyDatasetConfig()
    train_config = DummyTrainDatasetConfig()

    monkeypatch.setattr(
        "cccma_ppp.inference.dataloader._from_train",
        lambda value: converted,
    )

    cfg = InferenceDataloaderConfig(
        dataset_config=None,
    )

    cfg.read_datasetConfig_from_train(train_config)

    assert cfg.dataset_config is converted
    assert cfg.train_dataset_config is train_config


def test_input_metadata_requires_config():
    cfg = InferenceDataloaderConfig(
        dataset_config=None,
    )

    with pytest.raises(RuntimeError):
        _ = cfg.input_var_metadata


def test_input_metadata(dataset_config):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert cfg.input_var_metadata == {"tas": {}}
    assert dataset_config.ds_operator.input_calls == 1


def test_target_metadata_requires_train_config(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    with pytest.raises(RuntimeError):
        _ = cfg.target_var_metadata


def test_target_metadata_success(dataset_config):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    train_config = DummyTrainDatasetConfig()
    cfg.train_dataset_config = train_config

    assert cfg.target_var_metadata == {"obs": {}}
    assert train_config.ds_operator.target_calls == 1


@pytest.mark.parametrize(
    "model_name,condition_name,existing,expected",
    [
        (None, None, (), True),
        ("model", None, (), False),
        ("model", None, ("model",), True),
        (None, "condition", (), False),
        (
            None,
            "condition",
            ("condition",),
            True,
        ),
        (
            "model",
            "condition",
            (),
            False,
        ),
        (
            "model",
            "condition",
            ("model",),
            False,
        ),
        (
            "model",
            "condition",
            ("condition",),
            False,
        ),
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

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert cfg._input_preprocessor_exists(tmp_path) is expected


def test_input_preprocessor_exists_requires_config(
    tmp_path,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=None,
    )

    with pytest.raises(RuntimeError):
        cfg._input_preprocessor_exists(tmp_path)


def test_input_preprocessor_exists_default_runtime_directory(
    dataset_config,
    monkeypatch,
    tmp_path,
):
    preprocessing_dir = tmp_path / "preprocessing_pipeline"
    preprocessing_dir.mkdir()

    dataset_config.model = DummyDataSource("model")
    dataset_config.condition = None

    (preprocessing_dir / "model_preprocessing_pipeline.joblib").touch()

    monkeypatch.setattr(
        RuntimeContext,
        "GLOBAL_EXP_DIR",
        str(tmp_path),
    )

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert cfg._input_preprocessor_exists() is True


def test_setup_distributed_requires_config():
    cfg = InferenceDataloaderConfig(
        dataset_config=None,
    )

    with pytest.raises(RuntimeError):
        cfg.setup_distributed(
            DummyTrainLoader(),
            DummyDistributed(),
        )


@pytest.mark.parametrize(
    "exists,root,fit_expected",
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
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    monkeypatch.setattr(
        cfg,
        "_input_preprocessor_exists",
        lambda *args, **kwargs: exists,
    )

    train_loader = DummyTrainLoader()
    distributed = DummyDistributed(
        rank=0 if root else 1,
        world_size=2,
        root=root,
    )

    cfg.setup_distributed(
        train_loader,
        distributed,
    )

    assert train_loader.dataset_config.fit_called is fit_expected
    assert dataset_config.load_called is True
    assert distributed.barrier_called is True
    assert cfg.rank == distributed.rank
    assert cfg.world_size == distributed.world_size
    assert cfg._setup is True


def test_setup_distributed_passes_fit_arguments(
    dataset_config,
    monkeypatch,
    tmp_path,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    monkeypatch.setattr(
        cfg,
        "_input_preprocessor_exists",
        lambda *args, **kwargs: False,
    )

    train_loader = DummyTrainLoader()
    distributed = DummyDistributed(root=True)

    cfg.setup_distributed(
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


def test_setup_distributed_existing_preprocessors_skips_fit(
    dataset_config,
    monkeypatch,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    monkeypatch.setattr(
        cfg,
        "_input_preprocessor_exists",
        lambda *args, **kwargs: True,
    )

    train_loader = DummyTrainLoader()
    distributed = DummyDistributed(root=True)

    cfg.setup_distributed(
        train_loader,
        distributed,
    )

    assert train_loader.dataset_config.fit_called is False
    assert dataset_config.load_called is True


def test_build_loader_requires_setup(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    with pytest.raises(RuntimeError):
        cfg.build_inference_loader()


@pytest.mark.parametrize(
    "return_mask,reduce_mask",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_build_loader_argument_matrix(
    dataset_config,
    patched_dataloader,
    return_mask,
    reduce_mask,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
        batch_size=4,
    )

    cfg.rank = 2
    cfg.world_size = 8
    cfg._setup = True

    loader = cfg.build_inference_loader(
        return_spatial_mask=return_mask,
        reduce_spatial_mask=reduce_mask,
    )

    assert isinstance(
        loader,
        FakeDataloader,
    )
    assert loader.rank == 2
    assert loader.world_size == 8
    assert loader.shuffle is False
    assert loader.return_spatial_mask is return_mask
    assert loader.reduce_spatial_mask is reduce_mask
    assert loader.collate_fn is collate_batch
    assert dataset_config.build_called is True
    assert dataset_config.return_metadata is True


def test_build_loader_passes_default_years(
    dataset_config,
    patched_dataloader,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg.rank = 0
    cfg.world_size = 1
    cfg._setup = True

    cfg.build_inference_loader()

    np.testing.assert_array_equal(
        dataset_config.years,
        dataset_config.available_times,
    )


def test_build_loader_passes_explicit_years(
    dataset_config,
    patched_dataloader,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
        inference_years=(2001, 2002),
    )

    cfg.rank = 0
    cfg.world_size = 1
    cfg._setup = True

    cfg.build_inference_loader()

    np.testing.assert_array_equal(
        dataset_config.years,
        np.array([2001, 2002]),
    )

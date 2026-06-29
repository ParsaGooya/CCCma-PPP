import numpy as np
import pytest
import torch

from cccma_ppp.inference.dataloader import (
    BatchData,
    InferenceDataloaderConfig,
    collate_batch,
)


class DummyDSOperator:
    def get_input_var_metadata(self):
        return {"tas": {}}

    def get_target_var_metadata(self):
        return {"obs": {}}


class DummyPreprocessingPipeline:
    def __init__(self, name):
        self.name = name


class DummyDataSource:
    def __init__(self, name):
        self.preprocessing_pipeline = DummyPreprocessingPipeline(name)


class DummyDatasetConfig:
    def __init__(self):
        self.available_inference_time = np.array([2000, 2001, 2002, 2003])

        self.ds_operator = DummyDSOperator()

        self.model = None
        self.condition = None

        self.load_called = False
        self.build_called = False

    def _load_fitted_preprocessors(
        self,
        load_dir=None,
    ):
        self.load_called = True

    def build_dataset(
        self,
        years,
        return_metadata=False,
    ):
        self.build_called = True
        self.years = years
        self.return_metadata = return_metadata

        return ["dataset"]


class DummyTrainDatasetConfig:
    def __init__(self):
        self.fit_called = False

    def _fit_preprocessors(
        self,
        train_years,
        save=True,
        save_path=None,
    ):
        self.fit_called = True


class DummyTrainLoader:
    def __init__(self):
        self.dataset_config = DummyTrainDatasetConfig()

        self.train_years = [2000]


class DummyDistributed:
    def __init__(
        self,
        rank=0,
        world_size=1,
        root=True,
    ):
        self.rank = rank
        self.world_size = world_size
        self._root = root

        self.barrier_called = False

    def is_root(self):
        return self._root

    def barrier(self):
        self.barrier_called = True


@pytest.fixture
def dataset_config():
    return DummyDatasetConfig()


@pytest.mark.pruned
def test_batchdata_nan_replacement():
    batch = BatchData(input=torch.tensor([[1.0, float("nan")]]))

    assert torch.isnan(batch.input).sum() == 0
    assert batch.input[0, 1] == 0


@pytest.mark.pruned
def test_batchdata_spatial_mask_created():
    batch = BatchData(
        input=torch.tensor([[1.0, float("nan")]]),
        return_spatial_mask=True,
    )

    values, mask = batch.input

    assert mask.shape == values.shape
    assert mask[0, 0] == 1
    assert mask[0, 1] == 0


@pytest.mark.pruned
def test_batchdata_reduce_spatial_mask():

    with pytest.raises(RuntimeError):
        BatchData(
            input=torch.tensor(
                [
                    [1.0, 2.0],
                    [1.0, float("nan")],
                ]
            ),
            return_spatial_mask=True,
            reduce_spatial_mask=True,
        )


@pytest.mark.pruned
def test_batchdata_to_device():
    batch = BatchData(input=torch.ones(2, 2))

    returned = batch.to_device("cpu")

    assert returned is batch
    assert batch.input.device.type == "cpu"


def test_batchdata_to_device_with_added_features():
    batch = BatchData(
        input=torch.ones(2, 2),
        added_features=torch.ones(2),
    )

    batch.to_device("cpu")

    assert batch.added_features.device.type == "cpu"


def test_batchdata_to_device_with_spatial_mask():
    batch = BatchData(
        input=torch.ones(2, 2),
        return_spatial_mask=True,
    )

    batch.to_device("cpu")

    data, mask = batch.input

    assert data.device.type == "cpu"
    assert mask.device.type == "cpu"


@pytest.mark.pruned
def test_collate_batch_basic():
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


@pytest.mark.pruned
def test_collate_batch_with_metadata():
    batch = [
        (
            {
                "input": torch.ones(2, 2),
                "added_features": None,
            },
            {"year": 2000},
        ),
        (
            {
                "input": torch.zeros(2, 2),
                "added_features": None,
            },
            {"year": 2001},
        ),
    ]

    result = collate_batch(batch)

    assert len(result.metadata) == 2


@pytest.mark.pruned
def test_collate_batch_with_added_features():
    batch = [
        {
            "input": torch.ones(2, 2),
            "added_features": torch.tensor([1.0]),
        },
        {
            "input": torch.ones(2, 2),
            "added_features": torch.tensor([2.0]),
        },
    ]

    result = collate_batch(batch)

    assert result.added_features.shape == (2, 1)


@pytest.mark.pruned
def test_collate_batch_metadata_and_added_features():
    batch = [
        (
            {
                "input": torch.ones(2, 2),
                "added_features": torch.tensor([1.0]),
            },
            {"year": 2000},
        ),
        (
            {
                "input": torch.ones(2, 2),
                "added_features": torch.tensor([2.0]),
            },
            {"year": 2001},
        ),
    ]

    result = collate_batch(batch)

    assert len(result.metadata) == 2
    assert result.added_features.shape == (2, 1)


@pytest.mark.pruned
def test_collate_batch_spatial_mask():
    batch = [
        {
            "input": torch.tensor([1.0, float("nan")]),
            "added_features": None,
        }
    ]

    result = collate_batch(
        batch,
        return_spatial_mask=True,
    )

    assert isinstance(result.input, tuple)


@pytest.mark.pruned
def test_prefetch_factor_removed_when_no_workers(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
        num_data_workers=0,
        prefetch_factor=8,
    )

    assert cfg.prefetch_factor is None


def test_dataset_config_required():
    with pytest.raises(RuntimeError):
        InferenceDataloaderConfig(
            dataset_config=None,
        )


@pytest.mark.pruned
def test_inference_years_default(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert np.array_equal(
        cfg._inference_years,
        dataset_config.available_inference_time,
    )


def test_inference_years_range_current_bug(
    dataset_config,
):
    with pytest.raises(AttributeError):
        InferenceDataloaderConfig(
            dataset_config=dataset_config,
            inference_years=(2001, 2002),
        )


@pytest.mark.pruned
def test_inference_years_invalid_current_bug(
    dataset_config,
):
    with pytest.raises(AttributeError):
        InferenceDataloaderConfig(
            dataset_config=dataset_config,
            inference_years=(1990, 1991),
        )


@pytest.mark.pruned
def test_available_inference_years(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert np.array_equal(
        cfg.available_inference_years,
        dataset_config.available_inference_time,
    )


@pytest.mark.pruned
def test_input_preprocessor_exists_model_only(
    dataset_config,
    tmp_path,
):
    dataset_config.model = DummyDataSource("model")

    (tmp_path / "model_preprocessing_pipeline.joblib").touch()

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert cfg._input_preprocessor_exists(tmp_path)


@pytest.mark.pruned
def test_input_preprocessor_exists_condition_only(
    dataset_config,
    tmp_path,
):
    dataset_config.condition = DummyDataSource("condition")

    (tmp_path / "condition_preprocessing_pipeline.joblib").touch()

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert cfg._input_preprocessor_exists(tmp_path)


def test_input_preprocessor_exists_model_and_condition(
    dataset_config,
    tmp_path,
):
    dataset_config.model = DummyDataSource("model")

    dataset_config.condition = DummyDataSource("condition")

    (tmp_path / "model_preprocessing_pipeline.joblib").touch()

    (tmp_path / "condition_preprocessing_pipeline.joblib").touch()

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert cfg._input_preprocessor_exists(tmp_path)


@pytest.mark.pruned
def test_input_preprocessor_missing(
    dataset_config,
    tmp_path,
):
    dataset_config.model = DummyDataSource("model")

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert not cfg._input_preprocessor_exists(tmp_path)


@pytest.mark.pruned
def test_input_preprocessor_exists_with_no_sources(
    dataset_config,
    tmp_path,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert cfg._input_preprocessor_exists(tmp_path)


def test_target_metadata_requires_train_dataset(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    with pytest.raises(RuntimeError):
        _ = cfg.target_var_metadata


def test_target_metadata_success(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    class TrainConfig:
        ds_operator = DummyDSOperator()

    cfg.train_dataset_config = TrainConfig()

    assert cfg.target_var_metadata == {"obs": {}}


@pytest.mark.pruned
def test_input_metadata_success(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert cfg.input_var_metadata == {"tas": {}}


def test_setup_distributed_fits_when_missing(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg._input_preprocessor_exists = lambda *args, **kwargs: False

    train_loader = DummyTrainLoader()

    distributed = DummyDistributed(root=True)

    cfg.setup_distributed(
        train_loader,
        distributed,
    )

    assert train_loader.dataset_config.fit_called

    assert dataset_config.load_called
    assert cfg._setup is True


def test_setup_distributed_skip_fit_when_present(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg._input_preprocessor_exists = lambda *args, **kwargs: True

    train_loader = DummyTrainLoader()

    distributed = DummyDistributed(root=True)

    cfg.setup_distributed(
        train_loader,
        distributed,
    )

    assert not train_loader.dataset_config.fit_called

    assert dataset_config.load_called
    assert cfg._setup is True


@pytest.mark.pruned
def test_setup_distributed_non_root_branch(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg._input_preprocessor_exists = lambda *args, **kwargs: True

    train_loader = DummyTrainLoader()

    distributed = DummyDistributed(
        rank=1,
        world_size=2,
        root=False,
    )

    cfg.setup_distributed(
        train_loader,
        distributed,
    )

    assert cfg.rank == 1
    assert cfg.world_size == 2
    assert cfg._setup is True


def test_build_loader_before_setup_raises(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    with pytest.raises(RuntimeError):
        cfg.build_inference_loader()


@pytest.mark.pruned
def test_build_loader_after_setup(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg.rank = 0
    cfg.world_size = 1
    cfg._setup = True

    loader = cfg.build_inference_loader()

    assert loader is not None
    assert dataset_config.build_called


@pytest.mark.pruned
def test_build_loader_with_spatial_mask(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg.rank = 0
    cfg.world_size = 1
    cfg._setup = True

    loader = cfg.build_inference_loader(
        return_spatial_mask=True,
        reduce_spatial_mask=True,
    )

    assert loader is not None


@pytest.mark.pruned
def test_collate_batch_with_metadata_and_spatial_mask():
    batch = [
        (
            {
                "input": torch.tensor([1.0, float("nan")]),
                "added_features": None,
            },
            {"year": 2000},
        )
    ]

    result = collate_batch(
        batch,
        return_spatial_mask=True,
    )

    assert result.metadata == [{"year": 2000}]


def test_collate_batch_with_added_features_and_spatial_mask():
    batch = [
        {
            "input": torch.tensor([1.0, float("nan")]),
            "added_features": torch.tensor([1.0]),
        }
    ]

    result = collate_batch(
        batch,
        return_spatial_mask=True,
    )

    assert result.added_features is not None


@pytest.mark.pruned
def test_input_preprocessor_exists_condition_missing(
    dataset_config,
    tmp_path,
):
    dataset_config.condition = DummyDataSource("condition")

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert not cfg._input_preprocessor_exists(tmp_path)


@pytest.mark.pruned
def test_input_preprocessor_exists_model_present_condition_missing(
    dataset_config,
    tmp_path,
):
    dataset_config.model = DummyDataSource("model")

    dataset_config.condition = DummyDataSource("condition")

    (tmp_path / "model_preprocessing_pipeline.joblib").touch()

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert not cfg._input_preprocessor_exists(tmp_path)


def test_setup_distributed_non_root_missing_preprocessors(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg._input_preprocessor_exists = lambda *args, **kwargs: False

    train_loader = DummyTrainLoader()

    distributed = DummyDistributed(
        rank=1,
        world_size=2,
        root=False,
    )

    cfg.setup_distributed(
        train_loader,
        distributed,
    )

    assert cfg._setup


def test_build_loader_world_size_two(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg.rank = 1
    cfg.world_size = 2
    cfg._setup = True

    loader = cfg.build_inference_loader()

    assert loader is not None


@pytest.mark.pruned
def test_input_var_metadata_multiple_accesses(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert cfg.input_var_metadata == {"tas": {}}

    assert cfg.input_var_metadata == {"tas": {}}


@pytest.mark.pruned
def test_collate_batch_metadata_spatial_mask_added_features():
    batch = [
        (
            {
                "input": torch.tensor([1.0, float("nan")]),
                "added_features": torch.tensor([1.0]),
            },
            {"year": 2000},
        )
    ]

    result = collate_batch(
        batch,
        return_spatial_mask=True,
    )

    assert result.metadata == [{"year": 2000}]
    assert result.added_features is not None


def test_collate_batch_metadata_spatial_mask_reduce():
    batch = [
        (
            {
                "input": torch.tensor([1.0, float("nan")]),
                "added_features": None,
            },
            {"year": 2000},
        )
    ]

    with pytest.raises(RuntimeError):
        collate_batch(
            batch,
            return_spatial_mask=True,
            reduce_spatial_mask=True,
        )


@pytest.mark.pruned
def test_batchdata_to_device_metadata_only():
    batch = BatchData(
        input=torch.ones(2, 2),
        metadata=[{"year": 2000}],
    )

    batch.to_device("cpu")

    assert batch.metadata == [{"year": 2000}]


@pytest.mark.pruned
def test_input_preprocessor_exists_condition_present_model_missing(
    dataset_config,
    tmp_path,
):
    dataset_config.model = DummyDataSource("model")

    dataset_config.condition = DummyDataSource("condition")

    (tmp_path / "condition_preprocessing_pipeline.joblib").touch()

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert not cfg._input_preprocessor_exists(tmp_path)


@pytest.mark.pruned
def test_setup_distributed_root_world_size_two(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg._input_preprocessor_exists = lambda *args, **kwargs: True

    train_loader = DummyTrainLoader()

    distributed = DummyDistributed(
        rank=0,
        world_size=2,
        root=True,
    )

    cfg.setup_distributed(
        train_loader,
        distributed,
    )

    assert cfg.rank == 0
    assert cfg.world_size == 2


@pytest.mark.pruned
def test_setup_distributed_non_root_world_size_one(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg._input_preprocessor_exists = lambda *args, **kwargs: True

    train_loader = DummyTrainLoader()

    distributed = DummyDistributed(
        rank=0,
        world_size=1,
        root=False,
    )

    cfg.setup_distributed(
        train_loader,
        distributed,
    )

    assert cfg._setup


@pytest.mark.pruned
def test_input_var_metadata_multiple_access():
    ds = DummyDatasetConfig()

    cfg = InferenceDataloaderConfig(
        dataset_config=ds,
    )

    assert cfg.input_var_metadata == {"tas": {}}
    assert cfg.input_var_metadata == {"tas": {}}


@pytest.mark.pruned
def test_target_var_metadata_multiple_access():
    ds = DummyDatasetConfig()

    cfg = InferenceDataloaderConfig(
        dataset_config=ds,
    )

    class TrainCfg:
        ds_operator = DummyDSOperator()

    cfg.train_dataset_config = TrainCfg()

    assert cfg.target_var_metadata == {"obs": {}}
    assert cfg.target_var_metadata == {"obs": {}}


@pytest.mark.pruned
def test_build_loader_rank_zero_world_size_two(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg.rank = 0
    cfg.world_size = 2
    cfg._setup = True

    loader = cfg.build_inference_loader()

    assert loader is not None


@pytest.mark.pruned
def test_build_loader_spatial_mask_only(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg.rank = 0
    cfg.world_size = 1
    cfg._setup = True

    loader = cfg.build_inference_loader(
        return_spatial_mask=True,
    )

    assert loader is not None


@pytest.mark.pruned
def test_build_loader_reduce_spatial_mask_only(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg.rank = 0
    cfg.world_size = 1
    cfg._setup = True

    loader = cfg.build_inference_loader(
        reduce_spatial_mask=True,
    )

    assert loader is not None


@pytest.mark.pruned
def test_available_inference_years_multiple_access(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    years1 = cfg.available_inference_years
    years2 = cfg.available_inference_years

    assert np.array_equal(years1, years2)


@pytest.mark.pruned
def test_init_without_dataset_config_raises():
    with pytest.raises(
        RuntimeError,
        match="Inference dataset_config must be resolved",
    ):
        InferenceDataloaderConfig(
            dataset_config=None,
        )


@pytest.mark.pruned
def test_available_inference_years_property_direct(
    dataset_config,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    result = cfg.available_inference_years

    assert np.array_equal(
        result,
        np.array([2000, 2001, 2002, 2003]),
    )


@pytest.mark.parametrize(
    "exists,is_root,fit_expected",
    [
        (True, True, False),
        (True, False, False),
        (False, True, True),
        (False, False, False),
    ],
)
def test_setup_distributed_branch_matrix(
    dataset_config,
    exists,
    is_root,
    fit_expected,
):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg._input_preprocessor_exists = lambda *a, **k: exists

    train_loader = DummyTrainLoader()

    distributed = DummyDistributed(
        root=is_root,
        rank=0 if is_root else 1,
        world_size=2,
    )

    cfg.setup_distributed(
        train_loader,
        distributed,
    )

    assert train_loader.dataset_config.fit_called is fit_expected

    assert dataset_config.load_called
    assert distributed.barrier_called
    assert cfg._setup


@pytest.mark.pruned
def test_input_preprocessor_exists_empty_list_returns_true(
    dataset_config,
    tmp_path,
):
    dataset_config.model = None
    dataset_config.condition = None

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert cfg._input_preprocessor_exists(tmp_path) is True


def test_input_preprocessor_exists_default_runtime_dir(
    dataset_config,
    tmp_path,
):
    from cccma_ppp.generic import RuntimeContext

    RuntimeContext.GLOBAL_EXP_DIR = str(tmp_path)

    (tmp_path / "preprocessing_pipeline").mkdir()

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg._input_preprocessor_exists()

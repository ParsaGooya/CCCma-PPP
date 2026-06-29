# tests/inference/test_dataloader.py

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from cccma_ppp.inference.dataloader import (
    BatchData,
    InferenceDataloaderConfig,
    collate_batch,
)


# ------------------------------------------------------------------
# BatchData
# ------------------------------------------------------------------


def test_batchdata_nan_replacement():
    x = torch.tensor([[1.0, float("nan")]])

    batch = BatchData(input=x)

    assert torch.isnan(batch.input).sum() == 0
    assert batch.input[0, 1] == 0


def test_batchdata_spatial_mask_created():
    x = torch.tensor([[1.0, float("nan")]])

    batch = BatchData(
        input=x,
        return_spatial_mask=True,
    )

    assert isinstance(batch.input, tuple)

    values, mask = batch.input

    assert mask.shape == values.shape
    assert mask[0, 0] == 1
    assert mask[0, 1] == 0


def test_batchdata_reduce_spatial_mask():
    x = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, float("nan")],
        ]
    )

    batch = BatchData(
        input=x,
        return_spatial_mask=True,
        reduce_spatial_mask=True,
    )

    _, mask = batch.input

    assert mask.ndim < x.ndim


def test_batchdata_to_device():
    x = torch.ones(2, 2)

    batch = BatchData(input=x)

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


# ------------------------------------------------------------------
# collate_batch
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# InferenceDataloaderConfig
# ------------------------------------------------------------------


@pytest.fixture
def dataset_config():
    cfg = MagicMock()

    cfg.available_inference_time = np.array([2000, 2001, 2002, 2003])

    cfg.build_dataset.return_value = MagicMock()

    cfg.ds_operator.get_input_var_metadata.return_value = {"tas": {}}

    return cfg


def test_prefetch_factor_removed_when_no_workers(dataset_config):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
        num_data_workers=0,
    )

    assert cfg.prefetch_factor is None


def test_dataset_config_required():
    with pytest.raises(RuntimeError):
        InferenceDataloaderConfig(
            dataset_config=None,
        )


def test_inference_years_default(dataset_config):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    years = cfg._inference_years

    assert np.array_equal(
        years,
        dataset_config.available_inference_time,
    )


def test_inference_years_range(dataset_config):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
        inference_years=(2001, 2002),
    )

    years = cfg._inference_years

    assert np.array_equal(
        years,
        np.array([2001, 2002]),
    )


def test_inference_years_invalid(dataset_config):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
        inference_years=(1990, 1991),
    )

    cfg.available_inference_time = np.array([2000, 2001, 2002])

    with pytest.raises(ValueError):
        _ = cfg._inference_years


def test_available_inference_years(dataset_config):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    assert np.array_equal(
        cfg.available_inference_years,
        dataset_config.available_inference_time,
    )


# ------------------------------------------------------------------
# preprocessor existence
# ------------------------------------------------------------------


def test_input_preprocessor_exists_model_only(tmp_path):
    preprocessing = SimpleNamespace(name="model")

    model = SimpleNamespace(preprocessing_pipeline=preprocessing)

    ds = MagicMock()
    ds.model = model
    ds.condition = None

    file = tmp_path / "model_preprocessing_pipeline.joblib"
    file.touch()

    cfg = InferenceDataloaderConfig(
        dataset_config=ds,
    )

    assert cfg._input_preprocessor_exists(tmp_path)


def test_input_preprocessor_missing(tmp_path):
    preprocessing = SimpleNamespace(name="model")

    model = SimpleNamespace(preprocessing_pipeline=preprocessing)

    ds = MagicMock()
    ds.model = model
    ds.condition = None

    cfg = InferenceDataloaderConfig(
        dataset_config=ds,
    )

    assert not cfg._input_preprocessor_exists(tmp_path)


# ------------------------------------------------------------------
# target metadata
# ------------------------------------------------------------------


def test_target_metadata_requires_train_dataset(dataset_config):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    with pytest.raises(RuntimeError):
        _ = cfg.target_var_metadata


def test_target_metadata_success(dataset_config):
    train_ds = MagicMock()

    train_ds.ds_operator.get_target_var_metadata.return_value = {"obs": {}}

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg.train_dataset_config = train_ds

    assert cfg.target_var_metadata == {"obs": {}}


# ------------------------------------------------------------------
# setup_distributed
# ------------------------------------------------------------------


def test_setup_distributed_fits_when_missing(dataset_config):
    distributed = MagicMock()

    distributed.rank = 0
    distributed.world_size = 1
    distributed.is_root.return_value = True

    train_loader = MagicMock()

    train_loader.dataset_config = MagicMock()

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg._input_preprocessor_exists = MagicMock(return_value=False)

    cfg.setup_distributed(
        train_loader,
        distributed,
    )

    train_loader.dataset_config._fit_preprocessors.assert_called_once()
    dataset_config._load_fitted_preprocessors.assert_called_once()
    assert cfg._setup is True


def test_setup_distributed_skip_fit_when_present(dataset_config):
    distributed = MagicMock()

    distributed.rank = 0
    distributed.world_size = 1
    distributed.is_root.return_value = True

    train_loader = MagicMock()
    train_loader.dataset_config = MagicMock()

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg._input_preprocessor_exists = MagicMock(return_value=True)

    cfg.setup_distributed(
        train_loader,
        distributed,
    )

    train_loader.dataset_config._fit_preprocessors.assert_not_called()
    dataset_config._load_fitted_preprocessors.assert_called_once()


# ------------------------------------------------------------------
# build_inference_loader
# ------------------------------------------------------------------


def test_build_loader_before_setup_raises(dataset_config):
    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    with pytest.raises(RuntimeError):
        cfg.build_inference_loader()


def test_build_loader_after_setup(dataset_config, monkeypatch):
    fake_loader = object()

    cfg = InferenceDataloaderConfig(
        dataset_config=dataset_config,
    )

    cfg.rank = 0
    cfg.world_size = 1
    cfg._setup = True

    monkeypatch.setattr(
        "cccma_ppp.inference.dataloader.Dataloader",
        lambda **kwargs: fake_loader,
    )

    loader = cfg.build_inference_loader()

    assert loader is fake_loader

    dataset_config.build_dataset.assert_called_once()

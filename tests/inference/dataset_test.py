# tests/inference/test_dataset.py

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cccma_ppp.inference.dataset import (
    InferenceDataset,
    InferenceDatasetConfig,
)


# ============================================================
# InferenceDatasetConfig._check_model
# ============================================================


def test_same_member_with_ensemble_mean_model_raises():
    cfg = InferenceDatasetConfig(
        model=MagicMock(ensemble_mean=True),
        condition_method="same_member",
    )

    with pytest.raises(ValueError):
        cfg._check_model()


# ============================================================
# InferenceDatasetConfig._check_condition
# ============================================================


def test_condition_without_method_raises():
    cfg = InferenceDatasetConfig(
        model=MagicMock(),
        condition=MagicMock(),
        condition_method=None,
    )

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_same_member_requires_ensemble_dimension():
    condition = MagicMock()
    condition.ensemble_dim = None

    cfg = InferenceDatasetConfig(
        model=MagicMock(),
        condition=condition,
        condition_method="same_member",
    )

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_cross_ensemble_requires_ensemble_dimension():
    condition = MagicMock()
    condition.ensemble_dim = None

    cfg = InferenceDatasetConfig(
        model=MagicMock(),
        condition=condition,
        condition_method="cross_ensemble",
    )

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_static_condition_requires_dataset():
    cfg = InferenceDatasetConfig(
        model=MagicMock(),
        condition=None,
        condition_method="static",
    )

    with pytest.raises(ValueError):
        cfg._check_condition()


def test_static_condition_model_equals_condition_raises():
    model = MagicMock()

    cfg = InferenceDatasetConfig(
        model=model,
        condition=model,
        condition_method="static",
    )

    with pytest.raises(ValueError):
        cfg._check_condition()


# ============================================================
# _from_train
# ============================================================


def test_from_train_observation_only():
    train_cfg = MagicMock()

    train_cfg.input_dataset = MagicMock()
    train_cfg.condition_dataset = None

    cfg = InferenceDatasetConfig._from_train(train_cfg)

    assert cfg is not None


def test_from_train_with_condition_dataset():
    train_cfg = MagicMock()

    train_cfg.input_dataset = MagicMock()
    train_cfg.condition_dataset = MagicMock()

    cfg = InferenceDatasetConfig._from_train(train_cfg)

    assert cfg.condition is not None


def test_from_train_invalid_configuration():
    train_cfg = MagicMock()

    train_cfg.input_dataset = None
    train_cfg.condition_dataset = None

    with pytest.raises(Exception):
        InferenceDatasetConfig._from_train(train_cfg)


# ============================================================
# Dataset construction
# ============================================================


@pytest.fixture
def ds_operator():
    op = MagicMock()

    op.time = np.array([2000, 2001, 2002])

    return op


@pytest.fixture
def dataset_config(ds_operator):
    cfg = MagicMock()

    cfg.ds_operator = ds_operator

    cfg.model = MagicMock()
    cfg.condition = None

    cfg.return_metadata = False
    cfg.concat_condition = False

    cfg.model.load.return_value = np.ones((3, 5, 10))

    return cfg


# ============================================================
# Initialization branches
# ============================================================


def test_dataset_requires_fitted_preprocessors(dataset_config):
    dataset_config.preprocessors_loaded = False

    with pytest.raises(RuntimeError):
        InferenceDataset(
            dataset_config,
            years=np.array([2000]),
        )


def test_requested_year_outside_available(dataset_config):
    dataset_config.preprocessors_loaded = True

    with pytest.raises(ValueError):
        InferenceDataset(
            dataset_config,
            years=np.array([1990]),
        )


# ============================================================
# get_cond_indexes
# ============================================================


def test_same_member_indexes():
    ds = MagicMock(spec=InferenceDataset)

    ds.condition_method = "same_member"

    idx = InferenceDataset.get_cond_indexes(
        ds,
        sample_idx=3,
        ensemble_idx=2,
    )

    assert idx is not None


def test_cross_ensemble_indexes():
    ds = MagicMock(spec=InferenceDataset)

    ds.condition_method = "cross_ensemble"

    idx = InferenceDataset.get_cond_indexes(
        ds,
        sample_idx=3,
        ensemble_idx=2,
    )

    assert idx is not None


def test_static_indexes():
    ds = MagicMock(spec=InferenceDataset)

    ds.condition_method = "static"

    idx = InferenceDataset.get_cond_indexes(
        ds,
        sample_idx=3,
        ensemble_idx=2,
    )

    assert idx is not None


# ============================================================
# Shape helpers
# ============================================================


def test_get_input_shape_model_only(dataset_config):
    dataset_config.preprocessors_loaded = True

    ds = InferenceDataset(
        dataset_config,
        years=np.array([2000]),
    )

    shape = ds.get_input_shape()

    assert shape is not None


def test_get_added_features_dim_none(dataset_config):
    dataset_config.preprocessors_loaded = True

    ds = InferenceDataset(
        dataset_config,
        years=np.array([2000]),
    )

    assert ds.get_added_features_dim() is None


def test_get_added_features_dim_present(dataset_config):
    dataset_config.preprocessors_loaded = True

    dataset_config.added_features_dim = 5

    ds = InferenceDataset(
        dataset_config,
        years=np.array([2000]),
    )

    assert ds.get_added_features_dim() == 5


# ============================================================
# __getitem__
# ============================================================


def test_getitem_model_only(dataset_config):
    dataset_config.preprocessors_loaded = True

    ds = InferenceDataset(
        dataset_config,
        years=np.array([2000]),
    )

    sample = ds[0]

    assert "input" in sample


def test_getitem_with_metadata(dataset_config):
    dataset_config.preprocessors_loaded = True
    dataset_config.return_metadata = True

    ds = InferenceDataset(
        dataset_config,
        years=np.array([2000]),
    )

    sample, metadata = ds[0]

    assert metadata is not None


def test_getitem_condition_replaces_input(dataset_config):
    dataset_config.preprocessors_loaded = True

    dataset_config.condition = MagicMock()
    dataset_config.concat_condition = False

    ds = InferenceDataset(
        dataset_config,
        years=np.array([2000]),
    )

    sample = ds[0]

    assert "input" in sample


def test_getitem_condition_concatenated(dataset_config):
    dataset_config.preprocessors_loaded = True

    dataset_config.condition = MagicMock()
    dataset_config.concat_condition = True

    ds = InferenceDataset(
        dataset_config,
        years=np.array([2000]),
    )

    sample = ds[0]

    assert "input" in sample


# ============================================================
# condition indexing paths
# ============================================================


def test_index_condition_dataset_same_member(dataset_config):
    dataset_config.preprocessors_loaded = True

    dataset_config.condition = MagicMock()

    ds = InferenceDataset(
        dataset_config,
        years=np.array([2000]),
    )

    ds.condition_method = "same_member"

    result = ds._index_condition_dataset(
        sample_idx=0,
        ensemble_idx=0,
    )

    assert result is not None


def test_index_condition_dataset_cross_ensemble(dataset_config):
    dataset_config.preprocessors_loaded = True

    dataset_config.condition = MagicMock()

    ds = InferenceDataset(
        dataset_config,
        years=np.array([2000]),
    )

    ds.condition_method = "cross_ensemble"

    result = ds._index_condition_dataset(
        sample_idx=0,
        ensemble_idx=0,
    )

    assert result is not None


def test_index_condition_dataset_static(dataset_config):
    dataset_config.preprocessors_loaded = True

    dataset_config.condition = MagicMock()

    ds = InferenceDataset(
        dataset_config,
        years=np.array([2000]),
    )

    ds.condition_method = "static"

    result = ds._index_condition_dataset(
        sample_idx=0,
        ensemble_idx=0,
    )

    assert result is not None

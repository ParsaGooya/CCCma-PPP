# tests/inference/test_inference_configs.py

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from cccma_ppp.inference.inference_configs import (
    InferenceConfig,
)


# ============================================================
# output_preprocessor_dir
# ============================================================


def test_output_preprocessor_dir_from_model():
    model = MagicMock()
    model.preprocessing_pipeline.name = "model"

    cfg = InferenceConfig(
        model=model,
        condition=None,
    )

    result = cfg.output_preprocessor_dir

    assert result is not None


def test_output_preprocessor_dir_from_condition():
    model = MagicMock()
    condition = MagicMock()

    condition.preprocessing_pipeline.name = "condition"

    cfg = InferenceConfig(
        model=model,
        condition=condition,
    )

    result = cfg.output_preprocessor_dir

    assert result is not None


# ============================================================
# save_dir
# ============================================================


def test_save_dir():
    cfg = InferenceConfig(
        experiment_dir=Path("/tmp/experiment"),
    )

    assert "inference" in str(cfg.save_dir)


# ============================================================
# ensemble generation checks
# ============================================================


def test_deterministic_model_multiple_ensembles_raises():
    model = MagicMock()

    model.is_deterministic = True

    cfg = InferenceConfig(
        model=model,
        output_ensemble_size=10,
    )

    with pytest.raises(ValueError):
        cfg._check_ensemble_generation()


def test_deterministic_model_single_ensemble_allowed():
    model = MagicMock()

    model.is_deterministic = True

    cfg = InferenceConfig(
        model=model,
        output_ensemble_size=1,
    )

    cfg._check_ensemble_generation()


def test_probabilistic_model_multiple_ensembles_allowed():
    model = MagicMock()

    model.is_deterministic = False

    cfg = InferenceConfig(
        model=model,
        output_ensemble_size=25,
    )

    cfg._check_ensemble_generation()


# ============================================================
# condition method validation
# ============================================================


def test_condition_dataset_without_method_raises():
    cfg = InferenceConfig(
        model=MagicMock(),
        condition=MagicMock(),
        condition_method=None,
    )

    with pytest.raises(ValueError):
        cfg._check_inference_dataset()


def test_condition_dataset_with_method_allowed():
    cfg = InferenceConfig(
        model=MagicMock(),
        condition=MagicMock(),
        condition_method="same_member",
    )

    cfg._check_inference_dataset()


# ============================================================
# dataset resolution
# ============================================================


def test_resolve_dataset_from_train_config():
    train_cfg = MagicMock()

    train_cfg.dataloader.dataset_config = MagicMock()

    cfg = InferenceConfig(
        train_config=train_cfg,
    )

    result = cfg._resolve_inference_dataset_config()

    assert result is not None


def test_resolve_dataset_existing_dataset_config():
    dataset_cfg = MagicMock()

    cfg = InferenceConfig(
        inference_dataset_config=dataset_cfg,
    )

    result = cfg._resolve_inference_dataset_config()

    assert result is dataset_cfg


# ============================================================
# metadata validation
# ============================================================


def test_input_metadata_matches():
    dataset_cfg = MagicMock()

    dataset_cfg.input_var_metadata = {"tas": {}}

    cfg = InferenceConfig(
        inference_dataset_config=dataset_cfg,
    )

    cfg._validate_metadata()


def test_input_metadata_mismatch_raises():
    dataset_cfg = MagicMock()

    dataset_cfg.input_var_metadata = {"tas": {}}

    cfg = InferenceConfig(
        inference_dataset_config=dataset_cfg,
    )

    cfg.expected_input_metadata = {"pr": {}}

    with pytest.raises(ValueError):
        cfg._validate_metadata()


# ============================================================
# post-init
# ============================================================


def test_post_init_runs_validation(monkeypatch):
    cfg = object.__new__(InferenceConfig)

    ensemble_called = False
    dataset_called = False

    def fake_ensemble():
        nonlocal ensemble_called
        ensemble_called = True

    def fake_dataset():
        nonlocal dataset_called
        dataset_called = True

    monkeypatch.setattr(
        cfg,
        "_check_ensemble_generation",
        fake_ensemble,
    )

    monkeypatch.setattr(
        cfg,
        "_check_inference_dataset",
        fake_dataset,
    )

    InferenceConfig.__post_init__(cfg)

    assert ensemble_called
    assert dataset_called


# ============================================================
# cVAE branches
# ============================================================


def test_cvae_requires_condition_method():
    model = MagicMock()

    model.is_cvae = True

    cfg = InferenceConfig(
        model=model,
        condition=MagicMock(),
        condition_method=None,
    )

    with pytest.raises(ValueError):
        cfg._check_inference_dataset()


def test_cvae_condition_method_present():
    model = MagicMock()

    model.is_cvae = True

    cfg = InferenceConfig(
        model=model,
        condition=MagicMock(),
        condition_method="same_member",
    )

    cfg._check_inference_dataset()


# ============================================================
# save path creation
# ============================================================


def test_save_dir_under_experiment():
    cfg = InferenceConfig(
        experiment_dir=Path("/tmp/my_exp"),
    )

    save_dir = cfg.save_dir

    assert str(save_dir).startswith("/tmp/my_exp")


def test_save_dir_contains_inference():
    cfg = InferenceConfig(
        experiment_dir=Path("/tmp/my_exp"),
    )

    assert "inference" in str(cfg.save_dir).lower()


# ============================================================
# edge cases
# ============================================================


def test_missing_train_and_dataset_config_raises():
    cfg = InferenceConfig(
        train_config=None,
        inference_dataset_config=None,
    )

    with pytest.raises(RuntimeError):
        cfg._resolve_inference_dataset_config()


def test_none_model_raises():
    cfg = InferenceConfig(
        model=None,
    )

    with pytest.raises((ValueError, AttributeError)):
        cfg._check_ensemble_generation()


def test_zero_output_ensemble():
    model = MagicMock()

    model.is_deterministic = False

    cfg = InferenceConfig(
        model=model,
        output_ensemble_size=0,
    )

    with pytest.raises(ValueError):
        cfg._check_ensemble_generation()


def test_negative_output_ensemble():
    model = MagicMock()

    model.is_deterministic = False

    cfg = InferenceConfig(
        model=model,
        output_ensemble_size=-5,
    )

    with pytest.raises(ValueError):
        cfg._check_ensemble_generation()

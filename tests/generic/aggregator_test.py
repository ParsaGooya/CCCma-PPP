import pytest
import numpy as np
import torch
import warnings
import tempfile
from pathlib import Path

from src.cccma_ppp.generic.aggregator import MetricsAggregator


# ============================================================
# MOCK DISTRIBUTED
# ============================================================


class DummyDistributed:
    def __init__(self):
        self.device = "cpu"

    def all_reduce_sum(self, tensor):
        # simulate single-rank (no change)
        return tensor


# ============================================================
# INIT TESTS
# ============================================================


def test_init_basic():
    agg = MetricsAggregator(DummyDistributed(), "train")
    assert agg.num_epochs_seen == 0
    assert agg.epoch_loss_terms == {}
    assert agg.epoch_times == []


def test_init_with_existing_data():
    agg = MetricsAggregator(
        DummyDistributed(),
        "train",
        epoch_loss_terms={"loss": [1, 2]},
        epoch_times=[1.0, 2.0],
    )
    assert agg.num_epochs_seen == 2


def test_init_invalid_lengths():
    with pytest.raises(AssertionError):
        MetricsAggregator(
            DummyDistributed(),
            "train",
            epoch_loss_terms={"loss1": [1], "loss2": [1, 2]},
        )


def test_init_epoch_times_mismatch():
    with pytest.raises(AssertionError):
        MetricsAggregator(
            DummyDistributed(),
            "train",
            epoch_loss_terms={"loss": [1, 2]},
            epoch_times=[1.0],  # mismatch
        )


# ============================================================
# RECORD BATCH
# ============================================================


def test_record_numeric_and_tensor():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": 2.0, "val": torch.tensor([3.0, 5.0])})
    assert pytest.approx(agg.loss_terms["loss"]) == 2.0
    assert pytest.approx(agg.loss_terms["val"]) == 4.0  # mean
    assert agg.num_batches_seen == 1


def test_record_ignore_none():
    agg = MetricsAggregator(DummyDistributed(), "train")
    agg.record({"loss": None})
    assert "loss" not in agg.loss_terms


# ============================================================
# DIST COMPUTE
# ============================================================


def test_dist_compute_basic():
    agg = MetricsAggregator(DummyDistributed(), "train")
    agg.record({"loss": 4.0})
    logs = agg._dist_compute()

    assert logs["loss"] == 4.0
    assert agg._aggregated_across_ranks


def test_dist_compute_zero_batches():
    agg = MetricsAggregator(DummyDistributed(), "train")
    logs = agg._dist_compute()

    # no updates → nan
    for v in logs.values():
        assert np.isnan(v)


# ============================================================
# RECORD EPOCH
# ============================================================


def test_record_epoch_append():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": 1.0})
    logs = agg._dist_compute()

    agg.record_epoch(logs, time_elapsed=2.5)

    assert agg.epoch_loss_terms["loss"][0] == 1.0
    assert agg.epoch_times[0] == 2.5
    assert agg.num_epochs_seen == 1


def test_record_epoch_replace():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": 1.0})
    logs = agg._dist_compute()
    agg.record_epoch(logs)

    agg._aggregated_across_ranks = True
    agg.record_epoch({"loss": 10.0}, replace_index=0)

    assert agg.epoch_loss_terms["loss"][0] == 10.0


def test_record_epoch_missing_key_replace():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": 1.0})
    logs = agg._dist_compute()
    agg.record_epoch(logs)

    agg._aggregated_across_ranks = True

    with pytest.raises(ValueError):
        agg.record_epoch({"new_loss": 2.0}, replace_index=0)


def test_record_epoch_without_sync():
    agg = MetricsAggregator(DummyDistributed(), "train")

    with pytest.raises(RuntimeError):
        agg.record_epoch({"loss": 1.0})


# ============================================================
# RESET
# ============================================================


def test_reset_after_epoch():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": 1.0})
    logs = agg._dist_compute()
    agg.record_epoch(logs)

    agg.reset_batch_losses()

    assert agg.num_batches_seen == 0
    assert agg.loss_terms == {}


def test_reset_warning():
    agg = MetricsAggregator(DummyDistributed(), "train")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        agg.reset_batch_losses()
        assert len(w) > 0


# ============================================================
# PLOT
# ============================================================


def make_agg(name, values):
    agg = MetricsAggregator(DummyDistributed(), name)
    agg.epoch_loss_terms = {"loss": values}
    agg.epoch_times = [1] * len(values)
    agg.num_epochs_seen = len(values)
    return agg


def test_plot_basic(tmp_path):
    agg1 = make_agg("train", [1, 2])
    agg2 = make_agg("val", [2, 3])

    MetricsAggregator.plot([agg1, agg2], plot_dir=tmp_path)

    files = list(tmp_path.glob("*.png"))
    assert len(files) > 0


def test_plot_no_aggregators(monkeypatch, tmp_path):
    monkeypatch.setenv("GLOBAL_FIGURES_DIR", str(tmp_path))

    with pytest.raises(ValueError):
        MetricsAggregator.plot([])


def test_plot_inconsistent_epochs(tmp_path):
    agg1 = make_agg("train", [1, 2])
    agg2 = make_agg("val", [1])

    with pytest.raises(ValueError):
        MetricsAggregator.plot([agg1, agg2], plot_dir=tmp_path)


def test_plot_no_epochs_recorded(tmp_path):
    agg = MetricsAggregator(DummyDistributed(), "train")

    with pytest.raises(ValueError):
        MetricsAggregator.plot([agg], plot_dir=tmp_path)


# ============================================================
# STATE DICT
# ============================================================


def test_state_dict_roundtrip():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": 2.0})
    logs = agg._dist_compute()
    agg.record_epoch(logs, time_elapsed=1.2)

    state = agg.state_dict()

    new_agg = MetricsAggregator(DummyDistributed(), "new")
    new_agg.load_state_dict(state)

    assert new_agg.name == "train"
    assert new_agg.epoch_loss_terms == agg.epoch_loss_terms
    assert new_agg.epoch_times == agg.epoch_times
    assert new_agg.num_epochs_seen == agg.num_epochs_seen


# ============================================================
# OTHERS
# ============================================================


def test_dist_compute_nan_branch():
    agg = MetricsAggregator(DummyDistributed(), "train")
    agg.loss_terms = {"loss": 0.0}
    agg.num_batches_seen = 0

    logs = agg._dist_compute()

    assert np.isnan(logs["loss"])


def test_record_ignores_invalid_types():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": "invalid", "another": None})

    assert len(agg.loss_terms) == 0


def test_record_epoch_replace_with_nan_time():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": 1.0})
    logs = agg._dist_compute()
    agg.record_epoch(logs, time_elapsed=5.0)

    agg._aggregated_across_ranks = True
    agg.record_epoch({"loss": 2.0}, replace_index=0, time_elapsed=None)

    assert np.isnan(agg.epoch_times[0])


def test_plot_with_custom_styles(tmp_path):
    agg1 = make_agg("a", [1, 2])
    agg2 = make_agg("b", [2, 3])

    MetricsAggregator.plot(
        [agg1, agg2],
        color_styles_list=[("red", "solid"), ("blue", "dashed")],
        plot_dir=tmp_path,
    )

    assert len(list(tmp_path.glob("*.png"))) > 0


def test_plot_random_style_branch(tmp_path):
    agg1 = make_agg("foo", [1, 2])
    agg2 = make_agg("bar", [2, 3])

    MetricsAggregator.plot([agg1, agg2], plot_dir=tmp_path)

    assert len(list(tmp_path.glob("*.png"))) > 0


def test_plot_missing_loss_key(tmp_path):
    agg1 = make_agg("train", [1, 2])
    agg2 = make_agg("val", [2, 3])

    agg2.epoch_loss_terms = {"other": [3, 4]}

    MetricsAggregator.plot([agg1, agg2], plot_dir=tmp_path)

    assert len(list(tmp_path.glob("*.png"))) > 0


def test_plot_empty_epoch_times(tmp_path):
    agg = make_agg("train", [1, 2])
    agg.epoch_times = []

    MetricsAggregator.plot([agg], plot_dir=tmp_path)

    assert len(list(tmp_path.glob("*.png"))) > 0


def test_dist_compute_multiple_metrics():
    agg = MetricsAggregator(DummyDistributed(), "train")
    agg.record({"a": 2.0, "b": 4.0})


def test_load_state_dict_missing_keys():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.load_state_dict({})  # empty state

    assert agg.name is None

    assert agg.num_epochs_seen == 0
    assert agg.epoch_loss_terms is None
    assert agg.epoch_times is None


def test_dist_compute_multiple_metrics_assert():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"b": 4.0, "a": 2.0})  # unordered input

    logs = agg._dist_compute()

    assert list(sorted(logs.keys())) == ["a", "b"]
    assert logs["a"] == 2.0
    assert logs["b"] == 4.0


def test_record_mixed_valid_invalid():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": 2.0, "bad": "x", "none": None})

    assert agg.loss_terms["loss"] == 2.0
    assert "bad" not in agg.loss_terms


def test_record_epoch_append_nan_time():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": 1.0})
    logs = agg._dist_compute()

    agg.record_epoch(logs, time_elapsed=None)

    assert np.isnan(agg.epoch_times[-1])


def test_plot_inconsistent_loss_lengths(tmp_path):
    agg1 = make_agg("train", [1, 2])
    agg2 = make_agg("val", [1, 2])

    # manually break internal structure
    agg2.epoch_loss_terms = {"loss": [1]}  # shorter

    with pytest.raises(ValueError):
        MetricsAggregator.plot([agg1, agg2], plot_dir=tmp_path)


def test_plot_uses_env_dir(monkeypatch, tmp_path):
    agg = make_agg("train", [1, 2])

    monkeypatch.setenv("GLOBAL_FIGURES_DIR", str(tmp_path))

    MetricsAggregator.plot([agg])

    assert len(list(tmp_path.glob("*.png"))) > 0


def test_plot_multiple_loss_types(tmp_path):
    agg = MetricsAggregator(DummyDistributed(), "train")
    agg.epoch_loss_terms = {"a": [1, 2], "b": [2, 3]}
    agg.epoch_times = [1, 1]
    agg.num_epochs_seen = 2

    MetricsAggregator.plot([agg], plot_dir=tmp_path)

    files = list(tmp_path.glob("*.png"))
    assert any("a" in f.name for f in files)
    assert any("b" in f.name for f in files)


def test_load_state_dict_resets_batches():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": 5.0})  # batches exist

    state = {
        "name": "loaded",
        "epoch_loss_terms": {"loss": [1]},
        "epoch_times": [1.0],
        "num_epochs_seen": 1,
    }

    agg.load_state_dict(state)

    assert agg.name == "loaded"

    assert agg.num_batches_seen == 1
    assert agg.loss_terms["loss"] == 5.0


def test_init_empty_epoch_loss_terms():
    agg = MetricsAggregator(
        DummyDistributed(),
        "train",
        epoch_loss_terms={"loss": []},
    )

    assert agg.num_epochs_seen == 0


def test_plot_train_val_style(tmp_path):
    agg1 = make_agg("train_loss", [1, 2])
    agg2 = make_agg("val_loss", [2, 3])

    MetricsAggregator.plot([agg1, agg2], plot_dir=tmp_path)

    assert len(list(tmp_path.glob("*.png"))) > 0


def test_plot_multiple_losses_same_aggregator(tmp_path):
    agg = MetricsAggregator(DummyDistributed(), "train")
    agg.epoch_loss_terms = {
        "loss": [1, 2],
        "val_loss": [2, 3],
    }
    agg.epoch_times = [1, 1]
    agg.num_epochs_seen = 2

    MetricsAggregator.plot([agg], plot_dir=tmp_path)

    files = list(tmp_path.glob("*.png"))
    assert len(files) >= 2


def test_plot_skip_empty_times(tmp_path):
    agg = make_agg("train", [1, 2])
    agg.epoch_times = []  # triggers skip branch

    MetricsAggregator.plot([agg], plot_dir=tmp_path)


def test_reset_clears_aggregation_flag():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": 1.0})
    agg._dist_compute()
    agg.record_epoch({"loss": 1.0})

    assert agg._aggregated_across_ranks is False


def test_record_int_values():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": 1})

    assert agg.loss_terms["loss"] == 1.0


def test_load_state_dict_with_reset_trigger():
    agg = MetricsAggregator(DummyDistributed(), "train")

    agg.record({"loss": 5.0})

    # simulate epoch submission
    agg.epochs_submitted = True

    state = {
        "name": "loaded",
        "epoch_loss_terms": {"loss": [1]},
        "epoch_times": [1.0],
        "num_epochs_seen": 1,
    }

    agg.load_state_dict(state)

    assert agg.num_batches_seen == 0
    assert agg.loss_terms == {}

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from cccma_ppp.generic.aggregator import (
    MetricsAggregator,
    RunningCovariance,
)
from cccma_ppp.generic.runtime import RuntimeContext


class DummyDistributed:
    def __init__(self):
        self.device = "cpu"
        self.calls = []

    def all_reduce_sum(self, tensor):
        self.calls.append(tensor.clone())
        return tensor


class ScalingDistributed:
    def __init__(self, scale=2):
        self.device = "cpu"
        self.scale = scale
        self.calls = []

    def all_reduce_sum(self, tensor):
        self.calls.append(tensor.clone())
        tensor.mul_(self.scale)
        return tensor


def make_agg(
    name="train",
    metrics=None,
    epoch_times=None,
):
    if metrics is None:
        metrics = {
            "loss": [1.0, 2.0],
        }

    if epoch_times is None:
        number_of_epochs = len(next(iter(metrics.values())))
        epoch_times = [1.0] * number_of_epochs

    return MetricsAggregator(
        distributed=DummyDistributed(),
        name=name,
        epoch_metric_terms=metrics,
        epoch_times=epoch_times,
    )


@pytest.mark.pruned
def test_init_defaults():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    assert aggregator.name == "train"
    assert aggregator.epoch_metric_terms == {}
    assert aggregator.epoch_times == []
    assert aggregator.num_epochs_seen == 0
    assert aggregator.num_batches_seen == 0
    assert aggregator.loss_terms == {}
    assert aggregator.kwargs_terms == {}
    assert aggregator.lr_values == 0.0
    assert aggregator.epochs_submitted is False
    assert aggregator._aggregated_across_ranks is False


@pytest.mark.pruned
def test_init_with_existing_history_sets_epoch_count():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
        epoch_metric_terms={
            "loss": [1.0, 2.0],
            "rmse": [2.0, 3.0],
        },
        epoch_times=[4.0, 5.0],
    )

    assert aggregator.num_epochs_seen == 2
    assert aggregator.epoch_metric_terms["loss"] == [
        1.0,
        2.0,
    ]


@pytest.mark.pruned
def test_init_preserves_explicit_epoch_count():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
        epoch_metric_terms={
            "loss": [1.0, 2.0],
        },
        epoch_times=[1.0, 2.0],
        num_epochs_seen=7,
    )

    assert aggregator.num_epochs_seen == 7


@pytest.mark.pruned
def test_init_rejects_inconsistent_metric_lengths():
    with pytest.raises(AssertionError):
        MetricsAggregator(
            DummyDistributed(),
            "train",
            epoch_metric_terms={
                "loss": [1.0],
                "rmse": [2.0, 3.0],
            },
        )


@pytest.mark.pruned
def test_init_rejects_inconsistent_epoch_times():
    with pytest.raises(AssertionError):
        MetricsAggregator(
            DummyDistributed(),
            "train",
            epoch_metric_terms={
                "loss": [1.0, 2.0],
            },
            epoch_times=[1.0],
        )


@pytest.mark.pruned
def test_init_empty_metric_history():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
        epoch_metric_terms={
            "loss": [],
        },
    )

    assert aggregator.num_epochs_seen == 0
    assert aggregator.epoch_times == []


@pytest.mark.pruned
def test_record_numeric_metrics():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record(
        {
            "loss": 2.0,
            "count": 3,
        }
    )

    assert aggregator.loss_terms["loss"] == 2.0
    assert aggregator.loss_terms["count"] == 3.0
    assert aggregator.num_batches_seen == 1


def test_record_tensor_metrics_uses_mean():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record(
        {
            "loss": torch.tensor(
                [
                    [1.0, 3.0],
                    [5.0, 7.0],
                ]
            ),
        }
    )

    assert aggregator.loss_terms["loss"] == 4.0


def test_record_ignores_none_and_invalid_metrics():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record(
        {
            "missing": None,
            "invalid": "value",
            "valid": 2.0,
        }
    )

    assert aggregator.loss_terms == {
        "valid": 2.0,
    }
    assert aggregator.num_batches_seen == 1


@pytest.mark.pruned
def test_record_empty_dictionary_increments_batches():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record({})

    assert aggregator.num_batches_seen == 1
    assert aggregator.loss_terms == {}


@pytest.mark.pruned
def test_record_accumulates_multiple_batches():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record({"loss": 2.0})
    aggregator.record({"loss": 4.0})

    assert aggregator.loss_terms["loss"] == 6.0
    assert aggregator.num_batches_seen == 2

    logs = aggregator._dist_compute()

    assert logs["loss"] == 3.0


@pytest.mark.pruned
def test_record_kwargs_numeric_and_tensor():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record(
        {"loss": 1.0},
        kwargs={
            "beta": torch.tensor([0.2, 0.4]),
            "weight": 2,
        },
    )

    assert aggregator.kwargs_terms["beta"] == pytest.approx(0.3)
    assert aggregator.kwargs_terms["weight"] == 2.0


def test_record_kwargs_ignores_invalid_values():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record(
        {"loss": 1.0},
        kwargs={
            "missing": None,
            "invalid": "value",
            "valid": 3.0,
        },
    )

    assert aggregator.kwargs_terms == {
        "valid": 3.0,
    }


@pytest.mark.pruned
@pytest.mark.parametrize(
    ("learning_rate", "expected"),
    [
        (0.1, 0.1),
        (2, 2.0),
        (torch.tensor(0.25), 0.25),
        (torch.tensor([0.1, 0.3]), 0.2),
    ],
)
def test_record_learning_rate(
    learning_rate,
    expected,
):
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record(
        {"loss": 1.0},
        lr=learning_rate,
    )

    assert aggregator.lr_values == pytest.approx(expected)


@pytest.mark.parametrize(
    "learning_rate",
    [
        None,
        "invalid",
        object(),
    ],
)
def test_record_ignores_invalid_learning_rate(
    learning_rate,
):
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record(
        {"loss": 1.0},
        lr=learning_rate,
    )

    assert aggregator.lr_values == 0.0


def test_dist_compute_combines_all_metric_groups():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record(
        {
            "z_loss": 4.0,
            "a_loss": 2.0,
        },
        lr=0.1,
        kwargs={
            "beta": 0.5,
        },
    )

    logs = aggregator._dist_compute()

    assert list(logs) == [
        "a_loss",
        "z_loss",
        "beta",
        "lr",
    ]
    assert logs["a_loss"] == 2.0
    assert logs["z_loss"] == 4.0
    assert logs["beta"] == 0.5
    assert logs["lr"] == 0.1
    assert aggregator._aggregated_across_ranks is True


@pytest.mark.pruned
def test_dist_compute_zero_batches_returns_nan():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    logs = aggregator._dist_compute()

    assert set(logs) == {"lr"}
    assert np.isnan(logs["lr"])


@pytest.mark.pruned
def test_dist_average_calls_distributed_reduce():
    distributed = DummyDistributed()

    aggregator = MetricsAggregator(
        distributed,
        "train",
    )
    aggregator.record({"loss": 4.0})

    assert aggregator._dist_average(4.0) == 4.0
    assert len(distributed.calls) == 1

    np.testing.assert_array_equal(
        distributed.calls[0].numpy(),
        [4.0, 1.0],
    )


@pytest.mark.pruned
def test_dist_average_distributed_scaling_preserves_average():
    distributed = ScalingDistributed(scale=4)

    aggregator = MetricsAggregator(
        distributed,
        "train",
    )
    aggregator.record({"loss": 3.0})

    logs = aggregator._dist_compute()

    assert logs["loss"] == 3.0
    assert logs["lr"] == 0.0


@pytest.mark.pruned
def test_dist_compute_twice_without_new_records():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )
    aggregator.record({"loss": 4.0})

    first = aggregator._dist_compute()
    second = aggregator._dist_compute()

    assert first == second
    assert first["loss"] == 4.0


@pytest.mark.pruned
def test_record_epoch_requires_distributed_compute():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    with pytest.raises(
        RuntimeError,
        match="_dist_compute",
    ):
        aggregator.record_epoch({"loss": 1.0})


@pytest.mark.pruned
def test_record_epoch_appends_metrics():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record(
        {"loss": 2.0},
        lr=0.1,
        kwargs={"beta": 0.5},
    )
    logs = aggregator._dist_compute()

    result = aggregator.record_epoch(
        logs,
        time_elapsed=3.5,
    )

    assert result == logs
    assert aggregator.epoch_metric_terms == {
        "loss": [2.0],
        "beta": [0.5],
        "lr": [0.1],
    }
    assert aggregator.epoch_times == [3.5]
    assert aggregator.num_epochs_seen == 1
    assert aggregator.epochs_submitted is True


@pytest.mark.pruned
def test_record_epoch_none_time_becomes_nan():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record({"loss": 1.0})
    logs = aggregator._dist_compute()
    aggregator.record_epoch(
        logs,
        time_elapsed=None,
    )

    assert np.isnan(aggregator.epoch_times[0])


@pytest.mark.pruned
def test_record_epoch_resets_batch_state():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record(
        {"loss": 2.0},
        lr=0.2,
        kwargs={"beta": 0.5},
    )
    logs = aggregator._dist_compute()
    aggregator.record_epoch(logs)

    assert aggregator.loss_terms == {}
    assert aggregator.kwargs_terms == {}
    assert aggregator.lr_values == 0.0
    assert aggregator.num_batches_seen == 0
    assert aggregator._aggregated_across_ranks is False


@pytest.mark.pruned
def test_record_epoch_replaces_existing_metrics():
    aggregator = make_agg(
        metrics={
            "loss": [1.0],
            "lr": [0.1],
        },
        epoch_times=[2.0],
    )
    aggregator._aggregated_across_ranks = True

    aggregator.record_epoch(
        {
            "loss": 5.0,
            "lr": 0.2,
        },
        replace_index=0,
        time_elapsed=9.0,
    )

    assert aggregator.epoch_metric_terms["loss"] == [5.0]
    assert aggregator.epoch_metric_terms["lr"] == [0.2]
    assert aggregator.epoch_times == [9.0]


def test_record_epoch_replacement_none_time_becomes_nan():
    aggregator = make_agg(
        metrics={
            "loss": [1.0],
        },
        epoch_times=[2.0],
    )
    aggregator._aggregated_across_ranks = True

    aggregator.record_epoch(
        {"loss": 3.0},
        replace_index=0,
    )

    assert np.isnan(aggregator.epoch_times[0])


def test_record_epoch_rejects_new_metric_during_replace():
    aggregator = make_agg(
        metrics={
            "loss": [1.0],
        },
        epoch_times=[2.0],
    )
    aggregator._aggregated_across_ranks = True

    with pytest.raises(
        ValueError,
        match="not previously recorded",
    ):
        aggregator.record_epoch(
            {"new_metric": 3.0},
            replace_index=0,
        )


@pytest.mark.pruned
def test_record_epoch_replace_index_out_of_range():
    aggregator = make_agg(
        metrics={
            "loss": [1.0],
        },
        epoch_times=[2.0],
    )
    aggregator._aggregated_across_ranks = True

    with pytest.raises(IndexError):
        aggregator.record_epoch(
            {"loss": 3.0},
            replace_index=99,
        )


@pytest.mark.pruned
def test_reset_before_epoch_warns_and_preserves_batches():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )
    aggregator.record({"loss": 2.0})

    with pytest.warns(
        UserWarning,
        match="before submitting",
    ):
        aggregator.reset_batch_losses()

    assert aggregator.num_batches_seen == 1
    assert aggregator.loss_terms["loss"] == 2.0


@pytest.mark.pruned
def test_reset_after_epoch_clears_state():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record({"loss": 2.0})
    logs = aggregator._dist_compute()
    aggregator.record_epoch(logs)

    aggregator.record({"loss": 4.0})
    aggregator._aggregated_across_ranks = True
    aggregator.reset_batch_losses()

    assert aggregator.loss_terms == {}
    assert aggregator.kwargs_terms == {}
    assert aggregator.lr_values == 0.0
    assert aggregator.num_batches_seen == 0
    assert aggregator._aggregated_across_ranks is False


@pytest.mark.pruned
def test_state_dict_empty():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    assert aggregator.state_dict() == {
        "name": "train",
        "epoch_metric_terms": {},
        "epoch_times": [],
        "num_epochs_seen": 0,
    }


def test_state_dict_roundtrip():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    aggregator.record({"loss": 2.0})
    logs = aggregator._dist_compute()
    aggregator.record_epoch(
        logs,
        time_elapsed=1.5,
    )

    state = aggregator.state_dict()

    loaded = MetricsAggregator(
        DummyDistributed(),
        "other",
    )
    loaded.load_state_dict(state)

    assert loaded.name == "train"
    assert loaded.epoch_metric_terms == aggregator.epoch_metric_terms
    assert loaded.epoch_times == aggregator.epoch_times
    assert loaded.num_epochs_seen == 1


@pytest.mark.pruned
def test_load_state_dict_missing_keys():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    with pytest.warns(UserWarning):
        aggregator.load_state_dict({})

    assert aggregator.name is None
    assert aggregator.epoch_metric_terms is None
    assert aggregator.epoch_times is None
    assert aggregator.num_epochs_seen == 0


@pytest.mark.pruned
def test_load_state_dict_explicit_none_values():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    with pytest.warns(UserWarning):
        aggregator.load_state_dict(
            {
                "name": None,
                "epoch_metric_terms": None,
                "epoch_times": None,
                "num_epochs_seen": 0,
            }
        )

    assert aggregator.name is None
    assert aggregator.epoch_metric_terms is None
    assert aggregator.epoch_times is None


@pytest.mark.pruned
def test_load_state_preserves_batch_state_before_submission():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )
    aggregator.record({"loss": 3.0})

    with pytest.warns(UserWarning):
        aggregator.load_state_dict(
            {
                "name": "loaded",
                "epoch_metric_terms": {
                    "loss": [1.0],
                },
                "epoch_times": [1.0],
                "num_epochs_seen": 1,
            }
        )

    assert aggregator.num_batches_seen == 1
    assert aggregator.loss_terms["loss"] == 3.0


@pytest.mark.pruned
def test_load_state_resets_batches_after_submission():
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )
    aggregator.record({"loss": 3.0})
    aggregator.epochs_submitted = True

    aggregator.load_state_dict(
        {
            "name": "loaded",
            "epoch_metric_terms": {
                "loss": [1.0],
            },
            "epoch_times": [1.0],
            "num_epochs_seen": 1,
        }
    )

    assert aggregator.num_batches_seen == 1
    assert aggregator.loss_terms == {"loss": 3.0}


def test_plot_requires_aggregator(tmp_path):
    with pytest.raises(
        ValueError,
        match="at least one aggregator",
    ):
        MetricsAggregator.plot(
            [],
            plot_dir=tmp_path,
        )


def test_plot_rejects_inconsistent_epoch_counts(
    tmp_path,
):
    first = make_agg(
        "train",
        metrics={"loss": [1.0, 2.0]},
    )
    second = make_agg(
        "validation",
        metrics={"loss": [3.0]},
    )

    with pytest.raises(
        ValueError,
        match="same number of epochs",
    ):
        MetricsAggregator.plot(
            [first, second],
            plot_dir=tmp_path,
        )


def test_plot_rejects_zero_epochs(tmp_path):
    aggregator = MetricsAggregator(
        DummyDistributed(),
        "train",
    )

    with pytest.raises(
        ValueError,
        match="No epochs",
    ):
        MetricsAggregator.plot(
            [aggregator],
            plot_dir=tmp_path,
        )


@pytest.mark.pruned
def test_plot_creates_loss_and_time_files(
    tmp_path,
):
    train = make_agg(
        "train",
        metrics={
            "loss": [1.0, 2.0],
            "rmse": [2.0, 3.0],
        },
    )
    validation = make_agg(
        "validation",
        metrics={
            "loss": [2.0, 3.0],
            "rmse": [3.0, 4.0],
        },
    )

    MetricsAggregator.plot(
        [train, validation],
        plot_dir=tmp_path,
    )

    assert (tmp_path / "epoch_2_loss.png").exists()
    assert (tmp_path / "epoch_2_rmse.png").exists()
    assert (tmp_path / "epoch_2_times.png").exists()


def test_plot_uses_custom_styles(tmp_path):
    first = make_agg("first")
    second = make_agg("second")

    MetricsAggregator.plot(
        [first, second],
        color_styles_list=[
            ("red", "solid"),
            ("blue", "dashed"),
        ],
        plot_dir=tmp_path,
    )

    assert list(tmp_path.glob("*.png"))


def test_plot_random_style_branch(tmp_path):
    first = make_agg("alpha")
    second = make_agg("beta")

    MetricsAggregator.plot(
        [first, second],
        plot_dir=tmp_path,
    )

    assert list(tmp_path.glob("*.png"))


def test_plot_skips_none_aggregator(tmp_path):
    aggregator = make_agg("train")

    MetricsAggregator.plot(
        [aggregator, None],
        plot_dir=tmp_path,
    )

    assert list(tmp_path.glob("*.png"))


def test_plot_skips_missing_metric(tmp_path):
    first = make_agg(
        "train",
        metrics={
            "loss": [1.0, 2.0],
        },
    )
    second = make_agg(
        "validation",
        metrics={
            "rmse": [2.0, 3.0],
        },
    )

    MetricsAggregator.plot(
        [first, second],
        plot_dir=tmp_path,
    )

    assert (tmp_path / "epoch_2_loss.png").exists()
    assert (tmp_path / "epoch_2_rmse.png").exists()


def test_plot_skips_empty_epoch_times(tmp_path):
    aggregator = make_agg("train")
    aggregator.epoch_times = []

    MetricsAggregator.plot(
        [aggregator],
        plot_dir=tmp_path,
    )

    assert (tmp_path / "epoch_2_loss.png").exists()
    assert (tmp_path / "epoch_2_times.png").exists()


@pytest.mark.pruned
def test_plot_sanitizes_metric_name(tmp_path):
    aggregator = make_agg(
        "train",
        metrics={
            "loss/train": [1.0, 2.0],
        },
    )

    MetricsAggregator.plot(
        [aggregator],
        plot_dir=tmp_path,
    )

    assert (tmp_path / "epoch_2_loss_train.png").exists()


def test_plot_removes_old_loss_plot(tmp_path):
    old_plot = tmp_path / "epoch_1_loss.png"
    old_plot.write_text("old")

    MetricsAggregator.plot(
        [make_agg("train")],
        plot_dir=tmp_path,
    )

    assert old_plot.exists() is False


def test_plot_removes_old_time_plot(tmp_path):
    old_plot = tmp_path / "epoch_1_times.png"
    old_plot.write_text("old")

    MetricsAggregator.plot(
        [make_agg("train")],
        plot_dir=tmp_path,
    )

    assert old_plot.exists() is False


@pytest.mark.pruned
def test_plot_accepts_string_directory(tmp_path):
    MetricsAggregator.plot(
        [make_agg("train")],
        plot_dir=str(tmp_path),
    )

    assert list(tmp_path.glob("*.png"))


@pytest.mark.pruned
def test_plot_uses_runtime_context(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        RuntimeContext,
        "GLOBAL_FIGURES_DIR",
        str(tmp_path),
    )

    MetricsAggregator.plot(
        [make_agg("train")],
    )

    assert list(tmp_path.glob("*.png"))


@pytest.mark.pruned
def test_plot_closes_figures(tmp_path):
    MetricsAggregator.plot(
        [make_agg("train")],
        plot_dir=tmp_path,
    )

    assert plt.get_fignums() == []


@pytest.mark.pruned
def test_running_covariance_first_update():
    covariance = RunningCovariance(DummyDistributed())

    covariance.update(
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
    )

    assert covariance.count.item() == 2
    np.testing.assert_array_equal(
        covariance.sum_x.numpy(),
        [4.0, 6.0],
    )
    np.testing.assert_array_equal(
        covariance.sum_xxT.numpy(),
        [
            [10.0, 14.0],
            [14.0, 20.0],
        ],
    )


def test_running_covariance_multiple_updates():
    covariance = RunningCovariance(DummyDistributed())

    covariance.update(torch.tensor([[1.0, 2.0]]))
    covariance.update(torch.tensor([[3.0, 4.0]]))

    assert covariance.count.item() == 2
    np.testing.assert_array_equal(
        covariance.sum_x.numpy(),
        [4.0, 6.0],
    )


@pytest.mark.pruned
def test_running_covariance_detaches_input():
    covariance = RunningCovariance(DummyDistributed())

    value = torch.tensor(
        [[1.0, 2.0]],
        requires_grad=True,
    )

    covariance.update(value)

    assert covariance.sum_x.requires_grad is False
    assert covariance.sum_xxT.requires_grad is False


def test_running_covariance_finalize():
    covariance = RunningCovariance(DummyDistributed())

    covariance.update(
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
    )

    mean, result = covariance.finalize()

    np.testing.assert_allclose(
        mean.numpy(),
        [2.0, 3.0],
    )
    np.testing.assert_allclose(
        result.numpy(),
        [
            [2.0, 2.0],
            [2.0, 2.0],
        ],
    )


def test_running_covariance_finalize_rejects_one_sample():
    covariance = RunningCovariance(DummyDistributed())

    covariance.update(torch.tensor([[1.0, 2.0]]))

    with pytest.raises(
        ValueError,
        match="at least two samples",
    ):
        covariance.finalize()


@pytest.mark.pruned
def test_running_covariance_distributed_reduce():
    distributed = DummyDistributed()
    covariance = RunningCovariance(distributed)

    covariance.update(
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
    )
    covariance.distributed_reduce()

    assert len(distributed.calls) == 3

import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from cccma_ppp.generic.monitoring import Monitor, monitor
import threading

import cccma_ppp.generic.monitoring as module

from cccma_ppp.generic.monitoring import (
    build_distributed_monitor,
    combine_rank_results,
    distributed_monitoring,
    export_results,
    resolve_physical_gpu_index,
)


@pytest.fixture
def fake_pynvml(monkeypatch):
    pynvml = Mock()

    pynvml.nvmlInit = Mock()
    pynvml.nvmlDeviceGetCount = Mock(return_value=2)
    pynvml.nvmlDeviceGetHandleByIndex = Mock(
        side_effect=lambda index: "gpu-handle-" + str(index)
    )

    def get_utilization(handle):
        index = int(handle.rsplit("-", 1)[-1])

        if index == 0:
            return SimpleNamespace(gpu=75)

        return SimpleNamespace(gpu=25)

    def get_memory(handle):
        index = int(handle.rsplit("-", 1)[-1])

        if index == 0:
            return SimpleNamespace(
                used=4_294_967_296,
                total=8_589_934_592,
            )

        return SimpleNamespace(
            used=3_221_225_472,
            total=12_884_901_888,
        )

    pynvml.nvmlDeviceGetUtilizationRates = Mock(side_effect=get_utilization)
    pynvml.nvmlDeviceGetMemoryInfo = Mock(side_effect=get_memory)

    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        pynvml,
    )

    return pynvml


@pytest.fixture
def basic_monitor():
    instance = Monitor(
        cpu=True,
        ram=True,
        interval=0.01,
    )

    yield instance

    instance.stop(timeout=1.0)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        [
            {
                "t": 100.0,
                "stage": "main.work",
                "event": "start",
                "span_id": "span-1",
                "cpu": None,
                "ram": None,
                "gpu0": None,
                "vram0": None,
            },
            {
                "t": 100.1,
                "stage": "main.work",
                "event": "sample",
                "span_id": None,
                "cpu": 10.0,
                "ram": 1.0,
                "gpu0": 20.0,
                "vram0": 30.0,
            },
            {
                "t": 100.2,
                "stage": "main.work.ready",
                "event": "checkpoint",
                "span_id": None,
                "cpu": None,
                "ram": None,
                "gpu0": None,
                "vram0": None,
            },
            {
                "t": 100.3,
                "stage": "main.work",
                "event": "sample",
                "span_id": None,
                "cpu": 25.0,
                "ram": 1.5,
                "gpu0": 50.0,
                "vram0": 35.0,
            },
            {
                "t": 100.4,
                "stage": "main.work",
                "event": "sample",
                "span_id": None,
                "cpu": 15.0,
                "ram": 1.25,
                "gpu0": 40.0,
                "vram0": 40.0,
            },
            {
                "t": 100.5,
                "stage": "main.work",
                "event": "end",
                "span_id": "span-1",
                "cpu": None,
                "ram": None,
                "gpu0": None,
                "vram0": None,
            },
        ]
    )


def wait_until(condition, timeout=1.0, interval=0.01):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if condition():
            return True

        time.sleep(interval)

    return bool(condition())


@pytest.mark.pruned
def test_default_configuration():
    instance = Monitor()

    assert instance.cpu_enabled is True
    assert instance.ram_enabled is True
    assert instance.gpu_indices == ()
    assert instance.interval == pytest.approx(0.1)
    assert instance.metric_names == ("cpu", "ram")
    assert instance.running is False


@pytest.mark.pruned
def test_monitor_factory():
    instance = monitor(
        cpu=True,
        ram=False,
        interval=0.25,
    )

    assert isinstance(instance, Monitor)
    assert instance.cpu_enabled is True
    assert instance.ram_enabled is False
    assert instance.interval == pytest.approx(0.25)


@pytest.mark.pruned
def test_gpu_zero_configuration(fake_pynvml):
    instance = Monitor(
        cpu=False,
        ram=False,
        gpu0=True,
    )

    assert instance.gpu_indices == (0,)
    assert instance.metric_names == (
        "gpu0",
        "vram0",
    )


def test_gpu_one_configuration(fake_pynvml):
    instance = Monitor(
        cpu=False,
        ram=False,
        gpu1=True,
    )

    assert instance.gpu_indices == (1,)
    assert instance.metric_names == (
        "gpu1",
        "vram1",
    )


@pytest.mark.pruned
def test_multiple_gpu_configuration(fake_pynvml):
    instance = Monitor(
        cpu=False,
        ram=False,
        gpus=[1, 0, 1],
    )

    assert instance.gpu_indices == (0, 1)
    assert instance.metric_names == (
        "gpu0",
        "vram0",
        "gpu1",
        "vram1",
    )


@pytest.mark.pruned
def test_combined_gpu_configuration(fake_pynvml):
    instance = Monitor(
        cpu=False,
        ram=False,
        gpu0=True,
        gpus=[1],
    )

    assert instance.gpu_indices == (0, 1)


@pytest.mark.parametrize(
    "interval",
    [0.0, -0.1, -10.0],
)
def test_invalid_interval(interval):
    with pytest.raises(
        ValueError,
        match="interval must be greater than zero",
    ):
        Monitor(interval=interval)


@pytest.mark.parametrize(
    "indices",
    [
        [-1],
        [0, -1],
    ],
)
def test_negative_gpu_index(indices):
    with pytest.raises(
        ValueError,
        match="GPU indices must be non-negative",
    ):
        Monitor(gpus=indices)


@pytest.mark.pruned
@pytest.mark.parametrize(
    "indices",
    [
        [0.0],
        ["0"],
        [None],
        [True],
    ],
)
def test_invalid_gpu_index_type(indices):
    with pytest.raises(
        TypeError,
        match="GPU indices must be integers",
    ):
        Monitor(gpus=indices)


@pytest.mark.pruned
def test_ram_collection(basic_monitor):
    value = basic_monitor._ram_gb()

    assert isinstance(value, float)
    assert value >= 0.0


@pytest.mark.pruned
def test_cpu_collection(basic_monitor):
    basic_monitor._process.cpu_percent(interval=None)

    value = basic_monitor._cpu()

    assert isinstance(value, float)
    assert value >= 0.0


@pytest.mark.pruned
def test_cpu_and_ram_sample(basic_monitor, monkeypatch):
    monkeypatch.setattr(
        basic_monitor,
        "_cpu",
        lambda: 42.5,
    )
    monkeypatch.setattr(
        basic_monitor,
        "_ram_gb",
        lambda: 3.25,
    )

    sample = basic_monitor._collect_sample()

    assert sample["event"] == "sample"
    assert sample["stage"] == "root"
    assert sample["cpu"] == pytest.approx(42.5)
    assert sample["ram"] == pytest.approx(3.25)
    assert "gpu0" not in sample
    assert "vram0" not in sample


@pytest.mark.pruned
def test_disabled_metrics_are_not_collected():
    instance = Monitor(
        cpu=False,
        ram=False,
    )

    sample = instance._collect_sample()

    assert sample["event"] == "sample"
    assert "cpu" not in sample
    assert "ram" not in sample


@pytest.mark.pruned
def test_gpu_handles_are_initialized(fake_pynvml):
    instance = Monitor(
        cpu=False,
        ram=False,
        gpus=[0, 1],
    )

    fake_pynvml.nvmlInit.assert_called_once()

    assert instance._gpu_handles == {
        0: "gpu-handle-0",
        1: "gpu-handle-1",
    }


@pytest.mark.pruned
def test_gpu_zero_collection(fake_pynvml):
    instance = Monitor(
        cpu=False,
        ram=False,
        gpu0=True,
    )

    gpu, vram = instance._gpu(0)

    assert gpu == pytest.approx(75.0)
    assert vram == pytest.approx(50.0)


@pytest.mark.pruned
def test_gpu_one_collection(fake_pynvml):
    instance = Monitor(
        cpu=False,
        ram=False,
        gpu1=True,
    )

    gpu, vram = instance._gpu(1)

    assert gpu == pytest.approx(25.0)
    assert vram == pytest.approx(25.0)


def test_multiple_gpu_sample(fake_pynvml):
    instance = Monitor(
        cpu=False,
        ram=False,
        gpus=[0, 1],
    )

    sample = instance._collect_sample()

    assert sample["gpu0"] == pytest.approx(75.0)
    assert sample["vram0"] == pytest.approx(50.0)
    assert sample["gpu1"] == pytest.approx(25.0)
    assert sample["vram1"] == pytest.approx(25.0)


@pytest.mark.pruned
def test_unavailable_gpu_returns_nan(fake_pynvml):
    instance = Monitor(
        cpu=False,
        ram=False,
        gpus=[5],
    )

    gpu, vram = instance._gpu(5)

    assert np.isnan(gpu)
    assert np.isnan(vram)
    assert 5 in instance._unavailable_gpus


@pytest.mark.pruned
def test_failed_gpu_initialization(monkeypatch):
    pynvml = Mock()
    pynvml.nvmlInit.side_effect = RuntimeError("NVML unavailable")

    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        pynvml,
    )

    instance = Monitor(
        cpu=True,
        ram=True,
        gpu0=True,
    )

    gpu, vram = instance._gpu(0)

    assert np.isnan(gpu)
    assert np.isnan(vram)
    assert instance.cpu_enabled is True
    assert instance.ram_enabled is True


@pytest.mark.pruned
def test_gpu_query_failure(fake_pynvml):
    fake_pynvml.nvmlDeviceGetUtilizationRates.side_effect = RuntimeError(
        "GPU query failed"
    )

    instance = Monitor(
        cpu=False,
        ram=False,
        gpu0=True,
    )

    gpu, vram = instance._gpu(0)

    assert np.isnan(gpu)
    assert np.isnan(vram)
    assert 0 in instance._unavailable_gpus


@pytest.mark.pruned
def test_initial_stage_is_root(basic_monitor):
    assert basic_monitor.current_stage() == "root"


@pytest.mark.pruned
def test_span_records_start_and_end(basic_monitor):
    with basic_monitor.span("work"):
        assert basic_monitor.current_stage() == "main.work"

    assert basic_monitor.current_stage() == "root"

    dataframe = basic_monitor.get_dataframe()

    assert list(dataframe["event"]) == [
        "start",
        "end",
    ]
    assert list(dataframe["stage"]) == [
        "main.work",
        "main.work",
    ]

    start_span_id = dataframe.iloc[0]["span_id"]
    end_span_id = dataframe.iloc[1]["span_id"]

    assert isinstance(start_span_id, str)
    assert start_span_id
    assert end_span_id == start_span_id
    assert dataframe.iloc[1]["t"] >= dataframe.iloc[0]["t"]


@pytest.mark.pruned
def test_nested_spans(basic_monitor):
    with basic_monitor.span("outer"):
        assert basic_monitor.current_stage() == "main.outer"

        with basic_monitor.span("inner"):
            assert basic_monitor.current_stage() == "main.outer.inner"

        assert basic_monitor.current_stage() == "main.outer"

    assert basic_monitor.current_stage() == "root"

    dataframe = basic_monitor.get_dataframe()

    assert list(dataframe["stage"]) == [
        "main.outer",
        "main.outer.inner",
        "main.outer.inner",
        "main.outer",
    ]
    assert list(dataframe["event"]) == [
        "start",
        "start",
        "end",
        "end",
    ]


@pytest.mark.pruned
def test_span_ends_after_exception(basic_monitor):
    with pytest.raises(RuntimeError):
        with basic_monitor.span("failure"):
            raise RuntimeError("test failure")

    dataframe = basic_monitor.get_dataframe()

    assert list(dataframe["event"]) == [
        "start",
        "end",
    ]
    assert basic_monitor.current_stage() == "root"


def test_empty_span_name(basic_monitor):
    with pytest.raises(
        ValueError,
        match="span name cannot be empty",
    ):
        with basic_monitor.span("   "):
            pass


@pytest.mark.pruned
def test_invalid_span_name_type(basic_monitor):
    with pytest.raises(
        TypeError,
        match="span name must be a string",
    ):
        with basic_monitor.span(123):
            pass


@pytest.mark.pruned
def test_root_checkpoint(basic_monitor):
    basic_monitor.checkpoint("ready")

    dataframe = basic_monitor.get_dataframe()

    assert len(dataframe) == 1
    assert dataframe.iloc[0]["event"] == "checkpoint"
    assert dataframe.iloc[0]["stage"] == "main.ready"


@pytest.mark.pruned
def test_checkpoint_inside_span(basic_monitor):
    with basic_monitor.span("work"):
        basic_monitor.checkpoint("ready")

    dataframe = basic_monitor.get_dataframe()

    checkpoint = dataframe[dataframe["event"] == "checkpoint"].iloc[0]

    assert checkpoint["stage"] == "main.work.ready"


def test_empty_checkpoint_name(basic_monitor):
    with pytest.raises(
        ValueError,
        match="checkpoint name cannot be empty",
    ):
        basic_monitor.checkpoint("  ")


def test_invalid_checkpoint_name_type(basic_monitor):
    with pytest.raises(
        TypeError,
        match="checkpoint name must be a string",
    ):
        basic_monitor.checkpoint(123)


@pytest.mark.pruned
def test_observe_decorator(basic_monitor):
    @basic_monitor.observe
    def add(left, right):
        return left + right

    result = add(2, 3)

    assert result == 5
    assert add.__name__ == "add"

    dataframe = basic_monitor.get_dataframe()

    assert list(dataframe["event"]) == [
        "start",
        "end",
    ]
    assert list(dataframe["stage"]) == [
        "main.add",
        "main.add",
    ]


@pytest.mark.pruned
def test_observe_rejects_non_callable(basic_monitor):
    with pytest.raises(
        TypeError,
        match="observe expects a callable",
    ):
        basic_monitor.observe(123)


@pytest.mark.pruned
def test_start_and_stop_monitoring(basic_monitor):
    basic_monitor.start()

    assert basic_monitor.running is True

    collected = wait_until(lambda: len(basic_monitor.get_dataframe()) >= 2)

    basic_monitor.stop(timeout=1.0)

    assert collected is True
    assert basic_monitor.running is False

    dataframe = basic_monitor.get_dataframe()

    assert not dataframe.empty
    assert set(dataframe["event"]) == {"sample"}
    assert "cpu" in dataframe.columns
    assert "ram" in dataframe.columns


def test_start_twice_does_not_create_new_thread(
    basic_monitor,
):
    basic_monitor.start()
    first_thread = basic_monitor._thread

    basic_monitor.start()

    assert basic_monitor._thread is first_thread

    basic_monitor.stop(timeout=1.0)


@pytest.mark.pruned
def test_start_with_clear(basic_monitor):
    basic_monitor.checkpoint("old")

    assert len(basic_monitor.get_dataframe()) == 1

    basic_monitor.start(clear=True)

    collected = wait_until(lambda: len(basic_monitor.get_dataframe()) >= 1)

    basic_monitor.stop(timeout=1.0)

    assert collected is True

    dataframe = basic_monitor.get_dataframe()

    assert "checkpoint" not in set(dataframe["event"])


@pytest.mark.pruned
def test_stop_before_start(basic_monitor):
    basic_monitor.stop(timeout=1.0)

    assert basic_monitor.running is False


@pytest.mark.pruned
def test_negative_stop_timeout(basic_monitor):
    with pytest.raises(
        ValueError,
        match="timeout cannot be negative",
    ):
        basic_monitor.stop(timeout=-1.0)


@pytest.mark.pruned
def test_context_manager():
    instance = Monitor(
        cpu=True,
        ram=False,
        interval=0.01,
    )

    with instance:
        assert instance.running is True

        collected = wait_until(lambda: len(instance.get_dataframe()) >= 1)

        assert collected is True

    assert instance.running is False


@pytest.mark.pruned
def test_context_manager_does_not_suppress_exception():
    instance = Monitor(
        cpu=True,
        ram=False,
        interval=0.01,
    )

    with pytest.raises(RuntimeError):
        with instance:
            raise RuntimeError("failure")

    assert instance.running is False


@pytest.mark.pruned
def test_clear_data(basic_monitor):
    basic_monitor.checkpoint("ready")

    assert len(basic_monitor.get_dataframe()) == 1

    basic_monitor.clear()

    assert basic_monitor.get_dataframe().empty


@pytest.mark.pruned
def test_dataframe_returns_copy(basic_monitor):
    basic_monitor.checkpoint("ready")

    first = basic_monitor.get_dataframe()
    first.loc[0, "stage"] = "modified"

    second = basic_monitor.get_dataframe()

    assert second.loc[0, "stage"] == "main.ready"


@pytest.mark.pruned
def test_metric_names_with_all_resources(fake_pynvml):
    instance = Monitor(
        cpu=True,
        ram=True,
        gpus=[0, 1],
    )

    assert instance.metric_names == (
        "cpu",
        "ram",
        "gpu0",
        "vram0",
        "gpu1",
        "vram1",
    )


def test_empty_kalman_filter():
    result = Monitor._kalman_filter([])

    assert isinstance(result, np.ndarray)
    assert result.size == 0


@pytest.mark.pruned
def test_kalman_filter_preserves_length():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    result = Monitor._kalman_filter(values)

    assert len(result) == len(values)
    assert np.isfinite(result).all()


@pytest.mark.pruned
def test_kalman_filter_handles_nan():
    values = np.array([np.nan, 2.0, np.nan, 4.0])

    result = Monitor._kalman_filter(values)

    assert np.isnan(result[0])
    assert np.isfinite(result[1])
    assert np.isfinite(result[2])
    assert np.isfinite(result[3])


def test_kalman_filter_all_nan():
    values = np.array([np.nan, np.nan, np.nan])

    result = Monitor._kalman_filter(values)

    assert np.isnan(result).all()


def test_negative_process_variance():
    with pytest.raises(
        ValueError,
        match="process_variance cannot be negative",
    ):
        Monitor._kalman_filter(
            [1.0, 2.0],
            process_variance=-1.0,
        )


def test_negative_measurement_variance():
    with pytest.raises(
        ValueError,
        match="measurement_variance cannot be negative",
    ):
        Monitor._kalman_filter(
            [1.0, 2.0],
            measurement_variance=-1.0,
        )


def test_no_smoothing():
    series = pd.Series([1.0, 2.0, 3.0])

    result = Monitor._smooth(
        series,
        method=None,
    )

    assert result is series


@pytest.mark.pruned
def test_ema_smoothing():
    series = pd.Series([1.0, 2.0, 3.0])

    result = Monitor._smooth(
        series,
        method="ema",
        alpha=0.5,
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 3
    assert result.iloc[0] == pytest.approx(1.0)


def test_invalid_ema_alpha():
    series = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(
        ValueError,
        match="EMA alpha",
    ):
        Monitor._smooth(
            series,
            method="ema",
            alpha=0.0,
        )


def test_rolling_smoothing():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = Monitor._smooth(
        series,
        method="rolling",
        window=3,
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 4
    assert result.notna().all()


def test_invalid_rolling_window():
    series = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(
        ValueError,
        match="rolling window",
    ):
        Monitor._smooth(
            series,
            method="rolling",
            window=0,
        )


@pytest.mark.pruned
def test_kalman_smoothing():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = Monitor._smooth(
        series,
        method="kalman",
        process_variance=1.0,
        measurement_variance=25.0,
    )

    assert isinstance(result, np.ndarray)
    assert len(result) == 4


def test_unknown_smoothing_method():
    series = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(
        ValueError,
        match="Unknown smoothing method",
    ):
        Monitor._smooth(
            series,
            method="invalid",
        )


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("cpu", ("CPU", "%")),
        ("ram", ("RAM", "GB")),
        ("gpu0", ("GPU 0", "%")),
        ("gpu1", ("GPU 1", "%")),
        ("vram0", ("GPU 0 VRAM", "%")),
        ("vram1", ("GPU 1 VRAM", "%")),
        ("custom", ("custom", "")),
    ],
)
def test_metric_labels(metric, expected):
    assert Monitor._metric_label(metric) == expected


@pytest.mark.pruned
def test_available_metrics(sample_dataframe):
    instance = Monitor(
        cpu=True,
        ram=True,
    )

    instance.gpu_indices = (0,)

    result = instance._available_metrics(
        sample_dataframe[sample_dataframe["event"] == "sample"]
    )

    assert result == [
        "cpu",
        "ram",
        "gpu0",
        "vram0",
    ]


@pytest.mark.pruned
def test_span_intervals(sample_dataframe):
    sample_dataframe = sample_dataframe.copy()
    sample_dataframe["elapsed"] = sample_dataframe["t"] - sample_dataframe["t"].min()

    intervals = Monitor._span_intervals(sample_dataframe)

    assert len(intervals) == 1

    start, end, stage = intervals[0]

    assert start == pytest.approx(0.0)
    assert end == pytest.approx(0.5)
    assert stage == "main.work"


@pytest.mark.pruned
def test_plot_with_ema_smoothing(
    sample_dataframe,
):
    instance = Monitor(
        cpu=True,
        ram=True,
    )

    figure = instance.plot(
        sample_dataframe,
        metrics=["cpu"],
        show=False,
        smooth="ema",
        alpha=0.5,
    )

    assert figure is not None
    assert len(figure.axes) == 2
    assert len(figure.axes[1].lines) >= 2

    import matplotlib.pyplot as plt

    plt.close(figure)


@pytest.mark.pruned
def test_plot_with_rolling_smoothing(
    sample_dataframe,
):
    instance = Monitor(
        cpu=True,
        ram=True,
    )

    figure = instance.plot(
        sample_dataframe,
        metrics=["ram"],
        show=False,
        smooth="rolling",
        window=3,
    )

    assert figure is not None
    assert len(figure.axes) == 2
    assert figure.axes[1].get_ylabel() == "RAM (GB)"
    assert len(figure.axes[1].lines) >= 2

    import matplotlib.pyplot as plt

    plt.close(figure)


@pytest.mark.pruned
def test_plot_timeline_contains_span_bar(
    sample_dataframe,
):
    instance = Monitor(
        cpu=True,
        ram=True,
    )

    figure = instance.plot(
        sample_dataframe,
        metrics=["cpu"],
        show=False,
    )

    assert figure is not None

    timeline_axis = figure.axes[0]

    assert len(timeline_axis.collections) >= 1

    import matplotlib.pyplot as plt

    plt.close(figure)


@pytest.mark.pruned
def test_plot_saves_file(
    sample_dataframe,
    tmp_path,
):
    instance = Monitor(
        cpu=True,
        ram=True,
    )
    instance.gpu_indices = (0,)

    output_path = tmp_path / "plots" / "resource_usage.png"

    figure = instance.plot(
        sample_dataframe,
        metrics=["cpu", "gpu0"],
        save_path=output_path,
        show=False,
    )

    assert figure is not None
    assert len(figure.axes) == 3
    assert output_path.exists()
    assert output_path.is_file()
    assert output_path.stat().st_size > 0

    import matplotlib.pyplot as plt

    plt.close(figure)


@pytest.mark.pruned
def test_plot_calls_show(
    sample_dataframe,
    monkeypatch,
):
    instance = Monitor(
        cpu=True,
        ram=True,
    )

    show = Mock()

    monkeypatch.setattr(
        "matplotlib.pyplot.show",
        show,
    )

    figure = instance.plot(
        sample_dataframe,
        metrics=["cpu"],
        show=True,
    )

    assert figure is not None
    show.assert_called_once()

    import matplotlib.pyplot as plt

    plt.close(figure)


@pytest.mark.pruned
def test_plot_does_not_call_show(
    sample_dataframe,
    monkeypatch,
):
    instance = Monitor(
        cpu=True,
        ram=True,
    )

    show = Mock()

    monkeypatch.setattr(
        "matplotlib.pyplot.show",
        show,
    )

    figure = instance.plot(
        sample_dataframe,
        metrics=["cpu"],
        show=False,
    )

    assert figure is not None
    show.assert_not_called()

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_plot_empty_dataframe():
    instance = Monitor()

    result = instance.plot(
        pd.DataFrame(),
        show=False,
    )

    assert result is None


@pytest.mark.pruned
def test_plot_without_samples():
    instance = Monitor()

    dataframe = pd.DataFrame(
        [
            {
                "t": 1.0,
                "stage": "main.work",
                "event": "checkpoint",
                "span_id": None,
                "cpu": None,
                "ram": None,
            }
        ]
    )

    result = instance.plot(
        dataframe,
        show=False,
    )

    assert result is None


@pytest.mark.pruned
def test_plot_without_timeline_events():
    instance = Monitor(
        cpu=True,
        ram=False,
    )

    dataframe = pd.DataFrame(
        [
            {
                "t": 1.0,
                "stage": "root",
                "event": "sample",
                "span_id": None,
                "cpu": 10.0,
            },
            {
                "t": 2.0,
                "stage": "root",
                "event": "sample",
                "span_id": None,
                "cpu": 20.0,
            },
        ]
    )

    figure = instance.plot(
        dataframe,
        metrics=["cpu"],
        show=False,
    )

    assert figure is not None
    assert len(figure.axes) == 2
    assert figure.axes[0].get_ylabel() == "Stage"
    assert figure.axes[1].get_ylabel() == "CPU (%)"

    timeline_text = [text.get_text() for text in figure.axes[0].texts]

    assert "No execution stages recorded" in timeline_text

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_plot_missing_required_columns():
    instance = Monitor()

    dataframe = pd.DataFrame(
        [
            {
                "cpu": 10.0,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="Monitoring dataframe is missing columns",
    ):
        instance.plot(
            dataframe,
            show=False,
        )


@pytest.mark.pruned
def test_plot_missing_stage_column():
    instance = Monitor()

    dataframe = pd.DataFrame(
        [
            {
                "t": 1.0,
                "event": "sample",
                "cpu": 10.0,
                "ram": 1.0,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="Monitoring dataframe is missing columns",
    ):
        instance.plot(
            dataframe,
            show=False,
        )


@pytest.mark.pruned
def test_plot_rejects_string_metrics(
    sample_dataframe,
):
    instance = Monitor()

    with pytest.raises(
        TypeError,
        match="metrics must be an iterable",
    ):
        instance.plot(
            sample_dataframe,
            metrics="cpu",
            show=False,
        )


def test_plot_rejects_unknown_metric(
    sample_dataframe,
):
    instance = Monitor()

    with pytest.raises(
        ValueError,
        match="were not collected",
    ):
        instance.plot(
            sample_dataframe,
            metrics=["temperature"],
            show=False,
        )


@pytest.mark.pruned
def test_plot_removes_duplicate_metrics(
    sample_dataframe,
):
    instance = Monitor(
        cpu=True,
        ram=True,
    )

    figure = instance.plot(
        sample_dataframe,
        metrics=["cpu", "cpu", "ram"],
        show=False,
    )

    assert figure is not None
    assert len(figure.axes) == 3
    assert figure.axes[0].get_ylabel() == "Stage"
    assert figure.axes[1].get_ylabel() == "CPU (%)"
    assert figure.axes[2].get_ylabel() == "RAM (GB)"

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_plot_returns_none_for_nan_metric():
    instance = Monitor(
        cpu=True,
        ram=False,
    )

    dataframe = pd.DataFrame(
        [
            {
                "t": 1.0,
                "stage": "root",
                "event": "sample",
                "span_id": None,
                "cpu": np.nan,
            },
            {
                "t": 2.0,
                "stage": "root",
                "event": "sample",
                "span_id": None,
                "cpu": np.nan,
            },
        ]
    )

    result = instance.plot(
        dataframe,
        metrics=["cpu"],
        show=False,
    )

    assert result is None


@pytest.mark.pruned
def test_sample_stage_tracks_active_span(
    basic_monitor,
):
    with basic_monitor.span("processing"):
        stage = basic_monitor._current_sample_stage()

    assert stage == "main.processing"


@pytest.mark.pruned
def test_sample_stage_returns_root_without_span(
    basic_monitor,
):
    assert basic_monitor._current_sample_stage() == "root"


@pytest.mark.parametrize(
    "local_rank",
    [
        None,
        0.0,
        "0",
        True,
    ],
)
def test_resolve_physical_gpu_index_rejects_non_integer(
    local_rank,
):
    with pytest.raises(
        TypeError,
        match="local_rank must be an integer",
    ):
        resolve_physical_gpu_index(local_rank)


def test_resolve_physical_gpu_index_rejects_negative_rank():
    with pytest.raises(
        ValueError,
        match="local_rank must be non-negative",
    ):
        resolve_physical_gpu_index(-1)


@pytest.mark.parametrize(
    "visible_devices",
    [
        None,
        "",
    ],
)
def test_resolve_physical_gpu_index_without_visible_devices(
    monkeypatch,
    visible_devices,
):
    if visible_devices is None:
        monkeypatch.delenv(
            "CUDA_VISIBLE_DEVICES",
            raising=False,
        )
    else:
        monkeypatch.setenv(
            "CUDA_VISIBLE_DEVICES",
            visible_devices,
        )

    assert resolve_physical_gpu_index(3) == 3


@pytest.mark.pruned
def test_resolve_physical_gpu_index_uses_visible_device_mapping(
    monkeypatch,
):
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES",
        "5, 2, 7",
    )

    assert resolve_physical_gpu_index(0) == 5
    assert resolve_physical_gpu_index(1) == 2
    assert resolve_physical_gpu_index(2) == 7


def test_resolve_physical_gpu_index_ignores_empty_entries(
    monkeypatch,
):
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES",
        " 5, ,2 ,, 7 ",
    )

    assert resolve_physical_gpu_index(1) == 2


def test_resolve_physical_gpu_index_rejects_out_of_range_rank(
    monkeypatch,
):
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES",
        "5,2",
    )

    with pytest.raises(
        ValueError,
        match="outside CUDA_VISIBLE_DEVICES",
    ):
        resolve_physical_gpu_index(2)


@pytest.mark.pruned
@pytest.mark.parametrize(
    "visible_devices",
    [
        "GPU-abcd,1",
        "MIG-device,2",
        "0,GPU-1234",
    ],
)
def test_resolve_physical_gpu_index_rejects_non_numeric_device(
    monkeypatch,
    visible_devices,
):
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES",
        visible_devices,
    )

    rank = 1 if visible_devices.startswith("0,") else 0

    with pytest.raises(
        ValueError,
        match="numeric CUDA_VISIBLE_DEVICES",
    ):
        resolve_physical_gpu_index(rank)


@pytest.mark.parametrize(
    (
        "keyword",
        "value",
        "message",
    ),
    [
        (
            "rank",
            None,
            "rank must be an integer",
        ),
        (
            "rank",
            0.0,
            "rank must be an integer",
        ),
        (
            "rank",
            True,
            "rank must be an integer",
        ),
        (
            "local_rank",
            None,
            "local_rank must be an integer",
        ),
        (
            "local_rank",
            False,
            "local_rank must be an integer",
        ),
        (
            "world_size",
            1.0,
            "world_size must be an integer",
        ),
        (
            "world_size",
            True,
            "world_size must be an integer",
        ),
    ],
)
def test_monitor_rejects_invalid_distributed_value_types(
    keyword,
    value,
    message,
):
    arguments = {
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
    }
    arguments[keyword] = value

    with pytest.raises(
        TypeError,
        match=message,
    ):
        Monitor(**arguments)


@pytest.mark.pruned
def test_monitor_rejects_negative_rank():
    with pytest.raises(
        ValueError,
        match="rank must be non-negative",
    ):
        Monitor(
            rank=-1,
            world_size=2,
        )


def test_monitor_rejects_negative_local_rank():
    with pytest.raises(
        ValueError,
        match="local_rank must be non-negative",
    ):
        Monitor(
            local_rank=-1,
        )


@pytest.mark.pruned
@pytest.mark.parametrize(
    "world_size",
    [
        0,
        -1,
    ],
)
def test_monitor_rejects_nonpositive_world_size(
    world_size,
):
    with pytest.raises(
        ValueError,
        match="world_size must be greater than zero",
    ):
        Monitor(
            world_size=world_size,
        )


def test_monitor_rejects_rank_equal_to_world_size():
    with pytest.raises(
        ValueError,
        match="rank must be less than world_size",
    ):
        Monitor(
            rank=2,
            world_size=2,
        )


@pytest.mark.pruned
def test_monitor_stores_distributed_metadata():
    instance = Monitor(
        rank=2,
        local_rank=1,
        world_size=4,
    )

    assert instance.rank == 2
    assert instance.local_rank == 1
    assert instance.world_size == 4
    assert isinstance(
        instance.hostname,
        str,
    )
    assert instance.hostname
    assert instance.pid > 0


@pytest.mark.pruned
def test_create_event_contains_distributed_metadata():
    instance = Monitor(
        cpu=False,
        ram=False,
        rank=1,
        local_rank=0,
        world_size=2,
    )

    event = instance._create_event(
        stage="main.work",
        event="checkpoint",
        span_id="abc",
    )

    assert event["stage"] == "main.work"
    assert event["event"] == "checkpoint"
    assert event["span_id"] == "abc"
    assert event["rank"] == 1
    assert event["local_rank"] == 0
    assert event["world_size"] == 2
    assert event["hostname"] == instance.hostname
    assert event["pid"] == instance.pid


@pytest.mark.pruned
def test_create_event_adds_enabled_metric_placeholders(
    fake_pynvml,
):
    instance = Monitor(
        cpu=True,
        ram=True,
        gpus=[
            0,
            1,
        ],
    )

    event = instance._create_event(
        stage="root",
        event="sample",
    )

    assert event["cpu"] is None
    assert event["ram"] is None
    assert event["gpu0"] is None
    assert event["vram0"] is None
    assert event["gpu1"] is None
    assert event["vram1"] is None


@pytest.mark.pruned
def test_gpu_initialization_marks_handle_failure_unavailable(
    monkeypatch,
):
    pynvml = Mock()
    pynvml.nvmlInit = Mock()
    pynvml.nvmlDeviceGetCount = Mock(return_value=2)
    pynvml.nvmlDeviceGetHandleByIndex = Mock(
        side_effect=[
            "gpu-zero",
            RuntimeError("handle failed"),
        ]
    )

    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        pynvml,
    )

    instance = Monitor(
        cpu=False,
        ram=False,
        gpus=[
            0,
            1,
        ],
    )

    assert instance._gpu_handles == {
        0: "gpu-zero",
    }
    assert instance._unavailable_gpus == {
        1,
    }


def test_repeated_failed_gpu_query_does_not_log_again(
    fake_pynvml,
    monkeypatch,
):
    fake_pynvml.nvmlDeviceGetUtilizationRates.side_effect = RuntimeError("failure")

    exception_logger = Mock()
    monkeypatch.setattr(
        module.logger,
        "exception",
        exception_logger,
    )

    instance = Monitor(
        cpu=False,
        ram=False,
        gpu0=True,
    )

    instance._gpu(0)
    instance._gpu(0)

    exception_logger.assert_called_once()


@pytest.mark.pruned
def test_shutdown_gpus_returns_when_not_initialized():
    instance = Monitor(
        cpu=False,
        ram=False,
    )

    instance._shutdown_gpus()

    assert instance._nvml_initialized is False
    assert instance._pynvml is None


def test_shutdown_gpus_clears_nvml_state(
    fake_pynvml,
):
    fake_pynvml.nvmlShutdown = Mock()

    instance = Monitor(
        cpu=False,
        ram=False,
        gpu0=True,
    )

    instance._shutdown_gpus()

    fake_pynvml.nvmlShutdown.assert_called_once()
    assert instance._nvml_initialized is False
    assert instance._pynvml is None
    assert instance._gpu_handles == {}


@pytest.mark.pruned
def test_shutdown_gpus_clears_state_after_failure(
    fake_pynvml,
):
    fake_pynvml.nvmlShutdown = Mock(side_effect=RuntimeError("shutdown failed"))

    instance = Monitor(
        cpu=False,
        ram=False,
        gpu0=True,
    )

    instance._shutdown_gpus()

    assert instance._nvml_initialized is False
    assert instance._pynvml is None
    assert instance._gpu_handles == {}


@pytest.mark.pruned
def test_remove_empty_stack_preserves_nonempty_stack(
    basic_monitor,
):
    stack = basic_monitor._get_stack()
    stack.append("main.work")

    basic_monitor._remove_empty_stack()

    thread_id = threading.get_ident()

    assert thread_id in (basic_monitor._thread_stacks)


@pytest.mark.pruned
def test_span_recovers_from_modified_stack(
    basic_monitor,
):
    with basic_monitor.span("work"):
        stack = basic_monitor._get_stack()
        stack.clear()

    assert basic_monitor.current_stage() == "root"

    dataframe = basic_monitor.get_dataframe()

    assert list(dataframe["event"]) == [
        "start",
        "end",
    ]


def test_span_removes_itself_from_modified_stack(
    basic_monitor,
):
    with basic_monitor.span("outer"):
        stack = basic_monitor._get_stack()

        with basic_monitor.span("inner"):
            stack.append("main.external")

        assert "main.outer.inner" not in stack
        stack.clear()

    assert basic_monitor.current_stage() == "root"


@pytest.mark.pruned
def test_current_sample_stage_combines_unique_active_stages(
    basic_monitor,
):
    current_thread = threading.get_ident()

    with basic_monitor._lock:
        basic_monitor._thread_stacks = {
            current_thread: [
                "main.current",
            ],
            current_thread + 1: [
                "main.other",
            ],
            current_thread + 2: [
                "main.current",
            ],
            current_thread + 3: [],
        }

    assert basic_monitor._current_sample_stage() == "main.current | main.other"


@pytest.mark.pruned
def test_current_sample_stage_excludes_sampler_thread(
    basic_monitor,
):
    sampler = SimpleNamespace(ident=123)
    basic_monitor._thread = sampler

    with basic_monitor._lock:
        basic_monitor._thread_stacks = {
            123: [
                "main.sampler",
            ],
            456: [
                "main.worker",
            ],
        }

    assert basic_monitor._current_sample_stage() == "main.worker"

    basic_monitor._thread = None


@pytest.mark.pruned
def test_sampler_survives_collection_failure(
    basic_monitor,
    monkeypatch,
):
    calls = 0

    def collect():
        nonlocal calls
        calls += 1

        if calls == 1:
            raise RuntimeError("temporary failure")

        return basic_monitor._create_event(
            stage="root",
            event="sample",
        )

    monkeypatch.setattr(
        basic_monitor,
        "_collect_sample",
        collect,
    )

    basic_monitor.start()

    assert wait_until(lambda: calls >= 2)

    basic_monitor.stop(timeout=1.0)

    assert calls >= 2


@pytest.mark.pruned
def test_sampler_uses_zero_wait_when_collection_exceeds_interval(
    basic_monitor,
    monkeypatch,
):
    monotonic_values = iter(
        [
            1.0,
            2.0,
        ]
    )

    monkeypatch.setattr(
        module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    waits = []

    class StopEvent:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls > 1

        def wait(self, value):
            waits.append(value)

    basic_monitor._stop_event = StopEvent()

    monkeypatch.setattr(
        basic_monitor,
        "_collect_sample",
        lambda: {
            "event": "sample",
        },
    )

    basic_monitor._sampler()

    assert waits == [
        0.0,
    ]


@pytest.mark.pruned
def test_stop_warns_when_thread_remains_alive(
    basic_monitor,
    monkeypatch,
):
    fake_thread = Mock()
    fake_thread.is_alive.return_value = True

    basic_monitor._thread = fake_thread

    warning = Mock()
    monkeypatch.setattr(
        module.logger,
        "warning",
        warning,
    )

    basic_monitor.stop(timeout=0.01)

    fake_thread.join.assert_called_once_with(timeout=0.01)
    warning.assert_called_once()
    assert basic_monitor._thread is fake_thread


@pytest.mark.pruned
def test_close_stops_and_shuts_down(
    basic_monitor,
    monkeypatch,
):
    stop = Mock()
    shutdown = Mock()

    monkeypatch.setattr(
        basic_monitor,
        "stop",
        stop,
    )
    monkeypatch.setattr(
        basic_monitor,
        "_shutdown_gpus",
        shutdown,
    )

    basic_monitor.close()

    stop.assert_called_once_with()
    shutdown.assert_called_once_with()


@pytest.mark.pruned
def test_context_manager_returns_same_monitor():
    instance = Monitor(
        interval=0.01,
    )

    with instance as entered:
        assert entered is instance

    assert instance.running is False


@pytest.mark.pruned
def test_kalman_filter_zero_denominator():
    result = Monitor._kalman_filter(
        [
            1.0,
            2.0,
        ],
        process_variance=0.0,
        measurement_variance=-0.0,
    )

    assert np.isfinite(result).all()


@pytest.mark.pruned
@pytest.mark.parametrize(
    "alpha",
    [
        -1.0,
        1.1,
        2.0,
    ],
)
def test_invalid_ema_alpha_additional_cases(
    alpha,
):
    with pytest.raises(
        ValueError,
        match="EMA alpha",
    ):
        Monitor._smooth(
            pd.Series(
                [
                    1.0,
                    2.0,
                ]
            ),
            method="EMA",
            alpha=alpha,
        )


@pytest.mark.pruned
def test_smoothing_method_is_case_insensitive():
    series = pd.Series(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    result = Monitor._smooth(
        series,
        method="EMA",
        alpha=0.5,
    )

    assert isinstance(
        result,
        pd.Series,
    )


@pytest.mark.pruned
def test_plot_metric_without_smoothing():
    axis = Mock()
    values = pd.Series(
        [
            1.0,
            2.0,
        ]
    )

    Monitor._plot_metric(
        axis,
        pd.Series(
            [
                0.0,
                1.0,
            ]
        ),
        values,
        label="CPU",
        color="red",
        smooth=None,
    )

    axis.plot.assert_called_once()
    axis.legend.assert_called_once()


def test_span_intervals_without_span_id_column():
    dataframe = pd.DataFrame(
        [
            {
                "event": "sample",
                "elapsed": 0.0,
            }
        ]
    )

    assert Monitor._span_intervals(dataframe) == []


def test_span_intervals_ignores_unfinished_span():
    dataframe = pd.DataFrame(
        [
            {
                "event": "start",
                "elapsed": 0.0,
                "stage": "main.work",
                "span_id": "unfinished",
            }
        ]
    )

    assert Monitor._span_intervals(dataframe) == []


def test_plot_uses_internal_dataframe(
    basic_monitor,
):
    basic_monitor._data.extend(
        [
            {
                "t": 1.0,
                "stage": "root",
                "event": "sample",
                "span_id": None,
                "rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "hostname": "host",
                "pid": 1,
                "cpu": 10.0,
                "ram": 1.0,
            },
            {
                "t": 2.0,
                "stage": "root",
                "event": "sample",
                "span_id": None,
                "rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "hostname": "host",
                "pid": 1,
                "cpu": 20.0,
                "ram": 2.0,
            },
        ]
    )

    figure = basic_monitor.plot(
        df=None,
        metrics=[
            "cpu",
        ],
        show=False,
    )

    assert figure is not None

    import matplotlib.pyplot as plt

    plt.close(figure)


@pytest.mark.pruned
def test_plot_filters_selected_metric_with_only_nan_values(
    sample_dataframe,
):
    instance = Monitor(
        cpu=True,
        ram=True,
    )

    dataframe = sample_dataframe.copy()
    dataframe["ram"] = np.nan

    figure = instance.plot(
        dataframe,
        metrics=[
            "cpu",
            "ram",
        ],
        show=False,
    )

    assert figure is not None
    assert len(figure.axes) == 2
    assert figure.axes[1].get_ylabel() == "CPU (%)"

    import matplotlib.pyplot as plt

    plt.close(figure)


@pytest.mark.pruned
def test_plot_save_path_without_directory(
    sample_dataframe,
    monkeypatch,
    tmp_path,
):
    instance = Monitor(
        cpu=True,
        ram=True,
    )

    monkeypatch.chdir(tmp_path)

    figure = instance.plot(
        sample_dataframe,
        metrics=[
            "cpu",
        ],
        save_path="plot.png",
        show=False,
    )

    assert figure is not None
    assert (tmp_path / "plot.png").exists()

    import matplotlib.pyplot as plt

    plt.close(figure)


@pytest.mark.pruned
def test_build_distributed_monitor_without_cuda(
    monkeypatch,
):
    distributed = SimpleNamespace(
        rank=1,
        local_rank=0,
        world_size=2,
    )

    monkeypatch.setattr(
        module,
        "monitor",
        Mock(return_value=object()),
    )
    monkeypatch.setattr(
        "torch.cuda.is_available",
        lambda: False,
    )

    result = build_distributed_monitor(
        distributed,
        cpu=True,
        ram=False,
        gpu=True,
        interval=0.25,
    )

    assert result is module.monitor.return_value

    module.monitor.assert_called_once_with(
        cpu=True,
        ram=False,
        gpus=[],
        interval=0.25,
        rank=1,
        local_rank=0,
        world_size=2,
    )


@pytest.mark.pruned
def test_build_distributed_monitor_with_cuda(
    monkeypatch,
):
    distributed = SimpleNamespace(
        rank=1,
        local_rank=2,
        world_size=4,
    )

    constructor = Mock(return_value=object())

    monkeypatch.setattr(
        module,
        "monitor",
        constructor,
    )
    monkeypatch.setattr(
        "torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        module,
        "resolve_physical_gpu_index",
        Mock(return_value=7),
    )

    build_distributed_monitor(distributed)

    module.resolve_physical_gpu_index.assert_called_once_with(2)

    constructor.assert_called_once_with(
        cpu=True,
        ram=True,
        gpus=[
            7,
        ],
        interval=0.1,
        rank=1,
        local_rank=2,
        world_size=4,
    )


@pytest.mark.parametrize(
    "error",
    [
        TypeError("invalid rank"),
        ValueError("invalid rank"),
    ],
)
def test_build_distributed_monitor_recovers_from_gpu_resolution_error(
    monkeypatch,
    error,
):
    distributed = SimpleNamespace(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    constructor = Mock(return_value=object())

    monkeypatch.setattr(
        module,
        "monitor",
        constructor,
    )
    monkeypatch.setattr(
        "torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        module,
        "resolve_physical_gpu_index",
        Mock(side_effect=error),
    )

    build_distributed_monitor(distributed)

    assert constructor.call_args.kwargs["gpus"] == []


@pytest.mark.pruned
def test_build_distributed_monitor_gpu_disabled_does_not_query_cuda(
    monkeypatch,
):
    distributed = SimpleNamespace(
        rank=0,
        local_rank=0,
        world_size=1,
    )

    cuda_available = Mock(side_effect=AssertionError("CUDA check should not run"))

    monkeypatch.setattr(
        "torch.cuda.is_available",
        cuda_available,
    )
    monkeypatch.setattr(
        module,
        "monitor",
        Mock(return_value=object()),
    )

    build_distributed_monitor(
        distributed,
        gpu=False,
    )

    cuda_available.assert_not_called()


def test_export_results_empty_dataframe(
    tmp_path,
):
    resource_monitor = Mock()
    resource_monitor.rank = 3
    resource_monitor.get_dataframe.return_value = pd.DataFrame()

    result = export_results(
        resource_monitor,
        tmp_path,
    )

    assert result == (
        None,
        None,
    )


@pytest.mark.pruned
def test_export_results_writes_csv_and_plot(
    tmp_path,
    monkeypatch,
):
    resource_monitor = Mock()
    resource_monitor.rank = 2

    dataframe = pd.DataFrame(
        [
            {
                "t": 1.0,
                "event": "sample",
                "stage": "root",
                "cpu": 10.0,
            }
        ]
    )

    resource_monitor.get_dataframe.return_value = dataframe

    figure = Mock()
    resource_monitor.plot.return_value = figure

    close = Mock()
    monkeypatch.setattr(
        "matplotlib.pyplot.close",
        close,
    )

    csv_path, plot_path = export_results(
        resource_monitor,
        tmp_path,
    )

    assert csv_path == (tmp_path / "resource_monitoring_rank_0002.csv")
    assert plot_path == (tmp_path / "resource_monitoring_rank_0002.png")
    assert csv_path.exists()

    resource_monitor.plot.assert_called_once_with(
        df=dataframe,
        save_path=plot_path,
        show=False,
        smooth="kalman",
        process_variance=1.0,
        measurement_variance=25.0,
    )

    close.assert_called_once_with(figure)


@pytest.mark.pruned
def test_export_results_does_not_close_none_figure(
    tmp_path,
    monkeypatch,
):
    resource_monitor = Mock()
    resource_monitor.rank = 0
    resource_monitor.get_dataframe.return_value = pd.DataFrame(
        [
            {
                "t": 1.0,
                "event": "sample",
                "stage": "root",
            }
        ]
    )
    resource_monitor.plot.return_value = None

    close = Mock()
    monkeypatch.setattr(
        "matplotlib.pyplot.close",
        close,
    )

    export_results(
        resource_monitor,
        tmp_path,
    )

    close.assert_not_called()


@pytest.mark.parametrize(
    "world_size",
    [
        None,
        1.0,
        "1",
        True,
    ],
)
def test_combine_rank_results_rejects_invalid_world_size_type(
    tmp_path,
    world_size,
):
    with pytest.raises(
        TypeError,
        match="world_size must be an integer",
    ):
        combine_rank_results(
            tmp_path,
            world_size,
        )


@pytest.mark.parametrize(
    "world_size",
    [
        0,
        -1,
    ],
)
def test_combine_rank_results_rejects_nonpositive_world_size(
    tmp_path,
    world_size,
):
    with pytest.raises(
        ValueError,
        match="world_size must be greater than zero",
    ):
        combine_rank_results(
            tmp_path,
            world_size,
        )


def test_combine_rank_results_returns_none_without_files(
    tmp_path,
):
    assert (
        combine_rank_results(
            tmp_path,
            2,
        )
        is None
    )


@pytest.mark.pruned
def test_combine_rank_results_combines_and_sorts_files(
    tmp_path,
):
    pd.DataFrame(
        [
            {
                "t": 3.0,
                "rank": 0,
                "value": "late",
            },
            {
                "t": 1.0,
                "rank": 0,
                "value": "early",
            },
        ]
    ).to_csv(
        tmp_path / "resource_monitoring_rank_0000.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "t": 2.0,
                "rank": 1,
                "value": "middle",
            }
        ]
    ).to_csv(
        tmp_path / "resource_monitoring_rank_0001.csv",
        index=False,
    )

    result = combine_rank_results(
        tmp_path,
        2,
    )

    assert result == (tmp_path / "resource_monitoring_all_ranks.csv")
    assert result.exists()

    combined = pd.read_csv(result)

    assert list(combined["t"]) == [
        1.0,
        2.0,
        3.0,
    ]


@pytest.mark.pruned
def test_combine_rank_results_allows_partial_rank_files(
    tmp_path,
):
    pd.DataFrame(
        [
            {
                "t": 1.0,
                "rank": 0,
            }
        ]
    ).to_csv(
        tmp_path / "resource_monitoring_rank_0000.csv",
        index=False,
    )

    result = combine_rank_results(
        tmp_path,
        2,
    )

    assert result is not None
    assert result.exists()


def test_combine_rank_results_without_sort_columns(
    tmp_path,
):
    pd.DataFrame(
        [
            {
                "value": 2,
            },
            {
                "value": 1,
            },
        ]
    ).to_csv(
        tmp_path / "resource_monitoring_rank_0000.csv",
        index=False,
    )

    result = combine_rank_results(
        tmp_path,
        1,
    )

    combined = pd.read_csv(result)

    assert list(combined["value"]) == [
        2,
        1,
    ]


class DummyDistributed:
    def __init__(
        self,
        *,
        root=True,
        rank=0,
        local_rank=0,
        world_size=1,
    ):
        self.root = root
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.barrier_calls = 0

    def is_root(self):
        return self.root

    def barrier(self):
        self.barrier_calls += 1


def make_context_monitor():
    resource_monitor = Mock()
    resource_monitor.rank = 0
    resource_monitor.start = Mock()
    resource_monitor.stop = Mock()
    resource_monitor.close = Mock()
    resource_monitor.checkpoint = Mock()

    return resource_monitor


def test_distributed_monitoring_handles_export_failure(
    tmp_path,
    monkeypatch,
):
    distributed = DummyDistributed()
    resource_monitor = make_context_monitor()

    monkeypatch.setattr(
        module,
        "build_distributed_monitor",
        Mock(return_value=resource_monitor),
    )
    monkeypatch.setattr(
        module,
        "export_results",
        Mock(side_effect=RuntimeError("export failed")),
    )
    monkeypatch.setattr(
        module,
        "combine_rank_results",
        Mock(),
    )

    with distributed_monitoring(
        distributed,
        tmp_path,
    ):
        pass

    resource_monitor.close.assert_called_once_with()
    assert distributed.barrier_calls == 1
    module.combine_rank_results.assert_called_once_with(
        tmp_path,
        1,
    )


@pytest.mark.pruned
def test_distributed_monitoring_nonroot_does_not_combine(
    tmp_path,
    monkeypatch,
):
    distributed = DummyDistributed(
        root=False,
        rank=1,
        local_rank=1,
        world_size=2,
    )
    resource_monitor = make_context_monitor()

    monkeypatch.setattr(
        module,
        "build_distributed_monitor",
        Mock(return_value=resource_monitor),
    )
    monkeypatch.setattr(
        module,
        "export_results",
        Mock(
            return_value=(
                None,
                None,
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "combine_rank_results",
        Mock(),
    )

    with distributed_monitoring(
        distributed,
        tmp_path,
    ):
        pass

    assert distributed.barrier_calls == 1
    module.combine_rank_results.assert_not_called()


@pytest.mark.pruned
def test_distributed_monitoring_combine_disabled(
    tmp_path,
    monkeypatch,
):
    distributed = DummyDistributed()
    resource_monitor = make_context_monitor()

    monkeypatch.setattr(
        module,
        "build_distributed_monitor",
        Mock(return_value=resource_monitor),
    )
    monkeypatch.setattr(
        module,
        "export_results",
        Mock(
            return_value=(
                None,
                None,
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "combine_rank_results",
        Mock(),
    )

    with distributed_monitoring(
        distributed,
        tmp_path,
        combine_ranks=False,
    ):
        pass

    module.combine_rank_results.assert_not_called()


@pytest.mark.pruned
def test_distributed_monitoring_forwards_monitor_options(
    tmp_path,
    monkeypatch,
):
    distributed = DummyDistributed()
    resource_monitor = make_context_monitor()

    builder = Mock(return_value=resource_monitor)

    monkeypatch.setattr(
        module,
        "build_distributed_monitor",
        builder,
    )
    monkeypatch.setattr(
        module,
        "export_results",
        Mock(
            return_value=(
                None,
                None,
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "combine_rank_results",
        Mock(),
    )

    with distributed_monitoring(
        distributed,
        tmp_path,
        cpu=False,
        ram=False,
        gpu=False,
        interval=2.5,
    ):
        pass

    builder.assert_called_once_with(
        distributed,
        cpu=False,
        ram=False,
        gpu=False,
        interval=2.5,
    )

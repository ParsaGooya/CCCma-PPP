import time

import numpy as np
import pandas as pd
import pytest
import pynvml

import cccma_ppp.generic.monitoring as monitoring

from cccma_ppp.generic.monitoring import (
    _PROCESS,
    _data,
    _get_stack,
    _gpu,
    _kalman_filter,
    _plot_metric,
    _sampler,
    _smooth,
    _thread_state,
    checkpoint,
    current_stage,
    get_dataframe,
    observe,
    plot,
    span,
    start_monitoring,
    stop_monitoring,
)


@pytest.fixture(autouse=True)
def reset_monitoring_state():
    _data.clear()

    if hasattr(_thread_state, "stack"):
        _thread_state.stack = []

    monitoring._running = False

    yield

    _data.clear()

    if hasattr(_thread_state, "stack"):
        _thread_state.stack = []

    monitoring._running = False


@pytest.mark.pruned
def test_get_stack_creates_stack():

    if hasattr(_thread_state, "stack"):
        delattr(_thread_state, "stack")

    stack = _get_stack()

    assert stack == []
    assert hasattr(_thread_state, "stack")


@pytest.mark.pruned
def test_get_stack_reuses_stack():

    _thread_state.stack = ["abc"]

    stack = _get_stack()

    assert stack is _thread_state.stack


@pytest.mark.pruned
def test_current_stage_root():

    _thread_state.stack = []

    assert current_stage() == "root"


@pytest.mark.pruned
def test_current_stage_non_root():

    _thread_state.stack = ["main.train"]

    assert current_stage() == "main.train"


def test_gpu_unavailable(monkeypatch):

    monkeypatch.setattr(
        monitoring,
        "_GPU_AVAILABLE",
        False,
    )

    assert _gpu() == (0.0, 0.0)


def test_gpu_available(monkeypatch):

    monkeypatch.setattr(
        monitoring,
        "_GPU_AVAILABLE",
        True,
    )

    class Util:
        gpu = 50

    class Mem:
        used = 100
        total = 200

    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetHandleByIndex",
        lambda *_: object(),
    )

    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetUtilizationRates",
        lambda *_: Util(),
    )

    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetMemoryInfo",
        lambda *_: Mem(),
    )

    gpu, vram = _gpu()

    assert gpu == 50
    assert vram == 50.0


@pytest.mark.pruned
def test_gpu_available_zero_total_memory(monkeypatch):

    monkeypatch.setattr(
        monitoring,
        "_GPU_AVAILABLE",
        True,
    )

    class Util:
        gpu = 12

    class Mem:
        used = 0
        total = 0

    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetHandleByIndex",
        lambda *_: object(),
    )

    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetUtilizationRates",
        lambda *_: Util(),
    )

    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetMemoryInfo",
        lambda *_: Mem(),
    )

    gpu, vram = _gpu()

    assert gpu == 12
    assert vram == 0.0


@pytest.mark.pruned
def test_span_records_events():

    with span("load"):
        assert current_stage() == "main.load"

    assert len(_data) == 2
    assert _data[0]["event"] == "start"
    assert _data[1]["event"] == "end"
    assert _data[0]["stage"] == "main.load"


@pytest.mark.pruned
def test_nested_spans():

    with span("outer"):
        assert current_stage() == "main.outer"

        with span("inner"):
            assert current_stage() == "main.outer.inner"

    stages = [x["stage"] for x in _data]

    assert any("outer" in s for s in stages)
    assert any("inner" in s for s in stages)


@pytest.mark.pruned
def test_span_records_end_on_exception():

    with pytest.raises(RuntimeError):
        with span("boom"):
            raise RuntimeError()

    assert _data[0]["event"] == "start"
    assert _data[-1]["event"] == "end"


@pytest.mark.pruned
def test_observe_decorator():

    @observe
    def add(a, b):
        return a + b

    assert add(2, 3) == 5

    assert _data[0]["stage"] == "main.add"
    assert _data[0]["event"] == "start"
    assert _data[-1]["event"] == "end"


@pytest.mark.pruned
def test_checkpoint_root():

    checkpoint("save")

    assert _data[-1]["event"] == "checkpoint"
    assert _data[-1]["stage"] == "main.save"


@pytest.mark.pruned
def test_checkpoint_nested():

    with span("training"):
        checkpoint("epoch1")

    checkpoints = [x for x in _data if x["event"] == "checkpoint"]

    assert len(checkpoints) == 1
    assert checkpoints[0]["stage"].endswith(".epoch1")


def test_sampler_single_iteration(monkeypatch):

    monkeypatch.setattr(
        monitoring,
        "_ram_gb",
        lambda: 1.0,
    )

    monkeypatch.setattr(
        monitoring,
        "_gpu",
        lambda: (10.0, 20.0),
    )

    monkeypatch.setattr(
        _PROCESS,
        "cpu_percent",
        lambda interval=None: 30.0,
    )

    def stop_sleep(_):
        monitoring._running = False

    monkeypatch.setattr(
        time,
        "sleep",
        stop_sleep,
    )

    monitoring._running = True

    _sampler(0.001)

    samples = [x for x in _data if x["event"] == "sample"]

    assert len(samples) == 1

    sample = samples[0]

    assert sample["cpu"] == 30.0
    assert sample["ram"] == 1.0
    assert sample["gpu"] == 10.0
    assert sample["vram"] == 20.0


@pytest.mark.pruned
def test_start_and_stop_monitoring():

    start_monitoring(interval=0.01)

    assert monitoring._running is True

    stop_monitoring()

    assert monitoring._running is False


@pytest.mark.pruned
def test_get_dataframe():

    checkpoint("x")

    df = get_dataframe()

    assert not df.empty
    assert "event" in df.columns


def test_kalman_filter_empty():

    result = _kalman_filter([])

    assert len(result) == 0


@pytest.mark.pruned
def test_kalman_filter_constant():

    result = _kalman_filter([5, 5, 5, 5])

    assert len(result) == 4
    assert np.isfinite(result).all()


def test_smooth_none():

    series = pd.Series([1, 2, 3])

    result = _smooth(series)

    assert result is series


@pytest.mark.pruned
def test_smooth_ema():

    series = pd.Series([1, 2, 3])

    result = _smooth(
        series,
        method="ema",
        alpha=0.5,
    )

    assert len(result) == 3


def test_smooth_kalman():

    series = pd.Series([1, 2, 3, 4])

    result = _smooth(
        series,
        method="kalman",
    )

    assert len(result) == 4


@pytest.mark.pruned
def test_smooth_invalid():

    series = pd.Series([1, 2, 3])

    with pytest.raises(ValueError):
        _smooth(
            series,
            method="invalid",
        )


@pytest.mark.pruned
def test_plot_metric_no_smoothing():

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    _plot_metric(
        ax=ax,
        t=[0, 1, 2],
        values=[1, 2, 3],
        label="CPU",
        color="red",
    )

    assert len(ax.lines) == 1

    plt.close(fig)


@pytest.mark.pruned
def test_plot_metric_with_smoothing():

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    _plot_metric(
        ax=ax,
        t=[0, 1, 2],
        values=pd.Series([1, 2, 3]),
        label="CPU",
        color="red",
        smooth="ema",
    )

    assert len(ax.lines) == 2

    plt.close(fig)


def test_plot_empty_dataframe():

    plot(
        pd.DataFrame(),
        show=False,
    )


def test_plot_only_non_sample_events():

    df = pd.DataFrame([{"t": 0, "event": "start", "stage": "x"}])

    plot(
        df,
        show=False,
    )


def test_plot_sample_data(tmp_path):

    df = pd.DataFrame(
        [
            {
                "t": 0,
                "event": "start",
                "stage": "main.train",
                "cpu": None,
                "ram": None,
                "gpu": None,
                "vram": None,
            },
            {
                "t": 1,
                "event": "sample",
                "stage": "main.train",
                "cpu": 10,
                "ram": 1,
                "gpu": 0,
                "vram": 0,
            },
            {
                "t": 2,
                "event": "sample",
                "stage": "main.train",
                "cpu": 20,
                "ram": 2,
                "gpu": 5,
                "vram": 10,
            },
            {
                "t": 3,
                "event": "checkpoint",
                "stage": "main.train.ckpt",
                "cpu": None,
                "ram": None,
                "gpu": None,
                "vram": None,
            },
            {
                "t": 4,
                "event": "end",
                "stage": "main.train",
                "cpu": None,
                "ram": None,
                "gpu": None,
                "vram": None,
            },
        ]
    )

    output = tmp_path / "plot.png"

    plot(
        df,
        save_path=str(output),
        show=False,
        smooth="ema",
    )

    assert output.exists()


@pytest.mark.pruned
def test_plot_kalman_smoothing(tmp_path):

    df = pd.DataFrame(
        [
            {
                "t": 0,
                "event": "sample",
                "stage": "x",
                "cpu": 1,
                "ram": 1,
                "gpu": 1,
                "vram": 1,
            },
            {
                "t": 1,
                "event": "sample",
                "stage": "x",
                "cpu": 2,
                "ram": 2,
                "gpu": 2,
                "vram": 2,
            },
        ]
    )

    output = tmp_path / "kalman.png"

    plot(
        df,
        save_path=str(output),
        show=False,
        smooth="kalman",
    )

    assert output.exists()


@pytest.mark.pruned
def test_current_stage_after_nested_span_cleanup():

    with span("outer"):
        pass

    assert current_stage() == "root"


@pytest.mark.pruned
def test_kalman_filter_single_value():

    result = _kalman_filter([42])

    assert len(result) == 1
    assert result[0] == 42


@pytest.mark.pruned
def test_kalman_filter_custom_variances():

    result = _kalman_filter(
        [1, 2, 3],
        process_variance=0.01,
        measurement_variance=0.01,
    )

    assert len(result) == 3


@pytest.mark.pruned
def test_plot_metric_with_kalman():

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    _plot_metric(
        ax=ax,
        t=[0, 1, 2],
        values=pd.Series([1, 2, 3]),
        label="CPU",
        color="red",
        smooth="kalman",
    )

    assert len(ax.lines) == 2

    plt.close(fig)


def test_plot_show_branch(monkeypatch):

    called = {"show": False}

    import matplotlib.pyplot as plt

    monkeypatch.setattr(
        plt,
        "show",
        lambda: called.__setitem__("show", True),
    )

    df = pd.DataFrame(
        [
            {
                "t": 0,
                "event": "sample",
                "stage": "x",
                "cpu": 1,
                "ram": 1,
                "gpu": 1,
                "vram": 1,
            }
        ]
    )

    plot(df, show=True)

    assert called["show"]


@pytest.mark.pruned
def test_plot_stage_without_matching_end(tmp_path):

    df = pd.DataFrame(
        [
            {
                "t": 0,
                "event": "start",
                "stage": "main.train",
            },
            {
                "t": 1,
                "event": "sample",
                "stage": "main.train",
                "cpu": 1,
                "ram": 1,
                "gpu": 1,
                "vram": 1,
            },
        ]
    )

    output = tmp_path / "out.png"

    plot(
        df,
        save_path=str(output),
        show=False,
    )

    assert output.exists()


@pytest.mark.pruned
def test_plot_without_checkpoints(tmp_path):

    df = pd.DataFrame(
        [
            {
                "t": 0,
                "event": "sample",
                "stage": "x",
                "cpu": 1,
                "ram": 1,
                "gpu": 1,
                "vram": 1,
            }
        ]
    )

    output = tmp_path / "plot.png"

    plot(
        df,
        save_path=str(output),
        show=False,
    )

    assert output.exists()


def test_sampler_with_active_stage(monkeypatch):

    _thread_state.stack = ["main.training"]

    monkeypatch.setattr(
        monitoring,
        "_ram_gb",
        lambda: 1.0,
    )

    monkeypatch.setattr(
        monitoring,
        "_gpu",
        lambda: (0.0, 0.0),
    )

    monkeypatch.setattr(
        _PROCESS,
        "cpu_percent",
        lambda interval=None: 1.0,
    )

    def stop_sleep(_):
        monitoring._running = False

    monkeypatch.setattr(time, "sleep", stop_sleep)

    monitoring._running = True

    _sampler(0.01)

    assert _data[-1]["stage"] == "main.training"


@pytest.mark.pruned
def test_multiple_checkpoints():

    checkpoint("a")
    checkpoint("b")

    checkpoints = [x for x in _data if x["event"] == "checkpoint"]

    assert len(checkpoints) == 2
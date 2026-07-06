from __future__ import annotations

import functools
import logging
import os
import threading
import time
from contextlib import contextmanager

import psutil

logger = logging.getLogger(__name__)

_PROCESS = psutil.Process(os.getpid())

try:
    import pynvml

    pynvml.nvmlInit()
    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False


_data = []
_lock = threading.Lock()

_thread_state = threading.local()

_running = False


def _get_stack():
    """
    Retrieve thread-local stage stack.

    Returns
    -------
    list
        Stack of active monitoring stages for the current thread.
    """

    if not hasattr(_thread_state, "stack"):
        _thread_state.stack = []
    return _thread_state.stack


def _ram_gb():
    """
    Get current process memory usage.

    Returns
    -------
    float
        Resident memory usage in gigabytes.
    """
    return _PROCESS.memory_info().rss / (1024**3)


def _cpu():
    """
    Get current CPU utilization.

    Returns
    -------
    float
        CPU utilization percentage for the current process.
    """

    return _PROCESS.cpu_percent(interval=None)


def _gpu():
    """
    Get current GPU utilization statistics.

    Returns
    -------
    tuple of float
        GPU utilization percentage and VRAM utilization percentage.
    """
    if not _GPU_AVAILABLE:
        return 0.0, 0.0

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

    gpu_util = util.gpu
    vram = (mem.used / mem.total) * 100 if mem.total else 0.0

    return gpu_util, vram


def current_stage():
    """
    Get the currently active monitoring stage.

    Returns
    -------
    str
        Name of the current stage. Returns "root" if no stage
        is active.
    """

    stack = _get_stack()
    return stack[-1] if stack else "root"


@contextmanager
def span(name: str):
    """
    Create a monitored execution span.

    Parameters
    ----------
    name : str
        Name of the stage.

    Yields
    ------
    None
    """

    stack = _get_stack()

    parent = stack[-1] if stack else None
    full_name = f"{parent}.{name}" if parent else f"main.{name}"

    stack.append(full_name)

    start_t = time.time()

    with _lock:
        _data.append(
            {
                "t": start_t,
                "stage": full_name,
                "event": "start",
                "cpu": None,
                "ram": None,
                "gpu": None,
                "vram": None,
            }
        )

    try:
        yield

    finally:
        end_t = time.time()

        with _lock:
            _data.append(
                {
                    "t": end_t,
                    "stage": full_name,
                    "event": "end",
                    "cpu": None,
                    "ram": None,
                    "gpu": None,
                    "vram": None,
                }
            )

        stack.pop()


def observe(fn):
    """
    Decorator for automatic stage monitoring.

    Parameters
    ----------
    fn : callable
        Function to observe.

    Returns
    -------
    callable
        Wrapped function.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with span(fn.__name__):
            return fn(*args, **kwargs)

    return wrapper


def checkpoint(name: str):
    """
    Record a checkpoint event.

    Parameters
    ----------
    name : str
        Checkpoint identifier.

    Returns
    -------
    None
    """

    stack = _get_stack()

    parent = stack[-1] if stack else None
    full_name = f"{parent}.{name}" if parent else f"main.{name}"

    with _lock:
        _data.append(
            {
                "t": time.time(),
                "stage": full_name,
                "event": "checkpoint",
                "cpu": None,
                "ram": None,
                "gpu": None,
                "vram": None,
            }
        )


def _sampler(interval: float):
    """
    Background resource monitoring loop.

    Parameters
    ----------
    interval : float
        Sampling interval in seconds.

    Returns
    -------
    None
    """
    _PROCESS.cpu_percent(interval=None)

    while _running:
        cpu = _PROCESS.cpu_percent(interval=None)
        ram = _ram_gb()
        gpu, vram = _gpu()

        with _lock:
            _data.append(
                {
                    "t": time.time(),
                    "stage": current_stage(),
                    "event": "sample",
                    "cpu": cpu,
                    "ram": ram,
                    "gpu": gpu,
                    "vram": vram,
                }
            )

        time.sleep(interval)


def start_monitoring(interval: float = 0.1):
    """
    Start resource monitoring.

    Parameters
    ----------
    interval : float, optional
        Sampling interval in seconds.

    Returns
    -------
    None
    """
    global _running

    _running = True

    thread = threading.Thread(
        target=_sampler,
        args=(interval,),
        daemon=True,
    )
    thread.start()

    logger.info("Monitoring started")


def stop_monitoring():
    """
    Stop resource monitoring.

    Returns
    -------
    None
    """
    global _running

    _running = False

    logger.info("Monitoring stopped")


def get_dataframe():
    """
    Convert collected monitoring events to a dataframe.

    Returns
    -------
    pandas.DataFrame
        Monitoring events and sampled resource metrics.
    """
    import pandas as pd

    with _lock:
        return pd.DataFrame(_data)


def _kalman_filter(
    values,
    process_variance: float = 1.0,
    measurement_variance: float = 25.0,
):
    """
    Apply a simple 1D Kalman filter.

    Parameters
    ----------
    values : array-like
        Input observations.
    process_variance : float, optional
        Process noise variance.
    measurement_variance : float, optional
        Observation noise variance.

    Returns
    -------
    np.ndarray
        Smoothed values.
    """
    import numpy as np

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    x = values[0]
    p = 1.0

    filtered = np.empty_like(values)

    for i, z in enumerate(values):
        p = p + process_variance

        k = p / (p + measurement_variance)

        x = x + k * (z - x)
        p = (1 - k) * p

        filtered[i] = x

    return filtered


def _smooth(series, method=None, **kwargs):
    """
    Smooth a time series.

    Parameters
    ----------
    series : pandas.Series or array-like
        Input values.
    method : {"ema", "rolling", "kalman"} or None, optional
        Smoothing method. If None, the input series is returned
        unchanged.
    **kwargs
        Additional arguments passed to the selected smoothing
        method.

    Returns
    -------
    pandas.Series or np.ndarray
        Smoothed values.

    Raises
    ------
    ValueError
        If an unsupported smoothing method is requested.
    """

    if method is None:
        return series

    if method == "ema":
        alpha = kwargs.get("alpha", 0.2)
        return series.ewm(alpha=alpha).mean()

    if method == "rolling":
        window = kwargs.get("window", 5)
        return (
            series.rolling(window=window, center=True)
            .mean()
            .fillna(method="bfill")
            .fillna(method="ffill")
        )

    if method == "kalman":
        return _kalman_filter(
            series.to_numpy(),
            process_variance=kwargs.get("process_variance", 1.0),
            measurement_variance=kwargs.get(
                "measurement_variance",
                25.0,
            ),
        )

    raise ValueError(f"Unknown smoothing method: {method}")


def _plot_metric(
    ax,
    t,
    values,
    label: str,
    color: str,
    smooth: str | None = None,
    **smooth_kwargs,
):
    """
    Plot raw and optionally smoothed metric.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    t : array-like
        Time values.
    values : array-like
        Metric values.
    label : str
        Metric name.
    color : str
        Plot color.
    smooth : {"ema", "rolling", "kalman"} or None, optional
        Smoothing method.
    **smooth_kwargs
        Arguments passed to smoothing function.

    Returns
    -------
    None
    """

    ax.plot(
        t,
        values,
        color=color,
        alpha=0.25,
        linewidth=1,
        label=f"{label} (raw)",
    )

    if smooth is not None:
        smoothed = _smooth(
            values,
            method=smooth,
            **smooth_kwargs,
        )

        ax.plot(
            t,
            smoothed,
            color=color,
            linewidth=2.5,
            label=f"{label} ({smooth})",
        )

    # ax.legend(loc="upper right", fontsize=8)


def plot(
    df,
    save_path: str | None = None,
    show: bool = True,
    smooth: str | None = None,
    **smooth_kwargs,
):
    """
    Plot resource utilization over time.

    Parameters
    ----------
    df : pandas.DataFrame
        Monitoring dataframe generated by ``get_dataframe``.
    save_path : str or None, optional
        Path where the figure should be saved.
    show : bool, optional
        Whether to display the figure.
    smooth : {"ema", "rolling", "kalman"} or None, optional
        Smoothing method applied to utilization curves. If None,
        raw values are plotted.
    **smooth_kwargs
        Additional keyword arguments passed to the selected
        smoothing method.

        For ``smooth="ema"``:

        - alpha : float, optional
            Exponential moving average smoothing factor.

        For ``smooth="rolling"``:

        - window : int, optional
            Size of the centered rolling window.

        For ``smooth="kalman"``:

        - process_variance : float, optional
            Process noise variance controlling responsiveness.
        - measurement_variance : float, optional
            Observation noise variance controlling smoothing
            strength.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    df = df.copy()

    if df.empty:
        return

    t0 = df["t"].min()
    df["t"] = df["t"] - t0

    samples = df[df["event"] == "sample"]

    if samples.empty:
        return

    samples["t"] = samples["t"] - samples["t"].min()

    fig, ax = plt.subplots(
        3,
        1,
        figsize=(14, 8),
        sharex=True,
    )

    _plot_metric(
        ax[0],
        samples["t"],
        samples["cpu"],
        label="CPU",
        color="red",
        smooth=smooth,
        **smooth_kwargs,
    )

    _plot_metric(
        ax[1],
        samples["t"],
        samples["ram"],
        label="RAM",
        color="blue",
        smooth=smooth,
        **smooth_kwargs,
    )

    _plot_metric(
        ax[2],
        samples["t"],
        samples["gpu"],
        label="GPU",
        color="green",
        smooth=smooth,
        **smooth_kwargs,
    )

    ax[0].set_ylabel("CPU %")
    ax[1].set_ylabel("RAM GB")
    ax[2].set_ylabel("GPU %")
    ax[2].set_xlabel("Time (s)")

    starts = df[df["event"] == "start"]
    ends = df[df["event"] == "end"]

    for _, s in starts.iterrows():
        stage = s["stage"]

        e = ends[ends["stage"] == stage]

        if e.empty:
            continue

        start = s["t"]
        end = e.iloc[0]["t"]

        for a in ax:
            a.axvspan(start, end, alpha=0.08)

        ax[0].text(
            (start + end) / 2,
            ax[0].get_ylim()[1] * 0.9,
            stage,
            rotation=90,
            fontsize=8,
            ha="center",
        )

    checkpoints = df[df["event"] == "checkpoint"]

    for _, cp in checkpoints.iterrows():
        x = cp["t"]

        for a in ax:
            a.axvline(x, linestyle="--", alpha=0.4)

        ax[0].text(
            x,
            ax[0].get_ylim()[1] * 0.98,
            cp["stage"],
            rotation=90,
            fontsize=7,
            va="top",
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

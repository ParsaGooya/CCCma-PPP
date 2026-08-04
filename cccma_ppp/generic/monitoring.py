import functools
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar, cast

import psutil

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class Monitor:
    """
    Document this class.

    Parameters
    ----------
    cpu : bool
        Description not yet provided.
    ram : bool
        Description not yet provided.
    gpu0 : bool
        Description not yet provided.
    gpu1 : bool
        Description not yet provided.
    gpus : Iterable[int] | None
        Description not yet provided.
    interval : float
        Description not yet provided.
    """

    def __init__(
        self,
        *,
        cpu: bool = True,
        ram: bool = True,
        gpu0: bool = False,
        gpu1: bool = False,
        gpus: Iterable[int] | None = None,
        interval: float = 0.1,
    ) -> None:
        """
        Document this function.

        Parameters
        ----------
        cpu : bool
            Description not yet provided.
        ram : bool
            Description not yet provided.
        gpu0 : bool
            Description not yet provided.
        gpu1 : bool
            Description not yet provided.
        gpus : Iterable[int] | None
            Description not yet provided.
        interval : float
            Description not yet provided.

        Raises
        ------
        TypeError
            Description not yet provided.
        ValueError
            Description not yet provided.
        """
        if interval <= 0:
            raise ValueError("interval must be greater than zero")

        gpu_indices = set(gpus or ())

        if gpu0:
            gpu_indices.add(0)

        if gpu1:
            gpu_indices.add(1)

        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in gpu_indices
        ):
            raise TypeError("GPU indices must be integers")

        if any(index < 0 for index in gpu_indices):
            raise ValueError("GPU indices must be non-negative")

        self.cpu_enabled = bool(cpu)
        self.ram_enabled = bool(ram)
        self.gpu_indices = tuple(sorted(gpu_indices))
        self.interval = float(interval)

        self._process = psutil.Process(os.getpid())

        self._data: list[dict[str, Any]] = []
        self._lock = threading.RLock()

        self._thread_stacks: dict[int, list[str]] = {}

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._pynvml: Any | None = None
        self._gpu_handles: dict[int, Any] = {}
        self._unavailable_gpus: set[int] = set()

        if self.gpu_indices:
            self._initialize_gpus()

    def _initialize_gpus(self) -> None:
        """
        Document this function.
        """
        try:
            import pynvml

            pynvml.nvmlInit()

            self._pynvml = pynvml

            device_count = pynvml.nvmlDeviceGetCount()

            for index in self.gpu_indices:
                if index >= device_count:
                    logger.warning(
                        "GPU %s is unavailable. The system reports %s "
                        "NVIDIA GPU device(s).",
                        index,
                        device_count,
                    )
                    self._unavailable_gpus.add(index)
                    continue

                try:
                    self._gpu_handles[index] = pynvml.nvmlDeviceGetHandleByIndex(index)
                except Exception:
                    logger.exception(
                        "Unable to initialize monitoring for GPU %s",
                        index,
                    )
                    self._unavailable_gpus.add(index)

        except Exception:
            logger.warning(
                "NVIDIA GPU monitoring is unavailable",
                exc_info=True,
            )

            self._pynvml = None
            self._unavailable_gpus.update(self.gpu_indices)

    def _get_stack(self) -> list:
        """
        Document this function.

        Returns
        -------
        list
            Description not yet provided.
        """
        thread_id = threading.get_ident()

        with self._lock:
            return self._thread_stacks.setdefault(thread_id, [])

    def _remove_empty_stack(self) -> None:
        """
        Document this function.
        """
        thread_id = threading.get_ident()

        with self._lock:
            stack = self._thread_stacks.get(thread_id)

            if stack == []:
                self._thread_stacks.pop(thread_id, None)

    def current_stage(self) -> str:
        """
        Document this function.

        Returns
        -------
        str
            Description not yet provided.
        """
        stack = self._get_stack()
        return stack[-1] if stack else "root"

    def _current_sample_stage(self) -> str:
        """
        Document this function.

        Returns
        -------
        str
            Description not yet provided.
        """
        sampler_thread_id = self._thread.ident if self._thread is not None else None

        with self._lock:
            stages = [
                stack[-1]
                for thread_id, stack in self._thread_stacks.items()
                if thread_id != sampler_thread_id and stack
            ]

        if not stages:
            return "root"

        return " | ".join(sorted(set(stages)))

    def _ram_gb(self) -> float:
        """
        Document this function.

        Returns
        -------
        float
            Description not yet provided.
        """
        return self._process.memory_info().rss / (1024**3)

    def _cpu(self) -> float:
        """
        Document this function.

        Returns
        -------
        float
            Description not yet provided.
        """
        return float(self._process.cpu_percent(interval=None))

    def _gpu(self, index: int) -> tuple[float, float]:
        """
        Document this function.

        Parameters
        ----------
        index : int
            Description not yet provided.

        Returns
        -------
        tuple[float, float]
            Description not yet provided.
        """
        if (
            self._pynvml is None
            or index in self._unavailable_gpus
            or index not in self._gpu_handles
        ):
            return float("nan"), float("nan")

        try:
            handle = self._gpu_handles[index]

            utilization = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = self._pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpu_utilization = float(utilization.gpu)

            vram_utilization = (
                float(memory.used / memory.total * 100) if memory.total else 0.0
            )

            return gpu_utilization, vram_utilization

        except Exception:
            if index not in self._unavailable_gpus:
                logger.exception(
                    "Unable to collect utilization for GPU %s",
                    index,
                )
                self._unavailable_gpus.add(index)

            return float("nan"), float("nan")

    def _create_event(
        self,
        *,
        stage: str,
        event: str,
        span_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Document this function.

        Parameters
        ----------
        stage : str
            Description not yet provided.
        event : str
            Description not yet provided.
        span_id : str | None
            Description not yet provided.

        Returns
        -------
        dict[str, Any]
            Description not yet provided.
        """
        record: dict[str, Any] = {
            "t": time.time(),
            "stage": stage,
            "event": event,
            "span_id": span_id,
        }

        if self.cpu_enabled:
            record["cpu"] = None

        if self.ram_enabled:
            record["ram"] = None

        for index in self.gpu_indices:
            record[f"gpu{index}"] = None
            record[f"vram{index}"] = None

        return record

    def _append_event(self, event: dict[str, Any]) -> None:
        """
        Document this function.

        Parameters
        ----------
        event : dict[str, Any]
            Description not yet provided.
        """
        with self._lock:
            self._data.append(event)

    @contextmanager
    def span(self, name: str) -> Iterator:
        """
        Document this function.

        Parameters
        ----------
        name : str
            Description not yet provided.

        Yields
        ------
        Iterator
            Description not yet provided.

        Raises
        ------
        TypeError
            Description not yet provided.
        ValueError
            Description not yet provided.
        """
        if not isinstance(name, str):
            raise TypeError("span name must be a string")

        name = name.strip()

        if not name:
            raise ValueError("span name cannot be empty")

        stack = self._get_stack()

        with self._lock:
            parent = stack[-1] if stack else None
            full_name = f"{parent}.{name}" if parent else f"main.{name}"

            span_id = uuid.uuid4().hex
            stack.append(full_name)

            self._data.append(
                self._create_event(
                    stage=full_name,
                    event="start",
                    span_id=span_id,
                )
            )

        try:
            yield

        finally:
            self._append_event(
                self._create_event(
                    stage=full_name,
                    event="end",
                    span_id=span_id,
                )
            )

            with self._lock:
                if stack and stack[-1] == full_name:
                    stack.pop()
                else:
                    logger.warning(
                        "Monitoring span stack was modified while span %s was active",
                        full_name,
                    )

                    if full_name in stack:
                        stack.remove(full_name)

            self._remove_empty_stack()

    def observe(self, function: F) -> F:
        """
        Document this function.

        Parameters
        ----------
        function : F
            Description not yet provided.

        Returns
        -------
        F
            Description not yet provided.

        Raises
        ------
        TypeError
            Description not yet provided.
        """
        if not callable(function):
            raise TypeError("observe expects a callable")

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            """
            Document this function.

            Parameters
            ----------
            *args : Any
                Description not yet provided.
            **kwargs : Any
                Description not yet provided.

            Returns
            -------
            Any
                Description not yet provided.
            """
            with self.span(function.__name__):
                return function(*args, **kwargs)

        return cast(F, wrapper)

    def checkpoint(self, name: str) -> None:
        """
        Document this function.

        Parameters
        ----------
        name : str
            Description not yet provided.

        Raises
        ------
        TypeError
            Description not yet provided.
        ValueError
            Description not yet provided.
        """
        if not isinstance(name, str):
            raise TypeError("checkpoint name must be a string")

        name = name.strip()

        if not name:
            raise ValueError("checkpoint name cannot be empty")

        parent = self.current_stage()

        full_name = f"main.{name}" if parent == "root" else f"{parent}.{name}"

        self._append_event(
            self._create_event(
                stage=full_name,
                event="checkpoint",
            )
        )

    def _collect_sample(self) -> dict[str, Any]:
        """
        Document this function.

        Returns
        -------
        dict[str, Any]
            Description not yet provided.
        """
        sample = self._create_event(
            stage=self._current_sample_stage(),
            event="sample",
        )

        if self.cpu_enabled:
            sample["cpu"] = self._cpu()

        if self.ram_enabled:
            sample["ram"] = self._ram_gb()

        for index in self.gpu_indices:
            gpu_utilization, vram_utilization = self._gpu(index)

            sample[f"gpu{index}"] = gpu_utilization
            sample[f"vram{index}"] = vram_utilization

        return sample

    def _sampler(self) -> None:
        """
        Document this function.
        """
        if self.cpu_enabled:
            self._process.cpu_percent(interval=None)

        while not self._stop_event.is_set():
            sample_started = time.monotonic()

            try:
                self._append_event(self._collect_sample())
            except Exception:
                logger.exception("Unexpected error while collecting monitoring data")

            elapsed = time.monotonic() - sample_started
            wait_time = max(0.0, self.interval - elapsed)

            self._stop_event.wait(wait_time)

    def start(self, *, clear: bool = False) -> None:
        """
        Document this function.

        Parameters
        ----------
        clear : bool
            Description not yet provided.
        """
        if self.running:
            logger.warning("Monitoring is already running")
            return

        if clear:
            self.clear()

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._sampler,
            name="resource-monitor",
            daemon=True,
        )
        self._thread.start()

        logger.info(
            "Monitoring started: cpu=%s, ram=%s, gpus=%s, interval=%s",
            self.cpu_enabled,
            self.ram_enabled,
            list(self.gpu_indices),
            self.interval,
        )

    def stop(self, timeout: float | None = None) -> None:
        """
        Document this function.

        Parameters
        ----------
        timeout : float | None
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative")

        thread = self._thread

        if thread is None:
            return

        self._stop_event.set()
        thread.join(timeout=timeout)

        if thread.is_alive():
            logger.warning("Monitoring thread did not stop within the timeout")
            return

        self._thread = None

        logger.info("Monitoring stopped")

    @property
    def running(self) -> bool:
        """
        Document this function.

        Returns
        -------
        bool
            Description not yet provided.
        """
        return self._thread is not None and self._thread.is_alive()

    @property
    def metric_names(self) -> tuple[str, ...]:
        """
        Document this function.

        Returns
        -------
        tuple[str, ...]
            Description not yet provided.
        """
        metrics: list[str] = []

        if self.cpu_enabled:
            metrics.append("cpu")

        if self.ram_enabled:
            metrics.append("ram")

        for index in self.gpu_indices:
            metrics.append(f"gpu{index}")
            metrics.append(f"vram{index}")

        return tuple(metrics)

    def clear(self) -> None:
        """
        Document this function.
        """
        with self._lock:
            self._data.clear()

    def get_dataframe(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        import pandas as pd

        with self._lock:
            records = [event.copy() for event in self._data]

        return pd.DataFrame(records)

    def __enter__(self) -> "Monitor":
        """
        Document this function.

        Returns
        -------
        'Monitor'
            Description not yet provided.
        """
        self.start()
        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> bool:
        """
        Document this function.

        Parameters
        ----------
        exception_type : Any
            Description not yet provided.
        exception_value : Any
            Description not yet provided.
        traceback : Any
            Description not yet provided.

        Returns
        -------
        bool
            Description not yet provided.
        """
        self.stop()
        return False

    @staticmethod
    def _kalman_filter(
        values,
        process_variance: float = 1.0,
        measurement_variance: float = 25.0,
    ):
        """
        Document this function.

        Parameters
        ----------
        values : Any
            Description not yet provided.
        process_variance : float
            Description not yet provided.
        measurement_variance : float
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        import numpy as np

        if process_variance < 0:
            raise ValueError("process_variance cannot be negative")

        if measurement_variance < 0:
            raise ValueError("measurement_variance cannot be negative")

        values = np.asarray(values, dtype=float)

        if len(values) == 0:
            return values

        filtered = np.full_like(
            values,
            fill_value=np.nan,
            dtype=float,
        )

        valid_indices = np.flatnonzero(np.isfinite(values))

        if len(valid_indices) == 0:
            return filtered

        first_index = int(valid_indices[0])

        estimate = float(values[first_index])
        estimate_variance = 1.0

        for index in range(first_index, len(values)):
            observation = values[index]

            estimate_variance += process_variance

            if not np.isfinite(observation):
                filtered[index] = estimate
                continue

            denominator = estimate_variance + measurement_variance

            gain = estimate_variance / denominator if denominator else 0.0

            estimate += gain * (observation - estimate)

            estimate_variance = (1.0 - gain) * estimate_variance

            filtered[index] = estimate

        return filtered

    @classmethod
    def _smooth(
        cls,
        series,
        method: str | None = None,
        **kwargs,
    ):
        """
        Document this function.

        Parameters
        ----------
        series : Any
            Description not yet provided.
        method : str | None
            Description not yet provided.
        **kwargs : Any
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if method is None:
            return series

        method = method.lower()

        if method == "ema":
            alpha = float(kwargs.get("alpha", 0.2))

            if not 0 < alpha <= 1:
                raise ValueError(
                    "EMA alpha must be greater than 0 and less than or equal to 1"
                )

            return series.ewm(
                alpha=alpha,
                adjust=False,
            ).mean()

        if method == "rolling":
            window = int(kwargs.get("window", 5))

            if window <= 0:
                raise ValueError("rolling window must be greater than zero")

            return (
                series.rolling(
                    window=window,
                    center=True,
                    min_periods=1,
                )
                .mean()
                .bfill()
                .ffill()
            )

        if method == "kalman":
            return cls._kalman_filter(
                series.to_numpy(),
                process_variance=float(kwargs.get("process_variance", 1.0)),
                measurement_variance=float(
                    kwargs.get(
                        "measurement_variance",
                        25.0,
                    )
                ),
            )

        raise ValueError(f"Unknown smoothing method: {method}")

    @classmethod
    def _plot_metric(
        cls,
        ax,
        t,
        values,
        *,
        label: str,
        color: str,
        smooth: str | None = None,
        **smooth_kwargs,
    ) -> None:
        """
        Document this function.

        Parameters
        ----------
        ax : Any
            Description not yet provided.
        t : Any
            Description not yet provided.
        values : Any
            Description not yet provided.
        label : str
            Description not yet provided.
        color : str
            Description not yet provided.
        smooth : str | None
            Description not yet provided.
        **smooth_kwargs : Any
            Description not yet provided.
        """
        ax.plot(
            t,
            values,
            color=color,
            alpha=0.3,
            linewidth=1,
            label=f"{label} (raw)",
        )

        if smooth is not None:
            smoothed = cls._smooth(
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

        ax.legend(
            loc="upper right",
            fontsize=8,
        )

    @staticmethod
    def _metric_label(metric: str) -> tuple[str, str]:
        """
        Document this function.

        Parameters
        ----------
        metric : str
            Description not yet provided.

        Returns
        -------
        tuple[str, str]
            Description not yet provided.
        """
        if metric == "cpu":
            return "CPU", "%"

        if metric == "ram":
            return "RAM", "GB"

        if metric.startswith("gpu"):
            index = metric.removeprefix("gpu")
            return f"GPU {index}", "%"

        if metric.startswith("vram"):
            index = metric.removeprefix("vram")
            return f"GPU {index} VRAM", "%"

        return metric, ""

    def _available_metrics(self, samples) -> list:
        """
        Document this function.

        Parameters
        ----------
        samples : Any
            Description not yet provided.

        Returns
        -------
        list
            Description not yet provided.
        """
        return [
            metric
            for metric in self.metric_names
            if (metric in samples.columns and samples[metric].notna().any())
        ]

    @staticmethod
    def _span_intervals(df) -> list[tuple[float, float, str]]:
        """
        Document this function.

        Parameters
        ----------
        df : Any
            Description not yet provided.

        Returns
        -------
        list[tuple[float, float, str]]
            Description not yet provided.
        """
        intervals: list[tuple[float, float, str]] = []

        if "span_id" not in df.columns:
            return intervals

        starts = df[(df["event"] == "start") & df["span_id"].notna()]

        ends = df[(df["event"] == "end") & df["span_id"].notna()]

        end_times = {
            str(row["span_id"]): float(row["elapsed"]) for _, row in ends.iterrows()
        }

        for _, start_event in starts.iterrows():
            span_id = str(start_event["span_id"])

            if span_id not in end_times:
                continue

            intervals.append(
                (
                    float(start_event["elapsed"]),
                    end_times[span_id],
                    str(start_event["stage"]),
                )
            )

        return intervals

    def plot(
        self,
        df=None,
        metrics=None,
        save_path=None,
        show=True,
        smooth=None,
        **smooth_kwargs,
    ):
        """
        Document this function.

        Parameters
        ----------
        df : Any
            Description not yet provided.
        metrics : Any
            Description not yet provided.
        save_path : Any
            Description not yet provided.
        show : Any
            Description not yet provided.
        smooth : Any
            Description not yet provided.
        **smooth_kwargs : Any
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        TypeError
            Description not yet provided.
        ValueError
            Description not yet provided.
        """
        import matplotlib.pyplot as plt

        if df is None:
            df = self.get_dataframe()
        else:
            df = df.copy()

        if df.empty:
            logger.warning("There are no monitoring events to plot")
            return None

        required_columns = {"t", "event", "stage"}
        missing_columns = required_columns.difference(df.columns)

        if missing_columns:
            raise ValueError(
                "Monitoring dataframe is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        initial_time = float(df["t"].min())
        df["elapsed"] = df["t"] - initial_time

        samples = df[df["event"] == "sample"].copy()

        if samples.empty:
            logger.warning("There are no resource samples to plot")
            return None

        if isinstance(metrics, str):
            raise TypeError(
                "metrics must be an iterable of metric names, not a single string"
            )

        if metrics is None:
            selected_metrics = self._available_metrics(samples)
        else:
            selected_metrics = list(dict.fromkeys(metrics))

        unknown_metrics = [
            metric for metric in selected_metrics if metric not in samples.columns
        ]

        if unknown_metrics:
            raise ValueError(
                "The following metrics were not collected: "
                + ", ".join(unknown_metrics)
            )

        selected_metrics = [
            metric for metric in selected_metrics if samples[metric].notna().any()
        ]

        if not selected_metrics:
            logger.warning("None of the selected metrics contain data")
            return None

        span_intervals = self._span_intervals(df)

        checkpoints = df[df["event"] == "checkpoint"].copy()

        timeline_events = []

        for start, end, stage in span_intervals:
            timeline_events.append(
                {
                    "kind": "span",
                    "stage": stage,
                    "start": start,
                    "end": end,
                }
            )

        for _, checkpoint_event in checkpoints.iterrows():
            timeline_events.append(
                {
                    "kind": "checkpoint",
                    "stage": str(checkpoint_event["stage"]),
                    "start": float(checkpoint_event["elapsed"]),
                    "end": float(checkpoint_event["elapsed"]),
                }
            )

        timeline_events.sort(key=lambda event: event["start"])

        timeline_height = max(
            2.0,
            min(6.0, 0.32 * len(timeline_events) + 0.8),
        )

        metric_height = 2.7 * len(selected_metrics)

        figure = plt.figure(figsize=(16, timeline_height + metric_height))

        grid = figure.add_gridspec(
            nrows=len(selected_metrics) + 1,
            ncols=1,
            height_ratios=[
                timeline_height,
                *([2.7] * len(selected_metrics)),
            ],
            hspace=0.08,
        )

        timeline_axis = figure.add_subplot(grid[0, 0])

        axes = []

        for position in range(len(selected_metrics)):
            if position == 0:
                axis = figure.add_subplot(grid[position + 1, 0])
            else:
                axis = figure.add_subplot(
                    grid[position + 1, 0],
                    sharex=axes[0],
                )

            axes.append(axis)

        colors = (
            "tab:red",
            "tab:blue",
            "tab:green",
            "tab:orange",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        )

        for position, metric in enumerate(selected_metrics):
            label, unit = self._metric_label(metric)
            axis = axes[position]

            self._plot_metric(
                axis,
                samples["elapsed"],
                samples[metric],
                label=label,
                color=colors[position % len(colors)],
                smooth=smooth,
                **smooth_kwargs,
            )

            ylabel = f"{label} ({unit})" if unit else label

            axis.set_ylabel(ylabel)
            axis.grid(
                visible=True,
                axis="both",
                alpha=0.2,
            )

            if position < len(axes) - 1:
                axis.tick_params(
                    axis="x",
                    labelbottom=False,
                )

        axes[-1].set_xlabel("Time (s)")

        timeline_labels = []
        timeline_positions = []

        for position, event in enumerate(timeline_events):
            stage = event["stage"]

            if stage.startswith("main."):
                stage = stage[5:]

            stage = stage.replace("_", " ")

            timeline_labels.append(stage)
            timeline_positions.append(position)

            if event["kind"] == "span":
                start = event["start"]
                end = event["end"]
                duration = max(end - start, 0.001)

                timeline_axis.broken_barh(
                    [(start, duration)],
                    (
                        position - 0.35,
                        0.7,
                    ),
                    facecolors="tab:blue",
                    edgecolors="tab:blue",
                    alpha=0.45,
                    linewidth=0.8,
                )

                for axis in axes:
                    axis.axvspan(
                        start,
                        end,
                        color="tab:blue",
                        alpha=0.035,
                    )

            else:
                checkpoint_time = event["start"]

                timeline_axis.scatter(
                    checkpoint_time,
                    position,
                    marker="D",
                    color="tab:orange",
                    edgecolor="black",
                    linewidth=0.5,
                    s=35,
                    zorder=3,
                )

                for axis in axes:
                    axis.axvline(
                        checkpoint_time,
                        color="tab:orange",
                        linestyle="--",
                        linewidth=1,
                        alpha=0.5,
                    )

        if timeline_events:
            timeline_axis.set_yticks(timeline_positions)
            timeline_axis.set_yticklabels(
                timeline_labels,
                fontsize=8,
            )

            timeline_axis.set_ylim(
                len(timeline_events) - 0.4,
                -0.6,
            )
        else:
            timeline_axis.set_yticks([])
            timeline_axis.text(
                0.5,
                0.5,
                "No execution stages recorded",
                horizontalalignment="center",
                verticalalignment="center",
                transform=timeline_axis.transAxes,
            )

        timeline_axis.set_ylabel("Stage")
        timeline_axis.set_title(
            "Execution timeline",
            loc="left",
            fontsize=11,
        )
        timeline_axis.grid(
            visible=True,
            axis="x",
            alpha=0.2,
        )
        timeline_axis.tick_params(
            axis="x",
            labelbottom=False,
        )

        maximum_time = float(df["elapsed"].max())

        timeline_axis.set_xlim(
            0.0,
            maximum_time,
        )

        for axis in axes:
            axis.set_xlim(
                0.0,
                maximum_time,
            )

        figure.align_ylabels([timeline_axis, *axes])

        figure.subplots_adjust(
            left=0.22,
            right=0.98,
            top=0.97,
            bottom=0.08,
        )

        if save_path:
            save_path = os.fspath(save_path)
            directory = os.path.dirname(save_path)

            if directory:
                os.makedirs(
                    directory,
                    exist_ok=True,
                )

            figure.savefig(
                save_path,
                dpi=200,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        return figure


def monitor(
    *,
    cpu: bool = True,
    ram: bool = True,
    gpu0: bool = False,
    gpu1: bool = False,
    gpus: Iterable[int] | None = None,
    interval: float = 0.1,
) -> Monitor:
    """
    Document this function.

    Parameters
    ----------
    cpu : bool
        Description not yet provided.
    ram : bool
        Description not yet provided.
    gpu0 : bool
        Description not yet provided.
    gpu1 : bool
        Description not yet provided.
    gpus : Iterable[int] | None
        Description not yet provided.
    interval : float
        Description not yet provided.

    Returns
    -------
    Monitor
        Description not yet provided.
    """
    return Monitor(
        cpu=cpu,
        ram=ram,
        gpu0=gpu0,
        gpu1=gpu1,
        gpus=gpus,
        interval=interval,
    )

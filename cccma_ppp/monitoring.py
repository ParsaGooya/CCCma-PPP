from __future__ import annotations

import os
import time
from contextlib import contextmanager
import psutil
import pynvml
import logging


logger = logging.getLogger(__name__)


_PROCESS = psutil.Process(os.getpid())

pynvml.nvmlInit()
_GPU_AVAILABLE = True


def memory_gb() -> float:
    if _PROCESS is None:
        return -1.0
    return _PROCESS.memory_info().rss / (1024**3)


def cpu_percent() -> float:
    if _PROCESS is None:
        return -1.0
    return _PROCESS.cpu_percent(interval=None)


def gpu_usage():
    if not _GPU_AVAILABLE:
        return -1.0, -1.0

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

    gpu_util = util.gpu
    gpu_mem = (mem.used / mem.total) * 100 if mem.total else -1.0

    return gpu_util, gpu_mem


@contextmanager
def profile_stage(name: str):
    start_t = time.perf_counter()
    start_mem = memory_gb()
    start_cpu = cpu_percent()
    start_gpu_util, start_gpu_mem = gpu_usage()

    logger.info(
        "[START] %s | cpu=%.1f%% | mem=%.2f GB | gpu=%.1f%% | vram=%.1f%%",
        name,
        start_cpu,
        start_mem,
        start_gpu_util,
        start_gpu_mem,
    )

    try:
        yield
    except Exception:
        end_t = time.perf_counter()
        end_mem = memory_gb()
        end_cpu = cpu_percent()
        end_gpu_util, end_gpu_mem = gpu_usage()

        logger.exception(
            "[FAILED] %s | time=%.2f sec | cpu=%.1f%% | mem=%.2f -> %.2f GB | gpu=%.1f%% | vram=%.1f%%",
            name,
            end_t - start_t,
            start_cpu,
            start_mem,
            end_mem,
            end_gpu_util,
            end_gpu_mem,
        )
        raise
    else:
        end_t = time.perf_counter()
        end_mem = memory_gb()
        end_cpu = cpu_percent()
        end_gpu_util, end_gpu_mem = gpu_usage()

        logger.info(
            "[END] %s | time=%.2f sec | cpu=%.1f%% | mem=%.2f -> %.2f GB | gpu=%.1f%% | vram=%.1f%%",
            name,
            end_t - start_t,
            end_cpu,
            start_mem,
            end_mem,
            end_gpu_util,
            end_gpu_mem,
        )

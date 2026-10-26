Generic
=======

The ``cccma_ppp.generic`` package holds cross-cutting utilities used by training and inference: distributed setup, logging, metrics, runtime state, resource monitoring, and import registration.

Modules
-------

``distributed.py``
    ``Distributed`` wraps torch distributed setup. It reads ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE`` from the environment. It falls back to a single process when they are absent, and provides ``barrier``, ``all_reduce_sum``, ``broadcast``, and ``is_root``. Use ``Distributed.get_instance()`` for a shared object.

``logger.py``
    ``setup_logger`` creates a named logger. Rank 0 logs to stdout and to ``<name>.log`` in the log directory. Other ranks stay silent.

``runtime.py``
    ``RuntimeContext`` holds global directories (experiment, checkpoints, figures, logs, monitoring, output) and the input and target variable metadata. Module-level state is set here at startup and checked when checkpoints are loaded.

``aggregator.py``
    ``MetricsAggregator`` collects batch losses, reduces them across ranks, records per-epoch values, and plots them. ``RunningCovariance`` accumulates a mean and covariance online, for estimating residual and latent statistics.

``monitoring.py``
    ``Monitor`` samples CPU, RAM, and NVIDIA GPU use in a background thread, and records named stages with ``span`` and ``checkpoint``. Results are exported to CSV and a timeline plot. ``distributed_monitoring`` is a context manager for training and combines per-rank results.

``registery_imports.py``
    Imports modules for their registration side effects, so ``@register`` decorators run before configs are parsed.
Core
====

The ``cccma_ppp.core`` package holds the training-agnostic machinery: model wrappers, selectors, the trainer, the optimizer, and the writer that produces predictions.

Modules
-------

``core_abc.py``
    Base classes: ``moduleConfigABC`` (config that builds a module), ``moduleABC`` (wrapper module with loss initialization and checkpoint loading), ``GenerativeContext`` (tells losses whether the model is generative or uses a generator), and ``OutputABC``.

``registery.py``
    ``Registery``, a name-to-class registry. ``get`` builds a class from a dict using ``dacite`` in strict mode.

``selectors.py``
    Entry points that turn YAML into objects:

    - ``ModuleSelector``: picks a module type (``cvae`` or ``deterministic``) and builds it with ``build_module``.
    - ``ModelSelector``, ``cVAEModelSelector``, ``deterministicModelSelector``: pick an architecture by name. If ``load_dir`` is given, the saved config overrides the YAML.
    - ``FlowSelector``: picks a normalizing flow (``maf``, ``realnvp``).
    - ``_load_config_from_checkpoint``: reads the config, shapes, and metadata from a checkpoint.

``models/cvae.py``
    ``cVAEConfig`` and ``cVAE``. Wraps a conditional VAE model with an optional prior flow, posterior variance limits, and a CGCN loss weight. ``forward`` returns ``cVAEOutput``, and ``predict`` samples from the prior.

``models/deterministic.py``
    ``deterministicConfig`` and ``deterministic``. Wraps a deterministic model, with ``forward``, ``predict`` and ``deterministicOutput``.

``trainer.py``
    ``TrainerConfig`` and ``Trainer``. Runs the training loop with gradient accumulation, mixed precision, gradient clipping, validation, early stopping, best-model checkpointing, and resume. Also ``clear_memory``.

``optimization.py``
    ``OptimizerConfig`` builds an Adam or AdamW optimizer wrapped in ``OptimizerWrapper``. ``LRSchedulerConfig`` builds a warmup-then-cosine schedule through ``CosineAnnealingLRScheduler``.

``writer.py``
    ``WriterConfig`` and ``Writer``. Runs inference over a loader, writes per-batch temporary NetCDF files, then aggregates them year by year into final outputs with ``aggregate_predictions``. It can also compute training statistics for ensemble noise.
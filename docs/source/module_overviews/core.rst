Core
====

The ``S2D4P.core`` package holds the training-agnostic machinery: model wrappers, selectors, the trainer, the optimizer, and the writer that produces predictions.
The design separates the modeling framework (e.g. cVAE, deterministic, GAN, etc.) and architecture (e.g. MLP, UNet, transformer, etc.) as model vs. architecture.

Modules
-------

``core_abc.py``
    Base classes: ``moduleConfigABC`` (config that builds a model), ``moduleABC`` (wrapper model with loss initialization and checkpoint loading), ``GenerativeContext`` (tells losses whether the architecture is generative or uses a generator), and ``OutputABC``.

``registery.py``
    ``Registery``, a name-to-class registry. ``get`` builds a class from a dict using ``dacite`` in strict mode.

``selectors.py``
    Entry points that turn YAML into objects:

    - ``ModuleSelector``: picks a model type (``cvae`` or ``deterministic``) and builds it with ``build_module``.
    - ``ArchSelector``, ``cVAEArchSelector``, ``deterministicArchSelector``: pick an architecture by name. If ``load_dir`` is given, the saved config overrides the YAML.
    - ``FlowSelector``: picks a normalizing flow (``maf``, ``realnvp``).
    - ``_load_config_from_checkpoint``: reads the config, shapes, and metadata from a checkpoint.

``models/``
    Each modeling framework is desined as a ``nn.Module`` orchestrator and requrires a corresponding config class thst is registered to a the ``ModuleSelector``.

``models/cvae.py``
    ``cVAEConfig`` and ``cVAE``. Wraps a conditional VAE model with an optional prior flow, posterior variance limits, and a CGCN (combined posterior sampling and prior sampling during training) loss weight. ``forward`` returns ``cVAEOutput``, and ``predict`` samples from the prior.

``models/deterministic.py``
    ``deterministicConfig`` and ``deterministic``. Wraps a deterministic model, with ``forward``, ``predict`` and ``deterministicOutput``.

``trainer.py``
    ``TrainerConfig`` and ``Trainer``. Training orchestrator. Runs the training loop with gradient accumulation, mixed precision, gradient clipping, validation, early stopping, best-model checkpointing, and resume. Also ``clear_memory``.

``optimization.py``
    ``OptimizerConfig`` builds an Adam or AdamW optimizer wrapped in ``OptimizerWrapper``. ``LRSchedulerConfig`` builds a warmup-then-cosine schedule through ``CosineAnnealingLRScheduler``.

``writer.py``
    ``WriterConfig`` and ``Writer``. Inference orchestrator. Runs inference over a loader, writes per-batch temporary NetCDF files, then aggregates them year by year into final outputs with ``aggregate_predictions``. It can also compute training statistics for ensemble noise.

Development
------------------------

1- New models are expected to be added to ``models/``, and will be compatible with ``trainer.py`` orchestrator as long as their class and config class inherit from ``moduleConfigABC`` and ``moduleABC``. 

2- The new model's config class must be registrered with ``ModuleSelector``.

3- The model is an orchestrator that separates the modeling framework from architecture allowing flexibility. For instance, a future GAN model might be defined under ``models/gan.py`` that takes care of orchestrating parts and architecture modules  
specific to GAN model while allowing flexibility for architectural choices for the criric or generator networks. Those could in turn be defined under ``architectures``. An architecture selector specific to that model (e.g. ``GANArchSelector``)
must be created that inherit from ``ArchSelector`` and registers all architectures comptabible with that model type stored in e.g. ``architectures/unet/gan.py``. Finally, each new model requires its own ``predictor`` object
stored in ``inference/predictors``.
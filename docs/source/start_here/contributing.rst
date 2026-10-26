Contributing
============

Thank you for contributing to CCCma-PPP. This guide covers how to set up a development environment, how the package is organized, how to add new components, and what we expect in a pull request.

Development setup
-----------------

Follow the instructions in :doc:`../start_here/installation`

GPU features are optional. ``Distributed`` falls back to a single CPU process when ``RANK`` and ``WORLD_SIZE`` are not set, and ``Monitor`` logs a warning rather than failing when NVML is unavailable.

Project layout
--------------

Each top-level package has one job. See the module overview files in ``docs/`` for details.

- ``architectures``: neural network models and layers.
- ``configs``: project-wide settings loaded from ``configs.yaml``. Change dimension names here, not in code.
- ``core``: model wrappers, selectors, trainer, optimizer, and writer.
- ``data_modules``: data loading, dataset construction, and weights.
- ``generic``: distributed setup, logging, metrics, runtime state, and monitoring.
- ``inference``: prediction entry point, predictors, and output writing.
- ``loss``: loss terms, the loss pipeline, and the KL term.
- ``preprocessing``: reversible data transforms.
- ``train``: the training entry point and config wiring.

Keep dependencies one-directional where possible. ``generic`` and ``configs`` should not import from the higher-level packages.

Adding a component
------------------

Most extensions use a registry. Classes are registered with a decorator and then selected by name from a YAML config. Follow the pattern for the component you are adding.

**Model architecture**

1. Write a config dataclass that subclasses ``cVAEmodelConfigABC`` or ``modelConfigABC``. Set ``NUM_INPUT_DIMS``, ``NUM_OUTPUT_DIMS``, and an ``EXPECTS_MASK`` property. Validate arguments in ``__post_init__``.
2. Implement a ``build`` method that returns the module.
3. Write the module as a subclass of ``deterministicmodelsABC`` or ``cVAEmodelsABC``. Call ``_initialize_weights`` or ``_load_state_dict`` at the end of ``__init__`` depending on whether a checkpoint is provided.
4. Register it with ``@deterministicModelSelector.register("name")`` or ``@cVAEModelSelector.register("name")``.
5. Import the module in ``generic/registry_imports.py`` so the registration runs before configs are parsed.

**Loss term**

1. Subclass ``lossABC`` and implement ``forward``, ``_aggregate``, and ``_print_loss``.
2. Register it with ``@Losspipeline.register("name")``.
3. Accept the reserved arguments (``weights``, ``reduction``, ``num_output_dimensions``, ``generative_context``) explicitly. ``LosspipelineConfig`` rejects user-supplied values for them.

**Preprocessing step**

1. Subclass ``PreprocessModuleABC`` and implement ``fit``, ``transform``, and ``inverse_transform``.
2. Use ``_get_reduction_dims`` and ``_align_stat_for_transform`` rather than reimplementing them.
3. Register it with ``@PreprocessingStepSelector.register("name")``.
4. Make ``inverse_transform`` the exact inverse of ``transform``. Predictions depend on this.

**Normalizing flow**

Subclass ``flowABC`` and register it with ``@FlowSelector.register("name")``. Implement ``forward`` and ``inverse``, both returning a log-determinant, and make ``build`` accept ``dim`` and ``condition_size``.

Coding conventions
------------------

- Use type hints on all public functions and classes. The codebase uses ``X | None`` union syntax.
- Document every public class and function with a NumPy-style docstring following the `NumPy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html>`_. Include the sections that apply: Parameters, Returns, Yields, Raises, Warns, Notes, and Examples. Placeholder text such as "Description not yet provided." must be replaced in any code you touch.
- Put shared constants and dimension names in ``configs``, not in function defaults.
- Raise ``ValueError`` for bad user input and ``RuntimeError`` for inconsistent state, such as a checkpoint that does not match the data. Include the offending values in the message.
- Do not put tensors or large arrays in configs. Configs must stay serializable to YAML, since checkpoints store them.
- Keep distributed-aware code behind ``Distributed``. Use ``is_root()`` before writing files and ``barrier()`` before reading files that rank 0 has written.

Testing
-------

Unit tests are required for every change. Each module must reach at least **90% branch coverage**. Check coverage with branch measurement enabled, for example ``pytest --cov=cccma_ppp --cov-branch --cov-fail-under=90``, and do not merge a change that drops any module below the threshold.

Before opening a pull request, also check that:

- New configs round-trip through ``dacite.from_dict`` with ``strict=True``.
- A new model runs a forward pass and a ``predict`` pass on a small random tensor with the expected output shape.
- A new loss or preprocessing step passes a fit-then-transform-then-inverse check, with the inverse reproducing the input within floating-point tolerance.
- Checkpoint save and reload produce identical outputs.

Add tests alongside the code they cover, and run the full suite with GPU features disabled.

Build and pruning steps
-----------------------

Two steps are required before a pull request is ready for review:

1. **Rebuild the documentation.** Run the docs build script from the repository root after any change to docstrings or to the ``docs/`` files, and confirm it completes without warnings. Include the rebuilt output in your change if the build artifacts are tracked.

2. **Run the pruning pipeline.** Run the pruning pipeline on your branch and commit any changes it makes. Do not hand-edit its output.

Pull requests
-------------

- Keep changes focused. Separate refactors from behavior changes.
- Describe what changed and why, and note any change to saved checkpoint format or ``configs.yaml`` keys. Those break existing experiments and need a mention in the description.
- Update the matching module overview in ``docs/`` if you add a public component.
- Use a clear commit message in the imperative mood (for example, "Add Huber loss term").

Reporting issues
----------------

Include the config file (with paths redacted if needed), the full traceback, the output of ``python -c "import torch; print(torch.__version__)"``, and whether the run was single-process or distributed.
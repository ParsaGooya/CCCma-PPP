Configs
=======

The ``S2D4P.configs`` package loads project-wide settings from ``configs.yaml`` and exposes them as module-level constants. 

``configs/__init__.py``
-----------------------

On import, the module reads ``configs.yaml`` from its own directory and provides:

- ``required_sample_dimensions``: the names for the two sampling dimensions initialization time and lead time.
- ``lead_time_resolution``: ``"day"`` or ``"month"``. Any other value raises ``ValueError``. Supports subseasonal to decadal predictions.
- ``realization_dim``: name of the ensemble member dimension.
- ``supported_NN_dimensions_sorted``: spatial/temporal dimensions the networks accept, in order.
- ``forecast_data_allowed_dimensions``, ``reference_data_allowed_dimensions``, ``condition_data_allowed_dimensions``: allowed dimensions per data type, each merged with the supported NN dimensions.
- ``forecastl_data_required_dimensions``, ``reference_data_required_dimensions``, ``condition_data_required_dimensions``: dimensions each data type must have.
- ``save_deterministic_guess_only``: whether the cVAEUNe architecture saves only the deterministic guess branch.
- ``lead_time_unit``: the ``Literal["day", "month"]`` type alias.

Other modules import these constants instead of hard-coding dimension names, so changing the YAML file changes behavior across the package.

``configs/configs.yaml``
------------------------

The YAML file that supplies the values above. Edit it to change dimension names or the lead-time resolution for a project.

development
------------------------

Configs that need to be accesible project-wide must be added here. For instance, ``save_critic_prediction`` for a later GAN model.
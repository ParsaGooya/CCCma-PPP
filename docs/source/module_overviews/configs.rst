Configs
=======

The ``cccma_ppp.configs`` package loads project-wide settings from ``configs.yaml`` and exposes them as module-level constants.

``configs/__init__.py``
-----------------------

On import, the module reads ``configs.yaml`` from its own directory and provides:

- ``required_sample_dimensions``: the two sampling dimensions (initialization time and lead time).
- ``lead_time_resolution``: ``"day"`` or ``"month"``. Any other value raises ``ValueError``.
- ``realization_dim``: name of the ensemble member dimension.
- ``supported_NN_dimensions_sorted``: spatial/temporal dimensions the networks accept, in order.
- ``model_data_allowed_dimensions``, ``observation_data_allowed_dimensions``, ``condition_data_allowed_dimensions``: allowed dimensions per data type, each merged with the supported NN dimensions.
- ``model_data_required_dimensions``, ``observation_data_required_dimensions``, ``condition_data_required_dimensions``: dimensions each data type must have.
- ``save_deterministic_guess_only``: whether prediction saves only the deterministic guess branch.
- ``lead_time_unit``: the ``Literal["day", "month"]`` type alias.

Other modules import these constants instead of hard-coding dimension names, so changing the YAML file changes behavior across the package.

``configs/configs.yaml``
------------------------

The YAML file that supplies the values above. Edit it to change dimension names or the lead-time resolution for a project.
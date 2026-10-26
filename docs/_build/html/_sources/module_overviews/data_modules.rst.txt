Data Modules
============

The ``cccma_ppp.data_modules`` package reads NetCDF data, aligns model, observation, and conditioning sources in time, and produces batches for the models.

Modules
-------

``utils.py``
    Helpers for loading and masking data:

    - ``_load_xarray_data``: opens files with ``xr.open_mfdataset``, applies renames, realization selection, ensemble averaging, and time auxiliary coordinates (year, month, day of year).
    - ``_create_train_mask``: marks target times that fall after the last available initialization time, so training does not use incomplete targets.
    - ``_calculate_target_times``, ``add_lead_times``, ``_add_months``, ``_add_cftime_offset``: convert initialization times and lead times to valid times, for both ``datetime64`` and ``cftime`` calendars.
    - ``_validate_time_sequence``, ``get_time_representation``, ``infer_time_resolution``: time checks.
    - ``suppress_stderr``: silences noisy library output.

``weights.py``
    ``WeightsConfig`` builds spatial weights (uniform or cosine-latitude) and optional per-variable weights. Weights must be non-negative and finite.

``data/``
    Data source configs:

    - ``data_abc.py``: ``DataConfigABC`` opens files, validates dimensions and variables, and handles preprocessing pipeline fitting and loading.
    - ``data_configs.py``: ``ModelDataConfig``, ``ObsDataConfig``, and ``ConditionDataConfig``. Each declares its allowed and required dimensions. ``build_time_range`` gives the continuous time extent of a source.

``dataset/``
    Sample construction:

    - ``dataset_abc.py``: ``DatasetConfigABC`` combines model, observation, and condition sources, checks they are compatible, and chooses how the condition is used (``ensemble_mean``, ``cross_ensemble``, ``same_member``, or ``static``). ``DatasetABC`` is the PyTorch dataset base. ``AddedTimeFeatures`` adds normalized time, lead time, and cyclic month and day features. ``lead_time_config`` selects lead times.
    - ``operator.py``: ``DatasetOperator`` fits preprocessors across all sources, gathers variable metadata, and computes weights.

``dataloader.py``
    ``Dataloader`` wraps a PyTorch ``DataLoader`` with distributed sampling. ``DataloaderConfigABC`` and ``BatchDataABC`` define the shared config and batch interfaces.
Data Modules
============

The ``S2D4P.data_modules`` package prvides pipeline to read NetCDF data, aligns forecast, reference, and conditioning sources in time, index them and produces batches for the models.

Modules
-------

``utils.py``
    Helpers for loading and masking data:

    - ``_load_xarray_data``: (lazily) opens files with ``xr.open_mfdataset``, applies renames, realization selection, ensemble averaging, and time auxiliary coordinates (year, month, day of year).
    - ``_create_train_mask``: masks target times that fall after the first test sample's initialization time, so training does not get leakage from targets in test period.
    - ``_calculate_target_times``, ``add_lead_times``, ``_add_months``, ``_add_cftime_offset``: convert initialization times and lead times to valid times, for both ``datetime64`` and ``cftime`` calendars.
    - ``_validate_time_sequence``, ``get_time_representation``, ``infer_time_resolution``: time checks.
    - ``suppress_stderr``: silences noisy library output.

``weights.py``
    ``WeightsConfig`` builds spatial weights (uniform or cosine-latitude) and optional per-variable weights. Weights must be non-negative and finite.

``data/``
    Data source configs:

    - ``data_configs_abc.py``: ``DataConfigABC`` supports functionalities to opens files, validates dimensions and variables, and handles preprocessing pipeline fitting and loading reagardless of data type/source.
    - ``data_configs.py``: ``ForecastDataConfig``, ``ReferenceDataConfig``, and ``ConditionDataConfig``. Each declares its allowed and required dimensions. ``build_time_range`` gives the continuous time extent of a source. Inherit from ``DataConfigABC``.

``dataset/``
    Sample construction (indexing):

    - ``dataset_abc.py``: ``DatasetConfigABC`` ABS for combining (initial-time, lead-time) based forecast and condition sources, checking they are compatible, and choosing how the condition is used (``ensemble_mean``, ``cross_ensemble``, ``same_member``, or ``static``). ``DatasetABC`` is the PyTorch dataset base. ``AddedTimeFeatures`` adds normalized time, lead time, and cyclic month and day features. ``lead_time_config`` selects lead times.
    - ``operator.py``: ``DatasetOperator`` fits preprocessors across all sources, gathers variable metadata, and computes weights.

``dataloader.py``
    ``Dataloader`` wraps a PyTorch ``DataLoader`` with distributed sampling. ``DataloaderConfigABC`` and ``BatchDataABC`` define the shared config and batch interfaces compatable with the pipeline.

development
------------------------

1- Wrappers for configuring data sourced from other than existing forecast, reference and condition sources defined in ``data/data_configs.py`` must be added there and will be compatable with the data pipeline as long as they inhert from  ``data/data_configs_abc.py``.

2- New dataset classes and their config classes that index forecasts and conditons differently can added to the pipeline as long as they inherit from ``DatasetABC`` (e.g. adding new datasets supporting sequential indexing ). Dataset objects themselves and their config classes must be defined under ``train`` or ``inference`` libraries depending on the 
combination of sources their indexing must support (e.g. inference does not need a target).
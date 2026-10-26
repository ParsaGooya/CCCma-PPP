Preprocessing
=============

The ``cccma_ppp.preprocessing`` package transforms data before training and reverses the transforms on predictions. Steps are fitted on training data only, then saved and reloaded for inference.

Modules
-------

``preprocessing.py``
    ``PreprocessingPipeline``: an ordered list of steps. ``fit`` fits each step on the output of the previous one and saves the pipeline with joblib. ``transform`` and ``inverse_transform`` apply the steps forward or in reverse. ``to_dataset`` and ``inverse_rename`` restore the xarray structure of predictions. ``load_from_memory`` reloads a fitted pipeline.

``selector.py``
    ``PreprocessingStepSelector``: picks a step by registered name and arguments, and builds it.

``registery.py``
    The step registry, used by the selector.

``preprocessing_ABC.py``
    ``PreprocessModuleABC``: base class for steps. It handles reduction dimensions, the optional grouping coordinate (year, month, or day), aligning fitted statistics to data times, inverse-time computation for lead times, and zero-range checks.

``utils_preprocessing.py``
    Registered steps:

    - ``Normalizer`` (``"normalizer"``): min-max scaling.
    - ``Standardizer`` (``"standardizer"``): z-score scaling. Zero standard deviations are masked.
    - ``AnomaliesScaler`` (``"anomalies"``): subtracts the mean.
    - ``TrendRemover`` (``"trend_remover"``): removes a linear trend over time, globally or per month or day.
    - ``Flattennanremove`` (``"flattener"``): stacks spatial dimensions into one ``ref`` dimension and drops points with missing values. Required for 2D (flattened) network inputs.
    - ``align_stat_data_lead_time_inverse_transform``: aligns stored statistics with predicted valid times when inverting.
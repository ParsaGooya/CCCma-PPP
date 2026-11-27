Inference
=========

The ``S2D4P.inference`` package runs a trained model on new data and writes NetCDF predictions. It is run from the command line with ``python -m S2D4P.inference config.yaml``. At runtime ``dataloader.py`` supports loading
inference ready batches of data based on indexing logic from ``dataset.py``. The core of inference pipeline are the ``predictors/`` with one class for every model supporting model-specific behaviour. The pipeline creates
``Writer`` from ``core/writer.py`` that orchestrates inference using the above objects.

Modules
-------

``predict.py``
    Entry point. ``get_parser`` defines the CLI, and ``main`` loads the config, builds the writer, runs prediction, and cleans up distributed state.

``inference_configs.py``
    ``InferenceConfig`` reads the experiment's saved ``config.yaml`` and checks that the inference data's variables, preprocessing, and time features match training. It also provides ``load_module`` (restores the checkpoint, ``best.pt`` by default) and ``build_writer``, which assembles the writer, data loader, and post-processor.

``dataset.py``
    ``InferenceDatasetConfig`` and ``InferenceDataset``. Build pipeline for inference samples over requested initialization times, with conditioning handled as in training. ``_from_train`` derives the inference dataset configs from the training configs when a separate dataset config is not provided.

``dataloader.py``
    ``InferenceDataloaderConfig`` and ``BatchData``. Select inference years, set up preprocessors for distributed runs, and build a loader. ``collate_batch`` stacks samples and attaches metadata.

``predictors/``
    Predictors runs the models in inference mode to turn a batch into predictions. Each model might have a different protocol, e.g., cVAE needs to sample the latent space. Future models must define their own predictors.

    - ``predictor_abc.py``: ``PredictorABC`` handles output sampling, adds decoder noise for ensembles, and saves batches with ``save_batch_to_netcdf``.
    - ``deterministic.py``: ``DeterministicPredictorConfig`` and its predictor.
    - ``cvae.py``: ``cVAEPredictorConfig`` and its predictor. Supports latent samples (drawn from the prior or inferred from training statistics) and saving latent variables.

Output
------

Predictions are written first as per-batch temporary files in ``_temp``, then combined by year into files named ``<source>_<year>[_nstds<n>][_output_ensemble].nc`` under the output directory. Post-processing inverts the preprocessing steps used in training. The temporary files are removed afterwards.


Development
------------------------

New models require their own predictors defined under ``predictors/`` (e.g.  ``predictors/gan.py``) and will be compatible with the ``core/writer.py`` orchestrator as long as they inherit from ``predictor_abc.py``.
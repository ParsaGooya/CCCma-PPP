Inference
=========

The ``cccma_ppp.inference`` package runs a trained model on new data and writes NetCDF predictions. It is run from the command line with ``python -m cccma_ppp.inference config.yaml``.

Modules
-------

``predict.py``
    Entry point. ``get_parser`` defines the CLI, and ``main`` loads the config, builds the writer, runs prediction, and cleans up distributed state.

``inference_configs.py``
    ``InferenceConfig`` reads the experiment's saved ``config.yaml`` and checks that the inference data's variables, preprocessing, and time features match training. It also provides ``load_module`` (restores the checkpoint, ``best.pt`` by default) and ``build_writer``, which assembles the writer, data loader, and post-processor.

``dataset.py``
    ``InferenceDatasetConfig`` and ``InferenceDataset``. Build inference samples over requested initialization times, with conditioning handled as in training. ``_from_train`` derives the inference config from the training config.

``dataloader.py``
    ``InferenceDataloaderConfig`` and ``BatchData``. Select inference years, set up preprocessors for distributed runs, and build a loader. ``collate_batch`` stacks samples and attaches metadata.

``predictors/``
    Predictors turn a batch into predictions:

    - ``predictor_abc.py``: ``PredictorABC`` handles output sampling, adds decoder noise for ensembles, and saves batches with ``save_batch_to_netcdf``.
    - ``deterministic.py``: ``DeterministicPredictorConfig`` and its predictor.
    - ``cvae.py``: ``cVAEPredictorConfig`` and its predictor. Supports latent samples (drawn from the prior or inferred from training statistics) and saving latent variables.

Output
------

Predictions are written first as per-batch temporary files in ``_temp``, then combined by year into files named ``<source>_<year>[_nstds<n>][_output_ensemble].nc`` under the output directory. Post-processing inverts the preprocessing steps used in training. The temporary files are removed afterwards.
Train
=====

The ``cccma_ppp.train`` package wires the other packages together for training. It is run from the command line with ``python -m cccma_ppp.train config.yaml``.

Modules
-------

``train.py``
    Entry point. ``get_parser`` defines the CLI. ``main`` loads the YAML config, sets the seed, prepares directories, and builds the trainer inside a resource-monitoring context. It then calls ``setup_distributed`` and ``train``.

``train_configs.py``
    ``TrainConfig`` is the top-level config. It holds the experiment directory, maximum epochs, and the sections for the data loader, module, loss pipeline, trainer, optimizer, and weights. It validates that the pieces are compatible: generator models need CRPS, deterministic models need an observation target, cVAE models need a beta schedule, and input and output dimensions match the model. It also supports resuming from a halted experiment. ``build_trainer`` constructs every component in order. ``set_seed`` sets NumPy and torch seeds.

``dataloader.py``
    ``TrainDataloaderConfig`` splits available times into training and validation sets, fits preprocessors on the rank 0 process, and builds training and validation loaders. ``BatchData`` carries input, target, added features, and optional validity masks.

``dataset.py``
    ``TrainDatasetConfig`` and ``TrainDataset``. Pair model inputs with observation targets, or with the model itself for autoencoding, and build samples using the condition method.

Typical flow
------------

1. Load the YAML config into ``TrainConfig``.
2. Fit preprocessors on training times (rank 0), then load them on all ranks.
3. Build the training and validation loaders and the spatial weights.
4. Build the module, loss pipeline, optimizer, and trainer.
5. Train, saving the best checkpoint and metric plots to the experiment directory.
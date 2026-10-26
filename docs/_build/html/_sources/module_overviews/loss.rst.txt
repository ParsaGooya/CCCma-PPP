Loss
====

The ``cccma_ppp.loss`` package defines the training objectives. Losses are chained into a pipeline with per-term weights, and a KL divergence term is used for the cVAE.

Modules
-------

``loss.py``
    - ``LossStepConfig``: one loss term, given by registered name and arguments.
    - ``LosspipelineConfig``: a list of loss steps, their weights (which must sum to 1), reduction (``mean`` or ``sum``), and an optional saved output mask (``.pt`` or ``.nc``). ``build`` returns a ``Losspipeline``.
    - ``Losspipeline``: an ``nn.Module`` that runs each term, applies the output mask, and returns the weighted total and per-term values.

``loss_abc.py``
    ``lossABC``, the abstract base for loss terms, and the ``Reduction`` type.

``registery.py``
    A registry of loss classes, populated by ``@Losspipeline.register("name")``.

``utils_loss.py``
    Registered loss terms:

    - ``WeightedMSE`` (``"mse"``): weighted mean squared error, with optional thresholds and low-resolution pooling.
    - ``WeightedCRPS`` (``"crps"``): continuous ranked probability score for ensemble generators, computed from sorted samples.
    - ``Frobenius_norm`` (``"frobenius_norm"``): distance between predicted and true covariance matrices, by spatial or channel dimension.

``kld.py``
    - ``KLD``: KL divergence between the approximate posterior and the prior. Uses a standard normal prior, a conditional Gaussian prior, or a normalizing flow prior, estimated by sampling.
    - ``BetaAnnealing``: schedules the KL weight beta over training, with warmup and optional hold cycles.
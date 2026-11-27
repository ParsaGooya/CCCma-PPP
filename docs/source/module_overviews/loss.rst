Loss
====

The ``S2D4P.loss`` package defines the training objectives. Losses are chained into a pipeline with per-term weights using ``Losspipeline`` orchestrator. KL divergence term is used for the cVAE separately.

Modules
-------

``loss.py``
    - ``LossStepConfig``: Configs for one loss term, given by registered name and arguments are passed with this object.
    - ``LosspipelineConfig``: a list of loss steps, their weights (which must sum to 1), reduction (``mean`` or ``sum``), and an optional saved output mask (``.pt`` or ``.nc``). ``build`` returns a ``Losspipeline``. Registers individual loss terms.
    - ``Losspipeline``: an ``nn.Module`` that orchestrates running each term, applying output mask, and returning the weighted total and per-term values.

``loss_abc.py``
    ``lossABC``, the abstract base for loss terms, and the ``Reduction`` type.

``registery.py``
    A registry of loss classes, populated by ``@Losspipeline.register("name")``.

``utils_loss.py``
    Registered individual loss terms:

    - ``WeightedMSE`` (``"mse"``): weighted mean squared error, with optional thresholds and low-resolution pooling.
    - ``WeightedCRPS`` (``"crps"``): continuous ranked probability score for ensemble generators, computed from sorted samples.
    - ``Frobenius_norm`` (``"frobenius_norm"``): distance between predicted and true covariance matrices, by spatial or channel dimension.

``kld.py``
    - ``KLD``: KL divergence between the approximate posterior and the prior. Uses a standard normal prior, a conditional Gaussian prior, or a normalizing flow prior, estimated by sampling.
    - ``BetaAnnealing``: schedules the KL weight beta over training, with warmup and optional hold cycles.

Development
------------------------

New loss functions can be added to ``utils_loss.py`` and registered with ``Losspipeline`` and will be compatible with the ``loss.py`` orchestrator as long as they inherit from ``loss_abc.py``. 
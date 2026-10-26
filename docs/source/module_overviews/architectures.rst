Architectures
=============

The ``cccma_ppp.architectures`` package defines the neural network models and their building blocks. Each model has a config dataclass (subclassing ``modelConfigABC``) that validates settings and builds the model, plus a ``torch.nn.Module`` implementation.

Subpackages and modules
-----------------------

``models_abc.py``
    Abstract base classes and shared request/config types:

    - ``modelConfigABC`` and ``modelABC``: base classes for all configs and models. Provide checkpoint loading, weight initialization, and device lookup.
    - ``deterministicmodelsABC`` and ``cVAEmodelsABC``: base classes for deterministic and conditional VAE models.
    - ``DeterministicRequest``, ``cVAEForwardRequest``, ``cVAEPredictRequest``: typed request objects passed to ``forward``/``predict``.
    - ``GENERATORConfig``: settings for the ensemble generator (noise samples, noise level).
    - ``CheckpointConfig``: metadata for loading a saved model (shapes, variable metadata, freeze flag).
    - ``flowABC``: abstract base for normalizing flows.
    - ``weights_init``: applies kaiming, xavier, or truncated-normal initialization.

``normalized_flows.py``
    Normalizing flows used as learned priors in the cVAE:

    - ``NormalizedFlowConfig`` builds a ``NormalizedFlowModel`` from a list of flow steps.
    - ``MAF`` (masked autoregressive flow) and ``RealNVP`` (affine coupling) are registered flow types.
    - ``FCNN`` is the small base network used inside each flow step.

``layers/``
    Reusable building blocks:

    - ``generic.py``: type aliases (activation, normalization, padding, init methods) and helpers ``_build_activation``, ``_build_normalization``, ``LayerNorm2d``, ``DropPath``.
    - ``conv.py``: block configs (``ConvBlockConfig``, ``PartialConvBlockConfig``, ``ConvNeXtBlockConfig``) and block modules (``ConvBlock``, ``PartialConvBlock``, ``ConvNeXtBlock``), plus ``LatentBlock``/``LatentLayer`` for the bottleneck to latent vector. ``TensorMask`` and ``LatentVector`` are the data carriers.
    - ``partialconv2d.py``: ``PartialConv2d``, a convolution that tracks a validity mask, used for data with missing values.
    - ``unet.py``: ``DownBlock``, ``UpBlock``, ``UNetOutput``, ``UNetOutputSIC``, and ``build_conv_block``.
    - ``mlp.py``: ``build_mlp``, which builds a fully connected stack.
    - ``utils.py``: padding, mask broadcasting and merging, noise injection, and resizing helpers.

``mlp/``
    Fully connected models:

    - ``deterministic.py``: ``AutoencoderConfig`` / ``Autoencoder``, registered as ``"mlp"`` for deterministic prediction. ``append_mode`` controls where added features enter.
    - ``cvae.py``: ``cVAE_MLPConfig`` / ``cVAE_MLP``, a conditional VAE with MLP encoder and decoder.

``unet/``
    U-Net models:

    - ``deterministic.py``: ``UNetConfig`` / ``UNet`` and ``UNetSICConfig`` / ``UNetSIC`` (the SIC variant clamps output).
    - ``cvae.py``: ``cVAEUNetConfig`` / ``cVAEUNet`` and ``cVAEUNetSICEConfig`` / ``cVAEUNetSIC``. A conditional U-Net encodes the target into a latent, and can add a deterministic-guess U-Net branch.
    - ``utils.py``: config validation and mask repetition for ensemble generation.

Model selection
---------------

Models register with ``cVAEModelSelector`` or ``deterministicModelSelector`` using ``@register("name")``, so a YAML config can pick one by name (``"mlp"``, ``"unet"``, ``"unetsic"``).
Architectures
=============

The ``S2D4P.architectures`` package defines the neural network architectures and their building blocks. Each architecture has a config dataclass (subclassing ``ArchConfigABC``) that validates settings and builds the architecure, plus a ``torch.nn.Module`` implementation.
The design separates the modeling framework (e.g. cVAE, deterministic, GAN, etc.) and architecture (e.g. MLP, UNet, transformer, etc.) as model vs. architecture.
Each architecture is desined as a ``nn.Module`` child class and requrires a corresponding config class thst is registered to a selector class.

Subpackages and modules
-----------------------

``architectures_abc.py``
    Abstract base classes and shared request/config types:

    - ``ArchConfigABC`` and ``ArchABC``: base classes for all configs and architrectures. Provide checkpoint loading, weight initialization, and device lookup.
    - ``deterministicArchABC`` and ``cVAEArchsABC``: base classes for deterministic and conditional VAE model compatable architectures. Adding architectures for new models will require creating their own ABCs, e,g, GANArchABC.
    - ``DeterministicRequest``, ``cVAEForwardRequest``, ``cVAEPredictRequest``: typed request objects passed to ``forward``/``predict``. This makes it easy for developers and users to understand what allowed and required input combinations and arguments specific to each model.
    - ``GENERATORConfig``: settings for the ensemble generator (noise samples, noise level). It is up to the architecture to support a generator formulation.
    - ``CheckpointConfig``: metadata for loading a saved nn module (shapes, variable metadata, freeze flag).
    - ``flowABC``: abstract base for normalizing flows. This is used by cVAE for prior distrbution flexibility.
    - ``weights_init``: applies kaiming, xavier, or truncated-normal initialization.

``normalized_flows.py``
    Normalizing flows used as learned priors in the cVAE:

    - ``NormalizedFlowConfig`` builds a ``NormalizedFlowModel`` from a list of flow steps.
    - ``MAF`` (masked autoregressive flow) and ``RealNVP`` (affine coupling) are registered flow types.
    - ``FCNN`` is the small base network used inside each flow step.

``layers/``
    Reusable building blocks. It is expected that building blocks that might be later used for different architecures to be defined here under the architecture family name, e.g. ``unet.py`` .

    - ``generic.py``: type aliases (activation, normalization, padding, init methods) and helpers ``_build_activation``, ``_build_normalization``, ``LayerNorm2d``, ``DropPath``.
    - ``conv.py``: block configs (``ConvBlockConfig``, ``PartialConvBlockConfig``, ``ConvNeXtBlockConfig``) and block modules (``ConvBlock``, ``PartialConvBlock``, ``ConvNeXtBlock``), plus ``LatentBlock``/``LatentLayer`` for the bottleneck to latent vector. ``TensorMask`` and ``LatentVector`` are the data carriers.
    - ``partialconv2d.py``: ``PartialConv2d``, a convolution that tracks a validity mask, used for data with missing values.
    - ``unet.py``: ``DownBlock``, ``UpBlock``, ``UNetOutput``, ``UNetOutputSIC``, and ``build_conv_block``. Building blocks of a UNet architecture.
    - ``mlp.py``: ``build_mlp``, which builds a fully connected layers.  Building blocks of an MLP architecture.
    - ``utils.py``: padding, mask broadcasting and merging, noise injection, and resizing helpers.

``mlp/``
    Fully connected architectures family. Each model might require different architectural choices, e.g., cVAE needs separate encoder, decoder and potentially conditon branches. Many different architectures can be imagined but archirectures of the same multi-layer perceptron family can be defined here. 
    All architectures defined here will be registered to a Selector designed for that family of architectures (see below).
    - ``deterministic.py``: ``AutoencoderConfig`` / ``Autoencoder``, registered as ``"mlp"`` for deterministic prediction. ``append_mode`` controls where added features enter.
    - ``cvae.py``: ``cVAE_MLPConfig`` / ``cVAE_MLP``, a conditional VAE with MLP encoder and decoder.

``unet/``
    U-Net architectures family.  Each model might require different architectural choices, e.g., cVAE needs separate encoder, decoder and potentially conditon branches. Many different architectures can be imagined but archirectures of the same unet family can be defined here. 
    All architectures defined here will be registered to a Selector designed for that family of architectures (see below).
    - ``deterministic.py``: ``UNetConfig`` / ``UNet`` and ``UNetSICConfig`` / ``UNetSIC`` (the SIC variant clamps output).
    - ``cvae.py``: ``cVAEUNetConfig`` / ``cVAEUNet`` and ``cVAEUNetSICEConfig`` / ``cVAEUNetSIC``. A conditional U-Net encodes the target into a latent, and can add a deterministic-guess U-Net branch.
    - ``utils.py``: config validation and mask repetition for ensemble generation.

Model selection
---------------

Architectures register with ``cVAEArchSelector`` or ``deterministicArchSelector`` using ``@register("name")``, so a YAML config can pick one by name (``"mlp"``, ``"unet"``, ``"unetsic"``).

Development
---------------

Future use might require a new architecture. Each new architecture requires a class and config class and will be compatible with ``S2D4P``` as long as they inherit from ``ArchABC`` and ``ArchConfigABC`` respectively.
If an architecture falls under the existing architecture families (mlp and unet), it is expected that the building blocks be added to the corresponding family in ``layers/``, while the architecture object class and its config class 
are defined in their corresponding family's directory (``mlp/`` or ``unet/``, etc.) for their corresponding model (``deterministic.py`` or ``cvae.py``, etc.). Finelly, the config class must be registered with that model family's
selector class (``deterministicArchSelector`` or ``cVAEArchSelector``, etc.).  Should the design fall under a new family of architectures, it is expected its building blocks, architecture and config class to be defined under representative family names 
(e.g. ``layers/transformers.py``, ``transformer/cvae.py``), and the config class registered with the corresponidng model selector (e.g. ``cVAEArchSelector``).
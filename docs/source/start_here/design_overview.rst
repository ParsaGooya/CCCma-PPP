Design Logic and Software Architecture Overview
===============================================

Summary
-------

The design of S2D4P combines hierarchical configuration, factory-based construction, abstract interfaces, and registry-driven polymorphism.

These mechanisms serve complementary purposes:

+-----------------------------------------+----------------------------------------------------------+
| Mechanism                               | Main design purpose                                      |
+=========================================+==========================================================+
| Configuration dataclasses               | Typed, structured declaration of component settings      |
+-----------------------------------------+----------------------------------------------------------+
| ``__post_init__()``                     | Local validation and derived-parameter preparation       |
+-----------------------------------------+----------------------------------------------------------+
| ``build()`` methods                     | Deferred construction of runtime implementations         |
+-----------------------------------------+----------------------------------------------------------+
| Abstract base classes                   | Common interfaces and component interchangeability       |
+-----------------------------------------+----------------------------------------------------------+
| Selector registries                     | Dynamic selection of concrete implementations            |
+-----------------------------------------+----------------------------------------------------------+
| YAML                                    | Human-readable, reusable experiment specification        |
+-----------------------------------------+----------------------------------------------------------+
| ``dacite``                              | Conversion of nested dictionaries into typed dataclasses |
+-----------------------------------------+----------------------------------------------------------+
| ``TrainConfig`` and ``InferenceConfig`` | Coordination of the full experiment configuration        |
+-----------------------------------------+----------------------------------------------------------+

The central design principle is that **the experiment is described declaratively, its components are selected through registries, and its execution is delegated to runtime objects constructed through well-defined interfaces.**

This organization allows S2D4P to expand its supported data sources, architectures, learning objectives, and post-processing applications while preserving a common and reusable machine-learning infrastructure.

*This document and module docstrings are generated with the help of Artifical Intelligence.*

1. Introduction
---------------

S2D4P is designed as a modular, configuration-driven machine-learning framework for climate prediction post-processing. It supports different learning paradigms, neural network architectures, data sources, and preprocessing methods through a common training and inference infrastructure.

The framework follows four principal design choices:

1. **Separation of configuration and implementation:** Components are represented by configuration dataclasses that describe their parameters and provide interfaces for constructing the corresponding runtime objects.
2. **Hierarchical configuration:** An experiment is described by a nested configuration tree, with ``TrainConfig`` as its root. ``TrainConfig`` populates the condifuration classes for runtime objects. Each configuration object is responsible for a specific part of the pipeline.
3. **Registry-based component selection:** Selector registries associate symbolic names with concrete implementations, allowing components to be selected dynamically without hardcoding their dependencies throughout the framework.
4. **Declarative experiment definitions:** Experiments are specified through YAML files, which are interpreted and converted into typed Python configuration objects using ``dacite``.

Together, these choices separate experiment specification from implementation, promote code reuse, and make it easier to extend the framework without modifying its core training infrastructure.

2. Separation of Configuration and Runtime Classes
--------------------------------------------------

A fundamental design pattern inS2D4P is the separation between configuration classes and their associated runtime classes. The configuration classes are the levels at which the user interacts with the pipeline and the objects are created internally.

Configuration classes
~~~~~~~~~~~~~~~~~~~~~

Configuration classes are generally implemented as Python ``dataclasses``. They describe the parameters required to initialize a component, validate those parameters, and provide a ``build()`` method to construct the corresponding implementation. This design postpones the creation

of large and memory consuming objects until all checks have passed.

A configuration class typically has three responsibilities:

-  **Parameter declaration:** Define the supported configuration fields, their types, and default values.
-  **Validation and preparation:** Check parameter consistency and prepare derived settings, typically through ``__post_init__()``.
-  **Object construction:** Provide a ``build()`` method that instantiates the runtime component, optionally using information available only when the pipeline is assembled.

3. Separation of learning paradigm and architecture
---------------------------------------------------

In S2D4P, the learning paradigm ,i.e., the recipe/algorithm for training and inference is represented by ``model`` and separated from ``architecture`` which defines the backbone of the model and defines its forward processing algorithm and inductive bias.

The learning module defines the learning problem and associated computational logic. It determines how predictions are produced, how training objectives are evaluated, and, for generative models, how probabilistic samples are generated.

The neural architecture defines the computational mapping used by the learning module.

Runtime classes
~~~~~~~~~~~~~~~

Runtime classes implement the actual computational behavior of a component. Depending on the component, this may involve data loading, preprocessing, forward passes, optimization, sampling, or execution of a training loop.

Runtime objects can maintain mutable execution state, fitted parameters, neural network weights, or references to other components.

For example, the trainer follows this pattern:

.. code:: python

   @dataclasses.dataclass
   class TrainerConfig:
       gradient_accumulation_steps: int = 1
       mixed_precision: bool = True
       grad_clip: float | None = None
       def __post_init__(self):
           if self.grad_clip is not None:
               assert self.grad_clip > 0
       def build(
           self,
           train_data_loader,
           validation_data_loader,
           optimization,
           module,
           max_epochs,
       ):
           return Trainer(
               config=self,
               train_data_loader=train_data_loader,
               validation_data_loader=validation_data_loader,
               module=module,
               optimizer=optimization,
               max_epochs=max_epochs,
           )

Here, ``TrainerConfig`` contains the declarative settings, while ``Trainer`` implements the training loop, optimization steps, validation, and checkpoint management.

The ``build()`` method provides a bridge between these two representations.

Importantly, the configuration object does not necessarily contain all information required to construct the runtime object. Some information, such as input dimensions, the number of training batches, or references to other instantiated components, becomes available only during pipeline initialization.

Consequently, ``build()`` methods may accept additional arguments that represent runtime dependencies or data-derived properties.

This pattern resembles a *factory-based architecture*, in which configuration classes act as specialized factories for their corresponding implementations.

3. Hierarchical Configuration Architecture
------------------------------------------

An experiment in S2D4P is represented by a hierarchy of configuration objects.

At the top level, ``TrainConfig`` defines an experiment and contains the configuration objects associated with its major components.

A simplified representation of this hierarchy is:

.. code:: text

   TrainConfig
   │
   ├── train_loader: TrainDataloaderConfig
   │   └── dataset_config: TrainDatasetConfig
   │       ├── model: ModelDataConfig
   │       │   └── preprocessing_pipeline
   │       ├── observation: ObsDataConfig | None
   │       │   └── preprocessing_pipeline
   │       └── condition: ConditionDataConfig | None
   │           └── preprocessing_pipeline
   │
   ├── module: ModuleSelector
   │   └── ModuleConfig
   │       └── ModelConfig: ModelSelector
   │           └── ArchitectureConfig
   │
   ├── losspipeline: LosspipelineConfig
   │   └── loss_pipeline: list[LossStepConfig]
   │
   ├── trainer: TrainerConfig
   │
   ├── optimization: OptimizerConfig
   │   └── lr_scheduler_config
   │
   ├── weights: WeightsConfig
   │
   └── Experiment-level settings
       ├── experiment_dir
       ├── max_epochs
       ├── seed
       ├── save_checkpoint
       └── resume_dir

*This is a conceptual representation of the configuration hierarchy rather than a complete listing of every field.*

Responsibility boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~

Each branch is responsible for a distinct part of the experiment:

+-----------------------------------+-------------------------------------------------------------------------------------------------+
| Configuration                     | Responsibility                                                                                  |
+===================================+=================================================================================================+
| ``TrainConfig``                   | Defines and coordinates the experiment                                                          |
+-----------------------------------+-------------------------------------------------------------------------------------------------+
| ``TrainDataloaderConfig``         | Controls dataloader construction and batching                                                   |
+-----------------------------------+-------------------------------------------------------------------------------------------------+
| ``TrainDatasetConfig``            | Defines the dataset, input sources, temporal selections, and associated preprocessing           |
+-----------------------------------+-------------------------------------------------------------------------------------------------+
| Data configurations               | Describe forecast, reference, and conditioning data                                             |
+-----------------------------------+-------------------------------------------------------------------------------------------------+
| ``ModuleSelector``                | Selects the learning paradigm or ``model``, such as deterministic learning or a conditional VAE |
+-----------------------------------+-------------------------------------------------------------------------------------------------+
| ``ArchSelector``                  | Selects the neural network architecture                                                         |
+-----------------------------------+-------------------------------------------------------------------------------------------------+
| ``LosspipelineConfig``            | Specifies the loss terms and their composition                                                  |
+-----------------------------------+-------------------------------------------------------------------------------------------------+
| ``PreprocessingPipeline``         | Specifies the preprocessing steps and their configurations                                      |
+-----------------------------------+-------------------------------------------------------------------------------------------------+
| ``TrainerConfig``                 | Controls training-loop behavior                                                                 |
+-----------------------------------+-------------------------------------------------------------------------------------------------+
| ``OptimizerConfig``               | Specifies optimization and learning-rate scheduling                                             |
+-----------------------------------+-------------------------------------------------------------------------------------------------+
| ``WeightsConfig``                 | Defines spatial or variable-dependent loss weighting                                            |
+-----------------------------------+-------------------------------------------------------------------------------------------------+
| ``InferenceConfig``               | Defines and coordinates the inference loop                                                      |
+-----------------------------------+-------------------------------------------------------------------------------------------------+
| ``WriterConfig``                  | Controls inference-loop behavior                                                                |
+-----------------------------------+-------------------------------------------------------------------------------------------------+

This organization enables individual components to be modified independently.

For example, replacing a deterministic learning module with a conditional generative module should not require rewriting the dataloader or trainer, provided the new module satisfies the required interfaces.

Likewise, changing a U-Net architecture to another compatible model should primarily affect the architecture configuration, not the remainder of the experiment.

The nested structure also allows configuration validation to occur at multiple levels. Individual components can validate their own parameters, while higher-level configurations can enforce constraints involving multiple components.

4. Selector Registries and Component Extensibility
--------------------------------------------------

Although configuration dataclasses establish the structure of the pipeline, many configuration fields represent alternative implementations that share a common interface.

For example, a learning module might use a deterministic formulation or a conditional variational autoencoder. A neural architecture might be an MLP, U-Net, or another compatible model.

Hardcoding these choices through repeated ``if``/``elif`` statements would tightly couple the core framework to every available implementation.

S2D4P instead uses **selector registries** to associate names with implementation classes.

Registration
~~~~~~~~~~~~

A concrete configuration class can be registered under a symbolic name through a decorator.

For example:

.. code:: python

   @deterministicModelSelector.register("unet")
   @dataclasses.dataclass
   class UNetConfig(modelConfigABC):
       channels: list[int]
       bottleneck_dim: int | None = None
       def build(
           self,
           input_shape,
           output_shape=None,
           added_features_dim=None,
       ):
           return UNet(
               config=self,
               input_shape=input_shape,
               output_shape=output_shape,
               added_features_dim=added_features_dim,
           )

The registration associates the key ``"unet"`` with ``UNetConfig``.

A user can subsequently select this architecture through configuration without importing the concrete class into the experiment definition.

Selector responsibilities
~~~~~~~~~~~~~~~~~~~~~~~~~

Conceptually, a selector performs three operations:

1. Identifies the requested implementation using its registered name.
2. Resolves that name to the associated configuration class.
3. Constructs the appropriate configuration object from the provided parameters.

The resulting configuration object can then build its runtime implementation.

This gives two distinct layers of construction:

.. code:: text

   Registered name + configuration parameters
                     │
                     ▼
              Selector registry
                     │
                     ▼
          Concrete configuration
                     │
                 build()
                     │
                     ▼
              Runtime object

The selector resolves *which configuration class* to use, while the configuration class determines *how its runtime object is constructed*.

This distinction is important because the selected implementation may require additional runtime information that is not available during YAML parsing.

Benefits of registration
~~~~~~~~~~~~~~~~~~~~~~~~

Registry-based selection provides several advantages.

**Extensibility:** New models, preprocessing steps, and learning modules can be introduced by defining and registering new implementations as long as they comply with pipelines abstraction recipe (see below).

**Decoupling:** Parent components depend on common interfaces rather than concrete implementations.

**Configuration-driven selection:** Users can change components by editing configuration files rather than modifying core pipeline code.

**Code reuse:** Shared training infrastructure can operate on multiple implementations that satisfy the same interface.

**Discoverability:** A registry provides a centralized mapping between available component names and their corresponding classes.

Registration introduces an important implementation requirement: the module containing a registered class must be imported before that registry entry can be resolved. Registries do not automatically discover classes that have never been imported.

5. Abstract Interfaces and Polymorphism
---------------------------------------

The registry mechanism is complemented by abstract base classes and shared component interfaces.

Abstract interfaces establish the methods and properties that a component must provide to participate in the pipeline.

For instance, architecture configurations must inherit from ``ArchConfigABC``, while runtime architectures implement a corresponding object interface.

This allows the framework to work with an abstract model contract rather than a specific architecture.

An important example is the separation between the **learning module (model)** and the **neural network architecture (architecture)**.

A deterministic module and a conditional VAE can employ different architectures without requiring the architecture itself to implement the entire training procedure.

This separation is especially relevant to the framework’s support for different post-processing tasks, including bias adjustment, ensemble augmentation, downscaling, and conditional generation.

It also provides a natural extension point for additional generative approaches, such as diffusion-based models, provided they implement the interfaces expected by the surrounding infrastructure.

6. YAML-Based Configuration and ``dacite`` Deserialization
----------------------------------------------------------

S2D4P uses YAML as a human-readable format for specifying experiments.

Rather than defining every experiment programmatically, users describe the desired pipeline through configuration files.

These configurations can be saved, version-controlled, compared, and reused.

Configuration loading
~~~~~~~~~~~~~~~~~~~~~

The YAML configuration is first loaded into standard Python objects, typically nested dictionaries and lists.

The resulting dictionary is then converted into the typed configuration hierarchy using ``dacite``.

A simplified example is:

.. code:: python

   import yaml
   import dacite
   with open("train_config.yaml") as f:
       config_data = yaml.safe_load(f)
   config = dacite.from_dict(
       data_class=TrainConfig,
       data=config_data,
       config=dacite.Config(strict=True),
   )

``dacite`` can recursively construct nested configuration objects.

Recursive dataclass construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because ``TrainConfig`` contains other dataclass types, ``dacite`` can recursively construct nested configuration objects.

For example, a dictionary representing ``TrainerConfig`` can be converted into an instance of that dataclass as part of constructing ``TrainConfig``.

This allows the YAML structure to closely follow the Python configuration hierarchy.

However, recursive dataclass construction and registry-based selection solve different problems.

``dacite`` can construct a dataclass when its concrete type is known from the corresponding field annotation. It does not automatically determine which registered subclass should be instantiated from an arbitrary registry key.

Consequently, fields representing polymorphic components require an additional selection or conversion mechanism.

Integration with selectors
~~~~~~~~~~~~~~~~~~~~~~~~~~

Selector registries connect declarative YAML entries to the appropriate concrete configuration types.

Conceptually, the full loading process is:

.. code:: text

   YAML configuration file
             │
             ▼
       yaml.safe_load()
             │
             ▼
     Nested Python dictionary
             │
             ▼
       Selector resolution
       where polymorphism
          is required
             │
             ▼
     dacite.from_dict()
             │
             ▼
      Typed configuration tree
             │
             ▼
      Validation and build()
             │
             ▼
      Executable ML pipeline

The precise ordering of selector resolution and ``dacite`` construction depends on how a selector is implemented. A selector may resolve its target class during configuration conversion or through a custom construction step.

The essential design requirement is that each polymorphic entry resolves to an appropriate concrete configuration type before the corresponding runtime component is built.

Type safety and validation
~~~~~~~~~~~~~~~~~~~~~~~~~~

This loading mechanism provides several levels of error detection:

-  **YAML parsing:** Detects invalid YAML syntax.
-  **Dataclass deserialization:** Checks the structure and types of configuration fields.
-  **Configuration validation:** Applies component-specific constraints through methods such as ``__post_init__()``.
-  **Pipeline construction:** Checks requirements that depend on other components or runtime information.

These layers are complementary. ``dacite`` validates the configuration structure but does not replace semantic validation or guarantee compatibility between independently configured components.

Care is also required when defining forward references, union types, and abstract configuration fields, since these may require explicit type resolution or custom conversion logic.

7. Configuration Validation and Object Construction
---------------------------------------------------

The configuration lifecycle separates parsing, validation, and construction.

**Parsing** converts a declarative representation into Python configuration objects.

**Validation** determines whether the specified settings are internally consistent and compatible with the requested components.

**Construction** instantiates the runtime objects using both configuration settings and runtime dependencies. This is usually the computationally costly part and is delayed as much as possible to after when all existing checks pass.

For example, a U-Net configuration may validate its number of channels and derive the expected number of decoder stages in ``__post_init__()``.

However, its final construction may require input and output shapes inferred from the dataset.

Similarly, the trainer may require the training dataloader to be constructed before it can initialize a schedule that depends on the number of batches.

The use of ``build()`` therefore supports a form of **deferred construction**: a component’s settings are defined early, but the implementation is instantiated only when its dependencies are available.

The resulting dependency structure also gives the high-level training pipeline responsibility for assembling components in an appropriate order.

8. Reusability, Reproducibility, and Experiment Management
----------------------------------------------------------

Configuration-driven architecture is especially valuable in climate machine learning, where experiments frequently vary across datasets, initialization periods, lead times, variables, neural architectures, and training objectives.

A configuration file captures the intended structure of an experiment independently of the code responsible for executing it.

The separation of configuration and implementation supports reproducibility in several ways.

**Explicit experiment definitions:** Hyperparameters, data selections, preprocessing choices, and training settings can be recorded in structured configuration files.

**Consistent initialization:** The same configuration can be used to reconstruct compatible components across training and inference workflows.

**Reusable preprocessing:** Fitted preprocessing transformations can be stored and subsequently restored so that inference applies the transformations established during training.

**Modular comparisons:** Alternative models and training objectives can be compared while preserving common data-processing and execution settings.

**Version-controlled experiments:** YAML definitions can be tracked alongside the source code, making changes between experiments easier to identify.

The configuration alone is not a complete reproducibility record. Exact reproduction may also depend on the source-code revision, model checkpoints, fitted transformations, random seeds, software environment, and input-data versions.

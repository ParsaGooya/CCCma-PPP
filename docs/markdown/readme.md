# CCCma Post-Processing Package (PPP)

A machine learning pipeline for post-processing seasonal-to-decadal predictions produced by the Canadian Earth System Models (CanESM) at the Canadian Centre for Climate Modelling and Analysis (CCCma).

The training framework supports a suite of deterministic and probabilistic models across multiple neural network architectures. Current applications include bias adjustment, ensemble boosting, and statistical downscaling.

The package is currently under active development and is undergoing quality assurance (QA).

## Ownership and Use

Developed by Parsa Gooya at the Canadian Centre for Climate Modelling and Analysis (CCCma), Environment and Climate Change Canada (ECCC).
Contact: parsa.gooya@ec.gc.ca

All rights reserved. This software and associated materials may not be used, copied, modified, distributed, or shared without prior written permission from the authors.

Copyright © 2026 Parsa Gooya and CCCma.

# Installation

## Install `uv`

```bash
pip install uv
```

## Create a new virtual environment

```bash
uv venv
```

## Activate the virtual environment

- **Linux/macOS**

```bash
  source .venv/bin/activate
```

- **Windows (PowerShell)**

```powershell
  .venv\Scripts\Activate.ps1
```

## Install the package

```bash
uv pip install .
```

For development mode:

```bash
uv pip install -e .
```

# Usage

## Training

```bash
torchrun --nproc_per_node=gpu -m S2D4P.train path_to_train_config.yaml
```

## Inference

```bash
torchrun --nproc_per_node=gpu -m S2D4P.inference path_to_inference_config.yaml
```

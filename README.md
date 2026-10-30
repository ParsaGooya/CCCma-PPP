# S2D4P: Sub-seasonal to decadal predictions pre-processing package 

A machine learning pipeline for post-processing seasonal-to-decadal predictions produced by the Canadian Earth System Models (CanESM) at the Canadian Centre for Climate Modelling and Analysis (CCCma).

The training framework supports a suite of deterministic and probabilistic models across multiple neural network architectures. Current applications include bias adjustment, ensemble boosting, and statistical downscaling.

The package is currently under active development and is undergoing quality assurance (QA).

---
Note:
Packged name recently changed from cccma_ppp to S2D4P.

---
## Installation

`pip` installable with `pip install .`. Or for development mode, `pip install -e .`.

---
# Usage

## Job Scheduling on ECCC's Science machines 

```bash
qsub -I -lselect=1:ncpus={number_of_cpus}:mpiprocs={number_of_MPI_processes}:ngpus={number_of_gpus}:mem={gb_ram}G:vntype=gpu -lwalltime={hours}:{mins}:{seconds}
```

Example:

```bash
qsub -I -lselect=1:ncpus=96:mpiprocs=6:ngpus=1:mem=400G:vntype=gpu -lwalltime=3:0:0
```

## Training

```bash
torchrun --nproc_per_node=gpu -m cccma_ppp.train path_to_train_config.yaml
```

## Inference

```bash
torchrun --nproc_per_node=gpu -m cccma_ppp.inference path_to_inference_config.yaml
```
---
## Ownership and Use

Developed by Parsa Gooya at the Canadian Centre for Climate Modelling and Analysis (CCCma), Environment and Climate Change Canada (ECCC).
Contact: parsa.gooya@ec.gc.ca

All rights reserved. This software and associated materials may not be used, copied, modified, distributed, or shared without prior written permission from the authors.

Copyright © 2026 Parsa Gooya and CCCma.
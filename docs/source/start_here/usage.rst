Usage
=====
Training
--------
.. code-block:: bash

    torchrun --nproc_per_node=gpu -m S2D4P.train path_to_train_config.yaml

Inference
--------
.. code-block:: bash

    torchrun --nproc_per_node=gpu -m S2D4P.inference path_to_inference_config.yaml
Usage
=====
Training
--------
.. code-block:: bash

    torchrun --nproc_per_node=gpu -m cccma_ppp.train path_to_train_config.yaml

Inference
--------
.. code-block:: bash

    torchrun --nproc_per_node=gpu -m cccma_ppp.inference path_to_inference_config.yaml
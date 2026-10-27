Usage
=====
Job Scheduling
--------------
.. code-block:: bash

    qsub -I -lselect=1:ncpus={number_of_cpus}:mpiprocs={number_of_MPI_processes}:ngpus={number_of_gpus}:mem={gb_ram}G:vntype=gpu -lwalltime={hours}:{mins}:{seconds}

    qsub -I -lselect=1:ncpus=96:mpiprocs=6:ngpus=1:mem=400G:vntype=gpu -lwalltime=3:0:0

Training
--------
.. code-block:: bash

    torchrun --nproc_per_node=gpu -m cccma_ppp.train path_to_train_config.yaml

Inference
--------
.. code-block:: bash

    torchrun --nproc_per_node=gpu -m cccma_ppp.inference path_to_inference_config.yaml
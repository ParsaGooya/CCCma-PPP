# Usage

qsub -I -lselect=1:ncpus=96:mpiprocs=6:ngpus=1:mem=400G:vntype=gpu -lwalltime=3:0:0 #-v TERM

torchrun --nproc_per_node=gpu -m cccma_ppp.train scripts/integration_suite_train_config.yaml

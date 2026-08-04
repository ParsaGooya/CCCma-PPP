import torch
import torch.distributed as dist
import os


class Distributed:
    _instance = None

    def __init__(self):

        self.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

        if self.distributed:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])

            torch.cuda.set_device(self.local_rank)

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            self.device = torch.device(f"cuda:{self.local_rank}")

        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_instance(cls):

        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def cleanup(self):

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def is_root(self) -> bool:

        return self.rank == 0

    def barrier(self):

        if self.distributed:
            dist.barrier()

    def all_reduce_sum(self, local: torch.Tensor):

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)

    def broadcast(self, local: torch.Tensor, src=0):

        if dist.is_available() and dist.is_initialized():
            dist.broadcast(local, src=src)

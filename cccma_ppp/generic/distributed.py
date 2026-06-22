from __future__ import annotations
import torch
import torch.distributed as dist
import os


class Distributed:
    """
    Single class for managing distributed training setup.

    Attributes
    ----------
    distributed : bool
        Whether distributed training is enabled.
    rank : int
        Global process rank.
    local_rank : int
        Local process rank (GPU index).
    world_size : int
        Total number of processes.
    device : torch.device
        Device assigned to the current process.
    """

    _instance = None

    def __init__(self):
        """
        Initialize distributed environment.

        Returns
        -------
        None
        """

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
        """
        Retrieve singleton instance.

        Returns
        -------
        Distributed
            Shared instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def cleanup(cls):
        """
        Destroy distributed process group.

        Returns
        -------
        None
        """
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def is_root(self) -> bool:
        """
        Check if current process is the root process.

        Returns
        -------
        bool
            True if rank is 0.
        """
        return self.rank == 0

    def barrier(self):
        """
        Synchronize all processes.

        Returns
        -------
        None
        """
        if self.distributed:
            dist.barrier()

    def all_reduce_sum(self, local: torch.Tensor):
        """
        Perform all-reduce sum across processes.

        Parameters
        ----------
        local : torch.Tensor
            Tensor to be summed across all ranks.

        Returns
        -------
        None
        """

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)

    def broadcast(self, local: torch.Tensor, src=0):
        """
        Broadcast tensor from source process to all processes.

        Parameters
        ----------
        local : torch.Tensor
            Tensor to broadcast.
        src : int, optional
            Source rank.

        Returns
        -------
        None
        """
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(local, src=src)

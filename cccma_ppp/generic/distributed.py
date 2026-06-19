import torch
import torch.distributed as dist
import os


class Distributed:
    """
    Utility class for managing distributed training using PyTorch Distributed.

    Methods
    -------
    get_instance()
        Return singleton instance of the Distributed manager.
    cleanup()
        Destroy the distributed process group.
    is_root()
        Check if current process is the root rank.
    barrier()
        Synchronize all processes.
    all_reduce_sum(local)
        Perform sum reduction across all processes.
    broadcast(local, src=0)
        Broadcast tensor from source process to all processes.
    """
    _instance = None

    def __init__(self):
        """
        Initialize distributed environment and device configuration.

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
        Return singleton instance of Distributed.

        Returns
        -------
        Distributed
            Shared instance managing distributed state.
        """

        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def cleanup(cls):
        """
        Clean up distributed process group.

        Returns
        -------
        None
        """

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def is_root(self) -> bool:
        """
        Check if current process is the root rank.

        Returns
        -------
        bool
            True if rank equals zero.
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
        Perform sum reduction across all processes.

        Parameters
        ----------
        local : torch.Tensor
            Tensor to be reduced.
        Returns
        -------
        None
        """

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)

    def broadcast(self, lcoal: torch.Tensor, src=0):
        """
        Broadcast tensor from source rank to all processes.

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
            dist.broadcast(lcoal, src=src)

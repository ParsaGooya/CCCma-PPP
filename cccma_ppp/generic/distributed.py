import torch
import torch.distributed as dist
import os


class Distributed:
    """
    Manage distributed and single-process execution settings.

    Detects distributed execution from environment variables and configures the
    process rank, local CUDA device, world size, communication backend, and
    runtime device accordingly.
    """

    _instance = None

    def __init__(self):
        """
        Initialize the distributed execution environment.

        When distributed environment variables are present, configures the local CUDA
        device and initializes an NCCL process group if necessary. Otherwise, uses a
        single process on CUDA when available or CPU as a fallback.
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
        Return the shared distributed runtime instance.

        Returns
        -------
        Distributed
            Singleton distributed runtime instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def cleanup(self):
        """
        Destroy the active distributed process group.

        If distributed communication is unavailable or has not been initialized,
        this method has no effect.
        """
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def is_root(self) -> bool:
        """
        Indicate whether the current process is the root process.

        Returns
        -------
        bool
            ``True`` if the current process has rank zero, otherwise ``False``.
        """
        return self.rank == 0

    def barrier(self):
        """
        Synchronize all distributed processes.

        If distributed execution is disabled, this method has no effect.
        """
        if self.distributed:
            dist.barrier()

    def all_reduce_sum(self, local: torch.Tensor):
        """
        Sum a tensor across all distributed processes.

        The reduction is performed in place. If distributed communication has not
        been initialized, the tensor is left unchanged.

        Parameters
        ----------
        local : torch.Tensor
            Tensor containing the local value to sum across processes.
        """
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)

    def broadcast(self, local: torch.Tensor, src=0):
        """
        Broadcast a tensor from a source process to all distributed processes.

        The tensor is modified in place. If distributed communication has not been
        initialized, the tensor is left unchanged.

        Parameters
        ----------
        local : torch.Tensor
            Tensor to broadcast or receive.
        src : int
            Rank of the source process.
        """
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(local, src=src)

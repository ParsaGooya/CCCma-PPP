import torch
import torch.distributed as dist
import os


class Distributed:
    """
    Manage distributed execution state and collective communication.

    The class initializes distributed training from standard rank environment
    variables when available. Otherwise, it configures single-process execution
    on CUDA when available or CPU as a fallback.
    """

    _instance = None

    def __init__(self):
        """
        Initialize distributed or single-process execution.

        When ``RANK`` and ``WORLD_SIZE`` are available in the environment, initializes
        an NCCL process group, assigns the process to its local CUDA device, and
        records rank information. Otherwise, configures a single process on the best
        available device.
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
        Clean up distributed communication resources.

        Destroys the active process group when distributed communication has been
        initialized.
        """
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    def is_root(self) -> bool:
        """
        Indicate whether the current process is the root process.

        Returns
        -------
        bool
            ``True`` when the current process has rank zero.
        """
        return self.rank == 0

    def barrier(self):
        """
        Synchronize all distributed processes.

        Has no effect during single-process execution.
        """
        if self.distributed:
            dist.barrier()

    def all_reduce_sum(self, local: torch.Tensor):
        """
        Sum a tensor across all distributed processes.

        The input tensor is modified in place with the globally reduced sum. Has no
        effect when distributed communication is not initialized.

        Parameters
        ----------
        local : torch.Tensor
            Local tensor to sum across processes.
        """
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)

    def broadcast(self, local: torch.Tensor, src=0):
        """
        Broadcast a tensor from one process to all distributed processes.

        The input tensor is modified in place with the value from the source process.
        Has no effect when distributed communication is not initialized.

        Parameters
        ----------
        local : torch.Tensor
            Tensor containing the value to broadcast or receive.
        src : int
            Rank of the source process.
        """
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(local, src=src)

import numpy as np
import dataclasses
import torch
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import random
from pathlib import Path

from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext


@dataclasses.dataclass
class MetricsAggregator:
    """
    Document this class.

    Parameters
    ----------
    distributed : Distributed
        Description not yet provided.
    name : str
        Description not yet provided.
    epoch_metric_terms : dict[str, list[float]] | None
        Description not yet provided.
    epoch_times : list[float] | None
        Description not yet provided.
    num_epochs_seen : int
        Description not yet provided.
    """

    distributed: Distributed
    name: str

    epoch_metric_terms: dict[str, list[float]] | None = None
    epoch_times: list[float] | None = None
    num_epochs_seen: int = 0

    def __post_init__(self):
        """
        Document this function.

        Raises
        ------
        AssertionError
            Description not yet provided.
        """
        self.loss_terms = defaultdict(float)
        self.lr_values = 0.0
        self.kwargs_terms = defaultdict(float)
        self.num_batches_seen = 0

        self.epochs_submitted = False
        self._aggregated_across_ranks = False

        if self.epoch_metric_terms is not None:
            lengths = {len(v) for v in self.epoch_metric_terms.values()}
            assert len(lengths) == 1, (
                f"Not all loss lists have the same length: {lengths}"
            )
            if self.num_epochs_seen == 0:
                self.num_epochs_seen = next(iter(lengths), 0)

        else:
            self.epoch_metric_terms = {}

        if self.epoch_times is not None:
            assert self.epoch_metric_terms is not None, (
                "must specify corresponding loss terms."
            )
            assert len(self.epoch_times) == next(iter(lengths), 0), (
                "length of epoch_times is inconsistent with epoch_metric_terms."
            )

        else:
            self.epoch_times = []

    @torch.no_grad()
    def record(
        self,
        loss_dict: dict[str, torch.Tensor | int | float],
        lr: torch.Tensor | int | float | None = None,
        kwargs: dict[str, torch.Tensor | int | float] | None = None,
    ) -> None:
        """
        Document this function.

        Parameters
        ----------
        loss_dict : dict[str, torch.Tensor | int | float]
            Description not yet provided.
        lr : torch.Tensor | int | float | None
            Description not yet provided.
        kwargs : dict[str, torch.Tensor | int | float] | None
            Description not yet provided.
        """
        for name, value in loss_dict.items():
            if value is None:
                continue

            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()

            if isinstance(value, (int, float)):
                self.loss_terms[name] += float(value)

        if kwargs is not None:
            for name, value in kwargs.items():
                if value is None:
                    continue

                if isinstance(value, torch.Tensor):
                    value = value.detach().float().mean().item()

                if isinstance(value, (int, float)):
                    self.kwargs_terms[name] += float(value)

        if lr is not None:
            if isinstance(lr, torch.Tensor):
                lr = lr.detach().float().mean().item()

            if isinstance(lr, (int, float)):
                self.lr_values += float(lr)

        self.num_batches_seen += 1

    @torch.no_grad()
    def _dist_compute(self) -> dict[str, float]:
        """
        Document this function.

        Returns
        -------
        dict[str, float]
            Description not yet provided.
        """
        logs = {}

        for name in sorted(self.loss_terms):
            logs[name] = self._dist_average(self.loss_terms[name])

        for name in sorted(self.kwargs_terms):
            logs[name] = self._dist_average(self.kwargs_terms[name])

        logs["lr"] = self._dist_average(self.lr_values)

        self._aggregated_across_ranks = True
        return logs

    def _dist_average(self, tensor: float) -> float:
        """
        Document this function.

        Parameters
        ----------
        tensor : float
            Description not yet provided.

        Returns
        -------
        float
            Description not yet provided.
        """
        local = torch.tensor(
            [tensor, self.num_batches_seen],
            dtype=torch.float64,
            device=self.distributed.device,
        )

        self.distributed.all_reduce_sum(local)

        global_sum = local[0].item()
        global_count = int(local[1].item())

        if global_count == 0:
            return float("nan")
        else:
            return global_sum / global_count

    def record_epoch(
        self,
        logs: dict[str, float],
        replace_index: int = None,
        time_elapsed: float = None,
    ):
        """
        Document this function.

        Parameters
        ----------
        logs : dict[str, float]
            Description not yet provided.
        replace_index : int
            Description not yet provided.
        time_elapsed : float
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        RuntimeError
            Description not yet provided.
        ValueError
            Description not yet provided.
        """
        if not self._aggregated_across_ranks:
            raise RuntimeError(
                "Call _dist_compute() before record_epoch(), so losses are "
                "synchronized across all ranks."
            )

        if replace_index is None:
            for key, value in logs.items():
                self.epoch_metric_terms.setdefault(key, []).append(value)

            self.epoch_times.append(
                time_elapsed if time_elapsed is not None else np.nan
            )
            self.num_epochs_seen += 1

        else:
            for key, value in logs.items():
                if key not in self.epoch_metric_terms:
                    raise ValueError(
                        f"Cannot replace metric {key!r}; it was not previously recorded."
                    )

                self.epoch_metric_terms[key][replace_index] = value

            self.epoch_times[replace_index] = (
                time_elapsed if time_elapsed is not None else np.nan
            )

        self.epochs_submitted = True
        self.reset_batch_losses()

        return logs

    def reset_batch_losses(self):
        """
        Document this function.

        Warns
        -----
        UserWarning
            Description not yet provided.
        """
        if self.epochs_submitted:
            self.loss_terms = defaultdict(float)
            self.kwargs_terms = defaultdict(float)
            self.lr_values = 0.0
            self.num_batches_seen = 0
            self._aggregated_across_ranks = False
        else:
            warnings.warn(
                "\n you are resetting batch losses before submitting any epochs. \n"
            )

    @classmethod
    def plot(
        cls,
        aggregator_list: list["MetricsAggregator"],
        color_styles_list: list[tuple[str, str]] = None,
        plot_dir: str | Path | None = None,
        figsize=(8, 5),
    ) -> None:
        """
        Document this function.

        Parameters
        ----------
        aggregator_list : list['MetricsAggregator']
            Description not yet provided.
        color_styles_list : list[tuple[str, str]]
            Description not yet provided.
        plot_dir : str | Path | None
            Description not yet provided.
        figsize : Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        if plot_dir is None:
            plot_dir = Path(RuntimeContext.GLOBAL_FIGURES_DIR)
        else:
            plot_dir = Path(plot_dir)

        if len(aggregator_list) == 0:
            raise ValueError("Specify at least one aggregator.")

        num_epochs = {
            aggregator.num_epochs_seen
            for aggregator in aggregator_list
            if aggregator is not None
        }
        if len(num_epochs) != 1:
            raise ValueError(
                "All aggregators must have recorded the same number of epochs."
            )
        num_epochs = next(iter(num_epochs))

        if num_epochs == 0:
            raise ValueError("No epochs have been recorded yet.")

        metric_lengths = {
            len(list(aggregator.epoch_metric_terms.values())[0])
            for aggregator in aggregator_list
            if aggregator is not None
        }
        if len(metric_lengths) != 1:
            raise ValueError(
                "All aggregators must have the same number of epoch items."
            )
        metric_lengths = next(iter(metric_lengths))

        loss_kinds: set[str] = set()
        epochs_range = np.arange(num_epochs - metric_lengths + 1, num_epochs + 1)

        rng = random.Random()
        random_colors = [
            "tab:green",
            "tab:blue",
            "tab:orange",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        random_linestyles = [
            "solid",
            "dashed",
            "dashdot",
            "dotted",
        ]

        style_by_name: dict[str, tuple[str, str]] = {}

        for ind, aggregator in enumerate(aggregator_list):
            if aggregator is not None:
                loss_kinds.update(list(aggregator.epoch_metric_terms.keys()))

                if color_styles_list is None:
                    if "train" in aggregator.name.lower():
                        style_by_name[aggregator.name] = ("tab:blue", "solid")
                        random_linestyles.remove("solid")
                        random_colors.remove("tab:blue")

                    elif "val" in aggregator.name.lower():
                        style_by_name[aggregator.name] = ("tab:orange", "dashed")
                        random_linestyles.remove("dashed")
                        random_colors.remove("tab:orange")

                    else:
                        random_color = rng.choice(random_colors)
                        random_style = rng.choice(random_linestyles)
                        style_by_name[aggregator.name] = (random_color, random_style)
                        random_linestyles.remove(random_style)
                        random_colors.remove(random_color)

                else:
                    style_by_name[aggregator.name] = color_styles_list[ind]

        for loss_name in list(loss_kinds):
            fig, ax = plt.subplots(1, 1, figsize=figsize)

            for aggregator in aggregator_list:
                if aggregator is not None:
                    epoch_losses = aggregator.epoch_metric_terms.get(loss_name)

                    if epoch_losses is None or len(epoch_losses) == 0:
                        continue

                    color, style = style_by_name[aggregator.name]

                    ax.plot(
                        epochs_range,
                        epoch_losses,
                        color=color,
                        linestyle=style,
                        label=f"{aggregator.name}",
                    )

            ax.legend()
            ax.set_xlabel("Epochs")
            ax.set_ylabel(loss_name)

            safe_loss_name = loss_name.replace("/", "_")

            for old_plot in plot_dir.glob(f"epoch_*_{safe_loss_name}.png"):
                old_plot.unlink()

            plt.savefig(plot_dir / f"epoch_{num_epochs}_{safe_loss_name}.png")
            plt.close()

        _, ax = plt.subplots(1, 1, figsize=figsize)

        for aggregator in aggregator_list:
            if aggregator is not None:
                epoch_times = aggregator.epoch_times
                color, style = style_by_name[aggregator.name]

                if len(epoch_times) == 0:
                    continue

                ax.plot(
                    epochs_range,
                    epoch_times,
                    color=color,
                    linestyle=style,
                    label=f"{aggregator.name}",
                )

        ax.legend()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("epoch training times")

        for old_plot in plot_dir.glob("epoch_*_times.png"):
            old_plot.unlink()

        plt.savefig(plot_dir / f"epoch_{num_epochs}_times.png")
        plt.close()

    def state_dict(self):
        """
        Document this function.

        Returns
        -------
        Any
            Description not yet provided.
        """
        return {
            "name": self.name,
            "epoch_metric_terms": self.epoch_metric_terms,
            "epoch_times": self.epoch_times,
            "num_epochs_seen": self.num_epochs_seen,
        }

    def load_state_dict(self, state_dict):
        """
        Document this function.

        Parameters
        ----------
        state_dict : Any
            Description not yet provided.
        """
        self.name = state_dict.get("name")
        self.epoch_metric_terms = state_dict.get("epoch_metric_terms", None)
        self.epoch_times = state_dict.get("epoch_times", None)
        self.num_epochs_seen = state_dict.get("num_epochs_seen", 0)
        self.reset_batch_losses()


@dataclasses.dataclass
class RunningCovariance:
    """
    Document this class.

    Parameters
    ----------
    distributed : Distributed
        Description not yet provided.
    sum_x : torch.Tensor | None
        Description not yet provided.
    sum_xxT : torch.Tensor | None
        Description not yet provided.
    count : torch.Tensor | None
        Description not yet provided.
    """

    distributed: Distributed
    sum_x: torch.Tensor | None = None
    sum_xxT: torch.Tensor | None = None
    count: torch.Tensor | None = None

    def update(self, x: torch.Tensor):
        """
        Document this function.

        Parameters
        ----------
        x : torch.Tensor
            Description not yet provided.
        """
        x = x.detach().double()

        batch_sum = x.sum(dim=0)
        batch_xxT = x.T @ x
        batch_count = torch.tensor(
            x.shape[0],
            device=x.device,
            dtype=torch.float64,
        )

        if self.sum_x is None:
            self.sum_x = batch_sum
            self.sum_xxT = batch_xxT
            self.count = batch_count
        else:
            self.sum_x += batch_sum
            self.sum_xxT += batch_xxT
            self.count += batch_count

    def distributed_reduce(self):
        """
        Document this function.
        """
        self.distributed.all_reduce_sum(self.sum_x)
        self.distributed.all_reduce_sum(self.sum_xxT)
        self.distributed.all_reduce_sum(self.count)

    def finalize(self, print_checks=False):
        """
        Document this function.

        Parameters
        ----------
        print_checks : Any
            Description not yet provided.

        Returns
        -------
        Any
            Description not yet provided.

        Raises
        ------
        ValueError
            Description not yet provided.
        """
        mean = self.sum_x / self.count

        if self.count <= 1:
            raise ValueError("Need at least two samples to compute covariance.")

        cov = (self.sum_xxT - self.count * torch.outer(mean, mean)) / (self.count - 1)

        cov = 0.5 * (cov + cov.T)

        if print_checks:
            eigvals = torch.linalg.eigvalsh(cov.double())

            print("symmetry error:", (cov - cov.T).abs().max())
            print("minimum eigenvalue:", eigvals.min())
            print("count:", self.count)
            print("dimension:", cov.shape[-1])
            print("N:", int(self.count.item()))
            print("D:", cov.shape[-1])

        return mean.cpu(), cov.cpu()

import numpy as np
import dataclasses
import torch
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
from cccma_ppp.generic.distributed import Distributed
from cccma_ppp.generic.runtime import RuntimeContext
import random
from pathlib import Path


@dataclasses.dataclass
class MetricsAggregator:
    """
    Utility class for aggregating and tracking training metrics across batches and epochs,
    with support for distributed synchronization.

    Parameters
    ----------
    distributed : Distributed
        Distributed training utility used for cross-rank aggregation.
    name : str
        Name of the metric group (e.g., "train" or "val").
    epoch_loss_terms : dict of str to list of float, optional
        Pre-existing recorded epoch losses.
    epoch_times : list of float, optional
        Recorded epoch durations.
    num_epochs_seen : int, optional
        Number of epochs already processed.
    """

    distributed: Distributed
    name: str

    epoch_loss_terms: dict[str, list[float]] | None = None
    epoch_times: list[float] | None = None
    num_epochs_seen: int = 0

    def __post_init__(self):
        """
        Initialize internal state and validate provided historical metrics.

        Raises
        ------
        AssertionError
            If historical loss lists or timings are inconsistent in length.
        """

        self.loss_terms = defaultdict(float)
        self.num_batches_seen = 0

        self.epochs_submitted = False
        self._aggregated_across_ranks = False

        if self.epoch_loss_terms is not None:
            lengths = {len(v) for v in self.epoch_loss_terms.values()}
            assert len(lengths) == 1, (
                f"Not all loss lists have the same length: {lengths}"
            )
            if self.num_epochs_seen == 0:
                self.num_epochs_seen = next(iter(lengths), 0)

        else:
            self.epoch_loss_terms = {}

        if self.epoch_times is not None:
            assert self.epoch_loss_terms is not None, (
                "must specify corresponding loss terms."
            )
            assert len(self.epoch_times) == next(iter(lengths), 0), (
                "length of epoch_times is inconsistent with epoch_loss_terms."
            )

        else:
            self.epoch_times = []

    @torch.no_grad()
    def record(self, loss_dict: dict[str, torch.Tensor | int | float]) -> None:
        """
        Accumulate batch-level loss values.

        Parameters
        ----------
        loss_dict : dict of str to tensor or float
            Dictionary of loss components from a batch.

        Returns
        -------
        None
        """

        for name, value in loss_dict.items():
            if value is None:
                continue

            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()

            if isinstance(value, (int, float)):
                self.loss_terms[name] += float(value)

        self.num_batches_seen += 1

    @torch.no_grad()
    def _dist_compute(self) -> dict[str, float]:
        """
        Compute averaged metrics across all distributed ranks.

        Returns
        -------
        dict of str to float
            Averaged loss values.

        Notes
        -----
        Must be called before recording epoch-level statistics.
        """

        logs = {}

        for name in sorted(self.loss_terms):
            local = torch.tensor(
                [self.loss_terms[name], self.num_batches_seen],
                dtype=torch.float64,
                device=self.distributed.device,
            )

            self.distributed.all_reduce_sum(local)

            global_sum = local[0].item()
            global_count = int(local[1].item())

            if global_count == 0:
                logs[name] = float("nan")
            else:
                logs[name] = global_sum / global_count

        self._aggregated_across_ranks = True
        return logs

    def record_epoch(
        self,
        logs: dict[str, float],
        replace_index: int = None,
        time_elapsed: float = None,
    ):
        """
        Store epoch-level metrics.

        Parameters
        ----------
        logs : dict of str to float
            Aggregated metric values.
        replace_index : int or None, optional
            Index to replace existing epoch entry.
        time_elapsed : float or None, optional
            Time taken for the epoch.

        Returns
        -------
        dict
            Recorded metrics.

        Raises
        ------
        RuntimeError
            If distributed aggregation has not been performed.
        ValueError
            If attempting to replace a non-existing metric.
        """
        if not self._aggregated_across_ranks:
            raise RuntimeError(
                "Call _dist_compute() before record_epoch(), so losses are "
                "synchronized across all ranks."
            )

        if replace_index is None:
            for key, value in logs.items():
                self.epoch_loss_terms.setdefault(key, []).append(value)

            self.epoch_times.append(
                time_elapsed if time_elapsed is not None else np.nan
            )
            self.num_epochs_seen += 1

        else:
            for key, value in logs.items():
                if key not in self.epoch_loss_terms:
                    raise ValueError(
                        f"Cannot replace metric {key!r}; it was not previously recorded."
                    )

                self.epoch_loss_terms[key][replace_index] = value

            self.epoch_times[replace_index] = (
                time_elapsed if time_elapsed is not None else np.nan
            )

        self.epochs_submitted = True
        self.reset_batch_losses()

        return logs

    def reset_batch_losses(self):
        """
        Reset accumulated batch metrics.

        Returns
        -------
        None

        Warns
        -----
        UserWarning
            If called before any epoch has been submitted.
        """

        if self.epochs_submitted:
            self.loss_terms = defaultdict(float)
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
        Plot loss curves and epoch times for one or more aggregators.

        Parameters
        ----------
        aggregator_list : list of MetricsAggregator
            List of aggregators to plot.
        color_styles_list : list of tuple, optional
            List of (color, linestyle) for each aggregator.
        plot_dir : str or pathlib.Path or None, optional
            Directory to save plots.
        figsize : tuple, optional
            Size of the figure.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If aggregators are inconsistent or no data is available.
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

        loss_lengths = {
            len(list(aggregator.epoch_loss_terms.values())[0])
            for aggregator in aggregator_list
            if aggregator is not None
        }
        if len(loss_lengths) != 1:
            raise ValueError(
                "All aggregators must have the same number of epoch items."
            )
        loss_lengths = next(iter(loss_lengths))

        loss_kinds: set[str] = set()
        epochs_range = np.arange(num_epochs - loss_lengths + 1, num_epochs + 1)

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
                loss_kinds.update(list(aggregator.epoch_loss_terms.keys()))

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
                    epoch_losses = aggregator.epoch_loss_terms.get(loss_name)

                    if epoch_losses is None:
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
        Return serializable state of aggregator.

        Returns
        -------
        dict
            Aggregator state.
        """

        return {
            "name": self.name,
            "epoch_loss_terms": self.epoch_loss_terms,
            "epoch_times": self.epoch_times,
            "num_epochs_seen": self.num_epochs_seen,
        }

    def load_state_dict(self, state_dict):
        """
        Load aggregator state from dictionary.

        Parameters
        ----------
        state_dict : dict
            State dictionary.

        Returns
        -------
        None
        """

        self.name = state_dict.get("name")
        self.epoch_loss_terms = state_dict.get("epoch_loss_terms", None)
        self.epoch_times = state_dict.get("epoch_times", None)
        self.num_epochs_seen = state_dict.get("num_epochs_seen", 0)
        self.reset_batch_losses()

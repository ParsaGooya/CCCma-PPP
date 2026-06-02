import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from src.cccma_ppp.loss.loss import Losspipeline
from src.cccma_ppp.loss.loss_abc import lossABC


@Losspipeline.register("mse")
class WeightedMSE(lossABC):
    """
    Weighted mean squared error loss with optional low-resolution pooling.
    """

    def __init__(
        self,
        weights: xr.DataArray,
        reduction="mean",
        num_output_dimensions: int = 2,
        low_ress_kernel_size: int = None,
        hyperparam=1.0,
        min_threshold=0,
        max_threshold=0,
        **kwargs,
    ):
        """
        Initialize WeightedMSE loss.

        Parameters
        ----------
        weights : xr.DataArray
            Spatial or variable weighting array.
        reduction : str, optional
            Reduction method ('mean' or 'sum').
        num_output_dimensions : int, optional
            Number of spatial output dimensions.
        low_ress_kernel_size : int, optional
            Kernel size for downsampling.
        hyperparam : float, optional
            Scaling factor for thresholded regions.
        min_threshold : float, optional
            Lower threshold for conditional scaling.
        max_threshold : float, optional
            Upper threshold for conditional scaling.

        Returns
        -------
        None
        """

        super().__init__()
        self.reduction = reduction
        self.hyperparam = hyperparam
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.num_output_dimensions = num_output_dimensions
        self.low_ress_kernel_size = low_ress_kernel_size

        has_channels = "channels" in weights.dims

        weights_mask = torch.from_numpy(
            weights.where(np.isnan(weights), 1).fillna(0).to_numpy()
        ).float()
        weights = torch.from_numpy(weights.fillna(0).to_numpy()).float()

        if self.low_ress_kernel_size is not None:
            assert low_ress_kernel_size % 2 == 1, "choose odd kernel size"

            if not has_channels:
                weights_mask = weights_mask.unsqueeze(0)  # C x ...
                weights = weights.unsqueeze(0)  # C x ...

            if self.num_output_dimensions == 1:
                self.average_pool = F.avg_pool1d

            elif self.num_output_dimensions == 2:
                self.average_pool = F.avg_pool2d

            else:
                raise NotImplementedError(
                    "not designed for dataset with number of output dimensions > 2 (except channels)"
                )

            weights_mask = self._downsample(weights_mask)
            weights_mask = (
                weights_mask == 1
            ).float()  ## just keep grid weights where all high res grids are available
            weights = self._downsample(weights)

            if not has_channels:
                weights_mask = weights_mask.squeeze(0)

        weights = weights * weights_mask

        self.register_buffer("weights", weights)

    def _downsample(self, tensor):
        """
        Downsample tensor using average pooling.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Downsampled tensor.
        """

        squeeze = False
        if (
            len(tensor.shape) == self.num_output_dimensions + 1
        ):  # check for static masks (space + channel)
            tensor = tensor.unsqueeze(0)
            squeeze = True

        tensor_ = self.average_pool(
            tensor,
            kernel_size=self.low_ress_kernel_size,
            stride=self.low_ress_kernel_size // 2,
        )

        if squeeze:
            tensor_ = tensor_.squeeze(0)

        return tensor_

    def forward(
        self,
        data,
        target,
        target_mask=None,
        generative_modeling=False,
        generator=False,
        print_loss=False,
    ):
        """
        Compute weighted mean squared error.

        Parameters
        ----------
        data : torch.Tensor
            Predicted output.
        target : torch.Tensor
            Ground truth target.
        target_mask : torch.Tensor, optional
            Mask for valid targets.
        generative_modeling : bool, optional
            Whether latent ensemble structure is used.
        generator : bool, optional
            Whether input includes ensemble dimension.
        print_loss : bool, optional
            Whether to print loss value.

        Returns
        -------
        torch.Tensor
            Computed loss.

        Raises
        ------
        AssertionError
            If input shapes are mismatched.
        """

        if generator:
            _check_generator_structure(data, target)
            data = data.mean(0)

        assert data.shape == target.shape, (
            f"for MSE data and target must have the same shape, got {data.shape} vs {target.shape}"
        )

        y = target
        y_hat = data

        if self.low_ress_kernel_size is not None:
            if generative_modeling:  ##generative_modeling models generate a latent ensemble of outputs (Z x batch x channels x ...)
                if target_mask is not None and target_mask.shape == y.shape:
                    target_mask = torch.flatten(target_mask, start_dim=0, end_dim=1)
                y = torch.flatten(y, start_dim=0, end_dim=1)
                y_hat = torch.flatten(y_hat, start_dim=0, end_dim=1)

            y = self._downsample(y)
            if target_mask is not None:
                target_mask = self._downsample(target_mask)
            y_hat = self._downsample(y_hat)

        m = torch.ones_like(y)
        m[(y < self.min_threshold) & (y_hat >= 0)] *= self.hyperparam
        m[(y > self.max_threshold) & (y_hat <= 0)] *= self.hyperparam

        SE = (y_hat - y) ** 2 * m

        loss = self._aggregate(SE, target_mask)

        if print_loss:
            self._print_loss(loss)

        return loss

    def _print_loss(self, loss):
        """
        Print formatted loss.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value.

        Returns
        -------
        None
        """

        if self.low_ress_kernel_size is not None:
            print(f"MSE_lowress : {loss.item():.5f}")
        else:
            print(f"MSE : {loss.item():.5f}")

    def _aggregate(self, loss, mask):
        """
        Apply reduction to weighted loss.

        Parameters
        ----------
        loss : torch.Tensor
            Element-wise loss.
        mask : torch.Tensor or None
            Optional mask.

        Returns
        -------
        torch.Tensor
            Aggregated loss.
        """

        weight = self.weights
        loss = loss * weight
        if mask is not None:
            weight = weight * mask
            loss = loss * mask

        if self.reduction == "mean":
            loss = loss.sum() / (torch.ones_like(loss) * weight).sum()

        elif self.reduction == "sum":
            loss = torch.sum(
                loss,
                dim=tuple(-i for i in np.arange(1, self.num_output_dimensions + 1 + 1)),
            ).mean()  ## sum over output dimensions + channels

        else:
            raise NotImplementedError(
                "Other reduction methods than sum and mean are not defined."
            )

        return loss


@Losspipeline.register("crps")
class WeightedCRPS(lossABC):
    """
    Weighted Continuous Ranked Probability Score for ensemble predictions.
    """

    def __init__(
        self,
        weights: xr.DataArray,
        reduction="mean",
        num_output_dimensions: int = 2,
        low_ress_kernel_size: int = None,
        **kwargs,
    ):
        """
        Initialize WeightedCRPS loss.

        Parameters
        ----------
        weights : xr.DataArray
            Spatial or variable weighting array.
        reduction : str, optional
            Reduction method.
        num_output_dimensions : int, optional
            Number of spatial dimensions.
        low_ress_kernel_size : int, optional
            Kernel size for downsampling.

        Returns
        -------
        None
        """

        super().__init__()
        self.reduction = reduction
        self.num_output_dimensions = num_output_dimensions
        self.low_ress_kernel_size = low_ress_kernel_size

        has_channels = "channels" in weights.dims

        weights_mask = torch.from_numpy(
            weights.where(np.isnan(weights), 1).fillna(0).to_numpy()
        ).float()
        weights = torch.from_numpy(weights.fillna(0).to_numpy()).float()

        if self.low_ress_kernel_size is not None:
            assert low_ress_kernel_size % 2 == 1, "choose odd kernel size"

            if not has_channels:
                weights_mask = weights_mask.unsqueeze(0)  # C x ...
                weights = weights.unsqueeze(0)  # C x ...

            if self.num_output_dimensions == 1:
                self.average_pool = F.avg_pool1d

            elif self.num_output_dimensions == 2:
                self.average_pool = F.avg_pool2d

            else:
                raise NotImplementedError(
                    "not designed for dataset with number of output dimensions > 2 (except channels)"
                )

            weights_mask = self._downsample(weights_mask)
            weights_mask = (
                weights_mask == 1
            ).float()  ## just keep grid weights where all high res grids are available
            weights = self._downsample(weights)

            if not has_channels:
                weights_mask = weights_mask.squeeze(0)
                weights = weights.squeeze(0)

        weights = weights * weights_mask

        self.register_buffer("weights", weights)

    def _downsample(self, tensor):
        """
        Downsample tensor using average pooling.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Downsampled tensor.
        """

        squeeze = False
        if (
            len(tensor.shape) == self.num_output_dimensions + 1
        ):  # check for static masks (space + channel)
            tensor = tensor.unsqueeze(0)
            squeeze = True

        tensor_ = self.average_pool(
            tensor,
            kernel_size=self.low_ress_kernel_size,
            stride=self.low_ress_kernel_size // 2,
        )

        if squeeze:
            tensor_ = tensor_.squeeze(0)

        return tensor_

    def forward(
        self,
        data,
        target,
        target_mask=None,
        generative_modeling=False,
        generator=True,
        print_loss=False,
    ):
        """
        Compute CRPS for ensemble predictions.

        Parameters
        ----------
        data : torch.Tensor
            Ensemble predictions.
        target : torch.Tensor
            Ground truth.
        target_mask : torch.Tensor, optional
            Mask for valid targets.
        generative_modeling : bool, optional
            Whether latent ensemble structure is used.
        generator : bool, optional
            Whether generator output is used.
        print_loss : bool, optional
            Whether to print loss.

        Returns
        -------
        torch.Tensor
            Computed CRPS loss.

        Raises
        ------
        AssertionError
            If generator structure is invalid.
        """

        assert generator, (
            "generator cannot be False as a step_argument in Losspipeline.forward()"
        )
        _check_generator_structure(data, target)

        if generative_modeling:  ##generative_modeling models generate a latent ensemble  (Z x batch x channels x ...)
            if target_mask is not None and target_mask.shape == target.shape:
                target_mask = torch.flatten(target_mask, start_dim=0, end_dim=1)
            y = torch.flatten(target, start_dim=0, end_dim=1)
            y_hat = torch.flatten(
                data, start_dim=1, end_dim=2
            )  # for CRPS training an extra dimension of output ensembles should be on the left.
        else:
            y = target
            y_hat = data

        opts = dict(device=y_hat.device, dtype=y_hat.dtype)
        num_samples = y_hat.size(0)

        if self.low_ress_kernel_size is not None:
            y = self._downsample(y)
            if target_mask is not None:
                target_mask = self._downsample(target_mask)

            E, B = y_hat.shape[:2]
            y_hat = y_hat.reshape(
                E * B, *y_hat.shape[2:]
            )  ##reshape output ensemble to calcualte the average pool first and then reshape again.
            y_hat = self._downsample(y_hat)
            y_hat = y_hat.reshape(E, B, *y_hat.shape[1:])

        if num_samples == 1:
            crps = (y_hat[0] - y).abs()

        else:
            y_hat = y_hat.sort(dim=0).values
            diff = y_hat[1:] - y_hat[:-1]
            weight_crps = torch.arange(1, num_samples, **opts) * torch.arange(
                num_samples - 1, 0, -1, **opts
            )
            weight_crps = weight_crps.reshape(
                weight_crps.shape + (1,) * (diff.dim() - 1)
            )
            crps = (y_hat - y).abs().mean(0) - (diff * weight_crps).sum(
                0
            ) / num_samples**2

        loss = self._aggregate(crps, target_mask)

        if print_loss:
            self._print_loss(loss)

        return loss

    def _print_loss(self, loss):
        """
        Print formatted CRPS loss.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value.

        Returns
        -------
        None
        """

        if self.low_ress_kernel_size is not None:
            print(f"CRPS_lowress : {loss.item():.5f}")
        else:
            print(f"CRPS : {loss.item():.5f}")

    def _aggregate(self, loss, mask):
        """
        Apply reduction to CRPS loss.

        Parameters
        ----------
        loss : torch.Tensor
            Element-wise CRPS.
        mask : torch.Tensor or None
            Optional mask.

        Returns
        -------
        torch.Tensor
            Aggregated loss.
        """

        weight = self.weights
        loss = loss * weight
        if mask is not None:
            weight = weight * mask
            loss = loss * mask

        if self.reduction.lower() == "mean":
            loss = loss.sum() / (torch.ones_like(loss) * weight).sum()

        elif self.reduction.lower() == "sum":
            loss = torch.sum(
                loss,
                dim=tuple(-i for i in np.arange(1, self.num_output_dimensions + 1 + 1)),
            ).mean()  ## sum over output dimensions + channels

        else:
            raise NotImplementedError(
                "Other reduction methods than sum and mean are not defined."
            )

        return loss


@Losspipeline.register("frobenius_norm")
class Frobenius_norm(lossABC):
    """
    Frobenius norm loss between covariance matrices.
    """

    def __init__(
        self,
        weights: xr.DataArray,
        reduction="mean",
        num_output_dimensions: int = 2,
        covariance_dim: str = "spatial",
    ):
        """
        Initialize Frobenius norm loss.

        Parameters
        ----------
        weights : xr.DataArray
            Weighting array.
        reduction : str, optional
            Reduction method.
        num_output_dimensions : int, optional
            Number of spatial dimensions.
        covariance_dim : str, optional
            Dimension over which covariance is computed.

        Returns
        -------
        None
        """

        super().__init__()

        self.covariance_dim = covariance_dim
        self.num_output_dimensions = num_output_dimensions
        self.reduction = reduction
        self.output_size = np.prod(weights.shape[-self.num_output_dimensions :])
        # self.has_channels = 'channels' in weights.dims
        # weights = torch.from_numpy(weights.fillna(0).to_numpy()).float()
        # if  self.has_channels:
        #     weights = torch.mean(weights,  dim = tuple(-i for i in np.arange(1, self.num_output_dimensions + 1))) ## mean over output dimensions to get a weight for channels
        #     self.register_buffer("weights", weights)
        # else:
        #     self.weights = None
        assert self.covariance_dim in ["spatial", "channel"]

    def forward(
        self, data, target, generative_modeling=False, generator=False, print_loss=False
    ):
        """
        Compute Frobenius norm between covariance matrices.

        Parameters
        ----------
        data : torch.Tensor
            Predicted output.
        target : torch.Tensor
            Ground truth.
        generative_modeling : bool, optional
            Whether ensemble structure is used.
        generator : bool, optional
            Whether generator structure is used.
        print_loss : bool, optional
            Whether to print loss.

        Returns
        -------
        torch.Tensor
            Computed loss.

        Raises
        ------
        AssertionError
            If input shapes are inconsistent.
        """

        if generator:
            _check_generator_structure(data, target)
            data = data.mean(0)

        assert data.shape == target.shape

        y = torch.flatten(target, start_dim=-self.num_output_dimensions, end_dim=-1)
        y_hat = torch.flatten(data, start_dim=-self.num_output_dimensions, end_dim=-1)

        if generative_modeling:  ##generative_modeling models generate an ensemble of outputs (enx x batch x channels x ...)
            y = torch.flatten(y, start_dim=0, end_dim=1)
            y_hat = torch.flatten(y_hat, start_dim=0, end_dim=1)

        if self.covariance_dim.lower() == "spatial":
            covariance_truth = torch.stack(
                [torch.cov(y[:, channel, ...].T) for channel in range(y.shape[1])]
            )
            covariance_prediction = torch.stack(
                [
                    torch.cov(y_hat[:, channel, ...].T)
                    for channel in range(y_hat.shape[1])
                ]
            )
            output_size = self.output_size
        elif self.covariance_dim.lower() == "channel":
            covariance_truth = torch.cov(y.permute(1, 0, 2).reshape(y.shape[1], -1))
            covariance_prediction = torch.cov(
                y_hat.permute(1, 0, 2).reshape(y_hat.shape[1], -1)
            )
            output_size = y.shape[1]

        frobenius_norm = torch.linalg.matrix_norm(
            covariance_prediction - covariance_truth, ord="fro"
        )

        loss = self._aggregate(frobenius_norm, output_size)

        if print_loss:
            self._print_loss(loss)

        return loss

    def _print_loss(self, loss):
        """
        Print formatted Frobenius loss.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value.

        Returns
        -------
        None
        """

        print(f"FLN : {loss.item():.5f}")

    def _aggregate(self, loss, output_size=None):
        """
        Apply reduction to Frobenius loss.

        Parameters
        ----------
        loss : torch.Tensor
            Loss value.
        output_size : int, optional
            Normalization factor.

        Returns
        -------
        torch.Tensor
            Aggregated loss.
        """

        if output_size is None:
            output_size = self.output_size

        if self.reduction.lower() == "mean":
            loss = loss / output_size
            # if self.has_channels:
            #     loss = (loss * self.weights ).sum() / self.weights.sum()
            # else:
            loss = loss.mean()

        elif self.reduction.lower() == "sum":
            # if self.has_channels:
            #     loss  =loss * (self.weights / self.weights.sum())
            loss = loss.sum()

        return loss


def _check_generator_structure(data, target):
    """
    Validate generator output structure.

    Parameters
    ----------
    data : torch.Tensor
        Model output tensor.
    target : torch.Tensor
        Target tensor.

    Returns
    -------
    bool
        True if structure is valid.

    Raises
    ------
    ValueError
        If structure is invalid.
    """
    bool = data.shape[1:] == (1,) * (data.dim() - target.dim() - 1) + target.shape
    if bool:
        return True
    else:
        raise ValueError(
            "Expected data to have one extra sample dim on left. "
            "Actual shapes: {} versus {}".format(data.shape, target.shape)
        )

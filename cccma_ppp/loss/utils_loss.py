import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from typing import Literal

from cccma_ppp.loss.loss import Losspipeline
from cccma_ppp.loss.loss_abc import lossABC, Reduction

from cccma_ppp.core.core_abc import GenerativeContext


CovarianceDim = Literal["spatial", "channel"]


@Losspipeline.register("mse")
class WeightedMSE(lossABC):
    """
    Compute a weighted mean squared error loss.

    Supports optional spatial downsampling, target masking, threshold-dependent
    error scaling, and averaging across generator samples.

    Parameters
    ----------
    weights : xr.DataArray
        Spatial and variable weights applied to squared errors.
    reduction : Reduction
        Method used to aggregate weighted squared errors.
    num_output_dimensions : int
        Number of dimensions in each model output, including the channel
        dimension.
    low_ress_kernel_size : int | None
        Kernel size used to spatially downsample predictions, targets, masks, and
        weights before computing the loss. If ``None``, no downsampling is
        applied.
    generative_context : GenerativeContext | None
        Context describing generative modeling and generator-based sampling.
    hyperparam : float
        Multiplicative penalty applied to errors satisfying the configured
        threshold conditions.
    min_threshold : float
        Lower target threshold used to identify predictions receiving the
        additional penalty.
    max_threshold : float
        Upper target threshold used to identify predictions receiving the
        additional penalty.
    **kwargs : Any
        Additional keyword arguments accepted for loss-pipeline compatibility.
    """

    def __init__(
        self,
        weights: xr.DataArray,
        reduction: Reduction = "mean",
        num_output_dimensions: int = 2,
        low_ress_kernel_size: int = None,
        generative_context: GenerativeContext | None = None,
        hyperparam=1.0,
        min_threshold=0,
        max_threshold=0,
        **kwargs,
    ):
        """
        Initialize a weighted mean squared error loss.

        Parameters
        ----------
        weights : xr.DataArray
            Spatial and variable weights applied to squared errors.
        reduction : Reduction
            Method used to aggregate weighted squared errors.
        num_output_dimensions : int
            Number of dimensions in each model output, including the channel
            dimension.
        low_ress_kernel_size : int | None
            Kernel size used to spatially downsample loss inputs. If ``None``, no
            downsampling is applied.
        generative_context : GenerativeContext | None
            Context describing generative modeling and generator-based sampling.
        hyperparam : float
            Multiplicative penalty applied to threshold-dependent errors.
        min_threshold : float
            Lower threshold used to identify penalized prediction errors.
        max_threshold : float
            Upper threshold used to identify penalized prediction errors.
        **kwargs : Any
            Additional keyword arguments accepted for loss-pipeline compatibility.

        Raises
        ------
        NotImplementedError
            If low-resolution loss is requested for an unsupported number of output
            dimensions.
        ValueError
            If ``low_ress_kernel_size`` is not an odd integer.
        """
        super().__init__()
        self.reduction = reduction
        self.hyperparam = hyperparam
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.num_output_dimensions = num_output_dimensions
        self.low_ress_kernel_size = low_ress_kernel_size
        self.generative_context = (
            generative_context
            if generative_context is not None
            else GenerativeContext()
        )

        has_channels = "channels" in weights.dims

        weights_mask = torch.from_numpy(
            weights.where(np.isnan(weights), 1).fillna(0).to_numpy()
        ).float()
        weights = torch.from_numpy(weights.fillna(0).to_numpy()).float()

        if self.low_ress_kernel_size is not None:
            if not self.low_ress_kernel_size % 2 == 1:
                raise ValueError("choose odd kernel size")

            if not has_channels:
                weights_mask = weights_mask.unsqueeze(0)
                weights = weights.unsqueeze(0)

            if self.num_output_dimensions == 2:
                self.average_pool = F.avg_pool1d

            elif self.num_output_dimensions == 3:
                self.average_pool = F.avg_pool2d

            else:
                raise NotImplementedError(
                    "not designed for dataset with number of output spatiotemporal dimensions > 2 (except channels)"
                )

            weights_mask = self._downsample(weights_mask)
            weights_mask = (weights_mask == 1).float()
            weights = self._downsample(weights)

            if not has_channels:
                weights_mask = weights_mask.squeeze(0)

        weights = weights * weights_mask

        self.register_buffer("weights", weights)

    def _downsample(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Downsample a tensor using average pooling.

        The configured low-resolution kernel size is used as the pooling kernel, with
        a stride equal to half the kernel size using integer division.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to downsample.

        Returns
        -------
        torch.Tensor
            Downsampled tensor with any temporarily added leading dimension removed.
        """
        squeeze = False
        if len(tensor.shape) == self.num_output_dimensions:
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
        data: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        print_loss=False,
    ) -> torch.Tensor:
        """
        Compute the weighted mean squared error.

        When generator sampling is enabled, predictions are averaged across the
        generator sample dimension before the error is calculated. Optional
        low-resolution processing downsamples predictions, targets, and masks before
        threshold-dependent penalties and weights are applied.

        Parameters
        ----------
        data : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Target values.
        target_mask : torch.Tensor | None
            Optional mask restricting values contributing to the loss.
        print_loss : bool
            Whether to print the aggregated loss.

        Returns
        -------
        torch.Tensor
            Aggregated weighted mean squared error.

        Raises
        ------
        RuntimeError
            If prediction and target tensors do not have matching shapes after
            generator samples are handled.
        """
        if self.generative_context.generator:
            _check_generator_structure(data, target)
            data = data.mean(0)

        if not data.shape == target.shape:
            raise RuntimeError(
                f"for MSE data and target must have the same shape, got {data.shape} vs {target.shape}"
            )

        y = target
        y_hat = data

        if self.low_ress_kernel_size is not None:
            if self.generative_context.generative_modeling:
                if target_mask is not None and target_mask.shape == y.shape:
                    target_mask = torch.flatten(target_mask, start_dim=0, end_dim=1)
                y = torch.flatten(y, start_dim=0, end_dim=1)
                y_hat = torch.flatten(y_hat, start_dim=0, end_dim=1)

            y = self._downsample(y)
            if target_mask is not None:
                target_mask = self._downsample(target_mask)
            y_hat = self._downsample(y_hat)

        m = torch.ones_like(y)
        m[(y < self.min_threshold) & (y_hat >= self.min_threshold)] *= self.hyperparam
        m[(y > self.max_threshold) & (y_hat <= self.max_threshold)] *= self.hyperparam

        SE = (y_hat - y) ** 2 * m

        loss = self._aggregate(SE, target_mask)

        if print_loss:
            self._print_loss(loss)

        return loss

    def _print_loss(self, loss):
        """
        Print the weighted mean squared error.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss value to print.
        """
        if self.low_ress_kernel_size is not None:
            print(f"MSE_lowress : {loss.item():.5f}")
        else:
            print(f"MSE : {loss.item():.5f}")

    def _aggregate(self, loss, mask):
        """
        Apply weights, masking, and reduction to squared errors.

        Parameters
        ----------
        loss : torch.Tensor
            Element-wise squared errors.
        mask : torch.Tensor | None
            Optional mask restricting elements contributing to the loss.

        Returns
        -------
        torch.Tensor
            Aggregated weighted loss.

        Raises
        ------
        NotImplementedError
            If the configured reduction is neither ``"mean"`` nor ``"sum"``.
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
                dim=tuple(-i for i in np.arange(1, self.num_output_dimensions + 1)),
            ).mean()

        else:
            raise NotImplementedError(
                "Other reduction methods than 'sum' and 'mean' are not defined."
            )

        return loss


@Losspipeline.register("crps")
class WeightedCRPS(lossABC):
    """
    Compute a weighted continuous ranked probability score.

    The loss evaluates an ensemble of generator predictions against target values
    and supports optional spatial downsampling and masking.

    Parameters
    ----------
    weights : xr.DataArray
        Spatial and variable weights applied to CRPS values.
    reduction : Reduction
        Method used to aggregate weighted CRPS values.
    num_output_dimensions : int
        Number of dimensions in each model output, including the channel
        dimension.
    low_ress_kernel_size : int | None
        Kernel size used to spatially downsample predictions, targets, masks, and
        weights before computing the loss. If ``None``, no downsampling is
        applied.
    generative_context : GenerativeContext | None
        Context describing generative modeling and generator-based sampling.
    kwargs : Any
        Additional keyword arguments accepted for loss-pipeline compatibility.
    """

    def __init__(
        self,
        weights: xr.DataArray,
        reduction: Reduction = "mean",
        num_output_dimensions: int = 2,
        low_ress_kernel_size: int = None,
        generative_context: GenerativeContext | None = None,
        **kwargs,
    ):
        """
        Initialize a weighted continuous ranked probability score.

        Parameters
        ----------
        weights : xr.DataArray
            Spatial and variable weights applied to CRPS values.
        reduction : Reduction
            Method used to aggregate weighted CRPS values.
        num_output_dimensions : int
            Number of dimensions in each model output, including the channel
            dimension.
        low_ress_kernel_size : int | None
            Kernel size used to spatially downsample loss inputs. If ``None``, no
            downsampling is applied.
        generative_context : GenerativeContext | None
            Context describing generative modeling and generator-based sampling.
        kwargs : Any
            Additional keyword arguments accepted for loss-pipeline compatibility.

        Raises
        ------
        NotImplementedError
            If low-resolution loss is requested for an unsupported number of output
            dimensions.
        ValueError
            If ``low_ress_kernel_size`` is not an odd integer.
        """
        super().__init__()
        self.reduction = reduction
        self.num_output_dimensions = num_output_dimensions
        self.low_ress_kernel_size = low_ress_kernel_size
        self.generative_context = (
            generative_context
            if generative_context is not None
            else GenerativeContext()
        )

        has_channels = "channels" in weights.dims

        weights_mask = torch.from_numpy(
            weights.where(np.isnan(weights), 1).fillna(0).to_numpy()
        ).float()
        weights = torch.from_numpy(weights.fillna(0).to_numpy()).float()

        if self.low_ress_kernel_size is not None:
            if not low_ress_kernel_size % 2 == 1:
                raise ValueError("choose odd kernel size")

            if not has_channels:
                weights_mask = weights_mask.unsqueeze(0)
                weights = weights.unsqueeze(0)

            if self.num_output_dimensions == 2:
                self.average_pool = F.avg_pool1d

            elif self.num_output_dimensions == 3:
                self.average_pool = F.avg_pool2d

            else:
                raise NotImplementedError(
                    "not designed for dataset with number of output spatiotemporal dimensions > 2 (except channels)"
                )

            weights_mask = self._downsample(weights_mask)
            weights_mask = (weights_mask == 1).float()
            weights = self._downsample(weights)

            if not has_channels:
                weights_mask = weights_mask.squeeze(0)
                weights = weights.squeeze(0)

        weights = weights * weights_mask

        self.register_buffer("weights", weights)

    def _downsample(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Downsample a tensor using average pooling.

        The configured low-resolution kernel size is used as the pooling kernel, with
        a stride equal to half the kernel size using integer division.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to downsample.

        Returns
        -------
        torch.Tensor
            Downsampled tensor with any temporarily added leading dimension removed.
        """
        squeeze = False
        if len(tensor.shape) == self.num_output_dimensions:
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
        data: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        print_loss=False,
    ) -> torch.Tensor:
        """
        Compute the weighted continuous ranked probability score.

        For a single generator sample, the score reduces to absolute error. For
        multiple samples, predictions are sorted and an ensemble correction term is
        computed from differences between adjacent samples. Generative model sample
        and batch dimensions are combined before loss calculation when required.

        Parameters
        ----------
        data : torch.Tensor
            Ensemble of model predictions.
        target : torch.Tensor
            Target values.
        target_mask : torch.Tensor | None
            Optional mask restricting values contributing to the loss.
        print_loss : bool
            Whether to print the aggregated loss.

        Returns
        -------
        torch.Tensor
            Aggregated weighted continuous ranked probability score.

        Raises
        ------
        RuntimeError
            If generator-based sampling is not enabled.
        """
        if not self.generative_context.generator:
            raise RuntimeError(
                "generator cannot be False as a step_argument in Losspipeline.forward()"
            )
        _check_generator_structure(data, target)

        if self.generative_context.generative_modeling:
            if target_mask is not None and target_mask.shape == target.shape:
                target_mask = torch.flatten(target_mask, start_dim=0, end_dim=1)
            y = torch.flatten(target, start_dim=0, end_dim=1)
            y_hat = torch.flatten(data, start_dim=1, end_dim=2)
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
            y_hat = y_hat.reshape(E * B, *y_hat.shape[2:])
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
        Print the continuous ranked probability score.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss value to print.
        """
        if self.low_ress_kernel_size is not None:
            print(f"CRPS_lowress : {loss.item():.5f}")
        else:
            print(f"CRPS : {loss.item():.5f}")

    def _aggregate(self, loss, mask):
        """
        Apply weights, masking, and reduction to CRPS values.

        Parameters
        ----------
        loss : torch.Tensor
            Element-wise continuous ranked probability scores.
        mask : torch.Tensor | None
            Optional mask restricting elements contributing to the loss.

        Returns
        -------
        torch.Tensor
            Aggregated weighted continuous ranked probability score.

        Raises
        ------
        NotImplementedError
            If the configured reduction is neither ``"mean"`` nor ``"sum"``.
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
                dim=tuple(-i for i in np.arange(1, self.num_output_dimensions + 1)),
            ).mean()

        else:
            raise NotImplementedError(
                "Other reduction methods than 'sum' and 'mean' are not defined."
            )

        return loss


@Losspipeline.register("frobenius_norm")
class Frobenius_norm(lossABC):
    """
    Measure covariance mismatch using the Frobenius norm.

    Prediction and target covariance matrices are computed across either spatial
    features or output channels, and their difference is measured using the
    Frobenius matrix norm.

    Parameters
    ----------
    weights : xr.DataArray
        Weight array whose shape is used to determine the spatial output size for
        loss normalization.
    reduction : Reduction
        Method used to aggregate the covariance discrepancy.
    num_output_dimensions : int
        Number of dimensions in each model output, including the channel
        dimension.
    generative_context : GenerativeContext | None
        Context describing generative modeling and generator-based sampling.
    covariance_dim : CovarianceDim
        Dimension across which covariance structure is compared. Supported values
        are ``"spatial"`` and ``"channel"``.
    """

    def __init__(
        self,
        weights: xr.DataArray,
        reduction: Reduction = "mean",
        num_output_dimensions: int = 2,
        generative_context: GenerativeContext | None = None,
        covariance_dim: CovarianceDim = "spatial",
    ):
        """
        Initialize the covariance Frobenius norm loss.

        Parameters
        ----------
        weights : xr.DataArray
            Weight array whose shape is used to determine the spatial output size for
            normalization.
        reduction : Reduction
            Method used to aggregate the covariance discrepancy.
        num_output_dimensions : int
            Number of dimensions in each model output, including the channel
            dimension.
        generative_context : GenerativeContext | None
            Context describing generative modeling and generator-based sampling.
        covariance_dim : CovarianceDim
            Whether covariance is evaluated across spatial features or output
            channels.
        """
        super().__init__()

        self.covariance_dim = covariance_dim
        self.num_output_dimensions = num_output_dimensions
        self.reduction = reduction
        self.output_size = np.prod(weights.shape[-self.num_output_dimensions + 1 :])
        self.generative_context = (
            generative_context
            if generative_context is not None
            else GenerativeContext()
        )

    def forward(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        print_loss: bool = False,
    ) -> torch.Tensor:
        """
        Compute the Frobenius norm of the covariance difference.

        Generator samples are averaged before covariance calculation. For spatial
        covariance, a covariance matrix is calculated independently for each channel.
        For channel covariance, sample and spatial dimensions are combined before
        calculating covariance between channels.

        Parameters
        ----------
        data : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Target values.
        target_mask : torch.Tensor | None
            Optional target mask. This argument is accepted for loss-pipeline
            compatibility but is not used in the covariance calculation.
        print_loss : bool
            Whether to print the aggregated loss.

        Returns
        -------
        torch.Tensor
            Aggregated Frobenius norm of the difference between predicted and target
            covariance matrices.

        Raises
        ------
        AssertionError
            If prediction and target tensors do not have matching shapes.
        """
        if self.generative_context.generator:
            _check_generator_structure(data, target)
            data = data.mean(0)

        assert data.shape == target.shape

        y = torch.flatten(target, start_dim=-self.num_output_dimensions + 1, end_dim=-1)
        y_hat = torch.flatten(
            data, start_dim=-self.num_output_dimensions + 1, end_dim=-1
        )

        if self.generative_context.generative_modeling:
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
        Print the covariance Frobenius norm loss.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss value to print.
        """
        print(f"FLN : {loss.item():.5f}")

    def _aggregate(self, loss, output_size=None):
        """
        Aggregate the covariance Frobenius norm.

        For mean reduction, the norm is normalized by the applicable output size
        before averaging. Sum reduction directly sums the norm values.

        Parameters
        ----------
        loss : torch.Tensor
            Frobenius norm values to aggregate.
        output_size : int | None
            Size used to normalize the loss for mean reduction. If ``None``, the
            spatial output size determined during initialization is used.

        Returns
        -------
        torch.Tensor
            Aggregated covariance loss.

        Raises
        ------
        NotImplementedError
            If the configured reduction is neither ``"mean"`` nor ``"sum"``.
        """
        if output_size is None:
            output_size = self.output_size

        if self.reduction.lower() == "mean":
            loss = loss / output_size

            loss = loss.mean()

        elif self.reduction.lower() == "sum":
            loss = loss.sum()

        else:
            raise NotImplementedError(
                "Other reduction methods than 'sum' and 'mean' are not defined."
            )

        return loss


def _check_generator_structure(data: torch.Tensor, target: torch.Tensor):
    """
    Validate the relationship between generator output and target shapes.

    The prediction tensor must contain an additional leading generator sample
    dimension. Any additional dimensions between that sample dimension and the
    target dimensions must have size one.

    Parameters
    ----------
    data : torch.Tensor
        Generator output tensor.
    target : torch.Tensor
        Target tensor defining the expected prediction structure.

    Returns
    -------
    bool
        ``True`` when the generator output has the expected structure.

    Raises
    ------
    ValueError
        If the generator output shape is incompatible with the target shape.
    """
    bool = data.shape[1:] == (1,) * (data.dim() - target.dim() - 1) + target.shape
    if bool:
        return True
    else:
        raise ValueError(
            "Expected data to have one extra sample dim on left. "
            "Actual shapes: {} versus {}".format(data.shape, target.shape)
        )

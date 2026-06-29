from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from typing import Literal

from cccma_ppp.loss import Losspipeline
from cccma_ppp.loss.loss_abc import lossABC, Reduction


CovarianceDim = Literal["spatial", "channel"]


@Losspipeline.register("mse")
class WeightedMSE(lossABC):
    """
    Weighted mean squared error loss.

    Parameters
    ----------
    weights : xr.DataArray
        Spatial or variable weights.
    reduction : {"mean", "sum"}, optional
        Reduction method.
    num_output_dimensions : int, optional
        Number of spatial/output dimensions.
    low_ress_kernel_size : int or None, optional
        Kernel size for optional low-resolution downsampling.
    hyperparam : float, optional
        Scaling factor for asymmetric penalty.
    min_threshold : float, optional
        Lower threshold for asymmetric penalty.
    max_threshold : float, optional
        Upper threshold for asymmetric penalty.
    """

    def __init__(
        self,
        weights: xr.DataArray,
        reduction: Reduction = "mean",
        num_output_dimensions: int = 2,
        low_ress_kernel_size: int = None,
        hyperparam=1.0,
        min_threshold=0,
        max_threshold=0,
        **kwargs,
    ):
        """
        Initialize weighted MSE loss.

        Parameters
        ----------
        weights : xr.DataArray
            Spatial or variable weights.
        reduction : {"mean", "sum"}, optional
            Reduction method applied to the loss.
        num_output_dimensions : int, optional
            Number of spatial/output dimensions.
        low_ress_kernel_size : int or None, optional
            Kernel size for downsampling.
        hyperparam : float, optional
            Scaling factor for asymmetric penalty.
        min_threshold : float, optional
            Lower threshold for asymmetric weighting.
        max_threshold : float, optional
            Upper threshold for asymmetric weighting.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If kernel size is not odd.
        NotImplementedError
            If unsupported number of output dimensions is used.

        Notes
        -----
        Expected weight tensor layouts:

        - With channels: ``(C, O1, ..., On)``
        - Without channels: ``(O1, ..., On)``

        where:

        - ``C`` is the channel dimension
        - ``O1 ... On`` are spatial/output dimensions

        If channels are not present, a singleton channel dimension may be temporarily added during low-resolution downsampling.
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
            if not self.low_ress_kernel_size % 2 == 1:
                raise ValueError("choose odd kernel size")

            if not has_channels:
                weights_mask = weights_mask.unsqueeze(0)
                weights = weights.unsqueeze(0)

            if self.num_output_dimensions == 1:
                self.average_pool = F.avg_pool1d

            elif self.num_output_dimensions == 2:
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
        Apply pooling for downsampling.

        Parameters
        ----------
        tensor : torch.Tensor

        Returns
        -------
        torch.Tensor

        Notes
        -----
        Expected input shapes:

        - Static weights/masks:
        ``(C, O1, ..., On)``
        - Batched tensors:
        ``(B, C, O1, ..., On)``
        - Generative tensors:
        ``(Z, B, C, O1, ..., On)``

        The method temporarily inserts a batch dimension for static weight tensors so that PyTorch pooling operators can be applied. The original dimensionality is restored before returning.

        """

        squeeze = False
        if len(tensor.shape) == self.num_output_dimensions + 1:
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
        generative_modeling: bool = False,
        generator: bool = False,
        print_loss=False,
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss.

        Parameters
        ----------
        data : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth values.
        target_mask : torch.Tensor or None, optional
            Mask applied to targets.
        generative_modeling : bool, optional
            Whether inputs contain sample dimension.
        generator : bool, optional
            Whether predictions come from generator samples.
        print_loss : bool, optional
            Whether to print loss.

        Returns
        -------
        torch.Tensor
            Loss value.

        Raises
        ------
        RuntimeError
            If shapes do not match.

        Notes
        -----
        Expected tensor shapes are:

        - Standard mode: ``(B, C, ...)``
        - Generator mode: ``(E, B, C, ...)``
        - Generative modeling mode: ``(Z, B, C, ...)``
        - Generator + generative modeling: ``(E, Z, B, C, ...)``

        where ``E`` is the ensemble/sample dimension, ``Z`` is the latent sample dimension, ``B`` is batch size, and ``C`` is the channel dimension.
        """

        if generator:
            _check_generator_structure(data, target)
            data = data.mean(0)

        if not data.shape == target.shape:
            raise RuntimeError(
                f"for MSE data and target must have the same shape, got {data.shape} vs {target.shape}"
            )

        y = target
        y_hat = data

        if self.low_ress_kernel_size is not None:
            if generative_modeling:
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
        Print MSE loss.

        Parameters
        ----------
        loss : torch.Tensor

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
        Apply weighting, masking, and reduction.

        Parameters
        ----------
        loss : torch.Tensor
        mask : torch.Tensor or None

        Returns
        -------
        torch.Tensor

        Notes
        -----
        Expected loss shape before reduction:

        ``(B, C, O1, ..., On)``

        or, for generative modeling:

        ``(Z, B, C, O1, ..., On)``

        Weights are broadcast across all leading dimensions and applied
        along the channel and spatial dimensions.
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
            ).mean()

        else:
            raise NotImplementedError(
                "Other reduction methods than 'sum' and 'mean' are not defined."
            )

        return loss


@Losspipeline.register("crps")
class WeightedCRPS(lossABC):
    """
    Weighted continuous ranked probability score (CRPS).

    Designed for probabilistic predictions with multiple samples.

    Parameters
    ----------
    weights : xr.DataArray
        Spatial/feature weights applied to the loss.
    reduction : {"mean", "sum"}, optional
        Reduction method applied to the loss.
    num_output_dimensions : int, optional
        Number of output dimensions.
    low_ress_kernel_size : int or None, optional
        Kernel size for low-resolution smoothing (if used).
    """

    def __init__(
        self,
        weights: xr.DataArray,
        reduction: Reduction = "mean",
        num_output_dimensions: int = 2,
        low_ress_kernel_size: int = None,
        **kwargs,
    ):
        """
        Initialize weighted CRPS loss.

        Parameters
        ----------
        weights : xr.DataArray
            Spatial or variable weights.
        reduction : {"mean", "sum"}, optional
            Reduction method applied to the loss.
        num_output_dimensions : int, optional
            Number of spatial/output dimensions.
        low_ress_kernel_size : int or None, optional
            Kernel size for downsampling.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If kernel size is not odd.
        NotImplementedError
            If unsupported number of output dimensions is used.

        Notes
        -----
        Both `weights` and the internally constructed `weights_mask` are expected to follow the shape `(C, ...)`, where `C` is the channel dimension.
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
            if not low_ress_kernel_size % 2 == 1:
                raise ValueError("choose odd kernel size")

            if not has_channels:
                weights_mask = weights_mask.unsqueeze(0)
                weights = weights.unsqueeze(0)

            if self.num_output_dimensions == 1:
                self.average_pool = F.avg_pool1d

            elif self.num_output_dimensions == 2:
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
        Downsample tensor using average pooling.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to be downsampled.

        Returns
        -------
        torch.Tensor
            Downsampled tensor with the same dimensionality as input.
        """

        squeeze = False
        if len(tensor.shape) == self.num_output_dimensions + 1:
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
        generative_modeling: bool = False,
        generator: bool = True,
        print_loss=False,
    ) -> torch.Tensor:
        """
        Compute CRPS loss.

        Parameters
        ----------
        data : torch.Tensor
            Ensemble predictions (samples).
        target : torch.Tensor
            Ground truth.
        target_mask : torch.Tensor or None, optional
        generative_modeling : bool, optional
        generator : bool, optional
        print_loss : bool, optional

        Returns
        -------
        torch.Tensor

        Raises
        ------
        RuntimeError
            If generator flag is False.

        Notes
        -----
        CRPS requires an ensemble of predictions. The first dimension of ``data`` is interpreted as the ensemble/sample dimension.

        Expected tensor layouts:

        Standard ensemble prediction:
        - ``data``: ``(E, B, C, O1, ..., On)``
        - ``target``: ``(B, C, O1, ..., On)``

        Generative modeling:
        - ``data``: ``(E, Z, B, C, O1, ..., On)``
        - ``target``: ``(Z, B, C, O1, ..., On)``

        Where:
        - ``E`` = ensemble size
        - ``Z`` = latent realization dimension
        - ``B`` = batch dimension
        - ``C`` = channel dimension
        - ``O1...On`` = spatial/output dimensions
        """

        if not generator:
            raise RuntimeError(
                "generator cannot be False as a step_argument in Losspipeline.forward()"
            )
        _check_generator_structure(data, target)

        if generative_modeling:
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
        Print CRPS loss.

        Parameters
        ----------
        loss : torch.Tensor

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
        Apply weighting, masking, and reduction.

        Parameters
        ----------
        loss : torch.Tensor
        mask : torch.Tensor or None

        Returns
        -------
        torch.Tensor
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
            ).mean()

        else:
            raise NotImplementedError(
                "Other reduction methods than 'sum' and 'mean' are not defined."
            )

        return loss


@Losspipeline.register("frobenius_norm")
class Frobenius_norm(lossABC):
    """
    Frobenius norm loss between covariance matrices.

    Parameters
    ----------
    weights : xr.DataArray
        Spatial/feature weights applied to the loss.
    reduction : {"mean", "sum"}, optional
        Reduction method applied to the loss.
    num_output_dimensions : int, optional
        Number of output dimensions.
    covariance_dim : {"spatial", "channel"}, optional
        Dimension along which covariance is computed.
    """

    def __init__(
        self,
        weights: xr.DataArray,
        reduction: Reduction = "mean",
        num_output_dimensions: int = 2,
        covariance_dim: CovarianceDim = "spatial",
    ):
        """
        Initialize Frobenius norm loss size    Initialize Frobenius norm loss.
        based on spatial dimensions.

        Parameters
        ----------
        weights : xr.DataArray
            Reference weights used to infer output dimensionality.
        reduction : {"mean", "sum"}, optional
            Reduction method applied to the loss.
        num_output_dimensions : int, optional
            Number of spatial/output dimensions.
        covariance_dim : {"spatial", "channel"}, optional
            Dimension along which covariance is computed.

        Returns
        -------
        None
        """

        super().__init__()

        self.covariance_dim = covariance_dim
        self.num_output_dimensions = num_output_dimensions
        self.reduction = reduction
        self.output_size = np.prod(weights.shape[-self.num_output_dimensions :])

    def forward(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        generative_modeling: bool = False,
        generator: bool = False,
        print_loss=False,
    ) -> torch.Tensor:
        """
        Compute Frobenius norm of covariance difference.

        Parameters
        ----------
        data : torch.Tensor
        target : torch.Tensor
        generative_modeling : bool, optional
        generator : bool, optional
        print_loss : bool, optional

        Returns
        -------
        torch.Tensor

        Notes
        -----
        Expected tensor layouts:

        Standard mode:
        - ``data``: ``(B, C, O1, ..., On)``
        - ``target``: ``(B, C, O1, ..., On)``

        Generative modeling:
        - ``data``: ``(Z, B, C, O1, ..., On)``
        - ``target``: ``(Z, B, C, O1, ..., On)``

        Processing steps:

        1. Spatial dimensions ``O1...On`` are flattened into a single feature dimension.
        2. If generative modeling is enabled, the latent dimension ``Z`` is merged with the batch dimension.
        3. Covariance matrices are computed either:
        - across spatial features (``covariance_dim="spatial"``)
        - across channels (``covariance_dim="channel"``)
        4. The Frobenius norm of the covariance difference is returned.
        """

        if generator:
            _check_generator_structure(data, target)
            data = data.mean(0)

        assert data.shape == target.shape

        y = torch.flatten(target, start_dim=-self.num_output_dimensions, end_dim=-1)
        y_hat = torch.flatten(data, start_dim=-self.num_output_dimensions, end_dim=-1)

        if generative_modeling:
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
        Print Frobenius norm loss.

        Parameters
        ----------
        loss : torch.Tensor

        Returns
        -------
        None
        """

        print(f"FLN : {loss.item():.5f}")

    def _aggregate(self, loss, output_size=None):
        """
        Apply normalization and reduction.

        Parameters
        ----------
        loss : torch.Tensor
        output_size : int or None, optional

        Returns
        -------
        torch.Tensor
        """

        if output_size is None:
            output_size = self.output_size

        if self.reduction.lower() == "mean":
            loss = loss / output_size

            loss = loss.mean()

        elif self.reduction.lower() == "sum":
            loss = loss.sum()

        return loss


def _check_generator_structure(data: torch.Tensor, target: torch.Tensor):
    """
    Validate generator output structure.

    Parameters
    ----------
    data : torch.Tensor
        Generated samples.
    target : torch.Tensor
        Target tensor.

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        If structure is inconsistent.

    Notes
    -----
    Generator outputs must contain exactly one additional leading
    ensemble/sample dimension compared to the target.

    Valid examples:

    - target:
    ``(B, C, O1, O2)``
    - data:
    ``(E, B, C, O1, O2)``

    Generative example:

    - target:
    ``(Z, B, C, O1, O2)``
    - data:
    ``(E, Z, B, C, O1, O2)``

    Invalid examples:

    - data:
    ``(B, E, C, O1, O2)``
    - data:
    ``(E, C, O1, O2)``

    The ensemble dimension must always be the left-most dimension.
    """

    bool = data.shape[1:] == (1,) * (data.dim() - target.dim() - 1) + target.shape
    if bool:
        return True
    else:
        raise ValueError(
            "Expected data to have one extra sample dim on left. "
            "Actual shapes: {} versus {}".format(data.shape, target.shape)
        )

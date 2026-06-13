import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from typing import Literal

from cccma_ppp.loss.loss import Losspipeline
from cccma_ppp.loss.loss_abc import lossABC, Reduction


CovarianceDim = Literal["spatial", "channel"]


@Losspipeline.register("mse")
class WeightedMSE(lossABC):
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
                weights_mask = weights_mask.unsqueeze(0) # C x ...
                weights = weights.unsqueeze(0)  # C x ...

            if self.num_output_dimensions == 1:
                self.average_pool = F.avg_pool1d

            elif self.num_output_dimensions == 2:
                self.average_pool = F.avg_pool2d

            else:
                raise NotImplementedError(
                    "not designed for dataset with number of output spatiotemporal dimensions > 2 (except channels)"
                )

            weights_mask = self._downsample(weights_mask)
            weights_mask = (weights_mask == 1).float() ## just keep grid weights where all high res grids are available
            weights = self._downsample(weights)

            if not has_channels:
                weights_mask = weights_mask.squeeze(0)

        weights = weights * weights_mask

        self.register_buffer("weights", weights)

    def _downsample(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Downsample tenspr using the low-resolution kernel.


        For masks with 1 valid and 0 invalid, the output is a fractional-validity mask between 0 and 1.
        """
        squeeze = False
        if len(tensor.shape) == self.num_output_dimensions + 1:  #check for static masks (space + channel)
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
        data: torch.Tensor,    ## (E x) (Z x ) B x C x ....
        target: torch.Tensor,  ## (Z x ) B x C x ....
        target_mask: torch.Tensor | None = None,
        generative_modeling: bool = False,
        generator: bool = False,
        print_loss=False,
    ) -> torch.Tensor:

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
        if self.low_ress_kernel_size is not None:
            print(f"MSE_lowress : {loss.item():.5f}")
        else:
            print(f"MSE : {loss.item():.5f}")

    def _aggregate(self, loss, mask):
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
    def __init__(
        self,
        weights: xr.DataArray,
        reduction: Reduction ="mean",
        num_output_dimensions: int = 2,
        low_ress_kernel_size: int = None,
        **kwargs,
    ):

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
                weights_mask = weights_mask.unsqueeze(0) # C x ...
                weights = weights.unsqueeze(0) # C x ...

            if self.num_output_dimensions == 1:
                self.average_pool = F.avg_pool1d

            elif self.num_output_dimensions == 2:
                self.average_pool = F.avg_pool2d

            else:
                raise NotImplementedError(
                    "not designed for dataset with number of output spatiotemporal dimensions > 2 (except channels)"
                )

            weights_mask = self._downsample(weights_mask)
            weights_mask = (weights_mask == 1).float() ## just keep grid weights where all high res grids are available
            weights = self._downsample(weights)

            if not has_channels:
                weights_mask = weights_mask.squeeze(0)
                weights = weights.squeeze(0)

        weights = weights * weights_mask

        self.register_buffer("weights", weights)

    def _downsample(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Downsample tenspr using the low-resolution kernel.


        For masks with 1 valid and 0 invalid, the output is a fractional-validity mask between 0 and 1.
        """
        squeeze = False
        if len(tensor.shape) == self.num_output_dimensions + 1:  #check for static masks (space + channel)
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
        data: torch.Tensor,   ## E x (Z x ) B x C x ....  for crps an ensemble of outputs are required
        target: torch.Tensor,   ##  (Z x ) B x C x ....
        target_mask: torch.Tensor | None = None,
        generative_modeling: bool = False,
        generator: bool = True,
        print_loss=False,
    ) -> torch.Tensor:

        if not generator:
            raise RuntimeError('generator cannot be False as a step_argument in Losspipeline.forward()')
        _check_generator_structure(data, target)

        if generative_modeling:  ##generative_modeling models generate a latent ensemble  (Z x batch x channels x ...)
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
            y_hat = y_hat.reshape(E * B, *y_hat.shape[2:])   ##reshape output ensemble to calcualte the average pool first and then reshape again.
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
        if self.low_ress_kernel_size is not None:
            print(f"CRPS_lowress : {loss.item():.5f}")
        else:
            print(f"CRPS : {loss.item():.5f}")

    def _aggregate(self, loss, mask):

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
    def __init__(
        self,
        weights: xr.DataArray,
        reduction: Reduction ="mean",
        num_output_dimensions: int = 2,
        covariance_dim: CovarianceDim = "spatial",
    ):

        super().__init__()

        self.covariance_dim = covariance_dim
        self.num_output_dimensions = num_output_dimensions
        self.reduction = reduction
        self.output_size = np.prod(weights.shape[-self.num_output_dimensions :])

    def forward(
        self,
        data: torch.Tensor,  ## data shape is : in (Ens x) batch x channel x O_1, ... x O_n with n being the num_output_dimensions
        target: torch.Tensor,
        generative_modeling: bool = False,
        generator: bool = False,
        print_loss=False,
    ) -> torch.Tensor:

        if generator:
            _check_generator_structure(data, target)
            data = data.mean(0)

        assert data.shape == target.shape

        y = torch.flatten(target, start_dim=-self.num_output_dimensions, end_dim=-1)
        y_hat = torch.flatten(data, start_dim=-self.num_output_dimensions, end_dim=-1)

        if generative_modeling: ##generative_modeling models generate an ensemble of outputs (enx x batch x channels x ...)
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
        print(f"FLN : {loss.item():.5f}")

    def _aggregate(self, loss, output_size=None):
        if output_size is None:
            output_size = self.output_size

        if self.reduction.lower() == "mean":
            loss = loss / output_size

            loss = loss.mean()

        elif self.reduction.lower() == "sum":
            loss = loss.sum()

        return loss


def _check_generator_structure(data: torch.Tensor, target: torch.Tensor):
    bool = data.shape[1:] == (1,) * (data.dim() - target.dim() - 1) + target.shape
    if bool:
        return True
    else:
        raise ValueError(
            "Expected data to have one extra sample dim on left. "
            "Actual shapes: {} versus {}".format(data.shape, target.shape)
        )

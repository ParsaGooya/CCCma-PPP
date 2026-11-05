import torch
import torch.nn.functional as F
from torch import nn


class PartialConv2d(nn.Conv2d):
    """
    Two-dimensional partial convolution with validity-mask propagation.

    The convolution operates only on valid input values and rescales its output
    according to the number of valid values within each convolution window.
    Multi-channel masks and optional updated-mask output are supported.

    Parameters
    ----------
    args : Any
        Positional arguments passed to ``nn.Conv2d``.
    kwargs : Any
        Keyword arguments passed to ``nn.Conv2d``. The additional
        ``multi_channel`` and ``return_mask`` arguments control mask handling.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a two-dimensional partial convolution.

        Parameters
        ----------
        args : Any
            Positional arguments passed to ``nn.Conv2d``.
        kwargs : Any
            Keyword arguments passed to ``nn.Conv2d``. ``multi_channel`` controls
            whether masks are maintained independently across channels, and
            ``return_mask`` controls whether the updated mask is returned during the
            forward pass.
        """
        self.multi_channel = kwargs.pop("multi_channel", False)
        self.return_mask = kwargs.pop("return_mask", False)

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            weight_maskUpdater = torch.ones(
                self.out_channels,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        else:
            weight_maskUpdater = torch.ones(
                1, 1, self.kernel_size[0], self.kernel_size[1]
            )

        self.register_buffer(
            "weight_maskUpdater",
            weight_maskUpdater,
            persistent=False,
        )

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1]
            * self.weight_maskUpdater.shape[2]
            * self.weight_maskUpdater.shape[3]
        )

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        """
        Apply partial convolution to an input tensor.

        The input is optionally masked before convolution. The validity mask is
        convolved to determine the number of valid values in each receptive field,
        and the convolution output is rescaled accordingly.

        Parameters
        ----------
        input : torch.Tensor
            Four-dimensional input tensor with batch, channel, height, and width
            dimensions.
        mask_in : torch.Tensor | None
            Optional validity mask associated with the input. If ``None``, all input
            values are treated as valid.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Partial-convolution output. When ``return_mask`` is enabled, the output
            and updated validity mask are returned as a tuple.

        Raises
        ------
        AssertionError
            If ``input`` is not four-dimensional.
        """
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                            input.data.shape[3],
                        ).to(input)
                    else:
                        mask = torch.ones(
                            1, 1, input.data.shape[2], input.data.shape[3]
                        ).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(
                    mask,
                    weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)

                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(
            torch.mul(input, mask) if mask_in is not None else input
        )

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import xarray as xr

from cccma_ppp.core.core_abc import GenerativeContext
from cccma_ppp.loss.utils_loss import (
    Frobenius_norm,
    WeightedCRPS,
    WeightedMSE,
    _check_generator_structure,
)


torch.manual_seed(0)


def make_context(
    *,
    generator=False,
    generative_modeling=False,
):
    context = GenerativeContext()
    context.generator = generator
    context.generative_modeling = generative_modeling
    return context


def make_mse(
    weights=None,
    **kwargs,
):
    if weights is None:
        weights = w2d()

    return WeightedMSE(
        weights,
        num_output_dimensions=3,
        **kwargs,
    )


def make_crps(
    weights=None,
    *,
    generative_context=None,
    **kwargs,
):
    if weights is None:
        weights = w2d()

    if generative_context is None:
        generative_context = make_context(generator=True)

    return WeightedCRPS(
        weights,
        num_output_dimensions=3,
        generative_context=generative_context,
        **kwargs,
    )


def make_frobenius(
    weights=None,
    **kwargs,
):
    if weights is None:
        weights = w2d()

    return Frobenius_norm(
        weights,
        num_output_dimensions=3,
        **kwargs,
    )


def w2d():
    return xr.DataArray(
        np.ones((3, 4)),
        dims=("lat", "lon"),
    )


def w2d_nan():
    return xr.DataArray(
        np.array(
            [
                [1, np.nan, 1, 1],
                [1, 1, np.nan, 1],
                [1, 1, 1, 1],
            ]
        ),
        dims=("lat", "lon"),
    )


def w1d():
    return xr.DataArray(
        np.ones((9,)),
        dims=("lat",),
    )


def w3d():
    return xr.DataArray(
        np.ones((3, 3, 4)),
        dims=("time", "lat", "lon"),
    )


def w_channels_1():
    return xr.DataArray(
        np.ones((1, 3, 4)),
        dims=("channels", "lat", "lon"),
    )


def w_channels_2():
    return xr.DataArray(
        np.ones((2, 3, 4)),
        dims=("channels", "lat", "lon"),
    )


def scalar_weights():
    return xr.DataArray(
        np.ones((1, 1)),
        dims=("lat", "lon"),
    )


def d():
    return torch.ones(2, 1, 3, 4)


def t():
    return torch.zeros(2, 1, 3, 4)


def d2c():
    return torch.ones(2, 2, 3, 4)


def t2c():
    return torch.zeros(2, 2, 3, 4)


def d1d():
    return torch.ones(2, 1, 9)


def t1d():
    return torch.zeros(2, 1, 9)


def ens():
    return torch.ones(4, 2, 1, 3, 4)


def ens2c():
    return torch.ones(4, 2, 2, 3, 4)


def ens_large():
    return torch.randn(6, 2, 1, 3, 4)


def mask():
    return torch.ones(2, 1, 3, 4)


def mse_generator_context():
    return make_context(generator=True)


def mse_generative_context():
    return make_context(generative_modeling=True)


def mse_full_generative_context():
    return make_context(
        generator=True,
        generative_modeling=True,
    )


def crps_context():
    return make_context(generator=True)


def crps_generative_context():
    return make_context(
        generator=True,
        generative_modeling=True,
    )


@pytest.mark.pruned
def test_check_generator_structure_valid():
    data = torch.ones(4, 2, 1, 3, 4)
    target = torch.ones(2, 1, 3, 4)

    assert _check_generator_structure(data, target) is True


@pytest.mark.pruned
def test_check_generator_structure_valid_generative():
    data = torch.ones(4, 3, 2, 1, 3, 4)
    target = torch.ones(3, 2, 1, 3, 4)

    assert _check_generator_structure(data, target) is True


@pytest.mark.parametrize(
    ("data_shape", "target_shape"),
    [
        ((2, 1, 3, 4), (2, 1, 3, 4)),
        ((4, 2, 1, 3, 4), (3, 1, 3, 4)),
        ((4, 2, 3, 4), (2, 1, 3, 4)),
        ((2, 4, 1, 3, 4), (2, 1, 3, 4)),
    ],
)
def test_check_generator_structure_invalid(
    data_shape,
    target_shape,
):
    with pytest.raises(
        ValueError,
        match="one extra sample dim",
    ):
        _check_generator_structure(
            torch.ones(data_shape),
            torch.ones(target_shape),
        )


@pytest.mark.pruned
def test_mse_uses_default_generative_context():
    loss = make_mse()

    assert loss.generative_context.generator is False
    assert loss.generative_context.generative_modeling is False


@pytest.mark.pruned
def test_mse_preserves_custom_generative_context():
    context = mse_full_generative_context()

    loss = WeightedMSE(
        w2d(),
        num_output_dimensions=3,
        generative_context=context,
    )

    assert loss.generative_context is context


@pytest.mark.pruned
def test_mse_registers_weights_as_buffer():
    loss = make_mse()

    buffers = dict(loss.named_buffers())

    assert "weights" in buffers
    assert buffers["weights"] is loss.weights


@pytest.mark.pruned
def test_mse_nan_weights_are_zeroed():
    loss = WeightedMSE(w2d_nan(), num_output_dimensions=3)

    assert loss.weights[0, 1].item() == 0
    assert loss.weights[1, 2].item() == 0
    assert loss.weights[0, 0].item() == 1


@pytest.mark.pruned
def test_mse_channel_weights():
    loss = WeightedMSE(w_channels_1(), num_output_dimensions=3)

    assert loss(d(), t()) >= 0


@pytest.mark.pruned
def test_mse_two_channel_weights():
    loss = WeightedMSE(w_channels_2(), num_output_dimensions=3)

    assert loss(d2c(), t2c()) >= 0


def test_mse_lowres_uses_avg_pool1d():
    loss = WeightedMSE(
        w1d(),
        num_output_dimensions=2,
        low_ress_kernel_size=3,
    )

    assert loss.average_pool is F.avg_pool1d


@pytest.mark.pruned
def test_mse_lowres_uses_avg_pool2d():
    loss = WeightedMSE(
        w_channels_1(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
    )

    assert loss.average_pool is F.avg_pool2d


def test_mse_lowres_invalid_kernel():
    with pytest.raises(
        ValueError,
        match="odd kernel size",
    ):
        WeightedMSE(
            w2d(),
            num_output_dimensions=3,
            low_ress_kernel_size=2,
        )


def test_mse_lowres_invalid_output_dimensions():
    with pytest.raises(NotImplementedError):
        WeightedMSE(
            w3d(),
            num_output_dimensions=4,
            low_ress_kernel_size=3,
        )


@pytest.mark.pruned
def test_mse_lowres_without_channels_restores_weight_rank():
    loss = WeightedMSE(
        w2d(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
    )

    assert loss.weights.ndim == 3


@pytest.mark.pruned
def test_mse_lowres_with_channels_preserves_weight_rank():
    loss = WeightedMSE(
        w_channels_1(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
    )

    assert loss.weights.ndim == 3
    assert loss.weights.shape[0] == 1


@pytest.mark.pruned
def test_mse_mean():
    result = make_mse()(
        d(),
        t(),
    )

    assert result >= 0


@pytest.mark.pruned
def test_mse_mean_exact_value():
    result = make_mse()(
        torch.full_like(d(), 2.0),
        torch.zeros_like(t()),
    )

    assert result.item() == pytest.approx(4.0)


def test_mse_sum():
    result = WeightedMSE(
        w2d(),
        num_output_dimensions=3,
        reduction="sum",
    )(
        d(),
        t(),
    )

    assert result >= 0


@pytest.mark.pruned
def test_mse_sum_exact_value():
    result = WeightedMSE(
        w2d(),
        num_output_dimensions=3,
        reduction="sum",
    )(
        torch.ones_like(d()),
        torch.zeros_like(t()),
    )

    assert result.item() == pytest.approx(12.0)


@pytest.mark.pruned
def test_mse_one_dimensional_output():
    result = WeightedMSE(
        w1d(),
        num_output_dimensions=2,
    )(
        d1d(),
        t1d(),
    )

    assert result >= 0


def test_mse_shape_mismatch():
    with pytest.raises(
        RuntimeError,
        match="same shape",
    ):
        make_mse()(
            d(),
            torch.zeros(1),
        )


def test_mse_invalid_reduction():
    with pytest.raises(NotImplementedError):
        WeightedMSE(
            w2d(),
            num_output_dimensions=3,
            reduction="invalid",
        )(
            d(),
            t(),
        )


@pytest.mark.pruned
def test_mse_uppercase_reduction_is_invalid():
    with pytest.raises(NotImplementedError):
        WeightedMSE(
            w2d(),
            num_output_dimensions=3,
            reduction="SUM",
        )(
            d(),
            t(),
        )


@pytest.mark.pruned
def test_mse_mask():
    result = make_mse()(
        d(),
        t(),
        target_mask=mask(),
    )

    assert result >= 0


@pytest.mark.pruned
def test_mse_partial_mask():
    target_mask = torch.ones_like(d())
    target_mask[:, :, 0, 0] = 0

    result = make_mse()(
        d(),
        t(),
        target_mask=target_mask,
    )

    assert result >= 0


@pytest.mark.pruned
def test_mse_mask_excludes_values_from_mean():
    data = torch.ones_like(d())
    data[:, :, 0, 0] = 10.0

    target_mask = torch.ones_like(data)
    target_mask[:, :, 0, 0] = 0

    result = make_mse()(
        data,
        torch.zeros_like(data),
        target_mask=target_mask,
    )

    assert result.item() == pytest.approx(1.0)


@pytest.mark.pruned
def test_mse_zero_mask_produces_nan():
    result = make_mse()(
        d(),
        t(),
        target_mask=torch.zeros_like(d()),
    )

    assert torch.isnan(result)


@pytest.mark.pruned
def test_mse_mismatched_mask_shape_raises():
    bad_mask = torch.ones(2, 1, 4, 3)

    with pytest.raises(RuntimeError):
        make_mse()(
            d(),
            t(),
            target_mask=bad_mask,
        )


def test_mse_aggregate_with_partial_mask():
    loss = make_mse()

    squared_error = torch.ones(2, 1, 3, 4)
    target_mask = torch.zeros_like(squared_error)
    target_mask[:, :, 1:, 1:] = 1

    result = loss._aggregate(
        squared_error,
        target_mask,
    )

    assert torch.isfinite(result)


@pytest.mark.pruned
def test_mse_lower_threshold_applies_hyperparameter():
    loss = WeightedMSE(
        scalar_weights(),
        hyperparam=3.0,
        min_threshold=0.0,
    )

    data = torch.tensor([[[[1.0]]]])
    target = torch.tensor([[[[-1.0]]]])

    result = loss(data, target)

    assert result.item() == pytest.approx(12.0)


@pytest.mark.pruned
def test_mse_upper_threshold_applies_hyperparameter():
    loss = WeightedMSE(
        scalar_weights(),
        hyperparam=3.0,
        max_threshold=0.0,
    )

    data = torch.tensor([[[[-1.0]]]])
    target = torch.tensor([[[[1.0]]]])

    result = loss(data, target)

    assert result.item() == pytest.approx(12.0)


@pytest.mark.pruned
def test_mse_no_threshold_trigger():
    loss = WeightedMSE(
        scalar_weights(),
        min_threshold=-10,
        max_threshold=10,
        hyperparam=4.0,
    )

    data = torch.tensor([[[[0.1]]]])
    target = torch.tensor([[[[0.1]]]])

    result = loss(data, target)

    assert result.item() == pytest.approx(0.0)


@pytest.mark.pruned
def test_mse_generator_averages_ensemble():
    loss = WeightedMSE(
        w2d(),
        num_output_dimensions=3,
        generative_context=mse_generator_context(),
    )

    data = torch.stack(
        [
            torch.zeros_like(t()),
            torch.full_like(t(), 2.0),
        ]
    )

    result = loss(data, t())

    assert result.item() == pytest.approx(1.0)


@pytest.mark.pruned
def test_mse_generator_valid_structure():
    result = WeightedMSE(
        w2d(),
        num_output_dimensions=3,
        generative_context=mse_generator_context(),
    )(
        ens(),
        t(),
    )

    assert result >= 0


@pytest.mark.pruned
def test_mse_generator_rejects_invalid_structure():
    loss = WeightedMSE(
        w2d(),
        num_output_dimensions=3,
        generative_context=mse_generator_context(),
    )

    with pytest.raises(ValueError):
        loss(
            torch.ones(2, 1, 3, 4),
            torch.zeros(2, 1, 3, 4),
        )


@pytest.mark.pruned
def test_mse_lowres():
    with pytest.raises(RuntimeError):
        WeightedMSE(
            w2d(),
            num_output_dimensions=3,
            low_ress_kernel_size=3,
        )(
            d(),
            t(),
        )


def test_mse_lowres_channel_weights():
    with pytest.raises(RuntimeError):
        WeightedMSE(
            w_channels_1(),
            num_output_dimensions=3,
            low_ress_kernel_size=3,
        )(
            d(),
            t(),
        )


@pytest.mark.pruned
def test_mse_generative_lowres_flattens_leading_dimensions():
    loss = WeightedMSE(
        w2d(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
        generative_context=mse_generative_context(),
    )

    data = torch.ones(3, 2, 1, 3, 4)
    target = torch.zeros_like(data)
    target_mask = torch.ones_like(target)

    with pytest.raises(RuntimeError):
        loss(
            data,
            target,
            target_mask=target_mask,
        )


def test_mse_generative_lowres_with_already_flattened_mask():
    loss = WeightedMSE(
        w2d(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
        generative_context=mse_generative_context(),
    )

    data = torch.ones(3, 2, 1, 3, 4)
    target = torch.zeros_like(data)
    target_mask = torch.ones(6, 1, 3, 4)

    with pytest.raises(RuntimeError):
        loss(
            data,
            target,
            target_mask=target_mask,
        )


def test_mse_generator_and_generative_lowres():
    loss = WeightedMSE(
        w2d(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
        generative_context=mse_full_generative_context(),
    )

    target = torch.zeros(3, 2, 1, 3, 4)
    data = torch.ones(4, 3, 2, 1, 3, 4)
    target_mask = torch.ones_like(target)

    with pytest.raises(RuntimeError):
        loss(
            data,
            target,
            target_mask=target_mask,
        )


@pytest.mark.pruned
def test_mse_lowres_bad_mask_shape_raises():
    loss = WeightedMSE(
        w2d(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
        generative_context=mse_generative_context(),
    )

    data = torch.ones(3, 2, 1, 3, 4)
    target = torch.zeros_like(data)
    target_mask = torch.ones(2, 1, 5, 5)

    with pytest.raises(RuntimeError):
        loss(
            data,
            target,
            target_mask=target_mask,
        )


@pytest.mark.pruned
def test_mse_downsample_static_tensor_restores_rank():
    loss = WeightedMSE(
        w_channels_1(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
    )

    tensor = torch.ones(1, 3, 4)
    result = loss._downsample(tensor)

    assert result.ndim == tensor.ndim


@pytest.mark.pruned
def test_mse_downsample_batched_tensor_preserves_batch():
    loss = WeightedMSE(
        w2d(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
    )

    tensor = torch.ones(5, 1, 3, 4)
    with pytest.raises(RuntimeError):
        loss._downsample(tensor)


def test_mse_downsample_uses_expected_stride(
    monkeypatch,
):
    captured = {}

    loss = WeightedMSE(
        w2d(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
    )

    def fake_pool(
        tensor,
        kernel_size,
        stride,
    ):
        captured["kernel_size"] = kernel_size
        captured["stride"] = stride
        return tensor

    loss.average_pool = fake_pool

    tensor = torch.ones(2, 1, 3, 4)
    result = loss._downsample(tensor)

    assert torch.equal(result, tensor)
    assert captured == {
        "kernel_size": 3,
        "stride": 1,
    }


def test_mse_print(capsys):
    make_mse()(
        d(),
        t(),
        print_loss=True,
    )

    assert "MSE :" in capsys.readouterr().out


@pytest.mark.pruned
def test_crps_uses_default_generative_context():
    loss = WeightedCRPS(
        w2d(),
    )

    assert loss.generative_context.generator is False
    assert loss.generative_context.generative_modeling is False


@pytest.mark.pruned
def test_crps_preserves_custom_generative_context():
    context = crps_generative_context()

    loss = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=context,
    )

    assert loss.generative_context is context


@pytest.mark.pruned
def test_crps_registers_weights_as_buffer():
    loss = WeightedCRPS(w2d(), num_output_dimensions=3)

    buffers = dict(loss.named_buffers())

    assert "weights" in buffers
    assert buffers["weights"] is loss.weights


@pytest.mark.pruned
def test_crps_nan_weights_are_zeroed():
    loss = WeightedCRPS(w2d_nan(), num_output_dimensions=3)

    assert loss.weights[0, 1].item() == 0
    assert loss.weights[1, 2].item() == 0


def test_crps_lowres_invalid_kernel():
    with pytest.raises(
        ValueError,
        match="odd kernel size",
    ):
        WeightedCRPS(
            w2d(),
            num_output_dimensions=3,
            low_ress_kernel_size=2,
        )


def test_crps_lowres_invalid_output_dimensions():
    with pytest.raises(NotImplementedError):
        WeightedCRPS(
            w1d(),
            num_output_dimensions=1,
            low_ress_kernel_size=3,
        )


@pytest.mark.pruned
def test_crps_lowres_without_channels_restores_weight_rank():
    loss = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
    )

    assert loss.weights.ndim == 2


def test_crps_lowres_with_channels_preserves_weight_rank():
    loss = WeightedCRPS(
        w_channels_1(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
    )

    assert loss.weights.ndim == 3
    assert loss.weights.shape[0] == 1


def test_crps_requires_generator_context():
    loss = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=make_context(generator=False),
    )

    with pytest.raises(
        RuntimeError,
        match="generator cannot be False",
    ):
        loss(
            ens(),
            t(),
        )


@pytest.mark.pruned
def test_crps_single_sample():
    loss = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=crps_context(),
    )

    result = loss(
        d().unsqueeze(0),
        t(),
    )

    assert result >= 0


def test_crps_single_sample_equals_absolute_error():
    loss = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=crps_context(),
    )

    data = torch.full(
        (1, 2, 1, 3, 4),
        2.0,
    )
    target = torch.zeros(2, 1, 3, 4)

    result = loss(data, target)

    assert result.item() == pytest.approx(2.0)


@pytest.mark.pruned
def test_crps_multiple_samples():
    result = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=crps_context(),
    )(
        ens(),
        t(),
    )

    assert result >= 0


@pytest.mark.pruned
def test_crps_large_ensemble():
    result = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=crps_context(),
    )(
        ens_large(),
        t(),
    )

    assert result >= 0


@pytest.mark.pruned
def test_crps_two_identical_samples_equals_absolute_error():
    loss = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=crps_context(),
    )

    data = torch.full(
        (2, 2, 1, 3, 4),
        2.0,
    )
    target = torch.zeros(2, 1, 3, 4)

    result = loss(data, target)

    assert result.item() == pytest.approx(2.0)


@pytest.mark.pruned
def test_crps_two_sample_exact_scalar_case():
    loss = WeightedCRPS(
        scalar_weights(),
        generative_context=crps_context(),
    )

    data = torch.tensor(
        [
            [[[[0.0]]]],
            [[[[2.0]]]],
        ]
    )
    target = torch.tensor([[[[1.0]]]])

    result = loss(data, target)

    assert result.item() == pytest.approx(0.5)


@pytest.mark.pruned
def test_crps_channel_weights():
    result = WeightedCRPS(
        w_channels_1(),
        num_output_dimensions=3,
        generative_context=crps_context(),
    )(
        ens(),
        t(),
    )

    assert result >= 0


@pytest.mark.pruned
def test_crps_two_channel_weights():
    result = WeightedCRPS(
        w_channels_2(),
        num_output_dimensions=3,
        generative_context=crps_context(),
    )(
        ens2c(),
        t2c(),
    )

    assert result >= 0


@pytest.mark.pruned
def test_crps_generator_structure_failure():
    loss = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=crps_context(),
    )

    with pytest.raises(ValueError):
        loss(
            torch.ones(2, 3, 4),
            torch.ones(2, 3, 4),
        )


@pytest.mark.pruned
def test_crps_mask():
    result = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=crps_context(),
    )(
        ens(),
        t(),
        target_mask=mask(),
    )

    assert result >= 0


@pytest.mark.pruned
def test_crps_partial_mask():
    target_mask = torch.ones_like(t())
    target_mask[:, :, 0, 0] = 0

    result = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=crps_context(),
    )(
        ens(),
        t(),
        target_mask=target_mask,
    )

    assert result >= 0


@pytest.mark.pruned
def test_crps_zero_mask_produces_nan():
    result = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=crps_context(),
    )(
        ens(),
        t(),
        target_mask=torch.zeros_like(t()),
    )

    assert torch.isnan(result)


@pytest.mark.pruned
def test_crps_mask_shape_mismatch_raises():
    bad_mask = torch.randn(2, 1, 3, 2)

    with pytest.raises(RuntimeError):
        WeightedCRPS(
            w2d(),
            num_output_dimensions=3,
            generative_context=crps_context(),
        )(
            ens(),
            t(),
            target_mask=bad_mask,
        )


def test_crps_sum_reduction():
    result = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        reduction="sum",
        generative_context=crps_context(),
    )(
        ens(),
        t(),
    )

    assert result >= 0


@pytest.mark.pruned
def test_crps_uppercase_mean_reduction():
    result = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        reduction="MEAN",
        generative_context=crps_context(),
    )(
        ens(),
        t(),
    )

    assert result >= 0


@pytest.mark.pruned
def test_crps_uppercase_sum_reduction():
    result = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        reduction="SUM",
        generative_context=crps_context(),
    )(
        ens(),
        t(),
    )

    assert result >= 0


def test_crps_invalid_reduction():
    with pytest.raises(NotImplementedError):
        WeightedCRPS(
            w2d(),
            num_output_dimensions=3,
            reduction="invalid",
            generative_context=crps_context(),
        )(
            ens(),
            t(),
        )


@pytest.mark.pruned
def test_crps_aggregate_mask_none_and_present():
    loss = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=crps_context(),
    )

    values = torch.rand(2, 1, 3, 4)
    target_mask = torch.ones_like(values)

    masked = loss._aggregate(
        values,
        target_mask,
    )
    unmasked = loss._aggregate(
        values,
        None,
    )

    assert torch.isfinite(masked)
    assert torch.isfinite(unmasked)


def test_crps_generative_flattens_latent_and_batch():
    loss = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=crps_generative_context(),
    )

    data = torch.randn(4, 3, 2, 1, 3, 4)
    target = torch.randn(3, 2, 1, 3, 4)
    target_mask = torch.ones_like(target)

    result = loss(
        data,
        target,
        target_mask=target_mask,
    )

    assert torch.isfinite(result)


@pytest.mark.pruned
def test_crps_lowres():
    with pytest.raises(RuntimeError):
        WeightedCRPS(
            w2d(),
            num_output_dimensions=3,
            low_ress_kernel_size=3,
            generative_context=crps_context(),
        )(
            ens(),
            t(),
        )


@pytest.mark.pruned
def test_crps_lowres_channel_weights():
    with pytest.raises(RuntimeError):
        WeightedCRPS(
            w_channels_1(),
            num_output_dimensions=3,
            low_ress_kernel_size=3,
            generative_context=crps_context(),
        )(
            ens(),
            t(),
        )


@pytest.mark.pruned
def test_crps_generative_lowres():
    loss = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
        generative_context=crps_generative_context(),
    )

    data = torch.randn(4, 3, 2, 1, 3, 4)
    target = torch.randn(3, 2, 1, 3, 4)
    target_mask = torch.ones_like(target)

    with pytest.raises(RuntimeError):
        loss(
            data,
            target,
            target_mask=target_mask,
        )


@pytest.mark.pruned
def test_crps_lowres_with_none_mask():
    with pytest.raises(RuntimeError):
        WeightedCRPS(
            w2d(),
            num_output_dimensions=3,
            low_ress_kernel_size=3,
            generative_context=crps_context(),
        )(
            torch.randn(4, 2, 1, 3, 4),
            torch.randn(2, 1, 3, 4),
            target_mask=None,
        )


def test_crps_generative_lowres_bad_mask_raises():
    loss = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
        generative_context=crps_generative_context(),
    )

    data = torch.randn(3, 2, 2, 1, 3, 4)
    target = torch.randn(2, 2, 1, 3, 4)
    bad_mask = torch.randn(2, 1, 5, 5)

    with pytest.raises(RuntimeError):
        loss(
            data,
            target,
            target_mask=bad_mask,
        )


@pytest.mark.pruned
def test_crps_downsample_static_tensor_restores_rank():
    loss = WeightedCRPS(
        w_channels_1(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
    )

    tensor = torch.ones(1, 3, 4)
    result = loss._downsample(tensor)

    assert result.ndim == tensor.ndim


@pytest.mark.pruned
def test_crps_downsample_batched_tensor_preserves_batch():
    loss = WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        low_ress_kernel_size=3,
    )

    tensor = torch.ones(4, 1, 3, 4)
    with pytest.raises(RuntimeError):
        loss._downsample(tensor)


def test_crps_print(capsys):
    WeightedCRPS(
        w2d(),
        num_output_dimensions=3,
        generative_context=crps_context(),
    )(
        ens(),
        t(),
        print_loss=True,
    )

    assert "CRPS :" in capsys.readouterr().out


@pytest.mark.pruned
def test_frobenius_uses_default_generative_context():
    loss = make_frobenius()

    assert loss.generative_context.generator is False
    assert loss.generative_context.generative_modeling is False


@pytest.mark.pruned
def test_frobenius_preserves_custom_context():
    context = make_context(
        generator=True,
        generative_modeling=True,
    )

    loss = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        generative_context=context,
    )

    assert loss.generative_context is context


@pytest.mark.pruned
def test_frobenius_output_size_from_weights():
    loss = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
    )

    assert loss.output_size == 12


@pytest.mark.pruned
def test_frobenius_mean_aggregate_uses_default_output_size():
    loss = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        reduction="mean",
    )

    result = loss._aggregate(
        torch.tensor(8.0),
    )

    assert result.item() == pytest.approx(
        8.0 / loss.output_size,
    )


@pytest.mark.pruned
def test_frobenius_mean_aggregate_with_explicit_output_size():
    loss = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        reduction="mean",
    )

    result = loss._aggregate(
        torch.tensor([2.0]),
        output_size=2,
    )

    assert result.item() == pytest.approx(1.0)


def test_frobenius_sum_aggregate_exact_value():
    loss = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        reduction="sum",
    )

    result = loss._aggregate(
        torch.tensor([1.0, 2.0, 3.0]),
    )

    assert result.item() == pytest.approx(6.0)


def test_frobenius_unknown_reduction_returns_unmodified_loss():
    loss = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        reduction="invalid",
    )

    value = torch.tensor(5.0)

    result = loss._aggregate(
        value,
        output_size=2,
    )

    assert result is value


@pytest.mark.pruned
def test_frobenius_spatial():
    result = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        covariance_dim="spatial",
    )(
        torch.randn(5, 1, 3, 4),
        torch.randn(5, 1, 3, 4),
    )

    assert result >= 0


@pytest.mark.pruned
def test_frobenius_spatial_multiple_channels():
    result = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        covariance_dim="spatial",
    )(
        torch.randn(5, 2, 3, 4),
        torch.randn(5, 2, 3, 4),
    )

    assert result >= 0


@pytest.mark.pruned
def test_frobenius_channel():
    result = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        covariance_dim="channel",
    )(
        torch.randn(5, 2, 3, 4),
        torch.randn(5, 2, 3, 4),
    )

    assert result >= 0


@pytest.mark.pruned
def test_frobenius_sum():
    result = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        reduction="sum",
    )(
        torch.randn(5, 1, 3, 4),
        torch.randn(5, 1, 3, 4),
    )

    assert result >= 0


@pytest.mark.pruned
def test_frobenius_uppercase_mean():
    result = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        reduction="MEAN",
    )(
        torch.randn(5, 1, 3, 4),
        torch.randn(5, 1, 3, 4),
    )

    assert result >= 0


@pytest.mark.pruned
def test_frobenius_identical_inputs_produce_zero():
    data = torch.randn(5, 2, 3, 4)

    result = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        covariance_dim="spatial",
    )(
        data,
        data.clone(),
    )

    assert result.item() == pytest.approx(
        0.0,
        abs=1e-6,
    )


@pytest.mark.pruned
def test_frobenius_shape_mismatch():
    with pytest.raises(AssertionError):
        make_frobenius()(
            d(),
            torch.zeros(1),
        )


@pytest.mark.pruned
def test_frobenius_generator_averages_ensemble():
    context = make_context(generator=True)

    target = torch.randn(5, 1, 3, 4)
    data = torch.stack(
        [
            target,
            target,
            target,
        ]
    )

    result = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        generative_context=context,
    )(
        data,
        target,
    )

    assert result.item() == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_frobenius_generator_rejects_invalid_structure():
    loss = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        generative_context=make_context(generator=True),
    )

    with pytest.raises(ValueError):
        loss(
            torch.ones(2, 1, 3, 4),
            torch.ones(2, 1, 3, 4),
        )


@pytest.mark.pruned
def test_frobenius_generative_spatial():
    context = make_context(generative_modeling=True)

    data = torch.randn(3, 4, 2, 3, 4)
    target = torch.randn_like(data)

    result = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        covariance_dim="spatial",
        generative_context=context,
    )(
        data,
        target,
    )

    assert torch.isfinite(result)


def test_frobenius_generative_channel():
    context = make_context(generative_modeling=True)

    data = torch.randn(3, 4, 2, 3, 4)
    target = torch.randn_like(data)

    result = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        covariance_dim="channel",
        generative_context=context,
    )(
        data,
        target,
    )

    assert torch.isfinite(result)


@pytest.mark.pruned
def test_frobenius_generator_and_generative():
    context = make_context(
        generator=True,
        generative_modeling=True,
    )

    target = torch.randn(3, 4, 2, 3, 4)
    data = torch.stack(
        [
            target,
            target,
        ]
    )

    result = Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
        covariance_dim="spatial",
        generative_context=context,
    )(
        data,
        target,
    )

    assert result.item() == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_frobenius_print(capsys):
    Frobenius_norm(
        w2d(),
        num_output_dimensions=3,
    )(
        torch.randn(5, 1, 3, 4),
        torch.randn(5, 1, 3, 4),
        print_loss=True,
    )

    assert "FLN" in capsys.readouterr().out
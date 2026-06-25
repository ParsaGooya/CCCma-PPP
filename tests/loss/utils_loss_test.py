import pytest
import torch
import numpy as np
import xarray as xr

from cccma_ppp.loss.utils_loss import WeightedMSE, WeightedCRPS, Frobenius_norm


torch.manual_seed(0)


def w2d():
    return xr.DataArray(np.ones((3, 4)), dims=("lat", "lon"))


def w2d_nan():
    return xr.DataArray(
        np.array([[1, np.nan, 1, 1], [1, 1, np.nan, 1], [1, 1, 1, 1]]),
        dims=("lat", "lon"),
    )


def w1d():
    return xr.DataArray(np.ones((9,)), dims=("lat",))


def w_channels_1():
    return xr.DataArray(np.ones((1, 3, 4)), dims=("channels", "lat", "lon"))


def w_channels_2():
    return xr.DataArray(np.ones((2, 3, 4)), dims=("channels", "lat", "lon"))


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


@pytest.mark.pruned
def test_mse_mean():
    assert WeightedMSE(w2d())(d(), t()) >= 0


def test_mse_sum():
    assert WeightedMSE(w2d(), reduction="sum")(d(), t()) >= 0


@pytest.mark.pruned
def test_mse_nan_weights():
    assert WeightedMSE(w2d_nan())(d(), t()) >= 0


@pytest.mark.pruned
def test_mse_channel_weights():
    assert WeightedMSE(w_channels_1())(d(), t()) >= 0


@pytest.mark.pruned
def test_mse_channel_weights_two_channels():
    assert WeightedMSE(w_channels_2())(d2c(), t2c()) >= 0


@pytest.mark.pruned
def test_mse_mask():
    assert WeightedMSE(w2d())(d(), t(), target_mask=mask()) >= 0


@pytest.mark.pruned
def test_mse_partial_mask():
    m = torch.ones_like(d())
    m[:, :, 0, 0] = 0
    assert WeightedMSE(w2d())(d(), t(), target_mask=m) >= 0


@pytest.mark.pruned
def test_mse_thresholds():
    assert WeightedMSE(w2d(), min_threshold=0.5, hyperparam=2.0)(d(), t()) >= 0
    assert WeightedMSE(w2d(), max_threshold=-0.5, hyperparam=2.0)(-d(), t()) >= 0


@pytest.mark.pruned
def test_mse_both_thresholds_active():
    data = torch.tensor([[[[1.0]]]])
    target = torch.tensor([[[[-1.0]]]])

    loss = WeightedMSE(
        xr.DataArray(np.ones((1, 1)), dims=("lat", "lon")),
        min_threshold=0,
        max_threshold=0,
        hyperparam=2.0,
    )

    assert loss(data, target) >= 0


@pytest.mark.pruned
def test_mse_no_threshold_trigger():
    data = torch.full((1, 1, 1, 1), 0.1)
    target = torch.full((1, 1, 1, 1), 0.1)

    loss = WeightedMSE(
        xr.DataArray(np.ones((1, 1)), dims=("lat", "lon")),
        min_threshold=10,
        max_threshold=-10,
    )

    assert loss(data, target) >= 0


@pytest.mark.pruned
def test_mse_generator():
    assert WeightedMSE(w2d())(ens(), t(), generator=True) >= 0


@pytest.mark.pruned
def test_mse_generator_non_generative():
    assert (
        WeightedMSE(w2d())(ens(), t(), generator=True, generative_modeling=False) >= 0
    )


def test_mse_shape_fail():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        WeightedMSE(w2d())(d(), torch.zeros(1))


def test_mse_lowres_invalid_kernel():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        WeightedMSE(w2d(), low_ress_kernel_size=2)


def test_mse_invalid_output_dimensions():
    w = xr.DataArray(np.ones((2, 2, 2)), dims=("a", "b", "c"))
    with pytest.raises(NotImplementedError):
        WeightedMSE(w, num_output_dimensions=3, low_ress_kernel_size=3)


@pytest.mark.pruned
def test_mse_lowres():
    assert WeightedMSE(w2d(), low_ress_kernel_size=3)(d(), t()) >= 0


def test_mse_lowres_channel_weights():
    assert WeightedMSE(w_channels_1(), low_ress_kernel_size=3)(d(), t()) >= 0


@pytest.mark.pruned
def test_mse_lowres_generator():
    assert WeightedMSE(w2d(), low_ress_kernel_size=3)(ens(), t(), generator=True) >= 0


@pytest.mark.pruned
def test_mse_lowres_generative_no_mask():
    assert (
        WeightedMSE(w2d(), low_ress_kernel_size=3)(
            ens(), t(), generative_modeling=True, generator=True
        )
        >= 0
    )


@pytest.mark.pruned
def test_mse_full_branch_combo():
    m = torch.ones_like(t())
    assert (
        WeightedMSE(w2d(), low_ress_kernel_size=3)(
            ens(), t(), target_mask=m, generator=True
        )
        >= 0
    )


@pytest.mark.pruned
def test_mse_double_flatten_path():
    data = torch.randn(2, 2, 1, 3, 4)
    target = torch.randn(2, 1, 3, 4)

    assert (
        WeightedMSE(w2d(), low_ress_kernel_size=3)(
            data, target, generative_modeling=True, generator=True
        )
        >= 0
    )


@pytest.mark.pruned
def test_mse_1d():
    assert WeightedMSE(w1d(), num_output_dimensions=1)(d1d(), t1d()) >= 0


@pytest.mark.pruned
def test_mse_1d_lowres():
    assert (
        WeightedMSE(w1d(), num_output_dimensions=1, low_ress_kernel_size=3)(
            d1d(), t1d()
        )
        >= 0
    )


@pytest.mark.pruned
def test_mse_invalid_reduction():
    with pytest.raises(NotImplementedError):
        WeightedMSE(w2d(), reduction="invalid")(d(), t())


def test_mse_uppercase_reduction_invalid():
    with pytest.raises(NotImplementedError):
        WeightedMSE(w2d(), reduction="SUM")(d(), t())


@pytest.mark.pruned
def test_mse_mask_zeroing_nan():
    m = torch.zeros_like(d())
    out = WeightedMSE(w2d())(d(), t(), target_mask=m)
    assert torch.isnan(out)


def test_mse_mask_shape_mismatch_branch():
    bad_mask = torch.randn(2, 1, 5, 5)

    with pytest.raises(RuntimeError):
        WeightedMSE(w2d(), low_ress_kernel_size=3)(
            ens(), t(), target_mask=bad_mask, generative_modeling=True, generator=True
        )


@pytest.mark.pruned
def test_mse_mask_exists_but_not_equal_branch():
    bad_mask = torch.ones(2, 1, 12).reshape(2, 1, 3, 4).transpose(-1, -2)

    with pytest.raises(RuntimeError):
        WeightedMSE(w2d())(d(), t(), target_mask=bad_mask)


@pytest.mark.pruned
def test_mse_aggregate_with_mask_scaling():
    loss = WeightedMSE(w2d())
    x = torch.ones(2, 1, 3, 4)
    m = torch.zeros_like(x)
    m[:, :, 1:, 1:] = 1

    out = loss._aggregate(x, m)
    assert torch.isnan(out) or out >= 0


def test_mse_print(capsys):
    WeightedMSE(w2d())(d(), t(), print_loss=True)
    assert "MSE" in capsys.readouterr().out


@pytest.mark.pruned
def test_mse_lowres_print(capsys):
    WeightedMSE(w2d(), low_ress_kernel_size=3)(d(), t(), print_loss=True)
    assert "MSE_lowress" in capsys.readouterr().out


@pytest.mark.pruned
def test_downsample_squeeze_branch():
    loss = WeightedMSE(w2d(), low_ress_kernel_size=3)
    out = loss._downsample(torch.ones(1, 3, 4))
    assert out is not None


@pytest.mark.pruned
def test_downsample_no_squeeze_branch():
    loss = WeightedMSE(w2d(), low_ress_kernel_size=3)
    out = loss._downsample(torch.ones(2, 1, 3, 4))
    assert out is not None


@pytest.mark.pruned
def test_downsample_edge_condition_flip():
    loss = WeightedMSE(w2d(), low_ress_kernel_size=3)
    out = loss._downsample(torch.ones(1, 1, 3, 4))
    assert out is not None


def test_downsample_1d_branch():
    loss = WeightedMSE(w1d(), num_output_dimensions=1, low_ress_kernel_size=3)
    out = loss._downsample(torch.ones(1, 9))
    assert out is not None


@pytest.mark.pruned
def test_generator_structure_failure_mse():
    with pytest.raises(ValueError):
        WeightedMSE(w2d())(torch.ones(2, 3, 4), torch.ones(2, 3, 4), generator=True)


@pytest.mark.pruned
def test_generator_structure_success_mse():
    assert WeightedMSE(w2d())(ens(), t(), generator=True) >= 0


def test_crps_single():
    assert WeightedCRPS(w2d())(d().unsqueeze(0), t()) >= 0


@pytest.mark.pruned
def test_crps_single_exact():
    assert WeightedCRPS(w2d())(torch.randn(1, 2, 1, 3, 4), t()) >= 0


@pytest.mark.pruned
def test_crps_multi():
    assert WeightedCRPS(w2d())(ens(), t()) >= 0


@pytest.mark.pruned
def test_crps_two_samples_edge():
    assert WeightedCRPS(w2d())(torch.randn(2, 2, 1, 3, 4), t()) >= 0


@pytest.mark.pruned
def test_crps_large_ensemble():
    assert WeightedCRPS(w2d())(ens_large(), t()) >= 0


@pytest.mark.pruned
def test_crps_channel_weights():
    assert WeightedCRPS(w_channels_1())(ens(), t()) >= 0


@pytest.mark.pruned
def test_crps_channel_weights_two_channels():
    assert WeightedCRPS(w_channels_2())(ens2c(), t2c()) >= 0


def test_crps_requires_generator():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        WeightedCRPS(w2d())(ens(), t(), generator=False)


def test_crps_generator_structure_failure():
    with pytest.raises(ValueError):
        WeightedCRPS(w2d())(torch.ones(2, 3, 4), torch.ones(2, 3, 4))


@pytest.mark.pruned
def test_crps_mask():
    assert WeightedCRPS(w2d())(ens(), t(), target_mask=mask()) >= 0


@pytest.mark.pruned
def test_crps_partial_mask():
    m = torch.ones_like(t())
    m[:, :, 0, 0] = 0
    assert WeightedCRPS(w2d())(ens(), t(), target_mask=m) >= 0


@pytest.mark.pruned
def test_crps_lowres():
    assert WeightedCRPS(w2d(), low_ress_kernel_size=3)(ens(), t()) >= 0


def test_crps_lowres_channel_weights():
    assert WeightedCRPS(w_channels_1(), low_ress_kernel_size=3)(ens(), t()) >= 0


def test_crps_lowres_invalid_kernel():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        WeightedCRPS(w2d(), low_ress_kernel_size=2)


def test_crps_invalid_output_dimensions():
    w = xr.DataArray(np.ones((2, 2, 2)), dims=("a", "b", "c"))
    with pytest.raises(NotImplementedError):
        WeightedCRPS(w, num_output_dimensions=3, low_ress_kernel_size=3)


@pytest.mark.pruned
def test_crps_lowres_generative():
    assert (
        WeightedCRPS(w2d(), low_ress_kernel_size=3)(
            torch.randn(4, 2, 1, 3, 4),
            torch.randn(2, 1, 3, 4),
            generative_modeling=True,
        )
        >= 0
    )


@pytest.mark.pruned
def test_crps_lowres_non_generative():
    assert (
        WeightedCRPS(w2d(), low_ress_kernel_size=3)(
            torch.randn(4, 2, 1, 3, 4),
            torch.randn(2, 1, 3, 4),
            generative_modeling=False,
        )
        >= 0
    )


@pytest.mark.pruned
def test_crps_lowres_mask_none_branch():
    assert (
        WeightedCRPS(w2d(), low_ress_kernel_size=3)(
            torch.randn(4, 2, 1, 3, 4),
            torch.randn(2, 1, 3, 4),
            target_mask=None,
        )
        >= 0
    )


@pytest.mark.pruned
def test_crps_mask_exact_shape_generative_lowres():
    target = torch.randn(2, 1, 3, 4)
    m = torch.ones_like(target)

    assert (
        WeightedCRPS(w2d(), low_ress_kernel_size=3)(
            torch.randn(3, 2, 1, 3, 4),
            target,
            target_mask=m,
            generative_modeling=True,
        )
        >= 0
    )


def test_crps_mask_mismatch_lowres_generative():
    with pytest.raises(RuntimeError):
        WeightedCRPS(w2d(), low_ress_kernel_size=3)(
            torch.randn(3, 2, 1, 3, 4),
            torch.randn(2, 1, 3, 4),
            target_mask=torch.randn(2, 1, 5, 5),
            generative_modeling=True,
        )


@pytest.mark.pruned
def test_crps_mask_mismatch_non_crash_name_but_expected_error():
    bad_mask = torch.randn(2, 1, 3, 2)

    with pytest.raises(RuntimeError):
        WeightedCRPS(w2d())(ens(), t(), target_mask=bad_mask)


def test_crps_sum_reduce():
    assert WeightedCRPS(w2d(), reduction="sum")(ens(), t()) >= 0


def test_crps_invalid_reduction():
    with pytest.raises(NotImplementedError):
        WeightedCRPS(w2d(), reduction="invalid")(ens(), t())


@pytest.mark.pruned
def test_crps_reduction_uppercase():
    assert WeightedCRPS(w2d(), reduction="MEAN")(ens(), t()) >= 0


@pytest.mark.pruned
def test_crps_aggregate_mask_none_vs_present():
    loss = WeightedCRPS(w2d())
    x = torch.rand(2, 1, 3, 4)
    m = torch.ones_like(x)

    out1 = loss._aggregate(x, m)
    out2 = loss._aggregate(x, None)

    assert out1 >= 0 or torch.isnan(out1)
    assert out2 >= 0


@pytest.mark.pruned
def test_crps_print(capsys):
    WeightedCRPS(w2d())(ens(), t(), print_loss=True)
    assert "CRPS" in capsys.readouterr().out


def test_crps_lowres_print(capsys):
    WeightedCRPS(w2d(), low_ress_kernel_size=3)(ens(), t(), print_loss=True)
    assert "CRPS_lowress" in capsys.readouterr().out


@pytest.mark.pruned
def test_frobenius_spatial():
    assert Frobenius_norm(w2d())(d(), t()) >= 0


@pytest.mark.pruned
def test_frobenius_spatial_multi_channel_minimal():
    assert (
        Frobenius_norm(w2d(), covariance_dim="spatial")(
            torch.randn(3, 2, 3, 4),
            torch.randn(3, 2, 3, 4),
        )
        >= 0
    )


@pytest.mark.pruned
def test_frobenius_spatial_minimal():
    assert (
        Frobenius_norm(w2d(), covariance_dim="spatial")(
            torch.randn(2, 1, 3, 4),
            torch.randn(2, 1, 3, 4),
        )
        >= 0
    )


@pytest.mark.pruned
def test_frobenius_channel_valid():
    data = torch.randn(5, 3, 2, 3, 4)
    target = torch.randn(5, 3, 2, 3, 4)

    assert (
        Frobenius_norm(w2d(), covariance_dim="channel")(data.mean(0), target.mean(0))
        >= 0
    )


@pytest.mark.pruned
def test_frobenius_channel_no_reduce():
    data = torch.randn(2, 3, 2, 3, 4)
    target = torch.randn(2, 3, 2, 3, 4)

    assert Frobenius_norm(w2d(), covariance_dim="channel")(data[0], target[0]) >= 0


@pytest.mark.pruned
def test_frobenius_generative():
    assert (
        Frobenius_norm(w2d())(
            torch.randn(2, 2, 2, 3, 4),
            torch.randn(2, 2, 2, 3, 4),
            generative_modeling=True,
        )
        >= 0
    )


def test_frobenius_channel_generative():
    assert (
        Frobenius_norm(w2d(), covariance_dim="channel")(
            torch.randn(2, 2, 2, 3, 4),
            torch.randn(2, 2, 2, 3, 4),
            generative_modeling=True,
        )
        >= 0
    )


def test_frobenius_generator():
    assert Frobenius_norm(w2d())(ens(), t(), generator=True) >= 0


@pytest.mark.pruned
def test_frobenius_sum():
    assert Frobenius_norm(w2d(), reduction="sum")(d(), t()) >= 0


def test_frobenius_direct_aggregate_sum():
    f = Frobenius_norm(w2d(), reduction="sum")
    assert f._aggregate(torch.tensor([[1.0, 2.0]]), output_size=2) >= 0


@pytest.mark.pruned
def test_frobenius_direct_aggregate_mean():
    f = Frobenius_norm(w2d(), reduction="mean")
    assert f._aggregate(torch.tensor([2.0]), output_size=2) >= 0


@pytest.mark.pruned
def test_frobenius_internal_values():
    assert Frobenius_norm(w2d())._aggregate(torch.tensor(5.0), output_size=10) >= 0


@pytest.mark.pruned
def test_frobenius_reduction_uppercase():
    assert Frobenius_norm(w2d(), reduction="MEAN")(d(), t()) >= 0


@pytest.mark.pruned
def test_frobenius_shape_fail():
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        Frobenius_norm(w2d())(d(), torch.zeros(1))


@pytest.mark.pruned
def test_frobenius_print(capsys):
    Frobenius_norm(w2d())(d(), t(), print_loss=True)
    assert "FLN" in capsys.readouterr().out

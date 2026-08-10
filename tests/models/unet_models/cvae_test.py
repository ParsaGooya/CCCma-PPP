from builtins import object
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch
import torch.nn as nn

import cccma_ppp.models.unet_models.cvae as module
from cccma_ppp.models.layers.conv import (
    ConvBlockConfig,
    LatentVector,
    TensorMask,
)
from cccma_ppp.models.unet_models.cvae import (
    Generation,
    Recognition,
    cVAEUNet,
    cVAEUNetConfig,
)


def make_block_config():
    return ConvBlockConfig(
        name="standard_conv",
        num_convolutions=1,
        kernel_size=3,
        normalization="none",
        padding_method="zeros",
        activation="relu",
        dropout_rate=None,
        bias=False,
        group_norm_groups=1,
    )


def make_config(**overrides):
    values = {
        "channels": [4, 8],
        "latent_size": 3,
        "condition_embedding_channels": [4, 8],
        "condition_embedding_size": 3,
        "latent_normalization": "layer",
        "condition_dependant_latent": False,
        "condemb_to_decoder": True,
        "block_config": make_block_config(),
        "upsampling_method": "bilinear",
        "upsampling_alignment_method": "padd",
        "transpose_kernel_sizes": [3],
        "add_skip_latent": False,
        "mask_pooling": "any",
        "mask_fraction_threshold": 0.5,
        "output_activation": "identity",
        "output_block_hidden_channels": 32,
        "init_method": "trunc_normal",
        "GENERATOR": None,
    }
    values.update(overrides)
    values.pop("bottleneck_dim", None)
    return cVAEUNetConfig(**values)


def make_bare_model(**overrides):
    model = object.__new__(cVAEUNet)
    nn.Module.__init__(model)

    values = {
        "latent_size": 3,
        "condition_embedding_size": 3,
        "condition_dependant_latent": False,
        "condition_dependant_flow": False,
        "condemb_to_decoder": True,
    }
    values.update(overrides)
    values.pop("bottleneck_dim", None)

    for name, value in values.items():
        setattr(model, name, value)

    return model


def make_forward_request(**overrides):
    values = {
        "target": torch.ones(2, 1, 4, 4),
        "target_mask": None,
        "condition": torch.ones(2, 1, 4, 4),
        "condition_mask": None,
        "added_features": None,
        "latent_sample_size": 3,
        "posterior_variance_limits": None,
        "output_sample_size": 0,
    }
    if "sample_size" in overrides:
        overrides["latent_sample_size"] = overrides.pop("sample_size")
    if "min_posterior_variance" in overrides:
        minimum = overrides.pop("min_posterior_variance")
        overrides["posterior_variance_limits"] = (
            (minimum, torch.tensor(float("inf"))) if minimum is not None else None
        )
    values.update(overrides)
    values.pop("bottleneck_dim", None)
    return SimpleNamespace(**values)


def make_predict_request(**overrides):
    values = {
        "condition": torch.ones(2, 1, 4, 4),
        "condition_mask": None,
        "added_features": None,
        "prior_flow": None,
        "latent_samples": None,
        "nstds": 1.0,
        "latent_sample_size": 3,
        "latent_sample_size": 3,
        "output_sample_size": 0,
    }
    if "sample_size" in overrides:
        overrides["latent_sample_size"] = overrides.pop("sample_size")
    values.update(overrides)
    if "sample_size" in overrides:
        values["latent_sample_size"] = overrides["sample_size"]
    values.pop("bottleneck_dim", None)
    return SimpleNamespace(**values)


class IdentityTensorMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.received = []

    def forward(self, value):
        self.received.append(value)
        return value


class FixedTensorMaskModule(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.received = []

    def forward(self, value):
        self.received.append(value)
        return self.output


class FixedLatentModule(nn.Module):
    def __init__(self, mu, log_var):
        super().__init__()
        self.mu = mu
        self.log_var = log_var
        self.received = []

    def forward(self, value):
        self.received.append(value)
        return LatentVector(
            mu=self.mu,
            log_var=self.log_var,
        )


class ShapeDownBlock(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self._output_shape = tuple(output_shape)
        self.received = []

    def output_shape(self, shape):
        return self._output_shape

    def forward(self, value):
        self.received.append(value)
        return value


class RecordingUpBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        value,
        skip=None,
        resize_shape=None,
    ):
        self.calls.append(
            {
                "value": value,
                "skip": skip,
                "resize_shape": resize_shape,
            }
        )
        return value


class ConstantTensorOutput(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.received = []

    def forward(self, value):
        self.received.append(value)
        return self.output


class StubRecognition(nn.Module):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.args = args
        self.kwargs = kwargs

        height, width = kwargs["input_spatial_shape"]

        self.spatial_shapes = [
            (height, width),
            (
                max(height // 2, 1),
                max(width // 2, 1),
            ),
        ]

        type(self).instances.append(self)

    def forward(self, value):
        batch_size = value.tensor.shape[0]
        latent_size = self.kwargs["latent_size"]

        return LatentVector(
            mu=torch.zeros(
                batch_size,
                latent_size,
            ),
            log_var=torch.zeros(
                batch_size,
                latent_size,
            ),
        )


class StubGeneration(nn.Module):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.args = args
        self.kwargs = kwargs

        type(self).instances.append(self)

    def forward(
        self,
        latent_samples,
        num_output_samples=0,
    ):
        batch_size = latent_samples.shape[0]
        output_channels = self.kwargs["output_channels"]
        height, width = self.kwargs["resize_shapes"][-1]

        output = torch.zeros(
            batch_size,
            output_channels,
            height,
            width,
            device=latent_samples.device,
            dtype=latent_samples.dtype,
        )

        if num_output_samples > 0:
            return output.unsqueeze(0).expand(
                num_output_samples,
                *output.shape,
            )

        return output


class DummyFlow:
    def __init__(
        self,
        condition_size=None,
        result=None,
    ):
        self.condition_size = condition_size
        self.result = result
        self.samples = None
        self.condition = None

    def inverse(self, samples, condition):
        self.samples = samples
        self.condition = condition

        return SimpleNamespace(
            e_samples=(samples if self.result is None else self.result)
        )


@pytest.fixture
def stub_model_components(monkeypatch):
    StubRecognition.instances.clear()
    StubGeneration.instances.clear()

    monkeypatch.setattr(
        module,
        "Recognition",
        StubRecognition,
    )
    monkeypatch.setattr(
        module,
        "Generation",
        StubGeneration,
    )
    monkeypatch.setattr(
        cVAEUNet,
        "_validate_checkpoint_compatibility",
        Mock(),
    )
    monkeypatch.setattr(
        cVAEUNet,
        "_initialize_weights",
        Mock(),
    )


def test_config_post_init_calls_validation(
    monkeypatch,
):
    validation = Mock()

    monkeypatch.setattr(
        module,
        "_unet_config_checks",
        validation,
    )

    config = object.__new__(cVAEUNetConfig)

    config.channels = [4, 8]
    config.latent_size = 3
    config.condition_embedding_channels = [4, 8]
    config.condition_embedding_size = 3
    config.transpose_kernel_sizes = [3]

    cVAEUNetConfig.__post_init__(config)

    validation.assert_called_once_with(config)


@pytest.mark.parametrize(
    ("channels", "expected"),
    [
        (
            [4],
            [],
        ),
        (
            [4, 8],
            [5],
        ),
        (
            [4, 8, 16],
            [5, 5],
        ),
        (
            [4, 8, 16, 32],
            [5, 5, 5],
        ),
    ],
)
def test_config_post_init_expands_integer_kernel(
    monkeypatch,
    channels,
    expected,
):
    monkeypatch.setattr(
        module,
        "_unet_config_checks",
        Mock(),
    )

    config = object.__new__(cVAEUNetConfig)

    config.channels = channels
    config.latent_size = 3
    config.condition_embedding_channels = channels
    config.condition_embedding_size = 3
    config.transpose_kernel_sizes = 5

    cVAEUNetConfig.__post_init__(config)

    assert config.transpose_kernel_sizes == expected


def test_config_post_init_preserves_kernel_list(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "_unet_config_checks",
        Mock(),
    )

    kernels = [
        3,
        (4, 5),
    ]

    config = object.__new__(cVAEUNetConfig)

    config.channels = [4, 8, 16]
    config.latent_size = 3
    config.condition_embedding_channels = [
        4,
        8,
        16,
    ]
    config.condition_embedding_size = 3
    config.transpose_kernel_sizes = kernels

    cVAEUNetConfig.__post_init__(config)

    assert config.transpose_kernel_sizes is kernels


def test_config_post_init_defaults_condition_channels(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "_unet_config_checks",
        Mock(),
    )

    config = object.__new__(cVAEUNetConfig)

    config.channels = [4, 8, 16]
    config.latent_size = 3
    config.condition_embedding_channels = None
    config.condition_embedding_size = 3
    config.transpose_kernel_sizes = [
        3,
        3,
    ]

    cVAEUNetConfig.__post_init__(config)

    assert config.condition_embedding_channels == [4, 8, 16]


def test_config_post_init_preserves_condition_channels(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "_unet_config_checks",
        Mock(),
    )

    condition_channels = [2, 6]

    config = object.__new__(cVAEUNetConfig)

    config.channels = [4, 8]
    config.latent_size = 3
    config.condition_embedding_channels = condition_channels
    config.condition_embedding_size = 3
    config.transpose_kernel_sizes = [3]

    cVAEUNetConfig.__post_init__(config)

    assert config.condition_embedding_channels is condition_channels


def test_config_post_init_defaults_condition_size(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "_unet_config_checks",
        Mock(),
    )

    config = object.__new__(cVAEUNetConfig)

    config.channels = [4, 8]
    config.latent_size = 7
    config.condition_embedding_channels = [
        4,
        8,
    ]
    config.condition_embedding_size = None
    config.transpose_kernel_sizes = [3]

    cVAEUNetConfig.__post_init__(config)

    assert config.condition_embedding_size == 7


def test_config_post_init_preserves_condition_size(
    monkeypatch,
):
    monkeypatch.setattr(
        module,
        "_unet_config_checks",
        Mock(),
    )

    config = object.__new__(cVAEUNetConfig)

    config.channels = [4, 8]
    config.latent_size = 7
    config.condition_embedding_channels = [
        4,
        8,
    ]
    config.condition_embedding_size = 11
    config.transpose_kernel_sizes = [3]

    cVAEUNetConfig.__post_init__(config)

    assert config.condition_embedding_size == 11


def test_config_build_delegates_to_model(
    monkeypatch,
):
    config = make_config()
    expected = object()

    constructor = Mock(return_value=expected)

    monkeypatch.setattr(
        module,
        "cVAEUNet",
        constructor,
    )

    input_shape = np.asarray([2, 8, 8])
    output_shape = np.asarray([1, 8, 8])

    result = config.build(
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=4,
    )

    assert result is expected

    constructor.assert_called_once_with(
        config=config,
        input_shape=input_shape,
        output_shape=output_shape,
        added_features_dim=4,
    )


def test_config_build_passes_optional_defaults(
    monkeypatch,
):
    config = make_config()

    constructor = Mock(return_value=object())

    monkeypatch.setattr(
        module,
        "cVAEUNet",
        constructor,
    )

    input_shape = np.asarray([2, 8, 8])

    config.build(input_shape=input_shape)

    constructor.assert_called_once_with(
        config=config,
        input_shape=input_shape,
        output_shape=None,
        added_features_dim=None,
    )


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 8),
        (2, 8, 8, 1),
    ],
)
def test_model_rejects_invalid_input_rank(
    input_shape,
):
    with pytest.raises(
        RuntimeError,
        match="expects 3D input shapes",
    ):
        cVAEUNet(
            config=make_config(),
            input_shape=input_shape,
        )


@pytest.mark.parametrize(
    "output_shape",
    [
        (1, 8),
        (1, 8, 8, 1),
    ],
)
def test_model_rejects_invalid_output_rank(
    output_shape,
):
    with pytest.raises(
        RuntimeError,
        match="expects 3D output shapes",
    ):
        cVAEUNet(
            config=make_config(),
            input_shape=(2, 8, 8),
            output_shape=output_shape,
        )


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 0, 8),
        (2, 8, 0),
        (2, -1, 8),
    ],
)
def test_model_rejects_nonpositive_spatial_dimension(
    input_shape,
):
    with pytest.raises(
        ValueError,
        match="spatial dimensions must be positive",
    ):
        cVAEUNet(
            config=make_config(),
            input_shape=input_shape,
        )


def test_model_rejects_depth_larger_than_input():
    config = make_config(
        channels=[
            4,
            8,
            16,
            32,
            64,
        ],
        condition_embedding_channels=[
            4,
            8,
            16,
            32,
            64,
        ],
        transpose_kernel_sizes=[
            3,
            3,
            3,
            3,
        ],
    )

    with pytest.raises(
        ValueError,
        match="depth is too large",
    ):
        cVAEUNet(
            config=config,
            input_shape=(2, 4, 4),
        )


def test_model_defaults_output_shape_to_input(
    stub_model_components,
):
    model = cVAEUNet(
        config=make_config(),
        input_shape=(2, 8, 8),
    )

    assert model.input_shape == (
        2,
        8,
        8,
    )
    assert model.output_shape == (
        2,
        8,
        8,
    )


def test_model_uses_explicit_output_shape(
    stub_model_components,
):
    model = cVAEUNet(
        config=make_config(),
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
    )

    assert model.output_shape == (
        1,
        8,
        8,
    )


def test_model_defaults_added_features_to_zero(
    stub_model_components,
):
    model = cVAEUNet(
        config=make_config(),
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
        added_features_dim=None,
    )

    assert model.added_features_dim == 0

    recognition = StubRecognition.instances[0]
    condition = StubRecognition.instances[1]

    assert recognition.kwargs["input_channels"] == 3
    assert condition.kwargs["input_channels"] == 2


def test_model_includes_added_feature_channels(
    stub_model_components,
):
    model = cVAEUNet(
        config=make_config(),
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
        added_features_dim=3,
    )

    assert model.added_features_dim == 3

    recognition = StubRecognition.instances[0]
    condition = StubRecognition.instances[1]

    assert recognition.kwargs["input_channels"] == 6
    assert condition.kwargs["input_channels"] == 5


def test_model_condition_skips_log_variance_for_independent_latent(
    stub_model_components,
):
    cVAEUNet(
        config=make_config(condition_dependant_latent=False),
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
    )

    condition = StubRecognition.instances[1]

    assert condition.kwargs["get_log_var"] is False


def test_model_condition_embedding_is_added_to_decoder(
    stub_model_components,
):
    model = cVAEUNet(
        config=make_config(
            latent_size=3,
            condition_embedding_size=5,
            condemb_to_decoder=True,
        ),
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
    )

    assert model.add_condition_size == 5

    assert StubGeneration.instances[0].kwargs["latent_size"] == 8


def test_model_loads_checkpoint_when_configured(
    stub_model_components,
    monkeypatch,
):
    loader = Mock()
    initializer = Mock()

    monkeypatch.setattr(
        cVAEUNet,
        "_load_state_dict",
        loader,
    )
    monkeypatch.setattr(
        cVAEUNet,
        "_initialize_weights",
        initializer,
    )

    checkpoint = object()

    config = make_config()
    config.checkpoint_config = checkpoint

    cVAEUNet(
        config=config,
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
    )

    loader.assert_called_once_with(checkpoint)
    initializer.assert_not_called()


def test_prepare_input_without_optional_values():
    model = make_bare_model()

    tensor = torch.ones(
        2,
        3,
        4,
        5,
    )

    result = model._prepare_input(
        x=tensor,
        x_mask=None,
    )

    assert result.tensor is tensor
    assert result.mask is None


def test_prepare_input_broadcasts_input_mask(
    monkeypatch,
):
    model = make_bare_model()

    tensor = torch.ones(
        2,
        3,
        4,
        4,
    )
    mask = torch.ones(
        2,
        1,
        4,
        4,
    )
    broadcast = torch.ones_like(tensor)

    broadcast_mask = Mock(return_value=broadcast)

    monkeypatch.setattr(
        module,
        "_broadcast_mask",
        broadcast_mask,
    )

    result = model._prepare_input(
        x=tensor,
        x_mask=mask,
    )

    broadcast_mask.assert_called_once_with(
        mask,
        tensor,
    )

    assert result.mask is broadcast


def test_prepare_input_resizes_and_concatenates_condition(
    monkeypatch,
):
    model = make_bare_model()

    tensor = torch.ones(
        2,
        2,
        4,
        4,
    )
    condition = torch.ones(
        2,
        3,
        2,
        2,
    )
    resized = torch.full(
        (
            2,
            3,
            4,
            4,
        ),
        2.0,
    )

    monkeypatch.setattr(
        module,
        "_broadcast_mask",
        Mock(return_value=None),
    )

    resize = Mock(return_value=resized)

    monkeypatch.setattr(
        module,
        "_resize_tensor",
        resize,
    )

    result = model._prepare_input(
        x=tensor,
        x_mask=None,
        condition=condition,
    )

    resize.assert_called_once_with(
        condition,
        tensor.shape[-2:],
    )

    assert result.tensor.shape == (
        2,
        5,
        4,
        4,
    )

    torch.testing.assert_close(
        result.tensor[:, :2],
        tensor,
    )
    torch.testing.assert_close(
        result.tensor[:, 2:],
        resized,
    )

    assert result.mask is None


def test_prepare_input_creates_condition_mask_when_missing(
    monkeypatch,
):
    model = make_bare_model()

    tensor = torch.ones(
        2,
        2,
        4,
        4,
    )
    input_mask = torch.zeros_like(tensor)
    condition = torch.ones(
        2,
        1,
        4,
        4,
    )

    monkeypatch.setattr(
        module,
        "_broadcast_mask",
        Mock(return_value=input_mask),
    )
    monkeypatch.setattr(
        module,
        "_resize_tensor",
        Mock(return_value=condition),
    )

    result = model._prepare_input(
        x=tensor,
        x_mask=input_mask,
        condition=condition,
        condition_mask=None,
    )

    assert result.mask.shape == (
        2,
        3,
        4,
        4,
    )

    torch.testing.assert_close(
        result.mask[:, :2],
        input_mask,
    )
    torch.testing.assert_close(
        result.mask[:, 2:],
        torch.ones_like(condition),
    )


def test_prepare_input_resizes_and_broadcasts_condition_mask(
    monkeypatch,
):
    model = make_bare_model()

    tensor = torch.ones(
        2,
        2,
        4,
        4,
    )
    input_mask = torch.ones_like(tensor)
    condition = torch.ones(
        2,
        1,
        2,
        2,
    )
    condition_mask = torch.ones(
        2,
        1,
        2,
        2,
    )
    resized_condition = torch.ones(
        2,
        1,
        4,
        4,
    )
    resized_mask = torch.zeros(
        2,
        1,
        4,
        4,
    )

    monkeypatch.setattr(
        module,
        "_resize_tensor",
        Mock(return_value=resized_condition),
    )

    monkeypatch.setattr(
        module,
        "_broadcast_mask",
        Mock(
            side_effect=[
                input_mask,
                resized_mask,
            ]
        ),
    )

    result = model._prepare_input(
        x=tensor,
        x_mask=input_mask,
        condition=condition,
        condition_mask=condition_mask,
    )

    (condition_mask,)
    (input_mask.shape[-2:],)

    torch.testing.assert_close(
        result.mask[:, -1:],
        resized_mask,
    )


def test_prepare_input_resizes_and_concatenates_features_without_mask(
    monkeypatch,
):
    model = make_bare_model()

    tensor = torch.ones(
        2,
        2,
        4,
        4,
    )
    features = torch.ones(
        2,
        3,
        1,
        1,
    )
    resized = torch.full(
        (
            2,
            3,
            4,
            4,
        ),
        4.0,
    )

    monkeypatch.setattr(
        module,
        "_broadcast_mask",
        Mock(return_value=None),
    )

    resize = Mock(return_value=resized)

    monkeypatch.setattr(
        module,
        "_resize_tensor",
        resize,
    )

    result = model._prepare_input(
        x=tensor,
        x_mask=None,
        added_features=features,
    )

    resize.assert_called_once_with(
        features,
        tensor.shape[-2:],
        mode="nearest",
    )

    assert result.tensor.shape == (
        2,
        5,
        4,
        4,
    )
    assert result.mask is None


def test_prepare_input_adds_valid_feature_mask(
    monkeypatch,
):
    model = make_bare_model()

    tensor = torch.ones(
        2,
        2,
        4,
        4,
    )
    input_mask = torch.zeros_like(tensor)
    features = torch.ones(
        2,
        3,
        1,
        1,
    )
    resized_features = torch.full(
        (
            2,
            3,
            4,
            4,
        ),
        4.0,
    )

    monkeypatch.setattr(
        module,
        "_broadcast_mask",
        Mock(return_value=input_mask),
    )
    monkeypatch.setattr(
        module,
        "_resize_tensor",
        Mock(return_value=resized_features),
    )

    result = model._prepare_input(
        x=tensor,
        x_mask=input_mask,
        added_features=features,
    )

    assert result.mask.shape == (
        2,
        5,
        4,
        4,
    )

    torch.testing.assert_close(
        result.mask[:, -3:],
        torch.ones_like(resized_features),
    )


def test_prepare_input_combines_condition_and_features(
    monkeypatch,
):
    model = make_bare_model()

    tensor = torch.ones(
        2,
        2,
        4,
        4,
    )
    condition = torch.ones(
        2,
        1,
        2,
        2,
    )
    features = torch.ones(
        2,
        3,
        1,
        1,
    )

    resized_condition = torch.full(
        (
            2,
            1,
            4,
            4,
        ),
        2.0,
    )
    resized_features = torch.full(
        (
            2,
            3,
            4,
            4,
        ),
        3.0,
    )

    monkeypatch.setattr(
        module,
        "_broadcast_mask",
        Mock(return_value=None),
    )
    monkeypatch.setattr(
        module,
        "_resize_tensor",
        Mock(
            side_effect=[
                resized_condition,
                resized_features,
            ]
        ),
    )

    result = model._prepare_input(
        x=tensor,
        x_mask=None,
        condition=condition,
        added_features=features,
    )

    assert result.tensor.shape == (
        2,
        6,
        4,
        4,
    )

    torch.testing.assert_close(
        result.tensor[:, :2],
        tensor,
    )
    torch.testing.assert_close(
        result.tensor[:, 2:3],
        resized_condition,
    )
    torch.testing.assert_close(
        result.tensor[:, 3:],
        resized_features,
    )


def configure_forward_model(
    *,
    training=False,
    generator=None,
):
    model = make_bare_model()

    model.train(training)

    model.config = SimpleNamespace(GENERATOR=generator)

    cond_mu = torch.ones(
        2,
        3,
    )
    cond_log_var = torch.zeros(
        2,
        3,
    )
    mu = torch.full(
        (
            2,
            3,
        ),
        2.0,
    )
    log_var = torch.full(
        (
            2,
            3,
        ),
        -1.0,
    )
    samples = torch.ones(
        4,
        2,
        3,
    )
    output = torch.ones(
        4,
        2,
        1,
        4,
        4,
    )

    model._condition = Mock(
        return_value=(
            cond_mu,
            cond_log_var,
        )
    )
    model._recognition = Mock(
        return_value=(
            mu,
            log_var,
        )
    )
    model._sample = Mock(return_value=samples)
    model._generate = Mock(return_value=output)

    model.deterministic_guess = None
    model._output_block = Mock(side_effect=lambda input, sample_sizes: input)
    return SimpleNamespace(
        model=model,
        cond_mu=cond_mu,
        cond_log_var=cond_log_var,
        mu=mu,
        log_var=log_var,
        samples=samples,
        output=output,
    )


def test_forward_returns_complete_cvae_output():
    setup = configure_forward_model()

    result = setup.model.forward(
        make_forward_request(
            sample_size=4,
            output_sample_size=2,
        )
    )

    assert result.output is setup.output
    assert result.mu is setup.mu
    assert result.log_var is setup.log_var
    assert result.samples is setup.samples
    assert result.cond_mu is setup.cond_mu
    assert result.cond_log_var is setup.cond_log_var

    setup.model._sample.assert_called_once_with(
        setup.mu,
        setup.log_var,
        4,
    )

    setup.model._generate.assert_called_once_with(
        latent_samples=setup.samples,
        condition_embedding=setup.cond_mu,
        num_output_samples=2,
    )


def test_forward_passes_request_values():
    setup = configure_forward_model()

    target = torch.ones(
        2,
        1,
        4,
        4,
    )
    target_mask = torch.ones_like(target)
    condition = torch.full_like(
        target,
        2.0,
    )
    condition_mask = torch.zeros_like(target)
    features = torch.ones(
        2,
        2,
        4,
        4,
    )

    setup.model.forward(
        make_forward_request(
            target=target,
            target_mask=target_mask,
            condition=condition,
            condition_mask=condition_mask,
            added_features=features,
        )
    )

    setup.model._condition.assert_called_once_with(
        condition=condition,
        condition_mask=condition_mask,
        added_features=features,
    )

    setup.model._recognition.assert_called_once_with(
        x=target,
        x_mask=target_mask,
        condition=condition,
        condition_mask=condition_mask,
        added_features=features,
    )


def test_forward_without_minimum_preserves_log_variance():
    setup = configure_forward_model()

    setup.model.forward(make_forward_request(min_posterior_variance=None))

    passed_log_var = setup.model._sample.call_args.args[1]

    assert passed_log_var is setup.log_var


def test_forward_clamps_log_variance():
    setup = configure_forward_model()

    log_var = torch.tensor(
        [
            [
                -10.0,
                -2.0,
                1.0,
            ],
            [
                -4.0,
                0.0,
                2.0,
            ],
        ]
    )

    setup.model._recognition.return_value = (
        setup.mu,
        log_var,
    )

    setup.model.forward(make_forward_request(min_posterior_variance=torch.tensor(-3.0)))

    passed_log_var = setup.model._sample.call_args.args[1]

    assert torch.all(passed_log_var >= -3)


def test_forward_training_uses_generator_noise_count():
    generator = SimpleNamespace(num_training_noise_samples=5)

    setup = configure_forward_model(
        training=True,
        generator=generator,
    )

    setup.model.forward(make_forward_request(output_sample_size=9))

    assert setup.model._generate.call_args.kwargs["num_output_samples"] == 5


def test_forward_evaluation_uses_request_noise_count():
    generator = SimpleNamespace(num_training_noise_samples=5)

    setup = configure_forward_model(
        training=False,
        generator=generator,
    )

    setup.model.forward(make_forward_request(output_sample_size=9))

    assert setup.model._generate.call_args.kwargs["num_output_samples"] == 9


def test_forward_training_without_generator_uses_request_count():
    setup = configure_forward_model(
        training=True,
        generator=None,
    )

    setup.model.forward(make_forward_request(output_sample_size=7))

    assert setup.model._generate.call_args.kwargs["num_output_samples"] == 7


def configure_predict_model(
    **model_overrides,
):
    model = make_bare_model(**model_overrides)

    cond_mu = torch.ones(
        2,
        3,
    )
    cond_log_var = torch.zeros(
        2,
        3,
    )
    output = torch.ones(
        3,
        2,
        1,
        4,
        4,
    )

    model._condition = Mock(
        return_value=(
            cond_mu,
            cond_log_var,
        )
    )
    model._generate = Mock(return_value=output)

    model.deterministic_guess = None
    model.deterministic_guess = None
    model._output_block = Mock(side_effect=lambda input, sample_sizes: input)
    return SimpleNamespace(
        model=model,
        cond_mu=cond_mu,
        cond_log_var=cond_log_var,
        output=output,
    )


def test_predict_uses_condition_dependent_prior():
    setup = configure_predict_model(
        condition_dependant_latent=True,
        condition_dependant_flow=False,
    )

    samples = torch.ones(
        4,
        2,
        3,
    )

    setup.model._sample = Mock(return_value=samples)

    result = setup.model.predict(
        make_predict_request(
            sample_size=4,
            nstds=2.0,
        )
    )

    setup.model._sample.assert_called_once_with(
        setup.cond_mu,
        setup.cond_log_var,
        4,
        std=2.0,
    )

    assert result.output is setup.output
    assert result.mu is None
    assert result.log_var is None
    assert result.samples is None
    assert result.cond_mu is setup.cond_mu
    assert result.cond_log_var is setup.cond_log_var


def test_predict_accepts_user_latent_samples():
    setup = configure_predict_model()

    samples = torch.ones(
        4,
        2,
        3,
    )

    result = setup.model.predict(
        make_predict_request(
            latent_samples=samples,
            sample_size=4,
        )
    )

    assert result.output is setup.output

    assert setup.model._generate.call_args.args[0] is samples


@pytest.mark.parametrize(
    "shape",
    [
        (
            2,
            2,
            3,
        ),
        (
            4,
            1,
            3,
        ),
        (
            4,
            2,
            2,
        ),
        (
            4,
            2,
            3,
            1,
        ),
    ],
)
def test_predict_rejects_invalid_user_latent_shape(
    shape,
):
    setup = configure_predict_model()

    with pytest.raises(
        ValueError,
        match="expected shape",
    ):
        setup.model.predict(
            make_predict_request(
                latent_samples=torch.ones(shape),
                sample_size=4,
            )
        )


def test_predict_applies_unconditional_flow(
    monkeypatch,
):
    setup = configure_predict_model()

    original = torch.arange(
        18,
        dtype=torch.float32,
    ).reshape(
        3,
        2,
        3,
    )

    transformed = (
        original.reshape(
            6,
            3,
        )
        + 10
    )

    setup.model._sample = Mock(return_value=original)

    flow = DummyFlow(
        condition_size=None,
        result=transformed,
    )

    setup.model.predict(
        make_predict_request(
            prior_flow=flow,
            sample_size=3,
        )
    )

    assert flow.condition is None
    assert flow.samples.shape == (
        6,
        3,
    )

    generated_samples = setup.model._generate.call_args.args[0]

    torch.testing.assert_close(
        generated_samples,
        transformed.reshape(
            3,
            2,
            3,
        ),
    )


def test_predict_applies_conditioned_flow(
    monkeypatch,
):
    setup = configure_predict_model()

    setup.cond_mu[:] = torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
            ],
            [
                4.0,
                5.0,
                6.0,
            ],
        ]
    )

    samples = torch.ones(
        3,
        2,
        3,
    )

    setup.model._sample = Mock(return_value=samples)

    flow = DummyFlow(condition_size=3)

    setup.model.predict(
        make_predict_request(
            prior_flow=flow,
            sample_size=3,
        )
    )

    expected_condition = (
        setup.cond_mu.unsqueeze(0)
        .expand(
            3,
            -1,
            -1,
        )
        .reshape(
            6,
            3,
        )
    )

    torch.testing.assert_close(
        flow.condition,
        expected_condition,
    )


def test_predict_skips_flow_for_user_latent_samples():
    setup = configure_predict_model()

    flow = DummyFlow(condition_size=3)

    samples = torch.ones(
        3,
        2,
        3,
    )

    setup.model.predict(
        make_predict_request(
            prior_flow=flow,
            latent_samples=samples,
            sample_size=3,
        )
    )

    assert flow.samples is None
    assert flow.condition is None


def test_predict_passes_condition_request_values():
    setup = configure_predict_model()

    condition = torch.ones(
        2,
        1,
        4,
        4,
    )
    condition_mask = torch.zeros_like(condition)
    features = torch.ones(
        2,
        2,
        4,
        4,
    )
    samples = torch.ones(
        3,
        2,
        3,
    )

    setup.model.predict(
        make_predict_request(
            condition=condition,
            condition_mask=condition_mask,
            added_features=features,
            latent_samples=samples,
        )
    )

    setup.model._condition.assert_called_once_with(
        condition=condition,
        condition_mask=condition_mask,
        added_features=features,
    )


def test_predict_passes_output_sample_count():
    setup = configure_predict_model()

    samples = torch.ones(
        3,
        2,
        3,
    )

    setup.model.predict(
        make_predict_request(
            latent_samples=samples,
            output_sample_size=5,
        )
    )

    assert setup.model._generate.call_args.kwargs["num_output_samples"] == 5


def test_generate_appends_condition_embedding():
    model = make_bare_model(condemb_to_decoder=True)

    latent = torch.ones(
        3,
        2,
        4,
    )
    condition = torch.full(
        (
            2,
            2,
        ),
        5.0,
    )

    model.generation = Mock(
        return_value=torch.ones(
            6,
            1,
            4,
            4,
        )
    )

    result = model._generate(
        latent_samples=latent,
        condition_embedding=condition,
    )

    received = model.generation.call_args.args[0]

    assert received.shape == (
        6,
        6,
    )

    expected_condition = (
        condition.unsqueeze(0)
        .expand(
            3,
            -1,
            -1,
        )
        .reshape(
            6,
            2,
        )
    )

    torch.testing.assert_close(
        received[:, -2:],
        expected_condition,
    )

    assert result.shape == (
        3,
        2,
        1,
        4,
        4,
    )


def test_generate_does_not_append_condition_when_disabled():
    model = make_bare_model(condemb_to_decoder=False)

    model.generation = Mock(
        return_value=torch.ones(
            6,
            1,
            4,
            4,
        )
    )

    result = model._generate(
        latent_samples=torch.ones(
            3,
            2,
            4,
        ),
        condition_embedding=torch.ones(
            2,
            2,
        ),
    )

    received = model.generation.call_args.args[0]

    assert received.shape == (
        6,
        4,
    )
    assert result.shape == (
        3,
        2,
        1,
        4,
        4,
    )


def test_generate_does_not_append_missing_condition():
    model = make_bare_model(condemb_to_decoder=True)

    model.generation = Mock(
        return_value=torch.ones(
            6,
            1,
            4,
            4,
        )
    )

    model._generate(
        latent_samples=torch.ones(
            3,
            2,
            4,
        ),
        condition_embedding=None,
    )

    received = model.generation.call_args.args[0]

    assert received.shape == (
        6,
        4,
    )


def test_generate_restores_sample_and_batch_dimensions():
    model = make_bare_model(condemb_to_decoder=False)

    model.generation = Mock(
        return_value=torch.ones(
            6,
            5,
            7,
            8,
        )
    )

    result = model._generate(
        latent_samples=torch.ones(
            3,
            2,
            4,
        ),
        num_output_samples=0,
    )

    assert result.shape == (
        3,
        2,
        5,
        7,
        8,
    )


def test_generate_restores_output_noise_dimension():
    model = make_bare_model(condemb_to_decoder=False)

    model.generation = Mock(
        return_value=torch.ones(
            5,
            6,
            1,
            4,
            4,
        )
    )

    result = model._generate(
        latent_samples=torch.ones(
            3,
            2,
            4,
        ),
        num_output_samples=5,
    )

    assert result.shape == (
        5,
        3,
        2,
        1,
        4,
        4,
    )


def test_generate_passes_output_sample_count_to_generation():
    model = make_bare_model(condemb_to_decoder=False)

    model.generation = Mock(
        return_value=torch.ones(
            4,
            6,
            1,
            4,
            4,
        )
    )

    model._generate(
        latent_samples=torch.ones(
            3,
            2,
            4,
        ),
        num_output_samples=4,
    )

    assert model.generation.call_args.args[1] == 4


def make_recognition_config(
    **overrides,
):
    return make_config(
        channels=[
            4,
            8,
            16,
        ],
        condition_embedding_channels=[
            4,
            8,
            16,
        ],
        transpose_kernel_sizes=[
            3,
            3,
        ],
        **overrides,
    )


def test_recognition_builds_expected_down_blocks(
    monkeypatch,
):
    config = make_recognition_config()

    initial = IdentityTensorMask()
    bottleneck = FixedLatentModule(
        torch.ones(
            2,
            5,
        ),
        torch.zeros(
            2,
            5,
        ),
    )

    build = Mock(
        side_effect=[
            initial,
            bottleneck,
        ]
    )

    down_block = Mock(
        side_effect=[
            ShapeDownBlock((4, 4)),
            ShapeDownBlock((2, 2)),
        ]
    )

    monkeypatch.setattr(
        module,
        "build_conv_block",
        build,
    )
    monkeypatch.setattr(
        module,
        "DownBlock",
        down_block,
    )

    recognition = Recognition(
        input_channels=2,
        input_spatial_shape=(8, 8),
        channels=[
            4,
            8,
            16,
        ],
        latent_size=5,
        config=config,
    )

    assert len(recognition.down_blocks) == 2

    assert recognition.spatial_shapes == [
        (
            8,
            8,
        ),
        (
            4,
            4,
        ),
        (
            2,
            2,
        ),
    ]


def test_recognition_initial_mapping_arguments(
    monkeypatch,
):
    config = make_recognition_config()

    build = Mock(
        side_effect=[
            IdentityTensorMask(),
            FixedLatentModule(
                torch.ones(
                    2,
                    5,
                ),
                torch.zeros(
                    2,
                    5,
                ),
            ),
        ]
    )

    monkeypatch.setattr(
        module,
        "build_conv_block",
        build,
    )
    monkeypatch.setattr(
        module,
        "DownBlock",
        Mock(
            side_effect=[
                ShapeDownBlock((4, 4)),
                ShapeDownBlock((2, 2)),
            ]
        ),
    )

    Recognition(
        input_channels=7,
        input_spatial_shape=(8, 8),
        channels=[
            4,
            8,
            16,
        ],
        latent_size=5,
        config=config,
    )

    assert build.call_args_list[0] == call(
        7,
        4,
        config.block_config,
    )


@pytest.mark.parametrize(
    "get_log_var",
    [
        True,
        False,
    ],
)
def test_recognition_bottleneck_arguments(
    monkeypatch,
    get_log_var,
):
    config = make_recognition_config(latent_normalization="layer")

    build = Mock(
        side_effect=[
            IdentityTensorMask(),
            FixedLatentModule(
                torch.ones(
                    2,
                    5,
                ),
                torch.zeros(
                    2,
                    5,
                ),
            ),
        ]
    )

    monkeypatch.setattr(
        module,
        "build_conv_block",
        build,
    )
    monkeypatch.setattr(
        module,
        "DownBlock",
        Mock(
            side_effect=[
                ShapeDownBlock((4, 4)),
                ShapeDownBlock((2, 2)),
            ]
        ),
    )

    Recognition(
        input_channels=2,
        input_spatial_shape=(8, 8),
        channels=[
            4,
            8,
            16,
        ],
        latent_size=5,
        config=config,
        get_log_var=get_log_var,
    )

    assert build.call_args_list[1] == call(
        16,
        16,
        config.block_config,
        latent_size=5,
        block_output_shape=(
            16,
            2,
            2,
        ),
        get_log_var=get_log_var,
        latent_normalization="layer",
    )


def test_recognition_with_single_channel_level(
    monkeypatch,
):
    config = make_config(
        channels=[4],
        condition_embedding_channels=[4],
        transpose_kernel_sizes=[],
    )

    build = Mock(
        side_effect=[
            IdentityTensorMask(),
            FixedLatentModule(
                torch.ones(
                    2,
                    3,
                ),
                torch.zeros(
                    2,
                    3,
                ),
            ),
        ]
    )

    monkeypatch.setattr(
        module,
        "build_conv_block",
        build,
    )

    recognition = Recognition(
        input_channels=2,
        input_spatial_shape=(8, 8),
        channels=[4],
        latent_size=3,
        config=config,
    )

    assert len(recognition.down_blocks) == 0

    assert recognition.spatial_shapes == [
        (
            8,
            8,
        )
    ]


def test_recognition_forward_applies_all_modules():
    recognition = object.__new__(Recognition)
    nn.Module.__init__(recognition)

    initial = IdentityTensorMask()

    first = ShapeDownBlock((4, 4))
    second = ShapeDownBlock((2, 2))

    expected = LatentVector(
        mu=torch.ones(
            2,
            3,
        ),
        log_var=torch.zeros(
            2,
            3,
        ),
    )

    bottleneck = FixedLatentModule(
        expected.mu,
        expected.log_var,
    )

    recognition.initial_mapping = initial
    recognition.down_blocks = nn.ModuleList(
        [
            first,
            second,
        ]
    )
    recognition.bottleneck = bottleneck

    value = TensorMask(
        tensor=torch.ones(
            2,
            2,
            8,
            8,
        ),
        mask=None,
    )

    result = recognition(value)

    assert result.mu is expected.mu
    assert result.log_var is expected.log_var

    assert initial.received == [value]
    assert first.received == [value]
    assert second.received == [value]
    assert bottleneck.received == [value]


def test_recognition_forward_without_down_blocks():
    recognition = object.__new__(Recognition)
    nn.Module.__init__(recognition)

    initial = IdentityTensorMask()

    mu = torch.ones(
        2,
        3,
    )
    log_var = torch.zeros(
        2,
        3,
    )

    bottleneck = FixedLatentModule(
        mu,
        log_var,
    )

    recognition.initial_mapping = initial
    recognition.down_blocks = nn.ModuleList()
    recognition.bottleneck = bottleneck

    value = TensorMask(
        tensor=torch.ones(
            2,
            2,
            8,
            8,
        ),
        mask=None,
    )

    result = recognition(value)

    assert result.mu is mu
    assert result.log_var is log_var
    assert bottleneck.received == [value]


def make_generation_config(
    *,
    generator=None,
    **overrides,
):
    config = make_config(**overrides)
    config.GENERATOR = generator
    return config


def make_generator_config(
    *,
    noise_level,
    num_training_noise_samples=2,
):
    return SimpleNamespace(
        noise_level=noise_level,
        num_training_noise_samples=(num_training_noise_samples),
    )


def test_generation_builds_linear_mapping(
    monkeypatch,
):
    up_block = Mock(return_value=RecordingUpBlock())
    output = Mock(return_value=nn.Identity())

    monkeypatch.setattr(
        module,
        "UpBlock",
        up_block,
    )
    (output,)

    generation = Generation(
        latent_size=5,
        channels=[8],
        resize_shapes=[
            (
                2,
                3,
            ),
            (
                4,
                6,
            ),
        ],
        config=make_generation_config(),
    )

    assert generation.combine_latent.in_features == 5
    assert generation.combine_latent.out_features == 48
    assert generation.bottleneck_shape == (
        2,
        3,
    )
    assert generation.resize_shapes == [
        (
            4,
            6,
        )
    ]


def test_generation_builds_up_blocks_in_sequence(
    monkeypatch,
):
    up_block = Mock(
        side_effect=[
            RecordingUpBlock(),
            RecordingUpBlock(),
        ]
    )

    monkeypatch.setattr(
        module,
        "UpBlock",
        up_block,
    )

    config = make_generation_config(
        channels=[
            4,
            8,
            16,
        ],
        condition_embedding_channels=[
            4,
            8,
            16,
        ],
        transpose_kernel_sizes=[
            3,
            5,
        ],
    )

    Generation(
        latent_size=3,
        channels=[
            16,
            4,
            8,
        ],
        resize_shapes=[
            (
                2,
                2,
            ),
            (
                4,
                4,
            ),
            (
                8,
                8,
            ),
        ],
        config=config,
    )

    first = up_block.call_args_list[0].kwargs
    second = up_block.call_args_list[1].kwargs

    assert first["input_channels"] == 16
    assert first["out_channels"] == 4
    assert first["transpose_kernel_size"] == 3

    assert second["input_channels"] == 4
    assert second["out_channels"] == 8
    assert second["transpose_kernel_size"] == 5


@pytest.mark.parametrize(
    (
        "noise_level",
        "expected_block_noise",
        "expected_up_noise",
    ),
    [
        (
            "full",
            True,
            True,
        ),
        (
            "medium",
            True,
            True,
        ),
        (
            "low",
            False,
            True,
        ),
    ],
)
def test_generation_single_block_noise_configuration(
    monkeypatch,
    noise_level,
    expected_block_noise,
    expected_up_noise,
):
    up_block = Mock(return_value=RecordingUpBlock())

    monkeypatch.setattr(
        module,
        "UpBlock",
        up_block,
    )

    Generation(
        latent_size=3,
        channels=[8, 4],
        resize_shapes=[
            (
                2,
                2,
            ),
            (
                4,
                4,
            ),
        ],
        config=make_generation_config(
            generator=make_generator_config(noise_level=noise_level)
        ),
    )

    kwargs = up_block.call_args.kwargs

    assert kwargs["inject_noise_in_block"] is expected_block_noise
    assert kwargs["inject_noise"] is expected_up_noise


def test_generation_medium_noise_only_on_last_block(
    monkeypatch,
):
    up_block = Mock(
        side_effect=[
            RecordingUpBlock(),
            RecordingUpBlock(),
            RecordingUpBlock(),
        ]
    )

    monkeypatch.setattr(
        module,
        "UpBlock",
        up_block,
    )

    config = make_generation_config(
        generator=make_generator_config(noise_level="medium"),
        channels=[
            4,
            8,
            16,
            32,
        ],
        condition_embedding_channels=[
            4,
            8,
            16,
            32,
        ],
        transpose_kernel_sizes=[
            3,
            3,
            3,
        ],
    )

    Generation(
        latent_size=3,
        channels=[
            64,
            32,
            16,
            8,
        ],
        resize_shapes=[
            (
                1,
                1,
            ),
            (
                2,
                2,
            ),
            (
                4,
                4,
            ),
            (
                8,
                8,
            ),
        ],
        config=config,
    )

    assert [item.kwargs["inject_noise"] for item in up_block.call_args_list] == [
        False,
        False,
        True,
    ]


def test_generation_full_noise_on_all_blocks(
    monkeypatch,
):
    up_block = Mock(
        side_effect=[
            RecordingUpBlock(),
            RecordingUpBlock(),
        ]
    )

    monkeypatch.setattr(
        module,
        "UpBlock",
        up_block,
    )

    config = make_generation_config(
        generator=make_generator_config(noise_level="full"),
        channels=[
            4,
            8,
            16,
        ],
        condition_embedding_channels=[
            4,
            8,
            16,
        ],
        transpose_kernel_sizes=[
            3,
            3,
        ],
    )

    Generation(
        latent_size=3,
        channels=[
            32,
            16,
            8,
        ],
        resize_shapes=[
            (
                2,
                2,
            ),
            (
                4,
                4,
            ),
            (
                8,
                8,
            ),
        ],
        config=config,
    )

    assert [item.kwargs["inject_noise"] for item in up_block.call_args_list] == [
        True,
        True,
    ]


def test_generation_low_noise_uses_no_internal_noise(
    monkeypatch,
):
    up_block = Mock(
        side_effect=[
            RecordingUpBlock(),
            RecordingUpBlock(),
        ]
    )

    monkeypatch.setattr(
        module,
        "UpBlock",
        up_block,
    )

    config = make_generation_config(
        generator=make_generator_config(noise_level="low"),
        channels=[
            4,
            8,
            16,
        ],
        condition_embedding_channels=[
            4,
            8,
            16,
        ],
        transpose_kernel_sizes=[
            3,
            3,
        ],
    )

    Generation(
        latent_size=3,
        channels=[
            32,
            16,
            8,
        ],
        resize_shapes=[
            (
                2,
                2,
            ),
            (
                4,
                4,
            ),
            (
                8,
                8,
            ),
        ],
        config=config,
    )

    assert all(
        item.kwargs["inject_noise_in_block"] is False
        for item in up_block.call_args_list
    )


def test_generation_without_generator_disables_all_noise(
    monkeypatch,
):
    up_block = Mock(
        side_effect=[
            RecordingUpBlock(),
            RecordingUpBlock(),
        ]
    )

    monkeypatch.setattr(
        module,
        "UpBlock",
        up_block,
    )

    config = make_generation_config(
        generator=None,
        channels=[
            4,
            8,
            16,
        ],
        condition_embedding_channels=[
            4,
            8,
            16,
        ],
        transpose_kernel_sizes=[
            3,
            3,
        ],
    )

    Generation(
        latent_size=3,
        channels=[
            32,
            16,
            8,
        ],
        resize_shapes=[
            (
                2,
                2,
            ),
            (
                4,
                4,
            ),
            (
                8,
                8,
            ),
        ],
        config=config,
    )

    assert all(item.kwargs["inject_noise"] is False for item in up_block.call_args_list)

    assert all(
        item.kwargs["inject_noise_in_block"] is False
        for item in up_block.call_args_list
    )


def make_generation_without_constructor(
    *,
    generator=None,
    resize_shapes=None,
):
    generation = object.__new__(Generation)
    nn.Module.__init__(generation)

    generation.bottleneck_dim = 2
    generation.bottleneck_shape = (
        2,
        2,
    )
    generation.resize_shapes = (
        [
            (
                4,
                4,
            )
        ]
        if resize_shapes is None
        else resize_shapes
    )

    generation.config = SimpleNamespace(GENERATOR=generator)

    generation.combine_latent = nn.Linear(
        4,
        8,
    )

    generation.up_blocks = nn.ModuleList(
        [RecordingUpBlock() for _ in generation.resize_shapes]
    )

    generation.output = nn.Identity()

    return generation


def test_generation_forward_reshapes_linear_output():
    generation = make_generation_without_constructor()

    with torch.no_grad():
        generation.combine_latent.weight.zero_()
        generation.combine_latent.bias.copy_(
            torch.arange(
                8,
                dtype=torch.float32,
            )
        )

    generation(
        torch.ones(
            3,
            4,
        )
    )

    received = generation.up_blocks[0].calls[0]["value"]

    assert received.tensor.shape == (
        3,
        2,
        2,
        2,
    )
    assert received.mask is None


def test_generation_forward_passes_resize_shapes():
    generation = make_generation_without_constructor(
        resize_shapes=[
            (
                4,
                4,
            ),
            (
                8,
                8,
            ),
        ]
    )

    generation(
        torch.ones(
            3,
            4,
        )
    )

    assert generation.up_blocks[0].calls[0]["resize_shape"] == (
        4,
        4,
    )
    assert generation.up_blocks[1].calls[0]["resize_shape"] == (
        8,
        8,
    )


def test_generation_forward_passes_no_skip():
    generation = make_generation_without_constructor()

    generation(
        torch.ones(
            3,
            4,
        )
    )

    assert generation.up_blocks[0].calls[0]["skip"] is None


def test_generation_repeats_tensor_mask_for_generator(
    monkeypatch,
):
    generation = make_generation_without_constructor(
        generator=make_generator_config(noise_level="full")
    )

    repeated = TensorMask(
        tensor=torch.ones(
            6,
            2,
            2,
            2,
        ),
        mask=None,
    )

    repeat = Mock(return_value=repeated)

    monkeypatch.setattr(
        module,
        "_repeat_tensor_mask",
        repeat,
    )

    result = generation(
        torch.ones(
            3,
            4,
        ),
        num_output_samples=2,
    )

    repeat.assert_called_once()

    assert repeat.call_args.kwargs["repeats"] == 2

    assert result.shape == (
        2,
        3,
        2,
        2,
        2,
    )


def test_generation_does_not_repeat_without_generator(
    monkeypatch,
):
    generation = make_generation_without_constructor(generator=None)

    repeat = Mock()

    monkeypatch.setattr(
        module,
        "_repeat_tensor_mask",
        repeat,
    )

    generation(
        torch.ones(
            3,
            4,
        ),
        num_output_samples=2,
    )

    repeat.assert_not_called()


def test_generation_does_not_repeat_for_zero_output_samples(
    monkeypatch,
):
    generation = make_generation_without_constructor(
        generator=make_generator_config(noise_level="full")
    )

    repeat = Mock()

    monkeypatch.setattr(
        module,
        "_repeat_tensor_mask",
        repeat,
    )

    generation(
        torch.ones(
            3,
            4,
        ),
        num_output_samples=0,
    )

    repeat.assert_not_called()


def test_generation_generator_reshapes_output_samples():
    generation = make_generation_without_constructor(
        generator=make_generator_config(noise_level="full")
    )

    result = generation(
        torch.ones(
            3,
            4,
        ),
        num_output_samples=2,
    )

    assert result.shape == (
        2,
        3,
        2,
        2,
        2,
    )


def test_generation_without_generator_preserves_batch_dimension():
    generation = make_generation_without_constructor(generator=None)

    result = generation(
        torch.ones(
            3,
            4,
        ),
        num_output_samples=0,
    )

    assert result.shape == (
        3,
        2,
        2,
        2,
    )


def test_generation_forward_with_no_up_blocks():
    generation = make_generation_without_constructor(resize_shapes=[])

    result = generation(
        torch.ones(
            3,
            4,
        )
    )

    assert result.shape == (
        3,
        2,
        2,
        2,
    )


def test_config_expects_mask_for_partial_convolution():
    from cccma_ppp.models.layers.conv import PartialConvBlockConfig

    config = make_config(
        block_config=PartialConvBlockConfig(
            name="partial_conv",
            num_convolutions=1,
            kernel_size=3,
            normalization="none",
            padding_method="zeros",
            activation="relu",
            dropout_rate=None,
            bias=False,
            group_norm_groups=1,
        )
    )

    assert config.EXPECTS_MASK is True


def test_config_expects_mask_for_partial_convnext():
    from cccma_ppp.models.layers.conv import ConvNeXtBlockConfig

    config = make_config(
        block_config=ConvNeXtBlockConfig(
            name="convnext",
            num_blocks=1,
            kernel_size=3,
            expansion_ratio=2,
            padding_method="zeros",
            layer_scale_init=1e-6,
            dropout_rate=0.0,
            drop_path_rate=0.0,
            use_partial_conv=True,
        )
    )

    assert config.EXPECTS_MASK is True


def test_config_does_not_expect_mask_for_standard_convolution():
    config = make_config(
        block_config=make_block_config(),
    )

    assert config.EXPECTS_MASK is False


def test_config_does_not_expect_mask_for_standard_convnext():
    from cccma_ppp.models.layers.conv import ConvNeXtBlockConfig

    config = make_config(
        block_config=ConvNeXtBlockConfig(
            name="convnext",
            num_blocks=1,
            kernel_size=3,
            expansion_ratio=2,
            padding_method="zeros",
            layer_scale_init=1e-6,
            dropout_rate=0.0,
            drop_path_rate=0.0,
            use_partial_conv=False,
        )
    )

    assert config.EXPECTS_MASK is False


def test_config_without_deterministic_guess_disables_shared_output():
    config = make_config(
        deterministic_guess_config=None,
    )

    assert config.deterministic_guess_config is None
    assert config.share_output_block is False


def test_config_resolves_deterministic_guess():
    deterministic_model_config = SimpleNamespace(
        channels=[4, 8],
        GENERATOR=None,
        output_activation="identity",
        output_block_hidden_channels=32,
    )

    selector = SimpleNamespace(
        share_output_block=False,
        get_model_config=Mock(
            return_value=deterministic_model_config,
        ),
    )

    config = make_config(
        deterministic_guess_config=selector,
    )

    selector.get_model_config.assert_called_once_with()
    assert config.deterministic_guess_config is deterministic_model_config
    assert config.share_output_block is False


def test_config_rejects_deterministic_guess_channel_mismatch():
    deterministic_model_config = SimpleNamespace(
        channels=[5, 8],
        GENERATOR=None,
        output_activation="identity",
        output_block_hidden_channels=32,
    )

    selector = SimpleNamespace(
        share_output_block=False,
        get_model_config=Mock(
            return_value=deterministic_model_config,
        ),
    )

    with pytest.raises(
        ValueError,
        match="same number of channels",
    ):
        make_config(
            deterministic_guess_config=selector,
        )


def test_config_rejects_generator_in_deterministic_guess():
    deterministic_model_config = SimpleNamespace(
        channels=[4, 8],
        GENERATOR=object(),
        output_activation="identity",
        output_block_hidden_channels=32,
    )

    selector = SimpleNamespace(
        share_output_block=False,
        get_model_config=Mock(
            return_value=deterministic_model_config,
        ),
    )

    with pytest.raises(
        ValueError,
        match="cannot have GENERATOR",
    ):
        make_config(
            deterministic_guess_config=selector,
        )


@pytest.mark.parametrize(
    (
        "deterministic_activation",
        "deterministic_hidden_channels",
    ),
    [
        (
            "tanh",
            32,
        ),
        (
            "identity",
            16,
        ),
        (
            "sigmoid",
            64,
        ),
    ],
)
def test_config_rejects_shared_output_block_mismatch(
    deterministic_activation,
    deterministic_hidden_channels,
):
    deterministic_model_config = SimpleNamespace(
        channels=[4, 8],
        GENERATOR=None,
        output_activation=deterministic_activation,
        output_block_hidden_channels=deterministic_hidden_channels,
    )

    selector = SimpleNamespace(
        share_output_block=True,
        get_model_config=Mock(
            return_value=deterministic_model_config,
        ),
    )

    with pytest.raises(
        ValueError,
        match="share_output_block",
    ):
        make_config(
            deterministic_guess_config=selector,
            output_activation="identity",
            output_block_hidden_channels=32,
        )


def test_config_accepts_matching_shared_output_block():
    deterministic_model_config = SimpleNamespace(
        channels=[4, 8],
        GENERATOR=None,
        output_activation="identity",
        output_block_hidden_channels=32,
    )

    selector = SimpleNamespace(
        share_output_block=True,
        get_model_config=Mock(
            return_value=deterministic_model_config,
        ),
    )

    config = make_config(
        deterministic_guess_config=selector,
        output_activation="identity",
        output_block_hidden_channels=32,
    )

    assert config.share_output_block is True
    assert config.deterministic_guess_config is deterministic_model_config


def test_config_shared_output_mismatch_checks_activation_first():
    deterministic_model_config = SimpleNamespace(
        channels=[4, 8],
        GENERATOR=None,
        output_activation="tanh",
        output_block_hidden_channels=64,
    )

    selector = SimpleNamespace(
        share_output_block=True,
        get_model_config=Mock(
            return_value=deterministic_model_config,
        ),
    )

    with pytest.raises(
        ValueError,
        match="same output_blovk activation and hidden_channels",
    ):
        make_config(
            deterministic_guess_config=selector,
            output_activation="identity",
            output_block_hidden_channels=32,
        )


def test_model_condition_requests_log_variance_for_dependent_latent(
    stub_model_components,
):
    cVAEUNet(
        config=make_config(
            condition_dependant_latent=True,
        ),
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
    )

    condition = StubRecognition.instances[1]

    assert condition.kwargs["get_log_var"] is True


def test_model_builds_deterministic_guess(
    stub_model_components,
):
    deterministic_guess = SimpleNamespace(
        output_block=object(),
    )
    deterministic_config = SimpleNamespace(
        build=Mock(
            return_value=deterministic_guess,
        ),
    )

    config = make_config()
    config.deterministic_guess_config = deterministic_config
    config.share_output_block = False

    model = cVAEUNet(
        config=config,
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
        added_features_dim=3,
    )

    deterministic_config.build.assert_called_once_with(
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
        added_features_dim=3,
    )
    assert model.deterministic_guess is deterministic_guess


def test_model_uses_shared_deterministic_output_block(
    stub_model_components,
):
    shared_output = nn.Identity()
    deterministic_guess = SimpleNamespace(
        output_block=shared_output,
    )
    deterministic_config = SimpleNamespace(
        build=Mock(
            return_value=deterministic_guess,
        ),
    )

    config = make_config()
    config.deterministic_guess_config = deterministic_config
    config.share_output_block = True

    model = cVAEUNet(
        config=config,
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
    )

    assert model.output is shared_output


def test_model_builds_independent_output_block(
    stub_model_components,
    monkeypatch,
):
    output = nn.Identity()
    output_constructor = Mock(
        return_value=output,
    )

    monkeypatch.setattr(
        module,
        "UNetOutput",
        output_constructor,
    )

    config = make_config(
        output_block_hidden_channels=16,
        output_activation="tanh",
    )
    config.share_output_block = False

    model = cVAEUNet(
        config=config,
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
    )

    output_constructor.assert_called_once_with(
        in_channels=4,
        out_channels=1,
        hidden_channels=16,
        activation="tanh",
    )
    assert model.output is output


def test_model_initializes_weights_when_checkpoint_is_absent(
    stub_model_components,
    monkeypatch,
):
    initializer = Mock()

    monkeypatch.setattr(
        cVAEUNet,
        "_initialize_weights",
        initializer,
    )

    config = make_config(
        init_method="trunc_normal",
    )
    config.checkpoint_config = None

    model = cVAEUNet(
        config=config,
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
    )

    initializer.assert_called_once_with(
        "trunc_normal",
        exclude=(model.deterministic_guess,),
    )


def test_model_validates_checkpoint_compatibility(
    stub_model_components,
    monkeypatch,
):
    validation = Mock()

    monkeypatch.setattr(
        cVAEUNet,
        "_validate_checkpoint_compatibility",
        validation,
    )

    cVAEUNet(
        config=make_config(),
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
    )

    validation.assert_called_once_with(
        input_shape=(2, 8, 8),
        output_shape=(1, 8, 8),
    )


def test_prepare_input_does_not_broadcast_condition_mask_without_input_mask(
    monkeypatch,
):
    model = make_bare_model()

    tensor = torch.ones(
        2,
        2,
        4,
        4,
    )
    condition = torch.ones(
        2,
        1,
        4,
        4,
    )
    condition_mask = torch.zeros_like(
        condition,
    )

    broadcast = Mock(
        return_value=None,
    )
    monkeypatch.setattr(
        module,
        "_broadcast_mask",
        broadcast,
    )
    monkeypatch.setattr(
        module,
        "_resize_tensor",
        Mock(return_value=condition),
    )

    result = model._prepare_input(
        x=tensor,
        x_mask=None,
        condition=condition,
        condition_mask=condition_mask,
    )

    broadcast.assert_called_once_with(
        None,
        tensor,
    )
    assert result.mask is None


def test_prepare_input_feature_resize_uses_combined_spatial_shape(
    monkeypatch,
):
    model = make_bare_model()

    tensor = torch.ones(
        2,
        2,
        4,
        5,
    )
    condition = torch.ones(
        2,
        1,
        2,
        2,
    )
    features = torch.ones(
        2,
        3,
        1,
        1,
    )

    resized_condition = torch.ones(
        2,
        1,
        4,
        5,
    )
    resized_features = torch.ones(
        2,
        3,
        4,
        5,
    )

    resize = Mock(
        side_effect=[
            resized_condition,
            resized_features,
        ]
    )

    monkeypatch.setattr(
        module,
        "_resize_tensor",
        resize,
    )
    monkeypatch.setattr(
        module,
        "_broadcast_mask",
        Mock(return_value=None),
    )

    model._prepare_input(
        x=tensor,
        x_mask=None,
        condition=condition,
        added_features=features,
    )

    assert resize.call_args_list == [
        call(
            condition,
            (4, 5),
        ),
        call(
            features,
            (4, 5),
            mode="nearest",
        ),
    ]


def test_forward_uses_two_sample_dimensions_without_output_sampling():
    setup = configure_forward_model()

    setup.model.forward(
        make_forward_request(
            latent_sample_size=4,
            output_sample_size=0,
        )
    )

    setup.model._output_block.assert_called_once_with(
        setup.output,
        (
            4,
            2,
        ),
    )


def test_forward_uses_three_sample_dimensions_with_output_sampling():
    setup = configure_forward_model()

    setup.model.forward(
        make_forward_request(
            latent_sample_size=4,
            output_sample_size=5,
        )
    )

    setup.model._output_block.assert_called_once_with(
        setup.output,
        (
            5,
            4,
            2,
        ),
    )


def test_forward_adds_deterministic_guess_before_output_block():
    setup = configure_forward_model()

    deterministic = torch.full_like(
        setup.output,
        2.0,
    )
    setup.model._deterministic_guess = Mock(
        return_value=deterministic,
    )

    setup.model.forward(
        make_forward_request(
            latent_sample_size=4,
        )
    )

    passed = setup.model._output_block.call_args.args[0]

    torch.testing.assert_close(
        passed,
        setup.output + deterministic,
    )


def test_forward_passes_deterministic_guess_arguments():
    setup = configure_forward_model()

    setup.model._deterministic_guess = Mock(
        return_value=None,
    )

    condition = torch.ones(
        2,
        1,
        4,
        4,
    )
    condition_mask = torch.zeros_like(
        condition,
    )
    features = torch.ones(
        2,
        2,
        4,
        4,
    )

    setup.model.forward(
        make_forward_request(
            condition=condition,
            condition_mask=condition_mask,
            added_features=features,
        )
    )

    setup.model._deterministic_guess.assert_called_once_with(
        input=condition,
        input_mask=condition_mask,
        added_features=features,
    )


def test_forward_clamps_both_posterior_variance_limits():
    setup = configure_forward_model()

    log_var = torch.tensor(
        [
            [
                -10.0,
                0.0,
                10.0,
            ],
            [
                -5.0,
                1.0,
                5.0,
            ],
        ]
    )
    setup.model._recognition.return_value = (
        setup.mu,
        log_var,
    )

    setup.model.forward(
        make_forward_request(
            posterior_variance_limits=(
                torch.tensor(-2.0),
                torch.tensor(3.0),
            )
        )
    )

    passed = setup.model._sample.call_args.args[1]

    assert torch.all(passed >= -2)
    assert torch.all(passed <= 3)


def test_forward_variance_limits_use_mu_dtype():
    setup = configure_forward_model()

    setup.mu = setup.mu.double()
    setup.log_var = setup.log_var.double()

    setup.model._recognition.return_value = (
        setup.mu,
        setup.log_var,
    )

    setup.model.forward(
        make_forward_request(
            posterior_variance_limits=(
                torch.tensor(-2.0),
                torch.tensor(2.0),
            )
        )
    )

    passed = setup.model._sample.call_args.args[1]

    assert passed.dtype == torch.float64


def test_predict_uses_two_sample_dimensions_without_output_sampling():
    setup = configure_predict_model()

    samples = torch.ones(
        4,
        2,
        3,
    )

    setup.model.predict(
        make_predict_request(
            latent_samples=samples,
            latent_sample_size=4,
            output_sample_size=0,
        )
    )

    setup.model._output_block.assert_called_once_with(
        setup.output,
        (
            4,
            2,
        ),
    )


def test_predict_uses_three_sample_dimensions_with_output_sampling():
    setup = configure_predict_model()

    samples = torch.ones(
        4,
        2,
        3,
    )

    setup.model.predict(
        make_predict_request(
            latent_samples=samples,
            latent_sample_size=4,
            output_sample_size=5,
        )
    )

    setup.model._output_block.assert_called_once_with(
        setup.output,
        (
            5,
            4,
            2,
        ),
    )


def test_predict_adds_deterministic_guess_before_output_block():
    setup = configure_predict_model()

    samples = torch.ones(
        3,
        2,
        3,
    )
    deterministic = torch.full_like(
        setup.output,
        2.0,
    )

    setup.model._deterministic_guess = Mock(
        return_value=deterministic,
    )

    setup.model.predict(
        make_predict_request(
            latent_samples=samples,
        )
    )

    passed = setup.model._output_block.call_args.args[0]

    torch.testing.assert_close(
        passed,
        setup.output + deterministic,
    )


def test_predict_passes_deterministic_guess_arguments():
    setup = configure_predict_model()

    samples = torch.ones(
        3,
        2,
        3,
    )
    condition = torch.ones(
        2,
        1,
        4,
        4,
    )
    condition_mask = torch.zeros_like(
        condition,
    )
    features = torch.ones(
        2,
        2,
        4,
        4,
    )

    setup.model._deterministic_guess = Mock(
        return_value=None,
    )

    setup.model.predict(
        make_predict_request(
            condition=condition,
            condition_mask=condition_mask,
            added_features=features,
            latent_samples=samples,
        )
    )

    setup.model._deterministic_guess.assert_called_once_with(
        input=condition,
        input_mask=condition_mask,
        added_features=features,
    )


def test_deterministic_guess_returns_none_when_model_is_absent():
    model = make_bare_model()
    model.deterministic_guess = None

    result = model._deterministic_guess(
        input=torch.ones(
            2,
            1,
            4,
            4,
        )
    )

    assert result is None


def test_deterministic_guess_builds_request_and_calls_decoder(
    monkeypatch,
):
    model = make_bare_model()

    expected = torch.ones(
        2,
        1,
        4,
        4,
    )
    decoder = SimpleNamespace(
        forward_decoder=Mock(
            return_value=expected,
        )
    )
    model.deterministic_guess = decoder

    request = object()
    request_constructor = Mock(
        return_value=request,
    )

    monkeypatch.setattr(
        module,
        "DeterministicRequest",
        request_constructor,
    )

    input_tensor = torch.ones(
        2,
        1,
        4,
        4,
    )
    input_mask = torch.zeros_like(
        input_tensor,
    )
    features = torch.ones(
        2,
        2,
        4,
        4,
    )

    result = model._deterministic_guess(
        input=input_tensor,
        input_mask=input_mask,
        added_features=features,
    )

    request_constructor.assert_called_once_with(
        input_tensor,
        input_mask,
        features,
    )
    decoder.forward_decoder.assert_called_once_with(
        request,
    )
    assert result is expected


def test_generate_condition_embedding_is_repeated_for_each_latent_sample():
    model = make_bare_model(
        condemb_to_decoder=True,
    )

    latent = torch.zeros(
        3,
        2,
        4,
    )
    condition = torch.tensor(
        [
            [
                1.0,
                2.0,
            ],
            [
                3.0,
                4.0,
            ],
        ]
    )

    model.generation = Mock(
        return_value=torch.ones(
            6,
            1,
            4,
            4,
        )
    )

    model._generate(
        latent_samples=latent,
        condition_embedding=condition,
    )

    passed = model.generation.call_args.args[0]

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    torch.testing.assert_close(
        passed[:, -2:],
        expected,
    )


def test_recognition_down_block_arguments(
    monkeypatch,
):
    config = make_recognition_config(
        mask_pooling="fraction",
        mask_fraction_threshold=0.75,
    )

    monkeypatch.setattr(
        module,
        "build_conv_block",
        Mock(
            side_effect=[
                IdentityTensorMask(),
                FixedLatentModule(
                    torch.ones(2, 5),
                    torch.zeros(2, 5),
                ),
            ]
        ),
    )

    down_block = Mock(
        side_effect=[
            ShapeDownBlock((4, 4)),
            ShapeDownBlock((2, 2)),
        ]
    )
    monkeypatch.setattr(
        module,
        "DownBlock",
        down_block,
    )

    Recognition(
        input_channels=2,
        input_spatial_shape=(8, 8),
        channels=[4, 8, 16],
        latent_size=5,
        config=config,
    )

    assert down_block.call_args_list == [
        call(
            4,
            8,
            block_config=config.block_config,
            return_skip=False,
            mask_pooling="fraction",
            mask_fraction_threshold=0.75,
        ),
        call(
            8,
            16,
            block_config=config.block_config,
            return_skip=False,
            mask_pooling="fraction",
            mask_fraction_threshold=0.75,
        ),
    ]


def test_recognition_get_spatial_shapes_calls_blocks_in_order():
    recognition = object.__new__(
        Recognition,
    )
    nn.Module.__init__(
        recognition,
    )

    first = ShapeDownBlock(
        (5, 6),
    )
    second = ShapeDownBlock(
        (3, 3),
    )

    recognition.down_blocks = nn.ModuleList(
        [
            first,
            second,
        ]
    )

    result = recognition._get_spatial_shapes(
        (9, 11),
    )

    assert result == [
        (9, 11),
        (5, 6),
        (3, 3),
    ]


def test_generation_forwards_configuration_to_up_blocks(
    monkeypatch,
):
    up_block = Mock(
        side_effect=[
            RecordingUpBlock(),
            RecordingUpBlock(),
        ]
    )
    monkeypatch.setattr(
        module,
        "UpBlock",
        up_block,
    )

    config = make_generation_config(
        upsampling_method="transpose_conv",
        upsampling_alignment_method="strict",
        channels=[
            4,
            8,
            16,
        ],
        condition_embedding_channels=[
            4,
            8,
            16,
        ],
        transpose_kernel_sizes=[
            5,
            7,
        ],
    )

    Generation(
        latent_size=3,
        channels=[
            16,
            8,
            4,
        ],
        resize_shapes=[
            (2, 2),
            (4, 4),
            (8, 8),
        ],
        config=config,
    )

    assert up_block.call_args_list[0].kwargs["block_config"] is config.block_config
    assert up_block.call_args_list[0].kwargs["upsampling_method"] == "transpose_conv"
    assert up_block.call_args_list[0].kwargs["skip_alignment_method"] == "strict"
    assert up_block.call_args_list[0].kwargs["skip_channels"] is None

    assert up_block.call_args_list[1].kwargs["transpose_kernel_size"] == 7


def test_generation_forward_calls_up_blocks_in_sequence():
    generation = make_generation_without_constructor(
        resize_shapes=[
            (4, 4),
            (8, 8),
        ]
    )

    first = RecordingUpBlock()
    second = RecordingUpBlock()

    generation.up_blocks = nn.ModuleList(
        [
            first,
            second,
        ]
    )

    generation(
        torch.ones(
            3,
            4,
        )
    )

    first_output = first.calls[0]["value"]

    assert second.calls[0]["value"] is first_output
    assert first.calls[0]["resize_shape"] == (
        4,
        4,
    )
    assert second.calls[0]["resize_shape"] == (
        8,
        8,
    )


def test_generation_repeat_receives_initial_tensor_mask(
    monkeypatch,
):
    generation = make_generation_without_constructor(
        generator=make_generator_config(
            noise_level="full",
        )
    )

    captured = {}

    def repeat(
        value,
        repeats,
    ):
        captured["value"] = value
        captured["repeats"] = repeats

        return TensorMask(
            tensor=value.tensor.repeat_interleave(
                repeats,
                dim=0,
            ),
            mask=None,
        )

    monkeypatch.setattr(
        module,
        "_repeat_tensor_mask",
        repeat,
    )

    generation(
        torch.ones(
            3,
            4,
        ),
        num_output_samples=2,
    )

    assert isinstance(
        captured["value"],
        TensorMask,
    )
    assert captured["value"].tensor.shape == (
        3,
        2,
        2,
        2,
    )
    assert captured["value"].mask is None
    assert captured["repeats"] == 2

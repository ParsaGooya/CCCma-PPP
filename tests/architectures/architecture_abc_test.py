from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import S2D4P.architectures.architecture_abc as module
from S2D4P.architectures.architecture_abc import (
    CheckpointConfig,
    GENERATORConfig,
    cVAEArchConfigABC,
    cVAEArchABC,
    cVAEPredictRequest,
    deterministicArchABC,
    flowABC,
    ArchABC,
    ArchConfigABC,
    weights_init,
)
from S2D4P.generic.runtime import RuntimeContext


class ConcreteFlow(flowABC):
    def forward(self, x, condition=None):
        return x

    def inverse(self, z, condition=None):
        return SimpleNamespace(e_samples=z)


class ConcreteModelConfig(ArchConfigABC):
    activation = "relu"
    NUM_INPUT_DIMS = 2
    NUM_OUTPUT_DIMS = 2
    GENERATOR = None

    @property
    def EXPECTS_MASK(self):
        return False

    def build(self, input_shape, output_shape=None, added_features_dim=None, **kwargs):
        return {
            "input_shape": input_shape,
            "output_shape": output_shape,
            "added_features_dim": added_features_dim,
            **kwargs,
        }


class ConcreteModel(ArchABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generative_modeling = False
        self.linear = nn.Linear(2, 2)

    def forward(self, value=None):
        if value is None:
            return None
        return self.linear(value)


class ParameterlessModel(ArchABC):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(checkpoint_config=None)
        self.generative_modeling = False

    def forward(self):
        return None


class BufferedModel(ArchABC):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(checkpoint_config=None)
        self.generative_modeling = False
        self.register_buffer("stored", torch.ones(1))

    def forward(self):
        return self.stored


class NestedModel(ArchABC):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(checkpoint_config=None)
        self.generative_modeling = False
        self.first = nn.Linear(2, 3)
        self.container = nn.Sequential(nn.ReLU(), nn.Linear(3, 1))

    def forward(self, value):
        return self.container(self.first(value))


class ConcreteDeterministicModel(deterministicArchABC):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(2, 1)

    def forward(self, request):
        return self.linear(request.input)


class ConcreteCVAEConfig(cVAEArchConfigABC):
    activation = "relu"
    NUM_INPUT_DIMS = 2
    NUM_OUTPUT_DIMS = 2
    GENERATOR = None

    def __init__(
        self,
        *,
        latent_size=3,
        condition_embedding_size=3,
        condition_dependant_latent=False,
        condition_dependant_flow=False,
    ):
        self.latent_size = latent_size
        self.condition_embedding_size = condition_embedding_size
        self.condition_dependant_latent = condition_dependant_latent
        self.condition_embedding_dims = [4, 3]
        self.condemb_to_decoder = True
        self._resolve_flow_settings(condition_dependant_flow)

    @property
    def EXPECTS_MASK(self):
        return False

    def build(self, input_shape, output_shape=None, added_features_dim=None, **kwargs):
        return ConcreteCVAEModel(self)


class ConcreteCVAEModel(cVAEArchABC):
    def __init__(self, config):
        super().__init__(config)
        self.latent_size = config.latent_size
        self.condition_dependant_latent = config.condition_dependant_latent
        self.linear = nn.Linear(self.latent_size, self.latent_size)

    def forward(self, request):
        return None

    def predict(self, request):
        return None

    def _recognition(self, *args, **kwargs):
        return None

    def _condition(self, condition, condition_mask=None, added_features=None):
        """Return constant (mu, log_var) so tests can assert on the prior."""
        batch_size = condition.shape[0]
        kwargs = dict(dtype=condition.dtype, device=condition.device)

        if self.condition_dependant_latent and not self.condition_dependant_flow:
            return (
                torch.ones(batch_size, self.latent_size, **kwargs),
                torch.zeros(batch_size, self.latent_size, **kwargs),
            )

        return (torch.full((batch_size, self.latent_size), 2.0, **kwargs), None)

    def _generate(self, *args, **kwargs):
        return None


def make_predict_request(**overrides):
    values = {
        "condition": torch.ones(2, 1, dtype=torch.float32),
        "condition_mask": None,
        "added_features": None,
        "prior_flow": None,
        "latent_samples": None,
        "nstds": 1,
        "latent_sample_size": 4,
        "output_sample_size": 0,
    }
    values.update(overrides)
    return cVAEPredictRequest(**values)


def make_checkpoint(
    path,
    *,
    input_shape=(2, 3),
    output_shape=(1, 3),
    input_metadata=None,
    output_metadata=None,
):
    return CheckpointConfig(
        load_path=path,
        checkpoint_input_shape=np.array(input_shape),
        checkpoint_output_shape=np.array(output_shape),
        checkpoint_input_var_metadata=(
            {"tas": {}} if input_metadata is None else input_metadata
        ),
        checkpoint_output_var_metadata=(
            {"pr": {}} if output_metadata is None else output_metadata
        ),
    )


def make_load_config(path, *, strict=True, freeze_weights=False):
    return CheckpointConfig(
        load_path=path,
        checkpoint_input_shape=np.array([2]),
        checkpoint_output_shape=np.array([2]),
        checkpoint_input_var_metadata={},
        checkpoint_output_var_metadata={},
        strict=strict,
        freeze_weights=freeze_weights,
    )


@pytest.fixture
def matching_runtime_metadata(monkeypatch):
    """Make RuntimeContext metadata match the default checkpoint metadata."""
    monkeypatch.setattr(RuntimeContext, "INPUT_VAR_METADATA", {"tas": {}})
    monkeypatch.setattr(RuntimeContext, "TARGET_VAR_METADATA", {"pr": {}})


class TestGeneratorConfig:
    def test_default_values(self):
        config = GENERATORConfig()

        assert config.noise_level == "full"
        assert config.num_training_noise_samples == 10
        assert config.num_validation_noise_samples == 10

    def test_validation_samples_default_to_training_samples(self):
        config = GENERATORConfig(num_training_noise_samples=7)

        assert config.num_validation_noise_samples == 7

    def test_explicit_validation_samples_are_preserved(self):
        config = GENERATORConfig(
            num_training_noise_samples=7,
            num_validation_noise_samples=3,
        )

        assert config.num_validation_noise_samples == 3

    @pytest.mark.parametrize("noise_level", ["low", "medium", "full"])
    def test_noise_levels_are_stored(self, noise_level):
        config = GENERATORConfig(noise_level=noise_level)

        assert config.noise_level == noise_level


class TestCheckpointCompatibility:
    def _model_with_checkpoint(self, checkpoint):
        config = ConcreteModelConfig()
        config.checkpoint_config = checkpoint
        return ConcreteModel(config)

    def test_no_checkpoint_skips_validation(self):
        config = ConcreteModelConfig()
        config.checkpoint_config = None
        model = ConcreteModel(config)

        result = model._validate_checkpoint_compatibility(
            input_shape=np.array([999]),
            output_shape=np.array([888]),
        )

        assert result is None

    def test_matching_checkpoint_is_accepted(self, tmp_path, matching_runtime_metadata):
        model = self._model_with_checkpoint(make_checkpoint(tmp_path / "model.pt"))

        # Should not raise.
        model._validate_checkpoint_compatibility(
            input_shape=np.array([2, 3]),
            output_shape=np.array([1, 3]),
        )

    def test_input_shape_mismatch_raises(self, tmp_path):
        model = self._model_with_checkpoint(make_checkpoint(tmp_path / "model.pt"))

        with pytest.raises(RuntimeError, match="Requested input shape"):
            model._validate_checkpoint_compatibility(
                input_shape=np.array([5, 3]),
                output_shape=np.array([1, 3]),
            )

    def test_output_shape_mismatch_raises(self, tmp_path, matching_runtime_metadata):
        model = self._model_with_checkpoint(make_checkpoint(tmp_path / "model.pt"))

        with pytest.raises(RuntimeError, match="Requested output shape"):
            model._validate_checkpoint_compatibility(
                input_shape=np.array([2, 3]),
                output_shape=np.array([4, 3]),
            )

    def test_input_metadata_mismatch_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(RuntimeContext, "INPUT_VAR_METADATA", {"different": {}})
        monkeypatch.setattr(RuntimeContext, "TARGET_VAR_METADATA", {"pr": {}})
        model = self._model_with_checkpoint(make_checkpoint(tmp_path / "model.pt"))

        with pytest.raises(RuntimeError, match="Checkpoint input metadata"):
            model._validate_checkpoint_compatibility(
                input_shape=np.array([2, 3]),
                output_shape=np.array([1, 3]),
            )

    def test_output_metadata_mismatch_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(RuntimeContext, "INPUT_VAR_METADATA", {"tas": {}})
        monkeypatch.setattr(RuntimeContext, "TARGET_VAR_METADATA", {"different": {}})
        model = self._model_with_checkpoint(make_checkpoint(tmp_path / "model.pt"))

        with pytest.raises(RuntimeError, match="Checkpoint target metadata"):
            model._validate_checkpoint_compatibility(
                input_shape=np.array([2, 3]),
                output_shape=np.array([1, 3]),
            )


class TestInitializeWeights:
    def test_visits_model_and_all_nested_modules(self):
        model = NestedModel()

        with patch.object(module, "weights_init") as initializer:
            model._initialize_weights("xavier")

        visited = {id(call.args[0]) for call in initializer.call_args_list}

        assert id(model) in visited
        assert id(model.first) in visited
        assert id(model.container) in visited
        assert id(model.container[0]) in visited
        assert id(model.container[1]) in visited

    def test_passes_method_to_every_call(self):
        model = NestedModel()

        with patch.object(module, "weights_init") as initializer:
            model._initialize_weights("trunc_normal")

        assert all(
            call.kwargs["method"] == "trunc_normal"
            for call in initializer.call_args_list
        )

    def test_excluded_module_and_children_are_skipped(self):
        model = NestedModel()

        with patch.object(module, "weights_init") as initializer:
            model._initialize_weights("xavier", exclude=(model.container,))

        visited = {id(call.args[0]) for call in initializer.call_args_list}

        assert id(model.container) not in visited
        assert id(model.container[0]) not in visited
        assert id(model.container[1]) not in visited
        assert id(model.first) in visited


class TestGetDevice:
    def test_returns_parameter_device(self):
        model = ConcreteModel(ConcreteModelConfig())

        assert model._get_device() == model.linear.weight.device

    def test_returns_buffer_device_without_parameters(self):
        model = BufferedModel()

        assert model._get_device() == model.stored.device

    def test_defaults_to_cpu_without_parameters_or_buffers(self):
        model = ParameterlessModel()

        assert model._get_device() == torch.device("cpu")


class TestLoadStateDict:
    def test_missing_checkpoint_raises(self, tmp_path):
        model = ConcreteModel(ConcreteModelConfig())
        config = make_load_config(tmp_path / "missing.pt")

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            model._load_state_dict(config)

    def test_forwards_non_strict_flag(self, tmp_path):
        model = ConcreteModel(ConcreteModelConfig())
        path = tmp_path / "model.pt"
        path.touch()
        config = make_load_config(path, strict=False)

        with (
            patch.object(module.torch, "load", return_value={"module": {}}),
            patch.object(model, "load_state_dict") as load_state,
        ):
            model._load_state_dict(config)

        load_state.assert_called_once_with({}, strict=False)


class TestCVAEConfig:
    def test_flow_setting_defaults_to_false(self):
        config = ConcreteCVAEConfig()

        assert config.condition_dependant_flow is False

    def test_flow_setting_can_be_enabled(self):
        config = ConcreteCVAEConfig(condition_dependant_flow=True)

        assert config.condition_dependant_flow is True

    def test_resolve_flow_settings_returns_self(self):
        config = ConcreteCVAEConfig()

        result = config._resolve_flow_settings(True)

        assert result is config
        assert config.condition_dependant_flow is True

    def test_dependent_latent_without_flow_requires_matching_sizes(self):
        with pytest.raises(ValueError, match="must equal latent size"):
            ConcreteCVAEConfig(
                latent_size=3,
                condition_embedding_size=5,
                condition_dependant_latent=True,
                condition_dependant_flow=False,
            )

    def test_dependent_latent_with_flow_allows_different_sizes(self):
        config = ConcreteCVAEConfig(
            latent_size=3,
            condition_embedding_size=5,
            condition_dependant_latent=True,
            condition_dependant_flow=True,
        )

        assert config.latent_size == 3
        assert config.condition_embedding_size == 5

    def test_independent_latent_allows_different_sizes(self):
        config = ConcreteCVAEConfig(
            latent_size=3,
            condition_embedding_size=5,
            condition_dependant_latent=False,
            condition_dependant_flow=False,
        )

        assert config.condition_embedding_size == 5


class TestCVAEModelInitialization:
    def test_condition_flow_setting_is_copied_to_model(self):
        config = ConcreteCVAEConfig(condition_dependant_flow=True)

        model = ConcreteCVAEModel(config)

        assert model.condition_dependant_flow is True


class TestSample:
    def test_converts_log_variance_to_variance(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())
        mu = torch.zeros(2, 3)
        log_var = torch.log(torch.full((2, 3), 2.0))

        with patch.object(
            module, "_sample", return_value=torch.ones(4, 2, 3)
        ) as sample:
            result = model._sample(mu, log_var, sample_size=4, std=2)

        expected_variance = torch.exp(log_var) + 1e-4

        sample.assert_called_once()
        call = sample.call_args
        torch.testing.assert_close(call.args[0], mu)
        torch.testing.assert_close(call.args[1], expected_variance)
        assert call.args[2] == 4
        assert call.args[3] == 2
        assert result.shape == (4, 2, 3)


class TestSamplePrior:
    def test_dependent_latent_samples_condition_distribution(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig(condition_dependant_latent=True))
        request = make_predict_request(nstds=2)
        expected = torch.ones(4, 2, 3)

        with patch.object(model, "_sample", return_value=expected) as sample:
            latent, cond_mu, cond_log_var = model._sample_prior(request)

        sample.assert_called_once()
        call = sample.call_args
        assert call.args[2] == 4
        assert call.kwargs["std"] == 2
        assert latent is expected
        assert cond_mu.shape == (2, 3)
        assert cond_log_var.shape == (2, 3)

    def test_independent_latent_samples_standard_normal(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig(condition_dependant_latent=False))
        request = make_predict_request(latent_sample_size=5, nstds=3)

        distribution = Mock()
        distribution.sample.return_value = torch.zeros(5, 2, 3)

        with patch.object(
            module, "_get_normal", return_value=distribution
        ) as get_normal:
            latent, cond_mu, cond_log_var = model._sample_prior(request)

        get_normal.assert_called_once()
        reference = get_normal.call_args.args[0]
        assert reference.shape == (2, 3)
        assert get_normal.call_args.kwargs["std"] == 3
        distribution.sample.assert_called_once_with((5,))
        assert latent.shape == (5, 2, 3)
        assert cond_mu.shape == (2, 3)
        assert cond_log_var is None

    def test_condition_dependent_flow_uses_normal_prior(self):
        model = ConcreteCVAEModel(
            ConcreteCVAEConfig(
                condition_dependant_latent=True,
                condition_dependant_flow=True,
            )
        )
        request = make_predict_request()

        distribution = Mock()
        distribution.sample.return_value = torch.zeros(4, 2, 3)

        with (
            patch.object(module, "_get_normal", return_value=distribution),
            patch.object(model, "_sample") as sample,
        ):
            latent, _, _ = model._sample_prior(request)

        sample.assert_not_called()
        assert latent.shape == (4, 2, 3)

    def test_unconditional_prior_flow_is_applied(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())
        request = make_predict_request()

        distribution = Mock()
        distribution.sample.return_value = torch.zeros(4, 2, 3)

        transformed = torch.full((8, 3), 5.0)
        flow = Mock()
        flow.condition_size = None
        flow.inverse.return_value = SimpleNamespace(e_samples=transformed)
        request.prior_flow = flow

        with patch.object(module, "_get_normal", return_value=distribution):
            latent, _, _ = model._sample_prior(request)

        flow.inverse.assert_called_once()
        passed_latent = flow.inverse.call_args.args[0]
        passed_condition = flow.inverse.call_args.args[1]

        assert passed_latent.shape == (8, 3)
        assert passed_condition is None
        assert latent.shape == (4, 2, 3)
        assert torch.all(latent == 5)

    def test_conditional_prior_flow_receives_condition(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())
        request = make_predict_request()

        distribution = Mock()
        distribution.sample.return_value = torch.zeros(4, 2, 3)

        flow = Mock()
        flow.condition_size = 3
        flow.inverse.return_value = SimpleNamespace(e_samples=torch.zeros(8, 3))
        request.prior_flow = flow

        with patch.object(module, "_get_normal", return_value=distribution):
            model._sample_prior(request)

        passed_condition = flow.inverse.call_args.args[1]

        assert passed_condition.shape == (8, 3)
        assert torch.all(passed_condition == 2)

    def test_user_latent_samples_are_preserved(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())
        user_samples = torch.randn(4, 2, 3)
        request = make_predict_request(latent_samples=user_samples)

        latent, _, _ = model._sample_prior(request)

        assert latent is user_samples

    @pytest.mark.parametrize(
        "shape",
        [(2, 3), (4, 3, 3), (5, 2, 3), (4, 2, 4)],
    )
    def test_invalid_user_latent_shape_raises(self, shape):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())
        request = make_predict_request(latent_samples=torch.zeros(shape))

        with pytest.raises(ValueError, match="expected shape"):
            model._sample_prior(request)

    def test_user_samples_skip_prior_flow(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())
        flow = Mock()
        flow.condition_size = None
        request = make_predict_request(
            latent_samples=torch.zeros(4, 2, 3),
            prior_flow=flow,
        )

        model._sample_prior(request)

        flow.inverse.assert_not_called()

    def test_condition_receives_request_values(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())
        condition = torch.ones(2, 1)
        mask = torch.zeros_like(condition)
        features = torch.ones(2, 2)
        request = make_predict_request(
            condition=condition,
            condition_mask=mask,
            added_features=features,
        )

        with patch.object(
            model, "_condition", wraps=model._condition
        ) as condition_method:
            model._sample_prior(request)

        condition_method.assert_called_once_with(
            condition=condition,
            condition_mask=mask,
            added_features=features,
        )

    def test_prior_reference_matches_condition_dtype_and_device(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())
        condition = torch.ones(2, 1, dtype=torch.float64)
        request = make_predict_request(condition=condition)

        distribution = Mock()
        distribution.sample.return_value = torch.zeros(4, 2, 3, dtype=torch.float64)

        with patch.object(
            module, "_get_normal", return_value=distribution
        ) as get_normal:
            model._sample_prior(request)

        reference = get_normal.call_args.args[0]

        assert reference.dtype == torch.float64
        assert reference.device == condition.device


class TestWeightsInit:
    @pytest.mark.parametrize(
        "module_instance",
        [nn.ReLU(), nn.Identity(), nn.BatchNorm1d(3)],
    )
    def test_ignores_unsupported_modules(self, module_instance):
        with (
            patch.object(module.nn.init, "xavier_uniform_") as initializer,
            patch.object(module.nn.init, "constant_") as constant,
        ):
            weights_init(module_instance, method="xavier")

        initializer.assert_not_called()
        constant.assert_not_called()

    @pytest.mark.parametrize(
        "layer",
        [
            nn.Linear(3, 2),
            nn.Conv1d(2, 3, 3),
            nn.Conv2d(2, 3, 3),
            nn.Conv3d(2, 3, 3),
        ],
    )
    def test_xavier_initialization(self, layer):
        with (
            patch.object(module.nn.init, "xavier_uniform_") as initializer,
            patch.object(module.nn.init, "constant_") as constant,
        ):
            weights_init(layer, method="xavier")

        initializer.assert_called_once_with(layer.weight)
        constant.assert_not_called()

    @pytest.mark.parametrize("layer", [nn.Linear(3, 2), nn.Conv2d(2, 3, 3)])
    def test_truncated_normal_initialization(self, layer):
        with (
            patch.object(module, "trunc_normal_") as initializer,
            patch.object(module.nn.init, "constant_") as constant,
        ):
            weights_init(layer, method="trunc_normal")

        initializer.assert_called_once_with(layer.weight, std=0.02)
        constant.assert_not_called()

    def test_layer_without_bias_is_initialized(self):
        layer = nn.Linear(3, 2, bias=False)

        with (
            patch.object(module.nn.init, "xavier_uniform_") as initializer,
            patch.object(module.nn.init, "constant_") as constant,
        ):
            weights_init(layer, method="xavier")

        initializer.assert_called_once_with(layer.weight)
        constant.assert_not_called()

    def test_frozen_weight_is_not_initialized(self):
        layer = nn.Linear(3, 2)
        layer.weight.requires_grad_(False)

        with patch.object(module.nn.init, "xavier_uniform_") as initializer:
            weights_init(layer, method="xavier")

        initializer.assert_not_called()

    def test_frozen_bias_is_not_initialized(self):
        layer = nn.Linear(3, 2)
        layer.bias.requires_grad_(False)

        with patch.object(module.nn.init, "constant_") as constant:
            weights_init(layer, method="xavier")

        constant.assert_not_called()


class TestAdditionalBranchCoverage:
    @pytest.mark.parametrize("num_samples", [0, 1])
    def test_generator_config_rejects_too_few_validation_samples(
        self,
        num_samples,
    ):
        with pytest.raises(
            ValueError,
            match="at least two output ensemble samples",
        ):
            GENERATORConfig(
                num_validation_noise_samples=num_samples,
            )

    def test_add_checkpoint_config_stores_configuration(self, tmp_path):
        config = ConcreteModelConfig()
        checkpoint = make_checkpoint(tmp_path / "model.pt")

        result = config._add_checkpoint_config(checkpoint)

        assert result is None
        assert config.checkpoint_config is checkpoint

    def test_concrete_flow_forward_returns_input(self):
        flow = ConcreteFlow()
        value = torch.randn(2, 3)
        condition = torch.randn(2, 1)

        result = flow.forward(
            value,
            condition=condition,
        )

        assert result is value

    def test_concrete_flow_inverse_wraps_samples(self):
        flow = ConcreteFlow()
        latent = torch.randn(2, 3)

        result = flow.inverse(latent)

        assert result.e_samples is latent

    def test_model_config_build_forwards_all_arguments(self):
        config = ConcreteModelConfig()
        input_shape = np.asarray([2, 3])
        output_shape = np.asarray([1, 3])

        result = config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=4,
            custom_argument="value",
        )

        assert result["input_shape"] is input_shape
        assert result["output_shape"] is output_shape
        assert result["added_features_dim"] == 4
        assert result["custom_argument"] == "value"

    def test_deterministic_model_initialization_sets_expected_state(self):
        config = ConcreteModelConfig()

        model = ConcreteDeterministicModel(config)

        assert model.config is config
        assert model.generative_modeling is False

    def test_deterministic_model_forward_uses_request_input(self):
        from S2D4P.architectures.architecture_abc import DeterministicRequest

        model = ConcreteDeterministicModel(ConcreteModelConfig())
        request = DeterministicRequest(
            input=torch.ones(2, 2),
        )

        result = model(request)

        assert result.shape == (2, 1)

    def test_deterministic_request_defaults(self):
        from S2D4P.architectures.architecture_abc import DeterministicRequest

        input_tensor = torch.ones(2, 3)

        request = DeterministicRequest(input=input_tensor)

        assert request.input is input_tensor
        assert request.input_mask is None
        assert request.added_features is None
        assert request.output_sample_size == 1
        assert request.chunk_output_samples is False

    def test_cvae_forward_request_defaults(self):
        from S2D4P.architectures.architecture_abc import cVAEForwardRequest

        target = torch.ones(2, 1)
        condition = torch.ones(2, 1)

        request = cVAEForwardRequest(
            target=target,
            condition=condition,
        )

        assert request.target is target
        assert request.condition is condition
        assert request.target_mask is None
        assert request.condition_mask is None
        assert request.added_features is None
        assert request.latent_sample_size == 1
        assert request.output_sample_size == 1
        assert request.posterior_variance_limits is None

    def test_cvae_predict_request_defaults(self):
        condition = torch.ones(2, 1)

        request = cVAEPredictRequest(condition=condition)

        assert request.condition is condition
        assert request.condition_mask is None
        assert request.added_features is None
        assert request.prior_flow is None
        assert request.latent_samples is None
        assert request.nstds == 1
        assert request.latent_sample_size == 1
        assert request.output_sample_size == 0
        assert request.chunk_size is None

    def test_cvae_model_initialization_sets_generative_state(self):
        config = ConcreteCVAEConfig()

        model = ConcreteCVAEModel(config)

        assert model.config is config
        assert model.generative_modeling is True
        assert model.condition_dependant_flow is False

    def test_checkpoint_config_defaults(self, tmp_path):
        checkpoint = CheckpointConfig(
            load_path=tmp_path / "model.pt",
            checkpoint_input_shape=np.asarray([2]),
            checkpoint_output_shape=np.asarray([1]),
            checkpoint_input_var_metadata={},
            checkpoint_output_var_metadata={},
        )

        assert checkpoint.strict is True
        assert checkpoint.freeze_weights is False

    def test_load_state_dict_filters_non_architecture_keys(self, tmp_path):
        model = ConcreteModel(ConcreteModelConfig())
        path = tmp_path / "model.pt"
        path.touch()

        checkpoint = {
            "module": {
                "arch.linear.weight": torch.ones_like(model.linear.weight),
                "arch.linear.bias": torch.ones_like(model.linear.bias),
                "optimizer.state": torch.tensor(1),
                "unrelated": torch.tensor(2),
            }
        }

        config = make_load_config(path)

        with (
            patch.object(
                module.torch,
                "load",
                return_value=checkpoint,
            ) as torch_load,
            patch.object(model, "load_state_dict") as load_state,
        ):
            model._load_state_dict(config)

        torch_load.assert_called_once_with(
            path,
            map_location=model._get_device(),
            weights_only=True,
        )

        loaded_state = load_state.call_args.args[0]

        assert set(loaded_state) == {
            "linear.weight",
            "linear.bias",
        }
        load_state.assert_called_once_with(
            loaded_state,
            strict=True,
        )

    def test_load_state_dict_freezes_all_parameters(self, tmp_path):
        model = ConcreteModel(ConcreteModelConfig())
        path = tmp_path / "model.pt"
        path.touch()

        checkpoint = {
            "module": {
                "arch.linear.weight": model.linear.weight.detach().clone(),
                "arch.linear.bias": model.linear.bias.detach().clone(),
            }
        }

        config = make_load_config(
            path,
            freeze_weights=True,
        )

        with patch.object(
            module.torch,
            "load",
            return_value=checkpoint,
        ):
            model._load_state_dict(config)

        assert all(parameter.requires_grad is False for parameter in model.parameters())

    def test_load_state_dict_keeps_parameters_trainable_by_default(
        self,
        tmp_path,
    ):
        model = ConcreteModel(ConcreteModelConfig())
        path = tmp_path / "model.pt"
        path.touch()

        checkpoint = {
            "module": {
                "arch.linear.weight": model.linear.weight.detach().clone(),
                "arch.linear.bias": model.linear.bias.detach().clone(),
            }
        }

        config = make_load_config(
            path,
            freeze_weights=False,
        )

        with patch.object(
            module.torch,
            "load",
            return_value=checkpoint,
        ):
            model._load_state_dict(config)

        assert all(parameter.requires_grad is True for parameter in model.parameters())

    def test_load_state_dict_collects_garbage(self, tmp_path):
        model = ConcreteModel(ConcreteModelConfig())
        path = tmp_path / "model.pt"
        path.touch()
        config = make_load_config(path)

        checkpoint = {
            "module": {
                "arch.linear.weight": model.linear.weight.detach().clone(),
                "arch.linear.bias": model.linear.bias.detach().clone(),
            }
        }

        with (
            patch.object(
                module.torch,
                "load",
                return_value=checkpoint,
            ),
            patch.object(module.gc, "collect") as collect,
        ):
            model._load_state_dict(config)

        collect.assert_called_once_with()

    def test_sample_prior_passes_requested_standard_deviation(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())
        request = make_predict_request(nstds=2.5)
        distribution = Mock()
        distribution.sample.return_value = torch.zeros(4, 2, 3)

        with patch.object(
            module,
            "_get_normal",
            return_value=distribution,
        ) as get_normal:
            model._sample_prior(request)

        get_normal.assert_called_once()
        assert get_normal.call_args.kwargs["std"] == 2.5

    def test_conditional_flow_receives_repeated_condition_means(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())
        request = make_predict_request(
            latent_sample_size=3,
        )

        distribution = Mock()
        distribution.sample.return_value = torch.zeros(3, 2, 3)

        flow = Mock()
        flow.condition_size = 3
        flow.inverse.return_value = SimpleNamespace(
            e_samples=torch.zeros(6, 3),
        )
        request.prior_flow = flow

        with patch.object(
            module,
            "_get_normal",
            return_value=distribution,
        ):
            model._sample_prior(request)

        passed_condition = flow.inverse.call_args.args[1]

        assert passed_condition.shape == (6, 3)
        torch.testing.assert_close(
            passed_condition,
            torch.full((6, 3), 2.0),
        )

    def test_prior_flow_output_is_reshaped_to_sample_batch_layout(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())
        request = make_predict_request(
            latent_sample_size=3,
        )

        distribution = Mock()
        distribution.sample.return_value = torch.zeros(3, 2, 3)

        transformed = torch.arange(
            18,
            dtype=torch.float32,
        ).reshape(6, 3)

        flow = Mock()
        flow.condition_size = None
        flow.inverse.return_value = SimpleNamespace(
            e_samples=transformed,
        )
        request.prior_flow = flow

        with patch.object(
            module,
            "_get_normal",
            return_value=distribution,
        ):
            latent, _, _ = model._sample_prior(request)

        assert latent.shape == (3, 2, 3)
        torch.testing.assert_close(
            latent,
            transformed.reshape(3, 2, 3),
        )

    def test_weights_init_default_leaves_parameters_unchanged(self):
        layer = nn.Linear(3, 2)
        original_weight = layer.weight.detach().clone()
        original_bias = layer.bias.detach().clone()

        weights_init(
            layer,
            method="default",
        )

        torch.testing.assert_close(
            layer.weight,
            original_weight,
        )
        torch.testing.assert_close(
            layer.bias,
            original_bias,
        )

    def test_weights_init_xavier_zeros_trainable_bias(self):
        layer = nn.Linear(3, 2)

        with (
            patch.object(
                module.nn.init,
                "xavier_uniform_",
            ) as initializer,
            patch.object(
                module.nn.init,
                "zeros_",
            ) as zeros,
        ):
            weights_init(
                layer,
                method="xavier",
            )

        initializer.assert_called_once_with(layer.weight)
        zeros.assert_called_once_with(layer.bias)

    def test_weights_init_truncated_normal_zeros_trainable_bias(self):
        layer = nn.Linear(3, 2)

        with (
            patch.object(
                module,
                "trunc_normal_",
            ) as initializer,
            patch.object(
                module.nn.init,
                "zeros_",
            ) as zeros,
        ):
            weights_init(
                layer,
                method="trunc_normal",
            )

        initializer.assert_called_once_with(
            layer.weight,
            std=0.02,
        )
        zeros.assert_called_once_with(layer.bias)

    def test_weights_init_rejects_unknown_method(self):
        layer = nn.Linear(3, 2)

        with pytest.raises(
            NotImplementedError,
            match="is not implemented",
        ):
            weights_init(
                layer,
                method="unknown",
            )

    def test_weights_init_skips_frozen_weight_but_initializes_bias(self):
        layer = nn.Linear(3, 2)
        layer.weight.requires_grad_(False)

        with (
            patch.object(
                module.nn.init,
                "xavier_uniform_",
            ) as initializer,
            patch.object(
                module.nn.init,
                "zeros_",
            ) as zeros,
        ):
            weights_init(
                layer,
                method="xavier",
            )

        initializer.assert_not_called()
        zeros.assert_called_once_with(layer.bias)

    def test_weights_init_initializes_weight_but_skips_frozen_bias(self):
        layer = nn.Linear(3, 2)
        layer.bias.requires_grad_(False)

        with (
            patch.object(
                module.nn.init,
                "xavier_uniform_",
            ) as initializer,
            patch.object(
                module.nn.init,
                "zeros_",
            ) as zeros,
        ):
            weights_init(
                layer,
                method="xavier",
            )

        initializer.assert_called_once_with(layer.weight)
        zeros.assert_not_called()

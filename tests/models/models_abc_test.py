from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

import cccma_ppp.models.models_abc as module
from cccma_ppp.models.models_abc import (
    CheckpointConfig,
    DeterministicRequest,
    GENERATORConfig,
    cVAEForwardRequest,
    cVAEmodelConfigABC,
    cVAEmodelsABC,
    cVAEPredictRequest,
    deterministicmodelsABC,
    flowABC,
    modelABC,
    modelConfigABC,
    weights_init,
)
from cccma_ppp.generic.runtime import RuntimeContext


class ConcreteFlow(flowABC):
    def forward(
        self,
        x,
        condition=None,
    ):
        return x

    def inverse(
        self,
        z,
        condition=None,
    ):
        return SimpleNamespace(
            e_samples=z,
        )


class ConcreteModelConfig(modelConfigABC):
    activation = "relu"
    NUM_INPUT_DIMS = 2
    NUM_OUTPUT_DIMS = 2
    GENERATOR = None

    @property
    def EXPECTS_MASK(self):
        return False

    def build(
        self,
        input_shape,
        output_shape=None,
        added_features_dim=None,
        **kwargs,
    ):
        return {
            "input_shape": input_shape,
            "output_shape": output_shape,
            "added_features_dim": added_features_dim,
            **kwargs,
        }


class ConcreteModel(modelABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generative_modeling = False
        self.linear = nn.Linear(
            2,
            2,
        )

    def forward(self, value=None):
        if value is None:
            return None

        return self.linear(value)


class ParameterlessModel(modelABC):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            checkpoint_config=None,
        )
        self.generative_modeling = False

    def forward(self):
        return None


class BufferedModel(modelABC):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            checkpoint_config=None,
        )
        self.generative_modeling = False
        self.register_buffer(
            "stored",
            torch.ones(1),
        )

    def forward(self):
        return self.stored


class NestedModel(modelABC):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            checkpoint_config=None,
        )
        self.generative_modeling = False
        self.first = nn.Linear(
            2,
            3,
        )
        self.container = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                3,
                1,
            ),
        )

    def forward(self, value):
        return self.container(self.first(value))


class ConcreteDeterministicModel(deterministicmodelsABC):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(
            2,
            1,
        )

    def forward(
        self,
        request,
    ):
        return self.linear(request.input)


class ConcreteCVAEConfig(cVAEmodelConfigABC):
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
        self.condition_embedding_dims = [
            4,
            3,
        ]
        self.condemb_to_decoder = True
        self._resolve_flow_settings(condition_dependant_flow)

    @property
    def EXPECTS_MASK(self):
        return False

    def build(
        self,
        input_shape,
        output_shape=None,
        added_features_dim=None,
        **kwargs,
    ):
        return ConcreteCVAEModel(self)


class ConcreteCVAEModel(cVAEmodelsABC):
    def __init__(self, config):
        super().__init__(config)
        self.latent_size = config.latent_size
        self.condition_dependant_latent = config.condition_dependant_latent
        self.linear = nn.Linear(
            self.latent_size,
            self.latent_size,
        )

    def forward(
        self,
        request,
    ):
        return None

    def predict(
        self,
        request,
    ):
        return None

    def _recognition(
        self,
        *args,
        **kwargs,
    ):
        return None

    def _condition(
        self,
        condition,
        condition_mask=None,
        added_features=None,
    ):
        batch_size = condition.shape[0]

        if self.condition_dependant_latent and not self.condition_dependant_flow:
            return (
                torch.ones(
                    batch_size,
                    self.latent_size,
                    dtype=condition.dtype,
                    device=condition.device,
                ),
                torch.zeros(
                    batch_size,
                    self.latent_size,
                    dtype=condition.dtype,
                    device=condition.device,
                ),
            )

        return (
            torch.full(
                (
                    batch_size,
                    self.latent_size,
                ),
                2.0,
                dtype=condition.dtype,
                device=condition.device,
            ),
            None,
        )

    def _generate(
        self,
        *args,
        **kwargs,
    ):
        return None


class TestGeneratorConfig:
    def test_default_values(self):
        config = GENERATORConfig()

        assert config.noise_level == "full"
        assert config.num_training_noise_samples == 10
        assert config.num_validation_noise_samples == 10

    def test_validation_samples_default_to_training_samples(
        self,
    ):
        config = GENERATORConfig(
            num_training_noise_samples=7,
        )

        assert config.num_validation_noise_samples == 7

    def test_explicit_validation_samples_are_preserved(
        self,
    ):
        config = GENERATORConfig(
            num_training_noise_samples=7,
            num_validation_noise_samples=3,
        )

        assert config.num_validation_noise_samples == 3

    @pytest.mark.parametrize(
        "noise_level",
        [
            "low",
            "medium",
            "full",
        ],
    )
    def test_noise_levels_are_stored(
        self,
        noise_level,
    ):
        config = GENERATORConfig(
            noise_level=noise_level,
        )

        assert config.noise_level == noise_level


class TestCheckpointConfig:
    def test_initialization(self, tmp_path):
        load_path = tmp_path / "checkpoint.pt"

        config = CheckpointConfig(
            load_path=load_path,
            checkpoint_input_shape=np.array(
                [
                    2,
                    3,
                ]
            ),
            checkpoint_output_shape=np.array(
                [
                    1,
                    3,
                ]
            ),
            checkpoint_input_var_metadata={
                "tas": {},
            },
            checkpoint_output_var_metadata={
                "pr": {},
            },
        )

        assert config.load_path == load_path
        np.testing.assert_array_equal(
            config.checkpoint_input_shape,
            np.array(
                [
                    2,
                    3,
                ]
            ),
        )
        assert config.strict is True
        assert config.freeze_weights is False

    def test_custom_flags(self, tmp_path):
        config = CheckpointConfig(
            load_path=tmp_path / "model.pt",
            checkpoint_input_shape=np.array(
                [
                    2,
                ]
            ),
            checkpoint_output_shape=np.array(
                [
                    1,
                ]
            ),
            checkpoint_input_var_metadata={},
            checkpoint_output_var_metadata={},
            strict=False,
            freeze_weights=True,
        )

        assert config.strict is False
        assert config.freeze_weights is True


class TestFlowABC:
    def test_concrete_flow_forward(self):
        flow = ConcreteFlow()
        value = torch.ones(
            2,
            3,
        )

        assert flow.forward(value) is value

    def test_concrete_flow_inverse(self):
        flow = ConcreteFlow()
        value = torch.ones(
            2,
            3,
        )

        result = flow.inverse(value)

        assert result.e_samples is value

    def test_abstract_flow_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            flowABC()


class TestModelConfigABC:
    def test_subclass_receives_checkpoint_config(self):
        assert ConcreteModelConfig.checkpoint_config is None

    def test_add_checkpoint_config(self, tmp_path):
        config = ConcreteModelConfig()
        checkpoint = CheckpointConfig(
            load_path=tmp_path / "model.pt",
            checkpoint_input_shape=np.array(
                [
                    2,
                ]
            ),
            checkpoint_output_shape=np.array(
                [
                    1,
                ]
            ),
            checkpoint_input_var_metadata={},
            checkpoint_output_var_metadata={},
        )

        result = config._add_checkpoint_config(checkpoint)

        assert result is None
        assert config.checkpoint_config is checkpoint

    def test_concrete_build(self):
        config = ConcreteModelConfig()

        result = config.build(
            input_shape=np.array(
                [
                    2,
                    3,
                ]
            ),
            output_shape=np.array(
                [
                    1,
                    3,
                ]
            ),
            added_features_dim=2,
            custom=True,
        )

        np.testing.assert_array_equal(
            result["input_shape"],
            np.array(
                [
                    2,
                    3,
                ]
            ),
        )
        assert result["added_features_dim"] == 2
        assert result["custom"] is True

    def test_abstract_config_cannot_be_instantiated(
        self,
    ):
        with pytest.raises(TypeError):
            modelConfigABC()


class TestCheckpointCompatibility:
    def make_checkpoint(
        self,
        tmp_path,
        *,
        input_shape=(2, 3),
        output_shape=(1, 3),
        input_metadata=None,
        output_metadata=None,
    ):
        if input_metadata is None:
            input_metadata = {
                "tas": {},
            }

        if output_metadata is None:
            output_metadata = {
                "pr": {},
            }

        return CheckpointConfig(
            load_path=tmp_path / "model.pt",
            checkpoint_input_shape=np.array(input_shape),
            checkpoint_output_shape=np.array(output_shape),
            checkpoint_input_var_metadata=(input_metadata),
            checkpoint_output_var_metadata=(output_metadata),
        )

    def test_no_checkpoint_returns_without_validation(
        self,
    ):
        config = ConcreteModelConfig()
        config.checkpoint_config = None
        model = ConcreteModel(config)

        assert (
            model._validate_checkpoint_compatibility(
                input_shape=np.array(
                    [
                        999,
                    ]
                ),
                output_shape=np.array(
                    [
                        888,
                    ]
                ),
            )
            is None
        )

    def test_matching_checkpoint_is_accepted(
        self,
        tmp_path,
        monkeypatch,
    ):
        checkpoint = self.make_checkpoint(tmp_path)
        config = ConcreteModelConfig()
        config.checkpoint_config = checkpoint
        model = ConcreteModel(config)

        monkeypatch.setattr(
            RuntimeContext,
            "INPUT_VAR_METADATA",
            {
                "tas": {},
            },
        )
        monkeypatch.setattr(
            RuntimeContext,
            "TARGET_VAR_METADATA",
            {
                "pr": {},
            },
        )

        model._validate_checkpoint_compatibility(
            input_shape=np.array(
                [
                    2,
                    3,
                ]
            ),
            output_shape=np.array(
                [
                    1,
                    3,
                ]
            ),
        )

    def test_input_shape_mismatch(self, tmp_path):
        checkpoint = self.make_checkpoint(tmp_path)
        config = ConcreteModelConfig()
        config.checkpoint_config = checkpoint
        model = ConcreteModel(config)

        with pytest.raises(
            RuntimeError,
            match="Requested input shape",
        ):
            model._validate_checkpoint_compatibility(
                input_shape=np.array(
                    [
                        5,
                        3,
                    ]
                ),
                output_shape=np.array(
                    [
                        1,
                        3,
                    ]
                ),
            )

    def test_output_shape_mismatch(
        self,
        tmp_path,
        monkeypatch,
    ):
        checkpoint = self.make_checkpoint(tmp_path)
        config = ConcreteModelConfig()
        config.checkpoint_config = checkpoint
        model = ConcreteModel(config)

        monkeypatch.setattr(
            RuntimeContext,
            "INPUT_VAR_METADATA",
            {
                "tas": {},
            },
        )

        with pytest.raises(
            RuntimeError,
            match="Requested output shape",
        ):
            model._validate_checkpoint_compatibility(
                input_shape=np.array(
                    [
                        2,
                        3,
                    ]
                ),
                output_shape=np.array(
                    [
                        4,
                        3,
                    ]
                ),
            )

    def test_input_metadata_mismatch(
        self,
        tmp_path,
        monkeypatch,
    ):
        checkpoint = self.make_checkpoint(tmp_path)
        config = ConcreteModelConfig()
        config.checkpoint_config = checkpoint
        model = ConcreteModel(config)

        monkeypatch.setattr(
            RuntimeContext,
            "INPUT_VAR_METADATA",
            {
                "different": {},
            },
        )
        monkeypatch.setattr(
            RuntimeContext,
            "TARGET_VAR_METADATA",
            {
                "pr": {},
            },
        )

        with pytest.raises(
            RuntimeError,
            match="Checkpoint input metadata",
        ):
            model._validate_checkpoint_compatibility(
                input_shape=np.array(
                    [
                        2,
                        3,
                    ]
                ),
                output_shape=np.array(
                    [
                        1,
                        3,
                    ]
                ),
            )

    def test_output_metadata_mismatch(
        self,
        tmp_path,
        monkeypatch,
    ):
        checkpoint = self.make_checkpoint(tmp_path)
        config = ConcreteModelConfig()
        config.checkpoint_config = checkpoint
        model = ConcreteModel(config)

        monkeypatch.setattr(
            RuntimeContext,
            "INPUT_VAR_METADATA",
            {
                "tas": {},
            },
        )
        monkeypatch.setattr(
            RuntimeContext,
            "TARGET_VAR_METADATA",
            {
                "different": {},
            },
        )

        with pytest.raises(
            RuntimeError,
            match="Checkpoint target metadata",
        ):
            model._validate_checkpoint_compatibility(
                input_shape=np.array(
                    [
                        2,
                        3,
                    ]
                ),
                output_shape=np.array(
                    [
                        1,
                        3,
                    ]
                ),
            )


class TestInitializeWeights:
    def test_initialize_weights_visits_nested_modules(
        self,
    ):
        model = NestedModel()

        with patch.object(
            module,
            "weights_init",
        ) as initializer:
            model._initialize_weights("xavier")

        initialized_modules = {id(call.args[0]) for call in initializer.call_args_list}

        assert id(model) in initialized_modules
        assert id(model.first) in initialized_modules
        assert id(model.container) in initialized_modules
        assert id(model.container[0]) in initialized_modules
        assert id(model.container[1]) in initialized_modules

    def test_initialize_weights_passes_method(self):
        model = NestedModel()

        with patch.object(
            module,
            "weights_init",
        ) as initializer:
            model._initialize_weights("trunc_normal")

        assert all(
            call.kwargs["method"] == "trunc_normal"
            for call in initializer.call_args_list
        )

    def test_initialize_weights_excludes_module_and_children(
        self,
    ):
        model = NestedModel()

        with patch.object(
            module,
            "weights_init",
        ) as initializer:
            model._initialize_weights(
                "xavier",
                exclude=(model.container,),
            )

        initialized_modules = {id(call.args[0]) for call in initializer.call_args_list}

        assert id(model.container) not in initialized_modules
        assert id(model.container[0]) not in initialized_modules
        assert id(model.container[1]) not in initialized_modules
        assert id(model.first) in initialized_modules


class TestGetDevice:
    def test_returns_parameter_device(self):
        model = ConcreteModel(ConcreteModelConfig())

        assert model._get_device() == model.linear.weight.device

    def test_returns_buffer_device_without_parameters(
        self,
    ):
        model = BufferedModel()

        assert model._get_device() == model.stored.device

    def test_defaults_to_cpu(self):
        model = ParameterlessModel()

        assert model._get_device() == torch.device("cpu")


class TestLoadStateDict:
    def make_checkpoint_config(
        self,
        path,
        *,
        strict=True,
        freeze_weights=False,
    ):
        return CheckpointConfig(
            load_path=path,
            checkpoint_input_shape=np.array(
                [
                    2,
                ]
            ),
            checkpoint_output_shape=np.array(
                [
                    2,
                ]
            ),
            checkpoint_input_var_metadata={},
            checkpoint_output_var_metadata={},
            strict=strict,
            freeze_weights=freeze_weights,
        )

    def test_missing_checkpoint_raises(
        self,
        tmp_path,
    ):
        model = ConcreteModel(ConcreteModelConfig())
        config = self.make_checkpoint_config(tmp_path / "missing.pt")

        with pytest.raises(
            FileNotFoundError,
            match="Checkpoint not found",
        ):
            model._load_state_dict(config)

    def test_loads_only_model_prefixed_keys(
        self,
        tmp_path,
    ):
        model = ConcreteModel(ConcreteModelConfig())
        path = tmp_path / "model.pt"
        path.touch()

        checkpoint = {
            "module": {
                "model.linear.weight": (
                    torch.full_like(
                        model.linear.weight,
                        2.0,
                    )
                ),
                "model.linear.bias": (
                    torch.full_like(
                        model.linear.bias,
                        3.0,
                    )
                ),
                "optimizer.state": torch.tensor(1),
            }
        }

        config = self.make_checkpoint_config(path)

        with (
            patch.object(
                module.torch,
                "load",
                return_value=checkpoint,
            ) as load,
            patch.object(
                model,
                "load_state_dict",
            ) as load_state,
            patch.object(
                module.gc,
                "collect",
            ) as collect,
        ):
            model._load_state_dict(config)

        load.assert_called_once_with(
            path,
            map_location=model._get_device(),
            weights_only=True,
        )

        state = load_state.call_args.args[0]

        assert set(state) == {
            "linear.weight",
            "linear.bias",
        }
        load_state.assert_called_once_with(
            state,
            strict=True,
        )
        collect.assert_called_once()

    def test_forwards_non_strict_flag(
        self,
        tmp_path,
    ):
        model = ConcreteModel(ConcreteModelConfig())
        path = tmp_path / "model.pt"
        path.touch()

        config = self.make_checkpoint_config(
            path,
            strict=False,
        )

        with (
            patch.object(
                module.torch,
                "load",
                return_value={
                    "module": {},
                },
            ),
            patch.object(
                model,
                "load_state_dict",
            ) as load_state,
        ):
            model._load_state_dict(config)

        load_state.assert_called_once_with(
            {},
            strict=False,
        )

    def test_freezes_weights_when_requested(
        self,
        tmp_path,
    ):
        model = ConcreteModel(ConcreteModelConfig())
        path = tmp_path / "model.pt"
        path.touch()

        config = self.make_checkpoint_config(
            path,
            freeze_weights=True,
        )

        with (
            patch.object(
                module.torch,
                "load",
                return_value={
                    "module": {
                        "model.linear.weight": (model.linear.weight.detach().clone()),
                        "model.linear.bias": (model.linear.bias.detach().clone()),
                    },
                },
            ),
            patch.object(
                module.gc,
                "collect",
            ),
        ):
            model._load_state_dict(config)

        assert all(not parameter.requires_grad for parameter in model.parameters())

    def test_does_not_freeze_by_default(
        self,
        tmp_path,
    ):
        model = ConcreteModel(ConcreteModelConfig())
        path = tmp_path / "model.pt"
        path.touch()

        config = self.make_checkpoint_config(
            path,
            freeze_weights=False,
        )

        with (
            patch.object(
                module.torch,
                "load",
                return_value={
                    "module": {
                        "model.linear.weight": (model.linear.weight.detach().clone()),
                        "model.linear.bias": (model.linear.bias.detach().clone()),
                    },
                },
            ),
            patch.object(
                module.gc,
                "collect",
            ),
        ):
            model._load_state_dict(config)

        assert all(parameter.requires_grad for parameter in model.parameters())


class TestRequests:
    def test_deterministic_request_defaults(self):
        value = torch.ones(
            2,
            3,
        )

        request = DeterministicRequest(input=value)

        assert request.input is value
        assert request.input_mask is None
        assert request.added_features is None
        assert request.output_sample_size == 0

    def test_cvae_forward_request_defaults(self):
        target = torch.ones(
            2,
            1,
        )
        condition = torch.ones(
            2,
            1,
        )

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
        assert request.output_sample_size == 0
        assert request.posterior_variance_limits is None

    def test_cvae_forward_request_custom_values(self):
        target = torch.ones(
            2,
            1,
        )
        condition = torch.ones(
            2,
            1,
        )
        limits = [
            torch.tensor(-2.0),
            torch.tensor(2.0),
        ]

        request = cVAEForwardRequest(
            target=target,
            condition=condition,
            latent_sample_size=5,
            output_sample_size=4,
            posterior_variance_limits=limits,
        )

        assert request.latent_sample_size == 5
        assert request.output_sample_size == 4
        assert request.posterior_variance_limits is limits

    def test_cvae_predict_request_defaults(self):
        condition = torch.ones(
            2,
            1,
        )

        request = cVAEPredictRequest(condition=condition)

        assert request.condition is condition
        assert request.condition_mask is None
        assert request.added_features is None
        assert request.prior_flow is None
        assert request.latent_samples is None
        assert request.nstds == 1
        assert request.latent_sample_size == 1
        assert request.output_sample_size == 0


class TestDeterministicModelsABC:
    def test_initialization(self):
        config = ConcreteModelConfig()
        model = ConcreteDeterministicModel(config)

        assert model.config is config
        assert model.generative_modeling is False

    def test_forward(self):
        model = ConcreteDeterministicModel(ConcreteModelConfig())
        request = DeterministicRequest(
            input=torch.ones(
                2,
                2,
            )
        )

        output = model.forward(request)

        assert output.shape == (
            2,
            1,
        )


class TestCVAEConfig:
    def test_flow_setting_defaults_to_false(self):
        config = ConcreteCVAEConfig()

        assert config.condition_dependant_flow is False

    def test_flow_setting_can_be_enabled(self):
        config = ConcreteCVAEConfig(
            condition_dependant_flow=True,
        )

        assert config.condition_dependant_flow is True

    def test_resolve_flow_settings_returns_self(self):
        config = ConcreteCVAEConfig()

        result = config._resolve_flow_settings(True)

        assert result is config
        assert config.condition_dependant_flow is True

    def test_dependent_latent_without_flow_requires_matching_sizes(
        self,
    ):
        with pytest.raises(
            ValueError,
            match="must equal latent size",
        ):
            ConcreteCVAEConfig(
                latent_size=3,
                condition_embedding_size=5,
                condition_dependant_latent=True,
                condition_dependant_flow=False,
            )

    def test_dependent_latent_with_flow_allows_different_sizes(
        self,
    ):
        config = ConcreteCVAEConfig(
            latent_size=3,
            condition_embedding_size=5,
            condition_dependant_latent=True,
            condition_dependant_flow=True,
        )

        assert config.latent_size == 3
        assert config.condition_embedding_size == 5

    def test_independent_latent_allows_different_sizes(
        self,
    ):
        config = ConcreteCVAEConfig(
            latent_size=3,
            condition_embedding_size=5,
            condition_dependant_latent=False,
            condition_dependant_flow=False,
        )

        assert config.condition_embedding_size == 5


class TestCVAEModelInitialization:
    def test_initialization(self):
        config = ConcreteCVAEConfig()
        model = ConcreteCVAEModel(config)

        assert model.config is config
        assert model.generative_modeling is True
        assert model.condition_dependant_flow is False

    def test_condition_flow_setting_is_copied(
        self,
    ):
        config = ConcreteCVAEConfig(
            condition_dependant_flow=True,
        )
        model = ConcreteCVAEModel(config)

        assert model.condition_dependant_flow is True


class TestSample:
    def test_converts_log_variance_to_variance(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())

        mu = torch.zeros(
            2,
            3,
        )
        log_var = torch.log(
            torch.full(
                (
                    2,
                    3,
                ),
                2.0,
            )
        )

        with patch.object(
            module,
            "_sample",
            return_value=torch.ones(
                4,
                2,
                3,
            ),
        ) as sample:
            result = model._sample(
                mu,
                log_var,
                sample_size=4,
                std=2,
            )

        expected_variance = torch.exp(log_var) + 1e-4

        sample.assert_called_once()

        call = sample.call_args
        torch.testing.assert_close(
            call.args[0],
            mu,
        )
        torch.testing.assert_close(
            call.args[1],
            expected_variance,
        )
        assert call.args[2] == 4
        assert call.args[3] == 2
        assert result.shape == (
            4,
            2,
            3,
        )


class TestSamplePrior:
    def make_request(
        self,
        **overrides,
    ):
        values = {
            "condition": torch.ones(
                2,
                1,
                dtype=torch.float32,
            ),
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

    def test_dependent_latent_samples_condition_distribution(
        self,
    ):
        model = ConcreteCVAEModel(
            ConcreteCVAEConfig(
                condition_dependant_latent=True,
            )
        )
        request = self.make_request(
            nstds=2,
        )

        expected = torch.ones(
            4,
            2,
            3,
        )

        with patch.object(
            model,
            "_sample",
            return_value=expected,
        ) as sample:
            latent, cond_mu, cond_log_var = model._sample_prior(request)

        sample.assert_called_once()

        call = sample.call_args
        assert call.args[2] == 4
        assert call.kwargs["std"] == 2
        assert latent is expected
        assert cond_mu.shape == (
            2,
            3,
        )
        assert cond_log_var.shape == (
            2,
            3,
        )

    def test_independent_latent_samples_standard_normal(
        self,
    ):
        model = ConcreteCVAEModel(
            ConcreteCVAEConfig(
                condition_dependant_latent=False,
            )
        )
        request = self.make_request(
            latent_sample_size=5,
            nstds=3,
        )

        distribution = Mock()
        distribution.sample.return_value = torch.zeros(
            5,
            2,
            3,
        )

        with patch.object(
            module,
            "_get_normal",
            return_value=distribution,
        ) as get_normal:
            latent, cond_mu, cond_log_var = model._sample_prior(request)

        get_normal.assert_called_once()
        reference = get_normal.call_args.args[0]

        assert reference.shape == (
            2,
            3,
        )
        assert get_normal.call_args.kwargs["std"] == 3
        distribution.sample.assert_called_once_with((5,))
        assert latent.shape == (
            5,
            2,
            3,
        )
        assert cond_mu.shape == (
            2,
            3,
        )
        assert cond_log_var is None

    def test_condition_dependent_flow_uses_normal_prior(
        self,
    ):
        model = ConcreteCVAEModel(
            ConcreteCVAEConfig(
                condition_dependant_latent=True,
                condition_dependant_flow=True,
            )
        )
        request = self.make_request()

        distribution = Mock()
        distribution.sample.return_value = torch.zeros(
            4,
            2,
            3,
        )

        with (
            patch.object(
                module,
                "_get_normal",
                return_value=distribution,
            ),
            patch.object(
                model,
                "_sample",
            ) as sample,
        ):
            latent, _, _ = model._sample_prior(request)

        sample.assert_not_called()
        assert latent.shape == (
            4,
            2,
            3,
        )

    def test_unconditional_prior_flow(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())

        request = self.make_request()

        distribution = Mock()
        distribution.sample.return_value = torch.zeros(
            4,
            2,
            3,
        )

        transformed = torch.full(
            (
                8,
                3,
            ),
            5.0,
        )
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

        flow.inverse.assert_called_once()

        passed_latent = flow.inverse.call_args.args[0]
        passed_condition = flow.inverse.call_args.args[1]

        assert passed_latent.shape == (
            8,
            3,
        )
        assert passed_condition is None
        assert latent.shape == (
            4,
            2,
            3,
        )
        assert torch.all(latent == 5)

    def test_conditional_prior_flow(self):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())

        request = self.make_request()

        distribution = Mock()
        distribution.sample.return_value = torch.zeros(
            4,
            2,
            3,
        )

        flow = Mock()
        flow.condition_size = 3
        flow.inverse.return_value = SimpleNamespace(
            e_samples=torch.zeros(
                8,
                3,
            )
        )
        request.prior_flow = flow

        with patch.object(
            module,
            "_get_normal",
            return_value=distribution,
        ):
            model._sample_prior(request)

        passed_condition = flow.inverse.call_args.args[1]

        assert passed_condition.shape == (
            8,
            3,
        )
        assert torch.all(passed_condition == 2)

    def test_user_latent_samples_are_preserved(
        self,
    ):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())

        user_samples = torch.randn(
            4,
            2,
            3,
        )
        request = self.make_request(
            latent_samples=user_samples,
        )

        latent, _, _ = model._sample_prior(request)

        assert latent is user_samples

    @pytest.mark.parametrize(
        "shape",
        [
            (
                2,
                3,
            ),
            (
                4,
                3,
                3,
            ),
            (
                5,
                2,
                3,
            ),
            (
                4,
                2,
                4,
            ),
        ],
    )
    def test_invalid_user_latent_shape(
        self,
        shape,
    ):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())

        request = self.make_request(
            latent_samples=torch.zeros(shape),
        )

        with pytest.raises(
            ValueError,
            match="expected shape",
        ):
            model._sample_prior(request)

    def test_user_samples_skip_prior_flow(
        self,
    ):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())

        flow = Mock()
        flow.condition_size = None

        request = self.make_request(
            latent_samples=torch.zeros(
                4,
                2,
                3,
            ),
            prior_flow=flow,
        )

        model._sample_prior(request)

        flow.inverse.assert_not_called()

    def test_condition_receives_request_values(
        self,
    ):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())

        condition = torch.ones(
            2,
            1,
        )
        mask = torch.zeros_like(condition)
        features = torch.ones(
            2,
            2,
        )

        request = self.make_request(
            condition=condition,
            condition_mask=mask,
            added_features=features,
        )

        with patch.object(
            model,
            "_condition",
            wraps=model._condition,
        ) as condition_method:
            model._sample_prior(request)

        condition_method.assert_called_once_with(
            condition=condition,
            condition_mask=mask,
            added_features=features,
        )

    def test_prior_reference_matches_condition_dtype_and_device(
        self,
    ):
        model = ConcreteCVAEModel(ConcreteCVAEConfig())

        condition = torch.ones(
            2,
            1,
            dtype=torch.float64,
        )
        request = self.make_request(
            condition=condition,
        )

        distribution = Mock()
        distribution.sample.return_value = torch.zeros(
            4,
            2,
            3,
            dtype=torch.float64,
        )

        with patch.object(
            module,
            "_get_normal",
            return_value=distribution,
        ) as get_normal:
            model._sample_prior(request)

        reference = get_normal.call_args.args[0]

        assert reference.dtype == torch.float64
        assert reference.device == condition.device


class TestWeightsInit:
    @pytest.mark.parametrize(
        "module_instance",
        [
            nn.ReLU(),
            nn.Identity(),
            nn.BatchNorm1d(3),
        ],
    )
    def test_ignores_unsupported_modules(
        self,
        module_instance,
    ):
        with (
            patch.object(
                module.nn.init,
                "xavier_uniform_",
            ) as initializer,
            patch.object(
                module.nn.init,
                "constant_",
            ) as constant,
        ):
            weights_init(
                module_instance,
                method="xavier",
            )

        initializer.assert_not_called()
        constant.assert_not_called()

    @pytest.mark.parametrize(
        "layer",
        [
            nn.Linear(
                3,
                2,
            ),
            nn.Conv1d(
                2,
                3,
                3,
            ),
            nn.Conv2d(
                2,
                3,
                3,
            ),
            nn.Conv3d(
                2,
                3,
                3,
            ),
        ],
    )
    def test_xavier_initialization(
        self,
        layer,
    ):
        with (
            patch.object(
                module.nn.init,
                "xavier_uniform_",
            ) as initializer,
            patch.object(
                module.nn.init,
                "constant_",
            ) as constant,
        ):
            weights_init(
                layer,
                method="xavier",
            )

        initializer.assert_called_once_with(layer.weight)
        constant.assert_called_once_with(
            layer.bias,
            0,
        )

    @pytest.mark.parametrize(
        "layer",
        [
            nn.Linear(
                3,
                2,
            ),
            nn.Conv2d(
                2,
                3,
                3,
            ),
        ],
    )
    def test_truncated_normal_initialization(
        self,
        layer,
    ):
        with (
            patch.object(
                module,
                "trunc_normal_",
            ) as initializer,
            patch.object(
                module.nn.init,
                "constant_",
            ) as constant,
        ):
            weights_init(
                layer,
                method="trunc_normal",
            )

        initializer.assert_called_once_with(
            layer.weight,
            std=0.02,
        )
        constant.assert_called_once_with(
            layer.bias,
            0,
        )

    def test_layer_without_bias(self):
        layer = nn.Linear(
            3,
            2,
            bias=False,
        )

        with (
            patch.object(
                module.nn.init,
                "xavier_uniform_",
            ) as initializer,
            patch.object(
                module.nn.init,
                "constant_",
            ) as constant,
        ):
            weights_init(
                layer,
                method="xavier",
            )

        initializer.assert_called_once_with(layer.weight)
        constant.assert_not_called()

    def test_frozen_weight_is_not_initialized(
        self,
    ):
        layer = nn.Linear(
            3,
            2,
        )
        layer.weight.requires_grad_(False)

        with patch.object(
            module.nn.init,
            "xavier_uniform_",
        ) as initializer:
            weights_init(
                layer,
                method="xavier",
            )

        initializer.assert_not_called()

    def test_frozen_bias_is_not_initialized(
        self,
    ):
        layer = nn.Linear(
            3,
            2,
        )
        layer.bias.requires_grad_(False)

        with patch.object(
            module.nn.init,
            "constant_",
        ) as constant:
            weights_init(
                layer,
                method="xavier",
            )

        constant.assert_not_called()

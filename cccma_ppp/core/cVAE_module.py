import torch
import numpy as np
from pathlib import Path
import warnings
import dataclasses
import dacite
import gc

from cccma_ppp.loss.loss import Losspipeline
from cccma_ppp.loss.kld import KLD
from cccma_ppp.core.core_abc import moduleABC, moduleConfigABC, OutputABC
from cccma_ppp.core.selectors import (
    ModuleSelector,
    cVAEModelSelector,
    _load_config_from_checkpoint,
)
from cccma_ppp.models.normalized_flows import NormalizedFlowConfig
from cccma_ppp.models.models_abc import cVAEPredictRequest, cVAEForwardRequest
from cccma_ppp.train.dataloader import BatchData
from cccma_ppp.generic.runtime import RuntimeContext


@dataclasses.dataclass
class cVAEOutput(OutputABC):
    """
    Container for cVAE outputs.

    Parameters
    ----------
    output : torch.Tensor
        Generated samples.
    mu : torch.Tensor or None
        Posterior mean.
    log_var : torch.Tensor or None
        Posterior log-variance.
    cond_mu : torch.Tensor or None, optional
        Conditional prior mean.
    cond_log_var : torch.Tensor or None, optional
        Conditional prior log-variance.
    """

    output: torch.Tensor
    mu: torch.Tensor | None
    log_var: torch.Tensor | None
    samples: torch.Tensor | None = None
    cond_mu: torch.Tensor | None = None
    cond_log_var: torch.Tensor | None = None


@ModuleSelector.register("cvae")
@dataclasses.dataclass
class cVAEConfig(moduleConfigABC):
    """
    Configuration for conditional variational autoencoder (cVAE).

    Parameters
    ----------
    ModelConfig : cVAEModelSelector or None
        Model configuration selector.
    min_posterior_variance : float or None
        Minimum allowed posterior variance.
    prior_flow_config : NormalizedFlowConfig or None
        Configuration for optional prior flow.
    combined_CGCN_weight : float or None
        Weight for auxiliary CGCN loss.
    load_dir : str or None
        Path to checkpoint for loading configuration.
    """

    ModelConfig: cVAEModelSelector | None = None
    min_posterior_variance: float | None = None
    prior_flow_config: NormalizedFlowConfig | None = None
    combined_CGCN_weight: float = None
    load_dir: str | None = None

    def __post_init__(self):
        """
        Validate and initialize configuration.

        Raises
        ------
        ValueError
            If neither `ModelConfig` nor `load_dir` is provided.
        AssertionError
            If `combined_CGCN_weight` is not in [0, 1].
        """

        if self.load_dir is None:
            if self.ModelConfig is None:
                raise ValueError("provide loading dir or model configurations")

        else:
            self._load_from_checkpoint(self.load_dir)
            warnings.warn(
                f"Model and prior flow config overwritten by the saved module: \n {self.load_dir}"
            )

        if self.combined_CGCN_weight is None:
            self.combined_CGCN_weight = 0

        self.model_config = self.ModelConfig.get_model_config()

        self.latent_size = self.model_config.latent_size

        if self.prior_flow_config is not None:
            self.condition_dependant_flow = self.model_config.condition_dependant_latent

            if self.condition_dependant_flow:
                self.model_config._resolve_flow_settings(self.condition_dependant_flow)

        assert 0 <= self.combined_CGCN_weight <= 1, (
            "CGCN weight should be between [0,1]"
        )

    def build(
        self,
        input_shape: np.ndarray | tuple,
        output_shape: np.ndarray | tuple | None = None,
        added_features_dim: int = None,
    ):
        """
        Construct cVAE module instance.

        Parameters
        ----------
        input_shape : np.ndarray
            Input tensor shape.
        output_shape : np.ndarray or None, optional
            Output tensor shape.
        added_features_dim : int, optional
            Additional feature dimension.

        Returns
        -------
        cVAE
            Initialized cVAE module.
        """

        return cVAE(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

    def _load_from_checkpoint(self, load_path: Path | str):
        """
        Load configuration from checkpoint.

        Parameters
        ----------
        load_path : pathlib.Path or str
            Path to checkpoint.

        Returns
        -------
        cVAEConfig
            Updated configuration.
        """

        checkpoint_module, checkpoint_config = _load_config_from_checkpoint(
            Path(load_path)
        )
        self.checkpoint_config = checkpoint_config
        self.ModelConfig = dacite.from_dict(
            data_class=cVAEModelSelector,
            data=checkpoint_module.get("ModelConfig"),
            config=dacite.Config(strict=True),
        )

        self.prior_flow_config = checkpoint_module.get("prior_flow_config", None)
        self.prior_flow_config = dacite.from_dict(
            data_class=NormalizedFlowConfig,
            data=self.prior_flow_config,
            config=dacite.Config(strict=True),
        )

        if self.min_posterior_variance is None:
            self.min_posterior_variance = checkpoint_module.get(
                "min_posterior_variance", None
            )
        if self.combined_CGCN_weight is None:
            self.combined_CGCN_weight = checkpoint_module.get("combined_CGCN_weight", 0)

        del checkpoint_config, checkpoint_module
        gc.collect()
        return self


class cVAE(moduleABC):
    """
    Conditional variational autoencoder module.

    Parameters
    ----------
    config : cVAEConfig
        Configuration object.
    input_shape : np.ndarray
        Shape of input data.
    output_shape : np.ndarray or None
        Shape of output data.
    added_features_dim : int or None
        Additional feature dimension.
    """

    def __init__(
        self,
        config: cVAEConfig,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Initialize cVAE module and underlying model.

        Parameters
        ----------
        config : cVAEConfig
            Configuration instance.
        input_shape : np.ndarray
            Input shape.
        output_shape : np.ndarray or None
            Output shape.
        added_features_dim : int or None
            Additional feature dimension.

        Raises
        ------
        AssertionError
            If `min_posterior_variance` is non-positive.
        RuntimeError
            If checkpoint metadata or shapes are inconsistent.
        """

        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.latent_size = self.config.latent_size
        self.min_posterior_variance = self.config.min_posterior_variance
        self.prior_flow_config = self.config.prior_flow_config
        self.combined_CGCN_weight = self.config.combined_CGCN_weight

        if self.min_posterior_variance is not None:
            assert self.min_posterior_variance > 0, (
                "min_posterior_variance must be positive."
            )
        if getattr(self.config, "condition_dependant_flow", False):
            self.flow_condition_size = self.model_config.condition_embedding_size
        else:
            self.flow_condition_size = None

        self.criterion = None
        self.prior_flow = None

        if output_shape is None:
            output_shape = input_shape.copy()

        if self.config.load_dir is not None:
            if (
                RuntimeContext.INPUT_VAR_METADATA
                != self.config.checkpoint_config.checkpoint_input_var_metadata
            ):
                raise RuntimeError(
                    "the loaded module was not trained for the consistent input variables or preprocessing steps"
                )
            if (
                RuntimeContext.TARGET_VAR_METADATA
                != self.config.checkpoint_config.checkpoint_output_var_metadata
            ):
                raise RuntimeError(
                    "the loaded module was not trained for the consistent output variables or preprocessing steps"
                )

            if input_shape != self.config.checkpoint_config.checkpoint_input_shape:
                raise RuntimeError(
                    f"the requested input shape ({input_shape}) does not match the loaded module : {self.config.checkpoint_config.checkpoint_input_shape}"
                )
            if output_shape != self.config.checkpoint_config.checkpoint_output_shape:
                raise RuntimeError(
                    f"the requested output shape ({output_shape}) does not match the loaded module : {self.config.checkpoint_config.checkpoint_output_shape}"
                )

        if self.min_posterior_variance is not None:
            self.min_posterior_variance = torch.log(
                torch.tensor(self.min_posterior_variance)
            )

        self.model = self.model_config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

        if self.prior_flow_config is not None:
            self.prior_flow = self.prior_flow_config.build(
                latent_size=self.latent_size, condition_size=self.flow_condition_size
            )

        if self.config.load_dir is not None:
            self._load_state_dict(self.config.load_dir)

    def init_loss_function(self, reconstruction_loss: Losspipeline):
        """
        Initialize reconstruction and KL divergence losses.

        Parameters
        ----------
        reconstruction_loss : Losspipeline
            Loss pipeline instance.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If flow is used and reduction is not 'sum'.
        """

        if self.prior_flow_config is not None:
            if reconstruction_loss.reduction.lower() != "sum":
                raise RuntimeError(
                    "with normalized flow all loss reduction has to be sum."
                )

        self.criterion = reconstruction_loss.to(self._get_device())
        self.KLD = KLD(reduction=self.criterion.reduction).to(self._get_device())

    def _compute_loss(self, beta: float, data: BatchData):
        """
        Compute total training loss.

        Parameters
        ----------
        beta : float
            Weight applied to KL divergence term.
        data : BatchData
            Input batch.

        Returns
        -------
        tuple
            Total loss tensor and dictionary of loss terms.

        Raises
        ------
        RuntimeError
            If loss function is not initialized.
        """

        if self.criterion is None:
            raise RuntimeError(
                "crieterion should be specified before training is possible. Hint: call .init_loss_function() method in your module first."
            )

        output = self.forward(data)

        target = data.target
        target_mask = data.target_mask

        if target_mask is not None and target_mask.shape == target.shape:
            target_mask = target_mask.unsqueeze(0).expand_as(output.output)

        target = target.unsqueeze(0).expand_as(output.output)

        reconstruction_loss, indiv_losses = self.criterion(
            output.output,
            target,
            target_mask=target_mask,
            print_loss=False,
        )

        kld_loss = self.KLD(
            output.mu,
            output.log_var,
            output.cond_mu,
            output.cond_log_var,
            prior_flow=self.prior_flow,
            print_loss=False,
        )

        total_loss = reconstruction_loss + beta * kld_loss

        if self.combined_CGCN_weight > 0:
            output_CGCN = self.predict(data)
            reconstruction_loss_CGCN, _ = self.criterion(
                output_CGCN.output,
                target,
                print_loss=False,
            )

            total_loss = (
                total_loss * (1 - self.combined_CGCN_weight)
                + self.combined_CGCN_weight * reconstruction_loss_CGCN
            )
            indiv_losses["total_loss_CGCN"] = reconstruction_loss_CGCN.item()

        losses_dict = {"total_loss": total_loss.item(), "kld": kld_loss.item()}

        for key, value in indiv_losses.items():
            losses_dict[key] = value

        return total_loss, losses_dict

    def forward(self, data: BatchData, sample_size=1) -> cVAEOutput:
        """
        Perform forward pass.

        Parameters
        ----------
        data : BatchData
            Input batch.
        sample_size : int, optional
            Number of latent samples.

        Returns
        -------
        cVAEOutput
            Model outputs.
        """

        generator = self.model_config.GENERATOR
        output_sample_size = None
        if not self.training and generator is not None:
            output_sample_size = generator.num_validation_noise_samples

        return self.model(
            cVAEForwardRequest(
                target=data.target,
                target_mask=data.target_mask,
                added_features=data.added_features,
                condition=data.input,
                condition_mask=data.input_mask,
                min_posterior_variance=self.min_posterior_variance,
                sample_size=sample_size,
                output_sample_size=output_sample_size,
            )
        )

    def predict(
        self,
        data: BatchData,
        sample_size: int = 1,
        nstds: int = 1,
        latent_samples: torch.Tensor = None,
        output_sample_size: int = 1,
    ) -> cVAEOutput:
        """
        Generate predictions using the learned prior.

        Parameters
        ----------
        data : BatchData
            Input batch.
        sample_size : int, optional
            Number of samples to generate.

        Returns
        -------
        cVAEOutput
            Generated outputs.
        """
        generator = self.model_config.GENERATOR
        if self.training and generator is not None:
            output_sample_size = generator.num_training_noise_samples

        return self.model.predict(
            cVAEPredictRequest(
                condition=data.input,
                condition_mask=data.input_mask,
                added_features=data.added_features,
                prior_flow=self.prior_flow,
                sample_size=sample_size,
                nstds=nstds,
                latent_samples=latent_samples,
                output_sample_size=output_sample_size,
            )
        )

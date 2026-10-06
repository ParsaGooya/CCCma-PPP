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
from cccma_ppp.architectures.normalized_flows import NormalizedFlowConfig
from cccma_ppp.architectures.models_abc import cVAEPredictRequest, cVAEForwardRequest
from cccma_ppp.train.dataloader import BatchData
from cccma_ppp.generic.runtime import RuntimeContext


@dataclasses.dataclass
class cVAEOutput(OutputABC):
    """
    Container for outputs produced by a conditional variational autoencoder.

    Parameters
    ----------
    output : torch.Tensor
        Generated or reconstructed model output.
    mu : torch.Tensor | None
        Mean of the posterior latent distribution.
    log_var : torch.Tensor | None
        Log-variance of the posterior latent distribution.
    samples : torch.Tensor | None
        Latent samples used to generate the output.
    cond_mu : torch.Tensor | None
        Mean or embedding of the conditional latent representation.
    cond_log_var : torch.Tensor | None
        Log-variance of the conditional latent distribution when applicable.
    deterministic_guess : bool
        Whether the output represents only the deterministic guess branch.
    """

    output: torch.Tensor
    mu: torch.Tensor | None
    log_var: torch.Tensor | None
    samples: torch.Tensor | None = None
    cond_mu: torch.Tensor | None = None
    cond_log_var: torch.Tensor | None = None
    deterministic_guess: bool = False


@ModuleSelector.register("cvae")
@dataclasses.dataclass
class cVAEConfig(moduleConfigABC):
    """
    Configuration for a conditional variational autoencoder module.

    Parameters
    ----------
    ModelConfig : cVAEModelSelector | None
        Selector defining the underlying cVAE architecture. Required when a
        checkpoint directory is not provided.
    posterior_variance_limits : list[float | None, float | None] | tuple[float | None, float | None] | None
        Optional lower and upper bounds for posterior variance. Each bound may be
        ``None`` to leave that side unbounded.
    prior_flow_config : NormalizedFlowConfig | None
        Optional normalizing flow configuration used to transform the latent
        prior.
    combined_CGCN_weight : float | None
        Weight assigned to the prediction reconstruction loss when combining it
        with the standard cVAE training objective.
    load_dir : str | None
        Path to a saved module from which model and prior-flow configurations are
        restored.
    """

    ModelConfig: cVAEModelSelector | None = None
    posterior_variance_limits: (
        list[float | None, float | None] | tuple[float | None, float | None] | None
    ) = None
    prior_flow_config: NormalizedFlowConfig | None = None
    combined_CGCN_weight: float = None
    load_dir: str | None = None

    def __post_init__(self):
        """
        Resolve and validate the cVAE module configuration.

        When ``load_dir`` is provided, saved model and prior-flow configurations are
        restored from the checkpoint. Flow settings are then propagated to the model
        configuration.

        Raises
        ------
        AssertionError
            If ``combined_CGCN_weight`` is outside the interval [0, 1].
        ValueError
            If neither a model configuration nor a checkpoint directory is provided.

        Warns
        -----
        UserWarning
            If model and prior-flow configurations are restored from ``load_dir`` and
            overwrite the provided configuration.
        """
        self.condition_dependant_flow = False

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
        Build a conditional variational autoencoder module.

        Parameters
        ----------
        input_shape : np.ndarray | tuple
            Shape of the conditioning input excluding the batch dimension.
        output_shape : np.ndarray | tuple | None
            Shape of the model output excluding the batch dimension. If ``None``,
            ``input_shape`` is used.
        added_features_dim : int | None
            Number of additional features supplied to the model.

        Returns
        -------
        cVAE
            Constructed conditional variational autoencoder module.
        """
        return cVAE(
            config=self,
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

    def _load_from_checkpoint(self, load_path: Path | str):
        """
        Restore module configuration from a saved checkpoint.

        The model selector and prior-flow configuration are reconstructed from the
        saved module configuration. Posterior variance limits and the combined loss
        weight are restored when they have not been explicitly configured.

        Parameters
        ----------
        load_path : Path | str
            Path to the saved checkpoint.

        Returns
        -------
        cVAEConfig
            This configuration instance populated with the saved settings.
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

        if self.posterior_variance_limits is None:
            self.posterior_variance_limits = checkpoint_module.get(
                "posterior_variance_limits", None
            )
        if self.combined_CGCN_weight is None:
            self.combined_CGCN_weight = checkpoint_module.get("combined_CGCN_weight", 0)

        del checkpoint_config, checkpoint_module
        gc.collect()
        return self


class cVAE(moduleABC):
    """
    Conditional variational autoencoder training and inference module.

    The module builds the selected cVAE architecture, optionally constructs a
    normalizing flow for the latent prior, and manages reconstruction and
    Kullback-Leibler divergence losses.

    Parameters
    ----------
    config : cVAEConfig
        Configuration defining the cVAE architecture, latent prior, checkpoint,
        and training objective.
    input_shape : np.ndarray
        Shape of the conditioning input excluding the batch dimension.
    output_shape : np.ndarray | None
        Shape of the model output excluding the batch dimension. If ``None``,
        ``input_shape`` is used.
    added_features_dim : int | None
        Number of additional features supplied to the model.
    """

    def __init__(
        self,
        config: cVAEConfig,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Initialize a conditional variational autoencoder module.

        Parameters
        ----------
        config : cVAEConfig
            Configuration defining the model architecture, latent prior, checkpoint,
            and training settings.
        input_shape : np.ndarray
            Shape of the conditioning input excluding the batch dimension.
        output_shape : np.ndarray | None
            Shape of the model output excluding the batch dimension. If ``None``,
            ``input_shape`` is used.
        added_features_dim : int | None
            Number of additional features supplied to the model.

        Raises
        ------
        AssertionError
            If a configured posterior variance limit is nonpositive.
        RuntimeError
            If checkpoint input or output metadata is incompatible with the current
            runtime configuration, or if the requested input or output shape differs
            from that associated with the checkpoint.
        """
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.latent_size = self.config.latent_size
        self.posterior_variance_limits = self.config.posterior_variance_limits
        self.prior_flow_config = self.config.prior_flow_config
        self.combined_CGCN_weight = self.config.combined_CGCN_weight

        if self.posterior_variance_limits is not None:
            assert all(
                [
                    limit > 0
                    for limit in self.posterior_variance_limits
                    if limit is not None
                ]
            ), "posterior_variance_limits must be positive or None."
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

        if self.posterior_variance_limits is not None:
            min = (
                torch.log(torch.tensor(self.posterior_variance_limits[0]))
                if self.posterior_variance_limits[0] is not None
                else None
            )

            max = (
                torch.log(torch.tensor(self.posterior_variance_limits[1]))
                if self.posterior_variance_limits[1] is not None
                else None
            )

            self.posterior_variance_limits = [min, max]

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.added_features_dim = added_features_dim

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
        Initialize the reconstruction and Kullback-Leibler divergence losses.

        Parameters
        ----------
        reconstruction_loss : Losspipeline
            Reconstruction loss pipeline used to compare generated outputs with
            targets.

        Raises
        ------
        RuntimeError
            If a normalizing flow is configured and the reconstruction loss does not
            use sum reduction.
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
        Compute the cVAE training objective for a batch.

        The objective combines reconstruction loss with a beta-weighted
        Kullback-Leibler divergence. When configured, it is additionally combined
        with a reconstruction loss obtained from prediction.

        Parameters
        ----------
        beta : float
            Weight applied to the Kullback-Leibler divergence term.
        data : BatchData
            Batch containing targets, conditioning inputs, masks, and optional
            additional features.

        Returns
        -------
        tuple[torch.Tensor, dict]
            Total differentiable loss and a dictionary containing scalar loss
            components for reporting.

        Raises
        ------
        RuntimeError
            If the reconstruction loss has not been initialized before computing the
            training objective.
        """
        if self.criterion is None:
            raise RuntimeError(
                "crieterion should be specified before training is possible. Hint: call .init_loss_function() method in your module first."
            )

        latent_sample_size = 1
        output = self.forward(data, latent_sample_size)

        target = data.target
        target_mask = data.target_mask

        target = target.unsqueeze(0).expand(
            latent_sample_size,
            *target.shape,
        )

        if target_mask is not None and target_mask.shape != target.shape:
            target_mask = target_mask.unsqueeze(0).expand_as(target)

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

    def forward(self, data: BatchData, latent_sample_size=1) -> cVAEOutput:
        """
        Perform a cVAE forward pass for a batch.

        During evaluation, generator-enabled models use the configured number of
        validation noise samples. During training, the underlying model controls its
        configured generator sampling behavior.

        Parameters
        ----------
        data : BatchData
            Batch containing targets, conditioning inputs, masks, and optional
            additional features.
        latent_sample_size : int
            Number of posterior latent samples to draw for each batch element.

        Returns
        -------
        cVAEOutput
            Generated outputs together with posterior latent parameters, latent
            samples, and conditional latent information.
        """
        generator = self.model_config.GENERATOR
        output_sample_size = 1
        if not self.training and generator is not None:
            output_sample_size = generator.num_validation_noise_samples

        return self.model(
            cVAEForwardRequest(
                target=data.target,
                target_mask=data.target_mask,
                added_features=data.added_features,
                condition=data.input,
                condition_mask=data.input_mask,
                posterior_variance_limits=self.posterior_variance_limits,
                latent_sample_size=latent_sample_size,
                output_sample_size=output_sample_size,
            )
        )

    def predict(
        self,
        data: BatchData,
        latent_sample_size: int = 1,
        nstds: int = 1,
        latent_samples: torch.Tensor = None,
        output_sample_size: int = 1,
        chunk_size: int | None = None,
    ) -> cVAEOutput:
        """
        Generate predictions by sampling from the cVAE prior.

        User-provided latent samples may be supplied instead of sampling from the
        prior. Generator-enabled models can produce multiple stochastic outputs for
        each latent sample.

        Parameters
        ----------
        data : BatchData
            Batch containing conditioning inputs, masks, and optional additional
            features.
        latent_sample_size : int
            Number of latent samples used for each conditioning input.
        nstds : int
            Scale factor applied to the standard deviation of prior sampling.
        latent_samples : torch.Tensor | None
            Optional user-provided latent samples. If ``None``, latent values are
            sampled from the configured prior.
        output_sample_size : int
            Number of stochastic outputs requested for each latent sample.
        chunk_size : int | None
            Maximum number of latent samples processed together during chunked
            generation.

        Returns
        -------
        cVAEOutput
            Generated predictions and associated conditional latent information.
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
                latent_sample_size=latent_sample_size,
                nstds=nstds,
                latent_samples=latent_samples,
                output_sample_size=output_sample_size,
                chunk_size=chunk_size,
            )
        )

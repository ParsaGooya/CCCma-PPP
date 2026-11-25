import torch
import numpy as np
from pathlib import Path
import warnings
import dataclasses
import dacite
import gc

from S2D4P.loss.loss import Losspipeline
from S2D4P.loss.kld import KLD
from S2D4P.core.core_abc import moduleABC, moduleConfigABC, OutputABC
from S2D4P.core.selectors import (
    ModuleSelector,
    cVAEArchSelector,
    _load_config_from_checkpoint,
)
from S2D4P.architectures.normalized_flows import NormalizedFlowConfig
from S2D4P.architectures.architecture_abc import cVAEPredictRequest, cVAEForwardRequest
from S2D4P.train.dataloader import BatchData



@dataclasses.dataclass
class cVAEOutput(OutputABC):
    """
    Container for conditional variational autoencoder outputs.

    Parameters
    ----------
    output : torch.Tensor
        Generated model output.
    mu : torch.Tensor | None
        Mean of the posterior latent distribution.
    log_var : torch.Tensor | None
        Log-variance of the posterior latent distribution.
    samples : torch.Tensor | None
        Samples drawn from the posterior latent distribution.
    cond_mu : torch.Tensor | None
        Mean or embedding produced from the conditioning input.
    cond_log_var : torch.Tensor | None
        Log-variance of the conditional latent distribution when explicitly
        parameterized.
    deterministic_guess : bool
        Whether ``output`` represents only the deterministic guess branch.
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
    Architecture : cVAEArchSelector | None
        Selector defining the cVAE architecture.
    posterior_variance_limits : list[float | None, float | None] | tuple[float | None, float | None] | None
        Optional lower and upper bounds on posterior variance.
    prior_flow_config : NormalizedFlowConfig | None
        Optional normalizing-flow configuration defining the latent prior.
    combined_CGCN_weight : float | None
        Weight assigned to the prior-sampled reconstruction loss when combining
        it with the standard cVAE training objective.
    load_dir : str | None
        Path to a saved module checkpoint from which architecture and prior-flow
        settings are restored.
    """

    Architecture: cVAEArchSelector | None = None
    posterior_variance_limits: (
        list[float | None, float | None] | tuple[float | None, float | None] | None
    ) = None
    prior_flow_config: NormalizedFlowConfig | None = None
    combined_CGCN_weight: float = None
    load_dir: str | None = None

    def __post_init__(self):
        """
        Resolve and validate the cVAE module configuration.

        When a checkpoint is provided, its architecture and prior-flow configuration
        replace explicitly configured values. Flow conditioning is resolved from the
        architecture, and the combined reconstruction weight defaults to zero.

        Raises
        ------
        AssertionError
            If ``combined_CGCN_weight`` is outside the interval [0, 1].
        ValueError
            If neither an architecture configuration nor a checkpoint path is
            provided.

        Warns
        -----
        UserWarning
            If a checkpoint is loaded and its saved architecture and prior-flow
            configuration replace the current settings.
        """
        self.condition_dependant_flow = False

        if self.load_dir is None:
            if self.Architecture is None:
                raise ValueError("provide loading dir or architecture configurations")

        else:
            self._load_from_checkpoint(self.load_dir)
            warnings.warn(
                f"Architecture and prior flow config overwritten by the saved module: \n {self.load_dir}"
            )

        if self.combined_CGCN_weight is None:
            self.combined_CGCN_weight = 0

        self.architecture_config = self.Architecture.get_architecture_config()

        self.latent_size = self.architecture_config.latent_size

        if self.prior_flow_config is not None:
            self.condition_dependant_flow = (
                self.architecture_config.condition_dependant_latent
            )

        self.architecture_config._resolve_flow_settings(self.condition_dependant_flow)

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
            Shape of each model input excluding the batch dimension.
        output_shape : np.ndarray | tuple | None
            Shape of each model output excluding the batch dimension. If ``None``,
            the input shape is used.
        added_features_dim : int | None
            Number of additional features supplied with each sample.

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
        Restore cVAE configuration from a checkpoint.

        The architecture selector and prior-flow configuration are reconstructed from
        the saved module configuration. Posterior variance limits and the combined
        reconstruction weight are inherited when they are not explicitly configured.

        Parameters
        ----------
        load_path : Path | str
            Path to the checkpoint containing the saved module configuration.

        Returns
        -------
        cVAEConfig
            This configuration populated with checkpoint settings.
        """
        checkpoint_module, checkpoint_config = _load_config_from_checkpoint(
            Path(load_path)
        )
        self.checkpoint_config = checkpoint_config
        self.Architecture = dacite.from_dict(
            data_class=cVAEArchSelector,
            data=checkpoint_module.get("Architecture"),
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
    Conditional variational autoencoder module.

    Combines a configurable cVAE architecture with an optional normalizing-flow
    prior, reconstruction loss, and Kullback-Leibler divergence objective.

    Parameters
    ----------
    config : cVAEConfig
        Configuration defining the architecture, latent prior, variance limits,
        and training objective.
    input_shape : np.ndarray
        Shape of each model input excluding the batch dimension.
    output_shape : np.ndarray | None
        Shape of each model output excluding the batch dimension. If ``None``,
        the input shape is used.
    added_features_dim : int | None
        Number of additional features supplied with each sample.
    """

    def __init__(
        self,
        config: cVAEConfig,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Initialize the conditional variational autoencoder module.

        Builds the selected architecture and optional normalizing-flow prior. When a
        saved module is loaded, input and output shapes and variable metadata are
        validated against the checkpoint before model state is restored.

        Parameters
        ----------
        config : cVAEConfig
            Configuration defining the cVAE module.
        input_shape : np.ndarray
            Shape of each model input excluding the batch dimension.
        output_shape : np.ndarray | None
            Shape of each model output excluding the batch dimension. If ``None``,
            the input shape is used.
        added_features_dim : int | None
            Number of additional features supplied with each sample.

        Raises
        ------
        AssertionError
            If a configured posterior variance limit is not positive.
        RuntimeError
            If current input or target metadata, input shape, or output shape are
            incompatible with the loaded checkpoint.
        """
        super().__init__()

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
            self.flow_condition_size = self.architecture_config.condition_embedding_size
        else:
            self.flow_condition_size = None

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

        self.model_setup(config, input_shape, output_shape, added_features_dim)

        self.arch = self.architecture_config.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

        self.prior_flow = None
        if self.prior_flow_config is not None:
            self.prior_flow = self.prior_flow_config.build(
                latent_size=self.latent_size, condition_size=self.flow_condition_size
            )

        self._load_state_dict(self.config.load_dir)

    def init_loss_function(self, reconstruction_loss: Losspipeline):
        """
        Initialize the reconstruction and Kullback-Leibler losses.

        The reconstruction loss is moved to the module device and a matching KLD loss
        is constructed with the same reduction method.

        Parameters
        ----------
        reconstruction_loss : Losspipeline
            Reconstruction loss pipeline used during cVAE training.

        Raises
        ------
        RuntimeError
            If a normalizing-flow prior is configured but the reconstruction loss
            does not use sum reduction.
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

        The objective combines the reconstruction loss with beta-weighted
        Kullback-Leibler divergence. When ``combined_CGCN_weight`` is positive, this
        objective is additionally combined with a reconstruction loss calculated from
        prior-sampled predictions.

        Parameters
        ----------
        beta : float
            Weight applied to the Kullback-Leibler divergence.
        data : BatchData
            Training batch containing inputs, targets, masks, and optional additional
            features.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor | float]]
            Total differentiable loss and individual loss terms used for logging.

        Raises
        ------
        RuntimeError
            If the reconstruction loss has not been initialized.
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
        Run the cVAE training or validation forward pass.

        The batch target defines the posterior distribution while the model input
        provides conditioning information. During validation, a configured generator
        uses its validation output-sample count.

        Parameters
        ----------
        data : BatchData
            Batch containing targets, conditioning inputs, masks, and optional
            additional features.
        latent_sample_size : int
            Number of posterior latent samples to generate for each batch element.

        Returns
        -------
        cVAEOutput
            Predictions, posterior parameters, latent samples, and conditional latent
            parameters.
        """
        generator = self.architecture_config.GENERATOR
        output_sample_size = 1
        if not self.training and generator is not None:
            output_sample_size = generator.num_validation_noise_samples

        return self.arch(
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
        Generate predictions from the cVAE latent prior.

        Latent samples can be generated internally or supplied explicitly. An
        optional normalizing flow transforms prior samples before generation. During
        training, generator-enabled architectures use the configured number of
        training noise samples.

        Parameters
        ----------
        data : BatchData
            Batch containing conditioning inputs, masks, and optional additional
            features.
        latent_sample_size : int
            Number of latent samples generated for each batch element.
        nstds : int
            Standard-deviation multiplier used during latent prior sampling.
        latent_samples : torch.Tensor | None
            Optional explicitly supplied latent samples. If ``None``, samples are
            generated from the configured prior.
        output_sample_size : int
            Number of stochastic outputs generated for each latent sample.
        chunk_size : int | None
            Number of latent samples processed in each generation chunk. If ``None``,
            all latent samples are processed together.

        Returns
        -------
        cVAEOutput
            Generated predictions and associated conditional latent parameters.
        """
        generator = self.architecture_config.GENERATOR
        if self.training and generator is not None:
            output_sample_size = generator.num_training_noise_samples

        return self.arch.predict(
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

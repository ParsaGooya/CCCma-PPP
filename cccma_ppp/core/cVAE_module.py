import torch
import numpy as np
from pathlib import Path

from cccma_ppp.loss.loss import Losspipeline

from cccma_ppp.core.registery import *
from cccma_ppp.core.core_abc import moduleABC
from cccma_ppp.core.selectors import *
from cccma_ppp.models.normalized_flows import NormalizedFlowConfig
from cccma_ppp.data.dataloader import BatchData

from cccma_ppp.loss.kld import KLD


@ModuleSelector.register("cvae")
@dataclasses.dataclass
class cVAEConfig:
    """
    Configuration for constructing a conditional variational autoencoder module.
    """

    ModelConfig: cVAEModelSelector | None = None
    min_posterior_variance: float | None = None
    prior_flow_config: NormalizedFlowConfig | None = None
    combined_CGCN_weight: float = 0
    load_dir: str | None = None

    def __post_init__(self):
        """
        Validate configuration and initialize latent and conditioning settings.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If required configuration values are missing or invalid.
        """

        if self.load_dir is None:
            assert self.ModelConfig is not None, (
                "provide loading dir or model configurations"
            )

            self.latent_size = self.ModelConfig.args.get("latent_size")

            ## read condition_dependant_latent from the ModelConfig so that we know the flow should be conditional.
            self.condition_dependant_flow = self.ModelConfig.args.get(
                "condition_dependant_latent"
            )
            ## if prior flow is requested, set the condition_dependant_latent for the model to False because we don't want to generate cond_mu and cond_log_var.
            if self.prior_flow_config is not None:
                self.ModelConfig.args["condition_dependant_latent"] = False
            ##note that as long as .build() is not called, the cVAE module is not generated, thus the selected model is not created, so no issues with overwriting the config at this level.
            # When the model is loaded, if the flow is off,  condition_dependant_latent has been turned off and condition_dependant_flow is on, so even though the flow sees the condition, there is no requirement that the condition_embedding_size == latent_size.

        else:
            RuntimeWarning(
                f"all model config overwritten by the loaded model: \n {self.load_dir}"
            )

        assert 0 <= self.combined_CGCN_weight <= 1, (
            "CGCN weight should be between [0,1]"
        )
        self.model = self.ModelConfig.get_model()

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Build and return a cVAE module instance.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of input data.
        output_shape : np.ndarray or None, optional
            Shape of output data.
        added_features_dim : int, optional
            Dimension of additional features.

        Returns
        -------
        cVAE
            Constructed cVAE module.
        """

        return cVAE(self).build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )


@dataclasses.dataclass
class cVAEOutput:
    """
    Container for cVAE outputs and latent distribution parameters.
    """

    output: torch.Tensor
    mu: torch.Tensor | None
    log_var: torch.Tensor | None
    cond_mu: torch.Tensor | None = None
    cond_log_var: torch.Tensor | None = None


class cVAE(moduleABC):
    """
    Conditional variational autoencoder module with optional flow-based prior.
    """

    def __init__(self, config: cVAEConfig | None = None):
        """
        Initialize cVAE module.

        Parameters
        ----------
        config : cVAEConfig or None, optional
            Configuration for the module.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If configuration values are invalid.
        """

        super().__init__()
        self.config = config
        self.model = self.config.model
        self.latent_size = self.config.latent_size
        self.min_posterior_variance = self.config.min_posterior_variance
        self.prior_flow_config = self.config.prior_flow_config
        self.combined_CGCN_weight = self.config.combined_CGCN_weight

        if self.min_posterior_variance is not None:
            assert self.min_posterior_variance > 0, (
                "min_posterior_variance must be positive."
            )
        if self.config.condition_dependant_flow:
            self.flow_condition_size = self.model.condition_embedding_size
        else:
            self.flow_condition_size = None

        self.built = False
        self.criterion = None

    def build(
        self,
        input_shape: np.ndarray,
        output_shape: np.ndarray | None = None,
        added_features_dim: int = None,
    ):
        """
        Build encoder, decoder, and optional prior flow components.

        Parameters
        ----------
        input_shape : np.ndarray
            Shape of input data.
        output_shape : np.ndarray or None, optional
            Shape of output data.
        added_features_dim : int, optional
            Dimension of additional features.

        Returns
        -------
        cVAE
            Built module instance.
        """

        self.input_shape = input_shape
        self.output_shape = output_shape
        if self.min_posterior_variance is not None:
            self.min_posterior_variance = torch.log(
                torch.tensor(self.min_posterior_variance)
            )  # .expand(self.latent_size)

        self.model.build(
            input_shape=input_shape,
            output_shape=output_shape,
            added_features_dim=added_features_dim,
        )

        if self.prior_flow_config is not None:
            self.prior_flow = self.prior_flow_config.build(
                latent_size=self.latent_size, condition_size=self.flow_condition_size
            )

        self.built = True

        if self.config.load_dir is not None:
            self._load_from_state(self.config.load_dir)

        return self

    def init_loss_function(self, reconstruction_loss: Losspipeline):
        """
        Initialize reconstruction and KL divergence losses.

        Parameters
        ----------
        reconstruction_loss : Losspipeline
            Reconstruction loss pipeline.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If loss configuration is incompatible with flow-based prior.
        """

        if self.prior_flow_config is not None:
            if reconstruction_loss.reduction.lower() != "sum":
                raise RuntimeError(
                    "with normalized flow all loss reduction has to be sum."
                )

        self.criterion = reconstruction_loss
        self.KLD = KLD(
            reduction=self.criterion.reduction,
            prior_flow=getattr(self, "prior_flow", None),
        )

    def _compute_loss(self, beta: float, data: BatchData):
        """
        Compute total loss including reconstruction and KL divergence.

        Parameters
        ----------
        beta : float
            Weight for KL divergence term.
        data : BatchData
            Input batch data.

        Returns
        -------
        tuple
            Total loss tensor and dictionary of loss components.

        Raises
        ------
        AssertionError
            If loss function is not initialized.
        """

        assert self.criterion is not None, (
            "crieterion should be specified before training is possible. Hint: call .init_loss_function() method in your module first."
        )

        output = self.forward(data)

        if isinstance(data.target, (tuple, list)):
            target, target_mask = data.target
        else:
            target, target_mask = data.target, None

        if (
            target_mask is not None and target_mask.shape == target.shape
        ):  ## checking if target_mask is static
            target_mask = target_mask.unsqueeze(0).expand_as(output.output)
        target = target.unsqueeze(0).expand_as(
            output.output
        )  ## B x C x F -> Z x B x C x F

        step_arguments = {"generative_modeling": True}

        reconstruction_loss, indiv_losses = self.criterion(
            output.output,
            target,
            target_mask=target_mask,
            step_arguments=step_arguments,
            print_loss=False,
        )

        kld_loss = self.KLD(
            output.mu,
            output.log_var,
            output.cond_mu,
            output.cond_log_var,
            print_loss=False,
        )

        total_loss = reconstruction_loss + beta * kld_loss

        if self.combined_CGCN_weight > 0:
            output_CGCN = self.preidct(data)
            reconstruction_loss_CGCN, _ = self.criterion(
                output_CGCN.output,
                target,
                step_arguments=step_arguments,
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

    def forward(self, data: BatchData, sample_size=1):
        """
        Perform forward pass and sample latent variables.

        Parameters
        ----------
        data : BatchData
            Input batch data.
        sample_size : int, optional
            Number of latent samples.

        Returns
        -------
        cVAEOutput
            Model outputs and latent parameters.
        """

        return self.model(
            x=data.target,
            added_features=data.added_features,
            condition=data.input,
            min_posterior_variance=self.min_posterior_variance,
            sample_size=sample_size,
        )

    def preidct(self, data: BatchData, sample_size=1):
        """
        Generate samples from the prior distribution.

        Parameters
        ----------
        data : BatchData
            Conditioning input data.
        sample_size : int, optional
            Number of samples.

        Returns
        -------
        object
            Generated model outputs.
        """

        return self.model.predict(
            condition=data.input,
            added_features=data.added_features,
            prior_flow=self.prior_flow,
            sample_size=sample_size,
        )

    def _get_device(self):
        """
        Retrieve device of module parameters or buffers.

        Returns
        -------
        torch.device
            Device where module resides.
        """

        param = next(self.parameters(), None)

        if param is not None:
            return param.device

        buffer = next(self.buffers(), None)

        if buffer is not None:
            return buffer.device

        return torch.device("cpu")

    def _save_state_dict(self, save_path: Path | str):
        """
        Save model and flow state to disk.

        Parameters
        ----------
        save_path : Path or str
            Directory to save checkpoint.

        Returns
        -------
        None
        """

        """
        For DDP, save the underlying module, not the DDP wrapper.
        """

        prior_flow_state = None
        if hasattr(self, "prior_flow") and self.prior_flow is not None:
            prior_flow_state = self.prior_flow.state_dict()

        checkpoint = {
            "model": self.model.state_dict(),
            "latent_size": self.latent_size,
            "min_posterior_variance": self.min_posterior_variance,
            "prior_flow_config": self.config.prior_flow_config,
            "prior_flow": prior_flow_state,
            "combined_CGCN_weight": self.config.combined_CGCN_weight,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
        }

        path = Path(save_path) / "cVAE_module.pt"
        torch.save(checkpoint, path)

    def _load_from_state(self, load_path: Path | str, strict: bool = True):
        """
        Load model state from checkpoint.

        Parameters
        ----------
        load_path : Path or str
            Path to checkpoint file.
        strict : bool, optional
            Whether to enforce strict state loading.

        Returns
        -------
        dict
            Loaded checkpoint data.

        Raises
        ------
        AssertionError
            If module is not built before loading.
        FileNotFoundError
            If checkpoint file does not exist.
        """

        assert self.built, (
            "module stgate should be built for torch to load the weights into. Hint: call .build() method first."
        )

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(
            Path(load_path), map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model"], strict=strict)

        self.latent_size = checkpoint.get("latent_size")
        self.min_posterior_variance = checkpoint.get("min_posterior_variance", None)
        self.prior_flow_config = checkpoint.get("prior_flow_config", None)
        self.combined_CGCN_weight = checkpoint.get("prior_flow_config", 0)

        return checkpoint


import torch
import numpy as np
from pathlib import Path
import warnings
from cccma_ppp.loss.loss import Losspipeline
from cccma_ppp.core.registery import *
from cccma_ppp.core.core_abc import moduleABC, moduleConfigABC
from cccma_ppp.core.selectors import *
from cccma_ppp.models.normalized_flows import NormalizedFlowConfig
from cccma_ppp.data.dataloader import BatchData
from cccma_ppp.loss.kld import KLD
import gc

@ModuleSelector.register('cvae')
@dataclasses.dataclass
class cVAEConfig(moduleConfigABC):
    ModelConfig  : cVAEModelSelector | None = None
    min_posterior_variance: float | None = None
    prior_flow_config : NormalizedFlowConfig | None = None
    combined_CGCN_weight : float = None
    load_dir : str| None = None

    def __post_init__(self):
        if self.load_dir is None:
            assert self.ModelConfig is not None, 'provide loading dir or model configurations'

        else:
            self._load_from_checkpoint(self.load_dir)
            warnings.warn(f'Module config overwritten by the saved module: \n {self.load_dir}')
        
        if self.combined_CGCN_weight is None:
            self.combined_CGCN_weight = 0

        self.latent_size = self.ModelConfig.args.get('latent_size')

        if self.prior_flow_config is not None:
                ## read condition_dependant_latent from the ModelConfig so that we know the flow should be conditional.
                self.condition_dependant_flow = self.ModelConfig.args.get('condition_dependant_latent')
                ## if prior flow is requested, set the condition_dependant_flow tur for the model because we don't want to generate cond_mu and cond_log_var.
                self.ModelConfig.args['condition_dependant_flow'] = True

        assert 0 <= self.combined_CGCN_weight <= 1, 'CGCN weight should be between [0,1]'

        self.model = self.ModelConfig.get_model()
    def build(self,   ## this instantiates cVAE module and builds it at the same time.
              input_shape : np.ndarray,
              output_shape: np.ndarray|None = None,
              added_features_dim : int = None):

        return cVAE(self).build(input_shape= input_shape,
                                 output_shape = output_shape,
                                 added_features_dim = added_features_dim)


    def _load_from_checkpoint(self, load_path : Path | str):

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        
        checkpoint = torch.load( Path(load_path), weights_only=False)
        _checkpoint_config = checkpoint.get('module_config')

        self._checkpoint_input_shape = checkpoint.get('model_input_shape')
        self._checkpoint_output_shape = checkpoint.get('model_output_shape')
        
        self.ModelConfig  =  dacite.from_dict(
                                        data_class=cVAEModelSelector,
                                        data= _checkpoint_config.get("ModelConfig"),
                                        config=dacite.Config(strict=True))

        self.prior_flow_config = _checkpoint_config.get('prior_flow_config', None)
        if self.prior_flow_config is not None:
            self.prior_flow_config  =  dacite.from_dict(
                                            data_class=NormalizedFlowConfig,
                                            data= self.prior_flow_config,
                                            config=dacite.Config(strict=True))
        
        if self.min_posterior_variance is None:
            self.min_posterior_variance = _checkpoint_config.get('min_posterior_variance', None)
        if self.combined_CGCN_weight is None:
            self.combined_CGCN_weight = _checkpoint_config.get('combined_CGCN_weight', 0)

        del checkpoint, _checkpoint_config
        gc.collect()
        return self



@dataclasses.dataclass
class cVAEOutput:
    output : torch.Tensor
    mu : torch.Tensor  | None
    log_var : torch.Tensor | None
    cond_mu : torch.Tensor | None = None
    cond_log_var : torch.Tensor | None = None




class cVAE( moduleABC):
    def __init__(self,
                config : cVAEConfig| None = None):

        super().__init__()
        self.config = config
        self.model = self.config.model
        self.latent_size = self.config.latent_size
        self.min_posterior_variance = self.config.min_posterior_variance
        self.prior_flow_config = self.config.prior_flow_config
        self.combined_CGCN_weight = self.config.combined_CGCN_weight

        if self.min_posterior_variance is not None:
            assert self.min_posterior_variance > 0, 'min_posterior_variance must be positive.'
        if getattr(self.config , 'condition_dependant_flow', False):
            self.flow_condition_size = self.model.condition_embedding_size
        else:
            self.flow_condition_size = None

        self.built = False
        self.criterion = None

        self.prior_flow = None
    def build(self,
              input_shape : np.ndarray,
              output_shape: np.ndarray|None = None,
              added_features_dim : int = None):

        if output_shape is None:
            output_shape = input_shape.copy()

        self.input_shape = input_shape
        self.output_shape = output_shape

        if self.config.load_dir is not None:
            assert self.input_shape == self.config._checkpoint_input_shape, f'the requested input shape ({self.input_shape}) does not match the loaded module : {self.config._checkpoint_input_shape}'
            assert self.output_shape == self.config._checkpoint_output_shape, f'the requested output shape ({self.output_shape}) does not match the loaded module : {self.config._checkpoint_output_shape}'

        if self.min_posterior_variance is not None:
              self.min_posterior_variance = torch.log(torch.tensor(self.min_posterior_variance))  #.expand(self.latent_size)

        self.model.build(input_shape = input_shape,
                                            output_shape = output_shape,
                                            added_features_dim = added_features_dim)

        if self.prior_flow_config is not None:
            self.prior_flow = self.prior_flow_config.build(latent_size = self.latent_size,
                                                                condition_size = self.flow_condition_size)

        self.built = True

        if self.config.load_dir is not None:
            self._load_state_dict(self.config.load_dir)

        return self



    def init_loss_function(self,
                           reconstruction_loss : Losspipeline):


        if self.prior_flow_config is not None:
            if reconstruction_loss.reduction.lower() != 'sum':
                raise RuntimeError('with normalized flow all loss reduction has to be sum.')

        self.criterion = reconstruction_loss
        self.KLD = KLD( reduction = self.criterion.reduction)

    def _compute_loss(self,
                        beta: float,
                        data : BatchData):

        assert self.criterion is not None, 'crieterion should be specified before training is possible. Hint: call .init_loss_function() method in your module first.'

        output = self.forward(data)

        if isinstance(data.target, (tuple, list)):
            target, target_mask = data.target
        else:
            target, target_mask = data.target , None

        if (target_mask is not None and target_mask.shape == target.shape): ## checking if target_mask is static
            target_mask = target_mask.unsqueeze(0).expand_as(output.output)
        target = target.unsqueeze(0).expand_as(output.output) ## B x C x F -> Z x B x C x F

        step_arguments = {'generative_modeling' : True}


        reconstruction_loss, indiv_losses = self.criterion(
            output.output,
            target,
            target_mask = target_mask,
            step_arguments = step_arguments,
            print_loss = False)

        kld_loss = self.KLD(
            output.mu,
            output.log_var,
            output.cond_mu,
            output.cond_log_var,
            prior_flow =  self.prior_flow,
            print_loss = False)

        total_loss = reconstruction_loss + beta * kld_loss

        if self.combined_CGCN_weight > 0 :
            output_CGCN = self.preidct(data)
            reconstruction_loss_CGCN, _ = self.criterion(
                output_CGCN.output,
                target,
                step_arguments = step_arguments,
                print_loss = False)

            total_loss = total_loss * (1- self.combined_CGCN_weight) + self.combined_CGCN_weight * reconstruction_loss_CGCN
            indiv_losses['total_loss_CGCN']  = reconstruction_loss_CGCN.item()

        losses_dict = {
            "total_loss": total_loss.item(),
            "kld": kld_loss.item()}

        for key, value in indiv_losses.items():
            losses_dict[key] = value

        return total_loss, losses_dict

    def forward(self, data: BatchData,
                sample_size  = 1) -> cVAEOutput:


            return self.model(x = data.target,
                              added_features = data.added_features,
                              condition = data.input,
                              min_posterior_variance = self.min_posterior_variance,
                              sample_size = sample_size)

    def preidct(self, data: BatchData,
                sample_size = 1):


            return  self.model.predict( condition =data.input,
                added_features =data.added_features,
                prior_flow = self.prior_flow,
                sample_size = sample_size)








    # def _save_state_dict(self, save_path : Path | str):
    #     """
    #     For DDP, save the underlying module, not the DDP wrapper.
    #     """

    #     prior_flow_state = None
    #     if hasattr(self, "prior_flow") and self.prior_flow is not None:
    #         prior_flow_state = self.prior_flow.state_dict()

    #     checkpoint = {
    #             'model' : self.model.state_dict(),
    #             'latent_size' : self.latent_size,
    #             'min_posterior_variance' : self.min_posterior_variance,
    #             'prior_flow_config' : self.config.prior_flow_config,
    #             'prior_flow' :   prior_flow_state,
    #             'combined_CGCN_weight' : self.config.combined_CGCN_weight,
    #             'input_shape' : self.input_shape,
    #             'output_shape' : self.output_shape }


    #     path = Path(save_path) / f"cVAE_module.pt"
    #     torch.save(checkpoint, path)


    # def _load_from_state(self, load_path : Path | str, strict : bool = True):

    #     assert self.built, 'module stgate should be built for torch to load the weights into. Hint: call .build() method first.'

    #     if not Path(load_path).exists():
    #         raise FileNotFoundError(f"Checkpoint not found: {load_path}")

    #     checkpoint = torch.load( Path(load_path), map_location=self.device, weights_only=False)

    #     self.model.load_state_dict( checkpoint["model"],strict=strict)

    #     self.latent_size = checkpoint.get("latent_size")
    #     self.min_posterior_variance = checkpoint.get("min_posterior_variance", None)
    #     self.prior_flow_config = checkpoint.get("prior_flow_config", None)
    #     self.combined_CGCN_weight = checkpoint.get("prior_flow_config", 0)

    #     return checkpoint

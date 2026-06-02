import torch
import torch.nn as nn
import torch.nn.init as init
import math

import dataclasses

from src.cccma_ppp.models.models_abc import flowABC
from src.cccma_ppp.core.selectors import FlowSelector


@dataclasses.dataclass
class flowOutput:
    """
    Container for outputs of a flow transformation.
    """

    e_samples: torch.Tensor
    log_det: torch.Tensor


@dataclasses.dataclass
class NormalizedFlowConfig:
    """
    Configuration for constructing a sequence of normalizing flow transformations.
    """

    # registery: ClassVar[Registery] = Registery()  ####  registery machinery is not part of the FlowSelector #####

    list_flows: list[FlowSelector]
    flow_sample_size: int = 5000

    def build(self, latent_size, condition_size=None):
        """
        Build a NormalizedFlowModel.

        Parameters
        ----------
        latent_size : int
            Dimensionality of latent space.
        condition_size : int, optional
            Size of conditioning input.

        Returns
        -------
        NormalizedFlowModel
            Constructed flow model.
        """
        # this instatiates and build flow model at the same time.

        return NormalizedFlowModel(self).build(
            latent_size=latent_size, condition_size=condition_size
        )

    ####  registery machinery is not part of the FlowSelector #####
    # @classmethod
    # def register(cls, name: str) -> Callable[..., flowABC]:  # noqa: UP006
    #     return cls.registery.register(name.lower())   ##attentin: the return is on cls.registery. This means even when register is called on an instance of StepSelector, it will still register the type on the class-level registery, which is what we want.

    # def available(cls):
    #     return cls.registery.available()


class NormalizedFlowModel(flowABC):
    """
    Sequential container of normalizing flow transformations.
    """

    def __init__(self, config):
        """
        Initialize flow model from configuration.

        Parameters
        ----------
        config : NormalizedFlowConfig
            Flow configuration.

        Returns
        -------
        None
        """

        super().__init__()

        #### try this when registery machinery is part of NormalizedFlowConfig itself and not a separate selector #####
        # self.config = config
        # self.list_flows = self.config.list_flows
        # self.flow_sample_size = self.config.flow_sample_size
        ###############################################################################################################

        self.list_flows = config.list_flows
        self.flow_sample_size = config.flow_sample_size
        self.flows = []

        for step in self.list_flows:
            # self.flows.append(self.config.registery.get(name, args))  #### try this when registery machinery is part of NormalizedFlowConfig itself and not a separate selector #####
            self.flows.append(step.get_model())

    def build(self, latent_size, condition_size=None):
        """
        Build all flow components.

        Parameters
        ----------
        latent_size : int
            Dimensionality of latent space.
        condition_size : int, optional
            Conditioning feature size.

        Returns
        -------
        NormalizedFlowModel
            Built model.
        """

        self.condition_size = condition_size

        for ind, flow in enumerate(self.flows):
            self.flows[ind] = flow.build(
                dim=latent_size, condition_size=self.condition_size
            )

        self.flows = nn.ModuleList(self.flows)

        return self

    def forward(self, x, condition=None):
        """
        Apply forward flow transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        condition : torch.Tensor, optional
            Conditioning tensor.

        Returns
        -------
        flowOutput
            Transformed output and log-determinant of Jacobian.
        """

        bsz, _ = x.shape
        log_det = torch.zeros(bsz, device=x.device)
        for flow in self.flows:
            x, ld = flow.forward(x, condition=condition)
            log_det = log_det + ld
        # z, prior_logprob = x, self.prior.log_prob(x)

        return flowOutput(e_samples=x, log_det=log_det)

    def inverse(self, z, condition=None):
        """
        Apply inverse flow transformation.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor.
        condition : torch.Tensor, optional
            Conditioning tensor.

        Returns
        -------
        flowOutput
            Inverse-transformed output and log-determinant.
        """

        bsz, _ = z.shape
        log_det = torch.zeros(bsz, device=z.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, condition=condition)
            log_det = log_det + ld
        x = z

        return flowOutput(e_samples=x, log_det=log_det)


# from core.config_classes import NormalizedFlowConfig
########################################################################################################################################################
#    Code below sourced from https://github.com/tonyduan/normalizing-flows/tree/master?tab=readme-ov-file with adjustments to make the flows conditional.
########################################################################################################################################################


class FCNN(nn.Module):
    """
    Fully connected neural network used in flow parameterization.
    """

    def __init__(self, in_dim, out_dim, hidden_dim):
        """
        Initialize network.

        Parameters
        ----------
        in_dim : int
            Input dimension.
        out_dim : int
            Output dimension.
        hidden_dim : int
            Hidden layer size.

        Returns
        -------
        None
        """

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        """
        Apply forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        return self.network(x)


# @NormalizedFlowConfig.register('maf')


@FlowSelector.register("maf")
class MAF(flowABC):
    """
    Masked autoregressive flow (MAF).
    """

    def __init__(self, hidden_dim=16, base_network=FCNN):
        """
        Initialize MAF flow.

        Parameters
        ----------
        hidden_dim : int, optional
            Hidden layer size.
        base_network : callable, optional
            Network used for parameterization.

        Returns
        -------
        None
        """

        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_network = base_network

    def build(self, dim, condition_size=None):
        """
        Build autoregressive layers.

        Parameters
        ----------
        dim : int
            Input dimension.
        condition_size : int, optional
            Conditioning size.

        Returns
        -------
        MAF
            Built flow model.
        """

        self.dim = dim
        # self.layers = nn.ModuleList()
        layers = []
        added_features = condition_size if condition_size is not None else 0
        for i in range(1, self.dim):
            layers.append(self.base_network(i + added_features, 2, self.hidden_dim))
        self.layers = nn.Sequential(*layers)
        if condition_size is None:
            self.register_parameter(
                "initial_param", param=nn.Parameter(torch.Tensor(1, 2))
            )  ## make sure to register the newly defined parameters so that they are trainable!
            self.reset_parameters()
        else:
            self.initial_param = self.base_network(added_features, 2, self.hidden_dim)

        return self

    def reset_parameters(self):
        """
        Initialize learnable parameters.

        Returns
        -------
        None
        """

        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x, condition=None):
        """
        Apply forward autoregressive transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        condition : torch.Tensor, optional
            Conditioning tensor.

        Returns
        -------
        tuple
            Transformed tensor and log-determinant.
        """

        z = torch.zeros_like(x)
        log_det = torch.zeros(z.shape[0], device=x.device)
        for i in range(self.dim):
            if i == 0:
                if condition is None:
                    mu, alpha = self.initial_param[:, 0], self.initial_param[:, 1]
                else:
                    out = self.initial_param(condition)
                    mu, alpha = out[:, 0], out[:, 1]
            else:
                if condition is None:
                    x_in = x[:, :i].clone()
                else:
                    x_in = torch.cat([x[:, :i].clone(), condition], dim=-1)

                out = self.layers[i - 1](x_in)
                mu, alpha = out[:, 0], out[:, 1]
            z[:, i] = (x[:, i] - mu) / torch.exp(alpha)
            log_det -= alpha
        return z.flip(dims=(1,)), log_det

    def inverse(self, z, condition=None):
        """
        Apply inverse transformation.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor.
        condition : torch.Tensor, optional
            Conditioning tensor.

        Returns
        -------
        tuple
            Reconstructed input and log-determinant.
        """

        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0], device=z.device)
        z = z.flip(dims=(1,))
        for i in range(self.dim):
            if i == 0:
                if condition is None:
                    mu, alpha = self.initial_param[0], self.initial_param[1]
                else:
                    out = self.initial_param(condition)
                    mu, alpha = out[:, 0], out[:, 1]
            else:
                if condition is None:
                    x_in = x[:, :i].clone()
                else:
                    x_in = torch.cat([x[:, :i].clone(), condition], dim=-1)
                out = self.layers[i - 1](x_in)
                mu, alpha = out[:, 0], out[:, 1]
            x[:, i] = mu + torch.exp(alpha) * z[:, i]
            # print(alpha)
            log_det += alpha  # torch.clamp(-1*alpha, max=10)
        return x, log_det


# @NormalizedFlowConfig.register('realnvp')


@FlowSelector.register("realnvp")
class RealNVP(flowABC):
    """
    Real-valued non-volume preserving flow.
    """

    def __init__(self, hidden_dim=16, base_network=FCNN):
        """
        Initialize RealNVP flow.

        Parameters
        ----------
        hidden_dim : int, optional
            Hidden layer size.
        base_network : callable, optional
            Network used for scale and translation.

        Returns
        -------
        None
        """

        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_network = base_network

    def build(self, dim, condition_size=None):
        """
        Build coupling layers.

        Parameters
        ----------
        dim : int
            Input dimensionality.
        condition_size : int, optional
            Conditioning size.

        Returns
        -------
        RealNVP
            Built flow model.
        """

        self.dim = dim
        added_features = condition_size if condition_size is not None else 0
        self.t1 = self.base_network(
            added_features + self.dim // 2, self.dim // 2, self.hidden_dim
        )
        self.s1 = self.base_network(
            added_features + self.dim // 2, self.dim // 2, self.hidden_dim
        )
        self.t2 = self.base_network(
            added_features + self.dim // 2, self.dim // 2, self.hidden_dim
        )
        self.s2 = self.base_network(
            added_features + self.dim // 2, self.dim // 2, self.hidden_dim
        )

        return self

    def forward(self, x, condition=None):
        """
        Apply forward coupling transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        condition : torch.Tensor, optional
            Conditioning tensor.

        Returns
        -------
        tuple
            Transformed output and log-determinant.
        """

        lower, upper = x[:, : self.dim // 2].clone(), x[:, self.dim // 2 :].clone()
        if condition is not None:
            t1_transformed = self.t1(torch.cat([lower, condition], dim=-1))
            s1_transformed = self.s1(torch.cat([lower, condition], dim=-1))
        else:
            t1_transformed = self.t1(lower)
            s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        if condition is not None:
            t2_transformed = self.t2(torch.cat([upper, condition], dim=-1))
            s2_transformed = self.s2(torch.cat([upper, condition], dim=-1))
        else:
            t2_transformed = self.t2(upper)
            s2_transformed = self.s2(upper)

        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(s1_transformed, dim=1) + torch.sum(s2_transformed, dim=1)
        return z, log_det

    def inverse(self, z, condition=None):
        """
        Apply inverse coupling transformation.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor.
        condition : torch.Tensor, optional
            Conditioning tensor.

        Returns
        -------
        tuple
            Reconstructed input and log-determinant.
        """

        lower, upper = z[:, : self.dim // 2].clone(), z[:, self.dim // 2 :].clone()
        if condition is not None:
            t2_transformed = self.t2(torch.cat([upper, condition], dim=-1))
            s2_transformed = self.s2(torch.cat([upper, condition], dim=-1))
        else:
            t2_transformed = self.t2(upper)
            s2_transformed = self.s2(upper)

        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)

        if condition is not None:
            t1_transformed = self.t1(torch.cat([lower, condition], dim=-1))
            s1_transformed = self.s1(torch.cat([lower, condition], dim=-1))
        else:
            t1_transformed = self.t1(lower)
            s1_transformed = self.s1(lower)

        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + torch.sum(-s2_transformed, dim=1)
        return x, log_det

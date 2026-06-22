from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.init as init
import math

import dataclasses

from cccma_ppp.models.models_abc import flowABC
from cccma_ppp.core.selectors import FlowSelector


@dataclasses.dataclass
class flowOutput:
    """
    Output container for flow transformations.

    Parameters
    ----------
    e_samples : torch.Tensor
        Transformed samples.
    log_det : torch.Tensor
        Log-determinant of the Jacobian.
    """

    e_samples: torch.Tensor
    log_det: torch.Tensor


@dataclasses.dataclass
class NormalizedFlowConfig:
    """
    Configuration for a composed normalizing flow model.

    Parameters
    ----------
    list_flows : list of FlowSelector
        Sequence of flow components.
    flow_sample_size : int, optional
        Number of samples used for Monte Carlo estimates.
    """

    list_flows: list[FlowSelector]
    flow_sample_size: int = 5000

    def build(self, latent_size: int, condition_size: int = None):
        """
        Construct normalized flow model.

        Parameters
        ----------
        latent_size : int
            Dimensionality of latent space.
        condition_size : int or None, optional
            Conditioning feature size.

        Returns
        -------
        NormalizedFlowModel
        """

        return NormalizedFlowModel(
            config=self, latent_size=latent_size, condition_size=condition_size
        )


class NormalizedFlowModel(flowABC):
    """
    Composed normalizing flow model.
    """

    def __init__(
        self, config: NormalizedFlowConfig, latent_size, condition_size: int = None
    ):
        """
        Initialize normalized flow model.

        Parameters
        ----------
        config : NormalizedFlowConfig
        latent_size : int
        condition_size : int or None, optional
        """

        super().__init__()

        self.list_flows = config.list_flows
        self.flow_sample_size = config.flow_sample_size
        self.flows = []

        for step in self.list_flows:
            self.flows.append(step.get_model())

        self.condition_size = condition_size

        for ind, flow in enumerate(self.flows):
            self.flows[ind] = flow.build(
                dim=latent_size, condition_size=self.condition_size
            )

        self.flows = nn.ModuleList(self.flows)

    def forward(self, x, condition=None):
        """
        Apply forward transformation through all flow steps.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.
        condition : torch.Tensor or None, optional
            Conditioning input.

        Returns
        -------
        flowOutput
            Transformed samples and log-determinant.
        """

        bsz, _ = x.shape
        log_det = torch.zeros(bsz, device=x.device)
        for flow in self.flows:
            x, ld = flow.forward(x, condition=condition)
            log_det = log_det + ld

        return flowOutput(e_samples=x, log_det=log_det)

    def inverse(self, z, condition=None):
        """
        Apply inverse transformation through all flow steps.

        Parameters
        ----------
        z : torch.Tensor
            Latent samples.
        condition : torch.Tensor or None, optional

        Returns
        -------
        flowOutput
            Reconstructed samples and log-determinant.
        """
        bsz, _ = z.shape
        log_det = torch.zeros(bsz, device=z.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, condition=condition)
            log_det = log_det + ld
        x = z

        return flowOutput(e_samples=x, log_det=log_det)


class FCNN(nn.Module):
    """
    Fully connected neural network used in flow components.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.
    hidden_dim : int
        Hidden layer size.
    """

    def __init__(self, in_dim, out_dim, hidden_dim):
        """
        Initialize feedforward network.

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
        Compute forward pass.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return self.network(x)


@FlowSelector.register("maf")
class MAF(flowABC):
    """
    Masked autoregressive flow (MAF).
    """

    def __init__(self, hidden_dim=16, base_network=FCNN):
        """
        Initialize MAF.

        Parameters
        ----------
        hidden_dim : int, optional
        base_network : callable, optional
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_network = base_network

    def build(self, dim, condition_size=None):
        """
        Build MAF layers.

        Parameters
        ----------
        dim : int
            Input dimensionality.
        condition_size : int or None, optional

        Returns
        -------
        self
        """
        self.dim = dim

        layers = []
        added_features = condition_size if condition_size is not None else 0
        for i in range(1, self.dim):
            layers.append(self.base_network(i + added_features, 2, self.hidden_dim))
        self.layers = nn.Sequential(*layers)
        if condition_size is None:
            self.register_parameter(
                "initial_param", param=nn.Parameter(torch.Tensor(1, 2))
            )
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

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Progressive forward transformation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        condition : torch.Tensor or None, optional
            Conditioning input.

        Returns
        -------
        tuple
            (z, log_det)
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
        Apply autoregressive inverse transformation.

        Parameters
        ----------
        z : torch.Tensor
        condition : torch.Tensor or None, optional

        Returns
        -------
        tuple
            (x, log_det)
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

            log_det += alpha
        return x, log_det


@FlowSelector.register("realnvp")
class RealNVP(flowABC):
    """
    Real-valued non-volume preserving (RealNVP) flow.
    """

    def __init__(self, hidden_dim=16, base_network=FCNN):
        """
        Initialize RealNVP.

        Parameters
        ----------
        hidden_dim : int, optional
        base_network : callable, optional
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
        condition_size : int or None, optional

        Returns
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
        Apply forward coupling transformations.

        Parameters
        ----------
        x : torch.Tensor
        condition : torch.Tensor or None, optional

        Returns
        -------
        tuple
            (z, log_det)
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
        Apply inverse coupling transformations.

        Parameters
        ----------
        z : torch.Tensor
        condition : torch.Tensor or None, optional

        Returns
        -------
        tuple
            (x, log_det)
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

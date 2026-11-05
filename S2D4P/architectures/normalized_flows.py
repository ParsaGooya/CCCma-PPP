import torch
import torch.nn as nn
import torch.nn.init as init
import math

import dataclasses

from S2D4P.architectures.architecture_abc import flowABC
from S2D4P.core.selectors import FlowSelector


@dataclasses.dataclass
class flowOutput:
    """
    Container for the output of a normalizing-flow transformation.

    Parameters
    ----------
    e_samples : torch.Tensor
        Samples produced by the flow transformation.
    log_det : torch.Tensor
        Log-determinant of the transformation Jacobian for each sample.
    """

    e_samples: torch.Tensor
    log_det: torch.Tensor


@dataclasses.dataclass
class NormalizedFlowConfig:
    """
    Configuration for a sequence of normalizing-flow transformations.

    Parameters
    ----------
    list_flows : list[FlowSelector]
        Ordered flow configurations used to construct the normalizing-flow model.
    flow_sample_size : int
        Number of samples used when estimating flow-based quantities.
    """

    list_flows: list[FlowSelector]
    flow_sample_size: int = 5000

    def build(self, latent_size: int, condition_size: int = None):
        """
        Build a normalizing-flow model.

        Parameters
        ----------
        latent_size : int
            Number of latent dimensions transformed by the flow.
        condition_size : int | None
            Number of conditioning features supplied to each flow. If ``None``, the
            flow is unconditional.

        Returns
        -------
        NormalizedFlowModel
            Constructed sequence of normalizing-flow transformations.
        """
        return NormalizedFlowModel(
            config=self, latent_size=latent_size, condition_size=condition_size
        )


class NormalizedFlowModel(flowABC):
    """
    Apply a sequence of normalizing-flow transformations.

    Individual flow steps are evaluated in configuration order during the forward
    transformation and in reverse order during the inverse transformation.
    Log-determinants are accumulated across all steps.

    Parameters
    ----------
    config : NormalizedFlowConfig
        Configuration defining the sequence of flow transformations.
    latent_size : int
        Number of latent dimensions transformed by each flow.
    condition_size : int | None
        Number of conditioning features supplied to each flow. If ``None``, the
        flow is unconditional.
    """

    def __init__(
        self, config: NormalizedFlowConfig, latent_size, condition_size: int = None
    ):
        """
        Initialize a normalizing-flow model.

        Parameters
        ----------
        config : NormalizedFlowConfig
            Configuration defining the sequence of flow transformations.
        latent_size : int
            Number of latent dimensions transformed by each flow.
        condition_size : int | None
            Number of conditioning features supplied to each flow. If ``None``, the
            flow is unconditional.
        """
        super().__init__()

        self.list_flows = config.list_flows
        self.flow_sample_size = config.flow_sample_size
        self.flows = []

        for step in self.list_flows:
            self.flows.append(step.get_flow())

        self.condition_size = condition_size

        for ind, flow in enumerate(self.flows):
            self.flows[ind] = flow.build(
                dim=latent_size, condition_size=self.condition_size
            )

        self.flows = nn.ModuleList(self.flows)

    def forward(self, x, condition=None):
        """
        Apply the normalizing flows in forward order.

        Parameters
        ----------
        x : torch.Tensor
            Input samples with shape ``(batch, latent_size)``.
        condition : torch.Tensor | None
            Optional conditioning features supplied to each flow.

        Returns
        -------
        flowOutput
            Transformed samples and accumulated forward log-determinants.
        """
        bsz, _ = x.shape
        log_det = torch.zeros(bsz, device=x.device)
        for flow in self.flows:
            x, ld = flow.forward(x, condition=condition)
            log_det = log_det + ld

        return flowOutput(e_samples=x, log_det=log_det)

    def inverse(self, z, condition=None):
        """
        Apply the normalizing flows in reverse order.

        Parameters
        ----------
        z : torch.Tensor
            Transformed samples with shape ``(batch, latent_size)``.
        condition : torch.Tensor | None
            Optional conditioning features supplied to each inverse flow.

        Returns
        -------
        flowOutput
            Inverse-transformed samples and accumulated inverse log-determinants.
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
    Fully connected neural network used to parameterize flow transformations.

    The network contains two hidden layers with hyperbolic tangent activations.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    out_dim : int
        Number of output features.
    hidden_dim : int
        Number of features in each hidden layer.
    """

    def __init__(self, in_dim, out_dim, hidden_dim):
        """
        Initialize the fully connected network.

        Parameters
        ----------
        in_dim : int
            Number of input features.
        out_dim : int
            Number of output features.
        hidden_dim : int
            Number of features in each hidden layer.
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
        Apply the fully connected network.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Network output.
        """
        return self.network(x)


@FlowSelector.register("maf")
class MAF(flowABC):
    """
    Masked autoregressive flow transformation.

    Each latent dimension is transformed using location and log-scale parameters
    conditioned on preceding latent dimensions and, optionally, external
    conditioning features.

    Parameters
    ----------
    hidden_dim : int
        Number of hidden features used by the parameter networks.
    base_network : type[nn.Module]
        Network class used to predict location and log-scale parameters.
    """

    def __init__(
        self,
        hidden_dim=16,
        base_network=FCNN,
    ):
        """
        Initialize a masked autoregressive flow configuration.

        Parameters
        ----------
        hidden_dim : int
            Number of hidden features used by the parameter networks.
        base_network : type[nn.Module]
            Network class used to construct the autoregressive parameter networks.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_network = base_network
        self._conditional = False

    def build(self, dim, condition_size=None):
        """
        Build the autoregressive transformation.

        A parameter network is created for each latent dimension after the first.
        For an unconditional flow, the first dimension uses directly learned
        parameters. For a conditional flow, its parameters are predicted from the
        conditioning features.

        Parameters
        ----------
        dim : int
            Number of latent dimensions transformed by the flow.
        condition_size : int | None
            Number of conditioning features. If ``None``, the flow is unconditional.

        Returns
        -------
        MAF
            This flow instance with its parameter networks constructed.
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

        if added_features > 0:
            self._conditional = True

        return self

    def reset_parameters(self):
        """
        Initialize the learnable first-dimension parameters.

        The location and log-scale parameters are initialized from a uniform
        distribution.
        """
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x, condition=None):
        """
        Apply the autoregressive flow transformation.

        Each output dimension is standardized using parameters predicted from the
        preceding input dimensions. The transformed feature order is reversed before
        being returned.

        Parameters
        ----------
        x : torch.Tensor
            Input samples with shape ``(batch, dim)``.
        condition : torch.Tensor | None
            Conditioning features required by a conditional flow.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Transformed samples and the forward log-determinant for each sample.

        Raises
        ------
        ValueError
            If the flow was constructed as conditional but no conditioning features
            are provided.
        """
        if self._conditional:
            if condition is None:
                raise ValueError(
                    "The flow model is build as condition dependant but no condition "
                    "was provided in forward."
                )

        z = torch.zeros_like(x)
        log_det = torch.zeros(z.shape[0], device=x.device)
        for i in range(self.dim):
            if i == 0:
                if not self._conditional:
                    mu, alpha = self.initial_param.unbind(dim=-1)
                else:
                    out = self.initial_param(condition)
                    mu, alpha = out[:, 0], out[:, 1]
            else:
                if not self._conditional:
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
        Apply the inverse autoregressive flow transformation.

        Input feature order is reversed before each original dimension is
        reconstructed sequentially from previously reconstructed dimensions.

        Parameters
        ----------
        z : torch.Tensor
            Transformed samples with shape ``(batch, dim)``.
        condition : torch.Tensor | None
            Conditioning features required by a conditional flow.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Inverse-transformed samples and the inverse log-determinant for each
            sample.

        Raises
        ------
        ValueError
            If the flow was constructed as conditional but no conditioning features
            are provided.
        """
        if self._conditional:
            if condition is None:
                raise ValueError(
                    "The flow model is build as condition dependant but no condition "
                    "was provided in forward."
                )

        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0], device=z.device)
        z = z.flip(dims=(1,))
        for i in range(self.dim):
            if i == 0:
                if not self._conditional:
                    mu, alpha = self.initial_param.unbind(dim=-1)
                else:
                    out = self.initial_param(condition)
                    mu, alpha = out[:, 0], out[:, 1]
            else:
                if not self._conditional:
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
    Real-valued non-volume-preserving coupling flow.

    The latent representation is divided into two parts that are transformed
    sequentially using learned translation and log-scale functions. The coupling
    networks can optionally depend on external conditioning features.

    Parameters
    ----------
    hidden_dim : int
        Number of hidden features used by the coupling networks.
    base_network : type[nn.Module]
        Network class used to construct translation and log-scale functions.
    """

    def __init__(self, hidden_dim=16, base_network=FCNN):
        """
        Initialize a RealNVP coupling flow configuration.

        Parameters
        ----------
        hidden_dim : int
            Number of hidden features used by the coupling networks.
        base_network : type[nn.Module]
            Network class used to construct translation and log-scale functions.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_network = base_network
        self._conditional = False

    def build(self, dim, condition_size=None):
        """
        Build the RealNVP coupling networks.

        Translation and log-scale networks are constructed for both coupling
        transformations and optionally include conditioning features in their inputs.

        Parameters
        ----------
        dim : int
            Number of latent dimensions transformed by the flow.
        condition_size : int | None
            Number of conditioning features. If ``None``, the flow is unconditional.

        Returns
        -------
        RealNVP
            This flow instance with its coupling networks constructed.
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

        if added_features > 0:
            self._conditional = True

        return self

    def forward(self, x, condition=None):
        """
        Build the RealNVP coupling networks.

        Translation and log-scale networks are constructed for both coupling
        transformations and optionally include conditioning features in their inputs.

        Parameters
        ----------
        dim : int
            Number of latent dimensions transformed by the flow.
        condition_size : int | None
            Number of conditioning features. If ``None``, the flow is unconditional.

        Returns
        -------
        RealNVP
            This flow instance with its coupling networks constructed.
        """
        if self._conditional:
            if condition is None:
                raise ValueError(
                    "The flow model is build as condition dependant but no condition "
                    "was provided in forward."
                )

        lower, upper = x[:, : self.dim // 2].clone(), x[:, self.dim // 2 :].clone()
        if self._conditional:
            t1_transformed = self.t1(torch.cat([lower, condition], dim=-1))
            s1_transformed = self.s1(torch.cat([lower, condition], dim=-1))
        else:
            t1_transformed = self.t1(lower)
            s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        if self._conditional:
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
        Apply the inverse RealNVP coupling transformation.

        The two coupling transformations are inverted in reverse dependency order to
        recover the original latent samples.

        Parameters
        ----------
        z : torch.Tensor
            Transformed samples with shape ``(batch, dim)``.
        condition : torch.Tensor | None
            Conditioning features required by a conditional flow.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Inverse-transformed samples and the inverse log-determinant for each
            sample.

        Raises
        ------
        ValueError
            If the flow was constructed as conditional but no conditioning features
            are provided.
        """
        if self._conditional:
            if condition is None:
                raise ValueError(
                    "The flow model is build as condition dependant but no condition "
                    "was provided in forward."
                )

        lower, upper = z[:, : self.dim // 2].clone(), z[:, self.dim // 2 :].clone()
        if self._conditional:
            t2_transformed = self.t2(torch.cat([upper, condition], dim=-1))
            s2_transformed = self.s2(torch.cat([upper, condition], dim=-1))
        else:
            t2_transformed = self.t2(upper)
            s2_transformed = self.s2(upper)

        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)

        if self._conditional:
            t1_transformed = self.t1(torch.cat([lower, condition], dim=-1))
            s1_transformed = self.s1(torch.cat([lower, condition], dim=-1))
        else:
            t1_transformed = self.t1(lower)
            s1_transformed = self.s1(lower)

        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + torch.sum(-s2_transformed, dim=1)
        return x, log_det

import torch
import torch.nn as nn
import torch.nn.init as init
import math

import dataclasses

from cccma_ppp.architectures.models_abc import flowABC
from cccma_ppp.core.selectors import FlowSelector


@dataclasses.dataclass
class flowOutput:
    """
    Container for the output of a normalizing flow transformation.

    Parameters
    ----------
    e_samples : torch.Tensor
        Samples produced by the flow transformation.
    log_det : torch.Tensor
        Log absolute determinant of the transformation Jacobian for each sample.
    """

    e_samples: torch.Tensor
    log_det: torch.Tensor


@dataclasses.dataclass
class NormalizedFlowConfig:
    """
    Configuration for a sequence of normalizing flow transformations.

    Parameters
    ----------
    list_flows : list[FlowSelector]
        Ordered flow configurations composing the normalizing flow model.
    flow_sample_size : int
        Number of samples associated with flow-based sampling operations.
    """

    list_flows: list[FlowSelector]
    flow_sample_size: int = 5000

    def build(self, latent_size: int, condition_size: int = None):
        """
        Build a normalizing flow model from this configuration.

        Parameters
        ----------
        latent_size : int
            Dimensionality of the latent representation transformed by the flow.
        condition_size : int | None
            Dimensionality of the optional conditioning representation.

        Returns
        -------
        NormalizedFlowModel
            Constructed sequence of normalizing flow transformations.
        """
        return NormalizedFlowModel(
            config=self, latent_size=latent_size, condition_size=condition_size
        )


class NormalizedFlowModel(flowABC):
    """
    Compose multiple normalizing flow transformations.

    Individual flows are applied sequentially in the forward direction and in
    reverse order during inversion. Log-determinant contributions are accumulated
    across transformations.

    Parameters
    ----------
    config : NormalizedFlowConfig
        Configuration defining the ordered flow transformations.
    latent_size : int
        Dimensionality of the latent representation transformed by the flow.
    condition_size : int | None
        Dimensionality of the optional conditioning representation.
    """

    def __init__(
        self, config: NormalizedFlowConfig, latent_size, condition_size: int = None
    ):
        """
        Initialize a sequence of normalizing flow transformations.

        Parameters
        ----------
        config : NormalizedFlowConfig
            Configuration defining the ordered flow transformations.
        latent_size : int
            Dimensionality of the latent representation transformed by each flow.
        condition_size : int | None
            Dimensionality of the optional conditioning representation.
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
        Apply the sequence of flow transformations.

        Parameters
        ----------
        x : torch.Tensor
            Input samples with shape ``(N, D)``.
        condition : torch.Tensor | None
            Optional conditioning representation supplied to each flow.

        Returns
        -------
        flowOutput
            Transformed samples and the accumulated log absolute determinant of the
            forward transformations.
        """
        bsz, _ = x.shape
        log_det = torch.zeros(bsz, device=x.device)
        for flow in self.flows:
            x, ld = flow.forward(x, condition=condition)
            log_det = log_det + ld

        return flowOutput(e_samples=x, log_det=log_det)

    def inverse(self, z, condition=None):
        """
        Apply the inverse flow transformations in reverse order.

        Parameters
        ----------
        z : torch.Tensor
            Latent samples with shape ``(N, D)``.
        condition : torch.Tensor | None
            Optional conditioning representation supplied to each inverse flow.

        Returns
        -------
        flowOutput
            Inverse-transformed samples and the accumulated log absolute determinant
            of the inverse transformations.
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
        Initialize a fully connected flow network.

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
            Output produced by the network.
        """
        return self.network(x)


@FlowSelector.register("maf")
class MAF(flowABC):
    """
    Masked autoregressive flow transformation.

    Each feature is transformed using affine parameters derived from preceding
    features and, when configured, an additional conditioning representation.

    Parameters
    ----------
    hidden_dim : int
        Number of hidden features in each parameter network.
    base_network : type[nn.Module]
        Network class used to predict the affine transformation parameters.
    """

    def __init__(
        self,
        hidden_dim=16,
        base_network=FCNN,
    ):
        """
        Initialize a masked autoregressive flow.

        Parameters
        ----------
        hidden_dim : int
            Number of hidden features in each parameter network.
        base_network : type[nn.Module]
            Network class used to construct the affine parameter networks.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_network = base_network
        self._conditional = False

    def build(self, dim, condition_size=None):
        """
        Build the autoregressive transformation for a specified dimension.

        A separate parameter network is constructed for each feature after the first.
        When conditioning is enabled, the conditioning representation is included in
        the input to each parameter network.

        Parameters
        ----------
        dim : int
            Dimensionality of the representation transformed by the flow.
        condition_size : int | None
            Dimensionality of the optional conditioning representation.

        Returns
        -------
        MAF
            This flow instance with its transformation networks constructed.
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
        Initialize the unconditional affine parameters for the first feature.
        """
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x, condition=None):
        """
        Apply the autoregressive forward transformation.

        Each feature is standardized using affine parameters determined by preceding
        input features. The transformed feature order is reversed before returning.

        Parameters
        ----------
        x : torch.Tensor
            Input samples with shape ``(N, D)``.
        condition : torch.Tensor | None
            Conditioning representation required when the flow was built as
            conditional.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Transformed samples and the log absolute determinant of the forward
            transformation for each sample.

        Raises
        ------
        ValueError
            If the flow was built as conditional but no conditioning representation
            is provided.
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
        Apply the inverse autoregressive transformation.

        The latent feature order is first restored, after which input features are
        reconstructed sequentially from the affine parameters determined by
        previously reconstructed features.

        Parameters
        ----------
        z : torch.Tensor
            Latent samples with shape ``(N, D)``.
        condition : torch.Tensor | None
            Conditioning representation required when the flow was built as
            conditional.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Reconstructed samples and the log absolute determinant of the inverse
            transformation for each sample.

        Raises
        ------
        ValueError
            If the flow was built as conditional but no conditioning representation
            is provided.
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
    Real-valued non-volume preserving flow transformation.

    The input representation is divided into two parts that are transformed
    through successive affine coupling operations, optionally conditioned on an
    additional representation.

    Parameters
    ----------
    hidden_dim : int
        Number of hidden features in each coupling-network parameterization.
    base_network : type[nn.Module]
        Network class used to predict translation and scale parameters.
    """

    def __init__(self, hidden_dim=16, base_network=FCNN):
        """
        Initialize a RealNVP affine coupling transformation.

        Parameters
        ----------
        hidden_dim : int
            Number of hidden features in each coupling-network parameterization.
        base_network : type[nn.Module]
            Network class used to construct translation and scale networks.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_network = base_network
        self._conditional = False

    def build(self, dim, condition_size=None):
        """
        Build the affine coupling networks for a specified dimension.

        Parameters
        ----------
        dim : int
            Dimensionality of the representation transformed by the flow.
        condition_size : int | None
            Dimensionality of the optional conditioning representation.

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
        Apply the forward affine coupling transformation.

        The input is divided into two feature groups. Each group is transformed using
        translation and scale parameters computed from the other group and, when
        configured, the conditioning representation.

        Parameters
        ----------
        x : torch.Tensor
            Input samples with shape ``(N, D)``.
        condition : torch.Tensor | None
            Conditioning representation required when the flow was built as
            conditional.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Transformed samples and the log absolute determinant of the forward
            transformation for each sample.

        Raises
        ------
        ValueError
            If the flow was built as conditional but no conditioning representation
            is provided.
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
        Apply the inverse affine coupling transformation.

        The two transformed feature groups are reconstructed in reverse coupling
        order using the corresponding translation and scale networks.

        Parameters
        ----------
        z : torch.Tensor
            Latent samples with shape ``(N, D)``.
        condition : torch.Tensor | None
            Conditioning representation required when the flow was built as
            conditional.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Reconstructed samples and the log absolute determinant of the inverse
            transformation for each sample.

        Raises
        ------
        ValueError
            If the flow was built as conditional but no conditioning representation
            is provided.
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

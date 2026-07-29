"""Site-conditioned scalar-vector message passing for mode proposal."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def channel_linear(
    vectors: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("oc,ncd->nod", weight, vectors)


class RadialBasis(nn.Module):
    def __init__(self, count: int, cutoff: float) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        centers = torch.linspace(0.0, cutoff, count)
        self.register_buffer("centers", centers)
        spacing = float(centers[1] - centers[0])
        self.gamma = 1.0 / (spacing * spacing)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(
            -self.gamma * (distance[:, None] - self.centers[None, :]).square()
        )
        envelope = 0.5 * (torch.cos(math.pi * distance / self.cutoff) + 1.0)
        envelope = envelope * (distance < self.cutoff)
        return basis * envelope[:, None]


class ScalarVectorMessageLayer(nn.Module):
    """One O(3)-equivariant scalar/vector message-passing block."""

    def __init__(
        self,
        hidden_dim: int,
        radial_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scalar_source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.vector_source = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.vector_self = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.orthogonal_(self.vector_source)
        nn.init.orthogonal_(self.vector_self)
        self.filters = nn.Sequential(
            nn.Linear(radial_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.Tanh(),
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.vector_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.scalar_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        edge_index: torch.Tensor,
        edge_unit: torch.Tensor,
        radial_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        filters = self.filters(radial_features).reshape(-1, 4, self.hidden_dim)
        source_scalar = self.scalar_source(scalars[source])
        source_vector = channel_linear(vectors[source], self.vector_source)
        radial_projection = torch.sum(source_vector * edge_unit[:, None, :], dim=-1)
        scalar_message = (
            filters[:, 0] * source_scalar + filters[:, 1] * radial_projection
        )
        vector_message = (
            filters[:, 2, :, None] * source_vector
            + filters[:, 3, :, None] * source_scalar[:, :, None] * edge_unit[:, None, :]
        )

        scalar_aggregate = torch.zeros_like(scalars)
        vector_aggregate = torch.zeros_like(vectors)
        scalar_aggregate.index_add_(0, destination, scalar_message)
        vector_aggregate.index_add_(0, destination, vector_message)
        degree = torch.zeros(
            len(scalars),
            dtype=scalars.dtype,
            device=scalars.device,
        )
        degree.index_add_(
            0,
            destination,
            torch.ones(
                len(destination),
                dtype=scalars.dtype,
                device=scalars.device,
            ),
        )
        normalization = degree.clamp_min(1.0).sqrt()
        scalar_aggregate = scalar_aggregate / normalization[:, None]
        vector_aggregate = vector_aggregate / normalization[:, None, None]

        vector_invariant = torch.linalg.vector_norm(vector_aggregate, dim=-1)
        scalar_delta = self.scalar_update(
            torch.cat([scalars, scalar_aggregate, vector_invariant], dim=-1)
        )
        updated_scalars = self.scalar_norm(scalars + scalar_delta)
        mixed_vectors = channel_linear(vectors, self.vector_self)
        gate = self.vector_gate(updated_scalars)[:, :, None]
        updated_vectors = vectors + 0.25 * mixed_vectors + gate * vector_aggregate
        rms = torch.mean(updated_vectors.square(), dim=(1, 2), keepdim=True)
        updated_vectors = updated_vectors / torch.sqrt(1.0 + rms)
        return updated_scalars, updated_vectors


class ConditionedModeProposer(nn.Module):
    """Return K normalized, translation-free fields for each graph."""

    def __init__(
        self,
        hidden_dim: int = 32,
        layers: int = 3,
        radial_basis: int = 16,
        cutoff: float = 2.5,
        hypotheses: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hypotheses = hypotheses
        self.radial = RadialBasis(radial_basis, cutoff)
        self.scalar_embedding = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                ScalarVectorMessageLayer(hidden_dim, radial_basis, dropout)
                for _ in range(layers)
            ]
        )
        self.vector_readout = nn.Parameter(torch.empty(hypotheses, hidden_dim))
        nn.init.orthogonal_(self.vector_readout)
        self.atom_gate = nn.Linear(hidden_dim, hypotheses)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        graph_ptr = batch["graph_ptr"].to(dtype=torch.long)
        graph_sizes = graph_ptr[1:] - graph_ptr[:-1]
        if len(graph_sizes) == 0 or torch.any(graph_sizes <= 0):
            raise ValueError("graph_ptr must describe nonempty graphs")
        if not torch.all(graph_sizes == graph_sizes[0]):
            raise ValueError("one batch must contain graphs with the same atom count")
        graph_count = int(len(graph_sizes))
        atoms_per_graph = int(graph_sizes[0].item())
        if int(graph_ptr[-1].item()) != len(batch["species"]):
            raise ValueError("graph_ptr and node arrays disagree")

        species = F.one_hot(batch["species"] - 1, num_classes=2).to(
            batch["site_score"].dtype
        )
        scalar_input = torch.cat(
            [
                species,
                batch["site_score"][:, None],
                batch["coordination"][:, None],
            ],
            dim=1,
        )
        scalars = self.scalar_embedding(scalar_input)
        vectors = torch.zeros(
            (len(scalars), self.hidden_dim, 3),
            dtype=scalars.dtype,
            device=scalars.device,
        )
        distance = batch["edge_distances"]
        unit = batch["edge_vectors"] / distance.clamp_min(1.0e-8)[:, None]
        radial = self.radial(distance)
        for layer in self.layers:
            scalars, vectors = layer(
                scalars,
                vectors,
                batch["edge_index"],
                unit,
                radial,
            )

        vectors = vectors.reshape(
            graph_count,
            atoms_per_graph,
            self.hidden_dim,
            3,
        )
        gates = torch.sigmoid(self.atom_gate(scalars)).reshape(
            graph_count,
            atoms_per_graph,
            self.hypotheses,
        )
        fields = torch.einsum("kh,bnhd->bknd", self.vector_readout, vectors)
        fields = fields * (0.25 + gates.permute(0, 2, 1)[:, :, :, None])
        fields = fields - fields.mean(dim=2, keepdim=True)
        norm = torch.linalg.vector_norm(
            fields.reshape(graph_count, self.hypotheses, -1),
            dim=2,
            keepdim=True,
        )
        return fields / norm.clamp_min(1.0e-10)[:, :, None]

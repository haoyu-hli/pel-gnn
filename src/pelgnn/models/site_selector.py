"""Invariant periodic message passing for per-atom rearrangement scores."""

from __future__ import annotations

import math

import torch
from torch import nn


class RadialBasis(nn.Module):
    def __init__(self, count: int, cutoff: float) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, count)
        self.register_buffer("centers", centers)
        self.width = float(count) / cutoff
        self.cutoff = float(cutoff)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        radial = torch.exp(
            -(self.width**2) * (distance[:, None] - self.centers[None, :]).square()
        )
        envelope = 0.5 * (torch.cos(math.pi * distance / self.cutoff) + 1.0)
        return radial * envelope[:, None]


class RadialMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, radial_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + radial_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        source: torch.Tensor,
        destination: torch.Tensor,
        radial: torch.Tensor,
    ) -> torch.Tensor:
        message = self.message(torch.cat([hidden[source], radial], dim=1))
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, destination, message)
        degree = torch.zeros(len(hidden), dtype=hidden.dtype, device=hidden.device)
        degree.index_add_(
            0,
            destination,
            torch.ones(
                len(destination),
                dtype=hidden.dtype,
                device=hidden.device,
            ),
        )
        aggregate = aggregate / degree.clamp_min(1.0).sqrt()[:, None]
        update = self.update(torch.cat([hidden, aggregate], dim=1))
        return self.norm(hidden + update)


class RadialSiteGNN(nn.Module):
    """Assign an invariant rearrangement score to every atom."""

    def __init__(
        self,
        hidden_dim: int = 64,
        layers: int = 3,
        radial_dim: int = 16,
        cutoff: float = 2.5,
    ) -> None:
        super().__init__()
        self.species_embedding = nn.Embedding(3, hidden_dim)
        self.radial = RadialBasis(radial_dim, cutoff)
        self.layers = nn.ModuleList(
            [RadialMessageLayer(hidden_dim, radial_dim) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        species: torch.Tensor,
        edge_source: torch.Tensor,
        edge_destination: torch.Tensor,
        edge_distance: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.species_embedding(species)
        radial = self.radial(edge_distance)
        for layer in self.layers:
            hidden = layer(
                hidden,
                edge_source,
                edge_destination,
                radial,
            )
        return self.head(hidden).squeeze(1)

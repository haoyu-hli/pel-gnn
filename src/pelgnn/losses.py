"""Sign- and order-aware objectives for multi-hypothesis mode prediction."""

from __future__ import annotations

import torch


def normalize_fields(raw: torch.Tensor) -> torch.Tensor:
    """Project vector fields to zero translation and unit global norm."""

    raw = raw - raw.mean(dim=2, keepdim=True)
    norm = torch.linalg.vector_norm(
        raw.reshape(raw.shape[0], raw.shape[1], -1),
        dim=2,
        keepdim=True,
    )
    return raw / norm.clamp_min(1.0e-10)[:, :, None]


def mode_set_loss(
    predicted: torch.Tensor,
    targets: list[torch.Tensor],
    activity_weight: float = 0.20,
    diversity_weight: float = 0.05,
    diversity_margin: float = 0.80,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cover known target sets without treating extra hypotheses as false."""

    if predicted.ndim != 4 or predicted.shape[-1] != 3:
        raise ValueError("predicted must have shape (graphs, heads, atoms, 3)")
    if len(targets) != len(predicted):
        raise ValueError("one target tensor is required per graph")

    coverage_terms = []
    activity_terms = []
    diversity_terms = []
    for graph, target in enumerate(targets):
        if target.ndim != 3 or target.shape[1:] != predicted.shape[2:]:
            raise ValueError("target shape is incompatible with predictions")
        if len(target) == 0:
            raise ValueError("every graph requires at least one target")

        proposal = predicted[graph]
        overlap = torch.abs(torch.einsum("knd,mnd->km", proposal, target))
        best_overlap, best_head = torch.max(overlap, dim=0)
        coverage_terms.append(torch.mean(1.0 - best_overlap.square()))

        selected = proposal[best_head]
        target_activity = target.square().sum(dim=2)
        proposal_activity = selected.square().sum(dim=2)
        bhattacharyya = torch.sum(
            torch.sqrt((target_activity * proposal_activity).clamp_min(0.0)),
            dim=1,
        )
        activity_terms.append(torch.mean(1.0 - bhattacharyya))

        gram = torch.abs(torch.einsum("knd,lnd->kl", proposal, proposal))
        off_diagonal = ~torch.eye(
            len(proposal),
            dtype=torch.bool,
            device=proposal.device,
        )
        diversity_terms.append(
            torch.mean(torch.relu(gram[off_diagonal] - diversity_margin).square())
        )

    coverage = torch.stack(coverage_terms).mean()
    activity = torch.stack(activity_terms).mean()
    diversity = torch.stack(diversity_terms).mean()
    total = coverage + activity_weight * activity + diversity_weight * diversity
    return total, {
        "coverage": coverage,
        "activity": activity,
        "diversity": diversity,
    }

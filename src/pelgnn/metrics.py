"""Evaluation metrics for site rankings and vector proposals."""

from __future__ import annotations

import numpy as np
import torch


def best_absolute_overlap(
    proposals: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return the best sign-invariant proposal overlap per target."""

    if proposals.ndim != 3 or targets.ndim != 3:
        raise ValueError("proposals and targets must have shape (items, atoms, 3)")
    overlap = torch.abs(torch.einsum("knd,mnd->km", proposals, targets))
    return overlap.max(dim=0).values


def top_k_activity(
    scores: np.ndarray,
    target_probability: np.ndarray,
    k: int,
) -> dict[str, float]:
    """Measure activity captured by the top-k ranked atoms."""

    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target_probability, dtype=np.float64)
    if scores.ndim != 1 or target.shape != scores.shape:
        raise ValueError("scores and target_probability must be aligned vectors")
    if not 0 < k <= len(scores):
        raise ValueError("k must fall within the atom count")
    if np.any(target < 0.0) or not np.isfinite(target).all():
        raise ValueError("target probabilities must be finite and nonnegative")
    total = float(target.sum())
    if total <= 0.0:
        raise ValueError("target probability must have positive mass")
    target = target / total
    selected = np.argsort(scores, kind="stable")[-k:]
    captured = float(target[selected].sum())
    random_expectation = k / len(scores)
    return {
        "captured_activity": captured,
        "random_expectation": random_expectation,
        "uplift": captured / random_expectation,
    }

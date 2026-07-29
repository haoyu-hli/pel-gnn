"""Checkpoint inference on compact landscape samples."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from pelgnn.data import LandscapeSample
from pelgnn.metrics import best_absolute_overlap, top_k_activity
from pelgnn.models import ConditionedModeProposer, RadialSiteGNN

def load_checkpoint_pair(
    site_checkpoint: Union[str, Path],
    mode_checkpoint: Union[str, Path],
    device: torch.device,
) -> tuple[RadialSiteGNN, ConditionedModeProposer]:
    """Load the parent-block-held-out model pair."""

    site_model = RadialSiteGNN(
        hidden_dim=64,
        layers=3,
        radial_dim=16,
        cutoff=2.5,
    ).to(device)
    mode_model = ConditionedModeProposer(
        hidden_dim=32,
        layers=3,
        radial_basis=16,
        cutoff=2.5,
        hypotheses=8,
        dropout=0.05,
    ).to(device)
    site_state = torch.load(
        Path(site_checkpoint),
        map_location=device,
        weights_only=True,
    )
    mode_state = torch.load(
        Path(mode_checkpoint),
        map_location=device,
        weights_only=True,
    )
    site_model.load_state_dict(site_state)
    mode_model.load_state_dict(mode_state)
    site_model.eval()
    mode_model.eval()
    return site_model, mode_model


@torch.no_grad()
def evaluate_sample(
    sample_path: Union[str, Path],
    site_checkpoint: Union[str, Path],
    mode_checkpoint: Union[str, Path],
    device_name: str = "cpu",
) -> dict:
    """Run both models and return complete per-minimum diagnostics."""

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    sample = LandscapeSample.load(sample_path)
    site_model, mode_model = load_checkpoint_pair(
        site_checkpoint,
        mode_checkpoint,
        device,
    )

    records = []
    all_overlap = []
    site_uplift = []
    site_capture = []
    proposal_translation = []
    proposal_norm_error = []
    for minimum in range(len(sample)):
        graph = sample.neighbor_graph(minimum, cutoff=2.5)
        site_inputs = sample.site_tensors(minimum, graph, device)
        site_logits = site_model(*site_inputs)
        mode_batch = sample.mode_batch(
            minimum,
            graph,
            site_logits.cpu().numpy(),
            device,
        )
        proposals = mode_model(mode_batch)[0]
        targets = torch.as_tensor(
            sample.targets(minimum),
            dtype=torch.float32,
            device=device,
        )
        overlaps = best_absolute_overlap(proposals, targets)
        activity = top_k_activity(
            site_logits.cpu().numpy(),
            sample.site_target_probability[minimum],
            k=10,
        )
        translation = torch.linalg.vector_norm(proposals.mean(dim=1), dim=1)
        norms = torch.linalg.vector_norm(proposals.reshape(len(proposals), -1), dim=1)

        overlap_values = overlaps.cpu().numpy()
        all_overlap.extend(overlap_values.tolist())
        site_uplift.append(activity["uplift"])
        site_capture.append(activity["captured_activity"])
        proposal_translation.extend(translation.cpu().numpy().tolist())
        proposal_norm_error.extend(torch.abs(norms - 1.0).cpu().numpy().tolist())
        records.append(
            {
                "minimum_uid": str(sample.minimum_uid[minimum]),
                "network_id": str(sample.network_id[minimum]),
                "energy": float(sample.minimum_energy[minimum]),
                "atoms": sample.atoms_per_minimum,
                "directed_edges": int(graph.edge_index.shape[1]),
                "known_incident_modes": len(overlap_values),
                "mean_best_abs_overlap": float(overlap_values.mean()),
                "max_best_abs_overlap": float(overlap_values.max()),
                "site_top_10_activity": activity["captured_activity"],
                "site_top_10_uplift": activity["uplift"],
            }
        )

    return {
        "sample": Path(sample_path).name,
        "selection": "first eight stable N03 minimum indices",
        "minimum_count": len(sample),
        "target_mode_count": sample.target_count,
        "aggregate": {
            "mean_best_abs_overlap": float(np.mean(all_overlap)),
            "median_best_abs_overlap": float(np.median(all_overlap)),
            "fraction_overlap_ge_0p10": float(np.mean(np.asarray(all_overlap) >= 0.10)),
            "mean_site_top_10_activity": float(np.mean(site_capture)),
            "mean_site_top_10_uplift": float(np.mean(site_uplift)),
            "max_proposal_translation": float(np.max(proposal_translation)),
            "max_proposal_norm_error": float(np.max(proposal_norm_error)),
        },
        "per_minimum": records,
    }

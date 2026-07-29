"""Loading and validation for the compact held-out research sample."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch

from pelgnn.geometry import (
    NeighborGraph,
    build_neighbor_graph,
    standardize,
)


@dataclass(frozen=True)
class LandscapeSample:
    """A compact collection of minima and incident unstable-mode targets."""

    minimum_uid: np.ndarray
    network_id: np.ndarray
    minimum_energy: np.ndarray
    species: np.ndarray
    box: np.ndarray
    positions: np.ndarray
    site_target_probability: np.ndarray
    target_offsets: np.ndarray
    target_saddle_uid: np.ndarray
    target_modes: np.ndarray

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LandscapeSample":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as source:
            required = {
                "minimum_uid",
                "network_id",
                "minimum_energy",
                "species",
                "box",
                "positions",
                "site_target_probability",
                "target_offsets",
                "target_saddle_uid",
                "target_modes",
            }
            missing = required - set(source.files)
            if missing:
                raise ValueError(f"sample is missing arrays: {sorted(missing)}")
            sample = cls(
                minimum_uid=source["minimum_uid"].copy(),
                network_id=source["network_id"].copy(),
                minimum_energy=source["minimum_energy"].copy(),
                species=source["species"].copy(),
                box=source["box"].copy(),
                positions=source["positions"].copy(),
                site_target_probability=source["site_target_probability"].copy(),
                target_offsets=source["target_offsets"].copy(),
                target_saddle_uid=source["target_saddle_uid"].copy(),
                target_modes=source["target_modes"].copy(),
            )
        sample.validate()
        return sample

    def __len__(self) -> int:
        return len(self.minimum_uid)

    @property
    def atoms_per_minimum(self) -> int:
        return int(self.positions.shape[1])

    @property
    def target_count(self) -> int:
        return len(self.target_modes)

    def validate(self) -> None:
        minima = len(self.minimum_uid)
        if minima == 0:
            raise ValueError("sample contains no minima")
        if self.network_id.shape != (minima,):
            raise ValueError("network IDs are not aligned with minima")
        if self.minimum_energy.shape != (minima,):
            raise ValueError("minimum energies are not aligned")
        if self.positions.ndim != 3 or self.positions.shape[2] != 3:
            raise ValueError("positions must have shape (minima, atoms, 3)")
        atoms = self.positions.shape[1]
        if self.species.shape != (minima, atoms):
            raise ValueError("species and positions are not aligned")
        if self.box.shape != (minima, 3):
            raise ValueError("box must have shape (minima, 3)")
        if self.site_target_probability.shape != (minima, atoms):
            raise ValueError("site targets are not aligned")
        if self.target_offsets.shape != (minima + 1,):
            raise ValueError("target offsets must have minima + 1 entries")
        if (
            int(self.target_offsets[0]) != 0
            or int(self.target_offsets[-1]) != len(self.target_modes)
            or np.any(np.diff(self.target_offsets) <= 0)
        ):
            raise ValueError("every minimum needs a nonempty target range")
        if self.target_modes.ndim != 3:
            raise ValueError("target modes must have shape (targets, atoms, 3)")
        if self.target_modes.shape[1:] != (atoms, 3):
            raise ValueError("target modes use a different atom shape")
        if self.target_saddle_uid.shape != (len(self.target_modes),):
            raise ValueError("target saddle IDs are not aligned")
        if not np.isfinite(self.positions).all():
            raise ValueError("positions contain non-finite values")
        if not np.isfinite(self.target_modes).all():
            raise ValueError("target modes contain non-finite values")
        if np.any(self.box <= 0.0):
            raise ValueError("box lengths must be positive")
        if not np.isin(self.species, [1, 2]).all():
            raise ValueError("only species IDs 1 and 2 are expected")

        site_mass = self.site_target_probability.sum(axis=1)
        if np.max(np.abs(site_mass - 1.0)) > 2.0e-5:
            raise ValueError("site target probabilities must sum to one")
        mode_translation = np.linalg.norm(self.target_modes.mean(axis=1), axis=1)
        mode_norm = np.linalg.norm(
            self.target_modes.reshape(len(self.target_modes), -1),
            axis=1,
        )
        if mode_translation.max() > 2.0e-6:
            raise ValueError("target modes are not translation free")
        if np.max(np.abs(mode_norm - 1.0)) > 2.0e-5:
            raise ValueError("target modes are not normalized")

    def targets(self, minimum_index: int) -> np.ndarray:
        start, stop = self.target_offsets[minimum_index : minimum_index + 2]
        return self.target_modes[int(start) : int(stop)]

    def neighbor_graph(
        self,
        minimum_index: int,
        cutoff: float = 2.5,
    ) -> NeighborGraph:
        return build_neighbor_graph(
            self.positions[minimum_index],
            self.box[minimum_index],
            cutoff,
        )

    def site_tensors(
        self,
        minimum_index: int,
        graph: NeighborGraph,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.as_tensor(
                self.species[minimum_index],
                dtype=torch.long,
                device=device,
            ),
            torch.as_tensor(
                graph.edge_index[0],
                dtype=torch.long,
                device=device,
            ),
            torch.as_tensor(
                graph.edge_index[1],
                dtype=torch.long,
                device=device,
            ),
            torch.as_tensor(
                graph.edge_distances,
                dtype=torch.float32,
                device=device,
            ),
        )

    def mode_batch(
        self,
        minimum_index: int,
        graph: NeighborGraph,
        site_logits: np.ndarray,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        atoms = self.atoms_per_minimum
        return {
            "species": torch.as_tensor(
                self.species[minimum_index],
                dtype=torch.long,
                device=device,
            ),
            "site_score": torch.as_tensor(
                standardize(site_logits),
                dtype=torch.float32,
                device=device,
            ),
            "coordination": torch.as_tensor(
                standardize(graph.coordination),
                dtype=torch.float32,
                device=device,
            ),
            "edge_index": torch.as_tensor(
                graph.edge_index,
                dtype=torch.long,
                device=device,
            ),
            "edge_distances": torch.as_tensor(
                graph.edge_distances,
                dtype=torch.float32,
                device=device,
            ),
            "edge_vectors": torch.as_tensor(
                graph.edge_vectors,
                dtype=torch.float32,
                device=device,
            ),
            "graph_ptr": torch.as_tensor(
                [0, atoms],
                dtype=torch.long,
                device=device,
            ),
        }

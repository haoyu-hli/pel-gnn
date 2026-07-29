"""Periodic geometry utilities used by the atomic graph models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class NeighborGraph:
    """A directed periodic neighbor graph."""

    edge_index: np.ndarray
    edge_vectors: np.ndarray
    edge_distances: np.ndarray
    coordination: np.ndarray


def minimum_image(displacement: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Wrap Cartesian displacements into an orthorhombic minimum image."""

    displacement = np.asarray(displacement)
    box = np.asarray(box, dtype=displacement.dtype)
    if displacement.shape[-1] != 3 or box.shape != (3,):
        raise ValueError("displacement must end in 3 and box must have shape (3,)")
    if np.any(box <= 0.0):
        raise ValueError("box lengths must be positive")
    return displacement - box * np.rint(displacement / box)


def build_neighbor_graph(
    positions: np.ndarray,
    box: np.ndarray,
    cutoff: float,
) -> NeighborGraph:
    """Build a directed, no-self-edge graph under periodic boundaries."""

    positions = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (atoms, 3)")
    if box.shape != (3,) or np.any(box <= 0.0):
        raise ValueError("box must contain three positive lengths")
    if not np.isfinite(positions).all() or not np.isfinite(box).all():
        raise ValueError("positions and box must be finite")
    if not 0.0 < cutoff < 0.5 * float(box.min()) + 1.0e-12:
        raise ValueError("cutoff must be positive and no larger than half the box")

    wrapped = np.mod(positions, box)
    pairs = cKDTree(wrapped, boxsize=box).query_pairs(cutoff, output_type="ndarray")
    if len(pairs) == 0:
        raise ValueError("neighbor graph contains no edges")

    source = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int64)
    destination = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int64)
    vectors = minimum_image(positions[source] - positions[destination], box).astype(
        np.float32
    )
    distances = np.linalg.norm(vectors, axis=1).astype(np.float32)

    order = np.lexsort((destination, source))
    source = source[order]
    destination = destination[order]
    vectors = vectors[order]
    distances = distances[order]

    coordination = np.bincount(destination, minlength=len(positions)).astype(np.float32)
    return NeighborGraph(
        edge_index=np.stack([source, destination]),
        edge_vectors=vectors,
        edge_distances=distances,
        coordination=coordination,
    )


def standardize(values: np.ndarray, epsilon: float = 1.0e-6) -> np.ndarray:
    """Return zero-mean, unit-scale values without amplifying constants."""

    values = np.asarray(values, dtype=np.float32)
    scale = max(float(values.std()), epsilon)
    return ((values - values.mean()) / scale).astype(np.float32)

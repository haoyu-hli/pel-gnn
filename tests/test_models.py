import numpy as np
import torch

from pelgnn.geometry import (
    build_neighbor_graph,
    standardize,
)
from pelgnn.models import ConditionedModeProposer, RadialSiteGNN


def random_orthogonal(seed, reflection=False):
    rng = np.random.default_rng(seed)
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    determinant = np.linalg.det(matrix)
    if (reflection and determinant > 0) or (
        not reflection and determinant < 0
    ):
        matrix[:, 0] *= -1
    return matrix


def small_periodic_system():
    positions = np.asarray(
        [
            [5.0, 5.0, 5.0],
            [5.8, 5.1, 5.0],
            [4.7, 5.7, 5.2],
            [5.2, 4.4, 5.5],
            [4.4, 4.7, 4.6],
            [5.7, 5.7, 4.5],
        ],
        dtype=np.float64,
    )
    species = np.asarray([1, 1, 2, 1, 2, 1])
    box = np.asarray([10.0, 10.0, 10.0])
    return positions, species, box


def make_mode_batch(graph, species, site_score):
    return {
        "species": torch.as_tensor(species, dtype=torch.long),
        "site_score": torch.as_tensor(standardize(site_score), dtype=torch.float32),
        "coordination": torch.as_tensor(
            standardize(graph.coordination), dtype=torch.float32
        ),
        "edge_index": torch.as_tensor(graph.edge_index, dtype=torch.long),
        "edge_distances": torch.as_tensor(graph.edge_distances, dtype=torch.float32),
        "edge_vectors": torch.as_tensor(graph.edge_vectors, dtype=torch.float32),
        "graph_ptr": torch.as_tensor([0, len(species)], dtype=torch.long),
    }


def test_site_selector_is_rotation_invariant():
    torch.manual_seed(3)
    positions, species, box = small_periodic_system()
    center = box / 2.0
    rotation = random_orthogonal(seed=9)
    rotated = (positions - center) @ rotation + center
    graph = build_neighbor_graph(positions, box, cutoff=2.5)
    rotated_graph = build_neighbor_graph(rotated, box, cutoff=2.5)
    model = RadialSiteGNN().eval()

    def predict(current_graph):
        return model(
            torch.as_tensor(species, dtype=torch.long),
            torch.as_tensor(current_graph.edge_index[0], dtype=torch.long),
            torch.as_tensor(current_graph.edge_index[1], dtype=torch.long),
            torch.as_tensor(current_graph.edge_distances, dtype=torch.float32),
        )

    torch.testing.assert_close(
        predict(graph),
        predict(rotated_graph),
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_mode_proposer_is_orthogonally_equivariant():
    torch.manual_seed(5)
    positions, species, box = small_periodic_system()
    center = box / 2.0
    transform = random_orthogonal(seed=11, reflection=True)
    transformed_positions = (positions - center) @ transform + center
    graph = build_neighbor_graph(positions, box, cutoff=2.5)
    transformed_graph = build_neighbor_graph(
        transformed_positions,
        box,
        cutoff=2.5,
    )
    site_score = np.linspace(-1.0, 1.0, len(species))
    batch = make_mode_batch(graph, species, site_score)
    transformed_batch = make_mode_batch(
        transformed_graph,
        species,
        site_score,
    )
    model = ConditionedModeProposer().eval()

    output = model(batch)
    transformed_output = model(transformed_batch)
    expected = output @ torch.as_tensor(transform, dtype=output.dtype)

    torch.testing.assert_close(
        transformed_output,
        expected,
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        output.mean(dim=2),
        torch.zeros_like(output.mean(dim=2)),
        atol=2.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(
            output.reshape(1, output.shape[1], -1),
            dim=2,
        ),
        torch.ones((1, output.shape[1])),
        atol=2.0e-6,
        rtol=2.0e-6,
    )

import numpy as np

from pelgnn.geometry import build_neighbor_graph, minimum_image


def test_minimum_image_wraps_across_periodic_boundary():
    displacement = np.asarray([[9.8, -9.8, 0.0], [-5.1, 5.1, 0.2]])
    wrapped = minimum_image(displacement, np.asarray([10.0] * 3))
    expected = np.asarray([[-0.2, 0.2, 0.0], [4.9, -4.9, 0.2]])
    np.testing.assert_allclose(wrapped, expected)


def test_neighbor_graph_is_directed_and_translation_invariant():
    positions = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [9.9, 0.2, 0.3],
            [5.0, 5.0, 5.0],
        ]
    )
    box = np.asarray([10.0, 10.0, 10.0])
    graph = build_neighbor_graph(positions, box, cutoff=0.5)
    shifted = build_neighbor_graph(
        np.mod(positions + np.asarray([2.7, -1.3, 4.1]), box),
        box,
        cutoff=0.5,
    )

    np.testing.assert_array_equal(
        graph.edge_index,
        np.asarray([[0, 1], [1, 0]]),
    )
    np.testing.assert_allclose(
        graph.edge_vectors[0],
        -graph.edge_vectors[1],
        atol=1.0e-7,
    )
    np.testing.assert_array_equal(
        graph.edge_index,
        shifted.edge_index,
    )
    np.testing.assert_allclose(
        graph.edge_distances,
        shifted.edge_distances,
        atol=1.0e-6,
    )

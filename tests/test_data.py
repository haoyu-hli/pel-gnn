import numpy as np


def test_held_out_sample_shape(held_out_sample):
    assert len(held_out_sample) == 8
    assert held_out_sample.atoms_per_minimum == 1000
    assert held_out_sample.target_count == 33
    assert set(held_out_sample.network_id.tolist()) == {"N03"}


def test_every_minimum_has_normalized_targets(held_out_sample):
    for minimum in range(len(held_out_sample)):
        targets = held_out_sample.targets(minimum)
        assert len(targets) > 0
        np.testing.assert_allclose(
            targets.mean(axis=1),
            0.0,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            np.linalg.norm(targets.reshape(len(targets), -1), axis=1),
            1.0,
            atol=2.0e-5,
        )

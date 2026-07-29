import numpy as np
import torch

from pelgnn.inference import load_checkpoint_pair


def test_checkpoint_inference(repository_root, held_out_sample):
    checkpoint_root = repository_root / "checkpoints"
    site_model, mode_model = load_checkpoint_pair(
        checkpoint_root / "site_selector.pt",
        checkpoint_root / "mode_proposer.pt",
        torch.device("cpu"),
    )
    graph = held_out_sample.neighbor_graph(0)

    with torch.no_grad():
        site_logits = site_model(
            *held_out_sample.site_tensors(
                0,
                graph,
                torch.device("cpu"),
            )
        )
        proposals = mode_model(
            held_out_sample.mode_batch(
                0,
                graph,
                site_logits.numpy(),
                torch.device("cpu"),
            )
        )

    assert site_logits.shape == (1000,)
    assert proposals.shape == (1, 8, 1000, 3)
    assert np.isfinite(site_logits.numpy()).all()
    torch.testing.assert_close(
        proposals.mean(dim=2),
        torch.zeros_like(proposals.mean(dim=2)),
        atol=2.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(
            proposals.reshape(1, 8, -1),
            dim=2,
        ),
        torch.ones((1, 8)),
        atol=2.0e-6,
        rtol=2.0e-6,
    )

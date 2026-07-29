import torch

from pelgnn.losses import mode_set_loss, normalize_fields
from pelgnn.metrics import best_absolute_overlap


def test_mode_set_loss_ignores_target_sign_and_order():
    generator = torch.Generator().manual_seed(31)
    predicted = normalize_fields(torch.randn(1, 4, 7, 3, generator=generator))
    targets = normalize_fields(torch.randn(1, 3, 7, 3, generator=generator))[0]

    original, terms = mode_set_loss(predicted, [targets])
    transformed, transformed_terms = mode_set_loss(
        predicted,
        [-targets.flip(0)],
    )

    torch.testing.assert_close(original, transformed)
    for name in terms:
        torch.testing.assert_close(
            terms[name],
            transformed_terms[name],
        )


def test_best_overlap_is_sign_invariant():
    generator = torch.Generator().manual_seed(47)
    proposals = normalize_fields(torch.randn(1, 5, 8, 3, generator=generator))[0]
    targets = normalize_fields(torch.randn(1, 2, 8, 3, generator=generator))[0]

    torch.testing.assert_close(
        best_absolute_overlap(proposals, targets),
        best_absolute_overlap(proposals, -targets),
    )

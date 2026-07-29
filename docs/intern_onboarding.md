# Intern onboarding

Start by reading the root README and `technical_notes.md`, then run:

```bash
python -m pip install -e ".[dev]"
pelgnn-demo
pytest
```

Trace one minimum through:

1. periodic graph construction in `geometry.py`;
2. atom ranking in `site_selector.py`;
3. batch construction in `data.py`;
4. vector proposal in `mode_proposer.py`;
5. overlap evaluation in `metrics.py`.

A useful first task is to add one diagnostic without retraining the models.
Good choices are proposal diversity, per-minimum overlap, sensitivity to the
neighbor cutoff, or top-k target activity.

Keep comparisons on the same minimum IDs, do not use unobserved transitions as
negative examples, and add a small test when changing periodic geometry or
equivariance code.

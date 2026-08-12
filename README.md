# PEL-GNN

This project asks whether an atomic graph neural network can provide useful
starting directions for transition-state searches in a glass.

The system is a 1,000-atom Kob-Andersen binary Lennard-Jones mixture. Local
minima are connected by independently checked first-order saddles. The neural
network does not replace ARTn or the force calculation; it predicts where a
rearrangement is likely to occur and proposes full-system displacement fields.

## Models

The pipeline has two parts:

1. `RadialSiteGNN` ranks atoms by rearrangement activity using species and
   periodic neighbor distances.
2. `ConditionedModeProposer` takes the site scores and proposes eight
   normalized, translation-free vector fields.

The second model uses scalar and vector message passing, so its outputs rotate
and reflect with the atomic coordinates. Its loss compares each known unstable
mode with its best proposal using absolute overlap; mode sign and proposal
order are arbitrary.

## Results

The retrospective model evaluation holds out all cooling-rate siblings from
one parent glass.

| Metric | Learned model | Baseline |
|---|---:|---:|
| Site top-10 activity uplift | 8.083x random | 6.389x minimum soft modes |
| Site top-10 core recall | 12.88% | 5.55% minimum soft modes |
| Mean best mode overlap | 0.10697 | 0.05239 site-weighted random |
| Mean best mode overlap | 0.10697 | 0.03264 global random |

On the exact common subset, the eight lowest physical modes remain the best
full-vector baseline: 0.11491 mean overlap compared with 0.07927 for the
learned fields. The model is better at locating active atoms than reproducing
every component of the unstable mode.

The search policy was then tested prospectively. Each source received either
48 global-random ARTn attempts or a 48-attempt hybrid of 20 learned
directions, 20 site-weighted directions, and 8 conventional mixed-radius
attempts. ARTn and the physical certification procedure were unchanged.

| Prospective study | Hybrid channels | Random channels | Yield ratio |
|---|---:|---:|---:|
| First frozen test, 2 parent liquids | 906 | 668 | 1.356x |
| Independent verification, 10 parents | 900 | 593 | 1.518x |
| Retrained test, 6 untouched parents | 104 | 71 | 1.465x |

The main independent verification used 120 unseen source minima from 30
networks and was positive in all ten parent liquids. Its parent-bootstrap
interval for the yield ratio was 1.368 to 1.683, and the hybrid policy produced
1.254x as many certified channels per core-hour. After one retraining round,
a separately frozen test was positive in all six untouched parents and
retained a 1.168x compute-efficiency advantage.

These results validate the hybrid policy as a unit. They do not show that the
GNN alone replaces conventional or site-based proposals; the useful behavior
is their complementarity, with exact physics still deciding which searches
succeed.

## Run the example

The repository includes eight N03 minima and a checkpoint pair trained without
the N03-N05 parent block. This original held-out pair is kept for a clean,
small demonstration; the later prospective studies used frozen checkpoint
ensembles and are reported as research results rather than bundled here.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pelgnn-demo
pytest
```

The example rebuilds periodic neighbor graphs and evaluates 33 incident saddle
modes on CPU.

## Repository layout

```text
src/pelgnn/        graph construction, models, losses, and inference
data/example.npz   small held-out example
checkpoints/       one site model and one mode model
results/           compact summary of retrospective and prospective results
tests/             geometry, symmetry, data, and inference checks
docs/              technical notes and intern onboarding
```

The raw ARTn campaigns, Hessian jobs, large graph caches, and scheduler output
remain in the working research archive. See
[`docs/technical_notes.md`](docs/technical_notes.md) for data and evaluation
details.

## Author

Haoyu Li  
haoyuli@umich.edu

Licensed under the BSD 3-Clause License.

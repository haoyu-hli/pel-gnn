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

The main comparison holds out all cooling-rate siblings from one parent glass.

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

The follow-up ARTn test used 120 previously unused minima, with 20 from each
network. A 48-attempt hybrid policy combined 20 learned directions, 20
site-weighted random directions, and 8 conventional mixed-radius ARTn
attempts. It was compared with 48 global-random attempts on every minimum.

| Prospective metric | Hybrid policy | Global random |
|---|---:|---:|
| Distinct certified saddles per source | 7.55 | 5.57 |
| Attempts per source | 48 | 48 |

The hybrid policy produced 1.356x the saddle yield, and its mean advantage was
positive in all six networks. The result supports using the learned fields as
a complementary proposal source, not as a replacement for the physical and
site-based methods. The six networks come from two parent glasses, so broader
transfer should be tested on additional independently prepared systems.

## Run the example

The repository includes eight N03 minima and a checkpoint pair trained without
the N03-N05 parent block.

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

# Technical notes

## Dataset

The full research dataset is organized by glass sample in six independent
network folders, N03-N08. States within one network share atom identities and
a periodic cell; there are no transitions between different network folders.
The public repository contains only the small example needed to run inference.

The frozen training view contains:

| Quantity | Count |
|---|---:|
| Minima | 1,688 |
| Physical saddles | 1,712 |
| Directed endpoint targets | 3,424 |
| Equally exposed source minima | 300 |
| Source minima with observed site labels | 291 |
| ARTn attempts per exposed source | 64 |

Each minimum stores species, periodic coordinates, cell dimensions, energy,
network identity, and exposure status. Each saddle stores its energy,
coordinates, unstable mode, numerical residuals, and two relaxed endpoints.

A candidate enters the learning set after stationary-point refinement,
force reproduction, a one-negative-mode Hessian check, two-sided relaxation,
endpoint matching, and deduplication. An unsuccessful finite ARTn search is not
used as evidence that a physical transition is absent.

Training and evaluation roles are stored with the physical records rather than
in separate dataset versions. This keeps the frozen pre-frontier training data
separate from the later prospective campaign without duplicating the atomic
structures.

## Atomic graphs

Atoms are connected within 2.5 reduced length units under periodic boundary
conditions. Edge vectors follow the minimum-image convention.

The site model uses species and radial edge features. Its target is the
normalized atomwise squared amplitude summed over observed incident unstable
modes.

The mode model receives species, coordination, and frozen site-model scores.
It carries invariant scalar channels and equivariant vector channels through
three message-passing layers and returns eight fields. Each field is projected
to zero mean translation and normalized over all atoms and Cartesian
components.

The mode loss contains three terms:

- best-proposal coverage of each known target;
- agreement between atomwise activity distributions;
- a small penalty for duplicate proposals.

## Evaluation

Cooling-rate siblings from the same parent glass remain in one partition. The
primary result holds out a complete parent block and averages three seeds
within each split before giving the two parent blocks equal weight.

Controls are global random fields, site-weighted random fields, species-only
site rankings, and the eight lowest physical modes of each minimum.

The included example contains the first eight N03 minima and every certified
incident unstable mode for those endpoints. N03 is absent from the training
and validation networks of the included checkpoint pair. The example is meant
to exercise inference; the full grouped result is recorded in
`results/summary.json`.

## Prospective ARTn test

The final search test used 120 previously unused source minima, 20 from each
network. The policy was frozen before these outcomes were inspected. Every
source received:

- 20 learned proposals;
- 20 site-weighted random proposals;
- 8 conventional mixed-radius ARTn attempts;
- 48 global-random control attempts.

The first three components form one 48-attempt hybrid arm. It found 7.55
distinct strictly certified saddles per source, compared with 5.57 for the
equal-budget global-random arm, a yield ratio of 1.356. The mean paired
difference was 1.98 saddles per source; a network-and-root bootstrap interval
was 1.28 to 2.71. The difference was positive in all six networks.

This test supports a hybrid search policy. It does not show that the learned
proposer should replace site-weighted or conventional ARTn directions. The
network-level sample is also small: the six networks come from two parent
glasses.

## Units and conventions

- reduced Lennard-Jones units;
- periodic orthorhombic cells;
- species IDs 1 and 2;
- zero-based array indices;
- unstable modes have zero net translation and unit global norm;
- eigenvector sign is arbitrary.

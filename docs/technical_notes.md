# Technical notes

## Dataset

The original released learning dataset is organized by glass sample in six
independent network folders, N03-N08. States within one network share atom
identities and a periodic cell; there are no transitions between different
network folders. The public repository contains only the small example needed
to run inference.

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

A later retraining view added independent-parent results without changing the
model architectures. It contains 36 networks from 12 parent liquids, 2,958
minima, 2,882 physical saddles, 5,764 directed endpoint targets, and 411 site
samples. Site supervision uses only channels found by the matched global-random
control; guided discoveries do not become probability-like labels. Six other
parents were kept untouched for the next prospective test.

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

## Prospective ARTn tests

The search policy was frozen before outcomes were inspected. Every source in
all three tests received:

- 20 learned proposals;
- 20 site-weighted random proposals;
- 8 conventional mixed-radius ARTn attempts;
- 48 global-random control attempts.

The first frozen test used 120 previously unused minima in six networks. The
hybrid found 906 distinct certified source-saddle channels versus 668 for the
control, a yield ratio of 1.356. It established a useful policy signal but the
six networks represented only two parent glasses.

The main independent verification therefore used another 120 unseen minima in
30 networks from ten parent liquids absent from training. The hybrid found 900
channels versus 593 for global random: 7.500 versus 4.942 per source and a
1.518 yield ratio. The parent-bootstrap intervals were 1.925 to 3.158 for the
paired gain and 1.368 to 1.683 for the yield ratio. The mean gain was positive
in all ten parents. Including proposal inference, ARTn, refinement, Hessians,
and connectivity checks, the policy produced 26.833 versus 21.394 channels per
core-hour, a 1.254 ratio.

The unchanged architectures were then retrained as a three-pair ensemble on
36 networks from 12 parents. A separately frozen test used all three pristine
quenches from each of six untouched parents, 18 sources in total. It found 104
channels versus 71 for global random, a 1.465 ratio; the gain was positive in
all six parents with an exact one-sided sign-flip p-value of 0.015625. Its
compute-efficiency ratio was 1.168.

These experiments support the complete hybrid search policy, not replacement
of site-weighted or conventional ARTn by the learned proposer alone. The
parent liquid is the statistical unit; cooling-rate siblings and source minima
are correlated observations within a parent.

## What did not work

A later parameter-free spectral filter projected the structural proposals into
the source-minimum Hessian basis and improved retrospective vector overlap on
six independent parents. In a separately frozen equal-budget ARTn test,
however, it found 60 certified channels while the unchanged E(3) proposals
found 91. The branch was closed without tuning on the holdout. This is why the
project treats overlap as a diagnostic and prospective certified-channel yield
as the decision metric.

## Included checkpoints

The bundled checkpoint pair predates the later retraining and keeps N03 held
out, which makes the included N03 example a clean inference demonstration.
The retrained prospective policy used an ensemble of three checkpoint pairs;
those working research checkpoints and their private training data are not
included in this compact repository.

## Units and conventions

- reduced Lennard-Jones units;
- periodic orthorhombic cells;
- species IDs 1 and 2;
- zero-based array indices;
- unstable modes have zero net translation and unit global norm;
- eigenvector sign is arbitrary.

# Organoid Shape VAE Handoff Plan

This document is a technical and scientific handoff for building a VAE-style
model of intestinal organoid morphology from the `organograph.skeleton`
representation.  It records the biological assumptions, modeling choices,
decisions that were rejected, staged implementation plan, expected data format,
failure modes, diagnostics, and relevant literature.

The intended reader is a fresh coding agent or collaborator who has access to
this repository but not the prior design discussion.

## Biological Context

The project studies 3D outer-membrane meshes of intestinal organoids during
development.

Working biological assumptions:

- Each organoid has one main body/villus.
- The body can have multiple primary appendices, roughly up to 6.
- A primary appendix can be either a crypt or a branch.
- A crypt can be bulged/open or budded/constricted.
- A branch is connected to the body through a neck.
- A branch has daughter crypts, typically at least 2 and up to roughly 3-4.
- Branches are shallow: branches should not recursively have branch daughters.
- Crypts are the most defining morphological features and show the greatest
  variation.
- Crypts in one organoid are not independent.  Mechanical coordination through
  the tissue and lumen can produce symmetric or coordinated crypt growth.
- Structural development is assumed to be irreversible: once a crypt is
  established, it should not disappear in a developmental trajectory.

Scientific goals:

- Learn an interpretable morphospace of organoid shape.
- Keep the number of latent variables as small as possible.
- Preserve biological readability of topology and skeleton geometry.
- Later use time-course data to study developmental trajectories through the
  learned morphospace.
- Separate major biological structure from finer primitive/detail variation so
  the project can be built and debugged in stages.

## Existing Shape Representation

The skeleton package represents shape in three layers:

1. Skeleton topology and geometry:
   a biologically motivated graph/tree with nodes such as body, attachment,
   constriction, neck, branch, crypt, bend, and tip.

2. Primitive fitting:
   body/branch blobs and crypt tubes fitted to skeleton-defined components.

3. Blending:
   deterministic visualization-only connectors between fitted primitives.

Only the first two layers should be VAE-facing.  Blending is for rendering and
should not be encoded unless the project intentionally moves to visual surface
losses.

Use the following conceptual variables:

```text
C = context
    developmental age, cell count, treatment, batch, dataset, well, etc.

T = discrete topology
    number of primary appendices, crypt/branch labels, budded/bulged labels,
    branch daughter counts, parent/host relations.

S = core continuous skeleton morphology
    body scale and coarse body shape, attachment directions, node positions,
    crypt lengths, bend angles, branch geometry, neck/constriction locations.

P = primitive detail
    tube radii/taper profiles, detailed body/branch superellipsoid parameters,
    primitive residual refinements.
```

The boundary between `S` and `P` is biological, not purely technical.  Body
scale, body elongation, and any primitive-derived value needed for growth
decisions should be promoted into `S`, even if it comes from the fitted body or
crypt primitive.  Detailed radii and superellipsoid refinements can remain in
`P`.

## Central Modeling Choice

Do not force the entire problem into one smooth Euclidean latent space where
crypt existence is represented as a soft continuous coordinate.  Since topology
is biologically meaningful and irreversible, use a hybrid/stratified
morphospace:

```text
discrete topology T
  +
continuous morphology within topology T
  +
irreversible transitions between allowed topologies
```

Within one topology, morphology can change continuously.  Between topologies,
development proceeds through discrete events such as adding a crypt, forming a
constriction, or splitting a crypt into a branch with daughter crypts.

However, avoid fully independent VAEs for each topology.  That would create
unrelated latent spaces and make developmental trajectories hard to compare.
The preferred compromise is:

```text
shared low-dimensional morphospace coordinate u
  +
explicit discrete topology T
  +
shared compositional decoder modules
```

One useful factorization for snapshots is:

```text
p(T, S, P | C)
  = p(T | C)
    p(u)
    p(S | u, T, C)
    p(P | S, u, T, C)
```

But do not train this whole model first.  Start with observed `T` and `S`.

## Decisions: What To Do

### Keep Topology Discrete

`T` should be explicit and observed in the first model stages.  It should not
initially be hidden inside a continuous latent variable.

Reasons:

- Topology is the most important biological structure.
- Crypt creation and branch formation are event-like.
- Irreversible developmental constraints are easier with discrete topology.
- Failures are easier to diagnose when topology is not entangled with geometry.

### Use Shared Compositional Decoders

Do not make one complete decoder per exact topology.  Instead, build reusable
modules:

```text
body decoder
primary crypt decoder
branch decoder
daughter crypt decoder
primitive refinement decoder
```

The topology tells the model which modules to apply and how to connect them.

Reasons:

- Similar topologies share statistical strength.
- Rare topologies are less likely to overfit.
- The decoder matches the shallow biological grammar.
- The model can encode/decode the whole organoid while sharing crypt logic.

### Encode The Organoid As A Whole

Crypts should not be treated as independent samples.  Use a global latent
coordinate and, later, attention/message passing over appendices so the model
can learn symmetry and coordinated crypt growth.

### Separate Skeleton Core From Primitive Detail

Develop the first model on `T + S`.  Add `P` later.

Reasons:

- Smaller first problem.
- Skeleton geometry is the main biological signal.
- Primitive details can be noisy and may distract the latent space.
- It is easier to benchmark skeleton reconstruction before adding radii/taper
  details.

Exception:

- Include compact body size/shape descriptors in `S`.
- Include primitive-derived constriction descriptors in `S` if they are needed
  for topology/growth modeling and are reliably measured from primitives.

### Add Topology Generation Later

The first representation model should use:

```text
q(u | S, T, C)
p(S | u, T, C)
```

Only after that works should the project add:

```text
p(T | C)
```

or later:

```text
p(T_next | T, u, S, C)
```

The first topology generator should be a constrained grammar or irreversible
transition model, not a generic graph VAE.

## Decisions: What Not To Do First

### Do Not Start With A Full Junction Tree VAE

Junction Tree VAE is a useful analogy because it generates a scaffold before
details.  But organoid topology is shallower and more constrained than
molecular graph generation.

Reasons to avoid copying it directly:

- The biological grammar is known and small.
- A generic tree decoder adds unnecessary failure modes.
- Validity can be enforced directly.
- The available dataset may not support a large graph generator.

### Do Not Train Independent VAEs Per Topology

Separate VAEs per topology are tempting, but they create severe practical
problems:

- Sparse topology classes train poorly.
- Similar topologies fail to share information.
- Latent axes can rotate differently in each model.
- Cross-topology developmental trajectories become fragile.

Small topology-specific adapters may be useful later, but not fully independent
models.

### Do Not Encode Visualization Blends

Blend attachments are deterministic visualization helpers.  Keep them outside
the learned representation.

### Do Not Let Context Replace Latent Morphology

Context variables such as age and cell count are useful but dangerous.  If the
decoder can reconstruct average age-specific shape from `C` alone, the latent
coordinate `u` may become uninformative.

Always compare:

```text
model without C
model with C
model where C is predicted from u as an auxiliary diagnostic
```

## Portable Export Format

Raw skeleton/primitive exports are written by:

```text
src/organograph/skeleton/export.py
scripts/export_skeleton_primitives.py
```

The export is an interchange format, not the final packed VAE tensor format.
Downstream VAE projects should build their own versioned packing step from
this raw export.

Per-organoid directory:

```text
{output_root}/{timepoint}/{label_uid}/
    shape.json
    shape_nodes.csv
    shape_edges.csv
    shape_primitives.csv
    shape_arrays.npz
```

Top-level files:

```text
{output_root}/
    README.md
    manifest.csv
    failures.log
    run_settings_*.json
```

### `shape.json`

The full JSON payload contains:

```text
schema_version
created_at_utc
metadata
summary
graph
tables
skeletonization
primitive_fit
```

Important subfields:

- `metadata`: dataset, timepoint, well, organoid id, label uid, mesh path, and
  any extra metadata passed to the export.
- `summary`: node/edge/crypt/primitive counts.
- `graph`: faithful `SkeletonGraph.to_dict()` serialization.
- `tables.nodes`, `tables.edges`, `tables.primitives`: flat table records also
  written as CSV.
- `skeletonization.config`: detection and graph-building settings.
- `skeletonization.detections`: crypt detections used to build the graph.
- `primitive_fit.config`: primitive fitting settings.
- `primitive_fit.component_summary`: compact summary of component arrays.

The JSON payload intentionally keeps primitive-specific parameters in generic
dictionaries, so new primitive families can be exported without schema changes.

### CSV Tables

`shape_nodes.csv`:

```text
node_id,node_type,crypt_id,x,y,z,metadata_json,primitive_attachment_json
```

`shape_edges.csv`:

```text
edge_id,source,target,edge_type,crypt_id,metadata_json,primitive_attachment_json
```

`shape_primitives.csv`:

```text
attachment_id,attachment_type,primitive_type,target_ids_json,fit_error,
parameters_json,derived_parameters_json,residuals_json,metadata_json
```

The JSON columns are deliberate.  They keep the stable biological table schema
small while allowing the rapidly evolving primitive/skeleton internals to pass
through unchanged.

### `shape_arrays.npz`

The NPZ file is convenient for Python/NumPy loading:

```text
node_ids
node_types
node_crypt_ids
node_positions
edge_ids
edge_sources
edge_targets
edge_types
edge_crypt_ids
edge_index
primitive_attachment_ids
primitive_types
primitive_attachment_types
primitive_target_ids_json
primitive_parameters_json
primitive_derived_parameters_json
primitive_residuals_json
primitive_fit_error
```

Minimal loading:

```python
import json
import numpy as np

with open("day4p5/day4p5_B02_100/shape.json") as f:
    payload = json.load(f)

arrays = np.load("day4p5/day4p5_B02_100/shape_arrays.npz")
node_positions = arrays["node_positions"]
edge_index = arrays["edge_index"]
```

## Staged Implementation Roadmap

Each stage should produce artifacts that can be tested and reused.

### Stage 0: Raw Export And Non-Neural Baselines

Goal:
create clean, versioned data for downstream VAE work and establish baselines.

Tasks:

- Run `scripts/export_skeleton_primitives.py`.
- Validate the portable export on a small subset.
- Build a separate VAE packing script that maps raw export to `T/S/P/C`.
- Canonicalize coordinates:
  body center at origin, body orientation as canonical axes, body scale
  normalized.
- Promote compact body descriptors into `S`.
- Define deterministic ordering or matching for appendices.
- Build PCA/probabilistic PCA/factor-analysis baselines on `S`.
- Reconstruct skeletons from packed `T + S` before training neural models.

Deliverables:

- Raw export directory with manifest.
- Packed representation version.
- Data dictionary for packed features.
- PCA/factor-analysis notebook.
- Skeleton reconstruction visual checks.

Likely failure modes:

- Body canonical frame flips.
- Crypt ordering unstable across nearby shapes.
- Feature normalization inconsistent.
- Export includes duplicate derived variables as independent targets.

Diagnostics:

- Plot every feature distribution before/after normalization.
- Render sorted appendix slots with labels.
- Compare packed-to-reconstructed skeletons with original skeletons.

### Stage 1: Topology-Conditioned Skeleton VAE

Goal:
learn compact continuous morphology given observed topology.

Model:

```text
q(u | S, T)
p(S | u, T)
```

Add `C` only after benchmarking without it.

Architecture:

- Body encoder.
- Shared crypt/branch encoders.
- Masked pooling or attention over appendices.
- Low-dimensional global latent `u`, initially 2, 4, 6, or 8 dimensions.
- Shared compositional decoder conditioned on `T`.
- Masked losses for absent slots.

Deliverables:

- Training script.
- Reconstruction metrics by topology and feature group.
- Latent traversal renderer.
- Comparison to PCA/factor-analysis reconstruction.

Failure modes:

- Posterior collapse.
- Decoder ignores `u`.
- Latent axes rotate and become uninterpretable.
- Rare topology classes reconstruct poorly.
- Slot ordering artifacts dominate.

Diagnostics:

- Monitor KL per latent dimension.
- Render latent traversals for common topologies.
- Correlate latent axes with explicit descriptors.
- Train with and without `T`.
- Train with and without `C`.
- Report metrics by topology frequency.

### Stage 2: Explicit Compositional Grammar Decoder

Goal:
replace any overly flat padded decoder with modules that follow the shallow
organoid grammar.

Grammar:

```text
body
  -> primary appendix slots
primary appendix
  -> crypt | branch
branch
  -> daughter crypt slots
crypt
  -> bulged | budded
```

Tasks:

- Implement shared body, crypt, branch, and daughter-crypt modules.
- Route through modules using observed `T`.
- Add sibling communication with attention/message passing.
- Keep module-level tests for every valid topology case.

Deliverables:

- Compositional decoder.
- Comparison to Stage 1 decoder.
- Ablations with/without sibling attention.

Failure modes:

- Over-engineering before data quality is known.
- Shared modules underfit daughter crypts or branches.
- Attention improves reconstruction but hurts interpretability.

Diagnostics:

- Compare primary crypt and daughter crypt errors.
- Compare symmetric versus asymmetric organoids.
- Inspect common and rare topologies separately.

### Stage 3: Primitive Detail Model

Goal:
add primitive parameters after skeleton modeling is stable.

Start with:

```text
p(P | S, T)
```

Only add a primitive latent if residual primitive variation matters:

```text
q(z_P | P, S, T)
p(P | z_P, S, T)
```

Deliverables:

- Primitive reconstruction metrics.
- Rendered primitive reconstructions.
- Ablation measuring how much `P` adds beyond `T + S`.

Failure modes:

- Primitive fitting noise dominates.
- Tube radius/taper variables have incompatible scales.
- Primitive detail consumes capacity that should explain skeleton morphology.

Diagnostics:

- Compare direct regression versus residual VAE.
- Filter by primitive fit residuals.
- Plot primitive errors against skeleton errors.

### Stage 4: Topology Probability Model

Goal:
model or generate topology without yet doing time-course dynamics.

Start with a simple grammar:

```text
p(number_of_primary_appendices | C)
p(primary_type_i | count, C)
p(crypt_subtype_i | primary_type_i, C)
p(number_of_daughters | branch_i, C)
p(daughter_subtype_j | branch_i, C)
```

Candidate models:

- Empirical frequencies by timepoint.
- Multinomial/ordinal regression.
- Small autoregressive grammar network.

Deliverables:

- Topology likelihood/accuracy.
- Calibration plots.
- Samples from `p(T | C)`.

Failure modes:

- Rare topologies ignored.
- Batch/timepoint leakage.
- Valid but implausible samples.

Diagnostics:

- Compare to empirical topology frequencies.
- Confusion matrices by timepoint.
- Calibration by dataset and condition.

### Stage 5: Irreversible Transition Model

Goal:
use future time-course data to model developmental trajectories.

State:

```text
(T_t, u_t, S_t, optional P_t, C_t)
```

Within-topology dynamics:

```text
du/dt = f_T(u, C)
```

Topology event:

```text
T -> T'
hazard lambda_{T -> T'}(u, S, C)
```

Allowed topology transitions should form a directed acyclic graph, for example:

```text
body only -> add crypt
bulged crypt -> budded crypt
crypt -> branch with daughter crypts
add primary crypt
```

No reverse transitions should be allowed unless later biological evidence
requires them.

Deliverables:

- Explicit transition graph.
- Event hazard model.
- Time-course trajectory visualizations.
- Irreversibility checks.

Failure modes:

- Too little longitudinal data.
- Snapshot-trained `u` is not dynamically meaningful.
- Event timing is ambiguous between sampled frames.

Diagnostics:

- Start with discrete-time transition models.
- Compare transition probabilities to observed event frequencies.
- Check whether `u` changes smoothly when topology is unchanged.

## Packed Representation Guidance

The raw export should be transformed into a model-specific packed
representation.  This should have its own version string.

Suggested topology fields:

```text
num_primary_appendices
primary_slot_mask[K_primary]
primary_type[K_primary]              # absent, crypt, branch
primary_crypt_subtype[K_primary]     # bulged, budded, not crypt
branch_mask[K_primary]
num_daughter_crypts[K_primary]
daughter_slot_mask[K_primary, K_daughter]
daughter_crypt_subtype[K_primary, K_daughter]
host/parent indices
```

Suggested `S` fields:

- Body scale.
- Body axis ratios or elongation.
- Body canonical frame metadata.
- Primary attachment directions in body frame.
- Branch neck and branch center positions.
- Crypt attachment/constriction/crypt/tip coordinates in host-relative frame.
- Crypt length.
- Crypt bend angle or centerline control offsets.
- Constriction location and constriction ratio, if biologically important.
- Neighbor/crowding descriptors if useful for transition hazards.

Suggested `P` fields:

- `r_neck`, `r_body`, `r_taper`, `r_tip`.
- `s_body`, `s_taper`, optional `s_constriction`.
- Detailed body/branch asymmetric superellipsoid axes.
- Superellipsoid exponents.
- Primitive residual details if needed.

Suggested `C` fields:

- Dataset.
- Timepoint or continuous age.
- Well/plate metadata.
- Cell count.
- Treatment/condition.
- Batch/acquisition metadata.

Diagnostics such as primitive residuals, skeleton confidence, flags, and
component vertex counts should be stored separately and used for filtering,
weighting, or quality control.

## Training Principles

Use strong baselines first:

- PCA on `S`.
- Probabilistic PCA or factor analysis.
- Descriptor-only regressions against age/cell count.
- Empirical topology frequency models.

Keep `u` small:

```text
u_dim = 2, 4, 6, 8
```

Use feature-specific normalization and store normalization statistics.  Use
masked losses for all optional slots.  Use beta-VAE or capacity annealing and
monitor KL per latent dimension.

Evaluate by:

- topology class;
- timepoint;
- body-only/spherical samples;
- unbranched crypt samples;
- budded/constricted samples;
- branched samples;
- common versus rare topology classes.

## Common Failure Modes And Responses

### Reconstruction Is Poor

Possible causes:

- Wrong normalization.
- Bad canonical frame.
- Slot ordering instability.
- Noisy primitive targets.
- Incorrect masks.

Responses:

- Train on `T + S` only.
- Remove primitives and derived descriptors.
- Train on one common topology as a sanity check.
- Compare to PCA reconstruction.
- Render decoded skeletons during training.

### Latent Space Is Not Interpretable

Possible causes:

- Latent dimension too high.
- Decoder too powerful.
- Context absorbs developmental information.
- Axes rotate arbitrarily.

Responses:

- Reduce `u_dim`.
- Increase KL capacity slowly.
- Add auxiliary descriptor prediction from `u`.
- Compare models with and without context.
- Use explicit descriptors and PCA as anchors.

### Rare Topologies Fail

Possible causes:

- Too many topology-specific parameters.
- Sparse exact topology classes.
- Insufficient module sharing.

Responses:

- Use compositional shared modules.
- Group topologies coarsely early.
- Use topology-frequency-aware sampling or loss weighting.
- Delay topology generation.

### Generated Shapes Are Invalid

Possible causes:

- Decoder bypasses grammar.
- Continuous variables violate geometry constraints.
- Radii/lengths become negative or profile positions are unordered.

Responses:

- Decode through grammar modules only.
- Use positive transforms for lengths/radii.
- Use sigmoid/gap parameterizations for ordered tube positions.
- Add geometry validity checks before rendering.

### Trajectories Violate Irreversibility

Possible causes:

- Topology is represented only as soft occupancy.
- Transition model allows reverse moves.
- The latent was trained only for reconstruction.

Responses:

- Keep topology discrete in dynamics.
- Use a directed acyclic transition graph.
- Model topology events with hazards.
- Learn transition maps only between allowed topology pairs.

## Suggested Future Module Layout

Raw export remains close to skeleton code:

```text
src/organograph/skeleton/export.py
scripts/export_skeleton_primitives.py
```

Future VAE code can live separately:

```text
src/organograph/models/organoid_vae/
    data.py
    schema.py
    packing.py
    grammar.py
    encoders.py
    decoders.py
    losses.py
    render.py
    train_skeleton_vae.py
    train_primitive_model.py
    train_topology_model.py
    train_transition_model.py
```

Keep these responsibilities separate:

- `skeleton/export.py`: raw portable export.
- `packing.py`: model-specific conversion from raw export to tensors.
- `grammar.py`: topology grammar and validity checks.
- `render.py`: decoded tensor to skeleton/primitives for visualization.
- training scripts: experiment-specific model fitting.

## Relevant Literature And How It Applies

- Conditional VAE / structured output VAEs:
  relevant for `p(S | u, T, C)` where topology and context condition the
  decoder.
  See Sohn et al., "Learning Structured Output Representation using Deep
  Conditional Generative Models" (2015):
  https://arxiv.org/abs/1506.03703

- Beta-VAE and capacity annealing:
  relevant for compact, more interpretable latents and for avoiding immediate
  reconstruction/KL tradeoff failures.
  See Burgess et al., "Understanding disentangling in beta-VAE" (2018):
  https://arxiv.org/abs/1804.03599

- Junction Tree VAE:
  useful scaffold-first analogy, but likely too general for the shallow
  organoid grammar.
  See Jin et al. (2018):
  https://arxiv.org/abs/1802.04364

- Graph VAE:
  relevant as a baseline concept for graph-structured latent models, but not
  the first choice because validity and biological grammar are easier to
  enforce explicitly here.
  See Kipf and Welling (2016):
  https://arxiv.org/abs/1611.07308

- GRASS and StructureNet:
  relevant because they model shape as hierarchical structure plus geometry,
  and support the idea of shared part/structure modules.
  GRASS: https://arxiv.org/abs/1705.02090
  StructureNet: https://arxiv.org/abs/1908.00575

- Gumbel-Softmax / categorical relaxations:
  useful later if a joint differentiable topology generator is needed, but not
  necessary for the first observed-topology models.
  See Jang et al. (2016):
  https://arxiv.org/abs/1611.01144

- Neural jump / hybrid dynamics:
  relevant later for continuous latent motion interrupted by irreversible
  topology events.
  See Jia and Benson, "Neural Jump Stochastic Differential Equations" (2019):
  https://arxiv.org/abs/1905.10403

## Short Version For Future Agents

Build in this order:

1. Export raw skeleton/primitives with `scripts/export_skeleton_primitives.py`.
2. Build a separate versioned packer from raw export to `T/S/P/C` tensors.
3. Benchmark PCA/factor-analysis on `S`.
4. Train a topology-conditioned skeleton VAE on observed `T + S`.
5. Replace flat decoding with shared compositional grammar modules.
6. Add primitive detail `P` as a conditional refinement.
7. Add a grammar-based topology model `p(T | C)`.
8. When time courses exist, add irreversible transition dynamics.

Do not start with a full graph VAE, independent VAEs per topology, or a model
that treats crypts independently.  The central principle is:

```text
discrete biology-aware topology
  +
shared low-dimensional continuous morphospace
  +
compositional skeleton-first decoding
  +
primitive details added only after the skeleton model is stable
```

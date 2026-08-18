# OrganoGraph shape export v2: VAE handoff

This document is the integration contract between OrganoGraph and OrganoidVAE.
The producer schema is `organograph_shape_v2`.

## Producer stage

The export is created from the final `PrimitiveFitResult.graph` returned by:

```python
skeleton = skeletonize_organoid(...)
primitives = fit_primitives_for_skeletonization_result(skeleton, ...)
save_shape_export(primitives, output_dir, metadata=sample_metadata)
```

This is the same stage rendered by `plot_mesh_with_skeleton_and_primitives` in
`notebooks/tutorial_skeleton.ipynb`. It is after crypt-overlap merging, merged
geometry recomputation, graph rebuilding, crypt-node centerline updates, and the
final primitive refit. Visualization-only blending is not included.

The tutorial and batch exporter both use the factories in
`organograph.skeleton.profiles`, preventing configuration drift.

## Per-organoid layout

Each organoid directory contains one `shape.json`:

```text
schema_version: "organograph_shape_v2"
sample
coordinate_transform
summary
skeleton
  nodes[]
  edges[]
primitives[]
```

The batch script writes all configured source datasets beneath one configurable
`EXPORT_ROOT`. Samples use
`<dataset>/<timepoint>/<label_uid>/shape.json`, so labels shared by different
acquisitions cannot collide with one another or with old v1 exports.

Fit errors, residuals, detection fields, HKS arrays, component masks, mesh
vertex indices, fitting configs, and derived descriptors are intentionally not
part of this reconstructive file.

## Sample and eligibility

`sample` contains available identity/context fields:

```text
dataset, timepoint, well, organoid_id, label_uid
condition, replicate, age, cell_count          # when supplied
mesh_path, source_units                         # when supplied
has_branches, vae_eligible
```

The planned VAE dataset contains only organoids with:

```python
payload["sample"]["vae_eligible"] is True
payload["sample"]["has_branches"] is False
```

The raw producer can still export branched samples for analysis. The batch
script also supports `--unbranched-only` when those files should not be written.

## Coordinate systems

All exported skeleton and primitive values are in `prepared_mesh` coordinates,
the coordinates used during fitting and tutorial visualization.

`coordinate_transform` stores:

```text
center[3]
scale
rotation[3,3]
source_to_fitted[4,4]
fitted_to_source[4,4]
center_applied, rotation_applied
source_coordinate_system, fitted_coordinate_system
source_units, fitted_units
```

For a source-mesh column vector `x`, the fitted coordinate is:

```text
y = rotation @ (x - center) / scale
```

Use the homogeneous matrices as the authoritative convention. OrganoGraph can
restore the complete graph directly:

```python
from organograph.skeleton import load_shape_export_graph

fitted_graph = load_shape_export_graph("shape.json")
source_graph = load_shape_export_graph("shape.json", coordinate_system="source")
```

Restoration transforms node positions, primitive centers, centerline controls,
radii, semiaxes, and primitive orientation matrices. The spectral smoothing
step changes geometry but not the coordinate transform; `source` therefore
means the original mesh coordinate frame, not the unsmoothed surface.

## Skeleton

Nodes contain only:

```text
node_id, node_type, crypt_id, position[3]
```

Edges contain only:

```text
edge_id, source, target, edge_type, crypt_id
```

For VAE-eligible samples, every appendix is an unbranched crypt path. Typical
paths are:

```text
body -> attachment -> crypt -> tip
body -> attachment -> constriction -> crypt -> tip
```

The second path is a budded/constricted crypt. Node type and graph connectivity
are authoritative; do not infer topology by parsing node ID strings.

## Primitive records

Every primitive contains:

```text
primitive_id
primitive_type
role                         # body, branch, body_branch_neck, or crypt
attachment_scope             # node, edge, or graph
owner_id
target_node_ids[]
parameters
```

No fitting objective or residual is exported.

### Body

The definitive body primitive is normally a soft-barrier superellipsoid:

```text
center[3]
orientation[3,3]
axis_lengths[3]
epsilon_1
epsilon_2
```

The schema also supports ellipsoids and asymmetric superellipsoids. For an
asymmetric superellipsoid, `axis_lengths_negative[3]` and
`axis_lengths_positive[3]` replace `axis_lengths[3]`.

### Crypt

The crypt primitive is a tapered tube with deterministic distal closure:

```text
centerline_type                  # line, polyline, quadratic_bezier, cubic_bezier
centerline_control_points[K,3]   # K=2, variable, 3, or 4 respectively
centerline_samples               # reconstruction sampling count
r_neck
r_body
r_tip
s_body
s_taper
r_constriction                   # null when absent
s_constriction                   # null when absent
radius_profile                   # shape_preserving_cubic_squared_radius_v1
```

The dense 64-point centerline used during plotting is not stored. It is exactly
reconstructed from the line/Bézier controls. `r_neck`, `r_body`, and `r_tip`
are the fitted radial degrees of freedom. The surface closes to zero at the tip
deterministically; zero is not an additional fitted radius.

Length, bend angle, tortuosity, constriction ratio, and taper ratio are derived
from these parameters and must be recomputed rather than modeled as independent
degrees of freedom.

## Crypt ordering: important breaking change

Crypt order in `primitives[]`, `skeleton.nodes[]`, or any derived list has **no
semantic meaning**. `crypt_id` is an identity/linkage field for one export, not
a persistent angular slot shared across organoids.

The old OrganoidVAE v0 packer did the following:

1. fit PCA to non-body skeleton-node positions;
2. choose PCA signs using the most extreme node on each axis;
3. sort appendices by azimuth and polar angle in that frame;
4. assign the sorted list to fixed crypt slots.

That ordering is discontinuous for four reasons:

- crypt locations themselves determine the PCA frame;
- similar PCA eigenvalues permit axis swaps;
- a different extreme crypt can flip an axis sign;
- azimuth has a discontinuity at `-pi/pi`.

Small shape changes can therefore cause a large permutation of model targets.
Removing branches simplifies topology but does not fix this issue.

### Required consumer behavior

Preferred: represent crypts as an unordered set with a shared crypt encoder,
masked pooling/attention, and a permutation-invariant reconstruction objective.
For decoding, use set prediction with Hungarian matching or another assignment
loss between predicted and target crypts.

Acceptable temporary alternative: retain padded slots but match target crypts
to predicted slots per sample during the loss. Do not use PCA angle order as
slot identity, and do not feed numerical slot index as a biological feature.

If deterministic ordering is needed only for plotting, calculate it after model
inference. Such display ordering must never define the training target.

## Suggested model-facing fields

For each eligible sample, a compact factorization is:

```text
Body:
  primitive center/orientation/semiaxes/exponents

Crypt set:
  subtype from constriction-node presence
  attachment, crypt-center, and tip positions relative to body
  centerline type and control-point offsets
  r_neck, r_body, r_tip
  s_body, s_taper
  optional constriction mask, r_constriction, s_constriction
```

The raw export should remain in fitted coordinates. Canonicalization for a
specific model is a separate, versioned OrganoidVAE packing step. Preserve the
producer's coordinate transform alongside packed data so decoded shapes can be
returned to their source frame.

Body primitive PCA orientations can themselves have sign/permutation ambiguity
for near-symmetric bodies. Either use a symmetry-aware orientation loss or
define the model frame from external acquisition axes. Do not silently assume
the fitted orientation matrix gives globally stable semantic x/y/z axes.

## Strict JSON behavior

`shape.json` is written with `allow_nan=False`. All required reconstructive
geometry must be finite. Optional constriction values are represented together
as two JSON `null` values when no constriction exists. Invalid geometry fails
the export rather than emitting `NaN`, `Infinity`, or a partial primitive.

## Producer validation

OrganoGraph tests verify:

- compact graph and primitive reconstruction;
- exact Bézier centerline regeneration;
- source-coordinate position, scale, and orientation restoration;
- strict finite JSON;
- branch eligibility flags;
- absence of crypt-slot semantics.

Any future primitive family must add an explicit v2 parameter adapter and a
round-trip test before it can be exported.

# Organoid skeleton and primitives

This package implements one biology-aware shape pipeline. It is not a generic
medial-axis extractor.

## Stages

1. `skeletonize_organoid` fits a body soft-barrier superellipsoid before HKS
   detection, detects and validates crypt/branch candidates, fits ellipsoid
   barriers to accepted branches, and assigns each terminal crypt attachment.
   The default `host_surface` strategy projects the original candidate boundary
   onto the host primitive. The optional `embedded_boundary_plane` strategy
   first searches up to 1.5 initial crypt lengths for a circumference minimum
   or second-derivative transition, grows the crypt component to that boundary,
   and continues toward the host when its planar opening center is still
   outside. Its attachment is the maximum-clearance point in the refined
   boundary plane and may lie inside the body or branch primitive. If the 35%
   mesh-growth safeguard is reached before contact, only the attachment point
   falls back to the closest host-surface point.
2. The skeleton graph uses barrier centers for body and branch nodes. Terminal
   crypts contain an attachment, one crypt-center waypoint, and a surface tip.
   A constriction is represented by the fitted radius curve, not a graph node.
3. `fit_primitives_for_skeletonization_result` reuses the barrier primitives for
   body and branches by default, then fits body-branch neck cylinders and crypt
   tubes. Detection chooses one final tip; primitive fitting builds a
   boundary-to-tip distance-ratio field, measures ten cross-sectional contours,
   and fits one endpoint-normal cubic Hermite centerline with independently
   fitted proximal/distal tangent lengths and physical curvature regularization.
   After the centerline is fixed, each crypt grows competitively into connected
   mesh vertices not protected by a body/branch support mask. Growth is limited
   to 1.5 times the centerline length in restricted tip-geodesic distance, and
   contested vertices go to the nearest tip. The body support mask has its own
   primitive-stage level (`radius_support_body_level`), so this
   cannot change HKS detection or skeleton topology. A separate
   centerline-arc-length field then supplies transverse radius contours through
   `s=0.95`; these, rather than the geodesic ratio contours, drive the tube fit.
   The attachment-ring observation at `s=0` remains visible in diagnostics but
   is excluded from the radius objective by default because host overlap can
   make that transverse radius spuriously large. Eight positive radii on the
   shared grid `[0, .10, .20, .30, .45, .60, .75, .85]` are fitted to all retained
   transverse observations with asymmetric area error and mild log-radius
   smoothing. A deterministic squared-radius interpolation closes to zero at
   `s=1`. The crypt node is the volume center of this fitted profile. With
   both attachment strategies, the opening tangent uses the host normal at the
   closest primitive-surface point. An embedded attachment remains in place;
   the closest surface point supplies only its tangent frame.
   As a final topology check, same-host tubes whose intersection exceeds
   `PrimitiveFitConfig.crypt_overlap.threshold` of the smaller tube are merged;
   pairs whose host-to-attachment directions differ by more than
   `crypt_overlap.max_host_attachment_angle` are skipped before volume sampling;
   tip, distance field, neckline, projected opening, and centerline geometry
   are recomputed from the union component before the graph and primitives are
   refitted. Set
   `PrimitiveFitConfig.refine_host_primitives=True` only for an explicit
   second-stage host fit.
4. `blend_primitives_for_visualization` is separate and should not contribute
   parameters to the future VAE representation.

## Public API

```python
from organograph.skeleton import (
    PrimitiveFitConfig,
    SkeletonizationConfig,
    fit_primitives_for_skeletonization_result,
    skeletonize_organoid,
)

skeleton = skeletonize_organoid(
    mesh,
    vocab,
    geodesic_fn=compute_geodesics_dijkstra,
    config=SkeletonizationConfig(),
)
primitives = fit_primitives_for_skeletonization_result(
    skeleton,
    config=PrimitiveFitConfig(),
)
```

Configuration is grouped into candidate detection, neck profiling, branch
validation, barrier ownership, mesh preparation, and graph topology. Failed
opening projections are available through `skeleton.failed_attachments`;
they are never hidden by the workflow.

Choose attachment semantics through the barrier configuration:

```python
config = SkeletonizationConfig()
config.detection.barriers.attachment_strategy = "embedded_boundary_plane"
config.detection.barriers.boundary_refinement_max_distance_factor = 1.5
config.detection.barriers.boundary_refinement_max_mesh_fraction = 0.35
```

The maintained default is `"host_surface"`. The tutorial and export script
expose all three settings in their configuration sections. The mesh-fraction
limit prevents a failed host-contact search from consuming most of an organoid.

`notebooks/tutorial_skeleton.ipynb` is the maintained interactive example.
`scripts/export_skeleton_primitives.py` applies the same workflow to the
datasets and timepoints declared in its `DATASET_TIMEPOINTS` configuration. It
writes one combined export beneath its configurable `EXPORT_ROOT`, retaining
the source dataset in every sample path and manifest row.

`notebooks/audit_crypt_primitive_fits.ipynb` audits an existing export and can
compare it with a complete or partial candidate export using matched organoids.
New exports keep optimizer, support, and residual diagnostics in a separate
`quality.json`; the compact VAE-facing `shape.json` remains reconstructive only.

## Reconstructive export

The tutorial and batch script share the maintained factories in `profiles.py`.
`save_shape_export` writes the strict, compact `organograph_shape_v6` schema:
sample identity and branch eligibility, reversible original/prepared coordinate
transforms, graph nodes and edges, and only the primitive parameters needed to
reconstruct the final visualization. Fit errors, detections, component masks,
and derived descriptors are not exported.

Crypt order is explicitly non-semantic. Downstream models must use graph
identity and permutation-invariant crypt handling rather than PCA-angle slots.
See `VAE_EXPORT_V6.md` for the OrganoidVAE integration contract. The loader
continues to accept v2-v5 files so baseline fitting audits remain reproducible.

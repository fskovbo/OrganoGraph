# Organoid skeleton and primitives

This package implements one biology-aware shape pipeline. It is not a generic
medial-axis extractor.

## Stages

1. `skeletonize_organoid` fits a body soft-barrier superellipsoid before HKS
   detection, detects and validates crypt/branch candidates, fits ellipsoid
   barriers to accepted branches, and assigns terminal attachments at the first
   persistent crossing of the appropriate host boundary.
2. The skeleton graph uses barrier centers for body and branch nodes. Terminal
   crypts contain an attachment, an optional true constriction, one crypt-center
   waypoint, and a surface tip. Graph edges remain straight.
3. `fit_primitives_for_skeletonization_result` reuses the barrier primitives for
   body and branches by default, then fits body-branch neck cylinders and crypt
   tubes. As a final topology check, same-host tubes whose intersection exceeds
   `PrimitiveFitConfig.crypt_overlap.threshold` of the smaller tube are merged;
   pairs whose host-to-attachment directions differ by more than
   `crypt_overlap.max_host_attachment_angle` are skipped before volume sampling;
   tip, distance field, neckline, attachment crossing, and centerline geometry
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
validation, barrier ownership, mesh preparation, and graph geometry. Failed
host-boundary crossings are available through `skeleton.failed_attachments`;
they are never hidden by the workflow.

`notebooks/tutorial_skeleton.ipynb` is the maintained interactive example.
`scripts/export_skeleton_primitives.py` applies the same workflow to the
datasets and timepoints declared in its `DATASET_TIMEPOINTS` configuration. It
writes one combined export beneath its configurable `EXPORT_ROOT`, retaining
the source dataset in every sample path and manifest row.

`notebooks/audit_crypt_primitive_fits.ipynb` audits an existing export and can
compare it with a complete or partial candidate export using matched organoids.

## Reconstructive export

The tutorial and batch script share the maintained factories in `profiles.py`.
`save_shape_export` writes the strict, compact `organograph_shape_v2` schema:
sample identity and branch eligibility, reversible original/prepared coordinate
transforms, graph nodes and edges, and only the primitive parameters needed to
reconstruct the final visualization. Fit errors, detections, component masks,
and derived descriptors are not exported.

Crypt order is explicitly non-semantic. Downstream models must use graph
identity and permutation-invariant crypt handling rather than PCA-angle slots.
See `VAE_EXPORT_V2.md` for the OrganoidVAE integration contract.

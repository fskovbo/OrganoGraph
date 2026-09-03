# OrganoGraph Shape Export v5

This document describes the compact reconstruction contract consumed by the
OrganoidVAE project. Shape files use schema `organograph_shape_v5`. Historical
v2-v4 files remain readable, but newly fitted data should be exported as v5.

## Pipeline

1. Fit body and optional branch barrier primitives.
2. Detect crypt candidates and place each attachment on its host primitive.
3. Select the final crypt tip and form a boundary-to-tip ratio coordinate.
4. Measure cross-sectional contour centers and radii.
5. Fit one cubic Hermite centerline from attachment to tip. Its start tangent
   is the outward host-primitive normal and its end tangent is the distal
   tip-plane normal. Only one shared tangent length is fitted.
6. Fit the interpretable radius profile and place the crypt skeleton node at
   the fitted tube volume centroid, where volume density is proportional to
   `r(s)^2`.

The Hermite construction is deliberately restrictive. It preserves the two
biological endpoint directions without exporting free spline control points
that can oscillate or fold.

## Crypt Primitive

A crypt record contains:

- attachment and tip endpoint positions as `centerline_control_points`;
- `centerline_start_tangent` and `centerline_end_tangent`;
- `r_attachment`, `r_center`, and `r_distal`;
- `s_center` and fixed `s_taper`;
- optional `r_constriction` and `s_constriction`.

`centerline_type` is `tangent_hermite`. Standard cubic Hermite basis functions
reconstruct the sampled centerline. Tangent vectors transform as vectors under
rotation and scale, without translation.

The graph's crypt-node position is the volume center of the fitted primitive:

`s_node = integral(s * r(s)^2 ds) / integral(r(s)^2 ds)`.

The mesh contour area center is used only to initialize radius fitting and is
not the final crypt-node definition. The explicit graph node position remains
the authoritative reconstructive value.

## Coordinates And Ordering

Every file includes reversible fitted-to-source and source-to-fitted transforms.
Positions use the full affine transform, tangent vectors use its linear part,
normals use rotation, and all radii/axis lengths use uniform scale.

Crypt array order has no biological meaning. Match appendices by `crypt_id` and
graph connectivity, or use a permutation-invariant representation. Organoids
with branches can be filtered by graph node type when preparing a branch-free
VAE dataset.

## Excluded Data

Detection fields, mesh ownership arrays, residuals, and optimization traces are
not part of the VAE shape payload. They are written to the quality sidecar for
auditing only. JSON export rejects NaN and infinite values.

# OrganoGraph Shape Export v4

`shape.json` contains the final skeleton and reconstructive primitive degrees
of freedom. `quality.json` contains diagnostics and must not be encoded by the
VAE. Crypt array order has no biological meaning; use graph connectivity and
`crypt_id`, or a permutation-invariant appendix encoder.

## Crypt Geometry

Detection produces one final tip. It starts from the vertex furthest from the
candidate boundary and accepts the maximum-HKS vertex in the distal 60% only
when HKS improves by at least 5%. Primitive fitting does not repeat tip
selection.

The candidate boundary is projected to the fitted host primitive. Its
attachment is the maximum-clearance host-surface point enclosed by that ring.
Restricted mesh distances to the boundary and tip define

`s = d_boundary / (d_boundary + d_tip)`.

Ten iso-contours provide centers, circumferences, equivalent radii, and areas.
Their centers fit either a line or one minor circular arc with fixed attachment
and tip endpoints. A curved centerline is represented by one transverse
midpoint sagitta vector, so it cannot oscillate or fold back on itself. The
crypt skeleton node is derived at

`s_center = integral A(s) * s ds / integral A(s) ds`.

It is not a centerline constraint. The opening ring uses the fitted host
primitive's outward surface normal and transitions deterministically into the
transported centerline frame.

## Crypt Radius

Each terminal crypt is exported as one `tapered_capped_tube` with:

- `r_attachment` at the host boundary;
- optional `r_constriction` and `s_constriction` for a genuine waist;
- `r_center` at the area-derived `s_center`;
- learnable `r_distal` at the distal cap onset;
- deterministic `s_taper = 0.85` and zero radius at the tip.

The radius curve is a shape-preserving interpolation of squared radius. Radius
fitting uses contour-derived cross-sectional radii and penalizes overfill twice
as strongly as underfill. A constriction remains a radius-profile degree of
freedom and is no longer a skeleton node.

## VAE Inputs

For each crypt, encode attachment and tip geometry in a body-relative frame,
the two transverse components of the sagitta normalized by chord length,
`r_attachment`, `r_center`, `r_distal`, `s_center`, and optional constriction
controls. Do not learn `s_taper`, the opening normal, rendering frame blend, fit
errors, contour samples, or optimizer diagnostics.

## Compatibility

The loader and audit utilities still read historical line, Bezier, and
sinusoidal v2/v3 exports through `legacy_curves.py`. Those samplers are decoding
support only and are not available to current fitting. In v2/v3, `s_body` or
`s_center` was based on a projected point or radius landmark and does not have
the v4 area-center meaning.

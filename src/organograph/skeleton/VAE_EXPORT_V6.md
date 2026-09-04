# OrganoGraph Shape Export v6

This note is the migration contract for the OrganoidVAE importer. New shape
files use schema `organograph_shape_v6`; OrganoGraph can still read v2-v5
files for baseline comparisons.

## What Changed

Crypt radius is no longer encoded by the variable semantic landmarks
`r_attachment`, `r_center`, `r_distal`, and optional constriction fields.
Each crypt now has a fixed-width radius vector:

```text
radius_control_s     = [0.00, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.85]
radius_control_radii = [r0,   r1,   r2,   r3,   r4,   r5,   r6,   r7]
```

All eight radii are positive fitted parameters. Reconstruction applies a PCHIP
interpolant to squared radius and appends the deterministic point
`(s=1, r=0)`. This produces a smooth rounded distal closure. The positions are
shared by every crypt and therefore do not need to be predicted by the VAE.

There is no separate constriction radius/position and no separate
`constriction` skeleton node for budded crypts. A budded constriction is
represented naturally by a local minimum in the eight radius values. Do not
expect `r_constriction`, `s_constriction`, `r_center`, or `s_center` in a
v6 compact primitive record.

The ordinary `crypt` skeleton node remains. Its position is not an independent
radius landmark: it is derived from the fitted tube volume centroid,

```text
crypt_node_s = integral(s * r(s)^2 ds) / integral(r(s)^2 ds).
```

## Crypt Centerline

The compact centerline remains one cubic Hermite curve and is fully
reconstructible from:

- `centerline_control_points`: attachment and tip, shape `[2, 3]`;
- `centerline_start_tangent`: proximal derivative vector, shape `[3]`;
- `centerline_end_tangent`: distal derivative vector, shape `[3]`;
- `centerline_type = "tangent_hermite"`;
- `centerline_samples`: visualization/reconstruction sampling density.

The tangent vectors include their fitted magnitudes. They are vectors, so
coordinate restoration applies rotation and uniform scale but no translation.
The radius controls scale uniformly.

## Minimal Crypt Record

```json
{
  "primitive_type": "tapered_capped_tube",
  "parameters": {
    "centerline_type": "tangent_hermite",
    "centerline_control_points": [[0, 0, 0], [1, 0, 0]],
    "centerline_start_tangent": [0.4, 0, 0],
    "centerline_end_tangent": [0.3, 0, 0],
    "centerline_samples": 64,
    "radius_profile": "fixed_grid_shape_preserving_squared_radius_v1",
    "radius_control_s": [0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.85],
    "radius_control_radii": [0.2, 0.25, 0.32, 0.4, 0.5, 0.42, 0.3, 0.18],
    "s_taper": 0.85
  }
}
```

`opening_normal` and `opening_frame_blend_fraction` may also be present for
faithful surface rendering. Fit errors, observations, support masks, and
optimizer traces remain in `quality.json`, not in the VAE-facing shape.

## VAE Import Guidance

Use the eight radius values as one ordered fixed-width feature vector. Normalize
them by the same organoid scale used for positions. The fixed `radius_control_s`
grid and `s_taper` are reconstruction constants and should not be learned.
The crypt-node arc-length coordinate is recomputed from decoded radii; it is
not exported as an independent primitive parameter or treated as a latent
target. Its reconstructed 3D position can be evaluated on the decoded
centerline. The skeleton record also carries the fitted node position.

Crypt list order remains non-semantic. Use graph identity/permutation-invariant
appendix handling rather than PCA-angle slots. The current VAE dataset may
exclude samples containing branch nodes, but the export format itself still
supports them.

Every shape includes reversible fitted/source coordinate transforms. JSON
serialization rejects NaN and infinity.

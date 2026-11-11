# Consistent family frame and cap phase — 2026-11-11

Implemented and executed `notebooks/crypt_morphology/05_consistent_family_alignment.ipynb`.
The controlled comparison repeats notebook 04's two real crypts, all seventeen
openings, degree-12 basis, frozen cohort references, size conventions, interpolation
paths and fractions. It changes only how the known variants are aligned before
interpolating their coefficients. No encoder or decoder model is trained.

## What changed

The original condition independently aligns every cut to the same cohort template.
The new condition aligns the baseline once, then holds its physical origin and
rotation fixed for the entire boundary family. In physical mode the poses are
identical. In unit-RMS mode each cut still uses its own measured radius, so the
physical origin is fixed while numerical translation scales with the radius.
Normalization sensitivity has deliberately not been removed from this comparison.

Cap phase comes from matched original membrane vertices on the unchanged core.
This is 2,590 vertices for the ordinary crypt and 4,954 for the difficult crypt.
Core triangles are the same ones used in notebook 04: all three vertices have
signed edge distance less than `-0.1 R0` from the baseline rim. Fixed triangle-area
weights are lumped to their vertices. Inserted cut vertices are excluded.
Source IDs and physical coordinates must both agree.

For complex disk coordinates `z=u+iv`, the fitted phase minimizes
`sum(w * abs(z_variant * exp(-i*delta) - z_baseline)**2)`.
This removes one global disk rotation, without a radial/nonrigid chart warp.
The variants' final phases are the baseline phase plus their fitted offset.
The minimum coherence was 0.99491 (ordinary) and 0.99097 (difficult). Maximum
remaining disk RMS residuals were 0.08173 and 0.07539 on the unit disk. High
coherence means agreement about the angle, not identical charts.

New reusable functions are `shared_vertex_phase`, `reframe_representation`, and
`align_known_boundary_variant` in `family_alignment.py`. Reframing is exact
coefficient algebra, not a refit. Real-case checks found a maximum source-surface
change of `7.2e-15 R0` at verification samples after inverse pose/phase, and a
maximum physical-origin discrepancy of `1.3e-15 R0`.

The original independent coefficients were regenerated, agreeing with notebook
04 to a maximum absolute coefficient difference of `1.45e-6` in coefficient
units. Checks required coefficient and decoded verification-grid agreement
within `1e-6 R0`; numerical phase minimization changes the final digits between
runs. Prior outputs were used for this comparison only, not as the new results.

## Results

The cases remain `20251201/day4p5_B02_40`, crypt 0 (ordinary), and
`20250929/day4p5-more_C04_109`, crypt 0 (difficult). All seventeen cuts were
accepted for both. Interpolations use 25%, 50% and 75% mixtures between uniform
inward/outward `0.1 R0` cuts and between opposite sideways `0.1 R0` cuts.
Generated outputs are never realigned or renormalized.

The main metric is area-weighted RMS distance from actual source-core triangle
centroids to the generated triangle mesh, divided by baseline RMS radius in
the corresponding coefficient units. Centroids differ from fitting quadrature
points. Both methods use the same baseline frame, core and evaluation grid.

| Crypt | Size mode | Baseline fitting error | Previous worst interpolation | Shared frame/phase worst interpolation |
| --- | --- | ---: | ---: | ---: |
| Ordinary | physical | 0.175% | 7.351% | 0.593% |
| Ordinary | unit RMS | 0.176% | 2.559% | 1.759% |
| Difficult | physical | 1.038% | 6.730% | 3.092% |
| Difficult | unit RMS | 1.038% | 23.068% | 3.145% |

These are maxima over six interior mixtures, excluding observed endpoints.
Every case/size combination improves on that measure; not every individual
mesh or endpoint improves. For example, the ordinary unit-RMS uniform endpoints
have about 3.51% and 3.25% error in the shared frame because each is compared to
the baseline's normalized core. Their measured radii differ by about -4.23% and
+4.12% from baseline. The difficult crypt's radius range is -1.78% to +1.75%.
Source geometry and angular phase corrections do not remove this size effect.

Physical-size shared-frame interpolation errors are 0.306% (uniform) and
0.593% (sideways) for the ordinary crypt, and 1.986% (uniform) and 3.092%
(sideways) for the difficult crypt. The difficult observed physical endpoints
remain at approximately 0.88–1.41% core error. The larger midpoint error is
consistent with remaining chart differences causing deformation when averaging
coordinates, beyond the original fitting error. This is an interpretation,
not a separate decomposition proving the cause of every residual.

RMS averages conceal local differences. The largest sampled core distances
among shared-frame mixtures are 2.01%/3.59% (physical/unit RMS) for the ordinary
crypt and 11.21%/11.32% for the difficult crypt. These are sampled maxima, not
certified Hausdorff bounds or anatomical acceptability thresholds.

## Resolution and geometry checks

All **184 standard-grid evaluations** use 24 rings × 96 sectors:
two cases × two size modes × two methods × (17 observations + 6 mixtures).
**66 selected evaluations** were repeated at 48 × 192, including baselines,
all four endpoints, both midpoints and the worst interpolation on each path.
Counts include the common baseline in both alignment conditions.

There were **no detected self-intersections, collapsed triangles, or >90-degree
face-normal turns in any of the 250 evaluations**. The largest absolute change
in core RMS error on refinement was 0.210 percentage points. Among the selected
fine-grid shared-frame mixtures, the worst errors were 0.560%/1.794% for the
ordinary crypt and 2.921%/2.973% for the difficult crypt (physical/unit RMS).
The improvement and remaining difficult-case error persist under refinement.

Intersection checks retain the NumPy bounding-box sweep and triangle
separating-axis tests. Expected incident-face contact is excluded with `1e-7`
contraction; per-pair interval tolerance is `1e-12`. There is no `rtree`
dependency. Neither resolution certifies the entire continuous expansion.

The morphology suite passed **47 tests and 14 subtests**, including the opt-in
five-group real-data integration test. Six added test methods cover known
phase rotations across angular wrapping, radial residuals, reordered source
IDs and inserted vertices, invalid/ambiguous correspondences, exact inverse
pose/phase/scale changes, and preserved family origin with either per-variant
or explicitly fixed normalization. Notebook assertions check the real cases.

## Interpretation and next use

Consistent physical frame and phase substantially reduce the earlier
interpolation errors. Independent pose/phase selection contributed to those
errors; they should not be read as intrinsic instability of coefficient-to-mesh
decoding. The shared-frame interpolations generate complete surfaces without
the extrapolated bands seen with fixed-chart fitting.

This supports aligning an augmentation family's baseline once and keeping its
variants consistent before training/testing an encoder. It does **not** make
their coefficients identical or prove that a learned encoder will place them
nearby. Remaining chart deformation and boundary-dependent normalization still
need to be considered. The difficult crypt is not automatically an acceptable
reconstruction just because its mesh is free of numerical geometry warnings.

The control uses known parent-mesh correspondences. It does not solve alignment
between unrelated crypts, guarantee valid arbitrary coefficients, or establish
population-wide robustness from two cases. Keep organoids and all their variants
together when creating training/test splits in the learning project.

## Reproduction and artifacts

Use the `organoid-mesh` environment and run notebook 05 after notebook 04.
Set `ORGANOGRAPH_DATA_ROOT` if the read-only data live outside `../NicoleData`.
Notebook 05 checks the prior basis ID, source-geometry fingerprints and cohort
reference hashes before fitting. It writes separately from notebook 04.

- Source: `notebooks/crypt_morphology/05_consistent_family_alignment.ipynb`.
- Executed copy: `exports/crypt_morphology/05_consistent_family_alignment_executed.ipynb`.
- Reports: `exports/crypt_morphology/family_alignment/summary.csv`,
  `per_path_summary.csv`, `decoding_checks.csv`, `fine_mesh_checks.csv`,
  `phase_diagnostics.csv`, `opening_fits.csv`, `flagged_meshes.csv`.
- Reproducibility: `settings.json`, `basis.json`, both `*_core.npz` files with
  original vertex/face IDs and weights, and `decoded_test_coefficients.npz` with
  explicit case/size/method/scenario labels (numeric data; no pickle needed).
- Figures: error curves plus interactive HTML and static PNG comparisons of
  both paths in the chosen plotting size mode. The notebook displays uniform
  comparisons by default; its controls also display sideways and unit-RMS cases.

Plots are implemented in `organograph.plotting.crypt_morphology`. The coefficient
files are validation artifacts; no batch training-data exporter was added.

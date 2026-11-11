# Focused decoding stability — 2026-11-10

Implemented and executed `notebooks/crypt_morphology/04_decoding_stability.ipynb`
with the existing `organoid-mesh` environment. The experiment uses recut/hard
representations and direct coefficient evaluation only. It trains no latent
model and does not change the boundary-fitting or alignment algorithms.

## Cases and experiment

- Ordinary: `20251201/day4p5_B02_40`, crypt 0. Selected near the previous pilot's
  median core drift among 20251201 cases with all boundary variants available.
- Difficult: `20250929/day4p5-more_C04_109`, crypt 0. Selected by the previous
  pilot's largest recut-core drift.

The exact saved degree-12 basis (103 XYZ modes) and frozen physical/unit-RMS
cohort references were reused. All seventeen openings were accepted for each
case. Twenty original pilot crypts were refitted for a limited within/between
input-geometry comparison; all remain in the requested datasets and day ≥3.5
selection. The focused experiment is not a population-wide validation.

For each case and each size convention, 35 meshes were decoded:

- Seventeen fitted observations, including the baseline.
- Six coefficient interpolations: 25%, 50% and 75% between inward/outward
  openings, and between opposite sideways openings.
- Twelve opening-only errors: Fourier rim modes 1–3 or 1–12, two signs, and
  three amplitudes. Interior coefficients remain fixed.

Amplitudes were calibrated to 25%, 50% and 100% of the median observed rim RMS
variation across the physical-size boundary variants. The maximum calibrated
rim RMS was 6.806% of original RMS radius for the ordinary crypt and 5.942% for
the difficult crypt. RMS uses uniform cap angle, not physical rim arclength.
These are controlled error directions, not observed errors from a trained model.
The same physical errors were transported into both alignment/size conventions.

All **140 meshes** were checked at 24 rings × 96 sectors. **38 selected meshes**
were checked again at 48 × 192, including baselines, uniform endpoints,
interpolation midpoints, all largest rim errors, the largest core-error case,
and flagged outputs. Generated arrays were not refitted, realigned or rescaled.

## Findings

**No self-intersections or collapsed triangles were detected** in either mesh
resolution. All sampled interpolation meshes had no geometric flags. The
largest finer-mode opening error in one direction on the difficult crypt
produced >90-degree face-normal turns relative to the baseline: 3 faces at the
standard grid in each size mode; 9 physical/8 unit-RMS faces at the fine grid.
The two size modes describe the same physical perturbation; the slightly
different counts reflect the phase and tessellation of that surface.

The affected mesh has a visibly corrugated opening. A normal turn is an
inspection warning, not proof of self-intersection or a collapsed surface.
Smooth opening errors produced no geometric flags. Direct decoding did not
produce the large extrapolated bands observed with the fixed-chart fits.

Core points are centroids of fixed original membrane triangles deeper than the
largest permitted inward shift, covering 86.7% and 94.5% of the two source areas.
These points differ from the quadrature points used to fit coefficients. The
main error is point-to-generated-triangle distance from those actual source
points, expressed in the baseline's aligned frame. Values below are percentages
of the original RMS radius (in the corresponding size convention).

| Crypt | Size mode | Baseline core error | Worst interpolation core error | Worst opening-error core error |
| --- | --- | ---: | ---: | ---: |
| Ordinary | physical | 0.175% | 7.351% | 1.437% |
| Ordinary | unit RMS | 0.176% | 2.559% | 1.437% |
| Difficult | physical | 1.038% | 6.730% | 1.639% |
| Difficult | unit RMS | 1.038% | 23.068% | 1.639% |

**The interpolation errors include alignment and normalization effects.** They
cannot be read as intrinsic shape distortion alone. Independently aligning
different cuts to the same template does not guarantee a stable physical pose.
The original parent coordinates allow a separate control on fitted observations:
restore each fit to source coordinates, then apply the baseline's pose and
scale. This control is not available for arbitrary generated arrays and does
not modify the plotted interpolation meshes.

| Crypt | Size mode | Worst source-frame core error of observed variants | Largest observed rotation change |
| --- | --- | ---: | ---: |
| Ordinary | physical | 0.198% | 142.9° |
| Ordinary | unit RMS | 0.198% | 12.6° |
| Difficult | physical | 1.412% | 17.8° |
| Difficult | unit RMS | 1.412% | 160.4° |

These controls establish that the fitted source surfaces remain accurate while
some pose/phase choices change substantially. Averaging coefficients after
different pose/phase choices can itself deform a shape, so the generated
interpolation errors should not simply be subtracted away or declared rigid
motion. Alignment consistency remains part of the representation problem.

The median within-crypt boundary variation divided by the distance to the nearest
other pilot baseline was 0.481/0.678 for the ordinary crypt (physical/unit RMS)
and 0.153/0.346 for the difficult crypt. Maximum ratios were 1.159/1.168 and
0.249/0.950 respectively. These use the same fixed cap-grid metric and reference
weights for every comparison. Some ordinary-crypt cuts are therefore farther
from their baseline than its nearest other pilot crypt is. This is not a latent
encoding result or a guarantee of biological distinguishability.

## Inspection and reproduction

Open `exports/crypt_morphology/04_decoding_stability_executed.ipynb` to inspect
the completed tables, Plotly meshes and PNG response plot. The source notebook
is `notebooks/crypt_morphology/04_decoding_stability.ipynb`; it requires the
selection, basis, alignment references and boundary report from notebooks 02/03.
It recomputes its fits and measurements rather than silently reusing results.

Default figures show the uniform interpolation and finer opening-error sequence
for both crypts. `PLOT_SIZE_MODE`, `PLOT_PATHS` and `PLOT_RIM_MODES` control which
already-tested results are displayed. Each figure uses the same spatial limits
across panels; gray is the baseline, blue is decoded geometry, black is the rim.

Artifacts under `exports/crypt_morphology/decoding_stability/` include:

- `summary.csv`, `decoding_checks.csv`, `fine_mesh_checks.csv`.
- `source_pose_control.csv`, `within_between.csv`, `rim_error_calibration.csv`.
- `opening_fits.csv`, `settings.json`, `basis.json` and interactive/static figures.
- `decoded_test_coefficients.npz`: 140 generated/observed test coefficient arrays
  with basis ID, case, size mode and scenario labels, readable without pickle.

Source fingerprints, reference hashes, numerical tolerances and sampling
settings are recorded. The coefficient artifact is for this validation; no batch
training-data export script was added.

## Checks and limits

The targeted suite passed **41 tests and 9 subtests**, including eight new tests
covering direct linear decoding, interpolation, calibrated rim errors, error
transport between poses/scales, independent convex-feasibility checks for
intersection detection, shared-edge/vertex contacts, known crossings, collapsed
faces and source-frame controls. Existing morphology and mesh-format tests pass.
All eight code cells of the final notebook executed successfully, including the
interactive and static plot outputs.

A subsequent portability fix removed the `rtree` dependency from the candidate
face search, replacing it with a NumPy bounding-box sweep. The targeted suite
then passed **42 tests and 9 subtests**, including exhaustive candidate-pair
comparisons and known crossings with `rtree` imports blocked. The entire
notebook was rerun with `rtree` unavailable; all seven numerical CSV reports
matched the preceding run exactly. Intersection checks remain enabled.

Intersection checks use the
[separating-axis method](https://www.geometrictools.com/Documentation/MethodOfSeparatingAxes.pdf)
with a NumPy bounding-box sweep and per-pair normalization. Incident triangles are
contracted by 1e-7 toward their centroids to exclude expected topological contact;
overlap confined to that tiny strip can be missed. Interval tolerance is 1e-12
of pair extent. Checks are numerical and apply to sampled triangles, not a proof
about the continuous expansion or every interpolation fraction.

The results support direct coefficient decoding as a workable mechanism for
these tested perturbations. They do not establish boundary-invariant latent
encoding, reliable arbitrary coefficient sampling, or a universal acceptable
error threshold. Finer opening errors can strongly alter local normals, and
cohort alignment can change substantially across cuts. These observations remain
visible rather than being hidden by post-decoding alignment or mesh repair.

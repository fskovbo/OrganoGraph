# Opening-family pilot — 2026-11-10

This experiment replaces anatomical boundary optimization with prescribed
perturbations of the saved HKS opening. The parent membrane is unchanged. It
does not use detector tips, axes, normalized longitudinal coordinates or fitted
primitives. Neither a preferred opening nor a downstream training representation
is selected automatically.

## Design and coverage

The same twenty size-stratified crypts as step 1 were used: datasets 20250929 and
20251201, biological day 3.5 onward, with four crypts per eligible dataset/day
group. One exact degree-12 hemisphere basis (103 vector modes), regularization
1e-8, and seventeen openings per crypt were used:

- Baseline and inward/outward offsets at 5% and 10% of the original RMS radius.
- Four first-harmonic directions at each amplitude.
- Four second-harmonic phases at 10%.

Distances are edge-graph approximations from the original rim. Angular fields
are extended from rim arclength using an inverse-edge-length harmonic solve.
The baseline computational collar is 30% of the original RMS radius; logistic
blur width is 5%. These are experimental sensitivity settings, not estimates of
biological boundary uncertainty.

There were **340 requested openings**. All twenty baselines and 279 shifted
openings were accepted. Forty-one shifted openings touched triangles protected
for neighboring crypts and were rejected. Rejections occurred in four crypts:
`20250929/day4p5_A06_55` crypt 1, `20250929/day4p5_B06_45` crypt 0,
`20250929/day4p5_B06_9` crypt 0, and `20251201/day4p5_B05_19` crypt 1.
Every accepted opening completed all three fits and alignments: **897 successful
fits, 123 rejection rows, no additional fit failures**. Rejections remain in the
report denominator. Neighbor protection is conservative at whole-parent-triangle
resolution even though cuts themselves subdivide triangles.

The fixed physical core retained 77.1–94.5% of original surface area (median
85.2%). The largest displacement was only 1.04–2.78 median mesh edges, so
subtriangle clipping matters and discretization should not be ignored. Cut RMS
radii ranged from 0.929 to 1.069 times the baseline radius; cut areas ranged from
0.857 to 1.151 times baseline area.

## Results

All entries below are **percentages**, pooled over the same 279 accepted
nonbaseline crypt/variant pairs. Medians do not weight crypts equally when some
openings are rejected. These are paired sensitivity summaries, not independent
biological observations or significance tests.

| Method | Median core drift / R₀ | Median core fidelity error / R₀ | Median whole-cap drift / R₀ | Worst whole-cap drift / R₀ |
| --- | ---: | ---: | ---: | ---: |
| Recut, hard membership | 0.138% | 0.190% | 4.424% | 12.926% |
| Fixed chart, hard membership | 0.015% | 0.133% | 0.595% | 198.134% |
| Fixed chart, soft membership | 0.013% | 0.140% | 0.165% | 22.095% |

Core drift compares predictions at identical physical membrane quadrature
points, in original spatial coordinates. Fidelity compares those predictions
against the actual membrane. These measurements use the same core across all
variants and methods. The largest core fidelity errors were 1.794%, 3.140% and
3.401% respectively: smaller drift does not necessarily mean greater accuracy.
These core measurements reuse the triangle quadrature rule used in fitting;
they are not an independent held-out surface-distance validation or a bound on
errors between sampled points. Step 1's bidirectional geometry checks remain
separate from this paired stability experiment.

Whole-cap drift uses fixed per-method baseline area weights on the same cap
sampling grid, with rigid/phase alignment to a frozen per-crypt/per-method
baseline. The sampled *physical extent* differs between recut and fixed-chart
methods: fixed-chart fits include extrapolated or weakly constrained outer
regions. Thus their whole-cap scores are diagnostic and do not compare identical
physical point sets across methods; the core metric does.

Freezing the chart accounts for most of the median improvement in core
stability. Soft weights further reduced core drift relative to fixed-chart hard
fits in 240 of 279 paired variants, with a small increase in median fidelity
error. The fixed-chart hard fit can become severely unstable outside observed
geometry. Its worst unregularized condition number was about 41,764; for soft
fits it was 718, and for recut fits 28. All accepted designs retained numerical
rank 103, which alone is therefore insufficient as a quality check.

With recomputed cut RMS normalization, median whole-cap drift was 3.764%, 0.988%
and 0.417% respectively. Fixed-original-RMS and recomputed-RMS conventions must
remain separate: normalization itself changes sensitivity. For soft fits the
normalizer comes from the corresponding hard cut, not the computational support.

## Inspection and controls

`03_boundary_stability.ipynb` selects `20250929/day4p5-more_C04_109`, crypt 0, as
the largest recut-core-drift example. It shows the opening family, actual soft
fit weights, and three-method reconstructions for both the largest core change
and the worst outer extrapolation. The latter is the inward 10% offset, also the
worst fixed-chart whole-cap drift in this pilot.

An analytic asymmetric surface was tested at 16×32, 24×48 and 32×64 rings/sectors.
All 153 requested fits succeeded. Median whole-cap drift for soft fits was
0.174%, 0.164% and 0.165%; for recutting it was 2.802%, 2.798% and 2.796%.
This is a useful resolution control, not proof of convergence on arbitrary real
meshes. The analytic planar-disk unit test separately verifies distance and
clipped-area refinement. Edge-graph distance retains mesh-direction bias.

The inspected real crypt also has a 2×2 sweep of blur widths (2.5%, 5%) and
support collars (30%, 40%), using baseline plus four selected offsets/sideways
variants. The displacement fields are held exactly fixed with
`family.with_collar_width`; simply recomputing them for each collar would also
change the angular extension and confound the experiment. The recut control is
checked for identical results across these settings. All sixty sweep fits
completed. The difficult crypt remains sensitive to the collar and blur width;
widening the support is not a general cure for extrapolation.

## Reproduction and scope

Run `notebooks/crypt_morphology/03_boundary_stability.ipynb` in the `organoid-mesh`
environment. The notebook can reuse matching reports; set `RECOMPUTE=True` after
changing input data or implementation. Artifacts live under
`exports/crypt_morphology/boundary_stability/`:

- `real_stability.csv`, `diagnostics.json`, `settings.json`: all requested
  variants, failures, source fingerprints, exact basis ID and settings.
- `synthetic_resolution.csv`, `blur_support_sweep.csv`: numerical controls.
- `real_stability.png` and interactive HTML figures: inspection artifacts.

The executed notebook, including Plotly and PNG outputs, is at
`exports/crypt_morphology/03_boundary_stability_executed.ipynb`.

The targeted test suite passes **33 tests and 9 subtests**, including the five
existing real-data cases. New checks cover exact triangle clipping and
provenance, uniform and asymmetric shifts, proper rigid equivariance, protected
neighbors, fixed-chart interpolation, mesh refinement, zero-weight outlier
exclusion, reversible fixed-radius normalization, fixed cores, explicit rejected
variants and isolation of the support-width control.

The results support using this machinery for robustness experiments and keeping
chart changes separate from soft weighting. They do **not** establish complete
boundary independence or validate all cap coefficients as reliable training
targets. Outer extrapolation, baseline-support dependence, local chart distortion
and mesh discretization remain limitations. Scalar fields, batch exporters and
PCA/VAE training were not added.

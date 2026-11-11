# Step 1 validation — 2026-11-09

The implementation was tested with the existing `organoid-mesh` environment.
Both inspection notebooks were executed cell-by-cell with IPython, including
their Plotly and PNG display outputs. Executed copies and generated reports are
under `exports/crypt_morphology/`.

## Real-data pilot

Twenty crypts were selected reproducibly from 2,330 eligible saved HKS crypts,
after applying the existing blacklists and restricting biological age to at
least day 3.5. Selection used four vertex-count strata per dataset/timepoint,
favoring distinct organoids. The groups were:

- 20250929: day3p5, day4, day4p5, day4p5-more.
- 20251201: day4p5.

All 20 patches passed disk topology and mapping checks. All 60 fits (three
expansion orders per patch) completed. No degenerate reconstruction triangles
were reported. The metric below is bidirectional area-weighted point-to-triangle
RMSE, divided by the original surface RMS radius; it is not a Hausdorff bound.

| Maximum degree | Median relative surface RMSE | Worst relative surface RMSE |
| --- | ---: | ---: |
| 4 | 1.181% | 9.631% |
| 8 | 0.401% | 2.876% |
| 12 | 0.226% | 0.980% |

The worst degree-12 example was `20250929/day4p5-more_C04_109`, crypt 0; the
real-data notebook selects it for visual inspection by default. The degree-12
dictionary contains 103 vector modes. These results establish accuracy for this
pilot, not for all crypts in the datasets.

Weighted rigid/phase cohort alignment converged in both size conventions, with
the common physical orientation and first sample's angular origin anchored:

| Size convention | Iterations | Converged |
| --- | ---: | --- |
| physical | 106 | yes |
| unit_rms | 52 | yes |

Settings: 12 reference rings, 48 sectors, 32 phase-search starts, maximum 200
iterations, relative template-change tolerance 1e-6. Frozen reference artifacts
and the exact basis are retained in the validation output directory.

## Synthetic and automated checks

Six independent analytical meshes cover a hemisphere, elongation, constriction,
bending, asymmetry and a broad bottom. All were evaluated at degrees 4, 8 and 12.
The worst degree-12 relative surface error was 1.357% (the bent example).

The targeted test run passed **25 tests and 9 subtests**, including five real-data
cases. It included `tests/crypt_morphology/` and
`tests/test_mesh_format_normalization.py`. Coverage includes exact inverse poses,
proper rotations, zero-weight exclusions, reflection rejection, cap phase,
symmetry ambiguity, frozen-reference reuse, size scaling, mesh relabeling,
remeshing, exact surface moments, basis normalization and serialization.

An additional check of `tests/test_crypt_segmentation_settings.py` exposed an
existing unrelated mismatch: that test expects `min_percent_greater=2.0`, while
the committed segmentation script sets `4.0`. Both values were verified in HEAD;
neither file was changed for this work.

## Remaining scientific checks

The subsequent opening-family and soft-membership experiments are recorded in
[BOUNDARY_VALIDATION.md](BOUNDARY_VALIDATION.md). They measure sensitivity rather
than establish boundary independence. Marker fields are step 3. Successful
reconstruction/alignment does not establish anatomical correspondence. Current
diagnostics do not certify absence of self-intersections or validity of arbitrary
coefficient vectors. Batch training-data export remains deferred.

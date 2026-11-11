# Crypt morphology: fixed-cap representations

Step 1 implements open-surface extraction, a shared cap basis, mesh reconstruction,
reversible size conventions, and weighted rigid/phase alignment. Detection only
supplies vertex membership. Detector tips, axes, HKS scores and fitted primitives
are not used to construct the representation.

Step 2 tests representation stability across prescribed families of openings,
including soft membership. No anatomical boundary optimum is assumed or fitted.
Scalar fields remain a later stage. The learning project owns PCA/VAE training and latent decoding; this
package supplies the reusable coefficient-to-mesh reconstruction underneath it.
There is no batch exporter yet.

## Minimal use

```python
from organograph.crypt_morphology import (
    CapBasis, extract_patch, parameterize_patch, fit_representation,
    reconstruct, with_size_mode, align_representations,
    make_alignment_reference, align_to_reference,
)

basis = CapBasis.build(max_degree=12)  # fixed hemisphere for every sample
patch = extract_patch(original_vertices, original_faces, crypt_vertex_ids)
mapping = parameterize_patch(patch, half_angle=basis.half_angle)
physical = fit_representation(patch, mapping, basis, size_mode="physical")
shape_only = with_size_mode(physical, "unit_rms")  # no refitting
vertices, faces = reconstruct(physical, basis, original_coordinates=True)

result = align_representations(physical_representations, basis)
reference = make_alignment_reference(result, basis)
held_out_result = align_to_reference(held_out_representations, basis, reference)
```

Run separate cohorts for `physical` and `unit_rms`. The former retains all lengths
in source mesh units. The latter divides by the area-weighted RMS radius of the
original piecewise planar surface; its centroid and second moment are integrated
exactly over triangles. This radius, physical surface area and reversible pose
remain available. The current radius depends on the selected patch; its boundary
sensitivity is measured in step 2, not solved by normalization. For paired
experiments, `with_size_mode(rep, "unit_rms", radius=fixed_radius)` lets all
variants share a normalization; `rep.pose.scale` records the actual divisor.

Core imports require NumPy and SciPy only. Original VTP loading uses the existing
`OrganoidMesh` dependencies (VTK/libigl). Bidirectional surface-distance diagnostics
use libigl. Plotly/Matplotlib are optional and imported inside functions in
`organograph.plotting.crypt_morphology`. The existing `organoid-mesh` environment
contains these dependencies.
Numerical self-intersection checks use NumPy for bounding-box candidate pairs
and triangle tests; no `rtree` installation is required.

## Surface patch and chart

Extraction retains source triangles whose three vertices belong to the supplied
HKS region. It records original vertex/face IDs and reports selected vertices
unused by retained faces. It requires one connected, consistently oriented,
manifold topological disk. Holes, handles, duplicate faces, pinched vertices and
degenerate triangles produce errors; extraction does not silently repair them.

The opening is mapped by physical boundary arclength to a circle. Positive
mean-value coordinates place the interior in that disk; inverse stereographic
projection maps it to the fixed spherical cap. The numerical boundary start
vertex determines angular phase, which alignment subsequently handles. The cap
pole is a coordinate choice, not an anatomical landmark. No closing surface is
added. Area/angle distortion and triangle flips are inspected explicitly.

This is an initial mapping method, not a claim of optimal anatomical
correspondence. It can distort narrow-necked or long surfaces, especially at low
expansion order. Boundary changes may redistribute coordinates throughout the
patch. Step 2 tests these effects on common physical surface regions.

## Basis and fitting convention

For cap colatitude `theta <= theta_c`, define stereographic disk radius
`r = tan(theta/2) / tan(theta_c/2)`. The shared dictionary consists of:

1. The constant mode and rim extensions `r**m cos(m phi)`, `r**m sin(m phi)`.
2. Normalized real Dirichlet cap harmonics
   `P_nu^m(cos(theta)) cos(m phi)` and the corresponding sine modes, where
   `P_nu^m(cos(theta_c)) = 0`.

The first group permits a freely shaped 3D opening. The second describes the
interior residual with zero boundary values. This avoids collapsing the rim with
a Dirichlet-only expansion, or forcing zero transverse derivatives there with a
finite Neumann-only expansion. The **combined dictionary is not orthogonal**.
Only the interior modes are orthonormal under the cap area measure.

At the hemisphere, interior degrees are exactly `nu = m + 1 + 2j`. At smaller
fixed angles the corresponding roots are computed numerically. This pilot
supports cap half-angles from pi/6 through pi/2 and maximum degrees 1 through 32.
The degree setting specifies the same number/order of radial modes for each
angle; fractional eigenvalues at a smaller cap can exceed that setting. A
degree-12 hemisphere has 103 vector modes (309 coordinate coefficients).

Fit observations use three interior quadrature points per source triangle and
physical area weights, not vertex counts. Weighted least squares uses an
optional frequency penalty (`regularization=1e-8` by default) and records the
unregularized data rank and condition number. There is no per-crypt fitted basis.

The cap harmonic construction follows the domain/boundary distinction discussed
in [Shaqfa, Choi and Beyer (2021)](https://arxiv.org/abs/2107.14629).
The rim-extension plus interior dictionary is our implementation choice, not the
paper's Neumann-only surface expansion.

## Alignment and downstream geometry losses

`weighted_procrustes` is proper weighted Kabsch: rotation and translation only,
with reflections prohibited. `generalized_procrustes` handles matched arrays,
optionally normalizing their sampled RMS size. For representations, use
`align_representations`: it also searches cap angular phase by rotating each
sine/cosine coefficient pair exactly, with multiple local search starts.

The default alignment metric uses fixed physical vertex-area weights from the
initial template on a common cap sampling grid. These weights remain fixed
during optimization. The common rigid gauge is anchored to the initial surface;
the first training sample's cap phase is fixed to define the angular origin.
Every incoming sample has a free phase when using an already frozen reference.
The final template, weights, grid, basis ID and size convention form a reusable
`AlignmentReference`. Passing this reference for new crypts does not refit it.
Check the returned convergence flag/history. Angular search is numerical, not a
global-optimality guarantee. Approximately axial symmetry is reported separately;
it does not define a unique biological clock direction.

Spatial transforms use `y = x @ R.T / scale + translation` and have exact inverses.
Cap phase uses `X_new(theta, phi) = X_old(theta, phi + phase)` before spatial
alignment. An aligned representation's pose restores spatial coordinates but
leaves its angular reparameterization in place. To evaluate at an original
source vertex, subtract the stored phase from that vertex's cap angle first.

Because the dictionary is not orthogonal, raw Euclidean distances between its
coefficients are not surface distances. For the learning project, evaluate a
surface loss on the common grid with fixed weights, or use the metric
`G = B.T @ W @ B` with `B = basis.evaluate(reference_coordinates)`. Choosing and
training a latent model is outside this package.

## Interchange and reproducibility

`save_basis` serializes the domain, mode ordering, numerical eigenvalues and
normalizations. Its SHA-256 identifier must match each representation. Reuse this
saved object instead of independently rebuilding a supposedly equivalent basis
in a second environment. The cap sampling grid is independent of the analytic
basis; frozen alignment references also store its ring/sector resolution.

`save_representation` and `load_representation` handle individual numeric NPZ
files with JSON metadata (`allow_pickle=False`). Supplying the source patch and
mapping also stores source IDs, geometry and disk/cap coordinates for later field
transfer. The optional correspondence arrays remain accessible with `numpy.load`.
The HKS reader records source paths, dataset/sample IDs and a fingerprint of the
original mesh geometry. No changes are written to experimental input files.

## Families of openings and soft membership

```python
from organograph.crypt_morphology import (
    prepare_boundary_family, opening_shifts, evaluate_opening_family, surface_moments,
)
radius = surface_moments(patch.vertices, patch.faces)[2]
family = prepare_boundary_family(
    original_vertices, original_faces, crypt_vertex_ids,
    max_displacement=0.10 * radius, collar_width=0.30 * radius,
    forbidden_vertex_ids=other_crypt_vertex_ids,
)
result = evaluate_opening_family(
    family, basis, opening_shifts(radius), blur_width=0.05 * radius,
    store_samples=True,  # retain geometry for interactive inspection
)
```

The baseline rim defines signed **edge-graph distance** (negative inside) and an
arclength coordinate around the rim. Low-order sine/cosine displacement patterns
are extended off it with an inverse-edge-length harmonic graph solve. The cut is
the zero set of `signed_distance - displacement`. Constant displacements expand
or contract the opening; first angular modes move opposite sides in opposite
directions; second modes test more local asymmetry. These are sensitivity probes,
not estimates of anatomical uncertainty. Angular zero is an arbitrary mesh index;
several phases are tested and are not biological directions. Neither a crypt tip
nor a detector's longitudinal coordinate enters this construction.

Distance and displacement fields are interpolated linearly through parent
triangles. Cuts insert shared edge vertices and subdivide affected triangles,
preserving the original piecewise planar surface and winding. `ClippedPatch`
retains original face IDs plus a sparse barycentric interpolation matrix back to
the parent mesh; inserted vertices have `source_vertex_ids == -1`. Do not use
those IDs as direct parent indices. `transfer` supports interpolation of parent
data; biological field representation is still deferred to step 3.

Every accepted cut must remain a single manifold disk and preserve a fixed core.
Any triangle touching a neighboring crypt's selected vertices is protected;
entry rejects the cut. This is conservative and can reject cuts that only enter
the unlabelled portion of a mixed triangle. Rejections remain in the result table.
The finite surrounding support must itself be a disk. A setup failure should be
reported for the crypt, not silently repaired or dropped.

Three methods use the **same dictionary, shifts and physical core**:

| Method | Chart | Fit observations |
| --- | --- | --- |
| `recut_hard` | recomputed on each cut | exact cut surface |
| `fixed_chart_hard` | frozen on a padded parent patch | exact cut surface |
| `fixed_chart_soft` | the same frozen chart | padded surface, logistic membership |

For soft fitting, `w = expit(-level / blur_width)` multiplies the area-weighted
**squared residual** at each quadrature point, never the XYZ coordinates. Neighbor
faces have zero weight. The membrane is not geometrically smoothed. The two
fixed-chart methods deliberately share their computational support, so their
difference tests blurring rather than changing both chart and weights at once.
That support depends on the baseline opening: freezing it tests local robustness
around one detection, not independence from all possible initial detections.
Use `family.with_collar_width(new_width)` for support-width controls: it changes
the computational support while preserving the exact displacement fields.
Re-preparing a family with another collar would also change the angular
extension's solve band, confounding that comparison.

The unchanged physical core contains original triangles whose vertices all lie
deeper than the maximum permitted inward displacement. Core fidelity compares
predictions with actual membrane points; core drift compares predictions at
those **same points** with the method's baseline. Whole-cap drift uses the same
sampling grid and frozen baseline area weights, optimizing proper rigid pose
and cap phase against a frozen per-crypt/per-method baseline. No template is
relearned for a variant. These are paired sensitivity references, not the learned
cohort alignment reference used for downstream comparisons between crypts.

Physical units, fixed-original-RMS and recomputed-cut-RMS responses are reported.
For all methods the latter radius comes from the hard cut, never from the padded
support. Because the combined basis is nonorthogonal, coefficient-vector Euclidean
distances are not used. Baseline rows, failures, rank, conditioning, core fraction,
effective area, source provenance and settings are retained.

The fixed-chart hard fit extrapolates to unobserved cap regions; the soft fit only
weakly constrains its outer collar. A stable core does **not** establish a reliable
whole-cap opening. Inspect both metrics and reconstruction plots, and vary blur
width, support extent and mesh resolution before choosing a downstream training
representation. Edge-graph distances retain directional mesh bias; refinement
controls diagnose some discretization effects but do not remove that bias.

The existing HKS segmentation format itself contains object arrays and is read
with `allow_pickle=True`; use it only for trusted project-generated files.

## Validation and inspection

See [VALIDATION.md](VALIDATION.md) for step 1 and
[BOUNDARY_VALIDATION.md](BOUNDARY_VALIDATION.md) for the opening-family pilot and
its observed limitations. [DECODING_VALIDATION.md](DECODING_VALIDATION.md) records
the focused coefficient-decoding experiment.

- `notebooks/crypt_morphology/01_synthetic_validation.ipynb`: independent analytic
  shapes, resolution sweeps, original-unit restoration, rigid/phase alignment.
- `notebooks/crypt_morphology/02_real_crypt_inspection.ipynb`: HKS masks from
  20250929 and 20251201 at biological day 3.5 or later, blacklist-aware pilot
  selection, failure reporting, worst-case inspection and both size conventions.
- `notebooks/crypt_morphology/03_boundary_stability.ipynb`: prescribed uniform and
  asymmetric openings, three fit controls, fixed-core fidelity, size sensitivity,
  synthetic resolution checks, and blur/support sweeps on real examples.
- `notebooks/crypt_morphology/04_decoding_stability.ipynb`: recut/hard coefficient
  interpolation, controlled opening errors, geometric warnings at two mesh
  resolutions, source-pose controls and within/between-crypt input comparisons.
- Plot helpers are in `organograph.plotting.crypt_morphology`.
- Validation figures, CSVs, selected samples and executed notebook copies are
  generated under `exports/crypt_morphology/` (already ignored by Git).

```bash
PYTHONPATH=src python -m pytest tests/crypt_morphology -q
ORGANOGRAPH_DATA_ROOT=/path/to/NicoleData PYTHONPATH=src \
    python -m pytest tests/crypt_morphology -q
```

Unit tests cover independent geometric reconstructions, quadrature/size behavior,
mesh relabeling and remeshing, known rigid transforms, reflection rejection,
phase shifts, symmetric ambiguity, frozen references and artifact round trips.
The opt-in real-data test includes five dataset/timepoint groups and the pilot's
worst degree-12 example.

Geometric validation uses deterministic, area-weighted, bidirectional
point-to-triangle distances. The reported maximum is sampled, not a certified
Hausdorff bound. Degenerate reconstructed triangles are counted, but the current
diagnostics do **not** certify absence of self-intersections. Arbitrary coefficients
are not guaranteed to generate valid crypts. Reconstruction accuracy does not
establish boundary independence or biological correspondence.

## Direct decoding and imperfect coefficient predictions

```python
from organograph.crypt_morphology import (
    decode_coefficients, interpolate_coefficients, rim_coefficient_error,
    mesh_quality_metrics,
)
# Endpoints must already share a basis, size convention, and aligned frame.
middle = interpolate_coefficients(coefficients_a, coefficients_b, 0.5, basis)
vertices, faces = decode_coefficients(middle, basis)
delta = rim_coefficient_error(basis, rms_displacement=0.05, max_mode=3, seed=2026)
perturbed_vertices, _ = decode_coefficients(middle + delta, basis)
checks = mesh_quality_metrics(perturbed_vertices, faces, reference_vertices=vertices)
```

`decode_coefficients` returns XYZ by evaluating the fixed basis. No inverse mesh
parameterization, fitting, rigid alignment or renormalization is performed.
Generated arrays have no source pose, area or provenance; the caller supplies
their frame and units. Do not clone an observed `CapRepresentation` and assume
its physical measurements still describe arbitrary predicted coefficients.

Opening-error amplitudes specify RMS displacement under **uniform rim angle**.
Only nonconstant Fourier rim modes are modified; interior coefficients stay
unchanged. `transport_coefficient_error` expresses the same physical error under
another aligned pose, phase and size convention, allowing paired comparisons.
`coefficient_surface_distance` measures RMS XYZ changes on an explicitly frozen
grid/weight metric, rather than using a raw Euclidean coefficient norm.

`DecodingProbe` evaluates a fixed original core at source-triangle centroids.
Its correspondence displacement includes parameterization changes; its
point-to-triangle core error compares actual membrane points against generated
geometry. Both are measured in the baseline's frame, so the latter can still
include differing alignment or normalization of observed variants.
`source_frame_control` restores an **observed** variant to its known source pose
and the baseline size convention to distinguish fitting error from those
effects. It is unavailable for arbitrary generated arrays and never modifies
interpolation outputs. In particular, an accurate fitted shape can have a
substantially different rotation/phase when independently aligned to the same
cohort template.

`mesh_quality_metrics` reports collapsed triangles, intersections, area ratios,
edge stretches, and normal turns relative to a baseline. A >90-degree normal
turn is an inspection warning, not proof of a fold. No automatic biological
acceptance cutoff is assigned to errors or edge stretches. Topology/connectivity
is fixed by the cap grid; geometrical validity is a separate question.

The intersection checker implements the
[separating-axis method for convex polygons in 3D](https://www.geometrictools.com/Documentation/MethodOfSeparatingAxes.pdf),
including coplanar triangle overlap. Expected shared-edge/vertex contact is
excluded by contracting incident triangles by 1e-7 toward their centroids;
overlap confined to that tiny excluded strip can be missed. Axes/interval tests
are normalized separately per candidate pair (interval tolerance 1e-12).
This is a numerical test of the sampled triangle mesh, not a certificate for
the continuous expansion. The focused notebook checks selected cases on a
finer grid and keeps warnings separate from successful execution.

## Consistent alignment of known boundary families

`notebooks/crypt_morphology/05_consistent_family_alignment.ipynb` repeats the
interpolation test with a shared physical origin/orientation and a cap phase
determined by corresponding original core vertices. It compares this against
independent alignment of every cut to the same frozen cohort reference. The
baseline is aligned once; generated mixtures are never realigned. See
`FAMILY_ALIGNMENT_VALIDATION.md` for the executed experiment and its limits.

```python
from organograph.crypt_morphology import align_known_boundary_variant, vertex_areas

# core_face_mask selects the SAME original core triangles for the whole family.
weights = vertex_areas(baseline_patch.vertices, baseline_patch.faces[core_face_mask])
active = weights > 0
aligned_variant, phase_fit = align_known_boundary_variant(
    variant_representation, variant_patch, variant_mapping,
    aligned_baseline, baseline_patch, baseline_mapping, basis,
    baseline_patch.source_vertex_ids[active], weights[active],
)
```

The two mappings are their raw, unshifted charts. Representations may already
have nonzero stored phases or different poses. Original source IDs and matched
physical coordinates must agree; inserted cut vertices (`source_vertex_ids=-1`)
are excluded. At least three positively weighted core correspondences are
required. These are known correspondences on one parent membrane, not inferred
anatomical landmarks on unrelated crypts.

`shared_vertex_phase` fits one rotation of the disk by minimizing
`sum(w * abs(z_variant * exp(-i*delta) - z_baseline)**2)`, with `z=u+iv`.
The resulting variant phase is the baseline's stored phase plus `delta`.
Near-pole coordinates naturally have low angular leverage. The function reports
phase coherence and before/after disk RMS residuals; it rejects an undetermined
or incoherent phase. Even high coherence can coexist with nonrigid chart
differences. No radial/nonrigid warp is applied.

`reframe_representation` makes the required exact XYZ coefficient transform,
including the constant mode for translation and sine/cosine pairs for cap phase.
It preserves the fitted source surface and source measurements; inverse pose
plus the stored phase recovers its original coordinates. It does not refit.

By default `align_known_boundary_variant` preserves each variant's existing
normalization scale. All variants share `pose.rotation` and the physical origin
`pose.inverse([0, 0, 0])`; their numerical translations can differ when scale
differs. In physical mode all poses are identical. In unit-RMS mode, the original
per-cut radii remain in use, so size-normalization sensitivity remains measurable.
`preserve_variant_scale=False` additionally freezes the baseline scale as an
explicit normalization control; notebook 05 does not use that control in its
primary comparison.

This is suitable for keeping known augmentation families consistent. It does
not make the representation boundary invariant, establish valid arbitrary
coefficients, or solve orientation/correspondence between unrelated crypts.

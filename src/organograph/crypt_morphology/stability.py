"""Paired opening-sensitivity experiments, without boundary optimization.

Three controls separate recutting/reparameterization, chart freezing, and soft
membership. A fixed physical core supplies known point correspondences; the
whole-cap comparison also includes chart/extent changes and extrapolation.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.special import expit

from .alignment import align_representations, weighted_procrustes
from .parameterization import disk_to_cap, parameterize_patch
from .patches import surface_moments, triangle_areas, vertex_areas
from .representation import fit_representation, with_size_mode
from .boundaries import restrict_mapping


BARYCENTRIC = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
METHODS = ("recut_hard", "fixed_chart_hard", "fixed_chart_soft")


@dataclass
class StabilityResult:
    rows: list[dict]
    diagnostics: dict
    # Optional inspection samples: (representation, patch, mapping).
    samples: dict = field(default_factory=dict)
    support: object = None
    support_mapping: object = None


def _rmse(delta, weights):
    return float(np.sqrt(np.sum(weights[:, None] * delta**2)))


def _core_prediction(rep, patch, mapping, basis, parent_faces, parent_count):
    lookup = np.full(parent_count, -1, int)
    keep = patch.source_vertex_ids >= 0
    lookup[patch.source_vertex_ids[keep]] = np.flatnonzero(keep)
    local = lookup[parent_faces]
    if np.any(local < 0):
        raise ValueError("Variant failed to preserve the fixed physical core")
    uv = np.einsum("qa,fad->fqd", BARYCENTRIC, mapping.uv[local]).reshape(-1, 2)
    return rep.pose.inverse(basis.evaluate(disk_to_cap(uv, basis.half_angle)) @ rep.coefficients)


def evaluate_opening_family(family, basis, shifts, *, blur_width, regularization=1e-8,
                            store_samples=False):
    """Evaluate every requested variant and retain failures as explicit rows.

    ``shifts[0]`` must be the zero shift named 'baseline'. Distances, amplitudes,
    blur width and the original RMS radius are in parent-mesh physical units.
    The core is the original set of whole triangles whose vertices lie strictly
    deeper than the largest permitted inward shift. It never changes by variant.

    Physical and fixed-RMS cap drifts differ only by the fixed original radius.
    Recomputed-RMS uses the *hard cut's* radius for all three methods, so padding
    is not mistaken for crypt size. References are frozen per crypt and method;
    these paired references measure sensitivity, not cross-crypt correspondence.

    Soft membership is logistic(-level / blur_width), integrated at three points
    per support triangle. Protected neighbor faces receive zero weight. The
    finite support and regularization are experiment settings, not an inferred
    anatomical opening. Whole-cap reconstructions can extrapolate, especially
    for fixed_chart_hard; core fidelity and conditioning must be inspected too.
    """
    shifts = list(shifts)
    if not shifts or shifts[0].name != "baseline" or shifts[0].bound != 0:
        raise ValueError("The first shift must be the zero 'baseline'")
    if len({s.name for s in shifts}) != len(shifts):
        raise ValueError("Shift names must be unique")
    if not np.isfinite(blur_width) or blur_width <= 0:
        raise ValueError("blur_width must be finite and positive")
    for shift in shifts:
        family.level_set(shift)  # Validate settings before running expensive fits.
    original = family.original
    area0, _, radius0 = surface_moments(original.vertices, original.faces)
    original_faces = family.faces[original.source_face_ids]
    core_mask = np.max(family.signed_distance[original_faces], axis=1) < -family.max_displacement
    core_faces = original_faces[core_mask]
    if not len(core_faces):
        raise ValueError("No whole-triangle core remains at the requested displacement")
    core_area = triangle_areas(family.vertices, core_faces)
    core_weights = np.repeat(core_area / (3 * core_area.sum()), 3)
    truth = np.einsum("qa,fad->fqd", BARYCENTRIC, family.vertices[core_faces]).reshape(-1, 3)
    support = family.padded_patch()
    support_mapping = parameterize_patch(support.patch, half_angle=basis.half_angle)
    support_forbidden = family.forbidden_faces[support.patch.source_face_ids]
    rows, samples, fitted, predictions = [], {}, {}, {}
    diagnostics = dict(family.diagnostics, original_rms_radius=radius0, original_area=area0,
                       core_area_fraction=float(core_area.sum() / area0), core_n_faces=len(core_faces),
                       blur_width=float(blur_width), basis_id=basis.identifier,
                       max_degree=basis.max_degree, regularization=float(regularization),
                       protected_support_area_fraction=float(
                           triangle_areas(support.patch.vertices, support.patch.faces)[support_forbidden].sum()
                           / surface_moments(support.patch.vertices, support.patch.faces)[0]),
                       reference="frozen_per_crypt_per_method_baseline")
    for shift in shifts:
        common = dict(variant=shift.name, offset=shift.offset, displacement_bound=shift.bound,
                      displacement_fraction=shift.bound/radius0, original_rms_radius=radius0,
                      core_area_fraction=diagnostics["core_area_fraction"], blur_width=blur_width)
        try:
            cut = family.cut(shift)
            area, _, radius = surface_moments(cut.patch.vertices, cut.patch.faces)
            common.update(area_fraction=area/area0, rms_fraction=radius/radius0, cut_rms_radius=radius)
        except ValueError as error:
            rows.extend(dict(common, method=method, status="rejected_cut", error=str(error)) for method in METHODS)
            continue
        for method in METHODS:
            row = dict(common, method=method, status="ok", error="")
            try:
                confidence = None
                if method == "recut_hard":
                    patch = cut.patch
                    mapping = parameterize_patch(patch, half_angle=basis.half_angle)
                elif method == "fixed_chart_hard":
                    patch = cut.patch
                    mapping = restrict_mapping(cut, support, support_mapping)
                else:
                    patch, mapping = support.patch, support_mapping
                    level = support.transfer(family.level_set(shift))
                    quad_level = np.einsum("qa,fa->fq", BARYCENTRIC, level[patch.faces])
                    confidence = expit(-quad_level / blur_width)
                    confidence[support_forbidden] = 0
                rep = fit_representation(patch, mapping, basis, regularization=regularization,
                                         quadrature_weights=confidence,
                                         metadata={"opening_variant": shift.name, "method": method})
                pred = _core_prediction(rep, patch, mapping, basis, core_faces, len(family.vertices))
                key = (shift.name, method)
                fitted[key], predictions[key] = rep, pred
                if store_samples:
                    samples[key] = (rep, patch, mapping)
                row.update(core_fidelity_relative=_rmse(pred-truth, core_weights)/radius0,
                           condition_number=rep.diagnostics["condition_number"],
                           data_rank=rep.diagnostics["data_rank"], n_modes=basis.n_modes,
                           effective_area_fraction=rep.diagnostics["effective_area"]/area0,
                           fit_rmse_relative_rms=rep.diagnostics["fit_rmse_relative_rms"])
            except (ValueError, np.linalg.LinAlgError) as error:
                row.update(status="failed_fit", error=str(error))
            rows.append(row)

    # Freeze each method's baseline, all sampling weights, and the physical core.
    q, reference_faces = basis.reference_mesh(12, 48)
    matrix = basis.evaluate(q)
    for method in METHODS:
        baseline_key = ("baseline", method)
        valid = [row for row in rows if row["method"] == method and row["status"] == "ok"]
        if baseline_key not in fitted:
            for row in valid:
                row.update(status="missing_baseline", error="The method's baseline fit failed")
            continue
        base, base_pred = fitted[baseline_key], predictions[baseline_key]
        template = matrix @ base.coefficients
        weights = vertex_areas(template, reference_faces)
        reps = [fitted[(r["variant"], method)] for r in valid]
        try:
            physical = align_representations(reps, basis, template=template, weights=weights)
            scaled = [with_size_mode(rep, "unit_rms", radius=row["cut_rms_radius"])
                      for rep, row in zip(reps, valid)]
            normalized = align_representations(scaled, basis, template=template/radius0, weights=weights)
        except (ValueError, np.linalg.LinAlgError) as error:
            for row in valid:
                row.update(status="failed_alignment", error=str(error))
            continue
        for i, row in enumerate(valid):
            pred = predictions[(row["variant"], method)]
            rigid = weighted_procrustes(pred, base_pred, core_weights)
            scaled_pred = pred / row["cut_rms_radius"]
            scaled_pose = weighted_procrustes(scaled_pred, base_pred/radius0, core_weights)
            row.update(
                core_drift_relative=_rmse(pred-base_pred, core_weights)/radius0,
                core_drift_rigid_relative=_rmse(rigid.apply(pred)-base_pred, core_weights)/radius0,
                core_drift_unit_rms=_rmse(scaled_pose.apply(scaled_pred)-base_pred/radius0, core_weights),
                cap_drift_physical=_rmse(physical.aligned[i]-template, physical.weights),
                cap_drift_fixed_rms=_rmse(physical.aligned[i]-template, physical.weights)/radius0,
                cap_drift_unit_rms=_rmse(normalized.aligned[i]-template/radius0, normalized.weights),
                alignment_phase=float(physical.phases[i]), phase_ambiguous=bool(physical.phase_ambiguous[i]),
            )
    return StabilityResult(rows, diagnostics, samples, support, support_mapping)

"""Controlled alignment of known cuts from the same parent membrane.

Shared source IDs supply actual correspondence. This is a diagnostic and an
augmentation-family operation, not a solution for unrelated crypts. Neither
detector tips nor fitted biological axes are used. A single cap phase is fitted;
no chart warping, coefficient refitting or surface modification is performed.
"""
from dataclasses import dataclass, replace
import numpy as np

from .datatypes import SimilarityTransform


@dataclass(frozen=True)
class SharedVertexPhase:
    """Phase of the variant's raw chart relative to the reference's raw chart."""

    phase_offset: float
    coherence: float
    disk_rmse_before: float
    disk_rmse_after: float
    n_vertices: int


def _source_indices(patch, mapping, source_ids):
    ids = np.asarray(patch.source_vertex_ids)
    if ids.shape != (len(patch.vertices),) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Patch source IDs must be an integer vector matching its vertices")
    if mapping.uv.shape != (len(ids), 2) or not np.isfinite(mapping.uv).all():
        raise ValueError("Mapping must provide finite disk coordinates for every patch vertex")
    original = np.flatnonzero(ids >= 0)  # Inserted cut vertices have source ID -1.
    order = original[np.argsort(ids[original])]
    ordered = ids[order]
    if len(np.unique(ordered)) != len(ordered):
        raise ValueError("Original source vertex IDs must be unique within each patch")
    at = np.searchsorted(ordered, source_ids)
    if np.any(at == len(ordered)) or np.any(ordered[at] != source_ids):
        raise ValueError("Every requested core source vertex must be present in both patches")
    return order[at]


def shared_vertex_phase(patch, mapping, reference_patch, reference_mapping, source_ids, weights=None,
                        *, min_coherence=1e-6):
    """Find one disk rotation from matched original core vertices.

    Minimize sum_i w_i |z_variant_i exp(-i delta) - z_reference_i|²,
    with z = u + i v. The solution is arg(sum_i w_i z_variant_i conj(z_reference_i)).
    Disk radii naturally downweight points near the cap pole, where azimuth is
    poorly defined. Fixed physical-area weights should be supplied from the
    same original core for every variant. A low-coherence fit is rejected.

    IDs must refer to the same parent mesh: matched physical coordinates are
    checked as well as IDs. The nonzero residual includes radial and nonrigid
    chart differences that a single angular shift cannot remove.
    """
    source_ids = np.asarray(source_ids)
    if (source_ids.ndim != 1 or not np.issubdtype(source_ids.dtype, np.integer)
            or len(source_ids) < 3 or np.any(source_ids < 0)
            or len(np.unique(source_ids)) != len(source_ids)):
        raise ValueError("At least three distinct nonnegative source vertex IDs are required")
    if not np.isfinite(min_coherence) or not 0 < min_coherence <= 1:
        raise ValueError("min_coherence must lie in (0, 1]")
    if abs(mapping.half_angle-reference_mapping.half_angle) > 1e-12:
        raise ValueError("Mappings must use the same cap domain")
    a = _source_indices(patch, mapping, source_ids)
    b = _source_indices(reference_patch, reference_mapping, source_ids)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, float)
    if (w.shape != (len(a),) or not np.isfinite(w).all() or np.any(w < 0)
            or np.count_nonzero(w) < 3 or not np.isfinite(w.sum()) or w.sum() <= 0):
        raise ValueError("At least three positive finite weights matching source_ids are required")
    w = w/w.sum()
    xyz, reference_xyz = np.asarray(patch.vertices)[a], np.asarray(reference_patch.vertices)[b]
    extent = np.linalg.norm(np.ptp(reference_xyz, axis=0))
    if (not np.isfinite(xyz).all() or not np.isfinite(reference_xyz).all() or extent <= 0
            or np.linalg.norm(xyz-reference_xyz, axis=1).max() > 1e-9*extent):
        raise ValueError("Matched source IDs must identify the same physical membrane vertices")
    za = mapping.uv[a, 0]+1j*mapping.uv[a, 1]
    zb = reference_mapping.uv[b, 0]+1j*reference_mapping.uv[b, 1]
    cross = np.sum(w*za*np.conjugate(zb))
    lever = np.sum(w*np.abs(za)*np.abs(zb))
    if lever <= np.finfo(float).eps:
        raise ValueError("Core coordinates do not determine an angular phase")
    coherence = float(np.clip(abs(cross)/lever, 0, 1))
    if coherence < min_coherence:
        raise ValueError("Shared-vertex phase is ambiguous (low coherence)")
    phase = float(np.angle(cross))
    before = float(np.sqrt(np.sum(w*np.abs(za-zb)**2)))
    after = float(np.sqrt(np.sum(w*np.abs(za*np.exp(-1j*phase)-zb)**2)))
    return SharedVertexPhase(phase, coherence, before, after, len(a))


def reframe_representation(representation, basis, pose, *, phase=None):
    """Express an observed surface in a specified source-to-output pose and phase.

    This is exact algebra on XYZ coefficients, including their constant mode.
    Restoring original coordinates recovers the same surface, with the stated
    change of cap phase. Source measurements/provenance remain valid. The
    caller retains responsibility for the size-mode label if changing scale.
    """
    rep = representation
    if rep.basis_id != basis.identifier:
        raise ValueError("Representation basis mismatch")
    c = np.asarray(rep.coefficients, float)
    if c.shape != (basis.n_modes, 3) or not np.isfinite(c).all():
        raise ValueError("Invalid coordinate coefficients")
    phase = rep.phase if phase is None else float(phase)
    if not np.isfinite(phase):
        raise ValueError("phase must be finite")
    # Row-vector convention: source_pose.inverse(X), followed by target_pose.apply.
    linear = (rep.pose.rotation @ pose.rotation.T)*(rep.pose.scale/pose.scale)
    shifted = basis.phase_shift(c, phase-rep.phase)
    coefficients = shifted @ linear
    coefficients[0] += pose.translation-rep.pose.translation @ linear
    target_pose = SimilarityTransform(pose.rotation, pose.translation, pose.scale)
    return replace(rep, coefficients=coefficients, pose=target_pose, phase=phase % (2*np.pi))


def align_known_boundary_variant(representation, patch, mapping, reference, reference_patch,
                                reference_mapping, basis, source_ids, weights=None,
                                *, preserve_variant_scale=True):
    """Share a family's physical origin/orientation and fit phase by correspondence.

    ``reference`` is the baseline representation in the desired output frame.
    The mappings are raw patch charts, regardless of stored representation
    phases. By default each variant keeps its existing normalization scale;
    changing that scale preserves the SAME physical origin, not a new centroid.
    Setting preserve_variant_scale=False uses the baseline scale too, as an
    explicit normalization control. No independent rotation/translation search
    is performed. This requires known common-parent source correspondences.
    """
    if reference.basis_id != basis.identifier or representation.basis_id != basis.identifier:
        raise ValueError("Reference/variant basis mismatch")
    if representation.size_mode != reference.size_mode:
        raise ValueError("Reference and variant must use the same size convention")
    if abs(mapping.half_angle-basis.half_angle) > 1e-12:
        raise ValueError("Mapping and basis must use the same cap domain")
    fit = shared_vertex_phase(patch, mapping, reference_patch, reference_mapping, source_ids, weights)
    scale = representation.pose.scale if preserve_variant_scale else reference.pose.scale
    pose = SimilarityTransform(reference.pose.rotation,
                               reference.pose.translation*(reference.pose.scale/scale), scale)
    aligned = reframe_representation(representation, basis, pose, phase=reference.phase+fit.phase_offset)
    aligned.diagnostics = dict(aligned.diagnostics, alignment="known_parent_frame_shared_vertex_phase",
                               phase_coherence=fit.coherence, phase_disk_rmse=fit.disk_rmse_after,
                               phase_core_vertices=fit.n_vertices,
                               preserve_variant_scale=bool(preserve_variant_scale))
    return aligned, fit

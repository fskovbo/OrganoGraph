"""Proper rigid alignment, weighted cohort means and cap angular phase search."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.optimize import minimize_scalar

from .datatypes import AlignmentReference, ProcrustesResult, SimilarityTransform
from .patches import vertex_areas


def _weights(weights, n):
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, length N, with positive sum")
    if np.count_nonzero(w) < 3:
        raise ValueError("At least three positively weighted points are required")
    return w / w.sum()


def _rotation(covariance):
    u, singular, vt = np.linalg.svd(covariance)
    sign = 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0
    return (vt.T * np.array([1, 1, sign])) @ u.T, singular


def weighted_procrustes(source, target, weights=None):
    """Weighted Kabsch: translations/rotations only, never a reflection or scaling."""
    x, y = np.asarray(source, float), np.asarray(target, float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or not np.isfinite([x, y]).all():
        raise ValueError("source and target must be matching finite (N, 3) arrays")
    w = _weights(weights, len(x))
    cx, cy = w @ x, w @ y
    xc, yc = x - cx, y - cy
    rotation, _ = _rotation(xc.T @ (w[:, None] * yc))
    if np.linalg.matrix_rank(xc * np.sqrt(w[:, None])) < 2:
        raise ValueError("Rigid orientation is undefined for collinear/constant samples")
    return SimilarityTransform(rotation, cy - cx @ rotation.T)


def generalized_procrustes(shapes, weights=None, *, size_mode="physical", template=None,
                           max_iterations=100, tolerance=1e-9):
    """Align matched surface samples; a supplied template is frozen.

    Prefer physical template-area weights on common cap samples. This array API
    normalizes by sampled RMS when requested; fitting a CapRepresentation instead
    uses exact source-triangle moments. Per-sample weights can be (S, N).
    """
    x = np.asarray(shapes, float)
    if x.ndim != 3 or x.shape[2] != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("shapes must be finite (S, N, 3)")
    if size_mode not in ("physical", "unit_rms"):
        raise ValueError("Unknown size mode")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("Invalid convergence settings")
    raw_w = np.ones(x.shape[:2]) if weights is None else np.asarray(weights, float)
    if raw_w.ndim == 1:
        raw_w = np.broadcast_to(raw_w, x.shape[:2])
    if raw_w.shape != x.shape[:2]:
        raise ValueError("Weights must be (N,) or (S,N)")
    w = np.array([_weights(row, x.shape[1]) for row in raw_w])
    centered, normalization = [], []
    for points, wi in zip(x, w):
        center = wi @ points
        rms = np.sqrt(np.sum(wi[:, None] * (points - center) ** 2))
        if rms <= 0:
            raise ValueError("Zero-size shape")
        scale = rms if size_mode == "unit_rms" else 1.0
        pose = SimilarityTransform(translation=-center / scale, scale=scale)
        normalization.append(pose)
        centered.append(pose.apply(points))
    centered = np.array(centered)
    frozen = template is not None
    mean = centered[0].copy() if template is None else np.asarray(template, float).copy()
    if mean.shape != x.shape[1:] or not np.isfinite(mean).all():
        raise ValueError("Template must be finite (N,3) in the selected size convention")
    history, converged = [], False
    for _ in range(1 if frozen else max_iterations):
        poses = [weighted_procrustes(p, mean, wi) for p, wi in zip(centered, w)]
        aligned = np.array([pose.apply(p) for pose, p in zip(poses, centered)])
        denominator = w.sum(axis=0)
        updated = mean.copy()
        if not frozen:
            valid = denominator > 0
            updated[valid] = np.sum(w[:, :, None] * aligned, axis=0)[valid] / denominator[valid, None]
        history.append(float(np.sum(w[:, :, None] * (aligned - updated) ** 2)))
        change = np.linalg.norm(updated - mean) / max(np.linalg.norm(mean), 1e-15)
        mean = updated
        if frozen or change < tolerance:
            converged = True
            break
    transforms = [a.then(b) for a, b in zip(normalization, poses)]
    return ProcrustesResult(mean, aligned, transforms, history, converged, weights=w)


def _phase_alignment(coefficients, basis, matrix, weights, target, n_phase, previous_phase):
    """Evaluate the rigid fitting objective in coefficient space for efficiency."""
    mu = weights @ matrix
    gram = matrix.T @ (weights[:, None] * matrix) - np.outer(mu, mu)
    cy = weights @ target
    yc = target - cy
    cross = matrix.T @ (weights[:, None] * yc)
    target_var = np.sum(weights[:, None] * yc ** 2)

    def evaluate(phase, return_pose=False):
        c = basis.phase_shift(coefficients, phase)
        cx = mu @ c
        cov = c.T @ cross
        rotation, _ = _rotation(cov)
        value = max(0.0, float(np.sum(c * (gram @ c)) + target_var - 2 * np.trace(rotation @ cov)))
        if return_pose:
            return SimilarityTransform(rotation, cy - cx @ rotation.T), value
        return value

    phases = np.r_[np.arange(n_phase) * 2 * np.pi / n_phase, previous_phase]
    values = np.array([evaluate(a) for a in phases])
    best = int(np.argmin(values))
    phase, value = float(phases[best]), float(values[best])
    step = 2 * np.pi / n_phase
    # Multiple starts matter for shapes with approximate axial symmetry.
    for index in np.argsort(values)[:3]:
        fit = minimize_scalar(evaluate, bounds=(phases[index] - step, phases[index] + step),
                              method="bounded", options={"xatol": 1e-10})
        if fit.fun < value:
            phase, value = float(fit.x), float(fit.fun)
    pose, _ = evaluate(phase, True)
    ambiguous = np.ptp(values[:-1]) <= 1e-7 * max(float(target_var), 1e-15)
    return phase % (2 * np.pi), pose, bool(ambiguous)


def align_representations(representations, basis, *, template=None, weights=None,
                          n_rings=12, n_sectors=48, n_phase=32,
                          max_iterations=200, tolerance=1e-6):
    """Joint physical rotation and cap-phase alignment on a fixed reference grid.

    Default weights are physical vertex areas of the initial/frozen template and
    remain fixed throughout optimization. Save the template, weights and grid
    settings together for reproducible alignment of future samples.
    All inputs must share one size mode and exact basis identifier.
    When learning a template, its first sample anchors the angular origin. For a
    frozen template every incoming sample has a freely fitted angular phase.
    """
    reps = list(representations)
    if not reps or any(r.basis_id != basis.identifier for r in reps):
        raise ValueError("All representations must use this exact common basis")
    if len({r.size_mode for r in reps}) != 1:
        raise ValueError("Align physical and unit_rms cohorts separately")
    if n_phase < 8 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("Invalid phase/convergence settings")
    q, faces = basis.reference_mesh(n_rings, n_sectors)
    matrix = basis.evaluate(q)
    frozen = template is not None
    mean = matrix @ reps[0].coefficients if template is None else np.asarray(template, float).copy()
    if mean.shape != (len(q), 3) or not np.isfinite(mean).all():
        raise ValueError("Template must match the reference sampling grid")
    w = _weights(vertex_areas(mean, faces) if weights is None else weights, len(q))
    anchor = mean.copy()
    phases = np.zeros(len(reps))
    history, converged = [], False
    for _ in range(1 if frozen else max_iterations):
        poses, transformed, ambiguous = [], [], []
        for i, rep in enumerate(reps):
            phase, pose, amb = _phase_alignment(rep.coefficients, basis, matrix, w, mean, n_phase, phases[i])
            if i == 0 and not frozen:
                # Fix the cohort angular origin. Otherwise near-axial shapes
                # permit very slow common phase drift even after rigid anchoring.
                phase = 0.0
                pose = weighted_procrustes(matrix @ rep.coefficients, mean, w)
            phases[i] = phase
            c = basis.phase_shift(rep.coefficients, phase) @ pose.rotation.T
            c[0] += pose.translation  # constant rim mode equals one
            transformed.append(c)
            poses.append(pose)
            ambiguous.append(amb)
        aligned = np.array([matrix @ c for c in transformed])
        updated = mean if frozen else aligned.mean(axis=0)
        if not frozen:
            # The common rigid pose is a gauge freedom. Anchor it to the initial
            # reference so convergence measures shape change, not cohort drift.
            gauge = weighted_procrustes(updated, anchor, w)
            updated = gauge.apply(updated)
            aligned = gauge.apply(aligned)
            for i, c in enumerate(transformed):
                transformed[i] = c @ gauge.rotation.T
                transformed[i][0] += gauge.translation
            poses = [pose.then(gauge) for pose in poses]
        history.append(float(np.sum(w[None, :, None] * (aligned - updated) ** 2)))
        change = np.linalg.norm(updated - mean) / max(np.linalg.norm(mean), 1e-15)
        mean = updated
        if frozen or change < tolerance:
            converged = True
            break
    result_reps = [replace(r, coefficients=c, pose=r.pose.then(p),
                           phase=(r.phase + phase) % (2 * np.pi))
                   for r, c, p, phase in zip(reps, transformed, poses, phases)]
    return ProcrustesResult(mean, aligned, poses, history, converged, phases,
                            result_reps, np.asarray(ambiguous), w, (n_rings, n_sectors, n_phase))


def make_alignment_reference(result, basis, *, n_rings=None, n_sectors=None, n_phase=None):
    if result.representations is None or result.weights is None or result.sampling is None:
        raise ValueError("Reference requires a cap-representation alignment result")
    settings = tuple(given if given is not None else used for given, used in
                     zip((n_rings, n_sectors, n_phase), result.sampling))
    if settings != result.sampling:
        raise ValueError("Sampling settings do not match the result")
    n_rings, n_sectors, n_phase = settings
    if any(r.basis_id != basis.identifier for r in result.representations):
        raise ValueError("Basis mismatch")
    return AlignmentReference(result.template.copy(), result.weights.copy(), basis.identifier,
                              result.representations[0].size_mode, n_rings, n_sectors, n_phase)


def align_to_reference(representations, basis, reference):
    reps = list(representations)
    if reference.basis_id != basis.identifier or any(r.size_mode != reference.size_mode for r in reps):
        raise ValueError("Reference basis/size mode does not match inputs")
    return align_representations(reps, basis, template=reference.vertices, weights=reference.weights,
                                 n_rings=reference.n_rings, n_sectors=reference.n_sectors,
                                 n_phase=reference.n_phase)

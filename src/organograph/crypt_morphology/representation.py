"""Area-weighted surface fitting and reversible physical/shape-only modes."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.linalg import lstsq

from .datatypes import CapRepresentation, SimilarityTransform
from .patches import surface_moments, triangle_areas


def with_size_mode(representation, size_mode, *, radius=None):
    """Switch size convention without refitting; optionally use a fixed RMS radius.

    ``radius`` allows a boundary experiment to distinguish changes in geometry
    from changes in the patch's measured size. The source ``rms_radius`` remains
    unchanged; the actual normalization is always recorded in ``pose.scale``.
    """
    if size_mode not in ("physical", "unit_rms"):
        raise ValueError("size_mode must be 'physical' or 'unit_rms'")
    if radius is not None and (not np.isfinite(radius) or radius <= 0 or size_mode != "unit_rms"):
        raise ValueError("radius must be positive and is only valid for unit_rms")
    scale = (representation.rms_radius if radius is None else float(radius)) if size_mode == "unit_rms" else 1.0
    factor = scale / representation.pose.scale
    return replace(representation, coefficients=representation.coefficients / factor,
                   pose=representation.pose.then(SimilarityTransform(scale=factor)), size_mode=size_mode)


def fit_representation(patch, mapping, basis, *, size_mode="physical", regularization=1e-8,
                       metadata=None, quadrature_weights=None):
    """Fit a fixed dictionary using degree-two triangle quadrature.

    ``unit_rms`` divides by the exact area-weighted RMS radius of the original
    piecewise planar surface. Both modes remove translation and store its inverse.
    The original mesh geometry and topology are never modified.
    Optional nonnegative ``quadrature_weights`` have shape (n_faces, 3), in
    barycentric order (2/3,1/6,1/6) and its cyclic permutations. They weight
    residuals, never coordinates; effective area is recorded separately.
    """
    if size_mode not in ("physical", "unit_rms"):
        raise ValueError("size_mode must be 'physical' or 'unit_rms'")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if abs(mapping.half_angle - basis.half_angle) > 1e-12:
        raise ValueError("Mapping and basis use different cap domains")
    if mapping.uv.shape != (len(patch.vertices), 2):
        raise ValueError("Mapping does not match patch vertices")
    area, center, rms = surface_moments(patch.vertices, patch.faces)
    scale = rms if size_mode == "unit_rms" else 1.0
    pose = SimilarityTransform(translation=-center / scale, scale=scale)
    normalized = pose.apply(patch.vertices)
    # Sample actual triangle interiors, not just vertices used to construct the chart.
    from .parameterization import disk_to_cap
    bary = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
    uv = np.einsum("qa,fad->fqd", bary, mapping.uv[patch.faces]).reshape(-1, 2)
    targets = np.einsum("qa,fad->fqd", bary, normalized[patch.faces]).reshape(-1, 3)
    weights = np.repeat(triangle_areas(patch.vertices, patch.faces) / (3 * area), 3)
    if quadrature_weights is not None:
        confidence = np.asarray(quadrature_weights, float)
        if confidence.shape != (len(patch.faces), 3) or not np.isfinite(confidence).all() or np.any(confidence < 0):
            raise ValueError("quadrature_weights must be finite, nonnegative, and shaped (n_faces, 3)")
        weights *= confidence.ravel()
    weight_fraction = float(weights.sum())
    if weight_fraction <= 0:
        raise ValueError("At least one surface observation must have positive weight")
    weights /= weight_fraction
    matrix = basis.evaluate(disk_to_cap(uv, basis.half_angle))
    matrix *= np.sqrt(weights[:, None])
    rhs = targets * np.sqrt(weights[:, None])
    if matrix.shape[0] < basis.n_modes:
        raise ValueError("Insufficient surface observations for requested expansion order")
    # Compute data rank independently: regularization must not hide unobserved modes.
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * np.finfo(float).eps * max(matrix.shape)))
    if regularization:
        penalty = np.sqrt(regularization) * basis.regularization_weights()
        augmented = np.vstack([matrix, np.diag(penalty)])
        rhs_augmented = np.vstack([rhs, np.zeros((basis.n_modes, 3))])
    else:
        augmented, rhs_augmented = matrix, rhs
    coefficients = lstsq(augmented, rhs_augmented, lapack_driver="gelsy")[0]
    error = matrix @ coefficients - rhs
    physical_rmse = float(np.sqrt(np.sum(error ** 2)) * scale)
    diagnostics = dict(mapping.diagnostics)
    diagnostics.update({
        "data_rank": rank, "n_modes": basis.n_modes,
        "condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf"),
        "fit_rmse_physical": physical_rmse,
        "fit_rmse_relative_rms": physical_rmse / rms,
        "regularization": float(regularization),
        "effective_area": area * weight_fraction,
        "weight_fraction": weight_fraction,
        "n_vertices": len(patch.vertices), "n_faces": len(patch.faces),
        "unused_selected_vertices": len(patch.unused_selected_vertex_ids),
    })
    return CapRepresentation(coefficients, basis.identifier, pose, size_mode, area, rms,
                             diagnostics, dict(metadata or {}))


def reconstruct(representation, basis, theta_phi=None, *, original_coordinates=False,
                n_rings=24, n_sectors=96):
    """Return vertices and common faces, or vertices only at supplied cap points.

    An aligned representation retains its phase; original_coordinates restores
    spatial pose, while its surface is still parameterized with that phase.
    """
    if representation.basis_id != basis.identifier:
        raise ValueError("Representation and basis identifiers do not match")
    faces = None
    if theta_phi is None:
        theta_phi, faces = basis.reference_mesh(n_rings, n_sectors)
    vertices = basis.evaluate(theta_phi) @ representation.coefficients
    if original_coordinates:
        vertices = representation.pose.inverse(vertices)
    return (vertices, faces) if faces is not None else vertices

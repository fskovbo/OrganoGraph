"""Host-surface attachment points for detected crypt components.

The HKS detector supplies a surface patch and therefore an opening boundary.
This module projects that boundary to the fitted body or branch primitive and
places the attachment at the maximum-clearance point enclosed by the projected
ring. No geodesic-axis crossing or later attachment search is performed.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.optimize import minimize

from organograph.skeleton.detection.common import _coerce_patch
from organograph.skeleton.detection.mesh_regions import _boundary_vertices_from_patch
from organograph.skeleton.geometry import as_points
from organograph.skeleton.primitive.barriers import BarrierPrimitiveFit, barrier_primitive_level
from organograph.skeleton.primitive.blobs import blob_surface_radius


def _primitive_parameters(fit: BarrierPrimitiveFit) -> dict[str, Any]:
    return fit.to_primitive_parameters()


def _surface_point_from_angles(theta: float, phi: float, fit: BarrierPrimitiveFit) -> np.ndarray:
    local_direction = np.array(
        [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)],
        dtype=float,
    )
    world_direction = local_direction @ np.asarray(fit.axes, dtype=float).T
    radius = blob_surface_radius(
        _primitive_parameters(fit), fit.primitive_type, world_direction
    )
    return np.asarray(fit.center, dtype=float) + float(radius) * world_direction


def project_points_to_barrier_surface(points, fit: BarrierPrimitiveFit) -> np.ndarray:
    """Return approximate Euclidean closest points on a convex host primitive."""
    points = as_points(points)
    center = np.asarray(fit.center, dtype=float)
    axes = np.asarray(fit.axes, dtype=float)
    parameters = _primitive_parameters(fit)
    projected = []
    for point in points:
        offset = point - center
        norm = float(np.linalg.norm(offset))
        direction = axes[:, 0] if norm <= 1e-12 else offset / norm
        local = direction @ axes
        theta0 = float(np.arctan2(local[1], local[0]))
        phi0 = float(np.arctan2(local[2], np.hypot(local[0], local[1])))
        radial_radius = blob_surface_radius(parameters, fit.primitive_type, direction)
        radial = center + float(radial_radius) * direction

        def objective(angles):
            candidate = _surface_point_from_angles(angles[0], angles[1], fit)
            return float(np.sum((candidate - point) ** 2))

        result = minimize(
            objective,
            np.array([theta0, phi0]),
            method="L-BFGS-B",
            bounds=((-np.pi, np.pi), (-0.5 * np.pi, 0.5 * np.pi)),
            options={"maxiter": 80, "ftol": 1e-12},
        )
        projected.append(
            _surface_point_from_angles(result.x[0], result.x[1], fit)
            if result.success and np.all(np.isfinite(result.x))
            else radial
        )
    return np.asarray(projected, dtype=float)


def _radial_projection(points, fit: BarrierPrimitiveFit) -> np.ndarray:
    points = as_points(points)
    center = np.asarray(fit.center, dtype=float)
    offsets = points - center[None, :]
    levels = barrier_primitive_level(points, fit)
    valid = np.isfinite(levels) & (levels > 1e-12)
    projected = points.copy()
    projected[valid] = center + offsets[valid] / levels[valid, None]
    return projected


def barrier_surface_normal(point, fit: BarrierPrimitiveFit) -> np.ndarray:
    """Estimate the outward primitive normal from the level-set gradient."""
    point = np.asarray(point, dtype=float).reshape(3)
    step = 1e-5 * max(float(np.min(fit.radii)), 1.0)
    gradient = np.zeros(3, dtype=float)
    for axis in range(3):
        delta = np.zeros(3, dtype=float)
        delta[axis] = step
        hi = float(barrier_primitive_level((point + delta)[None, :], fit)[0])
        lo = float(barrier_primitive_level((point - delta)[None, :], fit)[0])
        gradient[axis] = (hi - lo) / (2.0 * step)
    norm = float(np.linalg.norm(gradient))
    if norm <= 1e-12 or not np.all(np.isfinite(gradient)):
        gradient = point - np.asarray(fit.center, dtype=float)
        norm = float(np.linalg.norm(gradient))
    return gradient / max(norm, 1e-12)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, reference)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.cross(normal, first)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    return first, second


def _inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        crossing = ((y1 > y) != (y0 > y)) & (
            x < (x0 - x1) * (y - y1) / (y0 - y1 + 1e-15) + x1
        )
        inside ^= crossing
        x0, y0 = x1, y1
    return inside


def _maximum_clearance_surface_point(
    projected_boundary: np.ndarray,
    fit: BarrierPrimitiveFit,
    *,
    grid_resolution: int = 31,
) -> tuple[np.ndarray, float]:
    direction = np.mean(projected_boundary, axis=0) - np.asarray(fit.center)
    if np.linalg.norm(direction) <= 1e-12:
        direction = projected_boundary[0] - np.asarray(fit.center)
    seed = _radial_projection(
        (np.asarray(fit.center) + direction)[None, :], fit
    )[0]
    normal = barrier_surface_normal(seed, fit)
    first, second = _plane_basis(normal)
    uv = np.column_stack(
        [(projected_boundary - seed) @ first, (projected_boundary - seed) @ second]
    )
    uv_center = np.mean(uv, axis=0)
    angles = np.arctan2(uv[:, 1] - uv_center[1], uv[:, 0] - uv_center[0])
    polygon = uv[np.argsort(angles)]
    lo = np.min(polygon, axis=0)
    hi = np.max(polygon, axis=0)
    resolution = max(9, int(grid_resolution))
    uu, vv = np.meshgrid(
        np.linspace(lo[0], hi[0], resolution),
        np.linspace(lo[1], hi[1], resolution),
    )
    candidates_uv = np.column_stack([uu.ravel(), vv.ravel()])
    candidates_uv = candidates_uv[_inside_polygon(candidates_uv, polygon)]
    if candidates_uv.size == 0:
        candidates_uv = uv_center[None, :]
    plane_points = (
        seed[None, :]
        + candidates_uv[:, :1] * first[None, :]
        + candidates_uv[:, 1:] * second[None, :]
    )
    # The dense search uses the primitive's homogeneous radial map. Exact
    # closest-point optimization is reserved for the observed boundary above.
    surface_points = _radial_projection(plane_points, fit)
    clearances = np.min(
        np.linalg.norm(surface_points[:, None, :] - projected_boundary[None, :, :], axis=2),
        axis=1,
    )
    selected = int(np.argmax(clearances))
    return surface_points[selected], float(clearances[selected])


def find_projected_opening_attachment(
    vertices,
    faces,
    crypt_vertices,
    host_fit: BarrierPrimitiveFit,
    *,
    grid_resolution: int = 31,
) -> dict[str, Any]:
    """Fit one host-surface attachment from a crypt candidate boundary."""
    vertices = as_points(vertices)
    patch = _coerce_patch(crypt_vertices)
    boundary = _boundary_vertices_from_patch(faces, patch)
    diagnostics: dict[str, Any] = {
        "found": False,
        "reason": "empty_candidate_boundary",
        "n_candidate_vertices": int(patch.size),
        "n_boundary_vertices": int(boundary.size),
    }
    if boundary.size < 3:
        return diagnostics
    projected = project_points_to_barrier_surface(vertices[boundary], host_fit)
    attachment, clearance = _maximum_clearance_surface_point(
        projected, host_fit, grid_resolution=grid_resolution
    )
    normal = barrier_surface_normal(attachment, host_fit)
    diagnostics.update(
        found=True,
        reason="projected_boundary_maximum_clearance",
        position=attachment,
        surface_normal=normal,
        clearance=clearance,
        primitive_level=float(barrier_primitive_level(attachment[None, :], host_fit)[0]),
        boundary_vertex_ids=boundary,
    )
    return diagnostics


def assign_crypt_attachments_from_projected_boundaries(
    vertices,
    faces,
    detections: list[dict[str, Any]],
    body_fit: BarrierPrimitiveFit,
    *,
    branch_fits: dict[str, BarrierPrimitiveFit] | None = None,
    assign_body_roots: bool = True,
    assign_branch_daughters: bool = True,
    grid_resolution: int = 31,
    metadata_key: str = "projected_opening_attachment",
) -> list[dict[str, Any]]:
    """Assign body- and branch-hosted crypt openings without changing patches."""
    vertices = as_points(vertices)
    branch_fits = dict(branch_fits or {})
    out = copy.deepcopy(detections)

    def update(detection, host_fit, host_id: str):
        metadata = dict(detection.get("metadata", {}))
        if host_fit is None:
            result = {"found": False, "reason": "missing_host_primitive"}
        else:
            result = find_projected_opening_attachment(
                vertices,
                faces,
                detection.get("crypt_vertices", []),
                host_fit,
                grid_resolution=grid_resolution,
            )
        result["host_id"] = str(host_id)
        if result.get("found"):
            detection["attachment_position"] = np.asarray(result["position"], dtype=float)
            detection["attachment_surface_normal"] = np.asarray(
                result["surface_normal"], dtype=float
            )
            detection["candidate_boundary_vertices"] = np.asarray(
                result["boundary_vertex_ids"], dtype=np.int64
            )
            detection["candidate_crypt_vertices"] = _coerce_patch(
                detection.get("crypt_vertices", [])
            )
        metadata[metadata_key] = {
            key: value for key, value in result.items() if key != "boundary_vertex_ids"
        }
        detection["metadata"] = metadata

    for detection in out:
        if assign_body_roots:
            update(detection, body_fit, "body")
        if not assign_branch_daughters:
            continue
        daughters = detection.get("daughters") or []
        if not daughters:
            continue
        branch_id = f"crypt_{detection.get('crypt_id')}_branch"
        for daughter in daughters:
            update(daughter, branch_fits.get(branch_id), branch_id)
    return out


def attachment_projection_diagnostics(
    detections: list[dict[str, Any]],
    *,
    metadata_key: str = "projected_opening_attachment",
) -> list[dict[str, Any]]:
    """Flatten projected-opening diagnostics for batch quality control."""
    records: list[dict[str, Any]] = []

    def collect(detection, parent_id=None):
        crypt_id = detection.get("crypt_id")
        records.append(
            {
                "crypt_id": crypt_id,
                "parent_crypt_id": parent_id,
                **dict(detection.get("metadata", {}).get(metadata_key, {})),
            }
        )
        for daughter in detection.get("daughters") or []:
            collect(daughter, crypt_id)

    for detection in detections:
        collect(detection)
    return records


__all__ = [
    "assign_crypt_attachments_from_projected_boundaries",
    "attachment_projection_diagnostics",
    "barrier_surface_normal",
    "find_projected_opening_attachment",
    "project_points_to_barrier_surface",
]

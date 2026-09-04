"""Attachment points and boundary refinement for detected crypt components.

The HKS detector supplies a surface patch and therefore an opening boundary.
This module projects that boundary to the fitted body or branch primitive and
places the attachment at the maximum-clearance point enclosed by the projected
ring. The embedded strategy first grows the candidate to a circumference-based
boundary and, when necessary, continues growing its geodesic sublevel region
until the planar opening center reaches the host primitive. Its opening frame
always comes from the closest point on the host surface.
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


def _maximum_clearance_polygon_point(
    polygon: np.ndarray,
    *,
    grid_resolution: int,
) -> tuple[np.ndarray, float]:
    """Return a grid approximation of the pole of inaccessibility in 2D."""
    polygon = np.asarray(polygon, dtype=float)
    center = np.mean(polygon, axis=0)
    angles = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
    polygon = polygon[np.argsort(angles)]
    lo = np.min(polygon, axis=0)
    hi = np.max(polygon, axis=0)
    resolution = max(9, int(grid_resolution))
    uu, vv = np.meshgrid(
        np.linspace(lo[0], hi[0], resolution),
        np.linspace(lo[1], hi[1], resolution),
    )
    candidates = np.column_stack([uu.ravel(), vv.ravel()])
    candidates = candidates[_inside_polygon(candidates, polygon)]
    if candidates.size == 0:
        candidates = center[None, :]
    clearances = np.min(
        np.linalg.norm(candidates[:, None, :] - polygon[None, :, :], axis=2),
        axis=1,
    )
    selected = int(np.argmax(clearances))
    return candidates[selected], float(clearances[selected])


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


def find_embedded_opening_attachment(
    vertices,
    faces,
    crypt_vertices,
    host_fit: BarrierPrimitiveFit,
    *,
    tip_position=None,
    grid_resolution: int = 31,
) -> dict[str, Any]:
    """Place an attachment at the planar center of the original boundary ring."""
    vertices = as_points(vertices)
    patch = _coerce_patch(crypt_vertices)
    boundary = _boundary_vertices_from_patch(faces, patch)
    diagnostics: dict[str, Any] = {
        "found": False,
        "reason": "empty_candidate_boundary",
        "strategy": "embedded_boundary_plane",
        "n_candidate_vertices": int(patch.size),
        "n_boundary_vertices": int(boundary.size),
    }
    if boundary.size < 3:
        return diagnostics
    points = vertices[boundary]
    plane_center = np.mean(points, axis=0)
    centered = points - plane_center
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if vh.shape != (3, 3) or singular_values[1] <= 1e-12:
        diagnostics["reason"] = "degenerate_candidate_boundary_plane"
        return diagnostics
    first = vh[0] / max(float(np.linalg.norm(vh[0])), 1e-12)
    normal = vh[-1] / max(float(np.linalg.norm(vh[-1])), 1e-12)
    second = np.cross(normal, first)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    preferred = None
    if tip_position is not None:
        candidate = np.asarray(tip_position, dtype=float).reshape(-1)
        if candidate.size == 3 and np.all(np.isfinite(candidate)):
            preferred = candidate - plane_center
    if preferred is None or np.linalg.norm(preferred) <= 1e-12:
        preferred = plane_center - np.asarray(host_fit.center, dtype=float)
    if float(np.dot(normal, preferred)) < 0.0:
        normal = -normal
        second = -second
    uv = np.column_stack([centered @ first, centered @ second])
    center_uv, clearance = _maximum_clearance_polygon_point(
        uv, grid_resolution=grid_resolution
    )
    attachment = plane_center + center_uv[0] * first + center_uv[1] * second
    primitive_level = float(barrier_primitive_level(attachment[None, :], host_fit)[0])
    plane_residuals = np.abs(centered @ normal)
    diagnostics.update(
        found=True,
        reason="embedded_boundary_plane_maximum_clearance",
        position=attachment,
        surface_normal=normal,
        normal_source="candidate_boundary_plane",
        clearance=clearance,
        primitive_level=primitive_level,
        embedded_in_host=bool(primitive_level <= 1.0 + 1e-6),
        boundary_plane_center=plane_center,
        boundary_plane_normal=normal,
        boundary_plane_rmse=float(np.sqrt(np.mean(plane_residuals**2))),
        boundary_vertex_ids=boundary,
    )
    return diagnostics


def _apply_closest_host_surface_frame(
    result: dict[str, Any],
    host_fit: BarrierPrimitiveFit,
) -> dict[str, Any]:
    """Use the closest host surface for the opening normal and outside fallback."""
    if not result.get("found", False):
        return result
    position = np.asarray(result["position"], dtype=float).reshape(3)
    surface_point = project_points_to_barrier_surface(position[None, :], host_fit)[0]
    normal = barrier_surface_normal(surface_point, host_fit)
    original_level = float(barrier_primitive_level(position[None, :], host_fit)[0])
    outside = bool(original_level > 1.0 + 1e-6)
    result.update(
        boundary_plane_normal=result.get("surface_normal"),
        host_surface_reference_position=surface_point,
        surface_normal=normal,
        normal_source="closest_host_primitive_surface_gradient",
        attachment_projected_to_host_surface=outside,
        unconstrained_primitive_level=original_level,
    )
    if outside:
        result["unconstrained_position"] = position
        result["position"] = surface_point
        result["primitive_level"] = float(
            barrier_primitive_level(surface_point[None, :], host_fit)[0]
        )
        result["embedded_in_host"] = True
        result["attachment_position_reason"] = (
            "closest_host_surface_fallback_after_growth_limit"
        )
    else:
        result["attachment_position_reason"] = "refined_boundary_plane_center"
    return result


def refine_embedded_opening_attachment(
    vertices,
    faces,
    detection: dict[str, Any],
    host_fit: BarrierPrimitiveFit,
    *,
    grid_resolution: int = 31,
    max_mesh_fraction: float = 0.35,
    max_search_evaluations: int = 64,
) -> tuple[dict[str, Any], np.ndarray]:
    """Grow a profiled crypt boundary until its planar center reaches its host.

    The circumference normalization performed upstream places the selected
    local minimum or second-derivative transition at distance level 1. The
    corresponding geodesic sublevel region is the first refined component. If
    its planar maximum-clearance center remains outside the host primitive, the
    level is advanced until the first host contact. Growth never consumes more
    than ``max_mesh_fraction`` of the complete mesh.
    """
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    initial_patch = _coerce_patch(detection.get("crypt_vertices", []))
    tip_position = _tip_position_for_attachment(vertices, detection)
    distance_field = detection.get("d_crypt")
    if distance_field is None:
        result = find_embedded_opening_attachment(
            vertices,
            faces,
            initial_patch,
            host_fit,
            tip_position=tip_position,
            grid_resolution=grid_resolution,
        )
        result.update(
            boundary_refined=False,
            boundary_refinement_reason="missing_distance_field",
            host_contact_found=bool(result.get("embedded_in_host", False)),
            initial_region_size=int(initial_patch.size),
            final_region_size=int(initial_patch.size),
        )
        return result, initial_patch

    distance_field = np.asarray(distance_field, dtype=float).reshape(-1)
    if distance_field.size != vertices.shape[0]:
        result = find_embedded_opening_attachment(
            vertices,
            faces,
            initial_patch,
            host_fit,
            tip_position=tip_position,
            grid_resolution=grid_resolution,
        )
        result.update(
            boundary_refined=False,
            boundary_refinement_reason="invalid_distance_field",
            host_contact_found=bool(result.get("embedded_in_host", False)),
            initial_region_size=int(initial_patch.size),
            final_region_size=int(initial_patch.size),
        )
        return result, initial_patch

    finite = np.isfinite(distance_field) & (distance_field >= 0.0)
    initial_boundary = _boundary_vertices_from_patch(faces, initial_patch)
    initial_boundary_values = distance_field[initial_boundary]
    initial_boundary_values = initial_boundary_values[
        np.isfinite(initial_boundary_values)
    ]
    initial_boundary_level = (
        float(np.mean(initial_boundary_values))
        if initial_boundary_values.size
        else 1.0
    )
    initial_boundary_level = max(initial_boundary_level, 1e-12)
    profile_level = 1.0
    profile_region = np.flatnonzero(finite & (distance_field <= profile_level))
    if initial_patch.size:
        profile_region = np.unique(np.concatenate([initial_patch, profile_region]))

    maximum_size = max(
        int(initial_patch.size),
        int(np.floor(float(max_mesh_fraction) * vertices.shape[0])),
        3,
    )
    profile_limited_by_size = bool(profile_region.size > maximum_size)
    if profile_limited_by_size:
        profile_region = initial_patch.copy()
    evaluations: list[dict[str, Any]] = []

    def evaluate(level: float):
        grown = np.flatnonzero(finite & (distance_field <= float(level)))
        region = (
            np.unique(np.concatenate([initial_patch, grown]))
            if initial_patch.size
            else grown
        )
        if region.size > maximum_size and region.size > profile_region.size:
            return None
        opening = find_embedded_opening_attachment(
            vertices,
            faces,
            region,
            host_fit,
            tip_position=tip_position,
            grid_resolution=grid_resolution,
        )
        if opening.get("found"):
            evaluations.append(
                {
                    "level": float(level),
                    "region_size": int(region.size),
                    "primitive_level": float(opening["primitive_level"]),
                    "inside_host": bool(opening["embedded_in_host"]),
                }
            )
        return opening, region

    selected_level = profile_level
    evaluated = evaluate(profile_level)
    if evaluated is None:
        evaluated = (
            find_embedded_opening_attachment(
                vertices,
                faces,
                profile_region,
                host_fit,
                tip_position=tip_position,
                grid_resolution=grid_resolution,
            ),
            profile_region,
        )
    selected_opening, selected_region = evaluated
    host_contact_found = bool(selected_opening.get("embedded_in_host", False))

    finite_levels = np.unique(distance_field[finite & (distance_field > profile_level)])
    if finite_levels.size and not host_contact_found:
        admissible = []
        for level in finite_levels:
            region_size = int(np.count_nonzero(finite & (distance_field <= level)))
            if region_size > maximum_size:
                break
            admissible.append(float(level))
        if admissible:
            admissible = np.asarray(admissible, dtype=float)
            count = min(max(2, int(max_search_evaluations)), admissible.size)
            coarse_indices = np.unique(
                np.linspace(0, admissible.size - 1, count).astype(np.int64)
            )
            previous_index = -1
            for index in coarse_indices:
                trial = evaluate(float(admissible[index]))
                if trial is None:
                    break
                opening, region = trial
                if opening.get("found") and (
                    not selected_opening.get("found")
                    or float(opening["primitive_level"])
                    < float(selected_opening["primitive_level"])
                ):
                    selected_opening = opening
                    selected_region = region
                    selected_level = float(admissible[index])
                if opening.get("embedded_in_host", False):
                    low = previous_index + 1
                    high = int(index)
                    while low < high:
                        middle = (low + high) // 2
                        middle_trial = evaluate(float(admissible[middle]))
                        if middle_trial is None:
                            high = middle
                            continue
                        middle_opening, _ = middle_trial
                        if middle_opening.get("embedded_in_host", False):
                            high = middle
                        else:
                            low = middle + 1
                    contact_trial = evaluate(float(admissible[low]))
                    if contact_trial is not None:
                        selected_opening, selected_region = contact_trial
                        selected_level = float(admissible[low])
                        host_contact_found = bool(
                            selected_opening.get("embedded_in_host", False)
                        )
                    break
                previous_index = int(index)

    neck_profile = detection.get("neck_profile")
    neck_profile = neck_profile if isinstance(neck_profile, dict) else {}
    selected_opening.update(
        strategy="embedded_boundary_plane",
        normal_source="refined_crypt_boundary_plane",
        boundary_refined=True,
        boundary_refinement_reason=(
            "profile_boundary_reached_host"
            if selected_level <= profile_level + 1e-12 and host_contact_found
            else "grown_to_host_primitive"
            if host_contact_found
            else "profile_boundary_exceeds_mesh_fraction_limit"
            if profile_limited_by_size
            else "host_not_reached_before_mesh_fraction_limit"
        ),
        host_contact_found=host_contact_found,
        profile_feature_kind=neck_profile.get("kind"),
        profile_feature_reason=neck_profile.get("reason"),
        initial_boundary_level=float(initial_boundary_level),
        profile_boundary_level=profile_level,
        profile_boundary_distance_factor=float(
            profile_level / initial_boundary_level
        ),
        final_boundary_level=float(selected_level),
        final_boundary_distance_factor=float(
            selected_level / initial_boundary_level
        ),
        initial_region_size=int(initial_patch.size),
        profile_region_size=int(profile_region.size),
        final_region_size=int(selected_region.size),
        maximum_region_size=int(maximum_size),
        max_mesh_fraction=float(max_mesh_fraction),
        profile_limited_by_mesh_fraction=profile_limited_by_size,
        growth_evaluations=evaluations,
    )
    return selected_opening, np.asarray(selected_region, dtype=np.int64)


def _tip_position_for_attachment(vertices, detection) -> np.ndarray | None:
    for key in ("tip_position", "tip_center", "bottom_position", "crypt_tip", "p_tip"):
        value = detection.get(key)
        if value is not None:
            point = np.asarray(value, dtype=float).reshape(-1)
            if point.size == 3 and np.all(np.isfinite(point)):
                return point
    for key in ("bottom_vertex_id", "tip_vertex_id", "boundary_distance_bottom_vertex_id"):
        value = detection.get(key)
        if value is None:
            continue
        try:
            vertex_id = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= vertex_id < vertices.shape[0]:
            return vertices[vertex_id]
    return None


def assign_crypt_attachments(
    vertices,
    faces,
    detections: list[dict[str, Any]],
    body_fit: BarrierPrimitiveFit,
    *,
    branch_fits: dict[str, BarrierPrimitiveFit] | None = None,
    assign_body_roots: bool = True,
    assign_branch_daughters: bool = True,
    grid_resolution: int = 31,
    boundary_refinement_max_mesh_fraction: float = 0.35,
    strategy: str = "host_surface",
    metadata_key: str = "opening_attachment",
) -> list[dict[str, Any]]:
    """Assign body- and branch-hosted openings using the selected strategy."""
    vertices = as_points(vertices)
    branch_fits = dict(branch_fits or {})
    strategy = str(strategy)
    if strategy not in {"host_surface", "embedded_boundary_plane"}:
        raise ValueError(
            "attachment strategy must be 'host_surface' or "
            "'embedded_boundary_plane'"
        )
    out = copy.deepcopy(detections)

    def update(detection, host_fit, host_id: str):
        metadata = dict(detection.get("metadata", {}))
        if host_fit is None:
            result = {"found": False, "reason": "missing_host_primitive"}
        elif strategy == "embedded_boundary_plane":
            result, refined_region = refine_embedded_opening_attachment(
                vertices,
                faces,
                detection,
                host_fit,
                grid_resolution=grid_resolution,
                max_mesh_fraction=boundary_refinement_max_mesh_fraction,
            )
            result = _apply_closest_host_surface_frame(result, host_fit)
            detection.setdefault(
                "initial_candidate_crypt_vertices",
                _coerce_patch(detection.get("crypt_vertices", [])),
            )
            detection["crypt_vertices"] = refined_region
            detection["attachment_region_vertices"] = refined_region
            detection["attachment_level"] = float(
                result.get("final_boundary_level", 1.0)
            )
            profile = detection.get("neck_profile")
            if isinstance(profile, dict):
                profile = dict(profile)
                profile.setdefault(
                    "profile_attachment_level", profile.get("attachment_level", 1.0)
                )
                profile["attachment_level"] = detection["attachment_level"]
                detection["neck_profile"] = profile
        else:
            result = find_projected_opening_attachment(
                vertices,
                faces,
                detection.get("crypt_vertices", []),
                host_fit,
                grid_resolution=grid_resolution,
            )
            result["strategy"] = "host_surface"
            result["normal_source"] = "host_primitive_surface_gradient"
        result["host_id"] = str(host_id)
        if result.get("found"):
            detection["attachment_position"] = np.asarray(result["position"], dtype=float)
            detection["attachment_surface_normal"] = np.asarray(
                result["surface_normal"], dtype=float
            )
            detection["attachment_normal_source"] = str(result["normal_source"])
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
    """Compatibility wrapper for the original host-surface strategy."""
    return assign_crypt_attachments(
        vertices,
        faces,
        detections,
        body_fit,
        branch_fits=branch_fits,
        assign_body_roots=assign_body_roots,
        assign_branch_daughters=assign_branch_daughters,
        grid_resolution=grid_resolution,
        boundary_refinement_max_mesh_fraction=1.0,
        strategy="host_surface",
        metadata_key=metadata_key,
    )


def attachment_projection_diagnostics(
    detections: list[dict[str, Any]],
    *,
    metadata_key: str = "opening_attachment",
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
    "assign_crypt_attachments",
    "assign_crypt_attachments_from_projected_boundaries",
    "attachment_projection_diagnostics",
    "barrier_surface_normal",
    "find_projected_opening_attachment",
    "find_embedded_opening_attachment",
    "refine_embedded_opening_attachment",
    "project_points_to_barrier_surface",
]

"""Assign appendix attachments from crossings of fitted host primitives.

The HKS and circumference machinery still identifies appendices, tips, and
genuine constrictions.  Host-side attachment points are instead defined by the
first persistent crossing of the body or branch barrier primitive while moving
away from the tip along the geodesic crypt coordinate.  This keeps component
ownership and skeleton junctions tied to the same coarse host representation.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from organograph.skeleton.barrier_ellipsoid import (
    SoftBarrierEllipsoidFit,
    barrier_primitive_level,
)
from organograph.skeleton.detection.common import _coerce_patch
from organograph.skeleton.detection.mesh_regions import _mesh_edges_from_faces
from organograph.skeleton.detection.neck_profiles import (
    _contour_center_from_distance_field,
)
from organograph.skeleton.geometry import as_points


def _tip_connected_region(
    faces,
    distance_field,
    attachment_level: float,
    tip_vertex_id: int | None,
) -> np.ndarray:
    """Return the thresholded component connected to the crypt tip."""
    dfield = np.asarray(distance_field, dtype=float).reshape(-1)
    allowed = np.isfinite(dfield) & (dfield <= float(attachment_level) + 1e-12)
    if not np.any(allowed):
        return np.empty(0, dtype=np.int64)

    if tip_vertex_id is None or not (0 <= int(tip_vertex_id) < dfield.size):
        return np.where(allowed)[0].astype(np.int64)
    tip = int(tip_vertex_id)
    if not allowed[tip]:
        return np.where(allowed)[0].astype(np.int64)

    edges = _mesh_edges_from_faces(faces)
    adjacency = [[] for _ in range(dfield.size)]
    valid_edges = edges[allowed[edges[:, 0]] & allowed[edges[:, 1]]]
    for a, b in valid_edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))

    visited = np.zeros(dfield.size, dtype=bool)
    visited[tip] = True
    stack = [tip]
    while stack:
        vertex = stack.pop()
        for neighbor in adjacency[vertex]:
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
    return np.where(visited)[0].astype(np.int64)


def _evaluate_contour_level(
    vertices,
    faces,
    distance_field,
    level: float,
    host_fit: SoftBarrierEllipsoidFit,
    *,
    prefer_vertices=None,
) -> tuple[np.ndarray | None, float | None]:
    center = _contour_center_from_distance_field(
        vertices,
        faces,
        distance_field,
        level=float(level),
        prefer_vertices=prefer_vertices,
    )
    if center is None:
        return None, None
    primitive_level = float(barrier_primitive_level(center[None, :], host_fit)[0])
    if not np.isfinite(primitive_level):
        return center, None
    return center, primitive_level


def find_barrier_boundary_crossing(
    vertices,
    faces,
    distance_field,
    host_fit: SoftBarrierEllipsoidFit,
    *,
    prefer_vertices=None,
    surface_level: float = 1.0,
    min_axis_level: float = 0.03,
    max_axis_level: float = 2.0,
    n_samples: int = 40,
    persistence: int = 2,
    bisection_iterations: int = 8,
) -> dict[str, Any]:
    """Find the first outside-to-inside host crossing along a crypt axis.

    Ring centers are sampled in increasing geodesic distance from the tip.  A
    crossing is accepted only when the requested number of subsequent valid
    rings remain inside the host primitive, which suppresses isolated noisy
    intersections.  The crossing level is then refined by bisection.
    """
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    dfield = np.asarray(distance_field, dtype=float).reshape(-1)
    target = float(surface_level)
    if dfield.size != vertices.shape[0]:
        raise ValueError("distance_field must contain one value per mesh vertex")
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("surface_level must be finite and positive")
    if int(n_samples) < 4:
        raise ValueError("n_samples must be at least 4")

    finite = dfield[np.isfinite(dfield)]
    diagnostics: dict[str, Any] = {
        "found": False,
        "reason": "not_evaluated",
        "surface_level": target,
        "min_axis_level": float(min_axis_level),
        "max_axis_level": float(max_axis_level),
        "n_samples": int(n_samples),
        "persistence": max(int(persistence), 1),
        "sample_axis_levels": [],
        "sample_primitive_levels": [],
        "sample_centers": [],
    }
    if finite.size == 0:
        diagnostics["reason"] = "empty_distance_field"
        return diagnostics

    lo = max(float(min_axis_level), float(np.nanmin(finite)))
    hi = min(float(max_axis_level), float(np.nanmax(finite)))
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        diagnostics["reason"] = "invalid_axis_interval"
        return diagnostics

    sample_levels = np.linspace(lo, hi, int(n_samples))
    samples = []
    for axis_level in sample_levels:
        center, primitive_level = _evaluate_contour_level(
            vertices,
            faces,
            dfield,
            axis_level,
            host_fit,
            prefer_vertices=prefer_vertices,
        )
        if center is None or primitive_level is None:
            continue
        samples.append((float(axis_level), center, float(primitive_level)))

    diagnostics["sample_axis_levels"] = [item[0] for item in samples]
    diagnostics["sample_centers"] = [item[1].tolist() for item in samples]
    diagnostics["sample_primitive_levels"] = [item[2] for item in samples]
    if len(samples) < 2:
        diagnostics["reason"] = "insufficient_valid_contours"
        return diagnostics

    required_inside = max(int(persistence), 1)
    bracket = None
    for i in range(1, len(samples)):
        previous = samples[i - 1]
        current = samples[i]
        if previous[2] <= target or current[2] > target:
            continue
        following = samples[i : i + required_inside]
        if len(following) < required_inside:
            continue
        if all(item[2] <= target for item in following):
            bracket = (previous, current)
            break

    if bracket is None:
        if samples[0][2] <= target:
            diagnostics["reason"] = "tip_side_contours_already_inside_host"
        elif all(item[2] > target for item in samples):
            diagnostics["reason"] = "host_boundary_not_reached"
        else:
            diagnostics["reason"] = "no_persistent_outside_to_inside_crossing"
        return diagnostics

    outside, inside = bracket
    outside_level, outside_center, outside_value = outside
    inside_level, inside_center, inside_value = inside
    for _ in range(max(int(bisection_iterations), 0)):
        middle_level = 0.5 * (outside_level + inside_level)
        middle_center, middle_value = _evaluate_contour_level(
            vertices,
            faces,
            dfield,
            middle_level,
            host_fit,
            prefer_vertices=prefer_vertices,
        )
        if middle_center is None or middle_value is None:
            break
        if middle_value > target:
            outside_level, outside_center, outside_value = (
                middle_level,
                middle_center,
                middle_value,
            )
        else:
            inside_level, inside_center, inside_value = (
                middle_level,
                middle_center,
                middle_value,
            )

    denominator = outside_value - inside_value
    fraction = (
        float(np.clip((outside_value - target) / denominator, 0.0, 1.0))
        if abs(denominator) > 1e-12
        else 0.5
    )
    crossing_level = outside_level + fraction * (inside_level - outside_level)
    crossing_position = outside_center + fraction * (inside_center - outside_center)
    final_primitive_level = float(
        barrier_primitive_level(crossing_position[None, :], host_fit)[0]
    )
    diagnostics.update(
        {
            "found": True,
            "reason": "first_persistent_host_boundary_crossing",
            "axis_level": float(crossing_level),
            "position": crossing_position,
            "primitive_level": final_primitive_level,
            "outside_bracket_level": float(outside_level),
            "inside_bracket_level": float(inside_level),
        }
    )
    return diagnostics


def assign_crypt_attachments_from_barrier_crossings(
    vertices,
    faces,
    detections: list[dict[str, Any]],
    body_fit: SoftBarrierEllipsoidFit,
    *,
    branch_fits: dict[str, SoftBarrierEllipsoidFit] | None = None,
    crossing_kwargs: dict[str, Any] | None = None,
    assign_body_roots: bool = True,
    assign_branch_daughters: bool = True,
    metadata_key: str = "barrier_boundary_crossing",
) -> list[dict[str, Any]]:
    """Assign root and daughter attachments from their host boundaries.

    Top-level appendices, including accepted split parents, cross the body
    primitive.  Daughters cross their fitted branch primitive.  Existing
    circumference-derived constrictions are preserved when they remain distal
    to the new host crossing; otherwise the impossible neck interval is
    collapsed to a transition.
    """
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    branch_fits = dict(branch_fits or {})
    kwargs = dict(crossing_kwargs or {})
    out = copy.deepcopy(detections)

    def update_detection(det, host_fit, *, host_id: str):
        metadata = dict(det.get("metadata", {}))
        diagnostics = {
            "applied": False,
            "host_id": str(host_id),
            "host_primitive_type": None if host_fit is None else host_fit.primitive_type,
        }
        if host_fit is None:
            diagnostics["reason"] = "missing_host_barrier_primitive"
            metadata[metadata_key] = diagnostics
            det["metadata"] = metadata
            return

        dfield = np.asarray(det.get("d_crypt", []), dtype=float).reshape(-1)
        if dfield.size != vertices.shape[0]:
            diagnostics["reason"] = "missing_or_invalid_distance_field"
            metadata[metadata_key] = diagnostics
            det["metadata"] = metadata
            return
        patch = _coerce_patch(det.get("crypt_vertices"))
        crossing = find_barrier_boundary_crossing(
            vertices,
            faces,
            dfield,
            host_fit,
            prefer_vertices=patch,
            **kwargs,
        )
        diagnostics.update(crossing)
        diagnostics["applied"] = bool(crossing.get("found", False))
        if not crossing.get("found", False):
            metadata[metadata_key] = diagnostics
            det["metadata"] = metadata
            return

        axis_level = float(crossing["axis_level"])
        position = np.asarray(crossing["position"], dtype=float)
        previous_level = float(det.get("attachment_level", 1.0))
        previous_position = det.get("attachment_position", det.get("neck_position"))
        det["attachment_level"] = axis_level
        det["attachment_position"] = position
        det["neck_position"] = position
        tip_vertex_id = det.get(
            "bottom_vertex_id",
            det.get("boundary_distance_bottom_vertex_id"),
        )
        root_region = _tip_connected_region(
            faces,
            dfield,
            axis_level,
            tip_vertex_id,
        )
        if "candidate_crypt_vertices" not in det:
            det["candidate_crypt_vertices"] = _coerce_patch(
                det.get("crypt_vertices")
            )
        det["crypt_vertices"] = root_region
        det["attachment_region_vertices"] = root_region
        det["root_region_vertices"] = root_region

        profile = det.get("neck_profile")
        constriction_collapsed = False
        if isinstance(profile, dict):
            profile = dict(profile)
            profile["attachment_level"] = axis_level
            profile["attachment_source"] = "host_primitive_boundary_crossing"
            constriction_level = profile.get("constriction_level")
            if profile.get("kind") == "constriction" and constriction_level is not None:
                if axis_level <= float(constriction_level) + 1e-6:
                    profile["kind"] = "transition"
                    profile["reason"] = "host_boundary_precedes_constriction"
                    constriction_collapsed = True
                    for key in (
                        "constriction_position",
                        "distal_neck_boundary_position",
                        "neck_region_vertices",
                    ):
                        det.pop(key, None)
                else:
                    distal_level = float(
                        profile.get("distal_boundary_level", constriction_level)
                    )
                    in_neck = (
                        np.isfinite(dfield[root_region])
                        & (dfield[root_region] >= distal_level)
                    )
                    det["neck_region_vertices"] = root_region[in_neck]
                    det["neck_position"] = np.asarray(
                        det.get("constriction_position", position),
                        dtype=float,
                    )
            det["neck_profile"] = profile

        old_length = det.get("L_crypt")
        if old_length is not None and np.isfinite(float(old_length)):
            det["L_crypt"] = float(old_length) * axis_level
        diagnostics.update(
            {
                "previous_attachment_level": previous_level,
                "previous_attachment_position": previous_position,
                "n_root_region_vertices": int(root_region.size),
                "constriction_collapsed": constriction_collapsed,
            }
        )
        metadata[metadata_key] = diagnostics
        det["metadata"] = metadata

    for detection in out:
        if assign_body_roots:
            update_detection(detection, body_fit, host_id="body")
        daughters = detection.get("daughters") or []
        if not daughters or not assign_branch_daughters:
            continue
        branch_id = f"crypt_{detection.get('crypt_id')}_branch"
        branch_fit = branch_fits.get(branch_id)
        for daughter in daughters:
            update_detection(daughter, branch_fit, host_id=branch_id)
    return out


__all__ = [
    "assign_crypt_attachments_from_barrier_crossings",
    "find_barrier_boundary_crossing",
]

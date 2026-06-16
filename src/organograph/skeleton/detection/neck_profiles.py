"""Circumference-profile and neckline geometry helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from organograph.skeleton.detection.common import (
    _centroid_from_vertex_keys,
    _coerce_patch,
    _first_present,
    _point_from_keys,
    _point_from_vertex,
)
from organograph.skeleton.detection.mesh_regions import _boundary_vertices_from_patch
from organograph.skeleton.geometry import as_points, centroid

def _crossing_level(
    levels: np.ndarray,
    values: np.ndarray,
    start_index: int,
    target: float,
    direction: int,
) -> float | None:
    """Return the first interpolated target crossing away from one index."""
    i = int(start_index)
    while 0 <= i + direction < levels.size:
        j = i + direction
        y0 = float(values[i] - target)
        y1 = float(values[j] - target)
        if y0 == 0.0:
            return float(levels[i])
        if y0 * y1 <= 0.0 and values[j] >= target:
            denom = float(values[j] - values[i])
            if abs(denom) <= 1e-12:
                return float(levels[j])
            fraction = float((target - values[i]) / denom)
            return float(levels[i] + fraction * (levels[j] - levels[i]))
        i = j
    return None

def analyze_neck_circumference_profile(
    levels,
    circumference,
    *,
    relation: str = "body_crypt",
    neck_search_window: tuple[float, float] = (0.8, 2.0),
    window_length: int = 9,
    polyorder: int = 3,
    min_prominence: float = 0.05,
    min_neck_length: float = 0.05,
    body_branch_cmax_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Classify a normalized neckline and bound genuine constrictions.

    The upstream segmentation has already normalized its selected neckline to
    d=1. This function never searches for or substitutes another minimum. It
    only classifies the fixed d=1 point as a constriction or transition. A
    minimum within one smoothing half-window is accepted to tolerate sampling
    offsets, but the constriction position remains d=1. For constrictions, the
    first half-depth circumference crossing on either side bounds the neck
    component. Necks shorter than ``min_neck_length`` on the attachment side
    collapse to one transition node.
    """
    levels = np.asarray(levels, dtype=float).reshape(-1)
    values = np.asarray(circumference, dtype=float).reshape(-1)
    result = {
        "kind": "transition",
        "relation": str(relation),
        "constriction_level": None,
        "attachment_level": 1.0,
        "distal_boundary_level": 1.0,
        "c_min": None,
        "c_max": None,
        "c_half": None,
        "prominence": 0.0,
        "second_derivative_peak_level": None,
        "second_derivative_peak_score": 0.0,
    }
    finite = np.isfinite(levels) & np.isfinite(values)
    if np.count_nonzero(finite) < 7:
        result["reason"] = "insufficient_circumference_samples"
        return result
    x = levels[finite]
    y = values[finite]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    wl = min(int(window_length) | 1, x.size if x.size % 2 else x.size - 1)
    wl = max(wl, 5)
    po = min(int(polyorder), wl - 2)
    smooth = savgol_filter(y, window_length=wl, polyorder=po, mode="interp")

    neck_index = int(np.argmin(np.abs(x - 1.0)))
    if neck_index == 0 or neck_index == x.size - 1:
        result["reason"] = "normalized_neck_outside_profile"
        return result

    spacing = float(np.median(np.diff(x)))
    classification_radius = max((wl // 2) * spacing, 2.0 * spacing, 1e-6)
    second_derivative = savgol_filter(
        y,
        window_length=wl,
        polyorder=po,
        deriv=2,
        delta=max(spacing, 1e-12),
        mode="interp",
    )
    lo, hi = map(float, neck_search_window)
    second_derivative_search = np.where((x >= lo) & (x <= hi))[0]
    second_derivative_near = np.where(
        np.abs(x - 1.0) <= classification_radius + 1e-12
    )[0]
    if second_derivative_search.size and second_derivative_near.size:
        peak_index = int(
            second_derivative_near[
                np.argmax(second_derivative[second_derivative_near])
            ]
        )
        search_peak = float(
            np.max(np.maximum(second_derivative[second_derivative_search], 0.0))
        )
        local_peak = max(float(second_derivative[peak_index]), 0.0)
        location_score = local_peak / max(search_peak, 1e-12)
        background = float(
            np.median(np.abs(second_derivative[second_derivative_search]))
        )
        peak_contrast = local_peak / max(background, 1e-12)
        peak_score = location_score * peak_contrast / (1.0 + peak_contrast)
        result.update(
            {
                "second_derivative_peak_level": float(x[peak_index]),
                "second_derivative_peak_score": float(np.clip(peak_score, 0.0, 1.0)),
                "second_derivative_peak_value": local_peak,
                "second_derivative_peak_contrast": peak_contrast,
            }
        )
    c_min = float(np.interp(1.0, x, smooth))
    nearby = np.where(np.abs(x - 1.0) <= classification_radius + 1e-12)[0]
    nearby = nearby[(nearby > 0) & (nearby < x.size - 1)]
    nearby_minima = nearby[
        (smooth[nearby - 1] >= smooth[nearby])
        & (smooth[nearby] <= smooth[nearby + 1])
    ]
    if nearby_minima.size == 0:
        result["reason"] = "second_derivative_transition"
        return result

    candidates = np.where((x >= lo) & (x <= hi))[0]
    left_candidates = candidates[x[candidates] <= 1.0]
    right_candidates = candidates[x[candidates] >= 1.0]
    if left_candidates.size == 0 or right_candidates.size == 0:
        result["reason"] = "normalized_neck_outside_search_window"
        return result

    left_reference = float(np.max(smooth[left_candidates]))
    right_reference = float(np.max(smooth[right_candidates]))
    reference = max(left_reference, right_reference, 1e-12)
    prominence = float((reference - c_min) / reference)
    if prominence < float(min_prominence):
        result.update(
            {
                "reason": "minimum_below_prominence",
                "prominence": prominence,
            }
        )
        return result

    crypt_side = smooth[x <= 1.0]
    if crypt_side.size == 0:
        result["reason"] = "missing_crypt_side"
        return result
    if str(relation) == "body_branch":
        multiplier = max(float(body_branch_cmax_multiplier), 1.0)
        c_max = multiplier * c_min
        cmax_source = "scaled_constriction_circumference"
        result["body_branch_cmax_multiplier"] = multiplier
    else:
        c_max = float(np.max(crypt_side))
        cmax_source = "maximum_within_crypt"
    c_half = 0.5 * (c_max + c_min)
    distal = _crossing_level(x, smooth, neck_index, c_half, -1)
    proximal = _crossing_level(x, smooth, neck_index, c_half, 1)
    distal_censored = distal is None
    proximal_censored = proximal is None
    if distal is None:
        distal = float(x[0])
    if proximal is None:
        proximal = float(x[-1])

    neck_length = max(float(proximal) - 1.0, 0.0)
    if neck_length < max(float(min_neck_length), 0.0):
        result.update(
            {
                "reason": "constricted_neck_below_min_length",
                "candidate_kind": "constriction",
                "candidate_constriction_level": 1.0,
                "candidate_attachment_level": float(proximal),
                "candidate_distal_boundary_level": float(distal),
                "candidate_neck_length": neck_length,
                "min_neck_length": max(float(min_neck_length), 0.0),
                "c_min": c_min,
                "c_max": c_max,
                "c_half": c_half,
                "c_max_source": cmax_source,
                "prominence": prominence,
                "classification_minimum_level": float(
                    x[nearby_minima[np.argmin(smooth[nearby_minima])]]
                ),
                "classification_radius": classification_radius,
            }
        )
        return result

    result.update(
        {
            "kind": "constriction",
            "reason": (
                "prominent_local_minimum_with_censored_boundary"
                if distal_censored or proximal_censored
                else "prominent_local_minimum"
            ),
            "constriction_level": 1.0,
            "attachment_level": float(proximal),
            "distal_boundary_level": float(distal),
            "c_min": c_min,
            "c_max": c_max,
            "c_half": c_half,
            "c_max_source": cmax_source,
            "prominence": prominence,
            "classification_minimum_level": float(
                x[nearby_minima[np.argmin(smooth[nearby_minima])]]
            ),
            "classification_radius": classification_radius,
            "neck_length": neck_length,
            "min_neck_length": max(float(min_neck_length), 0.0),
            "distal_boundary_censored": distal_censored,
            "attachment_boundary_censored": proximal_censored,
        }
    )
    return result

def _contour_center_from_distance_field(
    vertices,
    faces,
    dfield,
    *,
    level: float = 1.0,
    prefer_vertices=None,
    min_points: int = 3,
) -> np.ndarray | None:
    """Centroid of an isocontour component from triangle-edge intersections.

    The returned point is not forced to lie on the mesh.  For a closed neckline
    ring this produces the geometric center of the ring, which is the desired
    skeleton neck-node position.  The contour is extracted from the full mesh;
    `prefer_vertices` only helps choose a component if several are present.
    """
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    dfield = np.asarray(dfield, dtype=float).reshape(-1)
    if dfield.size != vertices.shape[0] or faces.size == 0:
        return None

    face_mask = np.isfinite(dfield[faces]).all(axis=1)
    candidate_faces = faces[face_mask]
    if candidate_faces.size == 0:
        return None

    level = float(level)
    eps = 1e-12
    segments = []

    def edge_intersection(a: int, b: int) -> np.ndarray | None:
        da = float(dfield[a])
        db = float(dfield[b])
        if abs(da - level) <= eps and abs(db - level) <= eps:
            return None
        if abs(da - level) <= eps:
            return vertices[a]
        if abs(db - level) <= eps:
            return vertices[b]
        if (da - level) * (db - level) > 0.0:
            return None
        t = (level - da) / (db - da)
        if -eps <= t <= 1.0 + eps:
            t = float(np.clip(t, 0.0, 1.0))
            return vertices[a] + t * (vertices[b] - vertices[a])
        return None

    for tri in candidate_faces:
        edge_points = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a = int(a)
            b = int(b)
            if abs(float(dfield[a]) - level) <= eps and abs(float(dfield[b]) - level) <= eps:
                edge_points.extend([vertices[a], vertices[b]])
                continue
            p = edge_intersection(a, b)
            if p is not None:
                edge_points.append(p)

        unique = []
        for p in edge_points:
            if not any(np.linalg.norm(p - q) <= 1e-10 for q in unique):
                unique.append(p)
        if len(unique) < 2:
            continue
        if len(unique) == 2:
            segments.append((unique[0], unique[1]))
        else:
            # Degenerate level-through-vertex cases can yield >2 points.  Connect
            # them around their centroid to keep the component closed enough for
            # center estimation.
            c = centroid(np.asarray(unique, dtype=float))
            normal = np.cross(unique[1] - unique[0], unique[2] - unique[0])
            if np.linalg.norm(normal) <= 1e-12:
                order = range(len(unique))
            else:
                axis0 = unique[0] - c
                n0 = np.linalg.norm(axis0)
                if n0 <= 1e-12:
                    order = range(len(unique))
                else:
                    axis0 = axis0 / n0
                    axis1 = np.cross(normal, axis0)
                    axis1 = axis1 / max(np.linalg.norm(axis1), 1e-12)
                    angles = [np.arctan2(np.dot(p - c, axis1), np.dot(p - c, axis0)) for p in unique]
                    order = np.argsort(angles)
            ordered = [unique[i] for i in order]
            for p0, p1 in zip(ordered, ordered[1:] + ordered[:1]):
                segments.append((p0, p1))

    if not segments:
        return None

    points = []
    adjacency = []
    key_to_index = {}

    def point_key(p: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.round(p / 1e-8).astype(np.int64).tolist())

    def add_point(p: np.ndarray) -> int:
        key = point_key(p)
        if key in key_to_index:
            return key_to_index[key]
        idx = len(points)
        key_to_index[key] = idx
        points.append(p)
        adjacency.append(set())
        return idx

    for p0, p1 in segments:
        i0 = add_point(np.asarray(p0, dtype=float))
        i1 = add_point(np.asarray(p1, dtype=float))
        if i0 == i1:
            continue
        adjacency[i0].add(i1)
        adjacency[i1].add(i0)

    if len(points) < int(min_points):
        return None

    visited = np.zeros(len(points), dtype=bool)
    components = []
    for start in range(len(points)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in adjacency[i]:
                if not visited[j]:
                    visited[j] = True
                    stack.append(j)
        if len(comp) >= int(min_points):
            components.append(comp)

    if not components:
        return None

    prefer = _coerce_patch(prefer_vertices)
    if prefer.size:
        prefer_points = vertices[prefer]

        def component_score(comp):
            pts = np.asarray([points[i] for i in comp], dtype=float)
            center = centroid(pts)
            min_dist = float(np.min(np.linalg.norm(prefer_points - center, axis=1)))
            return (-min_dist, len(comp))

        best = max(components, key=component_score)
    else:
        best = max(components, key=len)

    pts = np.asarray([points[i] for i in best], dtype=float)
    if pts.shape[0] < int(min_points):
        return None
    return centroid(pts)

def _neck_from_distance_field(
    vertices,
    faces,
    detection: dict[str, Any],
    *,
    tolerance: float = 0.05,
) -> np.ndarray | None:
    dfield = _first_present(detection, ("d_crypt", "distance_field", "dnorm", "dnorm_vertices"))
    if dfield is None:
        return None
    dfield = np.asarray(dfield, dtype=float).reshape(-1)
    if dfield.size != as_points(vertices).shape[0]:
        return None

    patch = _coerce_patch(
        _first_present(
            detection,
            ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
        )
    )

    ring_center = _contour_center_from_distance_field(
        vertices,
        faces,
        dfield,
        level=float(detection.get("neck_level", 1.0)),
        prefer_vertices=patch,
    )
    if ring_center is not None:
        return ring_center

    if patch.size:
        valid = patch[np.isfinite(dfield[patch])]
    else:
        valid = np.where(np.isfinite(dfield))[0]
    if valid.size == 0:
        return None

    delta = np.abs(dfield[valid] - 1.0)
    near = valid[delta <= float(tolerance)]
    if near.size == 0:
        best = np.nanmin(delta)
        near = valid[delta <= best + 1e-12]
    return centroid(as_points(vertices)[near])

def _add_neck_profile_geometry(
    vertices,
    faces,
    detection: dict[str, Any],
    levels,
    circumference,
    *,
    relation: str,
    window_length: int = 9,
    polyorder: int = 3,
    min_prominence: float = 0.05,
    min_neck_length: float = 0.05,
    body_branch_cmax_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Attach classified neck/attachment geometry to one detection."""
    profile = analyze_neck_circumference_profile(
        levels,
        circumference,
        relation=relation,
        window_length=window_length,
        polyorder=polyorder,
        min_prominence=min_prominence,
        min_neck_length=min_neck_length,
        body_branch_cmax_multiplier=body_branch_cmax_multiplier,
    )
    dfield = np.asarray(detection.get("d_crypt"), dtype=float).reshape(-1)
    patch = _coerce_patch(detection.get("crypt_vertices"))
    attachment_level = float(profile["attachment_level"])
    attachment = _contour_center_from_distance_field(
        vertices,
        faces,
        dfield,
        level=attachment_level,
        prefer_vertices=patch,
    )
    if attachment is None:
        attachment = _contour_center_from_distance_field(
            vertices,
            faces,
            dfield,
            level=1.0,
            prefer_vertices=patch,
        )
        attachment_level = 1.0
        profile["attachment_level"] = 1.0

    detection["neck_profile"] = profile
    detection["circumference_levels"] = np.asarray(levels, dtype=float)
    detection["circumference"] = np.asarray(circumference, dtype=float)
    detection["attachment_level"] = attachment_level
    if attachment is not None:
        detection["attachment_position"] = attachment

    if profile["kind"] == "constriction":
        constriction_level = float(profile["constriction_level"])
        constriction = _contour_center_from_distance_field(
            vertices,
            faces,
            dfield,
            level=constriction_level,
            prefer_vertices=patch,
        )
        distal = _contour_center_from_distance_field(
            vertices,
            faces,
            dfield,
            level=float(profile["distal_boundary_level"]),
            prefer_vertices=patch,
        )
        if constriction is not None:
            detection["constriction_position"] = constriction
            detection["neck_position"] = constriction
        if distal is not None:
            detection["distal_neck_boundary_position"] = distal
        detection["neck_region_vertices"] = np.where(
            np.isfinite(dfield)
            & (dfield >= float(profile["distal_boundary_level"]))
            & (dfield <= attachment_level)
        )[0].astype(np.int64)
    elif attachment is not None:
        detection["neck_position"] = attachment

    detection["attachment_region_vertices"] = np.where(
        np.isfinite(dfield) & (dfield <= attachment_level)
    )[0].astype(np.int64)
    return detection

def _neck_position(vertices, faces, detection: dict[str, Any]) -> np.ndarray:
    explicit = _point_from_keys(
        vertices,
        detection,
        ("neck_center", "neck_position", "neck", "neckline_center", "p_neck"),
    )
    if explicit is not None:
        return explicit

    from_vertices = _centroid_from_vertex_keys(
        vertices,
        detection,
        ("neck_vertices", "neckline_vertices", "boundary_vertices"),
    )
    if from_vertices is not None:
        return from_vertices

    from_distance = _neck_from_distance_field(vertices, faces, detection)
    if from_distance is not None:
        return from_distance

    patch = _coerce_patch(
        _first_present(
            detection,
            ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
        )
    )
    if patch.size:
        boundary = _boundary_vertices_from_patch(faces, patch)
        return centroid(as_points(vertices)[boundary])

    raise ValueError(
        "Crypt detection is missing neck coordinates, neck vertices, "
        "a normalized distance field, or patch vertices."
    )


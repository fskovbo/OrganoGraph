"""Shared geometry for current skeleton primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from organograph.skeleton.geometry import as_points


def component_points(vertices, component) -> np.ndarray:
    """Return component points from either coordinates or vertex indices."""
    if component is None:
        return as_points(vertices)
    array = np.asarray(component)
    if array.ndim == 2 and array.shape[1] == 3:
        return as_points(array)
    indices = np.asarray(list(component) if isinstance(component, set) else component, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("Component must be an (N, 3) array or 1D vertex indices")
    return as_points(vertices)[indices]


def polyline_lengths(points) -> tuple[np.ndarray, np.ndarray, float]:
    """Return segment lengths, cumulative arclengths, and total length."""
    points = as_points(points)
    if points.shape[0] < 2:
        return np.empty(0), np.zeros(points.shape[0]), 0.0
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    return lengths, cumulative, float(cumulative[-1])


def point_segment_projection(points, start, end):
    """Project points onto one segment and return points and parameters."""
    points = as_points(points)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-12:
        parameters = np.zeros(points.shape[0])
        return np.repeat(start[None, :], points.shape[0], axis=0), parameters
    parameters = np.clip(((points - start) @ segment) / denominator, 0.0, 1.0)
    return start + parameters[:, None] * segment, parameters


def project_points_to_polyline(points, centerline):
    """Project points to a polyline and return distances and arclength coordinates."""
    points = as_points(points)
    line = as_points(centerline)
    lengths, cumulative, total = polyline_lengths(line)
    if line.shape[0] == 0:
        raise ValueError("Centerline cannot be empty")
    if line.shape[0] == 1 or total <= 1e-12:
        closest = np.repeat(line[:1], points.shape[0], axis=0)
        return {
            "closest_points": closest,
            "distances": np.linalg.norm(points - closest, axis=1),
            "s": np.zeros(points.shape[0]),
            "arclength": np.zeros(points.shape[0]),
            "segment_index": np.zeros(points.shape[0], dtype=np.int64),
        }
    best_distance2 = np.full(points.shape[0], np.inf)
    best_closest = np.zeros_like(points)
    best_arclength = np.zeros(points.shape[0])
    best_segment = np.zeros(points.shape[0], dtype=np.int64)
    for index, length in enumerate(lengths):
        closest, parameter = point_segment_projection(points, line[index], line[index + 1])
        distance2 = np.sum((points - closest) ** 2, axis=1)
        update = distance2 < best_distance2
        best_distance2[update] = distance2[update]
        best_closest[update] = closest[update]
        best_arclength[update] = cumulative[index] + parameter[update] * length
        best_segment[update] = index
    return {
        "closest_points": best_closest,
        "distances": np.sqrt(best_distance2),
        "s": best_arclength / total,
        "arclength": best_arclength,
        "segment_index": best_segment,
    }


def point_at_polyline_arclength(points, fraction: float) -> np.ndarray:
    """Return the point at a normalized polyline arclength."""
    line = as_points(points)
    lengths, cumulative, total = polyline_lengths(line)
    if line.shape[0] == 0:
        raise ValueError("Polyline cannot be empty")
    if total <= 1e-12 or line.shape[0] == 1:
        return line[0].copy()
    target = float(np.clip(fraction, 0.0, 1.0)) * total
    segment = int(np.searchsorted(cumulative, target, side="right") - 1)
    segment = max(0, min(segment, lengths.size - 1))
    local = (target - cumulative[segment]) / max(lengths[segment], 1e-12)
    return line[segment] + local * (line[segment + 1] - line[segment])


def capped_tube_radius(
    s,
    r_neck: float,
    r_body: float,
    r_tip: float,
    *,
    body_s: float = 0.5,
    center_s: float | None = None,
    taper_start: float = 0.85,
    constriction_s: float | None = None,
    r_constriction: float | None = None,
) -> np.ndarray:
    """Evaluate the shape-preserving squared-radius crypt profile."""
    values = np.asarray(s, dtype=float)
    center = float(body_s if center_s is None else center_s)
    taper = float(taper_start)
    if not 0.0 < center < taper < 1.0:
        raise ValueError("Radius positions must satisfy 0 < center_s < s_taper < 1")
    clipped = np.clip(values, 0.0, 1.0)
    if constriction_s is not None and r_constriction is not None:
        constriction = float(constriction_s)
        if not 0.0 < constriction < center:
            raise ValueError("s_constriction must satisfy 0 < s_constriction < s_center")
        control_s = np.array([0.0, constriction, center, taper, 1.0])
        control_radius = np.array([r_neck, r_constriction, r_body, r_tip, 0.0])
    else:
        control_s = np.array([0.0, center, taper, 1.0])
        control_radius = np.array([r_neck, r_body, r_tip, 0.0])
    squared = PchipInterpolator(
        control_s, np.maximum(control_radius, 0.0) ** 2, extrapolate=False
    )(clipped)
    output = np.sqrt(np.maximum(squared, 0.0))
    output[values >= 1.0] = 0.0
    return output


def bend_angles_for_polyline(points) -> list[float]:
    """Return angles between consecutive polyline segments."""
    points = as_points(points)
    angles = []
    for index in range(1, points.shape[0] - 1):
        first = points[index] - points[index - 1]
        second = points[index + 1] - points[index]
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        angles.append(
            float("nan")
            if denominator <= 1e-12
            else float(np.arccos(np.clip(np.dot(first, second) / denominator, -1.0, 1.0)))
        )
    return angles


def sanitize_id(value: Any) -> str:
    """Convert an arbitrary component key into a stable id fragment."""
    return str(value).replace(" ", "_").replace("/", "_").replace(":", "_")

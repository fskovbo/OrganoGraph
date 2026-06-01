"""Geometry helpers for fitting and visualizing skeleton primitives."""

from __future__ import annotations

from typing import Any

import numpy as np

from organograph.skeleton.geometry import as_points


def component_points(vertices, component) -> np.ndarray:
    """Return component points from either point coordinates or vertex indices."""
    if component is None:
        return as_points(vertices)
    arr = np.asarray(component)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return as_points(arr)
    idx = np.asarray(list(component) if isinstance(component, set) else component, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("Component must be an (N, 3) point array or 1D vertex indices")
    return as_points(vertices)[idx]


def polyline_lengths(points) -> tuple[np.ndarray, np.ndarray, float]:
    """Return segment lengths, cumulative vertex arclengths, and total length."""
    pts = as_points(points)
    if pts.shape[0] < 2:
        return np.empty(0, dtype=float), np.zeros(pts.shape[0], dtype=float), 0.0
    seg = pts[1:] - pts[:-1]
    lengths = np.linalg.norm(seg, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    return lengths, cumulative, float(cumulative[-1])


def point_segment_projection(points, a, b):
    """Project points onto one segment and return closest points and parameters."""
    pts = as_points(points)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        t = np.zeros(pts.shape[0], dtype=float)
        closest = np.repeat(a[None, :], pts.shape[0], axis=0)
        return closest, t
    t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    return closest, t


def project_points_to_polyline(points, centerline):
    """Project points to a piecewise-linear centerline.

    Returns a dictionary with closest points, distances, normalized arclength
    coordinates, absolute arclength coordinates, and segment indices.
    """
    pts = as_points(points)
    line = as_points(centerline)
    seg_lengths, cumulative, total = polyline_lengths(line)
    if line.shape[0] == 0:
        raise ValueError("Centerline cannot be empty")
    if line.shape[0] == 1 or total <= 1e-12:
        closest = np.repeat(line[:1], pts.shape[0], axis=0)
        distances = np.linalg.norm(pts - closest, axis=1)
        return {
            "closest_points": closest,
            "distances": distances,
            "s": np.zeros(pts.shape[0], dtype=float),
            "arclength": np.zeros(pts.shape[0], dtype=float),
            "segment_index": np.zeros(pts.shape[0], dtype=np.int64),
        }

    best_dist2 = np.full(pts.shape[0], np.inf, dtype=float)
    best_closest = np.zeros_like(pts)
    best_arclength = np.zeros(pts.shape[0], dtype=float)
    best_segment = np.zeros(pts.shape[0], dtype=np.int64)
    for i, length in enumerate(seg_lengths):
        closest, t = point_segment_projection(pts, line[i], line[i + 1])
        dist2 = np.sum((pts - closest) ** 2, axis=1)
        update = dist2 < best_dist2
        best_dist2[update] = dist2[update]
        best_closest[update] = closest[update]
        best_arclength[update] = cumulative[i] + t[update] * length
        best_segment[update] = i

    return {
        "closest_points": best_closest,
        "distances": np.sqrt(best_dist2),
        "s": best_arclength / total,
        "arclength": best_arclength,
        "segment_index": best_segment,
    }


def quadratic_radius(s, r_neck: float, r_body: float, r_tip: float) -> np.ndarray:
    """Quadratic radius profile through s=0, 0.5, and 1."""
    s = np.asarray(s, dtype=float)
    rn = float(r_neck)
    rb = float(r_body)
    rt = float(r_tip)
    d1 = rt - rn
    d05 = rb - rn
    b = 4.0 * d05 - d1
    a = d1 - b
    return a * s**2 + b * s + rn


def bend_angles_for_polyline(points) -> list[float]:
    """Return angles between consecutive straight segments of a polyline."""
    pts = as_points(points)
    if pts.shape[0] < 3:
        return []
    angles = []
    for i in range(1, pts.shape[0] - 1):
        v0 = pts[i] - pts[i - 1]
        v1 = pts[i + 1] - pts[i]
        n0 = float(np.linalg.norm(v0))
        n1 = float(np.linalg.norm(v1))
        if n0 <= 1e-12 or n1 <= 1e-12:
            angles.append(float("nan"))
            continue
        cosang = float(np.dot(v0, v1) / (n0 * n1))
        angles.append(float(np.arccos(np.clip(cosang, -1.0, 1.0))))
    return angles


def sanitize_id(value: Any) -> str:
    """Convert an arbitrary component key into a stable id fragment."""
    return str(value).replace(" ", "_").replace("/", "_").replace(":", "_")

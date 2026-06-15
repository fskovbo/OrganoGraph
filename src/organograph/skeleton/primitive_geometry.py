"""Geometry helpers for fitting and visualizing skeleton primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

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


def sample_quadratic_bezier(
    start,
    control,
    end,
    *,
    n_samples: int = 64,
) -> np.ndarray:
    """Sample one quadratic Bézier centerline segment."""
    start = np.asarray(start, dtype=float)
    control = np.asarray(control, dtype=float)
    end = np.asarray(end, dtype=float)
    if start.shape != (3,) or control.shape != (3,) or end.shape != (3,):
        raise ValueError("Bézier start, control, and end must be 3-vectors")
    u = np.linspace(0.0, 1.0, max(2, int(n_samples)))
    return (
        (1.0 - u)[:, None] ** 2 * start
        + 2.0 * (1.0 - u)[:, None] * u[:, None] * control
        + u[:, None] ** 2 * end
    )


def fit_quadratic_bezier_control(
    start,
    end,
    observations,
    parameters,
    *,
    weights=None,
    regularization: float = 0.05,
) -> np.ndarray:
    """Fit the single control point of a quadratic Bézier segment."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    observations = as_points(observations)
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    if observations.shape[0] != parameters.size:
        raise ValueError("observations and parameters must have matching lengths")
    if observations.shape[0] == 0:
        return 0.5 * (start + end)

    u = np.clip(parameters, 1e-6, 1.0 - 1e-6)
    coefficient = 2.0 * (1.0 - u) * u
    target = (
        observations
        - (1.0 - u)[:, None] ** 2 * start
        - u[:, None] ** 2 * end
    )
    if weights is None:
        weights = np.ones_like(u)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != u.size:
        raise ValueError("weights must match the observation count")
    weights = np.maximum(weights, 0.0)

    prior = 0.5 * (start + end)
    scale = max(float(np.sum(weights * coefficient**2)), 1e-12)
    penalty = max(float(regularization), 0.0) * scale
    numerator = np.sum(
        (weights * coefficient)[:, None] * target,
        axis=0,
    ) + penalty * prior
    denominator = scale + penalty
    return numerator / max(denominator, 1e-12)


def estimate_smooth_crypt_centerline(
    vertices,
    component_vertices,
    distance_field,
    neck_position,
    tip_position,
    *,
    neck_level: float = 1.0,
    n_bands: int = 7,
    n_samples: int = 64,
    min_band_points: int = 3,
    control_regularization: float = 0.05,
) -> dict[str, Any]:
    """Estimate a smooth internal crypt centerline from geodesic bands.

    The normalized crypt-axis field is expected to be approximately zero at
    the geodesic bottom and ``neck_level`` at the neckline. Surface vertices
    are grouped into proximal-to-distal bands; each band centroid is an
    internal ring-center estimate. Those centers collectively fit the one
    control point of a quadratic Bézier segment with fixed neckline and tip.
    """
    vertices = as_points(vertices)
    indices = np.asarray(
        list(component_vertices) if isinstance(component_vertices, set) else component_vertices,
        dtype=np.int64,
    ).reshape(-1)
    field = np.asarray(distance_field, dtype=float).reshape(-1)
    neck = np.asarray(neck_position, dtype=float)
    tip = np.asarray(tip_position, dtype=float)
    if field.size != vertices.shape[0]:
        raise ValueError("distance_field must contain one value per mesh vertex")
    if indices.size == 0:
        raise ValueError("component_vertices cannot be empty")
    level = float(neck_level)
    if not np.isfinite(level) or level <= 0.0:
        raise ValueError("neck_level must be positive")

    valid = indices[(indices >= 0) & (indices < vertices.shape[0])]
    valid = valid[np.isfinite(field[valid])]
    if valid.size < 3:
        raise ValueError("Too few component vertices have finite axis distances")

    axis_s = 1.0 - np.clip(field[valid] / level, 0.0, 1.0)
    n_bands = max(3, int(n_bands))
    targets = np.linspace(0.0, 1.0, n_bands)
    half_width = 0.5 / float(n_bands - 1)
    minimum = max(1, int(min_band_points))
    ring_centers = []
    ring_parameters = []
    band_sizes = []

    for target in targets[1:-1]:
        selected = np.where(np.abs(axis_s - target) <= half_width)[0]
        if selected.size < minimum:
            order = np.argsort(np.abs(axis_s - target))
            selected = order[: min(max(minimum, valid.size // n_bands), valid.size)]
        if selected.size == 0:
            continue
        ring_centers.append(np.mean(vertices[valid[selected]], axis=0))
        ring_parameters.append(float(target))
        band_sizes.append(int(selected.size))

    ring_centers = np.asarray(ring_centers, dtype=float)
    ring_parameters = np.asarray(ring_parameters, dtype=float)
    control = fit_quadratic_bezier_control(
        neck,
        tip,
        ring_centers,
        ring_parameters,
        weights=np.asarray(band_sizes, dtype=float),
        regularization=control_regularization,
    )
    centerline = sample_quadratic_bezier(
        neck,
        control,
        tip,
        n_samples=n_samples,
    )
    return {
        "centerline_points": centerline,
        "control_points": np.vstack([neck, control, tip]),
        "control_parameters": np.array([0.0, 0.5, 1.0]),
        "bezier_control_point": control,
        "ring_centers": ring_centers,
        "ring_parameters": ring_parameters,
        "band_sizes": band_sizes,
        "method": "geodesic_band_centroids_quadratic_bezier",
    }


def point_at_polyline_arclength(points, fraction: float) -> np.ndarray:
    """Return the point at a normalized arc-length fraction of a polyline."""
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


def quadratic_radius(
    s,
    r_neck: float,
    r_body: float,
    r_tip: float,
    *,
    body_s: float = 0.5,
    tip_s: float = 1.0,
) -> np.ndarray:
    """Quadratic radius profile through s=0, ``body_s``, and ``tip_s``."""
    s = np.asarray(s, dtype=float)
    rn = float(r_neck)
    rb = float(r_body)
    rt = float(r_tip)
    sb = float(body_s)
    st = float(tip_s)
    if not (0.0 < sb < st):
        raise ValueError("Radius control positions must satisfy 0 < body_s < tip_s")

    # Lagrange interpolation through (0, rn), (sb, rb), and (st, rt).
    l0 = ((s - sb) * (s - st)) / (sb * st)
    l1 = (s * (s - st)) / (sb * (sb - st))
    lt = (s * (s - sb)) / (st * (st - sb))
    return rn * l0 + rb * l1 + rt * lt


def capped_tube_radius(
    s,
    r_neck: float,
    r_body: float,
    r_tip: float,
    *,
    body_s: float = 0.5,
    taper_start: float = 0.85,
) -> np.ndarray:
    """Shape-preserving smooth radius profile that closes at the crypt tip.

    A cubic Hermite interpolant is applied to squared radius through
    ``r_neck`` at s=0, ``r_body`` at ``body_s``, ``r_tip`` at
    ``taper_start``, and zero at s=1. Interpolating squared radius preserves
    non-negativity and gives a rounded square-root closure at the tip without
    introducing a separately parameterized cap interval.
    """
    values = np.asarray(s, dtype=float)
    body = float(body_s)
    start = float(taper_start)
    if not (0.0 < body < start < 1.0):
        raise ValueError("Radius positions must satisfy 0 < body_s < taper_start < 1")

    clipped = np.clip(values, 0.0, 1.0)
    control_s = np.array([0.0, body, start, 1.0], dtype=float)
    control_radius = np.maximum(
        np.array([r_neck, r_body, r_tip, 0.0], dtype=float),
        0.0,
    )
    squared_radius = PchipInterpolator(
        control_s,
        control_radius**2,
        extrapolate=False,
    )(clipped)
    out = np.sqrt(np.maximum(squared_radius, 0.0))
    out[values >= 1.0] = 0.0
    return out


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

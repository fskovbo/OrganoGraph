"""Geometry helpers for deterministic visual blend tubes."""

from __future__ import annotations

import numpy as np


def unit_vector(vector, fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        out = np.asarray(fallback, dtype=float)
        return out / max(float(np.linalg.norm(out)), 1e-12)
    return arr / norm


def sample_cubic_hermite(p0, p1, t0, t1, *, n_samples: int = 32) -> np.ndarray:
    """Sample a cubic Hermite curve from endpoint positions and tangents."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    t1 = np.asarray(t1, dtype=float)
    s = np.linspace(0.0, 1.0, max(2, int(n_samples)))[:, None]
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    return h00 * p0[None, :] + h10 * t0[None, :] + h01 * p1[None, :] + h11 * t1[None, :]


def sample_quadratic_through_midpoint(p0, pmid, p1, *, n_samples: int = 32) -> np.ndarray:
    """Sample a quadratic Bezier that passes through ``pmid`` at t=0.5."""
    p0 = np.asarray(p0, dtype=float)
    pmid = np.asarray(pmid, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    control = 2.0 * pmid - 0.5 * (p0 + p1)
    t = np.linspace(0.0, 1.0, max(2, int(n_samples)))[:, None]
    return (1.0 - t) ** 2 * p0[None, :] + 2.0 * (1.0 - t) * t * control[None, :] + t**2 * p1[None, :]


def smoothstep(values) -> np.ndarray:
    s = np.asarray(values, dtype=float)
    return s * s * (3.0 - 2.0 * s)


def blend_tube_radius(s, r_host, r_mid, r_crypt, *, s_mid: float = 0.5) -> np.ndarray:
    """Smooth host-to-crypt radius curve with monotone hostward widening.

    The renderer samples blend centerlines from host to crypt.  Biologically,
    the intended interpretation is the reverse: attachment regions should grow
    in radius as they move away from the crypt and into the body/branch.  This
    helper therefore clamps the middle control between the endpoint radii.
    """
    s = np.asarray(s, dtype=float)
    s_mid = float(np.clip(s_mid, 1e-6, 1.0 - 1e-6))
    r_host = float(max(r_host, 1e-8))
    r_crypt = float(max(r_crypt, 1e-8))
    lower = min(r_host, r_crypt)
    upper = max(r_host, r_crypt)
    r_mid = float(np.clip(float(r_mid), lower, upper))
    out = np.empty_like(s, dtype=float)
    left = s <= s_mid
    t_left = np.clip(s[left] / s_mid, 0.0, 1.0)
    t_right = np.clip((s[~left] - s_mid) / (1.0 - s_mid), 0.0, 1.0)
    out[left] = (1.0 - smoothstep(t_left)) * r_host + smoothstep(t_left) * r_mid
    out[~left] = (1.0 - smoothstep(t_right)) * r_mid + smoothstep(t_right) * r_crypt
    if r_host >= r_crypt:
        out = np.maximum.accumulate(out[::-1])[::-1]
    else:
        out = np.maximum.accumulate(out)
    return np.maximum(out, 1e-8)


def primitive_center(parameters: dict) -> np.ndarray | None:
    if "center" in parameters:
        return np.asarray(parameters["center"], dtype=float)
    centerline = parameters.get("centerline_points")
    if centerline is None:
        return None
    centerline = np.asarray(centerline, dtype=float)
    if centerline.ndim == 2 and centerline.shape[0]:
        return np.mean(centerline, axis=0)
    return None


def local_blob_cross_section_radius(attachment, axis_direction, *, default_radius: float) -> float:
    """Approximate the host blob radius perpendicular to a local tube axis.

    This is intentionally local to the blend direction instead of taking the
    minimum radius of the entire body/branch primitive.  It estimates the
    smallest semiaxis in the plane perpendicular to the extension axis.
    """
    axis = unit_vector(axis_direction)
    params = attachment.parameters
    primitive_type = attachment.primitive_type
    if primitive_type in {"ellipsoid", "superellipsoid_placeholder"}:
        orientation = np.asarray(params["orientation"], dtype=float)
        axes = np.asarray(params["axis_lengths"], dtype=float)
    elif primitive_type == "asymmetric_superellipsoid":
        orientation = np.asarray(params["orientation"], dtype=float)
        negative = np.asarray(params["axis_lengths_negative"], dtype=float)
        positive = np.asarray(params["axis_lengths_positive"], dtype=float)
        axes = 0.5 * (negative + positive)
    else:
        radius = params.get("radius")
        if radius is not None and np.isfinite(float(radius)) and float(radius) > 0.0:
            return float(radius)
        return float(default_radius)

    axis_local = axis @ orientation
    candidates = []
    for i in range(3):
        direction_local = np.zeros(3, dtype=float)
        direction_local[i] = 1.0
        perpendicular_weight = float(
            np.linalg.norm(direction_local - axis_local * np.dot(direction_local, axis_local))
        )
        if perpendicular_weight > 1e-3 and np.isfinite(axes[i]) and axes[i] > 0.0:
            candidates.append(float(axes[i] / perpendicular_weight))
    if not candidates:
        finite = axes[np.isfinite(axes) & (axes > 0.0)]
        if finite.size:
            return float(np.min(finite))
        return float(default_radius)
    return float(min(candidates))


def _orthonormal_frame(axis_direction) -> tuple[np.ndarray, np.ndarray]:
    axis = unit_vector(axis_direction)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    normal = np.cross(axis, reference)
    normal = unit_vector(normal)
    binormal = np.cross(axis, normal)
    return normal, unit_vector(binormal)


def _ellipsoid_ray_intersection_distance(point_local, direction_local, axes) -> float | None:
    axes = np.maximum(np.asarray(axes, dtype=float), 1e-12)
    p = np.asarray(point_local, dtype=float)
    u = np.asarray(direction_local, dtype=float)
    a = float(np.sum((u / axes) ** 2))
    b = float(2.0 * np.sum((p * u) / (axes**2)))
    c = float(np.sum((p / axes) ** 2) - 1.0)
    if a <= 1e-12:
        return None
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    sqrt_disc = float(np.sqrt(max(discriminant, 0.0)))
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    positive = [root for root in roots if root >= 1e-8]
    if not positive:
        return None
    return float(min(positive))


def _asymmetric_superellipsoid_level(point_local, negative_axes, positive_axes, epsilon_1, epsilon_2) -> float:
    point_local = np.asarray(point_local, dtype=float)
    axes = np.where(point_local >= 0.0, positive_axes, negative_axes)
    scaled = np.abs(point_local) / np.maximum(axes, 1e-12)
    xy = (
        scaled[0] ** (2.0 / epsilon_2)
        + scaled[1] ** (2.0 / epsilon_2)
    ) ** (epsilon_2 / epsilon_1)
    return float((xy + scaled[2] ** (2.0 / epsilon_1)) ** (epsilon_1 / 2.0))


def _superellipsoid_ray_intersection_distance(
    point_local,
    direction_local,
    negative_axes,
    positive_axes,
    epsilon_1,
    epsilon_2,
) -> float | None:
    p = np.asarray(point_local, dtype=float)
    u = unit_vector(direction_local)
    start = _asymmetric_superellipsoid_level(
        p,
        negative_axes,
        positive_axes,
        epsilon_1,
        epsilon_2,
    )
    if start >= 1.0:
        return None
    hi = float(max(np.max(negative_axes), np.max(positive_axes), 1e-6))
    for _ in range(32):
        level = _asymmetric_superellipsoid_level(
            p + hi * u,
            negative_axes,
            positive_axes,
            epsilon_1,
            epsilon_2,
        )
        if level >= 1.0:
            break
        hi *= 2.0
    else:
        return None
    lo = 0.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        level = _asymmetric_superellipsoid_level(
            p + mid * u,
            negative_axes,
            positive_axes,
            epsilon_1,
            epsilon_2,
        )
        if level >= 1.0:
            hi = mid
        else:
            lo = mid
    return float(hi)


def local_blob_radius_at_point(
    attachment,
    point,
    axis_direction,
    *,
    default_radius: float,
    n_angles: int = 64,
) -> float:
    """Expand a disk at ``point`` until it touches the host primitive.

    The disk lies in the plane perpendicular to ``axis_direction``.  The return
    value is the smallest positive radius to the primitive surface across
    sampled angular directions, making the blend local to its actual endpoint.
    """
    if attachment is None:
        return float(default_radius)
    point = np.asarray(point, dtype=float)
    normal, binormal = _orthonormal_frame(axis_direction)
    angles = np.linspace(0.0, 2.0 * np.pi, max(8, int(n_angles)), endpoint=False)
    directions = [
        np.cos(angle) * normal + np.sin(angle) * binormal
        for angle in angles
    ]
    params = attachment.parameters
    primitive_type = attachment.primitive_type
    distances = []
    if primitive_type in {"ellipsoid", "superellipsoid_placeholder"}:
        center = np.asarray(params["center"], dtype=float)
        orientation = np.asarray(params["orientation"], dtype=float)
        axes = np.asarray(params["axis_lengths"], dtype=float)
        point_local = (point - center) @ orientation
        for direction in directions:
            distance = _ellipsoid_ray_intersection_distance(
                point_local,
                np.asarray(direction, dtype=float) @ orientation,
                axes,
            )
            if distance is not None and np.isfinite(distance):
                distances.append(distance)
    elif primitive_type == "asymmetric_superellipsoid":
        center = np.asarray(params["center"], dtype=float)
        orientation = np.asarray(params["orientation"], dtype=float)
        negative = np.asarray(params["axis_lengths_negative"], dtype=float)
        positive = np.asarray(params["axis_lengths_positive"], dtype=float)
        point_local = (point - center) @ orientation
        for direction in directions:
            distance = _superellipsoid_ray_intersection_distance(
                point_local,
                np.asarray(direction, dtype=float) @ orientation,
                negative,
                positive,
                float(params.get("epsilon_1", 1.0)),
                float(params.get("epsilon_2", 1.0)),
            )
            if distance is not None and np.isfinite(distance):
                distances.append(distance)
    elif primitive_type == "straight_cylinder":
        radius = params.get("radius")
        if radius is not None and np.isfinite(float(radius)) and float(radius) > 0.0:
            return float(radius)
    if not distances:
        return float(default_radius)
    return float(max(min(distances), 1e-8))


def local_blob_surface_radius(attachment, direction, *, default_radius: float) -> float:
    """Distance from blob center to its surface along a world-space direction."""
    direction = unit_vector(direction)
    params = attachment.parameters
    primitive_type = attachment.primitive_type
    if primitive_type in {"ellipsoid", "superellipsoid_placeholder"}:
        orientation = np.asarray(params["orientation"], dtype=float)
        axes = np.asarray(params["axis_lengths"], dtype=float)
        local = direction @ orientation
        denom = float(np.sqrt(np.sum((local / np.maximum(axes, 1e-12)) ** 2)))
        return 1.0 / max(denom, 1e-12)
    if primitive_type == "asymmetric_superellipsoid":
        orientation = np.asarray(params["orientation"], dtype=float)
        negative = np.asarray(params["axis_lengths_negative"], dtype=float)
        positive = np.asarray(params["axis_lengths_positive"], dtype=float)
        epsilon_1 = float(params.get("epsilon_1", 1.0))
        epsilon_2 = float(params.get("epsilon_2", 1.0))
        local = direction @ orientation
        axes = np.where(local >= 0.0, positive, negative)
        scaled = np.abs(local) / np.maximum(axes, 1e-12)
        xy = (
            scaled[0] ** (2.0 / epsilon_2)
            + scaled[1] ** (2.0 / epsilon_2)
        ) ** (epsilon_2 / epsilon_1)
        denom = float((xy + scaled[2] ** (2.0 / epsilon_1)) ** (epsilon_1 / 2.0))
        return 1.0 / max(denom, 1e-12)
    return float(default_radius)

"""Low-dimensional crypt geometry from a detected surface component.

The maintained crypt model uses one final tip, a boundary-to-tip distance-ratio
coordinate, cross-sectional contour observations, and a tightly constrained
cubic Hermite centerline. Its endpoint tangent directions are fixed by the host
primitive and distal tip plane. Independent proximal and distal tangent lengths
are fitted with an optional physical bending-energy penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from organograph.skeleton.detection.mesh_regions import _boundary_vertices_from_patch
from organograph.skeleton.geometry import as_points
from organograph.skeleton.primitive_geometry import (
    point_at_polyline_arclength,
    polyline_lengths,
    project_points_to_polyline,
)


@dataclass(frozen=True)
class CryptGeometryFit:
    """Geometry shared by a crypt skeleton path and its tube primitive."""

    centerline_points: np.ndarray
    centerline_kind: str
    start_tangent: np.ndarray
    end_tangent: np.ndarray
    start_tangent_length: float
    end_tangent_length: float
    ratio_field: np.ndarray
    contour_s: np.ndarray
    contour_centers: np.ndarray
    contour_radii: np.ndarray
    contour_min_radii: np.ndarray
    contour_areas: np.ndarray
    contour_perimeters: np.ndarray
    diagnostic_s: np.ndarray
    diagnostic_radii: np.ndarray
    diagnostic_min_radii: np.ndarray
    radius_sample_s: np.ndarray
    radius_samples: np.ndarray
    radius_sample_weights: np.ndarray
    radius_contour_s: np.ndarray
    radius_mean_radii: np.ndarray
    radius_min_radii: np.ndarray
    radius_contour_perimeters: np.ndarray
    radius_contour_counts: np.ndarray
    initial_center_s: float
    initial_center: np.ndarray
    opening_normal: np.ndarray
    tip_normal: np.ndarray | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HermiteCenterlineFit:
    """Result of the two-tangent regularized Hermite fit."""

    centerline_points: np.ndarray
    start_tangent: np.ndarray
    end_tangent: np.ndarray
    start_tangent_length: float
    end_tangent_length: float
    contour_s: np.ndarray
    contour_closest_points: np.ndarray
    fit_rmse: float
    normalized_data_mse: float
    bending_energy: float
    dimensionless_bending_energy: float
    total_bend_angle: float
    max_curvature: float
    p95_curvature: float
    curvature_localization: float
    fold_penalty: float
    objective: float
    reference_length: float
    curvature_weight: float
    success: bool
    message: str


def restricted_surface_distance_field(vertices, faces, region, sources) -> np.ndarray:
    """Compute multi-source edge-geodesic distance within one mesh region."""
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    region = np.unique(np.asarray(region, dtype=np.int64).reshape(-1))
    sources = np.unique(np.asarray(sources, dtype=np.int64).reshape(-1))
    valid_region = region[(region >= 0) & (region < vertices.shape[0])]
    allowed = np.zeros(vertices.shape[0], dtype=bool)
    allowed[valid_region] = True
    sources = sources[(sources >= 0) & (sources < vertices.shape[0]) & allowed[sources]]
    field = np.full(vertices.shape[0], np.inf, dtype=float)
    if valid_region.size == 0 or sources.size == 0:
        return field
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edges = edges[allowed[edges[:, 0]] & allowed[edges[:, 1]]]
    local_index = np.full(vertices.shape[0], -1, dtype=np.int64)
    local_index[valid_region] = np.arange(valid_region.size)
    a = local_index[edges[:, 0]]
    b = local_index[edges[:, 1]]
    weights = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    graph = csr_matrix(
        (np.r_[weights, weights], (np.r_[a, b], np.r_[b, a])),
        shape=(valid_region.size, valid_region.size),
    )
    distances = dijkstra(
        graph,
        directed=False,
        indices=local_index[sources],
        return_predecessors=False,
        min_only=True,
    )
    field[valid_region] = np.asarray(distances, dtype=float)
    return field


def boundary_tip_ratio_field(vertices, faces, region, tip_vertex_id, boundary=None):
    """Return ``d_boundary / (d_boundary + d_tip)`` within a crypt patch."""
    region = np.unique(np.asarray(region, dtype=np.int64).reshape(-1))
    if boundary is None:
        boundary = _boundary_vertices_from_patch(faces, region)
    boundary = np.asarray(boundary, dtype=np.int64).reshape(-1)
    d_boundary = restricted_surface_distance_field(vertices, faces, region, boundary)
    d_tip = restricted_surface_distance_field(vertices, faces, region, [tip_vertex_id])
    denominator = d_boundary + d_tip
    ratio = np.full(np.asarray(vertices).shape[0], np.nan, dtype=float)
    valid = np.isfinite(denominator) & (denominator > 1e-12)
    ratio[valid] = np.clip(d_boundary[valid] / denominator[valid], 0.0, 1.0)
    ratio[int(tip_vertex_id)] = 1.0
    return ratio, d_boundary, d_tip, boundary


def _contour_segments(vertices, faces, scalar, level, region_mask) -> np.ndarray:
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    scalar = np.asarray(scalar, dtype=float)
    selected_faces = faces[np.all(region_mask[faces], axis=1)]
    segments = []
    for face in selected_faces:
        values = scalar[face]
        if not np.all(np.isfinite(values)) or level < np.min(values) or level > np.max(values):
            continue
        intersections = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            va, vb = values[first], values[second]
            if (va - level) * (vb - level) > 0.0 or abs(vb - va) <= 1e-14:
                continue
            fraction = float(np.clip((level - va) / (vb - va), 0.0, 1.0))
            intersections.append(
                vertices[face[first]] + fraction * (vertices[face[second]] - vertices[face[first]])
            )
        if len(intersections) >= 2:
            unique = []
            for point in intersections:
                if not any(np.linalg.norm(point - prior) <= 1e-10 for prior in unique):
                    unique.append(point)
            if len(unique) >= 2:
                segments.append([unique[0], unique[1]])
    return np.asarray(segments, dtype=float).reshape(-1, 2, 3)


def contour_observations(vertices, faces, region, ratio_field, *, n_contours=10):
    """Measure length-weighted centers and equivalent radii of ratio contours."""
    vertices = as_points(vertices)
    region = np.asarray(region, dtype=np.int64).reshape(-1)
    region_mask = np.zeros(vertices.shape[0], dtype=bool)
    region_mask[region] = True
    levels = np.linspace(0.05, 0.90, max(4, int(n_contours)))
    output = []
    for level in levels:
        segments = _contour_segments(vertices, faces, ratio_field, float(level), region_mask)
        if segments.size == 0:
            continue
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
        valid = lengths > 1e-12
        if np.sum(valid) < 2:
            continue
        segments = segments[valid]
        lengths = lengths[valid]
        perimeter = float(np.sum(lengths))
        center = np.average(np.mean(segments, axis=1), weights=lengths, axis=0)
        radius = perimeter / (2.0 * np.pi)
        output.append(
            {
                "level": float(level),
                "center": center,
                "perimeter": perimeter,
                "radius": radius,
                "area": float(np.pi * radius**2),
                "segments": segments,
                "points": segments.reshape(-1, 3),
            }
        )
    return output


def _boundary_contour_observation(vertices, faces, region, boundary):
    """Measure the induced crypt-patch boundary as the ``s=0`` contour."""
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    region = np.asarray(region, dtype=np.int64).reshape(-1)
    region_mask = np.zeros(vertices.shape[0], dtype=bool)
    region_mask[region] = True
    selected_faces = faces[np.all(region_mask[faces], axis=1)]
    if selected_faces.size:
        edges = np.vstack(
            [
                selected_faces[:, [0, 1]],
                selected_faces[:, [1, 2]],
                selected_faces[:, [2, 0]],
            ]
        )
        unique_edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
    else:
        boundary_edges = np.empty((0, 2), dtype=np.int64)
    if boundary_edges.size:
        segments = vertices[boundary_edges]
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
        valid = lengths > 1e-12
        segments = segments[valid]
        lengths = lengths[valid]
        if lengths.size:
            perimeter = float(np.sum(lengths))
            center = np.average(np.mean(segments, axis=1), weights=lengths, axis=0)
            radius = perimeter / (2.0 * np.pi)
            return {
                "level": 0.0,
                "center": center,
                "perimeter": perimeter,
                "radius": radius,
                "area": float(np.pi * radius**2),
                "segments": segments,
                "points": segments.reshape(-1, 3),
            }

    points = vertices[np.asarray(boundary, dtype=np.int64).reshape(-1)]
    center = np.mean(points, axis=0)
    radius = float(np.median(np.linalg.norm(points - center, axis=1)))
    return {
        "level": 0.0,
        "center": center,
        "perimeter": float(2.0 * np.pi * radius),
        "radius": radius,
        "area": float(np.pi * radius**2),
        "points": points,
    }


def _band_fallback_observations(vertices, region, ratio_field, *, n_contours=10):
    """Approximate contour geometry when a tiny patch has incomplete faces."""
    vertices = as_points(vertices)
    region = np.asarray(region, dtype=np.int64).reshape(-1)
    values = np.asarray(ratio_field, dtype=float)[region]
    finite = np.isfinite(values)
    region = region[finite]
    values = values[finite]
    if region.size < 6:
        return []
    levels = np.linspace(0.05, 0.90, max(4, int(n_contours)))
    half_width = 0.5 / max(levels.size - 1, 1)
    output = []
    for level in levels:
        selected = np.abs(values - level) <= half_width
        if np.count_nonzero(selected) < 3:
            nearest = np.argsort(np.abs(values - level))[: min(8, region.size)]
            points = vertices[region[nearest]]
        else:
            points = vertices[region[selected]]
        if points.shape[0] < 3:
            continue
        center = np.mean(points, axis=0)
        radius = float(np.median(np.linalg.norm(points - center, axis=1)))
        output.append(
            {
                "level": float(level),
                "center": center,
                "perimeter": float(2.0 * np.pi * radius),
                "radius": radius,
                "area": float(np.pi * radius**2),
                "points": points,
            }
        )
    return output


def minimum_contour_radius(contour_points, centerline_point) -> float:
    """Return the shortest distance from a centerline point to a contour.

    Exact mesh iso-contours are represented as consecutive line-segment endpoint
    pairs. Ratio-band fallback contours are unordered point samples, for which
    the minimum sampled distance is used instead.
    """
    raw = np.asarray(contour_points, dtype=float)
    center = np.asarray(centerline_point, dtype=float).reshape(3)
    if raw.ndim == 3 and raw.shape[1:] == (2, 3):
        segments = raw
        points = segments.reshape(-1, 3)
    else:
        points = as_points(raw)
        segments = None
    if points.shape[0] == 0:
        return float("nan")
    if segments is None:
        return float(np.min(np.linalg.norm(points - center, axis=1)))
    starts = segments[:, 0]
    directions = segments[:, 1] - starts
    denominator = np.sum(directions**2, axis=1)
    parameter = np.zeros(segments.shape[0], dtype=float)
    valid = denominator > 1e-20
    parameter[valid] = np.sum(
        (center - starts[valid]) * directions[valid], axis=1
    ) / denominator[valid]
    parameter = np.clip(parameter, 0.0, 1.0)
    closest = starts + parameter[:, None] * directions
    return float(np.min(np.linalg.norm(closest - center, axis=1)))


def _point_and_tangent_at_polyline_arclength(centerline, fraction):
    """Return a point and forward tangent at normalized polyline arc length."""
    line = as_points(centerline)
    lengths, cumulative, total = polyline_lengths(line)
    if line.shape[0] < 2 or total <= 1e-12:
        return line[0].copy(), np.array([1.0, 0.0, 0.0])
    target = float(np.clip(fraction, 0.0, 1.0)) * total
    segment = int(np.searchsorted(cumulative, target, side="right") - 1)
    segment = max(0, min(segment, lengths.size - 1))
    direction = line[segment + 1] - line[segment]
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    local = (target - cumulative[segment]) / max(lengths[segment], 1e-12)
    point = line[segment] + local * (line[segment + 1] - line[segment])
    return point, direction


def _transverse_radius_observation(contour, centerline, level):
    """Measure arc-length-weighted transverse radii for one mesh contour."""
    center, tangent = _point_and_tangent_at_polyline_arclength(centerline, level)
    segments = contour.get("segments")
    if segments is not None:
        segments = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
        lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
        valid = np.isfinite(lengths) & (lengths > 1e-12)
        segments = segments[valid]
        lengths = lengths[valid]
        if lengths.size == 0:
            return None
        sample_points = np.mean(segments, axis=1)
        weights = lengths / np.sum(lengths)
        relative_endpoints = segments - center[None, None, :]
        transverse_endpoints = relative_endpoints - (
            relative_endpoints @ tangent
        )[..., None] * tangent[None, None, :]
        starts = transverse_endpoints[:, 0]
        directions = transverse_endpoints[:, 1] - starts
        denominator = np.sum(directions**2, axis=1)
        closest_parameter = np.zeros(segments.shape[0], dtype=float)
        supported = denominator > 1e-20
        closest_parameter[supported] = -np.sum(
            starts[supported] * directions[supported], axis=1
        ) / denominator[supported]
        closest_parameter = np.clip(closest_parameter, 0.0, 1.0)
        closest_transverse = starts + closest_parameter[:, None] * directions
        minimum_radius = float(
            np.min(np.linalg.norm(closest_transverse, axis=1))
        )
        perimeter = float(np.sum(lengths))
    else:
        sample_points = as_points(contour["points"])
        if sample_points.shape[0] == 0:
            return None
        weights = np.full(sample_points.shape[0], 1.0 / sample_points.shape[0])
        relative = sample_points - center[None, :]
        transverse = relative - (relative @ tangent)[:, None] * tangent[None, :]
        minimum_radius = float(np.min(np.linalg.norm(transverse, axis=1)))
        perimeter = float(contour.get("perimeter", np.nan))

    relative = sample_points - center[None, :]
    transverse = relative - (relative @ tangent)[:, None] * tangent[None, :]
    radii = np.linalg.norm(transverse, axis=1)
    valid = np.isfinite(radii) & np.isfinite(weights) & (weights > 0.0)
    radii = radii[valid]
    weights = weights[valid]
    if radii.size == 0:
        return None
    weights /= np.sum(weights)
    mean_radius = float(np.sum(weights * radii))
    if not np.isfinite(perimeter):
        # A projected vertex band is only a sparse fallback, not an ordered
        # contour. Keep diagnostics JSON-safe with its circular-equivalent
        # perimeter while retaining the source label on the observation.
        perimeter = float(2.0 * np.pi * mean_radius)
    return {
        "s": float(level),
        "radii": radii,
        "weights": weights,
        "mean_radius": mean_radius,
        "min_radius": minimum_radius,
        "perimeter": perimeter,
        "count": int(radii.size),
    }


def centerline_radius_observations(
    vertices,
    faces,
    region,
    centerline,
    boundary,
    *,
    n_contours=19,
    max_s=0.95,
):
    """Extract transverse-radius samples at fixed centerline coordinates.

    The crypt mesh vertices are projected onto the final centerline to define a
    scalar arc-length field. Its iso-contours provide cross-sections whose
    longitudinal coordinates exactly match those used by the tube primitive.
    """
    vertices = as_points(vertices)
    region = np.unique(np.asarray(region, dtype=np.int64).reshape(-1))
    region_mask = np.zeros(vertices.shape[0], dtype=bool)
    region_mask[region] = True
    projection = project_points_to_polyline(vertices[region], centerline)
    centerline_s = np.full(vertices.shape[0], np.nan, dtype=float)
    centerline_s[region] = projection["s"]

    observations = []
    boundary_contour = _boundary_contour_observation(
        vertices, faces, region, boundary
    )
    boundary_observation = _transverse_radius_observation(
        boundary_contour, centerline, 0.0
    )
    if boundary_observation is not None:
        boundary_observation["source"] = "crypt_boundary"
        observations.append(boundary_observation)

    levels = np.linspace(0.05, float(max_s), max(4, int(n_contours)))
    level_half_width = 0.5 * float(levels[1] - levels[0])
    local_s = np.asarray(projection["s"], dtype=float)
    for level in levels:
        segments = _contour_segments(
            vertices, faces, centerline_s, float(level), region_mask
        )
        if segments.size:
            contour = {
                "segments": segments,
                "points": segments.reshape(-1, 3),
                "perimeter": float(
                    np.sum(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1))
                ),
            }
            source = "centerline_coordinate_isocontour"
        else:
            selected = np.abs(local_s - float(level)) <= level_half_width
            if np.count_nonzero(selected) < 3:
                nearest = np.argsort(np.abs(local_s - float(level)))[: min(8, region.size)]
                if (
                    nearest.size < 3
                    or abs(float(local_s[nearest[0]]) - float(level))
                    > 2.0 * level_half_width
                ):
                    continue
                sample_points = vertices[region[nearest]]
            else:
                sample_points = vertices[region[selected]]
            contour = {"points": sample_points, "perimeter": float("nan")}
            source = "centerline_coordinate_band_fallback"
        observation = _transverse_radius_observation(
            contour, centerline, float(level)
        )
        if observation is not None:
            observation["source"] = source
            observations.append(observation)
    return observations, centerline_s


def _oriented_unit_vector(vector, chord_direction) -> np.ndarray:
    """Normalize a tangent direction and orient it from attachment to tip."""
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.asarray(chord_direction, dtype=float).copy()
    vector = vector / norm
    if float(np.dot(vector, chord_direction)) < 0.0:
        vector = -vector
    return vector


def sample_tangent_hermite(
    start,
    end,
    start_tangent,
    end_tangent,
    *,
    n_samples=64,
) -> np.ndarray:
    """Sample a fixed-endpoint cubic Hermite centerline.

    ``start_tangent`` and ``end_tangent`` are derivative vectors, so their
    magnitudes control how strongly the curve follows each endpoint normal.
    """
    start = np.asarray(start, dtype=float).reshape(3)
    end = np.asarray(end, dtype=float).reshape(3)
    start_tangent = np.asarray(start_tangent, dtype=float).reshape(3)
    end_tangent = np.asarray(end_tangent, dtype=float).reshape(3)
    u = np.linspace(0.0, 1.0, max(2, int(n_samples)))
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    curve = (
        h00[:, None] * start
        + h10[:, None] * start_tangent
        + h01[:, None] * end
        + h11[:, None] * end_tangent
    )
    curve[0] = start
    curve[-1] = end
    return curve


def monotonic_project_points_to_polyline(points, centerline):
    """Project an ordered point sequence onto a polyline without reversing order.

    Independent continuous closest-point coordinates are made nondecreasing by
    least-squares isotonic regression. This decouples the mesh ratio coordinate
    used to order contours from centerline arc length, while preventing
    neighboring sections from swapping order.
    """
    points = as_points(points)
    line = as_points(centerline)
    if line.shape[0] == 0:
        raise ValueError("Centerline cannot be empty")
    if points.shape[0] == 0:
        return {
            "closest_points": np.empty((0, 3), dtype=float),
            "distances": np.empty(0, dtype=float),
            "s": np.empty(0, dtype=float),
        }
    projection = project_points_to_polyline(points, line)
    raw_s = np.asarray(projection["s"], dtype=float)

    # Pool-adjacent-violators algorithm with unit weights.
    means: list[float] = []
    weights: list[int] = []
    for value in raw_s:
        means.append(float(value))
        weights.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            combined_weight = weights[-2] + weights[-1]
            combined_mean = (
                weights[-2] * means[-2] + weights[-1] * means[-1]
            ) / combined_weight
            means[-2:] = [combined_mean]
            weights[-2:] = [combined_weight]
    monotonic_s = np.concatenate(
        [np.full(weight, mean, dtype=float) for mean, weight in zip(means, weights)]
    )
    monotonic_s = np.clip(monotonic_s, 0.0, 1.0)
    closest = np.asarray(
        [point_at_polyline_arclength(line, value) for value in monotonic_s],
        dtype=float,
    )
    return {
        "closest_points": closest,
        "distances": np.linalg.norm(points - closest, axis=1),
        "s": monotonic_s,
        "unconstrained_s": raw_s,
    }


def _hermite_derivatives(start, end, start_tangent, end_tangent, u):
    """Evaluate first and second derivatives of a cubic Hermite curve."""
    u = np.asarray(u, dtype=float).reshape(-1)
    first = (
        (6.0 * u**2 - 6.0 * u)[:, None] * start
        + (3.0 * u**2 - 4.0 * u + 1.0)[:, None] * start_tangent
        + (-6.0 * u**2 + 6.0 * u)[:, None] * end
        + (3.0 * u**2 - 2.0 * u)[:, None] * end_tangent
    )
    second = (
        (12.0 * u - 6.0)[:, None] * start
        + (6.0 * u - 4.0)[:, None] * start_tangent
        + (-12.0 * u + 6.0)[:, None] * end
        + (6.0 * u - 2.0)[:, None] * end_tangent
    )
    return first, second


def hermite_curvature_diagnostics(
    start,
    end,
    start_tangent,
    end_tangent,
    *,
    n_samples=257,
):
    """Measure physical curvature and bending energy for one Hermite curve."""
    start = np.asarray(start, dtype=float).reshape(3)
    end = np.asarray(end, dtype=float).reshape(3)
    start_tangent = np.asarray(start_tangent, dtype=float).reshape(3)
    end_tangent = np.asarray(end_tangent, dtype=float).reshape(3)
    u = np.linspace(0.0, 1.0, max(33, int(n_samples)))
    first, second = _hermite_derivatives(
        start, end, start_tangent, end_tangent, u
    )
    speed = np.linalg.norm(first, axis=1)
    length_scale = max(float(np.linalg.norm(end - start)), 1e-12)
    safe_speed = np.maximum(speed, 1e-8 * length_scale)
    cross_norm = np.linalg.norm(np.cross(first, second), axis=1)
    curvature = cross_norm / np.maximum(safe_speed**3, 1e-30)
    curve_length = float(np.trapezoid(speed, u))
    bending_energy = float(np.trapezoid(curvature**2 * speed, u))
    total_bend_angle = float(np.trapezoid(curvature * speed, u))
    rms_curvature = np.sqrt(bending_energy / max(curve_length, 1e-12))
    max_curvature = float(np.max(curvature))
    return {
        "curve_length": curve_length,
        "bending_energy": bending_energy,
        "total_bend_angle": total_bend_angle,
        "max_curvature": max_curvature,
        "p95_curvature": float(np.percentile(curvature, 95.0)),
        "curvature_localization": float(
            max_curvature / max(float(rms_curvature), 1e-12)
        ),
    }


def fit_tangent_constrained_hermite(
    start,
    end,
    centers,
    parameters,
    start_normal,
    end_normal,
    *,
    n_samples=64,
    max_tangent_length_fraction=1.25,
    curvature_weight=0.0,
    reference_length=None,
) -> HermiteCenterlineFit:
    """Fit endpoint tangent lengths with physical bending regularization.

    Contour centers are matched to monotonically ordered positions on the
    centerline. The objective is dimensionless: squared centerline residuals
    are divided by ``reference_length ** 2`` and the physical bending energy
    ``integral(kappa ** 2 dl)`` is multiplied by ``reference_length``.
    """
    start = np.asarray(start, dtype=float).reshape(3)
    end = np.asarray(end, dtype=float).reshape(3)
    centers = as_points(centers)
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    chord = end - start
    length = float(np.linalg.norm(chord))
    reference_length = length if reference_length is None else float(reference_length)
    if not np.isfinite(reference_length) or reference_length <= 0.0:
        raise ValueError("reference_length must be a positive finite value")
    curvature_weight = float(curvature_weight)
    if not np.isfinite(curvature_weight) or curvature_weight < 0.0:
        raise ValueError("curvature_weight must be a non-negative finite value")
    if length <= 1e-12:
        zero = np.zeros(3, dtype=float)
        centerline = np.repeat(start[None, :], max(2, int(n_samples)), axis=0)
        return HermiteCenterlineFit(
            centerline_points=centerline,
            start_tangent=zero,
            end_tangent=zero,
            start_tangent_length=0.0,
            end_tangent_length=0.0,
            contour_s=np.zeros(centers.shape[0], dtype=float),
            contour_closest_points=np.repeat(start[None, :], centers.shape[0], axis=0),
            fit_rmse=0.0,
            normalized_data_mse=0.0,
            bending_energy=0.0,
            dimensionless_bending_energy=0.0,
            total_bend_angle=0.0,
            max_curvature=0.0,
            p95_curvature=0.0,
            curvature_localization=0.0,
            fold_penalty=0.0,
            objective=0.0,
            reference_length=reference_length,
            curvature_weight=curvature_weight,
            success=True,
            message="degenerate zero-length centerline",
        )
    chord_direction = chord / length
    start_direction = _oriented_unit_vector(start_normal, chord_direction)
    end_direction = _oriented_unit_vector(end_normal, chord_direction)
    valid = np.isfinite(parameters)
    centers = centers[valid]
    parameters = np.clip(parameters[valid], 0.0, 1.0)
    order = np.argsort(parameters, kind="stable")
    centers = centers[order]

    def curve_for(length_fractions):
        start_length, end_length = length * np.asarray(length_fractions, dtype=float)
        return sample_tangent_hermite(
            start,
            end,
            float(start_length) * start_direction,
            float(end_length) * end_direction,
            n_samples=n_samples,
        )

    def objective_terms(length_fractions):
        start_length, end_length = length * np.asarray(length_fractions, dtype=float)
        start_tangent = float(start_length) * start_direction
        end_tangent = float(end_length) * end_direction
        curve = sample_tangent_hermite(
            start, end, start_tangent, end_tangent, n_samples=n_samples
        )
        if centers.shape[0]:
            projection = monotonic_project_points_to_polyline(centers, curve)
            data_mse = float(np.mean(projection["distances"] ** 2))
        else:
            projection = monotonic_project_points_to_polyline(centers, curve)
            data_mse = 0.0
        normalized_data_mse = data_mse / reference_length**2
        curvature = hermite_curvature_diagnostics(
            start, end, start_tangent, end_tangent
        )
        dimensionless_bending_energy = (
            reference_length * curvature["bending_energy"]
        )
        steps = np.diff(curve, axis=0)
        forward = steps @ chord_direction
        backwards = np.minimum(forward, 0.0)
        fold_penalty = 100.0 * float(np.mean(backwards**2)) / reference_length**2
        curvature_penalty = (
            0.0
            if curvature_weight == 0.0
            else curvature_weight * dimensionless_bending_energy
        )
        objective = normalized_data_mse + curvature_penalty + fold_penalty
        return objective, projection, curvature, normalized_data_mse, fold_penalty

    result = minimize(
        lambda values: objective_terms(values)[0],
        x0=np.ones(2, dtype=float),
        method="Powell",
        bounds=[
            (0.05, float(max_tangent_length_fraction)),
            (0.05, float(max_tangent_length_fraction)),
        ],
        options={"xtol": 1e-5, "ftol": 1e-9, "maxiter": 300},
    )
    fitted_values = np.asarray(result.x, dtype=float)
    fractions = (
        fitted_values
        if fitted_values.shape == (2,) and np.all(np.isfinite(fitted_values))
        else np.ones(2, dtype=float)
    )
    start_length, end_length = length * fractions
    start_tangent = float(start_length) * start_direction
    end_tangent = float(end_length) * end_direction
    centerline = curve_for(fractions)
    objective, projection, curvature, normalized_data_mse, fold_penalty = (
        objective_terms(fractions)
    )
    fit_rmse = (
        float(np.sqrt(np.mean(projection["distances"] ** 2)))
        if centers.shape[0]
        else 0.0
    )
    return HermiteCenterlineFit(
        centerline_points=centerline,
        start_tangent=start_tangent,
        end_tangent=end_tangent,
        start_tangent_length=float(start_length),
        end_tangent_length=float(end_length),
        contour_s=np.asarray(projection["s"], dtype=float),
        contour_closest_points=np.asarray(projection["closest_points"], dtype=float),
        fit_rmse=fit_rmse,
        normalized_data_mse=float(normalized_data_mse),
        bending_energy=float(curvature["bending_energy"]),
        dimensionless_bending_energy=float(
            reference_length * curvature["bending_energy"]
        ),
        total_bend_angle=float(curvature["total_bend_angle"]),
        max_curvature=float(curvature["max_curvature"]),
        p95_curvature=float(curvature["p95_curvature"]),
        curvature_localization=float(curvature["curvature_localization"]),
        fold_penalty=float(fold_penalty),
        objective=float(objective),
        reference_length=reference_length,
        curvature_weight=curvature_weight,
        success=bool(result.success),
        message=str(result.message),
    )


def _normal_from_contour(points, preferred_direction) -> np.ndarray | None:
    points = as_points(points)
    if points.shape[0] < 3:
        return None
    centered = points - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    preferred = np.asarray(preferred_direction, dtype=float)
    if np.dot(normal, preferred) < 0.0:
        normal = -normal
    return normal / max(float(np.linalg.norm(normal)), 1e-12)


def fit_crypt_geometry(
    vertices,
    faces,
    component_vertices,
    attachment,
    tip_vertex_id,
    *,
    boundary_vertices=None,
    opening_normal=None,
    n_contours=10,
    radius_n_contours=19,
    n_samples=64,
    curvature_weight=0.0,
    reference_length=None,
) -> CryptGeometryFit:
    """Fit ratio contours and an endpoint-normal constrained centerline."""
    vertices = as_points(vertices)
    region = np.unique(np.asarray(component_vertices, dtype=np.int64).reshape(-1))
    tip_vertex_id = int(tip_vertex_id)
    ratio, d_boundary, d_tip, boundary = boundary_tip_ratio_field(
        vertices, faces, region, tip_vertex_id, boundary=boundary_vertices
    )
    observations = contour_observations(
        vertices, faces, region, ratio, n_contours=n_contours
    )
    observation_source = "mesh_iso_contours"
    if len(observations) < 3:
        observations = _band_fallback_observations(
            vertices, region, ratio, n_contours=n_contours
        )
        observation_source = "ratio_band_fallback"
    if len(observations) < 3:
        raise ValueError("Crypt geometry needs at least three supported cross-sections")
    levels = np.asarray([item["level"] for item in observations], dtype=float)
    centers = np.asarray([item["center"] for item in observations], dtype=float)
    tip = vertices[tip_vertex_id]
    distal = min(observations, key=lambda item: abs(item["level"] - 0.85))
    tip_normal = _normal_from_contour(distal["points"], tip - np.asarray(attachment))
    chord = tip - np.asarray(attachment, dtype=float)
    chord_direction = chord / max(float(np.linalg.norm(chord)), 1e-12)
    if opening_normal is None:
        opening_normal = np.asarray(attachment) - np.mean(vertices[boundary], axis=0)
    opening_normal = _oriented_unit_vector(opening_normal, chord_direction)
    if tip_normal is None:
        tip_normal = chord_direction.copy()
    centerline_fit = fit_tangent_constrained_hermite(
        attachment,
        tip,
        centers,
        levels,
        opening_normal,
        tip_normal,
        n_samples=n_samples,
        curvature_weight=curvature_weight,
        reference_length=reference_length,
    )
    centerline = centerline_fit.centerline_points
    # Radius fitting remains in centerline arc-length coordinates. Because the
    # constrained Hermite curve only approximates the contour centers, their
    # closest centerline coordinates need not equal their ratio-field levels.
    contour_s = centerline_fit.contour_s
    minimum_radii = np.asarray(
        [
            minimum_contour_radius(
                item.get("segments", item["points"]),
                point_at_polyline_arclength(centerline, position),
            )
            for item, position in zip(observations, contour_s)
        ],
        dtype=float,
    )
    order = np.argsort(contour_s)
    contour_s = contour_s[order]
    centers = centers[order]
    radii = np.asarray([item["radius"] for item in observations], dtype=float)[order]
    minimum_radii = minimum_radii[order]
    areas = np.asarray([item["area"] for item in observations], dtype=float)[order]
    perimeters = np.asarray([item["perimeter"] for item in observations], dtype=float)[order]

    # Notebook diagnostics use the crypt-specific ratio levels directly. The
    # boundary and tip endpoints are included here only; they must not alter
    # the fitted radius profile.
    diagnostic_observations = [
        _boundary_contour_observation(vertices, faces, region, boundary),
        *observations,
        {
            "level": 1.0,
            "center": tip.copy(),
            "perimeter": 0.0,
            "radius": 0.0,
            "area": 0.0,
            "points": tip.reshape(1, 3),
        },
    ]
    diagnostic_s = np.asarray(
        [item["level"] for item in diagnostic_observations], dtype=float
    )
    diagnostic_radii = np.asarray(
        [item["radius"] for item in diagnostic_observations], dtype=float
    )
    diagnostic_min_radii = np.asarray(
        [
            minimum_contour_radius(
                item.get("segments", item["points"]),
                point_at_polyline_arclength(centerline, position),
            )
            for item, position in zip(diagnostic_observations, diagnostic_s)
        ],
        dtype=float,
    )
    radius_observations, centerline_coordinate = centerline_radius_observations(
        vertices,
        faces,
        region,
        centerline,
        boundary,
        n_contours=radius_n_contours,
        max_s=0.95,
    )
    radius_contour_s = np.asarray(
        [item["s"] for item in radius_observations], dtype=float
    )
    radius_mean_radii = np.asarray(
        [item["mean_radius"] for item in radius_observations], dtype=float
    )
    radius_min_radii = np.asarray(
        [item["min_radius"] for item in radius_observations], dtype=float
    )
    radius_contour_perimeters = np.asarray(
        [item["perimeter"] for item in radius_observations], dtype=float
    )
    radius_contour_counts = np.asarray(
        [item["count"] for item in radius_observations], dtype=np.int64
    )
    if radius_observations:
        radius_sample_s = np.concatenate(
            [
                np.full(item["radii"].size, item["s"], dtype=float)
                for item in radius_observations
            ]
        )
        radius_samples = np.concatenate(
            [np.asarray(item["radii"], dtype=float) for item in radius_observations]
        )
        radius_sample_weights = np.concatenate(
            [np.asarray(item["weights"], dtype=float) for item in radius_observations]
        )
    else:
        radius_sample_s = np.empty(0, dtype=float)
        radius_samples = np.empty(0, dtype=float)
        radius_sample_weights = np.empty(0, dtype=float)

    center_areas = np.pi * radius_mean_radii**2
    area_integral = (
        float(np.trapezoid(center_areas, radius_contour_s))
        if radius_contour_s.size >= 2
        else 0.0
    )
    if area_integral > 1e-12:
        center_s = float(
            np.trapezoid(
                center_areas * radius_contour_s, radius_contour_s
            )
            / area_integral
        )
    else:
        center_s = float(np.average(contour_s, weights=np.maximum(areas, 1e-12)))
    # The distal radius landmark remains fixed at 0.85, so the area center must
    # precede it by only a numerical margin rather than an old biological clamp.
    center_s = float(np.clip(center_s, 1e-3, 0.85 - 1e-3))
    crypt_center = point_at_polyline_arclength(centerline, center_s)
    return CryptGeometryFit(
        centerline_points=centerline,
        centerline_kind="tangent_hermite",
        start_tangent=centerline_fit.start_tangent,
        end_tangent=centerline_fit.end_tangent,
        start_tangent_length=centerline_fit.start_tangent_length,
        end_tangent_length=centerline_fit.end_tangent_length,
        ratio_field=ratio,
        contour_s=contour_s,
        contour_centers=centers,
        contour_radii=radii,
        contour_min_radii=minimum_radii,
        contour_areas=areas,
        contour_perimeters=perimeters,
        diagnostic_s=diagnostic_s,
        diagnostic_radii=diagnostic_radii,
        diagnostic_min_radii=diagnostic_min_radii,
        radius_sample_s=radius_sample_s,
        radius_samples=radius_samples,
        radius_sample_weights=radius_sample_weights,
        radius_contour_s=radius_contour_s,
        radius_mean_radii=radius_mean_radii,
        radius_min_radii=radius_min_radii,
        radius_contour_perimeters=radius_contour_perimeters,
        radius_contour_counts=radius_contour_counts,
        initial_center_s=center_s,
        initial_center=crypt_center,
        opening_normal=opening_normal,
        tip_normal=tip_normal,
        metadata={
            "method": "boundary_tip_ratio_contours_tangent_constrained_hermite",
            "centerline_fit_rmse": centerline_fit.fit_rmse,
            "centerline_normalized_data_mse": centerline_fit.normalized_data_mse,
            "centerline_bending_energy": centerline_fit.bending_energy,
            "centerline_dimensionless_bending_energy": (
                centerline_fit.dimensionless_bending_energy
            ),
            "centerline_total_bend_angle": centerline_fit.total_bend_angle,
            "centerline_max_curvature": centerline_fit.max_curvature,
            "centerline_p95_curvature": centerline_fit.p95_curvature,
            "centerline_curvature_localization": (
                centerline_fit.curvature_localization
            ),
            "centerline_fold_penalty": centerline_fit.fold_penalty,
            "centerline_objective": centerline_fit.objective,
            "centerline_reference_length": centerline_fit.reference_length,
            "centerline_curvature_weight": centerline_fit.curvature_weight,
            "centerline_optimizer_success": centerline_fit.success,
            "centerline_optimizer_message": centerline_fit.message,
            "center_initialization": "mesh_contour_area_weighted_center",
            "cross_section_source": observation_source,
            "fit_contour_coordinate": "monotonic_projected_centerline_arclength",
            "diagnostic_contour_coordinate": "crypt_boundary_tip_ratio",
            "radius_observation_coordinate": "centerline_normalized_arclength",
            "radius_observation_source": "centerline_coordinate_isocontours",
            "radius_observation_max_s": 0.95,
            "radius_supported_sections": int(radius_contour_s.size),
            "boundary_vertex_ids": boundary,
            "d_boundary": d_boundary,
            "d_tip": d_tip,
            "centerline_coordinate": centerline_coordinate,
        },
    )


__all__ = [
    "CryptGeometryFit",
    "HermiteCenterlineFit",
    "boundary_tip_ratio_field",
    "centerline_radius_observations",
    "contour_observations",
    "fit_crypt_geometry",
    "fit_tangent_constrained_hermite",
    "hermite_curvature_diagnostics",
    "minimum_contour_radius",
    "monotonic_project_points_to_polyline",
    "restricted_surface_distance_field",
    "sample_tangent_hermite",
]

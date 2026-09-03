"""Low-dimensional crypt geometry from a detected surface component.

The maintained crypt model uses one final tip, a boundary-to-tip distance-ratio
coordinate, cross-sectional contour observations, and a tightly constrained
cubic Hermite centerline. Its endpoint tangent directions are fixed by the host
primitive and distal tip plane; only one shared tangent length is fitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from organograph.skeleton.detection.mesh_regions import _boundary_vertices_from_patch
from organograph.skeleton.geometry import as_points
from organograph.skeleton.primitive_geometry import (
    point_at_polyline_arclength,
    project_points_to_polyline,
)


@dataclass(frozen=True)
class CryptGeometryFit:
    """Geometry shared by a crypt skeleton path and its tube primitive."""

    centerline_points: np.ndarray
    centerline_kind: str
    start_tangent: np.ndarray
    end_tangent: np.ndarray
    tangent_length: float
    ratio_field: np.ndarray
    contour_s: np.ndarray
    contour_centers: np.ndarray
    contour_radii: np.ndarray
    contour_areas: np.ndarray
    contour_perimeters: np.ndarray
    initial_center_s: float
    initial_center: np.ndarray
    opening_normal: np.ndarray
    tip_normal: np.ndarray | None
    metadata: dict[str, Any]


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
                "points": segments.reshape(-1, 3),
            }
        )
    return output


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
):
    """Fit one shared tangent length with endpoint directions held fixed."""
    start = np.asarray(start, dtype=float).reshape(3)
    end = np.asarray(end, dtype=float).reshape(3)
    centers = as_points(centers)
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    chord = end - start
    length = float(np.linalg.norm(chord))
    if length <= 1e-12:
        zero = np.zeros(3, dtype=float)
        return np.repeat(start[None, :], max(2, int(n_samples)), axis=0), zero, zero, 0.0, 0.0
    chord_direction = chord / length
    start_direction = _oriented_unit_vector(start_normal, chord_direction)
    end_direction = _oriented_unit_vector(end_normal, chord_direction)
    valid = np.isfinite(parameters)
    centers = centers[valid]
    parameters = np.clip(parameters[valid], 0.0, 1.0)

    def curve_for(tangent_length):
        return sample_tangent_hermite(
            start,
            end,
            float(tangent_length) * start_direction,
            float(tangent_length) * end_direction,
            n_samples=n_samples,
        )

    def objective(tangent_length):
        curve = curve_for(tangent_length)
        if centers.shape[0]:
            predicted = np.asarray(
                [point_at_polyline_arclength(curve, value) for value in parameters]
            )
            data_error = float(np.mean(np.sum((predicted - centers) ** 2, axis=1)))
        else:
            data_error = 0.0
        steps = np.diff(curve, axis=0)
        forward = steps @ chord_direction
        backwards = np.minimum(forward, 0.0)
        fold_penalty = 100.0 * float(np.sum(backwards**2))
        return data_error + fold_penalty

    result = minimize_scalar(
        objective,
        bounds=(0.05 * length, float(max_tangent_length_fraction) * length),
        method="bounded",
        options={"xatol": 1e-5 * max(length, 1.0)},
    )
    tangent_length = float(result.x) if result.success else float(length)
    centerline = curve_for(tangent_length)
    if centers.shape[0]:
        projected = project_points_to_polyline(centers, centerline)
        fit_rmse = float(np.sqrt(np.mean(projected["distances"] ** 2)))
    else:
        fit_rmse = 0.0
    return (
        centerline,
        tangent_length * start_direction,
        tangent_length * end_direction,
        tangent_length,
        fit_rmse,
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
    n_samples=64,
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
    centerline, start_tangent, end_tangent, tangent_length, fit_rmse = (
        fit_tangent_constrained_hermite(
            attachment,
            tip,
            centers,
            levels,
            opening_normal,
            tip_normal,
            n_samples=n_samples,
        )
    )
    projected = project_points_to_polyline(centers, centerline)
    contour_s = np.asarray(projected["s"], dtype=float)
    order = np.argsort(contour_s)
    contour_s = contour_s[order]
    centers = centers[order]
    radii = np.asarray([item["radius"] for item in observations], dtype=float)[order]
    areas = np.asarray([item["area"] for item in observations], dtype=float)[order]
    perimeters = np.asarray([item["perimeter"] for item in observations], dtype=float)[order]
    area_integral = float(np.trapezoid(areas, contour_s)) if contour_s.size >= 2 else 0.0
    if area_integral > 1e-12:
        center_s = float(np.trapezoid(areas * contour_s, contour_s) / area_integral)
    else:
        center_s = float(np.average(contour_s, weights=np.maximum(areas, 1e-12)))
    # The distal radius landmark remains fixed at 0.85, so the area center must
    # precede it by only a numerical margin rather than an old biological clamp.
    center_s = float(np.clip(center_s, 1e-3, 0.85 - 1e-3))
    crypt_center = point_at_polyline_arclength(centerline, center_s)
    return CryptGeometryFit(
        centerline_points=centerline,
        centerline_kind="tangent_hermite",
        start_tangent=start_tangent,
        end_tangent=end_tangent,
        tangent_length=tangent_length,
        ratio_field=ratio,
        contour_s=contour_s,
        contour_centers=centers,
        contour_radii=radii,
        contour_areas=areas,
        contour_perimeters=perimeters,
        initial_center_s=center_s,
        initial_center=crypt_center,
        opening_normal=opening_normal,
        tip_normal=tip_normal,
        metadata={
            "method": "boundary_tip_ratio_contours_tangent_constrained_hermite",
            "centerline_fit_rmse": fit_rmse,
            "center_initialization": "mesh_contour_area_weighted_center",
            "cross_section_source": observation_source,
            "boundary_vertex_ids": boundary,
            "d_boundary": d_boundary,
            "d_tip": d_tip,
        },
    )


__all__ = [
    "CryptGeometryFit",
    "boundary_tip_ratio_field",
    "contour_observations",
    "fit_crypt_geometry",
    "fit_tangent_constrained_hermite",
    "restricted_surface_distance_field",
    "sample_tangent_hermite",
]

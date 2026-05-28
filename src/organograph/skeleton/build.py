"""Build biology-aware organoid skeletons from crypt detections.

This module converts crypt detection outputs into a compact straight-edge graph:
body center -> neck center -> optional bend/branch nodes -> crypt tips.  It is
not a medial-axis extractor.  It deliberately keeps segmentation adapters thin
so parameters can be tuned upstream and the resulting skeleton can be rebuilt.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from organograph.skeleton.datatypes import SkeletonGraph
from organograph.skeleton.geometry import (
    as_points,
    as_vertex_indices,
    centroid,
    estimate_bend_position,
    surface_area_centroid,
)


def _first_present(mapping: dict[str, Any], names: tuple[str, ...], default=None):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _json_safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    out = {}
    for key, value in metadata.items():
        if isinstance(value, set):
            out[key] = sorted(map(int, value))
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _mesh_like(vertices, faces):
    """Small mesh object for segmentation helpers that only need v, f, areas."""
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)

    def vertex_areas(from_mass_matrix: bool = False):
        tri = vertices[faces]
        face_areas = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
            axis=1,
        )
        areas = np.zeros(vertices.shape[0], dtype=float)
        for i in range(3):
            np.add.at(areas, faces[:, i], face_areas / 3.0)
        return areas

    return SimpleNamespace(v=vertices, f=faces, vertex_areas=vertex_areas)


def _low_pass_smoothed_mesh_for_detection(
    mesh,
    *,
    lmax: int = 5,
    recompute_eigen: bool = True,
    eigen_k: int | None = None,
):
    """Return a shallow mesh copy with low-pass reconstructed coordinates."""
    lmax = int(lmax)
    if lmax <= 0:
        raise ValueError("smooth_lmax must be positive")

    mesh.compute_spectral_coefficients(lmax=lmax)
    vertices_smoothed = mesh.reconstruct_from_coeffs(mesh.coeffs_v, lmax=lmax)

    smoothed = copy.copy(mesh)
    smoothed.v = np.asarray(vertices_smoothed, dtype=float)
    if recompute_eigen:
        V = int(smoothed.v.shape[0])
        if eigen_k is None:
            eigen_k = len(mesh.eigvals) if getattr(mesh, "eigvals", None) is not None else lmax**2
        eigen_k = max(2, min(int(eigen_k), max(V - 2, 2)))
        smoothed.laplacian = None
        smoothed.mass_matrix = None
        smoothed.eigvals = None
        smoothed.eigvecs = None
        smoothed.coeffs_v = None
        smoothed.lmax = None
        smoothed._eig_decomp(k=eigen_k)
    return smoothed


def _mesh_edges_from_faces(faces) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def _weighted_vertex_adjacency_from_edges(
    vertices,
    n_vertices: int,
    edges: np.ndarray,
) -> list[dict[int, float]]:
    vertices = as_points(vertices)
    adjacency = [dict() for _ in range(int(n_vertices))]
    for a, b in np.asarray(edges, dtype=np.int64):
        a = int(a)
        b = int(b)
        length = float(np.linalg.norm(vertices[b] - vertices[a]))
        adjacency[a][b] = length
        adjacency[b][a] = length
    return adjacency


def _boundary_edges_for_region(edges: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    inside_a = region_mask[edges[:, 0]]
    inside_b = region_mask[edges[:, 1]]
    return edges[inside_a != inside_b]


def _boundary_length_and_center(vertices, boundary_edges) -> tuple[float, np.ndarray | None]:
    vertices = as_points(vertices)
    boundary_edges = np.asarray(boundary_edges, dtype=np.int64)
    if boundary_edges.size == 0:
        return 0.0, None
    p0 = vertices[boundary_edges[:, 0]]
    p1 = vertices[boundary_edges[:, 1]]
    lengths = np.linalg.norm(p1 - p0, axis=1)
    total = float(np.sum(lengths))
    if total <= 1e-12:
        return total, centroid(0.5 * (p0 + p1))
    midpoints = 0.5 * (p0 + p1)
    center = np.sum(midpoints * lengths[:, None], axis=0) / total
    return total, center


def _frontier_for_region(adjacency: list[dict[int, float]], region: set[int], region_mask: np.ndarray) -> set[int]:
    frontier = set()
    for v in region:
        frontier.update(n for n in adjacency[v] if not region_mask[n])
    return frontier


def _best_frontier_addition(
    adjacency: list[dict[int, float]],
    region_mask: np.ndarray,
    frontier: set[int],
    current_length: float,
) -> tuple[int | None, float | None, float | None]:
    best_vertex = None
    best_length = None
    best_delta = None
    for candidate in frontier:
        delta = 0.0
        for neighbor, edge_length in adjacency[candidate].items():
            delta += -edge_length if region_mask[neighbor] else edge_length
        predicted = float(current_length) + delta
        if (
            best_length is None
            or predicted < best_length
            or (np.isclose(predicted, best_length) and delta < best_delta)
        ):
            best_vertex = int(candidate)
            best_length = float(predicted)
            best_delta = float(delta)
    return best_vertex, best_length, best_delta


def _add_vertex_to_region(
    vertex: int,
    adjacency: list[dict[int, float]],
    region: set[int],
    region_mask: np.ndarray,
    frontier: set[int],
) -> None:
    vertex = int(vertex)
    region.add(vertex)
    region_mask[vertex] = True
    frontier.discard(vertex)
    frontier.update(n for n in adjacency[vertex] if not region_mask[n])
    frontier.difference_update(region)


def _hks_column_at_time(hks, ts_mesh, target_time: float) -> tuple[np.ndarray | None, int | None, float | None]:
    if hks is None:
        return None, None, None
    hks = np.asarray(hks, dtype=float)
    if hks.ndim != 2 or hks.shape[1] == 0:
        return None, None, None
    if ts_mesh is None:
        idx = int(np.argmin(np.abs(np.arange(hks.shape[1], dtype=float) - float(target_time))))
        return hks[:, idx], idx, None
    ts_mesh = np.asarray(ts_mesh, dtype=float).reshape(-1)
    if ts_mesh.size != hks.shape[1]:
        return None, None, None
    idx = int(np.nanargmin(np.abs(ts_mesh - float(target_time))))
    return hks[:, idx], idx, float(ts_mesh[idx])


def _select_hks_tips_from_axis(
    vertices,
    patches,
    dnorm_all,
    hks,
    ts_mesh,
    fallback_bottoms,
    *,
    hks_time: float = 1.0,
    bottom_fraction: float = 0.5,
    min_hks_percent_increase: float = 0.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select final skeleton tips by max HKS in the bottom part of a refined axis."""
    vertices = as_points(vertices)
    dnorm_all = np.asarray(dnorm_all, dtype=float)
    fallback_bottoms = np.asarray(fallback_bottoms, dtype=np.int64)
    hks_values, hks_time_index, hks_time_actual = _hks_column_at_time(hks, ts_mesh, hks_time)
    tips = fallback_bottoms.copy()
    info: list[dict[str, Any]] = []
    frac = float(bottom_fraction)
    if not (0.0 < frac <= 1.0):
        frac = 0.5
    min_increase = float(min_hks_percent_increase)
    if not np.isfinite(min_increase) or min_increase < 0.0:
        min_increase = 0.0

    for i, patch in enumerate(patches):
        patch_idx = _coerce_patch(patch)
        fallback = int(fallback_bottoms[i]) if i < fallback_bottoms.size else -1
        details = {
            "strategy": "hks_after_neck_refinement",
            "boundary_distance_bottom_vertex_id": fallback,
            "hks_time_requested": float(hks_time),
            "hks_time_actual": hks_time_actual,
            "hks_time_index": hks_time_index,
            "bottom_fraction": frac,
            "min_hks_percent_increase": min_increase,
            "n_patch_vertices": int(patch_idx.size),
            "n_candidate_vertices": 0,
            "initial_hks": None,
            "selected_hks": None,
            "hks_percent_increase": None,
            "update_accepted": False,
            "fallback": None,
        }
        if patch_idx.size == 0 or i >= dnorm_all.shape[0]:
            details["fallback"] = "empty_patch_or_axis"
            info.append(details)
            continue

        dnorm = dnorm_all[i]
        finite = patch_idx[np.isfinite(dnorm[patch_idx])]
        if finite.size == 0:
            details["fallback"] = "no_finite_axis_distances"
            info.append(details)
            continue

        n_keep = max(1, int(np.ceil(frac * finite.size)))
        order = np.argsort(dnorm[finite])
        candidates = finite[order[:n_keep]]
        details["n_candidate_vertices"] = int(candidates.size)

        if hks_values is None or hks_values.shape[0] != vertices.shape[0]:
            details["fallback"] = "missing_hks"
            info.append(details)
            continue

        candidate_hks = hks_values[candidates]
        finite_hks = np.isfinite(candidate_hks)
        if not np.any(finite_hks):
            details["fallback"] = "nonfinite_hks"
            info.append(details)
            continue

        valid_candidates = candidates[finite_hks]
        valid_hks = candidate_hks[finite_hks]
        tip = int(valid_candidates[int(np.argmax(valid_hks))])
        selected_hks = float(np.max(valid_hks))
        details["selected_hks"] = selected_hks
        details["dnorm_at_tip"] = float(dnorm[tip]) if np.isfinite(dnorm[tip]) else None

        if 0 <= fallback < hks_values.shape[0] and np.isfinite(hks_values[fallback]):
            initial_hks = float(hks_values[fallback])
            details["initial_hks"] = initial_hks
            if initial_hks != 0.0:
                percent_increase = 100.0 * (selected_hks - initial_hks) / abs(initial_hks)
            elif selected_hks > initial_hks:
                percent_increase = float("inf")
            else:
                percent_increase = 0.0
            details["hks_percent_increase"] = float(percent_increase)
            if percent_increase + 1e-12 < min_increase:
                details["bottom_vertex_id"] = fallback
                details["fallback"] = "hks_increase_below_threshold"
                info.append(details)
                continue

        tips[i] = tip
        details["bottom_vertex_id"] = tip
        details["update_accepted"] = True
        info.append(details)
    return tips, info


def _grow_parent_patch_to_neck(
    vertices,
    faces,
    patch_vertices,
    *,
    max_size_factor: float = 2.0,
    max_mesh_fraction: float = 0.35,
    smooth_perimeter: bool = True,
    smoothing_tolerance: float = 0.0,
    min_decrease_fraction: float = 0.0,
    min_prominence_fraction: float = 0.01,
    robust_window: int = 1,
) -> dict[str, Any]:
    """Grow a parent candidate patch and choose a robust boundary minimum.

    The boundary is first smoothed by adding frontier vertices that reduce the
    mesh cut length.  The patch is then grown one mesh-neighborhood ring at a
    time.  Growth continues past local minima until the allowed size limit is
    reached or the region can no longer grow; the neck is selected as a global
    boundary-length minimum with optional prominence/robustness checks.
    """
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    patch = _coerce_patch(patch_vertices)

    result = {
        "kept_as_split": False,
        "reason": "no_robust_boundary_minimum",
        "neck_position": None,
        "neck_region_vertices": [],
        "final_region_vertices": [],
        "smoothed_region_vertices": [],
        "raw_initial_size": int(patch.size),
        "initial_size": int(patch.size),
        "smoothed_initial_size": None,
        "neck_region_size": None,
        "final_region_size": None,
        "max_allowed_size": None,
        "max_mesh_fraction": float(max_mesh_fraction),
        "mesh_fraction_size_limit": None,
        "perimeter_smoothed": bool(smooth_perimeter),
        "perimeter_smoothing_added_vertices": [],
        "perimeter_smoothing_n_added": 0,
        "raw_initial_boundary_length": None,
        "smoothed_initial_boundary_length": None,
        "initial_boundary_length": None,
        "neck_boundary_length": None,
        "boundary_lengths": [],
        "region_sizes": [],
        "minimum_index": None,
        "minimum_boundary_length": None,
        "final_boundary_length": None,
        "min_decrease_fraction": float(min_decrease_fraction),
        "min_prominence_fraction": float(min_prominence_fraction),
        "robust_window": int(robust_window),
    }
    if patch.size == 0 or faces.size == 0:
        result["reason"] = "empty_parent_patch"
        return result

    edges = _mesh_edges_from_faces(faces)
    adjacency = _weighted_vertex_adjacency_from_edges(vertices, vertices.shape[0], edges)
    mesh_fraction = float(max_mesh_fraction)
    if not (np.isfinite(mesh_fraction) and mesh_fraction > 0):
        mesh_fraction = 1.0
    mesh_limit = int(np.floor(mesh_fraction * vertices.shape[0]))
    mesh_limit = max(1, min(mesh_limit, vertices.shape[0]))
    result["mesh_fraction_size_limit"] = mesh_limit
    factor_limit = int(np.floor(float(max_size_factor) * patch.size))
    factor_limit = max(factor_limit, int(patch.size))
    max_size = min(factor_limit, mesh_limit)
    result["max_allowed_size"] = max_size

    region = set(map(int, patch.tolist()))
    region_mask = np.zeros(vertices.shape[0], dtype=bool)
    region_mask[list(region)] = True
    boundary = _boundary_edges_for_region(edges, region_mask)
    current_length, _ = _boundary_length_and_center(vertices, boundary)
    result["raw_initial_boundary_length"] = current_length

    if len(region) > mesh_limit:
        result.update(
            {
                "reason": "initial_region_exceeds_mesh_fraction_limit",
                "initial_boundary_length": current_length,
                "smoothed_initial_boundary_length": current_length,
                "final_boundary_length": current_length,
                "final_region_vertices": sorted(region),
                "final_region_size": int(len(region)),
                "smoothed_region_vertices": sorted(region),
                "smoothed_initial_size": int(len(region)),
                "boundary_lengths": [current_length],
                "region_sizes": [int(len(region))],
            }
        )
        return result

    frontier = _frontier_for_region(adjacency, region, region_mask)
    smoothing_added = []
    if smooth_perimeter:
        while len(region) < max_size and frontier:
            best_vertex, best_length, best_delta = _best_frontier_addition(
                adjacency,
                region_mask,
                frontier,
                current_length,
            )
            if best_vertex is None or best_delta is None:
                break
            decrease_needed = max(float(smoothing_tolerance), 0.0)
            if best_delta >= -decrease_needed:
                break
            _add_vertex_to_region(best_vertex, adjacency, region, region_mask, frontier)
            current_length = float(best_length)
            smoothing_added.append(best_vertex)

    result["perimeter_smoothing_added_vertices"] = smoothing_added
    result["perimeter_smoothing_n_added"] = int(len(smoothing_added))
    result["smoothed_region_vertices"] = sorted(region)
    result["smoothed_initial_size"] = int(len(region))
    result["smoothed_initial_boundary_length"] = current_length
    result["initial_size"] = int(len(region))
    result["initial_boundary_length"] = current_length
    result["boundary_lengths"].append(current_length)
    result["region_sizes"].append(len(region))

    regions = [set(region)]
    stop_reason = "region_cannot_grow"

    while len(region) < max_size:
        grown = set(region)
        for v in region:
            grown.update(adjacency[v])
        if len(grown) == len(region):
            stop_reason = "region_cannot_grow"
            break
        if len(grown) > max_size:
            stop_reason = "reached_size_limit"
            break
        region = grown
        region_mask[:] = False
        region_mask[list(region)] = True
        boundary = _boundary_edges_for_region(edges, region_mask)
        current_length, _ = _boundary_length_and_center(vertices, boundary)
        result["boundary_lengths"].append(current_length)
        result["region_sizes"].append(len(region))
        regions.append(set(region))
    else:
        stop_reason = "reached_size_limit"

    lengths = np.asarray(result["boundary_lengths"], dtype=float)
    sizes = np.asarray(result["region_sizes"], dtype=int)
    result["final_boundary_length"] = float(lengths[-1]) if lengths.size else None
    result["final_region_vertices"] = sorted(regions[-1]) if regions else []
    result["final_region_size"] = int(len(regions[-1])) if regions else None
    if lengths.size == 0:
        result["reason"] = "empty_boundary_trace"
        return result

    min_index = int(np.argmin(lengths))
    min_length = float(lengths[min_index])
    result["minimum_index"] = min_index
    result["minimum_boundary_length"] = min_length

    required_drop = float(result["initial_boundary_length"]) * float(min_decrease_fraction)
    required_prominence = float(result["initial_boundary_length"]) * float(min_prominence_fraction)
    robust_window = max(1, int(robust_window))

    has_pre_min_growth = min_index > 0
    has_post_min_growth = min_index + robust_window < lengths.size
    before_size_limit = int(sizes[min_index]) < max_size
    initial_drop = float(result["initial_boundary_length"]) - min_length
    post_rise = float(np.max(lengths[min_index + 1 :]) - min_length) if min_index + 1 < lengths.size else 0.0

    result.update(
        {
            "stop_reason": stop_reason,
            "initial_drop": initial_drop,
            "post_minimum_rise": post_rise,
            "required_drop": required_drop,
            "required_prominence": required_prominence,
        }
    )

    if not has_pre_min_growth:
        result["reason"] = "minimum_at_initial_boundary"
        return result
    if not has_post_min_growth:
        result["reason"] = "minimum_not_observed_before_growth_end"
        return result
    if not before_size_limit:
        result["reason"] = "minimum_at_size_limit"
        return result
    if initial_drop < required_drop:
        result["reason"] = "minimum_not_deep_enough"
        return result
    if post_rise < required_prominence:
        result["reason"] = "minimum_not_prominent_enough"
        return result

    best_region = regions[min_index]
    best_mask = np.zeros(vertices.shape[0], dtype=bool)
    best_mask[list(best_region)] = True
    best_boundary = _boundary_edges_for_region(edges, best_mask)
    best_length, best_center = _boundary_length_and_center(vertices, best_boundary)

    if best_center is not None:
        result.update(
            {
                "kept_as_split": True,
                "reason": "boundary_minimum_found",
                "neck_position": np.asarray(best_center, dtype=float),
                "neck_region_vertices": sorted(best_region),
                "neck_region_size": int(len(best_region)),
                "neck_boundary_length": float(best_length),
            }
        )
    return result


def _coerce_patch(patch) -> np.ndarray:
    if patch is None:
        return np.empty(0, dtype=np.int64)
    return as_vertex_indices(patch)


def _point_from_vertex(vertices, vertex_id) -> np.ndarray | None:
    if vertex_id is None:
        return None
    vertex_id = int(vertex_id)
    if vertex_id < 0:
        return None
    return as_points(vertices)[vertex_id]


def _point_from_keys(vertices, detection: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    value = _first_present(detection, keys)
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape == (3,):
        return arr
    if arr.ndim == 0:
        return _point_from_vertex(vertices, int(arr))
    return None


def _centroid_from_vertex_keys(
    vertices,
    detection: dict[str, Any],
    keys: tuple[str, ...],
) -> np.ndarray | None:
    value = _first_present(detection, keys)
    if value is None:
        return None
    idx = _coerce_patch(value)
    if idx.size == 0:
        return None
    return centroid(as_points(vertices)[idx])


def _boundary_vertices_from_patch(faces, patch_vertices) -> np.ndarray:
    patch_vertices = _coerce_patch(patch_vertices)
    if patch_vertices.size == 0:
        return patch_vertices
    faces = np.asarray(faces, dtype=np.int64)
    keep = np.isin(faces, patch_vertices).all(axis=1)
    patch_faces = faces[keep]
    if patch_faces.size == 0:
        return patch_vertices
    edges = np.vstack(
        [
            patch_faces[:, [0, 1]],
            patch_faces[:, [1, 2]],
            patch_faces[:, [2, 0]],
        ]
    )
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if boundary_edges.size == 0:
        return patch_vertices
    return np.unique(boundary_edges.reshape(-1))


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


def _tip_position(vertices, detection: dict[str, Any]) -> np.ndarray:
    explicit = _point_from_keys(
        vertices,
        detection,
        ("tip_position", "tip_center", "tip", "bottom_position", "crypt_tip", "p_tip"),
    )
    if explicit is not None:
        return explicit

    vertex_id = _first_present(
        detection,
        ("tip_vertex_id", "bottom_vertex_id", "bottom", "bottom_vertex"),
    )
    by_vertex = _point_from_vertex(vertices, vertex_id)
    if by_vertex is not None:
        return by_vertex

    patch = _coerce_patch(
        _first_present(
            detection,
            ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
        )
    )
    if patch.size:
        return centroid(as_points(vertices)[patch])
    raise ValueError("Crypt detection is missing a tip/bottom position or vertex id.")


def _crypt_position(vertices, detection: dict[str, Any]) -> np.ndarray:
    explicit = _point_from_keys(
        vertices,
        detection,
        ("crypt_position", "crypt_center", "crypt_centroid", "p_crypt"),
    )
    if explicit is not None:
        return explicit

    patch = _coerce_patch(
        _first_present(
            detection,
            ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
        )
    )
    if patch.size:
        return centroid(as_points(vertices)[patch])
    return _tip_position(vertices, detection)


def _add_crypt_tip_path(
    graph: SkeletonGraph,
    vertices,
    *,
    path_prefix: str,
    source_id: str,
    tip_id: str,
    crypt_id,
    detection: dict[str, Any],
    crypt_vertices: np.ndarray,
    source_position: np.ndarray,
    tip_position: np.ndarray,
    metadata: dict[str, Any],
    bend_strategy: str,
    crypt_role: str = "crypt_centroid",
) -> None:
    """Connect a crypt neck-like source to a tip, optionally through a waypoint."""
    strategy = str(bend_strategy).lower()
    intermediate_position = None
    intermediate_type = None
    intermediate_role = None

    if strategy != "none":
        if strategy == "crypt_centroid":
            intermediate_position = _crypt_position(vertices, detection)
            intermediate_type = "crypt"
            intermediate_role = crypt_role
        else:
            intermediate_position = _point_from_keys(
                vertices,
                detection,
                ("bend_position", "bend_center", "bend", "p_bend"),
            )
            if intermediate_position is None:
                intermediate_position = estimate_bend_position(
                    vertices,
                    crypt_vertices,
                    source_position,
                    tip_position,
                    strategy=strategy,
                )
            if intermediate_position is not None:
                intermediate_type = "bend"
                intermediate_role = "bend"

    if intermediate_position is None:
        graph.add_edge(
            f"{path_prefix}_neck_to_tip",
            source_id,
            tip_id,
            edge_type="neck_to_tip",
            crypt_id=crypt_id,
        )
        return

    intermediate_id = f"{path_prefix}_{intermediate_type}"
    graph.add_node(
        intermediate_id,
        intermediate_type,
        intermediate_position,
        crypt_id=crypt_id,
        metadata={
            **metadata,
            "role": intermediate_role,
            "bend_strategy": strategy,
        },
    )
    graph.add_edge(
        f"{path_prefix}_neck_to_{intermediate_type}",
        source_id,
        intermediate_id,
        edge_type=f"neck_to_{intermediate_type}",
        crypt_id=crypt_id,
    )
    graph.add_edge(
        f"{path_prefix}_{intermediate_type}_to_tip",
        intermediate_id,
        tip_id,
        edge_type=f"{intermediate_type}_to_tip",
        crypt_id=crypt_id,
    )


def _branch_position(vertices, detection: dict[str, Any], neck, daughter_tips) -> np.ndarray:
    explicit = _point_from_keys(
        vertices,
        detection,
        ("branch_position", "branch_center", "branch", "split_position", "split_center"),
    )
    if explicit is not None:
        return explicit

    vertex_id = _first_present(detection, ("branch_vertex_id", "split_vertex_id"))
    by_vertex = _point_from_vertex(vertices, vertex_id)
    if by_vertex is not None:
        return by_vertex

    stem_vertices = _coerce_patch(_first_present(detection, ("stem_vertices", "trunk_vertices")))
    if stem_vertices.size:
        return centroid(as_points(vertices)[stem_vertices])

    daughter_mean = centroid(np.vstack(daughter_tips))
    return 0.5 * (np.asarray(neck, dtype=float) + daughter_mean)


def _crypt_side_region(detection: dict[str, Any]) -> np.ndarray:
    """Vertices on the crypt side of the root neck boundary."""
    region = _coerce_patch(
        _first_present(
            detection,
            ("neck_region_vertices", "neck_side_vertices", "root_region_vertices"),
        )
    )
    if region.size:
        return region

    dfield = _first_present(detection, ("d_crypt", "distance_field", "dnorm", "dnorm_vertices"))
    if dfield is not None:
        dfield = np.asarray(dfield, dtype=float).reshape(-1)
        level = float(detection.get("neck_level", 1.0))
        region = np.where(np.isfinite(dfield) & (dfield <= level))[0].astype(np.int64)
        if region.size:
            return region

    return _coerce_patch(
        _first_present(
            detection,
            ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
        )
    )


def _body_center_from_root_regions(vertices, detections: list[dict[str, Any]]) -> np.ndarray | None:
    vertices = as_points(vertices)
    excluded: set[int] = set()
    for detection in detections:
        region = _crypt_side_region(detection)
        if region.size:
            excluded.update(int(v) for v in region)

    if not excluded:
        return None
    body_vertices = np.setdiff1d(np.arange(vertices.shape[0], dtype=np.int64), np.fromiter(excluded, dtype=np.int64))
    if body_vertices.size == 0:
        return None
    return centroid(vertices[body_vertices])


def _branch_region_vertices(detection: dict[str, Any], daughters: list[dict[str, Any]]) -> np.ndarray:
    parent_region = _crypt_side_region(detection)
    if parent_region.size == 0:
        parent_region = _coerce_patch(_first_present(detection, ("stem_vertices", "trunk_vertices")))
    if parent_region.size == 0:
        return parent_region

    remove: set[int] = set()
    for daughter in daughters:
        daughter_region = _crypt_side_region(daughter)
        if daughter_region.size:
            remove.update(int(v) for v in daughter_region)
    if not remove:
        stem = _coerce_patch(_first_present(detection, ("stem_vertices", "trunk_vertices")))
        return stem if stem.size else parent_region

    branch_region = np.asarray([int(v) for v in parent_region if int(v) not in remove], dtype=np.int64)
    if branch_region.size:
        return branch_region
    stem = _coerce_patch(_first_present(detection, ("stem_vertices", "trunk_vertices")))
    return stem


def _branch_position_from_regions(vertices, detection: dict[str, Any], daughters: list[dict[str, Any]]) -> tuple[np.ndarray | None, np.ndarray]:
    branch_region = _branch_region_vertices(detection, daughters)
    if branch_region.size == 0:
        return None, branch_region
    return centroid(as_points(vertices)[branch_region]), branch_region


def _daughter_detections(detection: dict[str, Any]) -> list[dict[str, Any]]:
    daughters = _first_present(detection, ("daughters", "daughter_tips", "branches", "children"))
    if daughters is None:
        return []
    out = []
    for daughter in daughters:
        if isinstance(daughter, dict):
            out.append(dict(daughter))
        else:
            arr = np.asarray(daughter)
            if arr.shape == (3,):
                out.append({"tip_position": arr})
            else:
                out.append({"tip_vertex_id": int(arr)})
    return out


def normalize_crypt_detections(crypt_detections) -> list[dict[str, Any]]:
    """Normalize common segmentation outputs to a list of detection dicts.

    Accepted inputs include:
    - list of dicts with explicit neck/tip fields;
    - list of vertex-index patches;
    - segmentation dictionaries containing `crypts_mesh`, `crypts_ll`, or
      `crypts`, optionally with per-crypt arrays such as `bottom_vertex_ids`
      and `d_crypts`.
    """
    if crypt_detections is None:
        return []

    if isinstance(crypt_detections, dict):
        patches = _first_present(
            crypt_detections,
            ("crypt_detections", "crypts_mesh", "crypts_ll", "crypts", "patches"),
        )
        if patches is not None and not isinstance(patches, dict):
            if all(isinstance(patch, dict) for patch in patches):
                return [dict(patch, crypt_id=patch.get("crypt_id", i)) for i, patch in enumerate(patches)]
            detections = []
            for i, patch in enumerate(patches):
                det = {"crypt_id": i, "crypt_vertices": patch}
                for src_key, dst_key in (
                    ("bottom_vertex_ids", "bottom_vertex_id"),
                    ("tip_vertex_ids", "tip_vertex_id"),
                    ("d_crypts", "d_crypt"),
                    ("L_crypts", "L_crypt"),
                    ("circumference_crypts", "circumference"),
                    ("crypt_constrictions", "constriction"),
                    ("crypt_elongations", "elongation"),
                ):
                    if src_key in crypt_detections:
                        values = crypt_detections[src_key]
                        if len(values) > i:
                            det[dst_key] = values[i]
                detections.append(det)
            return detections
        return [dict(crypt_detections)]

    detections = []
    for i, item in enumerate(crypt_detections):
        if isinstance(item, dict):
            det = dict(item)
            det.setdefault("crypt_id", i)
        else:
            det = {"crypt_id": i, "crypt_vertices": item}
        detections.append(det)
    return detections


def _body_center(vertices, faces, body_vertices, body_faces, body_center) -> np.ndarray:
    if body_center is not None:
        center = np.asarray(body_center, dtype=float)
        if center.shape != (3,):
            raise ValueError("body_center must be a 3-vector")
        return center
    vertices = as_points(vertices)
    if body_vertices is not None:
        idx = _coerce_patch(body_vertices)
        if idx.size:
            return centroid(vertices[idx])
    if body_faces is not None:
        return surface_area_centroid(vertices, np.asarray(body_faces, dtype=np.int64))
    if faces is not None:
        return surface_area_centroid(vertices, faces)
    return centroid(vertices)


def build_skeleton_from_crypt_detections(
    vertices,
    faces,
    crypt_detections,
    body_vertices=None,
    body_faces=None,
    body_center=None,
    bend_strategy: str = "none",
    refine_body_center_from_necks: bool = True,
    refine_branch_centers_from_necks: bool = True,
    metadata: dict[str, Any] | None = None,
) -> SkeletonGraph:
    """Build a straight-edge organoid skeleton from crypt detections.

    Each non-split crypt is represented as `body -> neck -> tip` by default.
    When an intermediate waypoint is requested through ``bend_strategy``, the
    crypt path becomes either `neck -> bend -> tip` or, for
    ``bend_strategy="crypt_centroid"``, `neck -> crypt -> tip`.  Split
    detections with daughters use the same daughter-neck to daughter-tip rule.

    When enabled, body and branch node positions are refined from mesh regions:
    root necks bound crypt-side regions that are excluded from the villus body,
    and split branches are placed at the centroid of the parent region after
    subtracting daughter crypt-side regions.
    """
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    detections = normalize_crypt_detections(crypt_detections)

    graph = SkeletonGraph(
        metadata=_json_safe_metadata(metadata),
        coordinate_frame={
            "kind": "raw",
            "body_center_node": "body",
            "description": "Raw mesh/world coordinates; edges are straight segments.",
        },
    )
    body_position = _body_center(vertices, faces, body_vertices, body_faces, body_center)
    body_refined = False
    if body_center is None and body_vertices is None and body_faces is None and refine_body_center_from_necks:
        refined_body = _body_center_from_root_regions(vertices, detections)
        if refined_body is not None:
            body_position = refined_body
            body_refined = True

    graph.add_node(
        "body",
        "body",
        body_position,
        metadata={
            "role": "villus_body_center",
            "center_refined_from_neck_regions": body_refined,
        },
    )

    for i, detection in enumerate(detections):
        crypt_id = detection.get("crypt_id", i)
        crypt_prefix = f"crypt_{crypt_id}"
        crypt_vertices = _coerce_patch(
            _first_present(
                detection,
                ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
            )
        )
        common_meta = _json_safe_metadata(
            {
                "source_detection_index": i,
                "n_crypt_vertices": int(crypt_vertices.size),
                **dict(detection.get("metadata", {})),
            }
        )

        neck = _neck_position(vertices, faces, detection)
        neck_id = f"{crypt_prefix}_neck"
        graph.add_node(
            neck_id,
            "neck",
            neck,
            crypt_id=crypt_id,
            metadata=common_meta,
        )
        graph.add_edge(
            f"{crypt_prefix}_body_to_neck",
            "body",
            neck_id,
            edge_type="body_to_neck",
            crypt_id=crypt_id,
        )

        daughters = _daughter_detections(detection)
        if daughters:
            daughter_tips = [_tip_position(vertices, daughter) for daughter in daughters]
            branch_region = np.empty(0, dtype=np.int64)
            branch = None
            if refine_branch_centers_from_necks:
                branch, branch_region = _branch_position_from_regions(vertices, detection, daughters)
            if branch is None:
                branch = _branch_position(vertices, detection, neck, daughter_tips)
            branch_id = f"{crypt_prefix}_branch"
            graph.add_node(
                branch_id,
                "branch",
                branch,
                crypt_id=crypt_id,
                metadata={
                    **common_meta,
                    "n_daughters": len(daughters),
                    "center_refined_from_neck_regions": bool(branch_region.size),
                    "n_branch_region_vertices": int(branch_region.size),
                },
            )
            graph.add_edge(
                f"{crypt_prefix}_neck_to_branch",
                neck_id,
                branch_id,
                edge_type="neck_to_branch",
                crypt_id=crypt_id,
            )
            for j, daughter in enumerate(daughters):
                daughter_meta = _json_safe_metadata(
                    {
                        **common_meta,
                        "daughter_index": j,
                        **dict(daughter.get("metadata", {})),
                    }
                )
                daughter_neck = _neck_position(vertices, faces, daughter)
                daughter_neck_id = f"{crypt_prefix}_daughter_{j}_neck"
                graph.add_node(
                    daughter_neck_id,
                    "neck",
                    daughter_neck,
                    crypt_id=crypt_id,
                    metadata={
                        **daughter_meta,
                        "role": "daughter_neck",
                    },
                )
                graph.add_edge(
                    f"{crypt_prefix}_branch_to_daughter_{j}_neck",
                    branch_id,
                    daughter_neck_id,
                    edge_type="branch_to_neck",
                    crypt_id=crypt_id,
                )

                tip_id = f"{crypt_prefix}_tip_{j}"
                graph.add_node(
                    tip_id,
                    "tip",
                    daughter_tips[j],
                    crypt_id=crypt_id,
                    metadata=daughter_meta,
                )
                daughter_vertices = _coerce_patch(
                    _first_present(
                        daughter,
                        ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
                    )
                )
                _add_crypt_tip_path(
                    graph,
                    vertices,
                    path_prefix=f"{crypt_prefix}_daughter_{j}",
                    source_id=daughter_neck_id,
                    tip_id=tip_id,
                    crypt_id=crypt_id,
                    detection=daughter,
                    crypt_vertices=daughter_vertices,
                    source_position=daughter_neck,
                    tip_position=daughter_tips[j],
                    metadata=daughter_meta,
                    bend_strategy=bend_strategy,
                    crypt_role="daughter_crypt_centroid",
                )
            continue

        tip = _tip_position(vertices, detection)
        tip_id = f"{crypt_prefix}_tip"
        graph.add_node(
            tip_id,
            "tip",
            tip,
            crypt_id=crypt_id,
            metadata=common_meta,
        )
        _add_crypt_tip_path(
            graph,
            vertices,
            path_prefix=crypt_prefix,
            source_id=neck_id,
            tip_id=tip_id,
            crypt_id=crypt_id,
            detection=detection,
            crypt_vertices=crypt_vertices,
            source_position=neck,
            tip_position=tip,
            metadata=common_meta,
            bend_strategy=bend_strategy,
        )

    return graph


def detect_crypts_for_skeleton(
    mesh,
    vocab,
    *,
    geodesic_fn,
    L_ref=None,
    crypt_vocab_idx=None,
    threshold=0.5,
    filter_fn_list=None,
    refine_crypts=True,
    refine_threshold=0.0,
    refine_only_if_area_at_least=5.0,
    min_refined_frac_of_parent=0.1,
    geodesic_kwargs=None,
    final_tip_hks_time: float = 1.0,
    final_tip_bottom_fraction: float = 0.5,
    final_tip_min_hks_percent_increase: float = 0.0,
    extend_max=2.0,
    disc_resolution=200,
    neck_search_interval=(0.8, 2.0),
    validate_split_stems: bool = True,
    split_growth_max_size_factor: float = 2.0,
    split_growth_max_mesh_fraction: float = 0.35,
    split_growth_smooth_perimeter: bool = True,
    split_growth_smoothing_tolerance: float = 0.0,
    split_growth_min_decrease_fraction: float = 0.0,
    split_growth_min_prominence_fraction: float = 0.01,
    split_growth_robust_window: int = 1,
    smooth_mesh: bool = False,
    smooth_lmax: int = 5,
    smooth_recompute_eigen: bool = True,
    smooth_eigen_k: int | None = None,
    return_intermediates=False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run crypt detection pieces needed for skeleton construction.

    This adapter intentionally starts from a fresh HKS candidate screen.  Parent
    candidate patches are kept as skeleton crypt trunks; if local refinement
    splits a parent into multiple child patches, the output marks that parent as
    a split crypt with daughter tips.  This preserves stem/neck regions that may
    later be grouped with the villus in final saved segmentations.

    The geodesic axis and neckline are computed from the original
    boundary-distance bottom.  The skeleton tip is then updated to the max-HKS
    vertex near ``final_tip_hks_time`` in the bottom fraction of the refined
    crypt axis, provided the HKS increase over the initial tip clears
    ``final_tip_min_hks_percent_increase``.
    """
    from organograph.crypts.axis import compute_crypt_axis, normalize_crypt_axis_to_neckline
    from organograph.crypts.filters import apply_filters
    from organograph.crypts.vocab import detect_crypts_by_encoding, subdivide_crypts_by_encoding

    if geodesic_kwargs is None:
        geodesic_kwargs = {}

    detection_mesh = (
        _low_pass_smoothed_mesh_for_detection(
            mesh,
            lmax=smooth_lmax,
            recompute_eigen=smooth_recompute_eigen,
            eigen_k=smooth_eigen_k,
        )
        if smooth_mesh
        else mesh
    )

    parents, enc_vars = detect_crypts_by_encoding(
        vocab,
        detection_mesh,
        L_ref=L_ref,
        crypt_vocab_idx=crypt_vocab_idx,
        threshold=threshold,
        return_intermediates=True,
    )

    seg_vars = {
        "encoding": enc_vars.get("encoding"),
        "ts_mesh": enc_vars.get("ts_mesh"),
        "ts_vocab": enc_vars.get("ts_vocab"),
        "hks": enc_vars.get("hks"),
        "norm_hks": enc_vars.get("norm_hks"),
        "hks_segment": enc_vars.get("hks"),
        "normalised_hks_segment": enc_vars.get("norm_hks"),
        "vertex_areas": np.asarray(detection_mesh.vertex_areas(), float),
    }
    if filter_fn_list is not None:
        parents, filter_info, keep_idx = apply_filters(
            parents,
            filters=filter_fn_list,
            mesh=detection_mesh,
            seg_vars=seg_vars,
        )
        seg_vars["filter_info_initial"] = filter_info
        seg_vars["keep_idx_initial"] = keep_idx

    dnorm_parent, L_parent, bottom_parent = compute_crypt_axis(
        detection_mesh,
        parents,
        geodesic_fn,
        geodesic_kwargs=geodesic_kwargs,
    )
    bottom_parent = np.asarray(bottom_parent, dtype=np.int64)
    bottom_info_parent = [
        {
            "strategy": "boundary_distance",
            "bottom_vertex_id": int(bottom),
            "n_patch_vertices": int(len(patch)),
        }
        for bottom, patch in zip(bottom_parent, parents)
    ]
    d_levels = np.linspace(0.01, float(extend_max), int(disc_resolution))
    _, dnorm_parent, L_parent = normalize_crypt_axis_to_neckline(
        detection_mesh,
        dnorm_parent,
        d_levels,
        search_interval=neck_search_interval,
        L_crypt=L_parent,
    )
    final_bottom_parent, final_bottom_info_parent = _select_hks_tips_from_axis(
        detection_mesh.v,
        parents,
        dnorm_parent,
        seg_vars.get("hks"),
        seg_vars.get("ts_mesh"),
        bottom_parent,
        hks_time=final_tip_hks_time,
        bottom_fraction=final_tip_bottom_fraction,
        min_hks_percent_increase=final_tip_min_hks_percent_increase,
    )

    detections = []
    refined_by_parent = []
    split_validations = []
    for i, parent in enumerate(parents):
        daughters = []
        if refine_crypts:
            refined = subdivide_crypts_by_encoding(
                vocab,
                detection_mesh,
                L_ref=L_ref,
                crypt_vocab_idx=crypt_vocab_idx,
                patches=[parent],
                threshold=refine_threshold,
                refine_only_if_area_at_least=refine_only_if_area_at_least,
                min_refined_frac_of_parent=min_refined_frac_of_parent,
            )
        else:
            refined = [parent]
        refined_by_parent.append(refined)

        if len(refined) > 1:
            dnorm_child, L_child, bottom_child = compute_crypt_axis(
                detection_mesh,
                refined,
                geodesic_fn,
                geodesic_kwargs=geodesic_kwargs,
            )
            bottom_child = np.asarray(bottom_child, dtype=np.int64)
            bottom_info_child = [
                {
                    "strategy": "boundary_distance",
                    "bottom_vertex_id": int(bottom),
                    "n_patch_vertices": int(len(patch)),
                }
                for bottom, patch in zip(bottom_child, refined)
            ]
            _, dnorm_child, L_child = normalize_crypt_axis_to_neckline(
                detection_mesh,
                dnorm_child,
                d_levels,
                search_interval=neck_search_interval,
                L_crypt=L_child,
            )
            final_bottom_child, final_bottom_info_child = _select_hks_tips_from_axis(
                detection_mesh.v,
                refined,
                dnorm_child,
                seg_vars.get("hks"),
                seg_vars.get("ts_mesh"),
                bottom_child,
                hks_time=final_tip_hks_time,
                bottom_fraction=final_tip_bottom_fraction,
                min_hks_percent_increase=final_tip_min_hks_percent_increase,
            )
            daughter_union = set().union(*[set(child) for child in refined])
            stem_vertices = sorted(set(parent).difference(daughter_union))
            for j, child in enumerate(refined):
                daughters.append(
                    {
                        "crypt_id": f"{i}.{j}",
                        "crypt_vertices": child,
                        "boundary_distance_bottom_vertex_id": int(bottom_child[j]),
                        "bottom_vertex_id": int(final_bottom_child[j]),
                        "d_crypt": dnorm_child[j],
                        "L_crypt": float(L_child[j]),
                        "metadata": {
                            "boundary_distance_bottom_selection": bottom_info_child[j],
                            "final_tip_selection": final_bottom_info_child[j],
                        },
                    }
                )
        else:
            stem_vertices = []

        split_validation = {
            "kept_as_split": bool(len(daughters) > 0),
            "reason": "not_refined_split" if not daughters else "split_validation_disabled",
            "neck_position": None,
            "neck_region_vertices": [],
            "final_region_vertices": [],
            "smoothed_region_vertices": [],
            "raw_initial_size": int(len(parent)) if parent is not None else 0,
            "neck_region_size": None,
            "final_region_size": None,
            "smoothed_initial_size": None,
            "initial_size": int(len(parent)) if parent is not None else 0,
            "max_allowed_size": None,
            "max_mesh_fraction": float(split_growth_max_mesh_fraction),
            "mesh_fraction_size_limit": None,
            "perimeter_smoothed": bool(split_growth_smooth_perimeter),
            "perimeter_smoothing_added_vertices": [],
            "perimeter_smoothing_n_added": 0,
            "raw_initial_boundary_length": None,
            "smoothed_initial_boundary_length": None,
            "initial_boundary_length": None,
            "neck_boundary_length": None,
            "boundary_lengths": [],
            "region_sizes": [],
            "minimum_index": None,
            "minimum_boundary_length": None,
            "final_boundary_length": None,
            "min_decrease_fraction": float(split_growth_min_decrease_fraction),
            "min_prominence_fraction": float(split_growth_min_prominence_fraction),
            "robust_window": int(split_growth_robust_window),
        }
        if daughters and validate_split_stems:
            split_validation = _grow_parent_patch_to_neck(
                detection_mesh.v,
                detection_mesh.f,
                parent,
                max_size_factor=split_growth_max_size_factor,
                max_mesh_fraction=split_growth_max_mesh_fraction,
                smooth_perimeter=split_growth_smooth_perimeter,
                smoothing_tolerance=split_growth_smoothing_tolerance,
                min_decrease_fraction=split_growth_min_decrease_fraction,
                min_prominence_fraction=split_growth_min_prominence_fraction,
                robust_window=split_growth_robust_window,
            )
        split_validations.append(split_validation)

        if daughters and not split_validation.get("kept_as_split", False):
            for j, daughter in enumerate(daughters):
                daughter_meta = dict(daughter.get("metadata", {}))
                daughter_meta["split_validation"] = {
                    **split_validation,
                    "flattened_from_parent_crypt_id": i,
                    "daughter_index": j,
                }
                detections.append(
                    {
                        "crypt_id": f"{i}.{j}",
                        "crypt_vertices": daughter.get("crypt_vertices", []),
                        "bottom_vertex_id": daughter.get("bottom_vertex_id"),
                        "d_crypt": daughter.get("d_crypt"),
                        "L_crypt": daughter.get("L_crypt"),
                        "metadata": daughter_meta,
                    }
                )
            continue

        det = {
            "crypt_id": i,
            "crypt_vertices": parent,
            "boundary_distance_bottom_vertex_id": int(bottom_parent[i]),
            "bottom_vertex_id": int(final_bottom_parent[i]),
            "d_crypt": dnorm_parent[i],
            "L_crypt": float(L_parent[i]),
            "metadata": {
                "detection_stage": "fresh_initial_candidate",
                "split_validation": split_validation,
                "boundary_distance_bottom_selection": bottom_info_parent[i],
                "final_tip_selection": final_bottom_info_parent[i],
            },
        }
        if daughters:
            det["daughters"] = daughters
            det["stem_vertices"] = stem_vertices
            if split_validation.get("neck_position") is not None:
                det["neck_position"] = split_validation["neck_position"]
                det["neck_vertices"] = split_validation.get("neck_region_vertices", [])
                det["neck_region_vertices"] = split_validation.get("neck_region_vertices", [])
        detections.append(det)

    intermediates = {
        "initial_patches": parents,
        "refined_by_parent": refined_by_parent,
        "encoding": enc_vars,
        "d_levels": d_levels,
        "dnorm_parent": dnorm_parent,
        "L_parent": L_parent,
        "bottom_parent": bottom_parent,
        "final_bottom_parent": final_bottom_parent,
        "bottom_info_parent": bottom_info_parent,
        "final_bottom_info_parent": final_bottom_info_parent,
        "split_validations": split_validations,
        "detection_mesh_smoothed": bool(smooth_mesh),
        **seg_vars,
    }
    if return_intermediates:
        return detections, intermediates
    return detections


def build_skeleton_from_segmentation_parameters(
    mesh,
    vocab,
    *,
    geodesic_fn,
    build_kwargs: dict[str, Any] | None = None,
    detection_kwargs: dict[str, Any] | None = None,
) -> SkeletonGraph:
    """Convenience wrapper that reruns detection and builds a skeleton."""
    detection_kwargs = dict(detection_kwargs or {})
    build_kwargs = dict(build_kwargs or {})
    detection_kwargs.pop("return_intermediates", None)
    detections = detect_crypts_for_skeleton(
        mesh,
        vocab,
        geodesic_fn=geodesic_fn,
        **detection_kwargs,
    )
    return build_skeleton_from_crypt_detections(
        vertices=mesh.v,
        faces=mesh.f,
        crypt_detections=detections,
        **build_kwargs,
    )

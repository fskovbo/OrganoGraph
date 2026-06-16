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
from scipy.signal import savgol_filter

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


def _radial_distances_to_axis(points, origin, direction) -> np.ndarray:
    """Return perpendicular distances to an infinite 3D axis."""
    points = as_points(points)
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if points.size == 0 or norm <= 1e-12:
        return np.empty(0, dtype=float)
    unit = direction / norm
    offsets = points - origin[None, :]
    axial = offsets @ unit
    radial = offsets - axial[:, None] * unit[None, :]
    return np.linalg.norm(radial, axis=1)


def _validate_split_branch_geometry(
    vertices,
    faces,
    parent_vertices,
    daughters,
    split_validation: dict[str, Any],
    *,
    min_confidence: float = 0.6,
    max_neck_to_body_radius_ratio: float = 0.8,
) -> dict[str, Any]:
    """Score whether a growth-ring neck bounds a coherent split appendix.

    The decision uses the whole parent candidate, including daughter crypts,
    rather than the small residual stem left after daughter regions are
    removed.  This makes large daughter crypts compatible with a real branch.
    Residual-stem measurements are retained only as diagnostics.
    """
    result = dict(split_validation)
    result["branch_geometry_validation"] = {
        "applied": False,
        "accepted": bool(result.get("kept_as_split", False)),
        "reason": "growth_validation_rejected",
        "confidence": None,
        "min_confidence": float(min_confidence),
        "max_neck_to_body_radius_ratio": float(max_neck_to_body_radius_ratio),
        "envelope_constriction_score": None,
        "boundary_prominence_score": None,
        "daughter_side_score": None,
        "envelope_depth_score": None,
        "region_size_score": None,
        "neck_radius": None,
        "body_radius": None,
        "neck_to_body_radius_ratio": None,
        "parent_envelope_radius": None,
        "neck_to_parent_envelope_radius_ratio": None,
    }
    if not result.get("kept_as_split", False):
        return result

    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    parent = _coerce_patch(parent_vertices)
    root_region = _coerce_patch(result.get("neck_region_vertices"))
    if root_region.size == 0:
        root_region = parent

    # Residual-stem geometry is useful for diagnosis but is deliberately not a
    # branch acceptance criterion because large daughters may consume the stem.
    remove: set[int] = set()
    for daughter in daughters:
        daughter_region = _crypt_side_region(daughter)
        if daughter_region.size:
            remove.update(map(int, daughter_region.tolist()))
    branch_region = np.asarray(
        [int(v) for v in root_region if int(v) not in remove],
        dtype=np.int64,
    )
    if branch_region.size < 3:
        stem = set(map(int, _coerce_patch(parent_vertices).tolist()))
        for daughter in daughters:
            stem.difference_update(
                map(
                    int,
                    _coerce_patch(
                        _first_present(
                            daughter,
                            ("crypt_vertices", "patch_vertices", "vertices", "patch"),
                        )
                    ).tolist(),
                )
        )
        branch_region = np.asarray(sorted(stem), dtype=np.int64)

    region_mask = np.zeros(vertices.shape[0], dtype=bool)
    region_mask[root_region] = True
    boundary_edges = _boundary_edges_for_region(
        _mesh_edges_from_faces(faces),
        region_mask,
    )
    boundary_vertices = (
        np.unique(boundary_edges)
        if boundary_edges.size
        else np.empty(0, dtype=np.int64)
    )
    neck_center = result.get("neck_position")
    if (
        neck_center is None
        or parent.size < 3
        or root_region.size < 3
        or boundary_vertices.size < 3
    ):
        result["kept_as_split"] = False
        result["reason"] = "insufficient_branch_confidence_geometry"
        result["branch_geometry_validation"].update(
            {
                "accepted": False,
                "reason": "insufficient_branch_confidence_geometry",
                "confidence": 0.0,
                "n_branch_vertices": int(branch_region.size),
                "n_boundary_vertices": int(boundary_vertices.size),
            }
        )
        return result

    neck_center = np.asarray(neck_center, dtype=float)
    envelope_center = centroid(vertices[parent])
    axis = envelope_center - neck_center
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        centered = vertices[parent] - envelope_center[None, :]
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        if np.dot(centroid(vertices[root_region]) - neck_center, axis) < 0.0:
            axis = -axis
    unit_axis = axis / max(float(np.linalg.norm(axis)), 1e-12)

    neck_radial = _radial_distances_to_axis(
        vertices[boundary_vertices],
        neck_center,
        axis,
    )
    envelope_radial = _radial_distances_to_axis(
        vertices[parent],
        neck_center,
        axis,
    )
    neck_radial = neck_radial[np.isfinite(neck_radial)]
    envelope_radial = envelope_radial[np.isfinite(envelope_radial)]
    neck_radius = float(np.median(neck_radial))
    envelope_radius = (
        float(np.quantile(envelope_radial, 0.75))
        if envelope_radial.size
        else 0.0
    )
    radius_ratio = neck_radius / max(envelope_radius, 1e-12)
    raw_constriction = max(1.0 - radius_ratio, 0.0)
    envelope_constriction_score = float(np.clip(raw_constriction / 0.25, 0.0, 1.0))

    lengths = np.asarray(result.get("boundary_lengths", []), dtype=float)
    min_index = result.get("minimum_index")
    boundary_prominence = 0.0
    if lengths.size >= 3 and min_index is not None:
        min_index = int(min_index)
        if 0 < min_index < lengths.size - 1:
            minimum = float(lengths[min_index])
            left_reference = float(np.max(lengths[:min_index]))
            right_reference = float(np.max(lengths[min_index + 1 :]))
            left_drop = (left_reference - minimum) / max(left_reference, 1e-12)
            right_drop = (right_reference - minimum) / max(right_reference, 1e-12)
            boundary_prominence = max(min(left_drop, right_drop), 0.0)
    boundary_prominence_score = float(
        np.clip(boundary_prominence / 0.12, 0.0, 1.0)
    )

    root_set = set(map(int, root_region.tolist()))
    daughter_scores = []
    for daughter in daughters:
        daughter_patch = _coerce_patch(
            _first_present(
                daughter,
                ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
            )
        )
        if daughter_patch.size == 0:
            continue
        containment = float(
            np.mean([int(vertex) in root_set for vertex in daughter_patch])
        )
        projections = (vertices[daughter_patch] - neck_center[None, :]) @ unit_axis
        correct_side = float(np.mean(projections >= -1e-12))
        daughter_scores.append(0.5 * containment + 0.5 * correct_side)
    daughter_side_score = (
        float(np.min(daughter_scores)) if daughter_scores else 0.0
    )

    envelope_projection = (vertices[parent] - neck_center[None, :]) @ unit_axis
    envelope_depth = max(float(np.quantile(envelope_projection, 0.9)), 0.0)
    envelope_depth_ratio = envelope_depth / max(envelope_radius, 1e-12)
    envelope_depth_score = float(np.clip(envelope_depth_ratio / 1.0, 0.0, 1.0))

    region_fraction = float(root_region.size) / max(vertices.shape[0], 1)
    max_fraction = max(float(result.get("max_mesh_fraction", 0.35)), 1e-6)
    if region_fraction <= 0.2:
        region_size_score = 1.0
    else:
        region_size_score = float(
            np.clip((max_fraction - region_fraction) / max(max_fraction - 0.2, 1e-6), 0.0, 1.0)
        )

    confidence = (
        0.40 * envelope_constriction_score
        + 0.25 * boundary_prominence_score
        + 0.20 * daughter_side_score
        + 0.10 * envelope_depth_score
        + 0.05 * region_size_score
    )
    accepted = confidence >= float(min_confidence)
    reason = "branch_confidence_accepted" if accepted else "branch_confidence_below_threshold"

    body_vertices = np.setdiff1d(
        np.arange(vertices.shape[0], dtype=np.int64),
        root_region,
    )
    body_radius = None
    neck_to_body_ratio = None
    body_radius_check_passed = True
    if body_vertices.size >= 3:
        body_center = centroid(vertices[body_vertices])
        body_axis = neck_center - body_center
        body_radial = _radial_distances_to_axis(
            vertices[body_vertices],
            body_center,
            body_axis,
        )
        body_radial = body_radial[np.isfinite(body_radial)]
        if body_radial.size >= 3:
            body_radius = float(np.quantile(body_radial, 0.75))
            if body_radius > 1e-12:
                neck_to_body_ratio = neck_radius / body_radius
                body_radius_check_passed = (
                    neck_to_body_ratio
                    <= float(max_neck_to_body_radius_ratio)
                )

    if accepted and not body_radius_check_passed:
        accepted = False
        reason = "body_side_neck_too_broad_for_body"
    if not accepted:
        result["kept_as_split"] = False
        result["reason"] = reason

    residual_branch_radius = None
    residual_branch_depth = None
    residual_branch_depth_ratio = None
    if branch_region.size >= 3:
        branch_center = centroid(vertices[branch_region])
        residual_branch_depth = float(np.linalg.norm(branch_center - neck_center))
        residual_radial = _radial_distances_to_axis(
            vertices[branch_region],
            neck_center,
            branch_center - neck_center,
        )
        if residual_radial.size:
            residual_branch_radius = float(np.quantile(residual_radial, 0.75))
            residual_branch_depth_ratio = residual_branch_depth / max(
                residual_branch_radius,
                1e-12,
            )

    result["branch_geometry_validation"].update(
        {
            "applied": True,
            "accepted": bool(accepted),
            "reason": reason,
            "confidence": float(confidence),
            "body_radius_check_passed": bool(body_radius_check_passed),
            "envelope_constriction_score": envelope_constriction_score,
            "boundary_prominence_score": boundary_prominence_score,
            "daughter_side_score": daughter_side_score,
            "envelope_depth_score": envelope_depth_score,
            "region_size_score": region_size_score,
            "neck_radius": neck_radius,
            "body_radius": body_radius,
            "neck_to_body_radius_ratio": neck_to_body_ratio,
            "parent_envelope_radius": envelope_radius,
            "neck_to_parent_envelope_radius_ratio": float(radius_ratio),
            "boundary_prominence": boundary_prominence,
            "envelope_depth": envelope_depth,
            "envelope_depth_ratio": envelope_depth_ratio,
            "region_fraction": region_fraction,
            "n_branch_vertices": int(branch_region.size),
            "n_parent_envelope_vertices": int(parent.size),
            "n_boundary_vertices": int(boundary_vertices.size),
            "residual_branch_radius": residual_branch_radius,
            "residual_branch_depth": residual_branch_depth,
            "residual_branch_depth_ratio": residual_branch_depth_ratio,
            "parent_envelope_center": envelope_center.tolist(),
        }
    )
    return result


def _refine_broad_transition_opening(
    mesh,
    detection: dict[str, Any],
    levels,
    *,
    geodesic_fn,
    geodesic_kwargs: dict[str, Any],
    max_opening_to_crypt_body_ratio: float = 0.85,
    branch_max_opening_to_crypt_body_ratio: float = 0.95,
    min_linear_profile_r2: float = 0.985,
    max_linear_profile_deviation: float = 0.08,
    min_attachment_level: float = 0.35,
    window_length: int = 9,
    polyorder: int = 3,
) -> dict[str, Any]:
    """Move an overly broad transition attachment tipward without removing it."""
    profile = detection.get("neck_profile")
    relation = profile.get("relation") if isinstance(profile, dict) else None
    ratio_limit = float(
        branch_max_opening_to_crypt_body_ratio
        if relation == "branch_crypt"
        else max_opening_to_crypt_body_ratio
    )
    diagnostics = {
        "applied": False,
        "refined": False,
        "reason": "not_a_transition",
        "max_opening_to_crypt_body_ratio": ratio_limit,
        "attachment_relation": relation,
        "min_linear_profile_r2": float(min_linear_profile_r2),
        "max_linear_profile_deviation": float(max_linear_profile_deviation),
        "linear_profile_r2": None,
        "linear_profile_max_deviation": None,
        "opening_to_crypt_body_ratio": None,
        "original_attachment_level": float(detection.get("attachment_level", 1.0)),
        "refined_attachment_level": float(detection.get("attachment_level", 1.0)),
    }
    detection["broad_opening_validation"] = diagnostics
    if not isinstance(profile, dict) or profile.get("kind") != "transition":
        return detection
    if not (0.0 < ratio_limit < 1.0):
        diagnostics["reason"] = "disabled_by_ratio"
        return detection

    original_levels = np.asarray(
        detection.get("circumference_levels", levels),
        dtype=float,
    ).reshape(-1)
    original_circumference = np.asarray(
        detection.get("circumference", []),
        dtype=float,
    ).reshape(-1)
    original_attachment_level = float(detection.get("attachment_level", 1.0))
    if original_levels.size != original_circumference.size:
        diagnostics["reason"] = "missing_original_circumference_profile"
        return detection
    original_valid = (
        np.isfinite(original_levels)
        & np.isfinite(original_circumference)
        & (original_levels >= 0.05)
        & (original_levels <= original_attachment_level)
    )
    if np.count_nonzero(original_valid) < 7:
        diagnostics["reason"] = "insufficient_original_circumference_profile"
        return detection
    original_x = original_levels[original_valid]
    original_y = original_circumference[original_valid]
    original_order = np.argsort(original_x)
    original_x = original_x[original_order]
    original_y = original_y[original_order]
    original_wl = min(
        int(window_length) | 1,
        original_x.size if original_x.size % 2 else original_x.size - 1,
    )
    original_wl = max(original_wl, 5)
    original_po = min(int(polyorder), original_wl - 2)
    original_smooth = savgol_filter(
        original_y,
        window_length=original_wl,
        polyorder=original_po,
        mode="interp",
    )
    coefficients = np.polyfit(original_x, original_smooth, 1)
    fitted_line = np.polyval(coefficients, original_x)
    residual_sum = float(np.sum((original_smooth - fitted_line) ** 2))
    centered_sum = float(
        np.sum((original_smooth - np.mean(original_smooth)) ** 2)
    )
    linear_r2 = 1.0 - residual_sum / max(centered_sum, 1e-12)
    profile_span = max(float(np.ptp(original_smooth)), 1e-12)
    max_deviation = float(
        np.max(np.abs(original_smooth - fitted_line)) / profile_span
    )
    diagnostics.update(
        {
            "linear_profile_r2": linear_r2,
            "linear_profile_max_deviation": max_deviation,
            "linear_profile_slope": float(coefficients[0]),
        }
    )
    is_linear = (
        coefficients[0] > 0.0
        and linear_r2 >= float(min_linear_profile_r2)
        and max_deviation <= float(max_linear_profile_deviation)
    )
    if not is_linear:
        diagnostics["reason"] = "structured_transition_profile_preserved"
        return detection

    vertices = as_points(mesh.v)
    faces = np.asarray(mesh.f, dtype=np.int64)
    patch = _coerce_patch(detection.get("crypt_vertices"))
    tip_id = int(detection.get("bottom_vertex_id", -1))
    old_field = np.asarray(detection.get("d_crypt"), dtype=float).reshape(-1)
    if (
        tip_id < 0
        or tip_id >= vertices.shape[0]
        or old_field.size != vertices.shape[0]
        or patch.size < 3
    ):
        diagnostics["reason"] = "missing_tip_axis_or_patch"
        return detection

    old_level = float(detection.get("attachment_level", 1.0))
    old_mask = np.isfinite(old_field) & (old_field <= old_level)
    boundary_edges = _boundary_edges_for_region(
        _mesh_edges_from_faces(faces),
        old_mask,
    )
    boundary_vertices = (
        np.unique(boundary_edges)
        if boundary_edges.size
        else np.empty(0, dtype=np.int64)
    )
    if boundary_vertices.size < 3:
        diagnostics["reason"] = "missing_attachment_boundary"
        return detection

    distances = np.asarray(
        geodesic_fn(mesh, sources=[tip_id], **dict(geodesic_kwargs or {})),
        dtype=float,
    )
    if distances.ndim > 1:
        distances = distances[0]
    if distances.size != vertices.shape[0]:
        diagnostics["reason"] = "invalid_updated_tip_geodesics"
        return detection
    normalization_length = float(np.nanmedian(distances[boundary_vertices]))
    if not np.isfinite(normalization_length) or normalization_length <= 1e-12:
        diagnostics["reason"] = "invalid_attachment_distance"
        return detection

    from organograph.crypts.analysis import crypt_circumference

    updated_field = distances / normalization_length
    levels = np.asarray(levels, dtype=float).reshape(-1)
    circumference = crypt_circumference(mesh, updated_field, levels)
    finite = np.isfinite(levels) & np.isfinite(circumference)
    if np.count_nonzero(finite) < 7:
        diagnostics["reason"] = "insufficient_updated_circumference"
        return detection
    x = levels[finite]
    y = np.asarray(circumference, dtype=float)[finite]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    wl = min(int(window_length) | 1, x.size if x.size % 2 else x.size - 1)
    wl = max(wl, 5)
    po = min(int(polyorder), wl - 2)
    smooth = savgol_filter(y, window_length=wl, polyorder=po, mode="interp")

    body_mask = (x >= max(float(min_attachment_level) * 0.5, 0.05)) & (x <= 0.8)
    if np.count_nonzero(body_mask) < 3:
        diagnostics["reason"] = "insufficient_crypt_body_profile"
        return detection
    crypt_body_circumference = float(np.nanmax(smooth[body_mask]))
    opening_circumference = float(np.interp(1.0, x, smooth))
    if crypt_body_circumference <= 1e-12:
        diagnostics["reason"] = "invalid_crypt_body_circumference"
        return detection

    opening_ratio = opening_circumference / crypt_body_circumference
    diagnostics.update(
        {
            "applied": True,
            "reason": "opening_within_limit",
            "opening_circumference": opening_circumference,
            "crypt_body_circumference": crypt_body_circumference,
            "opening_to_crypt_body_ratio": float(opening_ratio),
            "updated_tip_vertex_id": tip_id,
        }
    )
    if opening_ratio <= ratio_limit:
        return detection

    threshold = ratio_limit * crypt_body_circumference
    candidate_mask = (
        (x >= max(float(min_attachment_level), float(np.min(x))))
        & (x <= 1.0)
        & (smooth <= threshold)
    )
    candidate_levels = x[candidate_mask]
    if candidate_levels.size == 0:
        refined_level = max(float(min_attachment_level), float(np.min(x)))
        diagnostics["reason"] = "opening_limited_at_minimum_level"
    else:
        refined_level = float(np.max(candidate_levels))
        diagnostics["reason"] = "opening_moved_tipward"

    attachment = _contour_center_from_distance_field(
        vertices,
        faces,
        updated_field,
        level=refined_level,
        prefer_vertices=patch,
    )
    if attachment is None:
        diagnostics["reason"] = "refined_contour_not_found"
        return detection

    detection["d_crypt"] = updated_field
    detection["L_crypt"] = normalization_length * refined_level
    detection["circumference_levels"] = levels
    detection["circumference"] = np.asarray(circumference, dtype=float)
    detection["attachment_level"] = refined_level
    detection["attachment_position"] = attachment
    detection["neck_position"] = attachment
    detection["attachment_region_vertices"] = np.where(
        np.isfinite(updated_field) & (updated_field <= refined_level)
    )[0].astype(np.int64)
    profile = dict(profile)
    profile["attachment_level"] = refined_level
    profile["broad_opening_refined"] = True
    profile["broad_opening_original_level"] = old_level
    detection["neck_profile"] = profile
    diagnostics.update(
        {
            "refined": True,
            "refined_attachment_level": refined_level,
            "allowed_opening_circumference": float(threshold),
            "refined_opening_circumference": float(np.interp(refined_level, x, smooth)),
        }
    )
    return detection


def _smoothed_circumference_profile(
    levels,
    circumference,
    *,
    max_level: float,
    min_level: float = 0.05,
    window_length: int = 9,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    levels = np.asarray(levels, dtype=float).reshape(-1)
    circumference = np.asarray(circumference, dtype=float).reshape(-1)
    if levels.size != circumference.size:
        return None, None
    valid = (
        np.isfinite(levels)
        & np.isfinite(circumference)
        & (levels >= float(min_level))
        & (levels <= float(max_level))
    )
    if np.count_nonzero(valid) < 7:
        return None, None
    x = levels[valid]
    y = circumference[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    wl = min(int(window_length) | 1, x.size if x.size % 2 else x.size - 1)
    wl = max(wl, 5)
    po = min(int(polyorder), wl - 2)
    return x, savgol_filter(y, window_length=wl, polyorder=po, mode="interp")


def _body_vertices_from_detections(n_vertices: int, detections: list[dict[str, Any]]) -> np.ndarray:
    excluded: set[int] = set()
    for detection in detections:
        region = _crypt_side_region(detection)
        if region.size:
            excluded.update(map(int, region.tolist()))
    if not excluded:
        return np.arange(int(n_vertices), dtype=np.int64)
    body = np.setdiff1d(
        np.arange(int(n_vertices), dtype=np.int64),
        np.fromiter(excluded, dtype=np.int64),
    )
    if body.size < 3:
        return np.arange(int(n_vertices), dtype=np.int64)
    return body


def _host_width_around_attachment(vertices, body_vertices, attachment, *, quantile: float = 0.75) -> float:
    vertices = as_points(vertices)
    body_vertices = _coerce_patch(body_vertices)
    if body_vertices.size < 3:
        return float("nan")
    body_center = centroid(vertices[body_vertices])
    radial = _radial_distances_to_axis(
        vertices[body_vertices],
        body_center,
        np.asarray(attachment, dtype=float) - body_center,
    )
    radial = radial[np.isfinite(radial)]
    if radial.size < 3:
        return float("nan")
    return 2.0 * float(np.quantile(radial, float(quantile)))


def _earlier_second_derivative_transition_level(
    levels,
    smooth,
    *,
    current_level: float,
    min_level: float,
    min_score: float,
    window_length: int,
) -> tuple[float | None, dict[str, Any]]:
    x = np.asarray(levels, dtype=float)
    y = np.asarray(smooth, dtype=float)
    details = {
        "candidate_level": None,
        "candidate_score": 0.0,
        "candidate_contrast": 0.0,
    }
    if x.size < 7:
        return None, details
    spacing = float(np.median(np.diff(x)))
    if not np.isfinite(spacing) or spacing <= 0:
        return None, details
    wl = min(int(window_length) | 1, x.size if x.size % 2 else x.size - 1)
    wl = max(wl, 5)
    po = min(3, wl - 2)
    second = savgol_filter(
        y,
        window_length=wl,
        polyorder=po,
        deriv=2,
        delta=spacing,
        mode="interp",
    )
    margin = max(2.0 * spacing, 0.03)
    search = np.where(
        (x >= float(min_level))
        & (x <= float(current_level) - margin)
        & np.isfinite(second)
    )[0]
    if search.size == 0:
        return None, details
    local = search[
        (search > 0)
        & (search < x.size - 1)
        & (second[search - 1] <= second[search])
        & (second[search] >= second[search + 1])
        & (second[search] > 0.0)
    ]
    if local.size == 0:
        local = search[second[search] > 0.0]
    if local.size == 0:
        return None, details
    background = float(np.median(np.abs(second[search])))
    best_index = int(local[np.argmax(second[local])])
    best_score = 0.0
    best_contrast = 0.0
    accepted_index = None
    accepted_levels = []
    for idx in local[np.argsort(x[local])]:
        idx = int(idx)
        positive_peak = max(float(second[idx]), 0.0)
        contrast = positive_peak / max(background, 1e-12)
        score = contrast / (1.0 + contrast)
        if score > best_score:
            best_index = idx
            best_score = score
            best_contrast = contrast
        if score >= float(min_score):
            accepted_levels.append(float(x[idx]))
            if accepted_index is None:
                accepted_index = idx
    report_index = accepted_index if accepted_index is not None else best_index
    report_peak = max(float(second[report_index]), 0.0)
    report_contrast = report_peak / max(background, 1e-12)
    report_score = report_contrast / (1.0 + report_contrast)
    details.update(
        {
            "candidate_level": float(x[report_index]),
            "candidate_score": float(np.clip(report_score, 0.0, 1.0)),
            "candidate_contrast": report_contrast,
            "strongest_candidate_level": float(x[best_index]),
            "strongest_candidate_score": float(np.clip(best_score, 0.0, 1.0)),
            "strongest_candidate_contrast": best_contrast,
            "accepted_candidate_levels": accepted_levels,
        }
    )
    if accepted_index is not None:
        return float(x[accepted_index]), details
    return None, details


def _refine_body_transition_width_outliers(
    mesh,
    detections: list[dict[str, Any]],
    *,
    max_crypt_to_host_width_ratio: float = 0.8,
    host_width_quantile: float = 0.75,
    min_second_derivative_score: float = 0.6,
    min_attachment_level: float = 0.35,
    window_length: int = 9,
    polyorder: int = 3,
) -> list[dict[str, Any]]:
    """Repair only body-attached transition crypts wider than their host."""
    ratio_limit = float(max_crypt_to_host_width_ratio)
    if not (np.isfinite(ratio_limit) and ratio_limit > 0.0):
        return detections
    vertices = as_points(mesh.v)
    faces = np.asarray(mesh.f, dtype=np.int64)
    body_vertices = _body_vertices_from_detections(vertices.shape[0], detections)

    for detection in detections:
        profile = detection.get("neck_profile")
        if (
            not isinstance(profile, dict)
            or profile.get("kind") != "transition"
            or profile.get("relation") != "body_crypt"
            or detection.get("daughters")
        ):
            continue

        attachment = _point_from_keys(
            vertices,
            detection,
            ("attachment_position", "neck_position"),
        )
        dfield = np.asarray(detection.get("d_crypt"), dtype=float).reshape(-1)
        levels = detection.get("circumference_levels")
        circumference = detection.get("circumference")
        current_level = float(detection.get("attachment_level", 1.0))
        diagnostics = {
            "applied": False,
            "refined": False,
            "reason": "not_evaluated",
            "max_crypt_to_host_width_ratio": ratio_limit,
            "host_width_quantile": float(host_width_quantile),
            "crypt_width": None,
            "host_width": None,
            "crypt_to_host_width_ratio": None,
        }
        detection["body_transition_width_validation"] = diagnostics
        if attachment is None or dfield.size != vertices.shape[0]:
            diagnostics["reason"] = "missing_attachment_or_distance_field"
            continue
        x, smooth = _smoothed_circumference_profile(
            levels,
            circumference,
            max_level=current_level,
            min_level=0.05,
            window_length=window_length,
            polyorder=polyorder,
        )
        if x is None:
            diagnostics["reason"] = "missing_circumference_profile"
            continue
        crypt_width = float(np.nanmax(smooth)) / np.pi
        host_width = _host_width_around_attachment(
            vertices,
            body_vertices,
            attachment,
            quantile=host_width_quantile,
        )
        if not (np.isfinite(crypt_width) and np.isfinite(host_width) and host_width > 0.0):
            diagnostics["reason"] = "invalid_widths"
            continue
        width_ratio = crypt_width / host_width
        diagnostics.update(
            {
                "applied": True,
                "reason": "within_width_limit",
                "crypt_width": crypt_width,
                "host_width": host_width,
                "crypt_to_host_width_ratio": float(width_ratio),
            }
        )
        if width_ratio <= ratio_limit:
            continue

        candidate_level, candidate_details = _earlier_second_derivative_transition_level(
            x,
            smooth,
            current_level=current_level,
            min_level=min_attachment_level,
            min_score=min_second_derivative_score,
            window_length=window_length,
        )
        target_level = candidate_level
        reason = "earlier_second_derivative_transition"
        if target_level is None:
            max_allowed_circumference = ratio_limit * host_width * np.pi
            running_max = np.maximum.accumulate(smooth)
            allowed = x[(x >= float(min_attachment_level)) & (running_max <= max_allowed_circumference)]
            if allowed.size:
                target_level = float(np.max(allowed))
                reason = "width_threshold_shrink"
            else:
                target_level = max(float(min_attachment_level), float(np.min(x)))
                reason = "width_threshold_minimum_level"

        attachment_new = _contour_center_from_distance_field(
            vertices,
            faces,
            dfield,
            level=float(target_level),
            prefer_vertices=_coerce_patch(detection.get("crypt_vertices")),
        )
        if attachment_new is None:
            diagnostics.update({**candidate_details, "reason": "target_contour_not_found"})
            continue
        detection["attachment_level"] = float(target_level)
        detection["attachment_position"] = attachment_new
        detection["neck_position"] = attachment_new
        detection["attachment_region_vertices"] = np.where(
            np.isfinite(dfield) & (dfield <= float(target_level))
        )[0].astype(np.int64)
        profile = dict(profile)
        profile["attachment_level"] = float(target_level)
        profile["body_transition_width_refined"] = True
        profile["body_transition_width_original_level"] = current_level
        detection["neck_profile"] = profile
        refined_mask = x <= float(target_level)
        refined_width = (
            float(np.nanmax(smooth[refined_mask])) / np.pi
            if np.any(refined_mask)
            else float("nan")
        )
        diagnostics.update(
            {
                **candidate_details,
                "refined": True,
                "reason": reason,
                "original_attachment_level": current_level,
                "refined_attachment_level": float(target_level),
                "refined_crypt_width": refined_width,
                "refined_crypt_to_host_width_ratio": (
                    refined_width / host_width
                    if np.isfinite(refined_width) and host_width > 0.0
                    else None
                ),
            }
        )
    return detections


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


def _penalize_short_crypt_bending(
    vertices,
    crypt_vertices,
    source_position,
    intermediate_position,
    tip_position,
    *,
    max_dimensionless_curvature: float | None,
    penalty_strength: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Softly pull a crypt waypoint toward its chord when bending is excessive."""
    source = np.asarray(source_position, dtype=float)
    candidate = np.asarray(intermediate_position, dtype=float)
    tip = np.asarray(tip_position, dtype=float)
    limit = (
        None
        if max_dimensionless_curvature is None
        else float(max_dimensionless_curvature)
    )
    diagnostics = {
        "applied": False,
        "max_dimensionless_curvature": limit,
        "penalty_strength": float(penalty_strength),
        "original_dimensionless_curvature": None,
        "final_dimensionless_curvature": None,
        "waypoint_lateral_scale": 1.0,
    }
    chord = tip - source
    chord_length = float(np.linalg.norm(chord))
    patch = _coerce_patch(crypt_vertices)
    if (
        limit is None
        or limit <= 0.0
        or float(penalty_strength) <= 0.0
        or chord_length <= 1e-12
        or patch.size < 3
    ):
        diagnostics["reason"] = "disabled_or_insufficient_geometry"
        return candidate, diagnostics

    unit = chord / chord_length
    longitudinal = float(
        np.clip(
            np.dot(candidate - source, unit),
            0.1 * chord_length,
            0.9 * chord_length,
        )
    )
    projection = source + longitudinal * unit
    radial = _radial_distances_to_axis(
        as_points(vertices)[patch],
        source,
        chord,
    )
    radial = radial[np.isfinite(radial)]
    if radial.size < 3:
        diagnostics["reason"] = "insufficient_radius_samples"
        return candidate, diagnostics
    crypt_radius = float(np.median(radial))
    if crypt_radius <= 1e-12:
        diagnostics["reason"] = "degenerate_crypt_radius"
        return candidate, diagnostics

    def curvature(point: np.ndarray) -> tuple[float, float, float]:
        first = point - source
        second = tip - point
        n_first = float(np.linalg.norm(first))
        n_second = float(np.linalg.norm(second))
        path_length = n_first + n_second
        if n_first <= 1e-12 or n_second <= 1e-12 or path_length <= 1e-12:
            return 0.0, 0.0, path_length
        cosine = float(np.clip(np.dot(first, second) / (n_first * n_second), -1.0, 1.0))
        angle = float(np.arccos(cosine))
        return angle * crypt_radius / path_length, angle, path_length

    original_curvature, original_angle, original_length = curvature(candidate)
    diagnostics.update(
        {
            "crypt_radius": crypt_radius,
            "original_dimensionless_curvature": original_curvature,
            "original_bend_angle": original_angle,
            "original_path_length": original_length,
        }
    )
    if original_curvature <= limit:
        diagnostics.update(
            {
                "reason": "within_curvature_limit",
                "final_dimensionless_curvature": original_curvature,
            }
        )
        return candidate, diagnostics

    lateral = candidate - projection
    alphas = np.linspace(0.0, 1.0, 101)
    objectives = []
    for alpha in alphas:
        trial = projection + float(alpha) * lateral
        trial_curvature, _, _ = curvature(trial)
        excess = max(trial_curvature / limit - 1.0, 0.0)
        objectives.append((1.0 - float(alpha)) ** 2 + float(penalty_strength) * excess**2)
    best_index = int(np.argmin(objectives))
    best_alpha = float(alphas[best_index])
    refined = projection + best_alpha * lateral
    final_curvature, final_angle, final_length = curvature(refined)
    diagnostics.update(
        {
            "applied": best_alpha < 1.0 - 1e-12,
            "reason": "curvature_penalty_applied",
            "waypoint_lateral_scale": best_alpha,
            "final_dimensionless_curvature": final_curvature,
            "final_bend_angle": final_angle,
            "final_path_length": final_length,
        }
    )
    return refined, diagnostics


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
    bend_max_dimensionless_curvature: float | None,
    bend_curvature_penalty: float,
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
        source_type = graph.node(source_id).node_type
        graph.add_edge(
            f"{path_prefix}_{source_type}_to_tip",
            source_id,
            tip_id,
            edge_type=f"{source_type}_to_tip",
            crypt_id=crypt_id,
        )
        return

    intermediate_position, bend_diagnostics = _penalize_short_crypt_bending(
        vertices,
        crypt_vertices,
        source_position,
        intermediate_position,
        tip_position,
        max_dimensionless_curvature=bend_max_dimensionless_curvature,
        penalty_strength=bend_curvature_penalty,
    )
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
            "bend_validation": _json_safe_metadata(bend_diagnostics),
        },
    )
    source_type = graph.node(source_id).node_type
    graph.add_edge(
        f"{path_prefix}_{source_type}_to_{intermediate_type}",
        source_id,
        intermediate_id,
        edge_type=f"{source_type}_to_{intermediate_type}",
        crypt_id=crypt_id,
    )
    graph.add_edge(
        f"{path_prefix}_{intermediate_type}_to_tip",
        intermediate_id,
        tip_id,
        edge_type=f"{intermediate_type}_to_tip",
        crypt_id=crypt_id,
    )


def _add_attachment_path(
    graph: SkeletonGraph,
    vertices,
    faces,
    *,
    path_prefix: str,
    host_id: str,
    crypt_id,
    detection: dict[str, Any],
    metadata: dict[str, Any],
    host_edge_prefix: str,
) -> tuple[str, np.ndarray]:
    """Add legacy neck or explicit attachment/constriction nodes."""
    profile = detection.get("neck_profile")
    if not isinstance(profile, dict):
        neck = _neck_position(vertices, faces, detection)
        neck_id = f"{path_prefix}_neck"
        graph.add_node(
            neck_id,
            "neck",
            neck,
            crypt_id=crypt_id,
            metadata=metadata,
        )
        graph.add_edge(
            f"{path_prefix}_{host_edge_prefix}_to_neck",
            host_id,
            neck_id,
            edge_type=f"{host_edge_prefix}_to_neck",
            crypt_id=crypt_id,
        )
        return neck_id, neck

    attachment = _point_from_keys(
        vertices,
        detection,
        ("attachment_position", "attachment_center", "p_attachment"),
    )
    if attachment is None:
        attachment = _neck_position(vertices, faces, detection)
    attachment_id = f"{path_prefix}_attachment"
    junction_meta = {
        **metadata,
        "neck_profile": _json_safe_metadata(profile),
        "attachment_level": float(detection.get("attachment_level", 1.0)),
    }
    graph.add_node(
        attachment_id,
        "attachment",
        attachment,
        crypt_id=crypt_id,
        metadata={**junction_meta, "role": "component_attachment"},
    )
    graph.add_edge(
        f"{path_prefix}_{host_edge_prefix}_to_attachment",
        host_id,
        attachment_id,
        edge_type=f"{host_edge_prefix}_to_attachment",
        crypt_id=crypt_id,
    )
    if profile.get("kind") != "constriction":
        return attachment_id, attachment

    constriction = _point_from_keys(
        vertices,
        detection,
        ("constriction_position", "constriction_center", "neck_position"),
    )
    if constriction is None:
        return attachment_id, attachment
    constriction_id = f"{path_prefix}_constriction"
    graph.add_node(
        constriction_id,
        "constriction",
        constriction,
        crypt_id=crypt_id,
        metadata={
            **junction_meta,
            "role": "narrowest_constriction",
            "constriction_level": profile.get("constriction_level"),
            "distal_boundary_level": profile.get("distal_boundary_level"),
            "c_min": profile.get("c_min"),
            "c_half": profile.get("c_half"),
        },
    )
    graph.add_edge(
        f"{path_prefix}_attachment_to_constriction",
        attachment_id,
        constriction_id,
        edge_type="attachment_to_constriction",
        crypt_id=crypt_id,
    )
    return constriction_id, constriction


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
            (
                "attachment_region_vertices",
                "neck_region_vertices",
                "neck_side_vertices",
                "root_region_vertices",
            ),
        )
    )
    if region.size:
        return region

    dfield = _first_present(detection, ("d_crypt", "distance_field", "dnorm", "dnorm_vertices"))
    if dfield is not None:
        dfield = np.asarray(dfield, dtype=float).reshape(-1)
        level = float(
            detection.get(
                "attachment_level",
                detection.get("neck_level", 1.0),
            )
        )
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
    bend_max_dimensionless_curvature: float | None = 0.5,
    bend_curvature_penalty: float = 5.0,
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
    Optional crypt waypoints are softly straightened when their bend is too
    sharp for the path length and estimated crypt radius.

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

        root_source_id, root_source_position = _add_attachment_path(
            graph,
            vertices,
            faces,
            path_prefix=crypt_prefix,
            host_id="body",
            crypt_id=crypt_id,
            detection=detection,
            metadata=common_meta,
            host_edge_prefix="body",
        )

        daughters = _daughter_detections(detection)
        if daughters:
            daughter_tips = [_tip_position(vertices, daughter) for daughter in daughters]
            branch_region = np.empty(0, dtype=np.int64)
            branch = None
            if refine_branch_centers_from_necks:
                branch, branch_region = _branch_position_from_regions(vertices, detection, daughters)
            if branch is None:
                branch = _branch_position(
                    vertices,
                    detection,
                    root_source_position,
                    daughter_tips,
                )
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
                    "branch_region_vertices": branch_region.tolist(),
                },
            )
            graph.add_edge(
                f"{crypt_prefix}_{graph.node(root_source_id).node_type}_to_branch",
                root_source_id,
                branch_id,
                edge_type=f"{graph.node(root_source_id).node_type}_to_branch",
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
                daughter_source_id, daughter_source_position = _add_attachment_path(
                    graph,
                    vertices,
                    faces,
                    path_prefix=f"{crypt_prefix}_daughter_{j}",
                    host_id=branch_id,
                    crypt_id=crypt_id,
                    detection=daughter,
                    metadata={
                        **daughter_meta,
                        "role": "daughter_junction",
                    },
                    host_edge_prefix="branch",
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
                    source_id=daughter_source_id,
                    tip_id=tip_id,
                    crypt_id=crypt_id,
                    detection=daughter,
                    crypt_vertices=daughter_vertices,
                    source_position=daughter_source_position,
                    tip_position=daughter_tips[j],
                    metadata=daughter_meta,
                    bend_strategy=bend_strategy,
                    bend_max_dimensionless_curvature=bend_max_dimensionless_curvature,
                    bend_curvature_penalty=bend_curvature_penalty,
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
            source_id=root_source_id,
            tip_id=tip_id,
            crypt_id=crypt_id,
            detection=detection,
            crypt_vertices=crypt_vertices,
            source_position=root_source_position,
            tip_position=tip,
            metadata=common_meta,
            bend_strategy=bend_strategy,
            bend_max_dimensionless_curvature=bend_max_dimensionless_curvature,
            bend_curvature_penalty=bend_curvature_penalty,
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
    neck_window_length: int = 9,
    neck_polyorder: int = 3,
    neck_min_prominence: float = 0.05,
    neck_min_length: float = 0.05,
    validate_split_stems: bool = True,
    validate_branch_geometry: bool = True,
    branch_min_confidence: float = 0.6,
    branch_max_neck_to_body_radius_ratio: float = 0.8,
    split_growth_max_size_factor: float = 2.0,
    split_growth_max_mesh_fraction: float = 0.35,
    split_growth_smooth_perimeter: bool = True,
    split_growth_smoothing_tolerance: float = 0.0,
    split_growth_min_decrease_fraction: float = 0.0,
    split_growth_min_prominence_fraction: float = 0.01,
    split_growth_robust_window: int = 1,
    refine_broad_crypt_openings: bool = True,
    max_opening_to_crypt_body_ratio: float = 0.85,
    branch_max_opening_to_crypt_body_ratio: float = 0.95,
    broad_opening_min_linear_profile_r2: float = 0.985,
    broad_opening_max_linear_profile_deviation: float = 0.08,
    broad_opening_min_attachment_level: float = 0.35,
    refine_body_transition_width_outliers: bool = True,
    body_transition_max_crypt_to_host_width_ratio: float = 0.8,
    body_transition_host_width_quantile: float = 0.75,
    body_transition_min_second_derivative_score: float = 0.6,
    body_transition_min_attachment_level: float = 0.25,
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
    ``final_tip_min_hks_percent_increase``. Transition-type crypt openings can
    then be shortened from that updated tip if their circumference is too
    broad relative to the crypt body. Split branches are retained only when
    the accepted growth-ring neck is both narrower and deep enough to define a
    distinct component.
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
    circumference_parent, dnorm_parent, L_parent = normalize_crypt_axis_to_neckline(
        detection_mesh,
        dnorm_parent,
        d_levels,
        search_interval=neck_search_interval,
        L_crypt=L_parent,
        window_length=neck_window_length,
        polyorder=neck_polyorder,
        min_prominence=neck_min_prominence,
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
            circumference_child, dnorm_child, L_child = normalize_crypt_axis_to_neckline(
                detection_mesh,
                dnorm_child,
                d_levels,
                search_interval=neck_search_interval,
                L_crypt=L_child,
                window_length=neck_window_length,
                polyorder=neck_polyorder,
                min_prominence=neck_min_prominence,
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
                daughter_detection = {
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
                daughter_detection = _add_neck_profile_geometry(
                    detection_mesh.v,
                    detection_mesh.f,
                    daughter_detection,
                    d_levels,
                    circumference_child[j],
                    relation="branch_crypt",
                    window_length=neck_window_length,
                    polyorder=neck_polyorder,
                    min_prominence=neck_min_prominence,
                    min_neck_length=neck_min_length,
                )
                if refine_broad_crypt_openings:
                    daughter_detection = _refine_broad_transition_opening(
                        detection_mesh,
                        daughter_detection,
                        d_levels,
                        geodesic_fn=geodesic_fn,
                        geodesic_kwargs=geodesic_kwargs,
                        max_opening_to_crypt_body_ratio=max_opening_to_crypt_body_ratio,
                        branch_max_opening_to_crypt_body_ratio=branch_max_opening_to_crypt_body_ratio,
                        min_linear_profile_r2=broad_opening_min_linear_profile_r2,
                        max_linear_profile_deviation=broad_opening_max_linear_profile_deviation,
                        min_attachment_level=broad_opening_min_attachment_level,
                        window_length=neck_window_length,
                        polyorder=neck_polyorder,
                    )
                daughters.append(daughter_detection)
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
        if daughters and validate_branch_geometry:
            split_validation = _validate_split_branch_geometry(
                detection_mesh.v,
                detection_mesh.f,
                parent,
                daughters,
                split_validation,
                min_confidence=branch_min_confidence,
                max_neck_to_body_radius_ratio=branch_max_neck_to_body_radius_ratio,
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
                flattened = dict(daughter)
                flattened["crypt_id"] = f"{i}.{j}"
                flattened["metadata"] = daughter_meta
                detections.append(flattened)
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
            validated_position = split_validation.get("neck_position")
            if validated_position is not None:
                det["neck_position"] = validated_position
                validated_region = split_validation.get(
                    "neck_region_vertices",
                    [],
                )
                det["neck_vertices"] = validated_region
                det["neck_region_vertices"] = validated_region
                region_mask = np.zeros(detection_mesh.v.shape[0], dtype=bool)
                region_mask[np.asarray(validated_region, dtype=np.int64)] = True
                boundary_edges = _boundary_edges_for_region(
                    _mesh_edges_from_faces(detection_mesh.f),
                    region_mask,
                )
                boundary_vertices = (
                    np.unique(boundary_edges)
                    if boundary_edges.size
                    else np.empty(0, dtype=np.int64)
                )
                boundary_levels = dnorm_parent[i, boundary_vertices]
                boundary_levels = boundary_levels[np.isfinite(boundary_levels)]
                if boundary_levels.size:
                    current_neck_level = float(np.median(boundary_levels))
                else:
                    nearest_vertex = int(
                        np.argmin(
                            np.linalg.norm(
                                detection_mesh.v
                                - np.asarray(validated_position, dtype=float)[None, :],
                                axis=1,
                            )
                        )
                    )
                    current_neck_level = float(dnorm_parent[i, nearest_vertex])
                if not np.isfinite(current_neck_level):
                    current_neck_level = 1.0
                neck_source = "validated_parent_patch_boundary"
            else:
                current_neck_level = 1.0
                neck_source = "normalized_parent_axis_neck"
            det["body_branch_circumference_levels"] = (
                np.asarray(d_levels, dtype=float) - current_neck_level
            )
            det["body_branch_circumference"] = np.asarray(
                circumference_parent[i],
                dtype=float,
            )
            det["body_branch_current_neck_level"] = current_neck_level
            det["body_branch_neck_position_source"] = neck_source
            det["body_branch_neck_logic"] = "legacy_single_neck"
        else:
            det = _add_neck_profile_geometry(
                detection_mesh.v,
                detection_mesh.f,
                det,
                d_levels,
                circumference_parent[i],
                relation="body_crypt",
                window_length=neck_window_length,
                polyorder=neck_polyorder,
                min_prominence=neck_min_prominence,
                min_neck_length=neck_min_length,
            )
            if refine_broad_crypt_openings:
                det = _refine_broad_transition_opening(
                    detection_mesh,
                    det,
                    d_levels,
                    geodesic_fn=geodesic_fn,
                    geodesic_kwargs=geodesic_kwargs,
                    max_opening_to_crypt_body_ratio=max_opening_to_crypt_body_ratio,
                    branch_max_opening_to_crypt_body_ratio=branch_max_opening_to_crypt_body_ratio,
                    min_linear_profile_r2=broad_opening_min_linear_profile_r2,
                    max_linear_profile_deviation=broad_opening_max_linear_profile_deviation,
                    min_attachment_level=broad_opening_min_attachment_level,
                    window_length=neck_window_length,
                    polyorder=neck_polyorder,
                )
        detections.append(det)

    if refine_body_transition_width_outliers:
        detections = _refine_body_transition_width_outliers(
            detection_mesh,
            detections,
            max_crypt_to_host_width_ratio=body_transition_max_crypt_to_host_width_ratio,
            host_width_quantile=body_transition_host_width_quantile,
            min_second_derivative_score=body_transition_min_second_derivative_score,
            min_attachment_level=body_transition_min_attachment_level,
            window_length=neck_window_length,
            polyorder=neck_polyorder,
        )

    intermediates = {
        "initial_patches": parents,
        "refined_by_parent": refined_by_parent,
        "encoding": enc_vars,
        "d_levels": d_levels,
        "dnorm_parent": dnorm_parent,
        "circumference_parent": circumference_parent,
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

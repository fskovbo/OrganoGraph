"""Split-branch growth and validation logic."""

from __future__ import annotations

from typing import Any

import numpy as np

from organograph.skeleton.detection.common import _coerce_patch, _first_present
from organograph.skeleton.detection.mesh_regions import (
    _add_vertex_to_region,
    _best_frontier_addition,
    _boundary_edges_for_region,
    _boundary_length_and_center,
    _crypt_side_region,
    _frontier_for_region,
    _mesh_edges_from_faces,
    _radial_distances_to_axis,
    _weighted_vertex_adjacency_from_edges,
)
from organograph.skeleton.geometry import as_points, centroid

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


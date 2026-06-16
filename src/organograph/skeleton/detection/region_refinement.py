"""Post-detection refinements for broad transition regions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from organograph.skeleton.detection.common import _coerce_patch, _point_from_keys
from organograph.skeleton.detection.mesh_regions import (
    _body_center_from_root_regions,
    _boundary_edges_for_region,
    _crypt_side_region,
    _mesh_edges_from_faces,
    _radial_distances_to_axis,
)
from organograph.skeleton.detection.neck_profiles import _contour_center_from_distance_field
from organograph.skeleton.geometry import as_points, centroid

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


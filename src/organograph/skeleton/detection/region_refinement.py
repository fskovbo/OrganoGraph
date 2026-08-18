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


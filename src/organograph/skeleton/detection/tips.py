"""Crypt-tip selection helpers for skeletonization."""

from __future__ import annotations

from typing import Any

import numpy as np

from organograph.skeleton.detection.common import _coerce_patch
from organograph.skeleton.geometry import as_points

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


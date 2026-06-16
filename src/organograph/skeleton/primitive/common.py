"""Shared helpers for primitive fitting."""

from __future__ import annotations

from typing import Any

import numpy as np

def _residual_summary(residuals: np.ndarray) -> dict[str, float]:
    residuals = np.asarray(residuals, dtype=float)
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "median_abs": float("nan"),
            "max_abs": float("nan"),
            "n_points": 0,
        }
    return {
        "rmse": float(np.sqrt(np.mean(finite**2))),
        "mae": float(np.mean(np.abs(finite))),
        "median_abs": float(np.median(np.abs(finite))),
        "max_abs": float(np.max(np.abs(finite))),
        "n_points": int(finite.size),
    }

def _coerce_indices(value) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.int64)
    arr = np.asarray(list(value) if isinstance(value, set) else value, dtype=np.int64)
    if arr.ndim != 1:
        return np.empty(0, dtype=np.int64)
    return arr

def _mesh_edges(faces) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)

def _region_boundary_vertices(faces, region, n_vertices: int) -> np.ndarray:
    region = _coerce_indices(region)
    if region.size == 0:
        return np.empty(0, dtype=np.int64)
    mask = np.zeros(int(n_vertices), dtype=bool)
    mask[region[(region >= 0) & (region < int(n_vertices))]] = True
    edges = _mesh_edges(faces)
    crossing = edges[mask[edges[:, 0]] != mask[edges[:, 1]]]
    return np.unique(crossing) if crossing.size else np.empty(0, dtype=np.int64)

def _first_detection_value(detection: dict[str, Any], keys: tuple[str, ...]):
    for key in keys:
        value = detection.get(key)
        if value is not None:
            return value
    return None


"""Shared helpers for skeleton detection and graph construction."""

from __future__ import annotations

from typing import Any

import numpy as np

from organograph.skeleton.geometry import as_points, as_vertex_indices, centroid

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


"""Connected mesh support regions used only for crypt radius fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np

from organograph.skeleton.geometry import as_points
from organograph.skeleton.primitive.crypt_geometry import (
    restricted_surface_distance_field,
)


@dataclass(frozen=True)
class CryptRadiusSupportResult:
    """Competitively grown radius-support regions and compact diagnostics."""

    regions: dict[Hashable, np.ndarray]
    diagnostics: dict[Hashable, dict[str, Any]]


def grow_crypt_radius_support_regions(
    vertices,
    faces,
    crypt_regions: dict[Hashable, Any],
    tip_vertex_ids: dict[Hashable, int],
    centerline_lengths: dict[Hashable, float],
    protected_mask,
    *,
    max_distance_factor: float = 1.5,
) -> CryptRadiusSupportResult:
    """Grow connected crypt regions into otherwise unassigned mesh vertices.

    Each original crypt is a fixed seed and blocks traversal by every other
    crypt. Growth may use only unassigned, non-host vertices connected to that
    seed. A restricted tip-geodesic assigns contested vertices to their nearest
    crypt and limits support to ``max_distance_factor * centerline_length``.
    The returned regions are intended for radius measurement only.
    """
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    n_vertices = vertices.shape[0]
    protected = np.asarray(protected_mask, dtype=bool).reshape(-1)
    if protected.size != n_vertices:
        raise ValueError("protected_mask must contain one value per mesh vertex")
    factor = float(max_distance_factor)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("max_distance_factor must be finite and positive")

    keys = list(crypt_regions)
    if not keys:
        return CryptRadiusSupportResult(regions={}, diagnostics={})

    base_regions: dict[Hashable, np.ndarray] = {}
    assigned = np.zeros(n_vertices, dtype=bool)
    for key in keys:
        region = np.unique(np.asarray(crypt_regions[key], dtype=np.int64).reshape(-1))
        region = region[(region >= 0) & (region < n_vertices)]
        base_regions[key] = region
        assigned[region] = True
    unassigned = ~protected & ~assigned

    distances = np.full((len(keys), n_vertices), np.inf, dtype=float)
    limits = np.empty(len(keys), dtype=float)
    for index, key in enumerate(keys):
        region = base_regions[key]
        if region.size == 0:
            limits[index] = 0.0
            continue
        tip = int(tip_vertex_ids[key])
        if tip < 0 or tip >= n_vertices or tip not in set(region.tolist()):
            tip_position = vertices[np.clip(tip, 0, n_vertices - 1)]
            tip = int(region[np.argmin(np.linalg.norm(vertices[region] - tip_position, axis=1))])
        allowed = unassigned.copy()
        allowed[region] = True
        distances[index] = restricted_surface_distance_field(
            vertices,
            faces,
            np.flatnonzero(allowed),
            [tip],
        )
        length = float(centerline_lengths[key])
        limits[index] = factor * length if np.isfinite(length) and length > 0.0 else 0.0

    admissible = distances <= limits[:, None]
    claim_distances = np.where(admissible & unassigned[None, :], distances, np.inf)
    winner = np.argmin(claim_distances, axis=0)
    winning_distance = np.min(claim_distances, axis=0)
    contested = np.sum(np.isfinite(claim_distances), axis=0) > 1

    regions: dict[Hashable, np.ndarray] = {}
    diagnostics: dict[Hashable, dict[str, Any]] = {}
    for index, key in enumerate(keys):
        base = base_regions[key]
        retained_base = base[admissible[index, base]]
        won = np.flatnonzero(
            unassigned
            & np.isfinite(winning_distance)
            & (winner == index)
        )
        expanded = np.unique(np.concatenate([retained_base, won])).astype(np.int64)
        regions[key] = expanded
        diagnostics[key] = {
            "original_vertices": int(base.size),
            "retained_original_vertices": int(retained_base.size),
            "added_vertices": int(won.size),
            "final_vertices": int(expanded.size),
            "contested_vertices_won": int(np.count_nonzero(contested[won])),
            "centerline_length": float(centerline_lengths[key]),
            "max_tip_geodesic_distance": float(limits[index]),
            "max_distance_factor": factor,
        }
    return CryptRadiusSupportResult(regions=regions, diagnostics=diagnostics)


__all__ = [
    "CryptRadiusSupportResult",
    "grow_crypt_radius_support_regions",
]

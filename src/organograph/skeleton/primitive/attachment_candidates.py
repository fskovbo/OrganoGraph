"""Candidate attachment points on fitted body and branch primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import brentq

from organograph.skeleton.detection.attachments import (
    barrier_surface_normal,
    project_points_to_barrier_surface,
)
from organograph.skeleton.primitive.barriers import (
    BarrierPrimitiveFit,
    barrier_primitive_level,
)


@dataclass
class CryptAttachmentCandidate:
    """One deterministic proximal endpoint proposed for a crypt primitive."""

    name: str
    position: np.ndarray
    outward_normal: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def first_tangent_surface_crossing(
    tip,
    tip_to_host_direction,
    host_fit: BarrierPrimitiveFit,
    *,
    reference_attachment=None,
    n_search_samples: int = 192,
) -> np.ndarray | None:
    """Return the first host-surface crossing along a ray from the crypt tip.

    The ray direction is oriented toward ``reference_attachment`` when one is
    supplied. The host primitive is convex in the maintained workflow, so the
    first outside-to-inside level-set crossing is the desired proximal point.
    """
    tip = np.asarray(tip, dtype=float).reshape(3)
    direction = np.asarray(tip_to_host_direction, dtype=float).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12 or not np.all(np.isfinite(direction)):
        return None
    direction = direction / norm
    if reference_attachment is not None:
        toward_reference = np.asarray(reference_attachment, dtype=float).reshape(3) - tip
        if float(np.dot(direction, toward_reference)) < 0.0:
            direction = -direction

    closest = project_points_to_barrier_surface(tip[None, :], host_fit)[0]
    closest_distance = float(np.linalg.norm(closest - tip))
    host_scale = max(float(np.max(np.asarray(host_fit.radii, dtype=float))), 1e-6)
    max_distance = max(3.0 * closest_distance, 2.5 * host_scale)
    distances = np.linspace(0.0, max_distance, max(16, int(n_search_samples)))
    points = tip[None, :] + distances[:, None] * direction[None, :]
    levels = np.asarray(barrier_primitive_level(points, host_fit), dtype=float) - 1.0
    if not np.all(np.isfinite(levels)):
        return None

    # A crypt tip already inside the host does not define an outside-to-inside
    # crossing. The closest-surface candidate remains available in that case.
    if levels[0] <= 0.0:
        return None
    crossing_indices = np.where((levels[:-1] > 0.0) & (levels[1:] <= 0.0))[0]
    if crossing_indices.size == 0:
        return None
    index = int(crossing_indices[0])

    def level_along_ray(distance):
        point = tip + float(distance) * direction
        return float(barrier_primitive_level(point[None, :], host_fit)[0] - 1.0)

    try:
        crossing_distance = brentq(
            level_along_ray,
            float(distances[index]),
            float(distances[index + 1]),
            xtol=1e-10,
            rtol=1e-10,
        )
    except ValueError:
        return None
    return tip + crossing_distance * direction


def crypt_attachment_candidates(
    current_attachment,
    tip,
    tip_to_host_direction,
    host_fit: BarrierPrimitiveFit,
    *,
    deduplication_tolerance_fraction: float = 1e-4,
) -> list[CryptAttachmentCandidate]:
    """Construct current, tip-tangent crossing, and closest-tip candidates."""
    current = np.asarray(current_attachment, dtype=float).reshape(3)
    tip = np.asarray(tip, dtype=float).reshape(3)
    raw = [("current", current, {})]

    crossing = first_tangent_surface_crossing(
        tip,
        tip_to_host_direction,
        host_fit,
        reference_attachment=current,
    )
    if crossing is not None:
        raw.append(
            (
                "tip_tangent_crossing",
                crossing,
                {"ray_direction": np.asarray(tip_to_host_direction, dtype=float)},
            )
        )
    closest = project_points_to_barrier_surface(tip[None, :], host_fit)[0]
    raw.append(("closest_surface_to_tip", closest, {}))

    host_scale = max(float(np.max(np.asarray(host_fit.radii, dtype=float))), 1.0)
    tolerance = max(float(deduplication_tolerance_fraction) * host_scale, 1e-8)
    candidates: list[CryptAttachmentCandidate] = []
    for name, position, metadata in raw:
        position = np.asarray(position, dtype=float).reshape(3)
        duplicate = next(
            (
                candidate
                for candidate in candidates
                if float(np.linalg.norm(candidate.position - position)) <= tolerance
            ),
            None,
        )
        if duplicate is not None:
            duplicate.metadata.setdefault("equivalent_candidates", []).append(name)
            continue
        candidates.append(
            CryptAttachmentCandidate(
                name=name,
                position=position,
                outward_normal=barrier_surface_normal(position, host_fit),
                metadata=dict(metadata),
            )
        )
    return candidates

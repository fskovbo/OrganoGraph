"""Soft-barrier ellipsoid helpers for experimental body/branch ownership.

The fitted ellipsoid is intended as an initial body/branch estimate before the
more expressive primitive layer.  It treats the mesh as a soft barrier: placing
the ellipsoid outside the observed surface is penalized more strongly than
underfilling it.  The resulting relative radial height field can protect body
or branch vertices from later crypt component assignment.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from organograph.skeleton.geometry import as_points, face_areas


@dataclass
class SoftBarrierEllipsoidConfig:
    """Settings for PCA-frame soft-barrier ellipsoid fitting."""

    barrier_weight: float = 100.0
    underfill_weight: float = 0.4
    center_regularization: float = 0.02
    anisotropy_regularization: float = 0.0
    center_shift_limit_frac: float = 0.45
    initial_radius_quantile: float = 0.78
    initial_radius_scale: float = 1.25
    min_radius_span_fraction: float = 0.02
    radius_lower_frac: float = 0.25
    radius_upper_max_frac: float = 1.35
    maxiter: int = 1200
    use_powell_retry: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SoftBarrierEllipsoidFit:
    """Fitted ellipsoid plus diagnostic fields."""

    center: np.ndarray
    axes: np.ndarray
    radii: np.ndarray
    solid_center_of_mass: np.ndarray | None = None
    solid_volume: float | None = None
    center0: np.ndarray | None = None
    center_of_mass_moved_inside: bool = False
    shift_local: np.ndarray | None = None
    initial_radii: np.ndarray | None = None
    success: bool = False
    status: int | None = None
    message: str | None = None
    objective: float | None = None
    nfev: int | None = None
    nit: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_primitive_parameters(self) -> dict[str, Any]:
        return {
            "center": self.center,
            "orientation": self.axes,
            "axis_lengths": self.radii,
            "axis_quantile": self.metadata.get("initial_radius_quantile"),
            "fit_family": "soft_barrier_ellipsoid",
        }

    def to_dict(self) -> dict[str, Any]:
        def convert(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value

        return {
            "center": convert(self.center),
            "axes": convert(self.axes),
            "radii": convert(self.radii),
            "solid_center_of_mass": convert(self.solid_center_of_mass),
            "solid_volume": convert(self.solid_volume),
            "center0": convert(self.center0),
            "center_of_mass_moved_inside": bool(self.center_of_mass_moved_inside),
            "shift_local": convert(self.shift_local),
            "initial_radii": convert(self.initial_radii),
            "success": bool(self.success),
            "status": self.status,
            "message": self.message,
            "objective": convert(self.objective),
            "nfev": self.nfev,
            "nit": self.nit,
            "metadata": convert(self.metadata),
        }


def vertex_areas_from_faces(vertices, faces) -> np.ndarray:
    """Barycentric per-vertex surface areas."""
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    areas = np.zeros(vertices.shape[0], dtype=float)
    if faces.size == 0:
        areas[:] = 1.0
        return areas
    f_areas = face_areas(vertices, faces)
    for i in range(3):
        np.add.at(areas, faces[:, i], f_areas / 3.0)
    return areas


def solid_center_of_mass(vertices, faces, *, fallback_weights=None) -> tuple[np.ndarray, float]:
    """Compute the signed-tetrahedra solid center of mass for a closed mesh."""
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        weights = fallback_weights
        if weights is None:
            return np.mean(vertices, axis=0), 0.0
        return np.average(vertices, weights=np.asarray(weights, dtype=float), axis=0), 0.0

    tri = vertices[faces]
    a = tri[:, 0]
    b = tri[:, 1]
    c = tri[:, 2]
    signed_volume = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    total_volume = float(np.sum(signed_volume))
    if abs(total_volume) <= np.finfo(float).eps:
        weights = fallback_weights
        if weights is None:
            return np.mean(vertices, axis=0), 0.0
        return np.average(vertices, weights=np.asarray(weights, dtype=float), axis=0), 0.0
    center = np.sum(
        signed_volume[:, None] * (a + b + c) / 4.0,
        axis=0,
    ) / total_volume
    return center, abs(total_volume)


def _signed_distance_to_mesh(points, vertices, faces):
    import igl

    points = np.atleast_2d(np.asarray(points, dtype=float))
    return igl.signed_distance(
        points,
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=np.int64),
        igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER,
    )


def inside_sign_calibration(vertices, faces) -> float:
    """Return the signed-distance sign observed at a far outside point."""
    vertices = as_points(vertices)
    center = np.mean(vertices, axis=0)
    span = np.linalg.norm(np.ptp(vertices, axis=0))
    direction = np.array([1.0, 0.37, -0.23])
    direction /= np.linalg.norm(direction)
    far_point = center + 4.0 * span * direction
    s_far, _, _, _ = _signed_distance_to_mesh(far_point, vertices, faces)
    outside_sign = np.sign(float(s_far[0]))
    return 1.0 if outside_sign == 0.0 else float(outside_sign)


def point_is_inside_mesh(point, vertices, faces, *, outside_sign: float | None = None) -> bool:
    if outside_sign is None:
        outside_sign = inside_sign_calibration(vertices, faces)
    s, _, _, _ = _signed_distance_to_mesh(point, vertices, faces)
    return np.sign(float(s[0])) != float(outside_sign)


def move_point_inside_mesh(
    point,
    vertices,
    faces,
    *,
    fallback_hints=None,
    outside_sign: float | None = None,
) -> tuple[np.ndarray, bool]:
    """Project an outside point to the closest surface point and nudge inward."""
    vertices = as_points(vertices)
    point = np.asarray(point, dtype=float)
    if outside_sign is None:
        outside_sign = inside_sign_calibration(vertices, faces)
    if point_is_inside_mesh(point, vertices, faces, outside_sign=outside_sign):
        return point, False

    _, _, closest, _ = _signed_distance_to_mesh(point, vertices, faces)
    closest = np.asarray(closest[0], dtype=float)
    scale = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    hints = [np.mean(vertices, axis=0)]
    if fallback_hints is not None:
        hints.extend(np.asarray(hint, dtype=float) for hint in fallback_hints)
    for hint in hints:
        direction = np.asarray(hint, dtype=float) - closest
        norm = float(np.linalg.norm(direction))
        if norm <= np.finfo(float).eps:
            continue
        direction /= norm
        for eps in (1e-5, 1e-4, 1e-3, 1e-2):
            candidate = closest + eps * scale * direction
            if point_is_inside_mesh(
                candidate,
                vertices,
                faces,
                outside_sign=outside_sign,
            ):
                return candidate, True
    return closest, True


def weighted_pca_frame(points, weights=None, center=None) -> np.ndarray:
    points = as_points(points)
    if center is None:
        center = np.mean(points, axis=0)
    centered = points - np.asarray(center, dtype=float)[None, :]
    if weights is None:
        weights = np.ones(points.shape[0], dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = np.maximum(weights, 0.0)
    if weights.size != points.shape[0]:
        raise ValueError("weights must match number of points")
    cov = (centered * weights[:, None]).T @ centered
    cov /= max(float(np.sum(weights)), np.finfo(float).eps)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    axes = evecs[:, order]
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return axes


def ellipsoid_level(points, center, axes, radii) -> np.ndarray:
    local = (as_points(points) - np.asarray(center, dtype=float)[None, :]) @ np.asarray(axes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    return np.sqrt(np.sum((local / np.maximum(radii[None, :], 1e-12)) ** 2, axis=1))


def project_points_to_ellipsoid(points, center, axes, radii) -> tuple[np.ndarray, np.ndarray]:
    points = as_points(points)
    center = np.asarray(center, dtype=float)
    axes = np.asarray(axes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    local = (points - center[None, :]) @ axes
    level = np.sqrt(np.sum((local / np.maximum(radii[None, :], 1e-12)) ** 2, axis=1))
    level = np.maximum(level, np.finfo(float).eps)
    projected_local = local / level[:, None]
    return center[None, :] + projected_local @ axes.T, level


def ellipsoid_vertices_like_mesh(vertices, fit: SoftBarrierEllipsoidFit) -> np.ndarray:
    projected, _ = project_points_to_ellipsoid(
        vertices,
        fit.center,
        fit.axes,
        fit.radii,
    )
    return projected


def sampled_vertex_indices(
    n_vertices: int,
    *,
    sample_fraction: float = 1.0,
    min_vertices: int = 4,
    random_seed: int | None = 0,
) -> np.ndarray:
    """Return deterministic random vertex indices for lightweight fitting."""
    n_vertices = int(n_vertices)
    if n_vertices <= 0:
        return np.empty(0, dtype=np.int64)
    sample_fraction = float(sample_fraction)
    if not np.isfinite(sample_fraction) or sample_fraction <= 0.0:
        sample_fraction = 1.0
    n_sample = int(np.ceil(sample_fraction * n_vertices))
    n_sample = min(n_vertices, max(int(min_vertices), n_sample))
    if n_sample >= n_vertices:
        return np.arange(n_vertices, dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    return np.sort(rng.choice(n_vertices, size=n_sample, replace=False)).astype(np.int64)


def fit_soft_barrier_ellipsoid_sampled(
    vertices,
    faces=None,
    *,
    sample_fraction: float = 1.0,
    min_vertices: int = 4,
    random_seed: int | None = 0,
    weights=None,
    config: SoftBarrierEllipsoidConfig | dict[str, Any] | None = None,
    require_inside_center: bool = True,
    center0=None,
    axes=None,
) -> SoftBarrierEllipsoidFit:
    """Fit a soft-barrier ellipsoid on a sampled vertex cloud.

    When faces are provided, the solid center of mass and inside correction are
    still computed from the full mesh.  The optimization itself then uses only
    the sampled vertices.  This keeps the body estimate anchored to the solid
    mesh while making the barrier fit cheaper.
    """
    vertices = as_points(vertices)
    sample_idx = sampled_vertex_indices(
        vertices.shape[0],
        sample_fraction=sample_fraction,
        min_vertices=min_vertices,
        random_seed=random_seed,
    )
    if sample_idx.size == vertices.shape[0]:
        fit = fit_soft_barrier_ellipsoid(
            vertices,
            faces,
            weights=weights,
            config=config,
            require_inside_center=require_inside_center,
            center0=center0,
            axes=axes,
        )
        fit.metadata.update(
            {
                "sample_fraction": 1.0,
                "sample_n_vertices": int(vertices.shape[0]),
                "sample_random_seed": random_seed,
            }
        )
        return fit

    sample_weights = None
    if weights is not None:
        sample_weights = np.asarray(weights, dtype=float).reshape(-1)[sample_idx]
    elif faces is not None:
        sample_weights = vertex_areas_from_faces(vertices, faces)[sample_idx]

    solid_com = None
    solid_volume = None
    moved_inside = False
    if center0 is None and faces is not None:
        solid_com, solid_volume = solid_center_of_mass(
            vertices,
            faces,
            fallback_weights=weights,
        )
        if require_inside_center:
            center0, moved_inside = move_point_inside_mesh(
                solid_com,
                vertices,
                faces,
                fallback_hints=[np.mean(vertices, axis=0)],
            )
        else:
            center0 = solid_com

    fit = fit_soft_barrier_ellipsoid(
        vertices[sample_idx],
        faces=None,
        weights=sample_weights,
        config=config,
        require_inside_center=False,
        center0=center0,
        axes=axes,
    )
    if solid_com is not None:
        fit.solid_center_of_mass = solid_com
        fit.solid_volume = solid_volume
        fit.center_of_mass_moved_inside = moved_inside
    fit.metadata.update(
        {
            "sample_fraction": float(sample_idx.size / max(vertices.shape[0], 1)),
            "requested_sample_fraction": float(sample_fraction),
            "sample_n_vertices": int(sample_idx.size),
            "full_n_vertices": int(vertices.shape[0]),
            "sample_random_seed": random_seed,
        }
    )
    return fit


def relative_height_field(vertices, fit: SoftBarrierEllipsoidFit) -> dict[str, np.ndarray]:
    """Return relative radial level and signed height over the ellipsoid."""
    projected, level = project_points_to_ellipsoid(
        vertices,
        fit.center,
        fit.axes,
        fit.radii,
    )
    vertices = as_points(vertices)
    signed_height = np.sign(level - 1.0) * np.linalg.norm(vertices - projected, axis=1)
    return {
        "level": level,
        "signed_height": signed_height,
        "projected_points": projected,
    }


def fit_soft_barrier_ellipsoid(
    vertices,
    faces=None,
    *,
    weights=None,
    config: SoftBarrierEllipsoidConfig | dict[str, Any] | None = None,
    require_inside_center: bool = True,
    center0=None,
    axes=None,
) -> SoftBarrierEllipsoidFit:
    """Fit a soft-barrier ellipsoid to mesh vertices or a point component."""
    vertices = as_points(vertices)
    if vertices.shape[0] < 4:
        raise ValueError("At least four points are required for ellipsoid fitting")
    if not isinstance(config, SoftBarrierEllipsoidConfig):
        config = SoftBarrierEllipsoidConfig(**dict(config or {}))
    if weights is None:
        if faces is not None:
            weights = vertex_areas_from_faces(vertices, faces)
        else:
            weights = np.ones(vertices.shape[0], dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if weights.size != vertices.shape[0]:
        raise ValueError("weights must match vertices")

    solid_volume = None
    solid_com = None
    moved_inside = False
    fallback_hints = [np.average(vertices, weights=np.maximum(weights, 1e-12), axis=0)]
    if center0 is None:
        if faces is not None:
            solid_com, solid_volume = solid_center_of_mass(
                vertices,
                faces,
                fallback_weights=weights,
            )
            if require_inside_center:
                center0, moved_inside = move_point_inside_mesh(
                    solid_com,
                    vertices,
                    faces,
                    fallback_hints=fallback_hints,
                )
            else:
                center0 = solid_com
        else:
            center0 = fallback_hints[0]
            solid_volume = 0.0
    else:
        center0 = np.asarray(center0, dtype=float)

    if axes is None:
        axes = weighted_pca_frame(vertices, weights, center0)
    else:
        axes = np.asarray(axes, dtype=float)
    local0 = (vertices - center0[None, :]) @ axes
    q = float(config.initial_radius_quantile)
    if not (0.0 < q <= 1.0):
        q = 0.78
    initial_radii = np.quantile(np.abs(local0), q, axis=0) * float(config.initial_radius_scale)
    max_radii = np.max(np.abs(local0), axis=0)
    span = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), np.finfo(float).eps)
    min_radius = float(config.min_radius_span_fraction) * span
    initial_radii = np.maximum(initial_radii, min_radius)
    max_radii = np.maximum(max_radii, initial_radii)
    area_weights = weights / max(float(np.sum(weights)), np.finfo(float).eps)
    shift_limit = float(config.center_shift_limit_frac) * initial_radii

    def unpack(params):
        shift_local = params[:3]
        radii = np.exp(params[3:])
        center = center0 + shift_local @ axes.T
        return center, radii, shift_local

    def objective(params):
        center, radii, shift_local = unpack(params)
        level = ellipsoid_level(vertices, center, axes, radii)
        residual = level - 1.0
        residual_weights = np.where(
            residual < 0.0,
            float(config.barrier_weight),
            float(config.underfill_weight),
        )
        data_loss = np.sum(area_weights * residual_weights * residual**2)
        data_loss /= max(float(np.sum(area_weights * residual_weights)), np.finfo(float).eps)
        shift_loss = float(config.center_regularization) * np.sum(
            (shift_local / np.maximum(initial_radii, np.finfo(float).eps)) ** 2
        )
        log_radii = np.log(np.maximum(radii, np.finfo(float).eps))
        anisotropy_loss = float(config.anisotropy_regularization) * float(
            np.mean((log_radii - np.mean(log_radii)) ** 2)
        )
        return float(data_loss + shift_loss + anisotropy_loss)

    x0 = np.r_[np.zeros(3), np.log(initial_radii)]
    bounds = [(-shift_limit[i], shift_limit[i]) for i in range(3)]
    bounds += [
        (
            np.log(float(config.radius_lower_frac) * initial_radii[i]),
            np.log(float(config.radius_upper_max_frac) * max_radii[i]),
        )
        for i in range(3)
    ]
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(config.maxiter)},
    )
    if bool(config.use_powell_retry) and not result.success:
        retry = minimize(
            objective,
            result.x,
            method="Powell",
            bounds=bounds,
            options={"maxiter": int(config.maxiter), "xtol": 1e-5, "ftol": 1e-6},
        )
        if retry.success or retry.fun <= result.fun * (1.0 + 1e-5):
            result = retry
    center, radii, shift_local = unpack(result.x)
    return SoftBarrierEllipsoidFit(
        center=center,
        axes=axes,
        radii=radii,
        solid_center_of_mass=solid_com,
        solid_volume=solid_volume,
        center0=center0,
        center_of_mass_moved_inside=moved_inside,
        shift_local=shift_local,
        initial_radii=initial_radii,
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        objective=float(result.fun),
        nfev=getattr(result, "nfev", None),
        nit=getattr(result, "nit", None),
        metadata={
            "fit_method": "soft_barrier_ellipsoid",
            "initial_radius_quantile": q,
            "anisotropy_regularization": float(config.anisotropy_regularization),
            "n_points": int(vertices.shape[0]),
            "has_faces": faces is not None,
            **dict(config.metadata),
        },
    )


def villus_mask_from_ellipsoid(
    vertices,
    fit: SoftBarrierEllipsoidFit,
    *,
    relative_height_threshold: float = 1.05,
) -> np.ndarray:
    """Return vertices whose radial level is within the ellipsoid threshold."""
    level = relative_height_field(vertices, fit)["level"]
    return np.isfinite(level) & (level <= float(relative_height_threshold))


def protect_patches_from_mask(
    patches,
    protected_mask,
    *,
    min_vertices: int = 1,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Remove protected vertices from candidate patches.

    This is used by the experimental barrier-ellipsoid workflow to keep the
    body estimate from being assigned to crypt candidates before later
    refinement and neckline computations.
    """
    protected = np.asarray(protected_mask, dtype=bool).reshape(-1)
    filtered = []
    diagnostics = []
    for i, patch in enumerate(patches):
        original = _coerce_indices(patch)
        valid = original[(original >= 0) & (original < protected.size)]
        kept = np.unique(valid[~protected[valid]]).astype(np.int64)
        keep_patch = bool(kept.size >= int(min_vertices))
        diagnostics.append(
            {
                "patch_index": int(i),
                "original_size": int(original.size),
                "filtered_size": int(kept.size),
                "removed_size": int(original.size - kept.size),
                "kept": keep_patch,
            }
        )
        if keep_patch:
            filtered.append(kept)
    return filtered, diagnostics


_REGION_KEYS = (
    "crypt_vertices",
    "attachment_region_vertices",
    "neck_region_vertices",
    "neck_side_vertices",
    "root_region_vertices",
)


def _coerce_indices(value) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.int64)

    flat = []

    def collect(item):
        if item is None:
            return
        if isinstance(item, dict):
            for subitem in item.values():
                collect(subitem)
            return
        if isinstance(item, (set, list, tuple)):
            for subitem in item:
                collect(subitem)
            return
        arr = np.asarray(item)
        if arr.ndim == 0:
            flat.append(item)
        else:
            for subitem in arr.reshape(-1):
                collect(subitem)

    collect(value)
    if not flat:
        return np.empty(0, dtype=np.int64)
    return np.asarray(flat, dtype=np.int64).reshape(-1)


def _filter_region(value, allowed_mask) -> list[int] | None:
    if value is None:
        return None
    indices = _coerce_indices(value)
    keep = indices[(indices >= 0) & (indices < allowed_mask.size)]
    keep = keep[allowed_mask[keep]]
    return np.unique(keep).astype(np.int64).tolist()


def protect_detection_regions_from_mask(
    detections: list[dict[str, Any]],
    protected_mask,
    *,
    recursive: bool = True,
    metadata_key: str = "protected_region_filter",
) -> list[dict[str, Any]]:
    """Remove protected vertices from crypt-side region keys in detections.

    This is an experimental adapter for barrier-ellipsoid ownership tests.  It
    intentionally does not alter distance fields or already-computed neck
    positions; it only prevents component extraction and graph construction
    from treating protected body/branch vertices as crypt mesh.
    """
    protected = np.asarray(protected_mask, dtype=bool).reshape(-1)
    allowed = ~protected

    def convert(det):
        out = copy.deepcopy(det)
        removed = {}
        for key in _REGION_KEYS:
            if key not in out:
                continue
            original = _coerce_indices(out.get(key))
            filtered = _filter_region(original, allowed)
            out[key] = filtered
            removed[key] = int(original.size - len(filtered))
        meta = dict(out.get("metadata", {}))
        meta[metadata_key] = {
            "n_protected_vertices": int(np.count_nonzero(protected)),
            "removed_by_key": removed,
        }
        out["metadata"] = meta
        if recursive and out.get("daughters"):
            out["daughters"] = [convert(daughter) for daughter in out["daughters"]]
        return out

    return [convert(det) for det in detections]

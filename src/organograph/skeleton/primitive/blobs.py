"""Blob primitive fitting for body and branch components."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from organograph.skeleton.geometry import as_points, centroid
from organograph.skeleton.primitive.common import _residual_summary
from organograph.skeleton.primitives import PrimitiveFit

def fit_ellipsoid_to_points(
    points,
    *,
    axis_quantile: float = 0.98,
    min_axis_length: float = 1e-6,
    metadata: dict[str, Any] | None = None,
) -> PrimitiveFit:
    """Fit a coarse PCA ellipsoid to component points.

    This is not a nonlinear algebraic ellipsoid fit.  The center is the point
    centroid, axes are PCA directions, and axis lengths are robust projected
    extents.  It is meant as a stable first blob primitive for body/branch
    components and can later be replaced by a superellipsoid or implicit blob.
    """
    pts = as_points(points)
    if pts.shape[0] < 3:
        raise ValueError("At least three points are required to fit an ellipsoid")

    center = centroid(pts)
    centered = pts - center[None, :]
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    orientation = eigvecs[:, order]
    if np.linalg.det(orientation) < 0.0:
        orientation[:, -1] *= -1.0

    coords = centered @ orientation
    q = float(axis_quantile)
    if not (0.0 < q <= 1.0):
        q = 0.98
    axes = np.quantile(np.abs(coords), q, axis=0)
    axes = np.maximum(axes, float(min_axis_length))

    normalized_radius = np.sqrt(np.sum((coords / axes[None, :]) ** 2, axis=1))
    residuals = normalized_radius - 1.0
    summary = _residual_summary(residuals)
    return PrimitiveFit(
        primitive_type="ellipsoid",
        parameters={
            "center": center,
            "orientation": orientation,
            "axis_lengths": axes,
            "axis_quantile": q,
            "superellipsoid_exponents": None,
        },
        fit_error=summary["rmse"],
        residuals=summary,
        metadata={
            "fit_method": "pca_projected_extent",
            "n_points": int(pts.shape[0]),
            "future_primitive_family": "superellipsoid_or_implicit_blob",
            **dict(metadata or {}),
        },
    )

def _asymmetric_superellipsoid_radius(
    coords: np.ndarray,
    negative_axes: np.ndarray,
    positive_axes: np.ndarray,
    epsilon_1: float,
    epsilon_2: float,
) -> np.ndarray:
    axes = np.where(coords >= 0.0, positive_axes[None, :], negative_axes[None, :])
    scaled = np.abs(coords) / np.maximum(axes, 1e-12)
    xy = (
        scaled[:, 0] ** (2.0 / epsilon_2)
        + scaled[:, 1] ** (2.0 / epsilon_2)
    ) ** (epsilon_2 / epsilon_1)
    return (xy + scaled[:, 2] ** (2.0 / epsilon_1)) ** (epsilon_1 / 2.0)

def fit_asymmetric_superellipsoid_to_points(
    points,
    *,
    axis_quantile: float = 0.98,
    exponent_bounds: tuple[float, float] = (0.3, 2.0),
    axis_regularization: float = 0.04,
    exponent_regularization: float = 0.02,
    min_axis_length: float = 1e-6,
    metadata: dict[str, Any] | None = None,
) -> PrimitiveFit:
    """Fit a PCA-aligned superellipsoid with six directional semiaxes.

    The orientation and center use the same stable PCA initialization as the
    ellipsoid fitter. Positive and negative extents are independent along each
    local axis, while two bounded exponents control axial and equatorial
    roundness. Regularization keeps the compact primitive close to robust
    directional extents instead of chasing individual mesh irregularities.
    """
    pts = as_points(points)
    if pts.shape[0] < 8:
        raise ValueError(
            "At least eight points are required to fit an asymmetric superellipsoid"
        )
    center = centroid(pts)
    centered = pts - center[None, :]
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    orientation = eigvecs[:, order]
    if np.linalg.det(orientation) < 0.0:
        orientation[:, -1] *= -1.0
    coords = centered @ orientation

    q = float(axis_quantile)
    if not (0.0 < q <= 1.0):
        q = 0.98
    negative = np.empty(3, dtype=float)
    positive = np.empty(3, dtype=float)
    for axis in range(3):
        neg_values = -coords[coords[:, axis] < 0.0, axis]
        pos_values = coords[coords[:, axis] >= 0.0, axis]
        fallback = float(np.quantile(np.abs(coords[:, axis]), q))
        negative[axis] = (
            float(np.quantile(neg_values, q)) if neg_values.size else fallback
        )
        positive[axis] = (
            float(np.quantile(pos_values, q)) if pos_values.size else fallback
        )
    negative = np.maximum(negative, float(min_axis_length))
    positive = np.maximum(positive, float(min_axis_length))
    initial_axes = np.concatenate([negative, positive])
    exponent_lo, exponent_hi = map(float, exponent_bounds)
    if not (0.0 < exponent_lo < exponent_hi):
        raise ValueError("exponent_bounds must satisfy 0 < lower < upper")

    def residual(parameters):
        neg = parameters[:3]
        pos = parameters[3:6]
        epsilon_1, epsilon_2 = parameters[6:8]
        surface = _asymmetric_superellipsoid_radius(
            coords,
            neg,
            pos,
            epsilon_1,
            epsilon_2,
        ) - 1.0
        axis_penalty = np.sqrt(max(float(axis_regularization), 0.0)) * (
            parameters[:6] / initial_axes - 1.0
        )
        exponent_penalty = np.sqrt(
            max(float(exponent_regularization), 0.0)
        ) * (parameters[6:8] - 1.0)
        return np.concatenate([surface, axis_penalty, exponent_penalty])

    lower_axes = np.maximum(0.35 * initial_axes, float(min_axis_length))
    upper_axes = np.maximum(2.5 * initial_axes, lower_axes + float(min_axis_length))
    initial = np.concatenate([initial_axes, [1.0, 1.0]])
    optimized = least_squares(
        residual,
        initial,
        bounds=(
            np.concatenate([lower_axes, [exponent_lo, exponent_lo]]),
            np.concatenate([upper_axes, [exponent_hi, exponent_hi]]),
        ),
        loss="soft_l1",
    ).x
    negative = optimized[:3]
    positive = optimized[3:6]
    epsilon_1, epsilon_2 = map(float, optimized[6:8])
    surface_residuals = _asymmetric_superellipsoid_radius(
        coords,
        negative,
        positive,
        epsilon_1,
        epsilon_2,
    ) - 1.0
    summary = _residual_summary(surface_residuals)
    return PrimitiveFit(
        primitive_type="asymmetric_superellipsoid",
        parameters={
            "center": center,
            "orientation": orientation,
            "axis_lengths_negative": negative,
            "axis_lengths_positive": positive,
            "epsilon_1": epsilon_1,
            "epsilon_2": epsilon_2,
            "axis_quantile": q,
        },
        fit_error=summary["rmse"],
        residuals=summary,
        derived_parameters={
            "axis_asymmetry_ratios": positive / np.maximum(negative, 1e-12),
            "mean_axis_lengths": 0.5 * (negative + positive),
        },
        metadata={
            "fit_method": "pca_frame_directional_axes_bounded_superellipsoid",
            "n_points": int(pts.shape[0]),
            "axis_regularization": float(axis_regularization),
            "exponent_regularization": float(exponent_regularization),
            **dict(metadata or {}),
        },
    )

def fit_blob_primitive_to_points(
    points,
    *,
    primitive_type: str = "ellipsoid",
    **kwargs,
) -> PrimitiveFit:
    """Fit a coarse blob primitive to body or branch component points."""
    primitive_type = str(primitive_type).lower()
    if primitive_type in {"ellipsoid", "superellipsoid_placeholder"}:
        fit = fit_ellipsoid_to_points(points, **kwargs)
        if primitive_type == "superellipsoid_placeholder":
            fit.primitive_type = primitive_type
            fit.metadata["base_fit"] = "ellipsoid"
        return fit
    if primitive_type == "asymmetric_superellipsoid":
        return fit_asymmetric_superellipsoid_to_points(points, **kwargs)
    raise ValueError(
        "Blob primitive_type must be 'ellipsoid' or "
        "'asymmetric_superellipsoid'"
    )


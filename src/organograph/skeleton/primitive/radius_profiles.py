"""Fixed-width longitudinal radius profiles for crypt tube primitives.

The maintained representation fits positive radii on a shared normalized
arc-length grid. A shape-preserving interpolation of squared radius connects
the controls and closes deterministically to zero at the crypt tip. This gives
the VAE a small, fixed-width vector while allowing broad, tapered, and locally
constricted crypts without fitting optional topology-specific landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from organograph.skeleton.primitive_geometry import (
    DEFAULT_CRYPT_RADIUS_CONTROL_S,
    fixed_grid_tube_radius,
)


@dataclass(frozen=True)
class RadiusProfileObservations:
    """Transverse-radius samples in normalized centerline coordinates."""

    s: np.ndarray
    radii: np.ndarray
    counts: np.ndarray
    bin_edges: np.ndarray
    n_input_points: int
    n_excluded_tip_points: int
    weights: np.ndarray | None = None
    section_s: np.ndarray | None = None
    section_mean_radii: np.ndarray | None = None
    section_min_radii: np.ndarray | None = None


@dataclass(frozen=True)
class RadiusProfileFitResult:
    """Fitted fixed-grid radii and their diagnostics."""

    control_s: np.ndarray
    control_radii: np.ndarray
    observations: RadiusProfileObservations
    diagnostics: dict[str, Any]

    @property
    def s_taper(self) -> float:
        return float(self.control_s[-1])

    @property
    def volume_center_s(self) -> float:
        return fitted_radius_volume_center(
            control_s=self.control_s,
            control_radii=self.control_radii,
        )

def fitted_radius_volume_center(
    *,
    control_s=None,
    control_radii=None,
    n_samples: int = 1025,
    **legacy_parameters,
) -> float:
    """Return the volume-centroid coordinate of a fitted radius profile."""
    s = np.linspace(0.0, 1.0, max(65, int(n_samples)))
    if control_s is not None and control_radii is not None:
        radii = fixed_grid_tube_radius(s, control_s, control_radii)
        fallback = float(np.asarray(control_s, dtype=float)[len(control_s) // 2])
    else:
        # Old exports remain readable, but no maintained fitter produces this
        # landmark representation.
        from organograph.skeleton.primitive_geometry import capped_tube_radius

        radii = capped_tube_radius(
            s,
            legacy_parameters["r_attachment"],
            legacy_parameters["r_center"],
            legacy_parameters["r_distal"],
            center_s=legacy_parameters["s_center"],
            taper_start=legacy_parameters["s_taper"],
            r_constriction=legacy_parameters.get("r_constriction"),
            constriction_s=legacy_parameters.get("s_constriction"),
        )
        fallback = float(legacy_parameters["s_center"])
    area = np.maximum(np.asarray(radii, dtype=float), 0.0) ** 2
    mass = float(np.trapezoid(area, s))
    if not np.isfinite(mass) or mass <= 1e-12:
        return fallback
    return float(np.trapezoid(s * area, s) / mass)


def estimate_equal_arclength_radius_profile(
    distances,
    s,
    *,
    n_bins: int = 20,
    min_points_per_bin: int = 2,
    radius_quantile: float = 0.5,
    tip_projection_tolerance: float = 1e-6,
) -> RadiusProfileObservations:
    """Aggregate projected point radii into normalized arc-length bins."""
    distances = np.asarray(distances, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    if distances.size != s.size:
        raise ValueError("distances and s must have matching lengths")
    finite = np.isfinite(distances) & np.isfinite(s) & (distances >= 0.0)
    tip_clamped = finite & (s >= 1.0 - max(float(tip_projection_tolerance), 0.0))
    valid = finite & ~tip_clamped
    distances = distances[valid]
    s = np.clip(s[valid], 0.0, 1.0)
    n_bins = max(4, int(n_bins))
    minimum = max(1, int(min_points_per_bin))
    quantile = float(radius_quantile)
    if not 0.0 < quantile <= 1.0:
        quantile = 0.5
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1)
    observed_s = []
    observed_radii = []
    counts = []
    for index in range(n_bins):
        selected = indices == index
        count = int(np.sum(selected))
        if count < minimum:
            continue
        observed_s.append(float(np.median(s[selected])))
        observed_radii.append(float(np.quantile(distances[selected], quantile)))
        counts.append(count)
    observed_radii_array = np.asarray(observed_radii, dtype=float)
    return RadiusProfileObservations(
        s=np.asarray(observed_s, dtype=float),
        radii=observed_radii_array,
        counts=np.asarray(counts, dtype=np.int64),
        bin_edges=edges,
        n_input_points=int(np.sum(finite)),
        n_excluded_tip_points=int(np.sum(tip_clamped)),
        weights=np.ones(len(observed_s), dtype=float),
        section_s=np.asarray(observed_s, dtype=float),
        section_mean_radii=observed_radii_array,
        section_min_radii=observed_radii_array,
    )


def _normalized_observation_weights(observations: RadiusProfileObservations):
    if observations.weights is None:
        return np.ones(observations.radii.size, dtype=float)
    weights = np.asarray(observations.weights, dtype=float).reshape(-1)
    if weights.size != observations.radii.size:
        raise ValueError("observation weights must match observed radii")
    valid = np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return np.ones(observations.radii.size, dtype=float)
    return np.where(valid, weights, 0.0)


def _initial_control_radii(
    observations: RadiusProfileObservations,
    control_s: np.ndarray,
    fallback_radius: float,
) -> np.ndarray:
    observed_s = np.asarray(observations.s, dtype=float)
    observed_radii = np.asarray(observations.radii, dtype=float)
    valid = np.isfinite(observed_s) & np.isfinite(observed_radii) & (observed_radii > 0.0)
    observed_s = observed_s[valid]
    observed_radii = observed_radii[valid]
    if observed_s.size == 0:
        return np.full(control_s.size, fallback_radius, dtype=float)
    order = np.argsort(observed_s)
    observed_s = observed_s[order]
    observed_radii = observed_radii[order]
    unique_s, inverse = np.unique(observed_s, return_inverse=True)
    aggregate = np.asarray(
        [np.median(observed_radii[inverse == index]) for index in range(unique_s.size)],
        dtype=float,
    )
    return np.maximum(
        np.interp(control_s, unique_s, aggregate, left=aggregate[0], right=aggregate[-1]),
        1e-8,
    )


def fit_fixed_grid_radius_profile(
    distances,
    s,
    *,
    radius_control_s=None,
    fixed_taper_position: float = 0.85,
    radius_profile_smoothness_weight: float = 0.05,
    radius_quantile: float = 0.5,
    optimize: bool = True,
    n_bins: int = 20,
    min_points_per_bin: int = 2,
    min_supported_bins: int = 6,
    outside_volume_weight: float = 2.0,
    tip_projection_tolerance: float = 1e-6,
    observations_override: RadiusProfileObservations | None = None,
    **_deprecated_options,
) -> RadiusProfileFitResult:
    """Fit fixed-grid radii with asymmetric area error and smoothness."""
    distances = np.asarray(distances, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    if distances.size != s.size:
        raise ValueError("distances and s must have matching lengths")
    if radius_control_s is None:
        controls = DEFAULT_CRYPT_RADIUS_CONTROL_S.copy()
        controls[1:] *= float(fixed_taper_position) / controls[-1]
    else:
        controls = np.asarray(radius_control_s, dtype=float).reshape(-1)
    if controls.size < 3 or controls[0] != 0.0 or not np.all(np.diff(controls) > 0.0):
        raise ValueError("radius_control_s must start at 0 and strictly increase")
    if not np.isclose(controls[-1], float(fixed_taper_position), atol=1e-10):
        raise ValueError("the final radius control must equal fixed_taper_position")
    if not 0.0 < controls[-1] < 1.0:
        raise ValueError("fixed_taper_position must lie in (0, 1)")

    q = float(radius_quantile) if 0.0 < float(radius_quantile) <= 1.0 else 0.5
    observations = observations_override or estimate_equal_arclength_radius_profile(
        distances,
        s,
        n_bins=n_bins,
        min_points_per_bin=min_points_per_bin,
        radius_quantile=q,
        tip_projection_tolerance=tip_projection_tolerance,
    )
    finite_distances = distances[np.isfinite(distances) & (distances > 0.0)]
    fallback_radius = max(
        float(np.median(finite_distances)) if finite_distances.size else 1e-8,
        1e-8,
    )
    initial = _initial_control_radii(observations, controls, fallback_radius)
    fitted = initial.copy()
    weights = _normalized_observation_weights(observations)
    weight_sum = max(float(np.sum(weights)), 1e-12)
    normalized_weights = weights / weight_sum
    observed_area = np.asarray(observations.radii, dtype=float) ** 2
    scale_area = max(
        float(np.sum(normalized_weights * observed_area)),
        fallback_radius**2,
        1e-12,
    )
    section_s = observations.section_s
    n_supported = int(np.asarray(section_s if section_s is not None else observations.s).size)
    smoothness_weight = max(float(radius_profile_smoothness_weight), 0.0)
    outside_weight = max(float(outside_volume_weight), 1.0)
    optimization = {
        "attempted": bool(optimize),
        "success": False,
        "message": "interpolated_fixed_grid_initialization",
        "nfev": 0,
        "outside_volume_weight": outside_weight,
        "smoothness_weight": smoothness_weight,
        "radius_control_s": controls.copy(),
        "s_taper_source": "fixed_visualization_parameter",
        "n_supported_bins": n_supported,
    }

    if optimize and n_supported >= max(3, int(min_supported_bins)):
        maximum_radius = max(float(np.max(observations.radii)), fallback_radius)

        def residual_function(log_radii):
            radii = np.exp(log_radii)
            predicted = fixed_grid_tube_radius(observations.s, controls, radii)
            area_error = (predicted**2 - observed_area) / scale_area
            asymmetric = np.where(area_error > 0.0, np.sqrt(outside_weight), 1.0)
            data_residual = np.sqrt(normalized_weights) * asymmetric * area_error
            if smoothness_weight <= 0.0 or radii.size < 3:
                return data_residual
            smoothness = np.diff(log_radii, n=2)
            return np.concatenate(
                [data_residual, np.sqrt(smoothness_weight) * smoothness]
            )

        try:
            result = least_squares(
                residual_function,
                np.log(initial),
                bounds=(
                    np.full(controls.size, np.log(1e-8)),
                    np.full(controls.size, np.log(10.0 * maximum_radius)),
                ),
                loss="soft_l1",
                max_nfev=500,
            )
            if result.success and np.all(np.isfinite(result.x)):
                fitted = np.exp(result.x)
                optimization.update(
                    success=True,
                    message=str(result.message),
                    nfev=int(result.nfev),
                    cost=float(result.cost),
                )
            else:
                optimization.update(message=str(result.message), nfev=int(result.nfev))
        except (FloatingPointError, ValueError) as exc:
            optimization["message"] = f"fallback_after_error: {exc}"
    elif optimize:
        optimization["message"] = "insufficient_supported_arc_length_bins"

    predicted = fixed_grid_tube_radius(observations.s, controls, fitted)
    area_delta = predicted**2 - observed_area
    outside_proxy = float(np.sum(normalized_weights * np.maximum(area_delta, 0.0)))
    missing_proxy = float(np.sum(normalized_weights * np.maximum(-area_delta, 0.0)))
    normalization = max(float(np.sum(normalized_weights * observed_area)), 1e-12)
    optimization.update(
        initial_control_radii=initial.copy(),
        fitted_control_radii=fitted.copy(),
        n_input_points=int(observations.n_input_points),
        n_excluded_tip_points=int(observations.n_excluded_tip_points),
        outside_volume_proxy=outside_proxy / normalization,
        missing_volume_proxy=missing_proxy / normalization,
        candidate_score=(outside_weight * outside_proxy + missing_proxy) / normalization,
        observation_rmse=(
            float(np.sqrt(np.sum(normalized_weights * (observations.radii - predicted) ** 2)))
            if observations.radii.size
            else float("nan")
        ),
    )
    return RadiusProfileFitResult(
        control_s=controls,
        control_radii=np.asarray(fitted, dtype=float),
        observations=observations,
        diagnostics=optimization,
    )

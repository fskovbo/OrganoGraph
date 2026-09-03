"""Interpretable longitudinal radius-profile fitting for crypt tubes.

The fitted degrees of freedom are radii at biological skeleton landmarks. The
landmark positions come from the fitted crypt geometry: attachment, optional
constriction, iterated tube-volume center, fixed distal cap onset, and tip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from organograph.skeleton.primitive_geometry import capped_tube_radius


@dataclass(frozen=True)
class RadiusProfileObservations:
    """Robust cross-sectional radii sampled at equal arc-length intervals."""

    s: np.ndarray
    radii: np.ndarray
    counts: np.ndarray
    bin_edges: np.ndarray
    n_input_points: int
    n_excluded_tip_points: int


@dataclass(frozen=True)
class RadiusProfileFitResult:
    """Fitted radii and fixed/derived longitudinal landmark positions."""

    r_attachment: float
    r_center: float
    r_distal: float
    s_center: float
    s_taper: float
    r_constriction: float | None
    s_constriction: float | None
    observations: RadiusProfileObservations
    diagnostics: dict[str, Any]

    @property
    def r_neck(self) -> float:
        """Deprecated name for ``r_attachment``."""
        return self.r_attachment

    @property
    def r_body(self) -> float:
        """Deprecated name for the crypt-center radius."""
        return self.r_center

    @property
    def s_body(self) -> float:
        """Deprecated name for ``s_center``."""
        return self.s_center


def fitted_radius_volume_center(
    *,
    r_attachment: float,
    r_center: float,
    r_distal: float,
    s_center: float,
    s_taper: float,
    r_constriction: float | None = None,
    s_constriction: float | None = None,
    n_samples: int = 1025,
) -> float:
    """Return the volume centroid coordinate of a fitted tube profile.

    The tube uses normalized arc length, so its differential volume is
    proportional to ``r(s)^2 ds``. The common factors ``pi`` and centerline
    length cancel in the centroid ratio.
    """
    s = np.linspace(0.0, 1.0, max(65, int(n_samples)))
    radii = capped_tube_radius(
        s,
        r_attachment,
        r_center,
        r_distal,
        center_s=s_center,
        taper_start=s_taper,
        r_constriction=r_constriction,
        constriction_s=s_constriction,
    )
    area = np.maximum(np.asarray(radii, dtype=float), 0.0) ** 2
    mass = float(np.trapezoid(area, s))
    if not np.isfinite(mass) or mass <= 1e-12:
        return float(s_center)
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
    """Aggregate projected mesh radii into equally weighted arc-length bins."""
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
    return RadiusProfileObservations(
        s=np.asarray(observed_s, dtype=float),
        radii=np.asarray(observed_radii, dtype=float),
        counts=np.asarray(counts, dtype=np.int64),
        bin_edges=edges,
        n_input_points=int(np.sum(finite)),
        n_excluded_tip_points=int(np.sum(tip_clamped)),
    )


def _radius_near(
    distances: np.ndarray,
    s: np.ndarray,
    observations: RadiusProfileObservations,
    position: float,
    *,
    half_width: float,
    quantile: float,
) -> float:
    selected = np.abs(observations.s - float(position)) <= float(half_width)
    if np.any(selected):
        return float(np.median(observations.radii[selected]))
    selected_raw = np.abs(s - float(position)) <= float(half_width)
    selected_raw &= np.isfinite(distances)
    if np.any(selected_raw):
        return float(np.quantile(distances[selected_raw], quantile))
    if observations.s.size:
        nearest = int(np.argmin(np.abs(observations.s - float(position))))
        return float(observations.radii[nearest])
    finite = distances[np.isfinite(distances)]
    return float(np.median(finite)) if finite.size else 1e-8


def fit_interpretable_radius_profile(
    distances,
    s,
    *,
    s_center: float = 0.5,
    fixed_taper_position: float = 0.85,
    radius_quantile: float = 0.5,
    neck_window: tuple[float, float] = (0.0, 0.05),
    tip_window: tuple[float, float] | None = None,
    optimize: bool = True,
    constriction_s: float | None = None,
    constriction_window_half_width: float = 0.04,
    n_bins: int = 20,
    min_points_per_bin: int = 2,
    min_supported_bins: int = 6,
    outside_volume_weight: float = 2.0,
    max_constriction_to_neighbor_fraction: float = 0.98,
    tip_projection_tolerance: float = 1e-6,
    observations_override: RadiusProfileObservations | None = None,
    **_deprecated_options,
) -> RadiusProfileFitResult:
    """Fit landmark radii using an asymmetric cross-sectional volume proxy.

    Squared radius is proportional to cross-sectional area. Over-predicted
    area approximates primitive volume outside the crypt and receives twice
    the weight of under-predicted (missing) area by default. No centerline
    curvature term is included.
    """
    distances = np.asarray(distances, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    if distances.size != s.size:
        raise ValueError("distances and s must have matching lengths")
    q = float(radius_quantile) if 0.0 < float(radius_quantile) <= 1.0 else 0.5
    taper_s = float(fixed_taper_position)
    center_s = float(s_center)
    if not 0.0 < center_s < taper_s < 1.0:
        raise ValueError("Landmarks must satisfy 0 < s_center < s_taper < 1")

    observations = observations_override
    if observations is None:
        observations = estimate_equal_arclength_radius_profile(
            distances,
            s,
            n_bins=n_bins,
            min_points_per_bin=min_points_per_bin,
            radius_quantile=q,
            tip_projection_tolerance=tip_projection_tolerance,
        )
    finite_distances = distances[np.isfinite(distances)]
    fallback_radius = max(
        float(np.median(finite_distances)) if finite_distances.size else 1e-8,
        1e-8,
    )
    neck_half_width = max(
        0.5 * abs(float(neck_window[1]) - float(neck_window[0])), 0.025
    )
    r_attachment = max(
        _radius_near(
            distances,
            s,
            observations,
            0.5 * (float(neck_window[0]) + float(neck_window[1])),
            half_width=neck_half_width,
            quantile=q,
        ),
        1e-8,
    )
    r_center_observed = max(
        _radius_near(
            distances, s, observations, center_s, half_width=0.05, quantile=q
        ),
        1e-8,
    )
    distal_position = taper_s
    if tip_window is not None:
        distal_position = 0.5 * (float(tip_window[0]) + float(tip_window[1]))
    r_distal_observed = max(
        _radius_near(
            distances,
            s,
            observations,
            distal_position,
            half_width=0.05,
            quantile=q,
        ),
        1e-8,
    )

    constriction_position = None
    r_constriction_observed = None
    if constriction_s is not None and np.isfinite(float(constriction_s)):
        candidate = float(constriction_s)
        if 0.0 < candidate < center_s:
            constriction_position = candidate
            r_constriction_observed = max(
                _radius_near(
                    distances,
                    s,
                    observations,
                    candidate,
                    half_width=max(float(constriction_window_half_width), 1e-3),
                    quantile=q,
                ),
                1e-8,
            )

    r_center = r_center_observed
    r_distal = r_distal_observed
    r_constriction = r_constriction_observed
    optimization = {
        "attempted": bool(optimize),
        "success": False,
        "message": "robust_binned_initial_profile",
        "nfev": 0,
        "outside_volume_weight": float(outside_volume_weight),
        "s_center_source": "iterated_fitted_radius_volume_center",
        "s_taper_source": "fixed_visualization_parameter",
    }
    enough_support = observations.s.size >= max(3, int(min_supported_bins))
    if optimize and enough_support:
        has_constriction = r_constriction is not None
        distal_fraction = float(
            np.clip(r_distal / max(r_center, 1e-12), 1e-6, 1.0 - 1e-6)
        )
        x0 = [
            np.log(r_center),
            float(
                np.clip(
                    np.log(distal_fraction / (1.0 - distal_fraction)),
                    -10.0,
                    10.0,
                )
            ),
        ]
        if has_constriction:
            x0.append(np.log(r_constriction))
        scale_area = max(
            float(np.median(observations.radii**2)), fallback_radius**2, 1e-12
        )
        outside_weight = max(float(outside_volume_weight), 1.0)

        def decode(x):
            center_radius = float(np.exp(x[0]))
            distal_radius = center_radius * float(expit(x[1]))
            constriction_radius = None
            if has_constriction:
                raw = float(np.exp(x[2]))
                maximum = float(max_constriction_to_neighbor_fraction) * min(
                    r_attachment, center_radius
                )
                constriction_radius = min(raw, max(maximum, 1e-8))
            return center_radius, distal_radius, constriction_radius

        def residual_function(x):
            center_radius, distal_radius, constriction_radius = decode(x)
            predicted = capped_tube_radius(
                observations.s,
                r_attachment,
                center_radius,
                distal_radius,
                center_s=center_s,
                taper_start=taper_s,
                constriction_s=constriction_position,
                r_constriction=constriction_radius,
            )
            area_error = (predicted**2 - observations.radii**2) / scale_area
            weights = np.where(area_error > 0.0, np.sqrt(outside_weight), 1.0)
            return weights * area_error

        maximum_radius = max(float(np.max(observations.radii)), fallback_radius)
        n_variables = len(x0)
        try:
            result = least_squares(
                residual_function,
                np.asarray(x0, dtype=float),
                bounds=(
                    np.array(
                        [np.log(1e-8), -12.0]
                        + ([np.log(1e-8)] if has_constriction else [])
                    ),
                    np.array(
                        [np.log(10.0 * maximum_radius), 12.0]
                        + ([np.log(10.0 * maximum_radius)] if has_constriction else [])
                    ),
                ),
                loss="soft_l1",
                max_nfev=300,
            )
            if result.success and np.all(np.isfinite(result.x)):
                r_center, r_distal, r_constriction = decode(result.x)
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

    # Distal width remains fitted, but must precede a closing cap rather than
    # becoming wider than the crypt-center cross-section.
    r_distal = min(float(r_distal), 0.999999 * float(r_center))

    predicted = capped_tube_radius(
        observations.s,
        r_attachment,
        r_center,
        r_distal,
        center_s=center_s,
        taper_start=taper_s,
        constriction_s=constriction_position,
        r_constriction=r_constriction,
    )
    area_delta = predicted**2 - observations.radii**2
    outside_proxy = (
        float(np.mean(np.maximum(area_delta, 0.0))) if area_delta.size else 0.0
    )
    missing_proxy = (
        float(np.mean(np.maximum(-area_delta, 0.0))) if area_delta.size else 0.0
    )
    normalization = max(
        float(np.mean(observations.radii**2)) if observations.radii.size else 0.0,
        1e-12,
    )
    optimization.update(
        n_supported_bins=int(observations.s.size),
        n_input_points=int(observations.n_input_points),
        n_excluded_tip_points=int(observations.n_excluded_tip_points),
        r_attachment_observed=float(r_attachment),
        r_center_observed=float(r_center_observed),
        r_distal_observed=float(r_distal_observed),
        r_constriction_observed=(
            None if r_constriction_observed is None else float(r_constriction_observed)
        ),
        outside_volume_proxy=outside_proxy / normalization,
        missing_volume_proxy=missing_proxy / normalization,
        candidate_score=(float(outside_volume_weight) * outside_proxy + missing_proxy)
        / normalization,
        observation_rmse=(
            float(np.sqrt(np.mean((observations.radii - predicted) ** 2)))
            if observations.radii.size
            else float("nan")
        ),
    )
    return RadiusProfileFitResult(
        r_attachment=float(r_attachment),
        r_center=float(r_center),
        r_distal=float(r_distal),
        s_center=float(center_s),
        s_taper=float(taper_s),
        r_constriction=None if r_constriction is None else float(r_constriction),
        s_constriction=(
            None if constriction_position is None else float(constriction_position)
        ),
        observations=observations,
        diagnostics=optimization,
    )

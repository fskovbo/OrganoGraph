"""Quality-control summaries for exported crypt tube primitives.

This module evaluates reconstructive v5 and legacy v2-v4 exports without
rerunning skeletonization or primitive fitting. The resulting records are
intended for retrospective audits and baseline-versus-candidate comparisons;
QC flags are not fitted biological parameters and should not be encoded by a
shape model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from organograph.skeleton.export import load_shape_export_json
from organograph.skeleton.legacy_curves import (
    sample_cubic_bezier,
    sample_quadratic_bezier,
    sample_sinusoidal_bend,
)
from organograph.skeleton.primitive_geometry import tube_radius_from_parameters
from organograph.skeleton.primitive.crypt_geometry import sample_tangent_hermite
from organograph.skeleton.legacy_curves import sample_circular_arc


@dataclass(frozen=True)
class CryptPrimitiveQCConfig:
    """Thresholds used to flag potentially unstable crypt primitive fits."""

    center_position_bounds: tuple[float, float] = (0.05, 0.80)
    fixed_taper_position: float = 0.85
    min_center_taper_gap: float = 0.05
    position_tolerance: float = 0.01
    max_neck_to_body_ratio: float = 2.0
    max_distal_to_body_ratio: float = 1.0
    max_crypt_body_to_host_scale: float = 0.75
    min_length_to_host_scale: float = 0.10
    constriction_margin_fraction: float = 0.0
    profile_samples: int = 256


QC_FLAG_COLUMNS = (
    "flag_s_center_lower_clamp",
    "flag_s_center_upper_clamp",
    "flag_nonfixed_taper",
    "flag_min_center_taper_gap",
    "flag_missing_budded_constriction",
    "flag_invalid_constriction_minimum",
    "flag_large_neck_to_body_ratio",
    "flag_large_distal_to_body_ratio",
    "flag_large_crypt_to_host_scale",
    "flag_short_centerline",
    "flag_line_centerline",
)


def _sample_key(sample: dict[str, Any]) -> str:
    return "::".join(
        str(sample.get(key, "")) for key in ("dataset", "timepoint", "label_uid")
    )


def _body_scale(payload: dict[str, Any]) -> float:
    for primitive in payload.get("primitives") or []:
        if primitive.get("role") != "body":
            continue
        parameters = primitive.get("parameters") or {}
        if "axis_lengths" in parameters:
            axes = np.asarray(parameters["axis_lengths"], dtype=float)
        elif "axis_lengths_negative" in parameters and "axis_lengths_positive" in parameters:
            negative = np.asarray(parameters["axis_lengths_negative"], dtype=float)
            positive = np.asarray(parameters["axis_lengths_positive"], dtype=float)
            axes = 0.5 * (negative + positive)
        else:
            continue
        axes = axes[np.isfinite(axes) & (axes > 0.0)]
        if axes.size:
            return float(np.exp(np.mean(np.log(axes))))
    return float("nan")


def _sample_centerline(parameters: dict[str, Any]) -> np.ndarray:
    controls = np.asarray(parameters.get("centerline_control_points"), dtype=float)
    curve_type = str(parameters.get("centerline_type", "polyline"))
    n_samples = max(2, int(parameters.get("centerline_samples", 64)))
    if curve_type == "quadratic_bezier" and controls.shape == (3, 3):
        return sample_quadratic_bezier(*controls, n_samples=n_samples)
    if curve_type == "cubic_bezier" and controls.shape == (4, 3):
        return sample_cubic_bezier(*controls, n_samples=n_samples)
    if curve_type == "sinusoidal_bend" and controls.shape == (2, 3):
        bend = np.asarray(parameters.get("centerline_bend_vector"), dtype=float)
        if bend.shape == (3,):
            return sample_sinusoidal_bend(
                controls[0], controls[1], bend, n_samples=n_samples
            )
    if curve_type == "circular_arc" and controls.shape == (2, 3):
        sagitta = np.asarray(parameters.get("centerline_sagitta_vector"), dtype=float)
        if sagitta.shape == (3,):
            return sample_circular_arc(
                controls[0], controls[1], sagitta, n_samples=n_samples
            )
    if curve_type == "tangent_hermite" and controls.shape == (2, 3):
        start_tangent = np.asarray(
            parameters.get("centerline_start_tangent"), dtype=float
        )
        end_tangent = np.asarray(
            parameters.get("centerline_end_tangent"), dtype=float
        )
        if start_tangent.shape == (3,) and end_tangent.shape == (3,):
            return sample_tangent_hermite(
                controls[0],
                controls[1],
                start_tangent,
                end_tangent,
                n_samples=n_samples,
            )
    if controls.ndim == 2 and controls.shape[1:] == (3,) and controls.shape[0] >= 2:
        return controls
    return np.empty((0, 3), dtype=float)


def _centerline_length(centerline: np.ndarray) -> float:
    if centerline.shape[0] < 2:
        return float("nan")
    return float(np.sum(np.linalg.norm(np.diff(centerline, axis=0), axis=1)))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None:
        return float("nan")
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def _profile_diagnostics(parameters: dict[str, Any], config: CryptPrimitiveQCConfig):
    if "radius_control_s" in parameters:
        control_s = np.asarray(parameters["radius_control_s"], dtype=float)
        control_radii = np.asarray(parameters["radius_control_radii"], dtype=float)
        if control_s.size < 3 or control_radii.shape != control_s.shape:
            return float("nan"), float("nan"), False
        interior = np.arange(1, control_radii.size - 1)
        minima = interior[
            (control_radii[interior] < control_radii[interior - 1])
            & (control_radii[interior] < control_radii[interior + 1])
        ]
        if minima.size == 0:
            return float("nan"), float("nan"), False
        index = int(minima[np.argmin(control_radii[minima])])
        neighbors = min(control_radii[index - 1], control_radii[index + 1])
        depth = _safe_ratio(neighbors - control_radii[index], neighbors)
        return float(control_s[index]), depth, True
    r_constriction = parameters.get("r_constriction")
    s_constriction = parameters.get("s_constriction")
    if r_constriction is None or s_constriction is None:
        return float("nan"), float("nan"), False
    try:
        r_attachment = float(parameters.get("r_attachment", parameters.get("r_neck")))
        r_center = float(parameters.get("r_center", parameters.get("r_body")))
        r_distal = float(parameters.get("r_distal", parameters.get("r_tip")))
        s_center = float(parameters.get("s_center", parameters.get("s_body")))
        s = np.linspace(0.0, 1.0, max(32, int(config.profile_samples)))
        radii = tube_radius_from_parameters(parameters, s)
    except (KeyError, TypeError, ValueError):
        return float("nan"), float("nan"), False

    body_s = float(parameters.get("s_center", parameters.get("s_body")))
    proximal = s <= body_s
    if not np.any(proximal):
        return float("nan"), float("nan"), False
    proximal_s = s[proximal]
    proximal_r = radii[proximal]
    minimum_index = int(np.argmin(proximal_r))
    minimum_s = float(proximal_s[minimum_index])
    neighboring_radius = min(r_attachment, r_center)
    depth = _safe_ratio(neighboring_radius - float(r_constriction), neighboring_radius)
    tolerance = max(2.0 / max(int(config.profile_samples), 32), 0.02)
    minimum_near_constriction = abs(minimum_s - float(s_constriction)) <= tolerance
    return minimum_s, depth, bool(minimum_near_constriction)


def crypt_primitive_qc_records(
    payload: dict[str, Any],
    *,
    shape_path: str | Path | None = None,
    config: CryptPrimitiveQCConfig | None = None,
    quality_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract one QC record per crypt primitive from v6 or a legacy payload."""
    config = config or CryptPrimitiveQCConfig()
    sample = dict(payload.get("sample") or {})
    node_types = {
        str(node["node_id"]): str(node["node_type"])
        for node in (payload.get("skeleton") or {}).get("nodes", [])
    }
    body_scale = _body_scale(payload)
    quality_by_primitive = {
        str(record.get("primitive_id")): record
        for record in (quality_payload or {}).get("crypt_primitives", [])
    }
    records = []
    for primitive in payload.get("primitives") or []:
        if primitive.get("role") != "crypt":
            continue
        parameters = dict(primitive.get("parameters") or {})
        quality = quality_by_primitive.get(str(primitive.get("primitive_id")), {})
        optimization = dict(quality.get("profile_optimization") or {})
        candidate_selection = dict(quality.get("candidate_selection") or {})
        residuals = dict(quality.get("residuals") or {})
        target_ids = [str(value) for value in primitive.get("target_node_ids") or []]
        target_types = [node_types.get(value, "unknown") for value in target_ids]
        if "radius_control_s" in parameters:
            radius_positions = np.asarray(parameters["radius_control_s"], dtype=float)
            radius_controls = np.asarray(
                parameters.get("radius_control_radii"), dtype=float
            )
            r_neck = float(radius_controls[0])
            r_body = float(np.max(radius_controls))
            r_distal = float(radius_controls[-1])
            dense_s = np.linspace(0.0, 1.0, max(65, int(config.profile_samples)))
            dense_radii = tube_radius_from_parameters(parameters, dense_s)
            area = dense_radii**2
            mass = float(np.trapezoid(area, dense_s))
            s_body = (
                float(np.trapezoid(dense_s * area, dense_s) / mass)
                if mass > 1e-12
                else float(radius_positions[len(radius_positions) // 2])
            )
        else:
            r_neck = float(
                parameters.get("r_attachment", parameters.get("r_neck", np.nan))
            )
            r_body = float(parameters.get("r_center", parameters.get("r_body", np.nan)))
            r_distal = float(parameters.get("r_distal", parameters.get("r_tip", np.nan)))
            s_body = float(parameters.get("s_center", parameters.get("s_body", np.nan)))
        r_constriction = parameters.get("r_constriction")
        s_constriction = parameters.get("s_constriction")
        s_taper = float(parameters.get("s_taper", np.nan))
        centerline = _sample_centerline(parameters)
        centerline_length = _centerline_length(centerline)
        controls = np.asarray(parameters.get("centerline_control_points"), dtype=float)
        bend_vector = np.asarray(parameters.get("centerline_bend_vector"), dtype=float)
        if parameters.get("centerline_type") == "circular_arc":
            bend_vector = np.asarray(
                parameters.get("centerline_sagitta_vector"), dtype=float
            )
        dimensionless_bend = float("nan")
        if controls.shape == (2, 3) and bend_vector.shape == (3,):
            dimensionless_bend = _safe_ratio(
                float(np.linalg.norm(bend_vector)),
                float(np.linalg.norm(controls[1] - controls[0])),
            )
        elif controls.shape == (2, 3) and centerline.shape[0] >= 2:
            chord = controls[1] - controls[0]
            chord_length = float(np.linalg.norm(chord))
            if chord_length > 1e-12:
                relative = centerline - controls[0]
                chord_s = np.clip((relative @ chord) / chord_length**2, 0.0, 1.0)
                baseline = controls[0] + chord_s[:, None] * chord
                dimensionless_bend = float(
                    np.max(np.linalg.norm(centerline - baseline, axis=1)) / chord_length
                )
        subtype = (
            "unclassified"
            if "radius_control_s" in parameters
            else ("budded" if "constriction" in target_types else "bulged")
        )
        has_constriction = r_constriction is not None and s_constriction is not None
        minimum_s, constriction_depth, minimum_near = _profile_diagnostics(
            parameters, config
        )
        margin = max(float(config.constriction_margin_fraction), 0.0)
        constriction_limit = (
            (1.0 - margin) * min(r_neck, r_body)
            if np.isfinite(r_neck) and np.isfinite(r_body)
            else float("nan")
        )

        flags = {
            "flag_s_center_lower_clamp": bool(
                np.isfinite(s_body)
                and s_body <= config.center_position_bounds[0] + config.position_tolerance
            ),
            "flag_s_center_upper_clamp": bool(
                np.isfinite(s_body)
                and s_body >= config.center_position_bounds[1] - config.position_tolerance
            ),
            "flag_nonfixed_taper": bool(
                np.isfinite(s_taper)
                and abs(s_taper - config.fixed_taper_position)
                > config.position_tolerance
            ),
            "flag_min_center_taper_gap": bool(
                np.isfinite(s_body)
                and np.isfinite(s_taper)
                and s_taper - s_body
                <= config.min_center_taper_gap + config.position_tolerance
            ),
            "flag_missing_budded_constriction": subtype == "budded" and not has_constriction,
            "flag_invalid_constriction_minimum": bool(
                has_constriction
                and (
                    not np.isfinite(constriction_limit)
                    or float(r_constriction) >= constriction_limit
                    or not minimum_near
                )
            ),
            "flag_large_neck_to_body_ratio": bool(
                _safe_ratio(r_neck, r_body) > config.max_neck_to_body_ratio
            ),
            "flag_large_distal_to_body_ratio": bool(
                _safe_ratio(r_distal, r_body) > config.max_distal_to_body_ratio
            ),
            "flag_large_crypt_to_host_scale": bool(
                _safe_ratio(r_body, body_scale) > config.max_crypt_body_to_host_scale
            ),
            "flag_short_centerline": bool(
                _safe_ratio(centerline_length, body_scale) < config.min_length_to_host_scale
            ),
            "flag_line_centerline": str(parameters.get("centerline_type")) == "line",
        }
        active_flags = [name.removeprefix("flag_") for name, active in flags.items() if active]
        severe = (
            3 * int(flags["flag_missing_budded_constriction"])
            + 3 * int(flags["flag_invalid_constriction_minimum"])
            + 2 * int(flags["flag_large_neck_to_body_ratio"])
            + 2 * int(flags["flag_large_distal_to_body_ratio"])
            + 2 * int(flags["flag_large_crypt_to_host_scale"])
            + int(flags["flag_short_centerline"])
            + int(flags["flag_line_centerline"])
            + sum(int(flags[name]) for name in QC_FLAG_COLUMNS[:4])
        )
        records.append(
            {
                "sample_key": _sample_key(sample),
                "dataset": sample.get("dataset"),
                "timepoint": sample.get("timepoint"),
                "well": sample.get("well"),
                "organoid_id": sample.get("organoid_id"),
                "label_uid": sample.get("label_uid"),
                "mesh_path": sample.get("mesh_path"),
                "shape_path": str(shape_path) if shape_path is not None else None,
                "primitive_id": str(primitive.get("primitive_id")),
                "crypt_id": next(
                    (
                        node.get("crypt_id")
                        for node in (payload.get("skeleton") or {}).get("nodes", [])
                        if str(node.get("node_id")) in target_ids
                        and node.get("crypt_id") is not None
                    ),
                    None,
                ),
                "subtype": subtype,
                "centerline_type": parameters.get("centerline_type"),
                "target_node_ids": target_ids,
                "body_scale": body_scale,
                "centerline_length": centerline_length,
                "length_to_body_scale": _safe_ratio(centerline_length, body_scale),
                "centerline_dimensionless_bend": dimensionless_bend,
                "r_neck": r_neck,
                "r_attachment": r_neck,
                "r_body": r_body,
                "r_center": r_body,
                "r_distal": r_distal,
                "r_constriction": r_constriction,
                "s_constriction": s_constriction,
                "s_body": s_body,
                "s_center": s_body,
                "s_taper": s_taper,
                "taper_gap": s_taper - s_body,
                "neck_to_body_ratio": _safe_ratio(r_neck, r_body),
                "distal_to_body_ratio": _safe_ratio(r_distal, r_body),
                "constriction_to_body_ratio": _safe_ratio(r_constriction, r_body),
                "crypt_body_to_host_scale": _safe_ratio(r_body, body_scale),
                "profile_proximal_min_s": minimum_s,
                "constriction_depth_fraction": constriction_depth,
                "n_qc_flags": len(active_flags),
                "qc_severity": severe,
                "qc_flags": "; ".join(active_flags),
                "has_qc_flag": bool(active_flags),
                "fit_rmse": residuals.get("rmse", quality.get("fit_error")),
                "fit_mae": residuals.get("mae"),
                "optimizer_success": optimization.get("success"),
                "optimizer_nfev": optimization.get("nfev"),
                "optimizer_cost": optimization.get("cost"),
                "n_fit_points": quality.get("n_points"),
                "n_supported_bins": optimization.get("n_supported_bins"),
                "profile_observation_rmse": optimization.get("observation_rmse"),
                "outside_volume_proxy": optimization.get("outside_volume_proxy"),
                "missing_volume_proxy": optimization.get("missing_volume_proxy"),
                "candidate_score": optimization.get("candidate_score"),
                "candidate_selection_score": candidate_selection.get(
                    "selected_score"
                ),
                "selected_tip_source": quality.get("tip_source"),
                "selected_tip_vertex_id": quality.get("tip_vertex_id"),
                "selected_centerline_kind": quality.get("centerline_kind"),
                "centerline_initial_tangent_source": quality.get(
                    "centerline_initial_tangent_source"
                ),
                "raw_centroid_projection_s": quality.get(
                    "raw_centroid_projection_s"
                ),
                "host_intersection_fraction": quality.get(
                    "host_intersection_fraction"
                ),
                "host_penetration_rms": quality.get("host_penetration_rms"),
                "host_penetration_max": quality.get("host_penetration_max"),
                "host_penetration_normalized": quality.get(
                    "host_penetration_normalized"
                ),
                "distal_constraint_active": optimization.get(
                    "distal_constraint_active"
                ),
                "constriction_constraint_active": optimization.get(
                    "constriction_constraint_active"
                ),
                "constriction_status": optimization.get("constriction_status"),
                "centerline_fallback_reason": quality.get(
                    "centerline_fallback_reason"
                ),
                **flags,
            }
        )
    return records


def discover_shape_exports(export_root: str | Path) -> list[Path]:
    """Find reconstructive ``shape.json`` files below an export root."""
    root = Path(export_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Export root does not exist: {root}")
    return sorted(root.rglob("shape.json"))


def load_crypt_primitive_qc(
    export_root: str | Path,
    *,
    config: CryptPrimitiveQCConfig | None = None,
    unbranched_only: bool = True,
    sample_keys: Iterable[str] | None = None,
    max_samples: int | None = None,
):
    """Load an export tree into a crypt-level pandas QC table.

    ``sample_keys`` and ``max_samples`` make candidate-export audits cheap: a
    partial candidate tree can be loaded and compared against matching rows in
    a full baseline tree without requiring either export to be complete.
    """
    import pandas as pd

    requested = None if sample_keys is None else {str(value) for value in sample_keys}
    records: list[dict[str, Any]] = []
    loaded_samples = 0
    for shape_path in discover_shape_exports(export_root):
        payload = load_shape_export_json(shape_path)
        sample = dict(payload.get("sample") or {})
        key = _sample_key(sample)
        if requested is not None and key not in requested:
            continue
        if unbranched_only and bool(sample.get("has_branches", False)):
            continue
        quality_payload = None
        quality_path = shape_path.with_name("quality.json")
        if quality_path.exists():
            with quality_path.open("r", encoding="utf-8") as handle:
                quality_payload = json.load(handle)
        records.extend(
            crypt_primitive_qc_records(
                payload,
                shape_path=shape_path,
                config=config,
                quality_payload=quality_payload,
            )
        )
        loaded_samples += 1
        if max_samples is not None and loaded_samples >= int(max_samples):
            break
    return pd.DataFrame.from_records(records)


def pair_crypt_primitive_qc(baseline, candidate):
    """Pair baseline and candidate crypt rows for a possibly partial candidate."""
    import pandas as pd

    keys = ["sample_key", "primitive_id"]
    if baseline.empty or candidate.empty:
        return pd.DataFrame()
    return baseline.merge(
        candidate,
        on=keys,
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )

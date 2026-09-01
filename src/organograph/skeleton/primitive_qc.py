"""Quality-control summaries for exported crypt tube primitives.

This module evaluates reconstructive ``organograph_shape_v2`` exports without
rerunning skeletonization or primitive fitting. The resulting records are
intended for retrospective audits and baseline-versus-candidate comparisons;
QC flags are not fitted biological parameters and should not be encoded by a
shape model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from organograph.skeleton.export import load_shape_export_json
from organograph.skeleton.primitive_geometry import (
    capped_tube_radius,
    sample_cubic_bezier,
    sample_quadratic_bezier,
)


@dataclass(frozen=True)
class CryptPrimitiveQCConfig:
    """Thresholds used to flag potentially unstable crypt primitive fits."""

    body_position_bounds: tuple[float, float] = (0.2, 0.7)
    max_taper_position: float = 0.9
    min_taper_gap: float = 0.1
    position_tolerance: float = 0.01
    max_neck_to_body_ratio: float = 2.0
    max_distal_to_body_ratio: float = 1.0
    max_crypt_body_to_host_scale: float = 0.75
    min_length_to_host_scale: float = 0.10
    constriction_margin_fraction: float = 0.0
    profile_samples: int = 256


QC_FLAG_COLUMNS = (
    "flag_s_body_lower_bound",
    "flag_s_body_upper_bound",
    "flag_s_taper_upper_bound",
    "flag_min_taper_gap",
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
    r_constriction = parameters.get("r_constriction")
    s_constriction = parameters.get("s_constriction")
    if r_constriction is None or s_constriction is None:
        return float("nan"), float("nan"), False
    try:
        s = np.linspace(0.0, 1.0, max(32, int(config.profile_samples)))
        radii = capped_tube_radius(
            s,
            float(parameters["r_neck"]),
            float(parameters["r_body"]),
            float(parameters["r_tip"]),
            body_s=float(parameters["s_body"]),
            taper_start=float(parameters["s_taper"]),
            constriction_s=float(s_constriction),
            r_constriction=float(r_constriction),
        )
    except (KeyError, TypeError, ValueError):
        return float("nan"), float("nan"), False

    body_s = float(parameters["s_body"])
    proximal = s <= body_s
    if not np.any(proximal):
        return float("nan"), float("nan"), False
    proximal_s = s[proximal]
    proximal_r = radii[proximal]
    minimum_index = int(np.argmin(proximal_r))
    minimum_s = float(proximal_s[minimum_index])
    neighboring_radius = min(float(parameters["r_neck"]), float(parameters["r_body"]))
    depth = _safe_ratio(neighboring_radius - float(r_constriction), neighboring_radius)
    tolerance = max(2.0 / max(int(config.profile_samples), 32), 0.02)
    minimum_near_constriction = abs(minimum_s - float(s_constriction)) <= tolerance
    return minimum_s, depth, bool(minimum_near_constriction)


def crypt_primitive_qc_records(
    payload: dict[str, Any],
    *,
    shape_path: str | Path | None = None,
    config: CryptPrimitiveQCConfig | None = None,
) -> list[dict[str, Any]]:
    """Extract one QC record per crypt primitive from a v2 payload."""
    config = config or CryptPrimitiveQCConfig()
    sample = dict(payload.get("sample") or {})
    node_types = {
        str(node["node_id"]): str(node["node_type"])
        for node in (payload.get("skeleton") or {}).get("nodes", [])
    }
    body_scale = _body_scale(payload)
    records = []
    for primitive in payload.get("primitives") or []:
        if primitive.get("role") != "crypt":
            continue
        parameters = dict(primitive.get("parameters") or {})
        target_ids = [str(value) for value in primitive.get("target_node_ids") or []]
        target_types = [node_types.get(value, "unknown") for value in target_ids]
        r_neck = float(parameters.get("r_neck", np.nan))
        r_body = float(parameters.get("r_body", np.nan))
        r_distal = float(parameters.get("r_tip", np.nan))
        r_constriction = parameters.get("r_constriction")
        s_constriction = parameters.get("s_constriction")
        s_body = float(parameters.get("s_body", np.nan))
        s_taper = float(parameters.get("s_taper", np.nan))
        centerline = _sample_centerline(parameters)
        centerline_length = _centerline_length(centerline)
        subtype = (
            "budded"
            if "constriction" in target_types
            else "bulged"
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
            "flag_s_body_lower_bound": bool(
                np.isfinite(s_body)
                and s_body <= config.body_position_bounds[0] + config.position_tolerance
            ),
            "flag_s_body_upper_bound": bool(
                np.isfinite(s_body)
                and s_body >= config.body_position_bounds[1] - config.position_tolerance
            ),
            "flag_s_taper_upper_bound": bool(
                np.isfinite(s_taper)
                and s_taper >= config.max_taper_position - config.position_tolerance
            ),
            "flag_min_taper_gap": bool(
                np.isfinite(s_body)
                and np.isfinite(s_taper)
                and s_taper - s_body <= config.min_taper_gap + config.position_tolerance
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
                "r_neck": r_neck,
                "r_body": r_body,
                "r_distal": r_distal,
                "r_constriction": r_constriction,
                "s_constriction": s_constriction,
                "s_body": s_body,
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
                **flags,
            }
        )
    return records


def discover_shape_exports(export_root: str | Path) -> list[Path]:
    """Find v2 ``shape.json`` files below an export root."""
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
        records.extend(
            crypt_primitive_qc_records(payload, shape_path=shape_path, config=config)
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

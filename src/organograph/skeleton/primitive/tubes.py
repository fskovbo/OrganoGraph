"""Crypt tapered-tube primitive fitting and graph attachment."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from organograph.skeleton.datatypes import SkeletonGraph
from organograph.skeleton.geometry import as_points
from organograph.skeleton.primitive.blobs import fit_blob_primitive_to_points
from organograph.skeleton.primitive.common import _residual_summary
from organograph.skeleton.primitive_geometry import (
    bend_angles_for_polyline,
    capped_tube_radius,
    component_points,
    estimate_smooth_crypt_centerline,
    point_at_polyline_arclength,
    polyline_lengths,
    project_points_to_polyline,
    sample_quadratic_bezier,
    sanitize_id,
)
from organograph.skeleton.primitives import PrimitiveAttachment, PrimitiveFit

def _children_for_crypt(graph: SkeletonGraph, crypt_id):
    children = defaultdict(list)
    for edge in graph.edges_for_crypt(crypt_id, include_body_edge=False):
        children[edge.source].append(edge.target)
    return children

def _root_necks(graph: SkeletonGraph, crypt_id):
    necks = [
        node
        for node in graph.nodes_for_crypt(crypt_id)
        if node.node_type in {"neck", "attachment"}
    ]
    incoming_necks = set()
    for edge in graph.edges_for_crypt(crypt_id, include_body_edge=False):
        if graph.node(edge.target).node_type in {"neck", "attachment"}:
            incoming_necks.add(edge.target)
    roots = [node for node in necks if node.node_id not in incoming_necks]
    return roots or necks

def _all_root_to_tip_paths(graph: SkeletonGraph, crypt_id) -> list[list[str]]:
    tips = {node.node_id for node in graph.nodes_for_crypt(crypt_id, node_type="tip")}
    if not tips:
        return []
    children = _children_for_crypt(graph, crypt_id)
    paths = []
    for root in _root_necks(graph, crypt_id):
        q = deque([(root.node_id, [root.node_id])])
        while q:
            node_id, path = q.popleft()
            if node_id in tips:
                paths.append(path)
                continue
            for child in children.get(node_id, []):
                if child not in path:
                    q.append((child, path + [child]))
    return paths

def crypt_terminal_paths(graph: SkeletonGraph, crypt_id) -> list[list[str]]:
    """Return terminal paths beginning at the final attachment before each tip."""
    out = []
    for path in _all_root_to_tip_paths(graph, crypt_id):
        last_neck_index = 0
        for i, node_id in enumerate(path[:-1]):
            if graph.node(node_id).node_type in {"neck", "attachment"}:
                last_neck_index = i
        out.append(path[last_neck_index:])
    return out

def _radius_from_window(
    distances: np.ndarray,
    s: np.ndarray,
    lo: float,
    hi: float,
    *,
    quantile: float,
) -> float:
    mask = (s >= float(lo)) & (s <= float(hi)) & np.isfinite(distances)
    if not np.any(mask):
        center = 0.5 * (float(lo) + float(hi))
        order = np.argsort(np.abs(s - center))
        keep = order[: max(1, min(10, order.size))]
        vals = distances[keep]
    else:
        vals = distances[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, quantile))

def _logit(value: float) -> float:
    value = float(np.clip(value, 1e-6, 1.0 - 1e-6))
    return float(np.log(value / (1.0 - value)))

def _decode_profile_positions(
    body_raw: float,
    taper_raw: float,
    *,
    body_position_bounds: tuple[float, float],
    min_taper_gap: float,
    max_taper_position: float,
) -> tuple[float, float]:
    body_min, body_max = map(float, body_position_bounds)
    body_s = body_min + (body_max - body_min) * float(expit(body_raw))
    taper_min = body_s + float(min_taper_gap)
    taper_room = max(float(max_taper_position) - taper_min, 0.0)
    taper_s = taper_min + taper_room * float(expit(taper_raw))
    return body_s, taper_s

def _initial_position_variables(
    body_s: float,
    taper_s: float,
    *,
    body_position_bounds: tuple[float, float],
    min_taper_gap: float,
    max_taper_position: float,
) -> tuple[float, float]:
    body_min, body_max = map(float, body_position_bounds)
    body_frac = (float(body_s) - body_min) / max(body_max - body_min, 1e-12)
    body_raw = _logit(body_frac)
    decoded_body, _ = _decode_profile_positions(
        body_raw,
        0.0,
        body_position_bounds=body_position_bounds,
        min_taper_gap=min_taper_gap,
        max_taper_position=max_taper_position,
    )
    taper_min = decoded_body + float(min_taper_gap)
    taper_frac = (float(taper_s) - taper_min) / max(
        float(max_taper_position) - taper_min,
        1e-12,
    )
    return body_raw, _logit(taper_frac)

def fit_crypt_tube_to_points(
    points,
    centerline_points,
    *,
    path_node_ids: list[str] | None = None,
    radius_quantile: float = 0.5,
    neck_window: tuple[float, float] = (0.0, 0.2),
    body_window: tuple[float, float] = (0.4, 0.6),
    tip_window: tuple[float, float] | None = None,
    optimize_radius_profile: bool = True,
    initial_body_position: float = 0.5,
    initial_taper_position: float = 0.85,
    body_position_bounds: tuple[float, float] = (0.2, 0.7),
    min_taper_gap: float = 0.1,
    max_taper_position: float = 0.9,
    distal_taper_start: float | None = None,
    constriction_s: float | None = None,
    constriction_window_half_width: float = 0.04,
    metadata: dict[str, Any] | None = None,
) -> PrimitiveFit:
    """Fit a tapered capped tube to crypt component points.

    The supplied centerline is represented by dense straight samples of the
    fitted smooth curve. Attachment, body, and distal taper radii are estimated
    from point distances to that curve. Budded crypt paths can additionally
    provide ``constriction_s`` to fit an explicit internal minimum radius.
    Radii and the ordered control positions ``s_body`` and ``s_taper`` are
    jointly optimized, and the profile closes smoothly at the crypt-tip node.
    """
    pts = as_points(points)
    centerline = as_points(centerline_points)
    if centerline.shape[0] < 2:
        raise ValueError("A crypt tube centerline needs at least two points")
    if pts.shape[0] == 0:
        raise ValueError("At least one component point is required")

    projection = project_points_to_polyline(pts, centerline)
    distances = projection["distances"]
    s = projection["s"]
    q = float(radius_quantile)
    if not (0.0 < q <= 1.0):
        q = 0.5
    if distal_taper_start is not None:
        initial_taper_position = float(distal_taper_start)
    body_min, body_max = map(float, body_position_bounds)
    gap = float(min_taper_gap)
    taper_max = float(max_taper_position)
    if not (0.0 < body_min < body_max < 1.0):
        raise ValueError("body_position_bounds must satisfy 0 < min < max < 1")
    if gap <= 0.0 or body_max + gap > taper_max or taper_max >= 1.0:
        raise ValueError(
            "Profile safeguards must allow body_s + min_taper_gap <= max_taper_position < 1"
        )
    constriction_position = None
    if constriction_s is not None:
        constriction_position = float(constriction_s)
        if not (0.0 < constriction_position < body_max):
            constriction_position = None
    effective_body_min = body_min
    if constriction_position is not None:
        effective_body_min = max(body_min, constriction_position + 0.05)
        if effective_body_min >= body_max:
            constriction_position = None
            effective_body_min = body_min
    effective_body_bounds = (effective_body_min, body_max)
    body_s = float(np.clip(initial_body_position, *effective_body_bounds))
    taper_start = float(
        np.clip(initial_taper_position, body_s + gap, taper_max)
    )
    if tip_window is None:
        tip_window = (max(0.5, taper_start - 0.1), taper_start)

    r_neck = _radius_from_window(distances, s, *neck_window, quantile=q)
    r_body = _radius_from_window(distances, s, *body_window, quantile=q)
    r_tip = _radius_from_window(distances, s, *tip_window, quantile=q)
    r_constriction = None
    if constriction_position is not None:
        half_width = max(float(constriction_window_half_width), 1e-3)
        r_constriction = _radius_from_window(
            distances,
            s,
            max(0.0, constriction_position - half_width),
            min(1.0, constriction_position + half_width),
            quantile=q,
        )
    radii = np.asarray(
        [r_neck, r_body, r_tip]
        + ([r_constriction] if r_constriction is not None else []),
        dtype=float,
    )
    fallback_radius = float(np.nanmedian(distances)) if np.any(np.isfinite(distances)) else 0.0
    radii = np.where(np.isfinite(radii), radii, fallback_radius)
    radii = np.maximum(radii, 1e-8)
    r_neck, r_body, r_tip = map(float, radii[:3])
    if radii.size > 3:
        r_constriction = float(radii[3])

    optimization_info = {
        "attempted": bool(optimize_radius_profile),
        "success": False,
        "message": "fixed_initial_profile",
        "nfev": 0,
    }
    if optimize_radius_profile and pts.shape[0] >= 10:
        body_raw, taper_raw = _initial_position_variables(
            body_s,
            taper_start,
            body_position_bounds=effective_body_bounds,
            min_taper_gap=gap,
            max_taper_position=taper_max,
        )
        fitted_radii = [r_neck, r_body, r_tip]
        if r_constriction is not None:
            fitted_radii.append(r_constriction)
        n_radii = len(fitted_radii)
        x0 = np.array(
            [*(np.log(fitted_radii)), body_raw, taper_raw],
            dtype=float,
        )
        finite_distances = distances[np.isfinite(distances)]
        scale = float(np.nanmedian(finite_distances)) if finite_distances.size else 1.0
        scale = max(scale, 1e-6)

        def profile_residuals(x):
            fitted = np.exp(x[:n_radii])
            rn, rb, rt = fitted[:3]
            rc = float(fitted[3]) if n_radii == 4 else None
            sb, st = _decode_profile_positions(
                x[n_radii],
                x[n_radii + 1],
                body_position_bounds=effective_body_bounds,
                min_taper_gap=gap,
                max_taper_position=taper_max,
            )
            predicted_radius = capped_tube_radius(
                s,
                rn,
                rb,
                rt,
                body_s=sb,
                taper_start=st,
                constriction_s=constriction_position,
                r_constriction=rc,
            )
            return (distances - predicted_radius) / scale

        max_radius = max(float(np.nanmax(finite_distances)) if finite_distances.size else scale, scale)
        lower = np.array(
            [*[np.log(1e-8)] * n_radii, -8.0, -8.0],
            dtype=float,
        )
        upper = np.array(
            [*[np.log(max_radius * 10.0)] * n_radii, 8.0, 8.0],
            dtype=float,
        )
        try:
            result = least_squares(
                profile_residuals,
                x0,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=500,
            )
            if result.success and np.all(np.isfinite(result.x)):
                fitted = np.exp(result.x[:n_radii])
                r_neck, r_body, r_tip = map(float, fitted[:3])
                if n_radii == 4:
                    r_constriction = float(fitted[3])
                body_s, taper_start = _decode_profile_positions(
                    result.x[n_radii],
                    result.x[n_radii + 1],
                    body_position_bounds=effective_body_bounds,
                    min_taper_gap=gap,
                    max_taper_position=taper_max,
                )
                optimization_info.update(
                    {
                        "success": True,
                        "message": str(result.message),
                        "nfev": int(result.nfev),
                        "cost": float(result.cost),
                    }
                )
            else:
                optimization_info.update(
                    {
                        "message": str(result.message),
                        "nfev": int(result.nfev),
                    }
                )
        except (ValueError, FloatingPointError) as exc:
            optimization_info["message"] = f"fallback_after_error: {exc}"

    predicted = capped_tube_radius(
        s,
        r_neck,
        r_body,
        r_tip,
        body_s=body_s,
        taper_start=taper_start,
        constriction_s=constriction_position,
        r_constriction=r_constriction,
    )
    residuals = distances - predicted
    summary = _residual_summary(residuals)

    _, _, length = polyline_lengths(centerline)
    straight = float(np.linalg.norm(centerline[-1] - centerline[0]))
    bend_angles = bend_angles_for_polyline(centerline)
    segments = np.diff(centerline, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    valid_segments = segments[segment_lengths > 1e-12]
    if valid_segments.shape[0] >= 2:
        first = valid_segments[0] / np.linalg.norm(valid_segments[0])
        last = valid_segments[-1] / np.linalg.norm(valid_segments[-1])
        bend_angle = float(
            np.arccos(np.clip(np.dot(first, last), -1.0, 1.0))
        )
    else:
        bend_angle = 0.0
    derived = {
        "length": float(length),
        "straight_distance": straight,
        "tortuosity": float(length / straight) if straight > 1e-12 else float("nan"),
        "bend_angle": bend_angle,
        "bend_angles": bend_angles,
        "constriction_ratio": (
            float(
                (r_constriction if r_constriction is not None else r_neck)
                / r_body
            )
            if r_body > 1e-12
            else float("nan")
        ),
        "taper_ratio": float(r_tip / r_body) if r_body > 1e-12 else float("nan"),
    }
    return PrimitiveFit(
        primitive_type="tapered_capped_tube",
        parameters={
            "centerline_points": centerline,
            "path_node_ids": list(path_node_ids or []),
            "r_neck": r_neck,
            "r_attachment": r_neck,
            "r_body": r_body,
            "r_tip": r_tip,
            "r_taper": r_tip,
            "r_constriction": r_constriction,
            "s_constriction": constriction_position,
            "s_body": body_s,
            "s_taper": taper_start,
            "radius_quantile": q,
            "radius_profile": "shape_preserving_cubic_squared_radius",
            "distal_taper_start": taper_start,
            "distal_taper": "smooth_squared_radius_to_zero",
            "cap": "integrated_squared_radius_closure",
            "neck_window": neck_window,
            "body_window": body_window,
            "tip_window": tip_window,
            "profile_safeguards": {
                "body_position_bounds": effective_body_bounds,
                "min_taper_gap": gap,
                "max_taper_position": taper_max,
            },
        },
        fit_error=summary["rmse"],
        residuals=summary,
        derived_parameters=derived,
        metadata={
            "fit_method": "point_distances_to_piecewise_linear_centerline",
            "n_points": int(pts.shape[0]),
            "profile_optimization": optimization_info,
            **dict(metadata or {}),
        },
    )

def attach_body_primitive(
    graph: SkeletonGraph,
    vertices,
    component=None,
    *,
    primitive_type: str = "ellipsoid",
    metadata: dict[str, Any] | None = None,
    **fit_kwargs,
) -> PrimitiveAttachment:
    """Fit and attach a body blob primitive to the body node."""
    points = component_points(vertices, component)
    fit = fit_blob_primitive_to_points(
        points,
        primitive_type=primitive_type,
        metadata={"component": "body", **dict(metadata or {})},
        **fit_kwargs,
    )
    attachment = fit.to_attachment(
        attachment_type="node",
        attachment_id="body",
        target_ids=[graph.body_node().node_id],
    )
    graph.body_node().primitive_attachment = attachment
    return attachment

def attach_branch_primitives(
    graph: SkeletonGraph,
    vertices,
    branch_components: dict[str, Any],
    *,
    primitive_type: str = "ellipsoid",
    **fit_kwargs,
) -> dict[str, PrimitiveAttachment]:
    """Fit blob primitives to branch components keyed by branch node id."""
    out = {}
    for branch_node_id, component in branch_components.items():
        branch_node_id = str(branch_node_id)
        node = graph.node(branch_node_id)
        points = component_points(vertices, component)
        fit = fit_blob_primitive_to_points(
            points,
            primitive_type=primitive_type,
            metadata={"component": "branch", "branch_node_id": branch_node_id},
            **fit_kwargs,
        )
        attachment = fit.to_attachment(
            attachment_type="node",
            attachment_id=branch_node_id,
            target_ids=[branch_node_id],
        )
        node.primitive_attachment = attachment
        out[branch_node_id] = attachment
    return out

def _path_id(path: list[str]) -> str:
    return "_to_".join(path)

def _resolve_crypt_paths(graph: SkeletonGraph, key) -> list[tuple[str, list[str]]]:
    if isinstance(key, (list, tuple)) and len(key) >= 2:
        path = [str(v) for v in key]
        return [(_path_id(path), path)]
    key_str = str(key)
    if key_str in graph.nodes and graph.node(key_str).node_type == "tip":
        for crypt_id in graph.crypt_ids():
            for path in crypt_terminal_paths(graph, crypt_id):
                if path[-1] == key_str:
                    return [(_path_id(path), path)]
        raise ValueError(f"No crypt path ends at tip node {key_str!r}")

    paths = crypt_terminal_paths(graph, key)
    if not paths:
        raise ValueError(f"No crypt paths found for crypt/component key {key!r}")
    return [(f"crypt_{sanitize_id(key)}_path_{i}", path) for i, path in enumerate(paths)]

def attach_crypt_tube_primitives(
    graph: SkeletonGraph,
    vertices,
    crypt_components: dict[Any, Any],
    *,
    centerline_data: dict[Any, dict[str, Any]] | None = None,
    smooth_centerline: bool = True,
    centerline_n_bands: int = 7,
    centerline_n_samples: int = 64,
    centerline_constriction_weight: float = 4.0,
    update_crypt_nodes: bool = True,
    **fit_kwargs,
) -> dict[str, PrimitiveAttachment]:
    """Fit tapered capped tubes to crypt components.

    ``crypt_components`` can be keyed by crypt id, by tip node id, or by an
    explicit tuple/list of path node ids.  Values can be vertex indices into
    ``vertices`` or direct ``(N, 3)`` point arrays. When normalized geodesic
    fields are provided in ``centerline_data``, ring centroids are sampled
    along the crypt axis and collectively fit one quadratic Bézier segment.
    """
    vertices = as_points(vertices)
    centerline_data = dict(centerline_data or {})
    out = {}
    for key, component in crypt_components.items():
        points = component_points(vertices, component)
        for attachment_id, path in _resolve_crypt_paths(graph, key):
            graph_centerline = np.vstack(
                [graph.node(node_id).position for node_id in path]
            )
            centerline = graph_centerline
            centerline_metadata = {
                "method": "straight_skeleton_path",
                "control_points": graph_centerline,
                "control_parameters": np.linspace(0.0, 1.0, graph_centerline.shape[0]),
            }
            data = centerline_data.get(key)
            if data is None and path[-1] in centerline_data:
                data = centerline_data[path[-1]]
            path_constrictions = [
                graph.node(node_id).position
                for node_id in path
                if graph.node(node_id).node_type == "constriction"
            ]

            if smooth_centerline and data is not None and data.get("distance_field") is not None:
                indices = data.get("vertex_indices", component)
                profile = data.get("neck_profile")
                constriction_level = (
                    profile.get("constriction_level")
                    if isinstance(profile, dict)
                    else None
                )
                try:
                    centerline_metadata = estimate_smooth_crypt_centerline(
                        vertices,
                        indices,
                        data["distance_field"],
                        graph_centerline[0],
                        graph_centerline[-1],
                        neck_level=float(data.get("neck_level", 1.0)),
                        n_bands=centerline_n_bands,
                        n_samples=centerline_n_samples,
                        constriction_position=(
                            path_constrictions[0] if path_constrictions else None
                        ),
                        constriction_level=constriction_level,
                        constriction_weight=centerline_constriction_weight,
                    )
                    centerline = centerline_metadata["centerline_points"]
                except ValueError as exc:
                    centerline_metadata = {
                        **centerline_metadata,
                        "fallback_reason": str(exc),
                    }
            elif smooth_centerline and graph_centerline.shape[0] >= 3:
                control = np.mean(graph_centerline[1:-1], axis=0)
                centerline = sample_quadratic_bezier(
                    graph_centerline[0],
                    control,
                    graph_centerline[-1],
                    n_samples=centerline_n_samples,
                )
                centerline_metadata = {
                    "method": "quadratic_bezier_from_skeleton_path",
                    "control_points": np.vstack(
                        [graph_centerline[0], control, graph_centerline[-1]]
                    ),
                    "control_parameters": np.array([0.0, 0.5, 1.0]),
                }

            if update_crypt_nodes and smooth_centerline:
                midpoint = point_at_polyline_arclength(centerline, 0.5)
                for node_id in path[1:-1]:
                    node = graph.node(node_id)
                    if node.node_type != "crypt":
                        continue
                    node.metadata.setdefault(
                        "position_before_centerline_refinement",
                        node.position.tolist(),
                    )
                    node.position = midpoint.copy()
                    node.metadata.update(
                        {
                            "position_refined_from_smooth_centerline": True,
                            "centerline_attachment_id": attachment_id,
                        }
                    )

            local_fit_kwargs = dict(fit_kwargs)
            if path_constrictions and "constriction_s" not in local_fit_kwargs:
                projected = project_points_to_polyline(
                    np.asarray(path_constrictions[:1], dtype=float),
                    centerline,
                )
                local_fit_kwargs["constriction_s"] = float(projected["s"][0])

            fit = fit_crypt_tube_to_points(
                points,
                centerline,
                path_node_ids=path,
                metadata={
                    "component": "crypt",
                    "component_key": str(key),
                    "centerline_method": centerline_metadata["method"],
                    "centerline_control_points": centerline_metadata["control_points"],
                    "centerline_control_parameters": centerline_metadata[
                        "control_parameters"
                    ],
                    "centerline_band_sizes": centerline_metadata.get("band_sizes", []),
                    "centerline_constriction_used": centerline_metadata.get(
                        "constriction_used",
                        False,
                    ),
                    "centerline_constriction_parameter": centerline_metadata.get(
                        "constriction_parameter"
                    ),
                    "centerline_constriction_weight": centerline_metadata.get(
                        "constriction_weight"
                    ),
                    "centerline_fallback_reason": centerline_metadata.get(
                        "fallback_reason"
                    ),
                },
                **local_fit_kwargs,
            )
            attachment = fit.to_attachment(
                attachment_type="path",
                attachment_id=attachment_id,
                target_ids=path,
            )
            graph.add_primitive_attachment(attachment_id, attachment)
            out[attachment_id] = attachment
    return out

def primitive_attachments_to_dataframe(graph: SkeletonGraph):
    """Return graph-level primitive attachments as a pandas DataFrame."""
    return graph.to_primitive_dataframe()


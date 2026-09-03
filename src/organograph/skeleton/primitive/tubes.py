"""Crypt tapered-tube primitive fitting and graph attachment."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from organograph.skeleton.datatypes import SkeletonGraph
from organograph.skeleton.geometry import as_points
from organograph.skeleton.primitive.blobs import (
    constrain_blob_fit_surface_radius,
    fit_blob_primitive_to_points,
)
from organograph.skeleton.primitive.common import _residual_summary
from organograph.skeleton.primitive.crypt_geometry import fit_crypt_geometry
from organograph.skeleton.primitive.radius_profiles import (
    RadiusProfileObservations,
    fitted_radius_volume_center,
    fit_interpretable_radius_profile,
)
from organograph.skeleton.primitive_geometry import (
    bend_angles_for_polyline,
    capped_tube_radius,
    component_points,
    point_at_polyline_arclength,
    polyline_lengths,
    project_points_to_polyline,
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

def fit_crypt_tube_to_points(
    points,
    centerline_points,
    *,
    path_node_ids: list[str] | None = None,
    center_s: float = 0.5,
    radius_quantile: float = 0.5,
    neck_window: tuple[float, float] = (0.0, 0.05),
    tip_window: tuple[float, float] | None = None,
    optimize_radius_profile: bool = True,
    fixed_taper_position: float = 0.85,
    outside_volume_weight: float = 2.0,
    constriction_s: float | None = None,
    constriction_window_half_width: float = 0.04,
    profile_n_bins: int = 20,
    profile_min_points_per_bin: int = 2,
    profile_min_supported_bins: int = 6,
    max_constriction_to_neighbor_fraction: float = 0.98,
    tip_projection_tolerance: float = 1e-6,
    opening_normal=None,
    opening_frame_blend_fraction: float = 0.15,
    contour_s=None,
    contour_radii=None,
    volume_center_max_iterations: int = 8,
    volume_center_tolerance: float = 1e-4,
    metadata: dict[str, Any] | None = None,
) -> PrimitiveFit:
    """Fit a tapered capped tube to crypt component points.

    The supplied centerline is represented by dense straight samples of the
    endpoint-normal constrained Hermite curve. Maintained workflows provide contour-derived radius
    observations, avoiding dependence on mesh tessellation density. Direct
    point-cloud callers fall back to equal-arclength bins. The distal cap onset
    is fixed, a detected constriction remains a local radius minimum, and the
    profile closes smoothly at the crypt-tip node.
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
    normalized_opening = None
    if opening_normal is not None:
        normalized_opening = np.asarray(opening_normal, dtype=float)
        if normalized_opening.shape != (3,) or not np.all(np.isfinite(normalized_opening)):
            raise ValueError("opening_normal must be a finite 3-vector")
        opening_norm = float(np.linalg.norm(normalized_opening))
        if opening_norm <= 1e-12:
            raise ValueError("opening_normal must be non-zero")
        normalized_opening = normalized_opening / opening_norm
    frame_blend = float(opening_frame_blend_fraction)
    if not 0.0 < frame_blend <= 1.0:
        raise ValueError("opening_frame_blend_fraction must lie in (0, 1]")
    observations_override = None
    if contour_s is not None and contour_radii is not None:
        observed_s = np.asarray(contour_s, dtype=float).reshape(-1)
        observed_radii = np.asarray(contour_radii, dtype=float).reshape(-1)
        valid = (
            np.isfinite(observed_s)
            & np.isfinite(observed_radii)
            & (observed_radii >= 0.0)
        )
        observed_s = observed_s[valid]
        observed_radii = observed_radii[valid]
        observations_override = RadiusProfileObservations(
            s=observed_s,
            radii=observed_radii,
            counts=np.ones(observed_s.size, dtype=np.int64),
            bin_edges=np.linspace(0.0, 1.0, max(observed_s.size + 1, 2)),
            n_input_points=int(observed_s.size),
            n_excluded_tip_points=0,
        )
    initial_center_s = float(center_s)
    current_center_s = initial_center_s
    center_history = []
    profile_fit = None
    maximum_iterations = max(1, int(volume_center_max_iterations))
    tolerance = max(float(volume_center_tolerance), 0.0)
    lower_center_bound = 1e-3
    if constriction_s is not None and np.isfinite(float(constriction_s)):
        lower_center_bound = max(lower_center_bound, float(constriction_s) + 1e-3)
    upper_center_bound = float(fixed_taper_position) - 1e-3
    for iteration in range(maximum_iterations):
        profile_fit = fit_interpretable_radius_profile(
            distances,
            s,
            s_center=current_center_s,
            fixed_taper_position=fixed_taper_position,
            radius_quantile=radius_quantile,
            neck_window=neck_window,
            tip_window=tip_window,
            optimize=bool(optimize_radius_profile and pts.shape[0] >= 10),
            outside_volume_weight=outside_volume_weight,
            constriction_s=constriction_s,
            constriction_window_half_width=constriction_window_half_width,
            n_bins=profile_n_bins,
            min_points_per_bin=profile_min_points_per_bin,
            min_supported_bins=profile_min_supported_bins,
            max_constriction_to_neighbor_fraction=max_constriction_to_neighbor_fraction,
            tip_projection_tolerance=tip_projection_tolerance,
            observations_override=observations_override,
        )
        volume_center_s = fitted_radius_volume_center(
            r_attachment=profile_fit.r_attachment,
            r_center=profile_fit.r_center,
            r_distal=profile_fit.r_distal,
            s_center=profile_fit.s_center,
            s_taper=profile_fit.s_taper,
            r_constriction=profile_fit.r_constriction,
            s_constriction=profile_fit.s_constriction,
        )
        volume_center_s = float(
            np.clip(volume_center_s, lower_center_bound, upper_center_bound)
        )
        center_history.append(
            {
                "iteration": iteration,
                "profile_center_s": float(current_center_s),
                "volume_center_s": volume_center_s,
            }
        )
        if abs(volume_center_s - current_center_s) <= tolerance:
            current_center_s = volume_center_s
            break
        current_center_s = volume_center_s
    assert profile_fit is not None
    if abs(profile_fit.s_center - current_center_s) > tolerance:
        profile_fit = fit_interpretable_radius_profile(
            distances,
            s,
            s_center=current_center_s,
            fixed_taper_position=fixed_taper_position,
            radius_quantile=radius_quantile,
            neck_window=neck_window,
            tip_window=tip_window,
            optimize=bool(optimize_radius_profile and pts.shape[0] >= 10),
            outside_volume_weight=outside_volume_weight,
            constriction_s=constriction_s,
            constriction_window_half_width=constriction_window_half_width,
            n_bins=profile_n_bins,
            min_points_per_bin=profile_min_points_per_bin,
            min_supported_bins=profile_min_supported_bins,
            max_constriction_to_neighbor_fraction=max_constriction_to_neighbor_fraction,
            tip_projection_tolerance=tip_projection_tolerance,
            observations_override=observations_override,
        )
    r_neck = profile_fit.r_neck
    r_body = profile_fit.r_body
    r_tip = profile_fit.r_distal
    r_constriction = profile_fit.r_constriction
    constriction_position = profile_fit.s_constriction
    body_s = profile_fit.s_center
    taper_start = profile_fit.s_taper
    volume_center_s = fitted_radius_volume_center(
        r_attachment=r_neck,
        r_center=r_body,
        r_distal=r_tip,
        s_center=body_s,
        s_taper=taper_start,
        r_constriction=r_constriction,
        s_constriction=constriction_position,
    )
    volume_center_s = float(
        np.clip(volume_center_s, lower_center_bound, upper_center_bound)
    )

    predicted = capped_tube_radius(
        s,
        r_neck,
        r_body,
        r_tip,
        center_s=body_s,
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
            "r_center": r_body,
            "r_tip": r_tip,
            "r_taper": r_tip,
            "r_distal": r_tip,
            "r_constriction": r_constriction,
            "s_constriction": constriction_position,
            "s_body": body_s,
            "s_center": body_s,
            "crypt_node_s": volume_center_s,
            "s_taper": taper_start,
            "radius_quantile": float(radius_quantile),
            "radius_profile": "semantic_landmarks_shape_preserving_squared_radius_v2",
            "distal_taper_start": taper_start,
            "distal_taper": "smooth_squared_radius_to_zero",
            "cap": "integrated_squared_radius_closure",
            "neck_window": neck_window,
            "tip_window": tip_window,
            "profile_safeguards": {
                "fixed_taper_position": float(taper_start),
                "outside_volume_weight": float(outside_volume_weight),
                "max_constriction_to_neighbor_fraction": float(
                    max_constriction_to_neighbor_fraction
                ),
                "tip_projection_tolerance": float(tip_projection_tolerance),
            },
            "opening_normal": normalized_opening,
            "opening_frame_blend_fraction": frame_blend,
        },
        fit_error=summary["rmse"],
        residuals=summary,
        derived_parameters=derived,
        metadata={
            "fit_method": "semantic_landmark_radii_with_asymmetric_volume_proxy",
            "n_points": int(pts.shape[0]),
            "profile_observations": {
                "s": profile_fit.observations.s,
                "radii": profile_fit.observations.radii,
                "counts": profile_fit.observations.counts,
                "bin_edges": profile_fit.observations.bin_edges,
            },
            "profile_optimization": profile_fit.diagnostics,
            "volume_center": {
                "source": "fitted_radius_squared_volume_centroid",
                "initial_mesh_center_s": initial_center_s,
                "final_profile_center_s": float(body_s),
                "crypt_node_s": volume_center_s,
                "iterations": center_history,
            },
            **dict(metadata or {}),
        },
    )

def _descendant_tip_radius_constraints(
    graph: SkeletonGraph,
    host_node_id: str,
    center,
    *,
    margin_fraction: float = 0.02,
) -> list[dict[str, Any]]:
    """Collect fitting-time radius bounds from host center toward crypt tips.

    For the main body, traversal stops at branch nodes so daughter crypts
    constrain the branch primitive rather than collapsing the body primitive.
    """
    host_node_id = str(host_node_id)
    if host_node_id not in graph.nodes:
        return []

    center = np.asarray(center, dtype=float).reshape(-1)
    if center.size != 3 or not np.all(np.isfinite(center)):
        return []

    children: dict[str, list[str]] = {}
    for edge in graph.edges.values():
        children.setdefault(edge.source, []).append(edge.target)

    host_type = graph.node(host_node_id).node_type
    constraints = []
    stack = list(children.get(host_node_id, []))
    seen = set()
    while stack:
        node_id = str(stack.pop())
        if node_id in seen:
            continue
        seen.add(node_id)
        node = graph.node(node_id)
        if node.node_type == "tip":
            direction = np.asarray(node.position, dtype=float) - center
            distance = float(np.linalg.norm(direction))
            if distance > 1e-12:
                constraints.append(
                    {
                        "tip_node_id": node_id,
                        "direction": direction / distance,
                        "max_radius": distance
                        * max(0.0, 1.0 - float(margin_fraction)),
                    }
                )
            continue
        if host_type != "branch" and node.node_type == "branch":
            continue
        stack.extend(children.get(node_id, []))
    return constraints


def _orthonormal_basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    normal = np.cross(axis, reference)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    binormal = np.cross(axis, normal)
    binormal = binormal / max(float(np.linalg.norm(binormal)), 1e-12)
    return axis, normal, binormal


def _first_child_of_type(
    children: dict[str, list[str]],
    graph: SkeletonGraph,
    node_id: str,
    node_types: tuple[str, ...],
) -> str | None:
    candidates = list(children.get(str(node_id), []))
    for node_type in node_types:
        for candidate in candidates:
            if graph.node(candidate).node_type == node_type:
                return candidate
    return None


def _attachment_cap_support_points(
    graph: SkeletonGraph,
    host_node_id: str,
    *,
    points_per_attachment: int = 64,
    radius_fraction: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate sparse unweighted support caps for host-side crypt openings.

    The cap is a simple dome from an attachment node to the next crypt-side
    skeleton node.  It is synthetic support for fitting the host blob, not mesh
    data and not a visualization primitive.
    """
    children: dict[str, list[str]] = {}
    for edge in graph.edges.values():
        children.setdefault(edge.source, []).append(edge.target)

    support = []
    attachment_ids = []
    n_per = max(1, int(points_per_attachment))
    n_rings = max(2, min(6, int(np.sqrt(n_per))))
    n_theta = max(4, int(np.ceil(max(n_per - 1, 1) / n_rings)))
    angles = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    s_values = np.linspace(0.0, 0.92, n_rings)

    for attachment_id in children.get(str(host_node_id), []):
        attachment_node = graph.node(attachment_id)
        if attachment_node.node_type != "attachment":
            continue
        next_id = _first_child_of_type(
            children,
            graph,
            attachment_id,
            ("constriction", "crypt", "bend", "tip"),
        )
        if next_id is None:
            continue
        next_node = graph.node(next_id)
        start = np.asarray(attachment_node.position, dtype=float)
        end = np.asarray(next_node.position, dtype=float)
        axis_vector = end - start
        length = float(np.linalg.norm(axis_vector))
        if length <= 1e-12:
            continue

        axis, normal, binormal = _orthonormal_basis_from_axis(axis_vector)
        base_radius = max(float(radius_fraction), 0.0) * length
        if base_radius <= 1e-12:
            continue

        for s in s_values:
            center = start + float(s) * length * axis
            radius = base_radius * np.sqrt(max(1.0 - float(s) ** 2, 0.0))
            for angle in angles:
                support.append(
                    center
                    + radius
                    * (np.cos(angle) * normal + np.sin(angle) * binormal)
                )
        support.append(end)
        attachment_ids.append(str(attachment_id))

    if support:
        support_points = np.asarray(support, dtype=float)
    else:
        support_points = np.empty((0, 3), dtype=float)
    metadata = {
        "enabled": True,
        "n_points": int(support_points.shape[0]),
        "n_attachments": len(attachment_ids),
        "attachment_ids": attachment_ids,
        "points_per_attachment": int(points_per_attachment),
        "radius_fraction": float(radius_fraction),
        "method": "unweighted_dome_attachment_to_next_crypt_node",
    }
    return support_points, metadata


def attach_body_primitive(
    graph: SkeletonGraph,
    vertices,
    component=None,
    *,
    primitive_type: str = "ellipsoid",
    add_attachment_cap_support: bool = True,
    cap_support_points_per_attachment: int = 64,
    cap_support_radius_fraction: float = 0.5,
    constrain_to_descendant_tips: bool = True,
    tip_constraint_margin_fraction: float = 0.02,
    tip_constraint_weight: float = 50.0,
    metadata: dict[str, Any] | None = None,
    **fit_kwargs,
) -> PrimitiveAttachment:
    """Fit and attach a body blob primitive to the body node."""
    points = component_points(vertices, component)
    fit_points = points
    cap_metadata = {
        "enabled": bool(add_attachment_cap_support),
        "n_points": 0,
        "n_attachments": 0,
    }
    if add_attachment_cap_support:
        cap_points, cap_metadata = _attachment_cap_support_points(
            graph,
            graph.body_node().node_id,
            points_per_attachment=cap_support_points_per_attachment,
            radius_fraction=cap_support_radius_fraction,
        )
        if cap_points.size:
            fit_points = np.vstack([points, cap_points])

    fit = fit_blob_primitive_to_points(
        fit_points,
        primitive_type=primitive_type,
        metadata={
            "component": "body",
            "n_real_points": int(points.shape[0]),
            "attachment_cap_support": cap_metadata,
            **dict(metadata or {}),
        },
        **fit_kwargs,
    )
    if constrain_to_descendant_tips:
        fit = constrain_blob_fit_surface_radius(
            fit,
            fit_points,
            _descendant_tip_radius_constraints(
                graph,
                graph.body_node().node_id,
                fit.parameters["center"],
                margin_fraction=tip_constraint_margin_fraction,
            ),
            constraint_weight=tip_constraint_weight,
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
    constrain_to_descendant_tips: bool = True,
    tip_constraint_margin_fraction: float = 0.02,
    tip_constraint_weight: float = 50.0,
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
        if constrain_to_descendant_tips:
            fit = constrain_blob_fit_surface_radius(
                fit,
                points,
                _descendant_tip_radius_constraints(
                    graph,
                    branch_node_id,
                    fit.parameters["center"],
                    margin_fraction=tip_constraint_margin_fraction,
                ),
                constraint_weight=tip_constraint_weight,
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
    mesh=None,
    geodesic_fn=None,
    geodesic_kwargs: dict[str, Any] | None = None,
    centerline_n_contours: int = 10,
    centerline_n_samples: int = 64,
    opening_frame_blend_fraction: float = 0.15,
    update_crypt_nodes: bool = True,
    **fit_kwargs,
) -> dict[str, PrimitiveAttachment]:
    """Fit one ratio-contour/Hermite tube for each terminal crypt.

    Tip selection is already final at detection time. The primitive stage has
    no competing tip or spline candidates: it derives one boundary-to-tip
    coordinate, measures its contours, and fits one endpoint-normal constrained
    curve. ``geodesic_fn`` and ``geodesic_kwargs`` are accepted only so callers
    can pass a shared workflow context; restricted mesh distances are computed
    internally and cannot shortcut outside the crypt component.
    """
    vertices = as_points(vertices)
    if mesh is None or not hasattr(mesh, "f"):
        raise ValueError("mesh with triangular faces is required for crypt contour fitting")
    centerline_data = dict(centerline_data or {})
    out = {}
    for key, component in crypt_components.items():
        points = component_points(vertices, component)
        for attachment_id, path in _resolve_crypt_paths(graph, key):
            data = centerline_data.get(key)
            if data is None and path[-1] in centerline_data:
                data = centerline_data[path[-1]]
            data = dict(data or {})
            crypt_nodes = [
                graph.node(node_id)
                for node_id in path[1:-1]
                if graph.node(node_id).node_type == "crypt"
            ]
            tip_node = graph.node(path[-1])
            tip_vertex_id = data.get("hks_tip_vertex_id")
            if tip_vertex_id is None:
                tip_vertex_id = int(
                    np.argmin(np.linalg.norm(vertices - tip_node.position[None, :], axis=1))
                )
            geometry = fit_crypt_geometry(
                vertices,
                np.asarray(mesh.f, dtype=np.int64),
                data.get("vertex_indices", component),
                graph.node(path[0]).position,
                tip_vertex_id,
                boundary_vertices=data.get("candidate_boundary_vertices"),
                opening_normal=data.get("attachment_surface_normal"),
                n_contours=centerline_n_contours,
                n_samples=centerline_n_samples,
            )
            local_fit_kwargs = dict(fit_kwargs)
            local_fit_kwargs["center_s"] = geometry.initial_center_s
            local_fit_kwargs["opening_normal"] = geometry.opening_normal
            local_fit_kwargs["opening_frame_blend_fraction"] = float(opening_frame_blend_fraction)
            profile = data.get("neck_profile")
            constriction_position = data.get("constriction_position")
            if isinstance(profile, dict) and profile.get("kind") == "constriction" and constriction_position is not None:
                local_fit_kwargs["constriction_s"] = float(
                    project_points_to_polyline(
                        np.asarray(constriction_position, dtype=float).reshape(1, 3),
                        geometry.centerline_points,
                    )["s"][0]
                )
            fit = fit_crypt_tube_to_points(
                points,
                geometry.centerline_points,
                path_node_ids=path,
                contour_s=geometry.contour_s,
                contour_radii=geometry.contour_radii,
                metadata={
                    "component": "crypt",
                    "component_key": str(key),
                    "tip_source": "detection_final_tip",
                    "tip_vertex_id": tip_vertex_id,
                    "centerline_kind": geometry.centerline_kind,
                    "centerline_method": geometry.metadata["method"],
                    "centerline_start_tangent": geometry.start_tangent,
                    "centerline_end_tangent": geometry.end_tangent,
                    "centerline_tangent_length": geometry.tangent_length,
                    "centerline_fit_rmse": geometry.metadata["centerline_fit_rmse"],
                    "opening_normal_source": "host_primitive_surface_gradient",
                    "tip_normal": geometry.tip_normal,
                    "ratio_contours": {
                        "s": geometry.contour_s,
                        "centers": geometry.contour_centers,
                        "radii": geometry.contour_radii,
                        "areas": geometry.contour_areas,
                        "perimeters": geometry.contour_perimeters,
                    },
                },
                **local_fit_kwargs,
            )
            tip_node.position = vertices[int(tip_vertex_id)].copy()
            if update_crypt_nodes:
                crypt_node_s = float(fit.parameters["crypt_node_s"])
                crypt_center = point_at_polyline_arclength(
                    geometry.centerline_points, crypt_node_s
                )
                for node in crypt_nodes:
                    node.position = crypt_center.copy()
                    node.metadata.update(
                        {
                            "position_source": "fitted_tube_volume_center",
                            "centerline_attachment_id": attachment_id,
                            "centerline_s": crypt_node_s,
                            "initial_mesh_centerline_s": geometry.initial_center_s,
                        }
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

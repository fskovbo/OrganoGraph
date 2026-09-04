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
from organograph.skeleton.primitive.crypt_geometry import (
    centerline_radius_observations,
    fit_crypt_geometry,
)
from organograph.skeleton.primitive.radius_profiles import (
    RadiusProfileObservations,
    fitted_radius_volume_center,
    fit_fixed_grid_radius_profile,
)
from organograph.skeleton.primitive.radius_support import (
    grow_crypt_radius_support_regions,
)
from organograph.skeleton.primitive_geometry import (
    CRYPT_RADIUS_PROFILE_TYPE,
    bend_angles_for_polyline,
    component_points,
    fixed_grid_tube_radius,
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
    radius_quantile: float = 0.5,
    optimize_radius_profile: bool = True,
    radius_control_s=None,
    fixed_taper_position: float = 0.85,
    radius_profile_smoothness_weight: float = 0.05,
    outside_volume_weight: float = 2.0,
    profile_n_bins: int = 20,
    profile_min_points_per_bin: int = 2,
    profile_min_supported_bins: int = 6,
    tip_projection_tolerance: float = 1e-6,
    opening_normal=None,
    opening_frame_blend_fraction: float = 0.15,
    radius_observations: RadiusProfileObservations | None = None,
    contour_s=None,
    contour_radii=None,
    metadata: dict[str, Any] | None = None,
) -> PrimitiveFit:
    """Fit a fixed-grid tapered tube around a fitted crypt centerline.

    The centerline and radius profile are independent compact parameterizations.
    Eight fixed longitudinal radius controls describe the tube; a deterministic
    shape-preserving cap closes the final control to zero at the tip. The crypt
    skeleton node is derived from the fitted tube's volume centroid.
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
    observations_override = radius_observations
    if observations_override is not None and not isinstance(
        observations_override, RadiusProfileObservations
    ):
        raise TypeError("radius_observations must be RadiusProfileObservations")
    if observations_override is None and contour_s is not None and contour_radii is not None:
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
            weights=np.ones(observed_s.size, dtype=float),
            section_s=observed_s,
            section_mean_radii=observed_radii,
            section_min_radii=observed_radii,
        )
    profile_fit = fit_fixed_grid_radius_profile(
        distances,
        s,
        radius_control_s=radius_control_s,
        fixed_taper_position=fixed_taper_position,
        radius_profile_smoothness_weight=radius_profile_smoothness_weight,
        radius_quantile=radius_quantile,
        optimize=bool(optimize_radius_profile and pts.shape[0] >= 10),
        outside_volume_weight=outside_volume_weight,
        n_bins=profile_n_bins,
        min_points_per_bin=profile_min_points_per_bin,
        min_supported_bins=profile_min_supported_bins,
        tip_projection_tolerance=tip_projection_tolerance,
        observations_override=observations_override,
    )
    control_s = profile_fit.control_s
    control_radii = profile_fit.control_radii
    taper_start = profile_fit.s_taper
    volume_center_s = fitted_radius_volume_center(
        control_s=control_s,
        control_radii=control_radii,
    )
    predicted = fixed_grid_tube_radius(s, control_s, control_radii)
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
    dense_s = np.linspace(0.0, 1.0, 513)
    dense_radii = fixed_grid_tube_radius(dense_s, control_s, control_radii)
    maximum_radius = float(np.max(dense_radii))
    distal_radius = float(control_radii[-1])
    minimum_pre_cap_radius = float(np.min(control_radii))
    derived = {
        "length": float(length),
        "straight_distance": straight,
        "tortuosity": float(length / straight) if straight > 1e-12 else float("nan"),
        "bend_angle": bend_angle,
        "bend_angles": bend_angles,
        "constriction_ratio": (
            minimum_pre_cap_radius / maximum_radius
            if maximum_radius > 1e-12
            else float("nan")
        ),
        "taper_ratio": (
            distal_radius / maximum_radius
            if maximum_radius > 1e-12
            else float("nan")
        ),
        "maximum_radius": maximum_radius,
    }
    return PrimitiveFit(
        primitive_type="tapered_capped_tube",
        parameters={
            "centerline_points": centerline,
            "path_node_ids": list(path_node_ids or []),
            "radius_control_s": control_s,
            "radius_control_radii": control_radii,
            "crypt_node_s": volume_center_s,
            "s_taper": taper_start,
            "radius_quantile": float(radius_quantile),
            "radius_profile": CRYPT_RADIUS_PROFILE_TYPE,
            "distal_taper_start": taper_start,
            "distal_taper": "smooth_squared_radius_to_zero",
            "cap": "integrated_squared_radius_closure",
            "profile_safeguards": {
                "fixed_taper_position": float(taper_start),
                "outside_volume_weight": float(outside_volume_weight),
                "radius_profile_smoothness_weight": float(
                    radius_profile_smoothness_weight
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
            "fit_method": "fixed_grid_radii_with_asymmetric_volume_proxy",
            "n_points": int(pts.shape[0]),
            "profile_observations": {
                "s": profile_fit.observations.s,
                "radii": profile_fit.observations.radii,
                "weights": profile_fit.observations.weights,
                "counts": profile_fit.observations.counts,
                "bin_edges": profile_fit.observations.bin_edges,
                "section_s": profile_fit.observations.section_s,
                "section_mean_radii": profile_fit.observations.section_mean_radii,
                "section_min_radii": profile_fit.observations.section_min_radii,
            },
            "profile_optimization": profile_fit.diagnostics,
            "volume_center": {
                "source": "fitted_radius_squared_volume_centroid",
                "crypt_node_s": volume_center_s,
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


def _radius_profile_observations(sections):
    """Convert centerline cross-sections into equally weighted fit samples."""
    if len(sections) < 3:
        return None
    section_s = np.asarray([item["s"] for item in sections], dtype=float)
    sample_s = np.concatenate(
        [np.full(item["radii"].size, item["s"], dtype=float) for item in sections]
    )
    samples = np.concatenate(
        [np.asarray(item["radii"], dtype=float) for item in sections]
    )
    weights = np.concatenate(
        [np.asarray(item["weights"], dtype=float) for item in sections]
    )
    return RadiusProfileObservations(
        s=sample_s,
        radii=samples,
        weights=weights,
        counts=np.ones(samples.size, dtype=np.int64),
        bin_edges=np.linspace(0.0, 1.0, len(sections) + 1),
        n_input_points=int(samples.size),
        n_excluded_tip_points=0,
        section_s=section_s,
        section_mean_radii=np.asarray(
            [item["mean_radius"] for item in sections], dtype=float
        ),
        section_min_radii=np.asarray(
            [item["min_radius"] for item in sections], dtype=float
        ),
    )


def _radius_fit_sections(sections, *, exclude_attachment):
    """Select measured sections used by the profile objective."""
    if not exclude_attachment:
        return list(sections)
    return [item for item in sections if float(item["s"]) > 1e-8]


def _radius_observation_metadata(sections):
    return {
        "s": np.asarray([item["s"] for item in sections], dtype=float),
        "mean_radii": np.asarray(
            [item["mean_radius"] for item in sections], dtype=float
        ),
        "min_radii": np.asarray(
            [item["min_radius"] for item in sections], dtype=float
        ),
        "perimeters": np.asarray(
            [item["perimeter"] for item in sections], dtype=float
        ),
        "counts": np.asarray([item["count"] for item in sections], dtype=np.int64),
        "sources": [str(item.get("source", "unknown")) for item in sections],
        "coordinate": "centerline_normalized_arclength",
    }


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
    radius_n_contours: int = 19,
    centerline_n_samples: int = 64,
    centerline_curvature_weight: float = 0.0,
    centerline_reference_length: float | None = None,
    radius_support_protected_mask=None,
    radius_support_max_distance_factor: float = 1.5,
    exclude_attachment_radius_observation: bool = True,
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
    faces = np.asarray(mesh.f, dtype=np.int64)
    centerline_data = dict(centerline_data or {})
    records = []
    for key, component in crypt_components.items():
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
                faces,
                data.get("vertex_indices", component),
                graph.node(path[0]).position,
                tip_vertex_id,
                boundary_vertices=data.get("candidate_boundary_vertices"),
                opening_normal=data.get("attachment_surface_normal"),
                n_contours=centerline_n_contours,
                radius_n_contours=radius_n_contours,
                n_samples=centerline_n_samples,
                curvature_weight=centerline_curvature_weight,
                reference_length=centerline_reference_length,
            )
            region = np.unique(
                np.asarray(data.get("vertex_indices", component), dtype=np.int64).reshape(-1)
            )
            records.append(
                {
                    "key": key,
                    "attachment_id": attachment_id,
                    "path": path,
                    "data": data,
                    "crypt_nodes": crypt_nodes,
                    "tip_node": tip_node,
                    "tip_vertex_id": int(tip_vertex_id),
                    "geometry": geometry,
                    "region": region,
                }
            )

    support_regions = {
        record["attachment_id"]: record["region"] for record in records
    }
    support_diagnostics = {
        record["attachment_id"]: {
            "enabled": False,
            "original_vertices": int(record["region"].size),
            "final_vertices": int(record["region"].size),
            "added_vertices": 0,
        }
        for record in records
    }
    if records and radius_support_protected_mask is not None:
        support = grow_crypt_radius_support_regions(
            vertices,
            faces,
            support_regions,
            {
                record["attachment_id"]: record["tip_vertex_id"]
                for record in records
            },
            {
                record["attachment_id"]: polyline_lengths(
                    record["geometry"].centerline_points
                )[2]
                for record in records
            },
            radius_support_protected_mask,
            max_distance_factor=radius_support_max_distance_factor,
        )
        support_regions = support.regions
        support_diagnostics = {
            key: {"enabled": True, **value}
            for key, value in support.diagnostics.items()
        }

    out = {}
    for record in records:
        key = record["key"]
        attachment_id = record["attachment_id"]
        path = record["path"]
        data = record["data"]
        crypt_nodes = record["crypt_nodes"]
        tip_node = record["tip_node"]
        tip_vertex_id = record["tip_vertex_id"]
        geometry = record["geometry"]
        support_region = support_regions[attachment_id]
        points = component_points(vertices, support_region)
        radius_sections, _ = centerline_radius_observations(
            vertices,
            faces,
            support_region,
            geometry.centerline_points,
            data.get("candidate_boundary_vertices"),
            n_contours=radius_n_contours,
            max_s=0.95,
        )
        fit_radius_sections = _radius_fit_sections(
            radius_sections,
            exclude_attachment=exclude_attachment_radius_observation,
        )
        radius_observations = _radius_profile_observations(fit_radius_sections)
        if radius_observations is None:
            radius_sections = [
                {
                    "s": float(s),
                    "radii": np.asarray([mean_radius], dtype=float),
                    "weights": np.ones(1, dtype=float),
                    "mean_radius": float(mean_radius),
                    "min_radius": float(min_radius),
                    "perimeter": float(perimeter),
                    "count": int(count),
                    "source": "original_component_fallback",
                }
                for s, mean_radius, min_radius, perimeter, count in zip(
                    geometry.radius_contour_s,
                    geometry.radius_mean_radii,
                    geometry.radius_min_radii,
                    geometry.radius_contour_perimeters,
                    geometry.radius_contour_counts,
                    )
                ]
            fit_radius_sections = _radius_fit_sections(
                radius_sections,
                exclude_attachment=exclude_attachment_radius_observation,
            )
            radius_observations = _radius_profile_observations(
                fit_radius_sections
            )
        local_fit_kwargs = dict(fit_kwargs)
        local_fit_kwargs["opening_normal"] = geometry.opening_normal
        local_fit_kwargs["opening_frame_blend_fraction"] = float(opening_frame_blend_fraction)
        fit = fit_crypt_tube_to_points(
            points,
            geometry.centerline_points,
            path_node_ids=path,
            radius_observations=radius_observations,
            metadata={
                "component": "crypt",
                "component_key": str(key),
                "tip_source": "detection_final_tip",
                "tip_vertex_id": tip_vertex_id,
                "centerline_kind": geometry.centerline_kind,
                "centerline_method": geometry.metadata["method"],
                "centerline_start_tangent": geometry.start_tangent,
                "centerline_end_tangent": geometry.end_tangent,
                "centerline_start_tangent_length": geometry.start_tangent_length,
                "centerline_end_tangent_length": geometry.end_tangent_length,
                "centerline_fit_rmse": geometry.metadata["centerline_fit_rmse"],
                "centerline_normalized_data_mse": geometry.metadata[
                    "centerline_normalized_data_mse"
                ],
                "centerline_bending_energy": geometry.metadata[
                    "centerline_bending_energy"
                ],
                "centerline_dimensionless_bending_energy": geometry.metadata[
                    "centerline_dimensionless_bending_energy"
                ],
                "centerline_total_bend_angle": geometry.metadata[
                    "centerline_total_bend_angle"
                ],
                "centerline_max_curvature": geometry.metadata[
                    "centerline_max_curvature"
                ],
                "centerline_p95_curvature": geometry.metadata[
                    "centerline_p95_curvature"
                ],
                "centerline_curvature_localization": geometry.metadata[
                    "centerline_curvature_localization"
                ],
                "centerline_fold_penalty": geometry.metadata[
                    "centerline_fold_penalty"
                ],
                "centerline_objective": geometry.metadata[
                    "centerline_objective"
                ],
                "centerline_reference_length": geometry.metadata[
                    "centerline_reference_length"
                ],
                "centerline_curvature_weight": geometry.metadata[
                    "centerline_curvature_weight"
                ],
                "opening_normal_source": data.get(
                    "attachment_normal_source",
                    "host_primitive_surface_gradient",
                ),
                "tip_normal": geometry.tip_normal,
                "ratio_contours": {
                    "s": geometry.contour_s,
                    "centers": geometry.contour_centers,
                    "radii": geometry.contour_radii,
                    "min_radii": geometry.contour_min_radii,
                    "areas": geometry.contour_areas,
                    "perimeters": geometry.contour_perimeters,
                    "diagnostic_s": geometry.diagnostic_s,
                    "diagnostic_radii": geometry.diagnostic_radii,
                    "diagnostic_min_radii": geometry.diagnostic_min_radii,
                },
                "centerline_radius_contours": {
                    **_radius_observation_metadata(radius_sections),
                    "attachment_observation_used_for_fit": bool(
                        not exclude_attachment_radius_observation
                    ),
                    "n_fit_sections": len(fit_radius_sections),
                },
                "radius_support": support_diagnostics[attachment_id],
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

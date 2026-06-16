"""Primitive fitting for biology-aware organoid skeletons.

The routines in this module are intentionally lightweight.  They fit coarse,
interpretable primitives to already-isolated mesh components and attach the
results to a skeleton graph without changing the skeleton topology.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from organograph.skeleton.datatypes import SkeletonGraph
from organograph.skeleton.geometry import as_points, centroid
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


def _residual_summary(residuals: np.ndarray) -> dict[str, float]:
    residuals = np.asarray(residuals, dtype=float)
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "median_abs": float("nan"),
            "max_abs": float("nan"),
            "n_points": 0,
        }
    return {
        "rmse": float(np.sqrt(np.mean(finite**2))),
        "mae": float(np.mean(np.abs(finite))),
        "median_abs": float(np.median(np.abs(finite))),
        "max_abs": float(np.max(np.abs(finite))),
        "n_points": int(finite.size),
    }


def _coerce_indices(value) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.int64)
    arr = np.asarray(list(value) if isinstance(value, set) else value, dtype=np.int64)
    if arr.ndim != 1:
        return np.empty(0, dtype=np.int64)
    return arr


def _mesh_edges(faces) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _region_boundary_vertices(faces, region, n_vertices: int) -> np.ndarray:
    region = _coerce_indices(region)
    if region.size == 0:
        return np.empty(0, dtype=np.int64)
    mask = np.zeros(int(n_vertices), dtype=bool)
    mask[region[(region >= 0) & (region < int(n_vertices))]] = True
    edges = _mesh_edges(faces)
    crossing = edges[mask[edges[:, 0]] != mask[edges[:, 1]]]
    return np.unique(crossing) if crossing.size else np.empty(0, dtype=np.int64)


def _first_detection_value(detection: dict[str, Any], keys: tuple[str, ...]):
    for key in keys:
        value = detection.get(key)
        if value is not None:
            return value
    return None


def component_region_from_detection(
    detection: dict[str, Any],
    n_vertices: int,
    *,
    region_keys: tuple[str, ...] = (
        "attachment_region_vertices",
        "neck_region_vertices",
        "neck_side_vertices",
        "root_region_vertices",
    ),
) -> np.ndarray:
    """Return the component vertices on the crypt side of a neckline.

    The preferred input is an explicit neck-side region.  If absent, a full-mesh
    normalized distance field is thresholded at ``neck_level`` (default 1.0).
    Finally, raw crypt patch vertices are used as a fallback.  This mirrors the
    skeleton builder's neck-bounded component logic and keeps appendices cut off
    before fitting body or branch primitives.
    """
    region = _coerce_indices(_first_detection_value(detection, region_keys))
    if region.size:
        return np.unique(region)

    dfield = _first_detection_value(
        detection,
        ("d_crypt", "distance_field", "dnorm", "dnorm_vertices"),
    )
    if dfield is not None:
        dfield = np.asarray(dfield, dtype=float).reshape(-1)
        if dfield.size == int(n_vertices):
            level = float(
                detection.get(
                    "attachment_level",
                    detection.get("neck_level", 1.0),
                )
            )
            region = np.where(np.isfinite(dfield) & (dfield <= level))[0].astype(np.int64)
            if region.size:
                return region

    return np.unique(
        _coerce_indices(
            _first_detection_value(
                detection,
                ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
            )
        )
    )


def primitive_components_from_crypt_detections(
    vertices,
    crypt_detections: list[dict[str, Any]],
    graph: SkeletonGraph | None = None,
    faces=None,
) -> dict[str, Any]:
    """Build neck-cut body, branch, and crypt component vertex sets.

    Body vertices are the mesh vertices left after detaching every root
    appendage at its body-side neckline.  Branch vertices are parent
    neck-side regions with daughter crypt-side regions removed.  Crypt tube
    components are the terminal regions after their final neckline.
    """
    vertices = as_points(vertices)
    n_vertices = int(vertices.shape[0])
    all_vertices = set(range(n_vertices))
    body_excluded: set[int] = set()
    branches: dict[str, list[int]] = {}
    crypts: dict[Any, list[int]] = {}
    crypt_centerlines: dict[Any, dict[str, Any]] = {}
    body_branch_necks: dict[str, dict[str, Any]] = {}

    for detection in crypt_detections:
        crypt_id = detection.get("crypt_id")
        daughters = detection.get("daughters") or []
        parent_region = component_region_from_detection(detection, n_vertices)

        if daughters:
            if parent_region.size:
                body_excluded.update(map(int, parent_region.tolist()))

            daughter_regions = []
            for j, daughter in enumerate(daughters):
                daughter_region = component_region_from_detection(daughter, n_vertices)
                daughter_regions.append(daughter_region)
                if daughter_region.size:
                    tip_node_id = f"crypt_{crypt_id}_tip_{j}"
                    crypts[tip_node_id] = sorted(map(int, daughter_region.tolist()))
                    crypt_centerlines[tip_node_id] = {
                        "vertex_indices": crypts[tip_node_id],
                        "distance_field": _first_detection_value(
                            daughter,
                            ("d_crypt", "distance_field", "dnorm", "dnorm_vertices"),
                        ),
                        "neck_level": float(
                            daughter.get(
                                "attachment_level",
                                daughter.get("neck_level", 1.0),
                            )
                        ),
                        "neck_profile": daughter.get("neck_profile"),
                    }

            remove = set()
            for daughter_region in daughter_regions:
                remove.update(map(int, daughter_region.tolist()))
            branch_region = [int(v) for v in parent_region.tolist() if int(v) not in remove]
            if not branch_region:
                stem = _coerce_indices(
                    _first_detection_value(detection, ("stem_vertices", "trunk_vertices"))
                )
                branch_region = sorted(map(int, stem.tolist()))

            branch_node_id = f"crypt_{crypt_id}_branch"
            if branch_region and (graph is None or branch_node_id in graph.nodes):
                branches[branch_node_id] = sorted(set(branch_region))
            neck_node_id = f"crypt_{crypt_id}_neck"
            if (
                faces is not None
                and graph is not None
                and neck_node_id in graph.nodes
                and branch_node_id in graph.nodes
            ):
                boundary_vertices = _region_boundary_vertices(
                    faces,
                    parent_region,
                    n_vertices,
                )
                if boundary_vertices.size:
                    attachment_id = f"{neck_node_id}_cylinder"
                    body_branch_necks[attachment_id] = {
                        "neck_node_id": neck_node_id,
                        "body_node_id": graph.body_node().node_id,
                        "branch_node_id": branch_node_id,
                        "boundary_vertices": boundary_vertices.tolist(),
                    }
            continue

        region = component_region_from_detection(detection, n_vertices)
        if region.size:
            body_excluded.update(map(int, region.tolist()))
            crypts[crypt_id] = sorted(map(int, region.tolist()))
            crypt_centerlines[crypt_id] = {
                "vertex_indices": crypts[crypt_id],
                "distance_field": _first_detection_value(
                    detection,
                    ("d_crypt", "distance_field", "dnorm", "dnorm_vertices"),
                ),
                "neck_level": float(
                    detection.get(
                        "attachment_level",
                        detection.get("neck_level", 1.0),
                    )
                ),
                "neck_profile": detection.get("neck_profile"),
            }

    body = sorted(all_vertices.difference(body_excluded))
    if len(body) < 3:
        body = sorted(all_vertices)

    return {
        "body": body,
        "branches": branches,
        "crypts": crypts,
        "crypt_centerlines": crypt_centerlines,
        "body_branch_necks": body_branch_necks,
        "metadata": {
            "n_body_vertices": len(body),
            "n_body_excluded_vertices": len(body_excluded),
            "component_source": "neck_cut_crypt_detections",
        },
    }


def fit_straight_neck_cylinder(
    vertices,
    boundary_vertices,
    body_center,
    neck_center,
    branch_center,
    *,
    radius_quantile: float = 0.5,
    expansion_factor: float = 1.35,
    max_extent_fraction: float = 0.25,
    min_extent_radius_fraction: float = 0.35,
    n_axial_bins: int = 24,
) -> PrimitiveFit:
    """Fit a local straight cylinder at a stored body-branch cut ring.

    The axis is fixed by the body and branch centers and passes through the
    existing neck node. Radius comes from the cut-ring vertices. Cylinder
    endpoints are the first local axial bins whose robust surface radius has
    expanded by ``expansion_factor``, bounded to remain near the neck.
    """
    vertices = as_points(vertices)
    boundary = vertices[_coerce_indices(boundary_vertices)]
    body = np.asarray(body_center, dtype=float)
    neck = np.asarray(neck_center, dtype=float)
    branch = np.asarray(branch_center, dtype=float)
    axis = branch - body
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        axis = branch - neck
        axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        raise ValueError("Body and branch centers do not define a neck axis")
    axis /= axis_norm
    if float(np.dot(branch - neck, axis)) < 0.0:
        axis *= -1.0

    boundary_delta = boundary - neck[None, :]
    boundary_axial = boundary_delta @ axis
    boundary_radial = np.linalg.norm(
        boundary_delta - boundary_axial[:, None] * axis[None, :],
        axis=1,
    )
    finite_radius = boundary_radial[np.isfinite(boundary_radial)]
    if finite_radius.size < 3:
        raise ValueError("Too few finite body-branch boundary vertices")
    radius = float(np.quantile(finite_radius, float(radius_quantile)))
    radius = max(radius, 1e-6)

    body_limit = max(
        float(np.linalg.norm(neck - body)) * float(max_extent_fraction),
        radius * float(min_extent_radius_fraction),
    )
    branch_limit = max(
        float(np.linalg.norm(branch - neck)) * float(max_extent_fraction),
        radius * float(min_extent_radius_fraction),
    )
    delta = vertices - neck[None, :]
    axial = delta @ axis
    radial = np.linalg.norm(delta - axial[:, None] * axis[None, :], axis=1)
    local = (
        (axial >= -body_limit)
        & (axial <= branch_limit)
        & (radial <= max(3.0 * radius, radius + 1e-6))
    )
    threshold = max(float(expansion_factor), 1.0) * radius

    def first_expansion(sign: float, limit: float) -> float:
        edges = np.linspace(0.0, limit, max(4, int(n_axial_bins)) + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            coord = sign * axial
            mask = local & (coord >= lo) & (coord < hi)
            if np.count_nonzero(mask) < 3:
                continue
            local_radius = float(np.median(radial[mask]))
            if local_radius >= threshold:
                return max(float(lo), radius * float(min_extent_radius_fraction))
        return float(limit)

    body_extent = min(first_expansion(-1.0, body_limit), body_limit)
    branch_extent = min(first_expansion(1.0, branch_limit), branch_limit)
    start = neck - body_extent * axis
    end = neck + branch_extent * axis
    supported = (
        local
        & (axial >= -body_extent)
        & (axial <= branch_extent)
        & (radial <= threshold)
    )
    residuals = radial[supported] - radius
    summary = _residual_summary(residuals)
    return PrimitiveFit(
        primitive_type="straight_cylinder",
        parameters={
            "centerline_points": np.vstack([start, end]),
            "radius": radius,
            "axis": axis,
            "body_extent": body_extent,
            "branch_extent": branch_extent,
            "neck_center": neck,
            "expansion_factor": float(expansion_factor),
        },
        fit_error=summary["rmse"],
        residuals=summary,
        derived_parameters={
            "length": float(body_extent + branch_extent),
            "diameter": float(2.0 * radius),
            "aspect_ratio": float((body_extent + branch_extent) / (2.0 * radius)),
        },
        metadata={
            "fit_method": "fixed_axis_cut_ring_radius_local_expansion",
            "n_boundary_vertices": int(boundary.shape[0]),
            "n_supported_vertices": int(np.count_nonzero(supported)),
        },
    )


def attach_body_branch_neck_primitives(
    graph: SkeletonGraph,
    vertices,
    neck_components: dict[str, dict[str, Any]],
    *,
    body_component,
    branch_components: dict[str, Any],
    trim_radius_factor: float = 1.5,
    **fit_kwargs,
) -> dict[str, Any]:
    """Fit body-branch neck cylinders first and trim later blob components."""
    vertices = as_points(vertices)
    body_indices = _coerce_indices(body_component)
    trimmed_branches = {
        str(key): _coerce_indices(value)
        for key, value in branch_components.items()
    }
    attachments = {}

    for attachment_id, spec in neck_components.items():
        body_node_id = str(spec["body_node_id"])
        neck_node_id = str(spec["neck_node_id"])
        branch_node_id = str(spec["branch_node_id"])
        fit = fit_straight_neck_cylinder(
            vertices,
            spec["boundary_vertices"],
            graph.node(body_node_id).position,
            graph.node(neck_node_id).position,
            graph.node(branch_node_id).position,
            **fit_kwargs,
        )
        attachment = fit.to_attachment(
            attachment_type="path",
            attachment_id=str(attachment_id),
            target_ids=[body_node_id, neck_node_id, branch_node_id],
            metadata={"component": "body_branch_neck"},
        )
        graph.add_primitive_attachment(str(attachment_id), attachment)
        attachments[str(attachment_id)] = attachment

        axis = np.asarray(fit.parameters["axis"], dtype=float)
        neck = np.asarray(fit.parameters["neck_center"], dtype=float)
        radius = float(fit.parameters["radius"]) * float(trim_radius_factor)
        body_extent = float(fit.parameters["body_extent"])
        branch_extent = float(fit.parameters["branch_extent"])

        def trim(indices, *, side: str):
            if indices.size == 0:
                return indices
            delta = vertices[indices] - neck[None, :]
            axial = delta @ axis
            radial = np.linalg.norm(
                delta - axial[:, None] * axis[None, :],
                axis=1,
            )
            if side == "body":
                remove = (axial > -body_extent) & (radial <= radius)
            else:
                remove = (axial < branch_extent) & (radial <= radius)
            kept = indices[~remove]
            return kept if kept.size >= 3 else indices

        body_indices = trim(body_indices, side="body")
        if branch_node_id in trimmed_branches:
            trimmed_branches[branch_node_id] = trim(
                trimmed_branches[branch_node_id],
                side="branch",
            )

    return {
        "attachments": attachments,
        "body": body_indices.tolist(),
        "branches": {
            key: value.tolist()
            for key, value in trimmed_branches.items()
        },
    }


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

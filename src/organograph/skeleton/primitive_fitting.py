"""Primitive fitting for biology-aware organoid skeletons.

The routines in this module are intentionally lightweight.  They fit coarse,
interpretable primitives to already-isolated mesh components and attach the
results to a skeleton graph without changing the skeleton topology.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

from organograph.skeleton.datatypes import SkeletonGraph
from organograph.skeleton.geometry import as_points, centroid
from organograph.skeleton.primitive_geometry import (
    bend_angles_for_polyline,
    component_points,
    polyline_lengths,
    project_points_to_polyline,
    quadratic_radius,
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
            level = float(detection.get("neck_level", 1.0))
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
            continue

        region = component_region_from_detection(detection, n_vertices)
        if region.size:
            body_excluded.update(map(int, region.tolist()))
            crypts[crypt_id] = sorted(map(int, region.tolist()))

    body = sorted(all_vertices.difference(body_excluded))
    if len(body) < 3:
        body = sorted(all_vertices)

    return {
        "body": body,
        "branches": branches,
        "crypts": crypts,
        "metadata": {
            "n_body_vertices": len(body),
            "n_body_excluded_vertices": len(body_excluded),
            "component_source": "neck_cut_crypt_detections",
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
    raise ValueError("Only 'ellipsoid' is implemented for blob primitives")


def _children_for_crypt(graph: SkeletonGraph, crypt_id):
    children = defaultdict(list)
    for edge in graph.edges_for_crypt(crypt_id, include_body_edge=False):
        children[edge.source].append(edge.target)
    return children


def _root_necks(graph: SkeletonGraph, crypt_id):
    necks = graph.nodes_for_crypt(crypt_id, node_type="neck")
    incoming_necks = set()
    for edge in graph.edges_for_crypt(crypt_id, include_body_edge=False):
        if graph.node(edge.target).node_type == "neck":
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
    """Return crypt-component paths trimmed to the last neckline before each tip."""
    out = []
    for path in _all_root_to_tip_paths(graph, crypt_id):
        last_neck_index = 0
        for i, node_id in enumerate(path[:-1]):
            if graph.node(node_id).node_type == "neck":
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


def fit_crypt_tube_to_points(
    points,
    centerline_points,
    *,
    path_node_ids: list[str] | None = None,
    radius_quantile: float = 0.5,
    neck_window: tuple[float, float] = (0.0, 0.2),
    body_window: tuple[float, float] = (0.4, 0.6),
    tip_window: tuple[float, float] = (0.75, 0.95),
    metadata: dict[str, Any] | None = None,
) -> PrimitiveFit:
    """Fit a tapered capped tube to crypt component points.

    The centerline is piecewise linear.  Three radii are estimated from point
    distances to the centerline near the neck, body, and distal tip; a quadratic
    radius profile interpolates between them.  The rounded cap is deterministic
    from ``r_tip`` and is represented by metadata rather than extra degrees of
    freedom.
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

    r_neck = _radius_from_window(distances, s, *neck_window, quantile=q)
    r_body = _radius_from_window(distances, s, *body_window, quantile=q)
    r_tip = _radius_from_window(distances, s, *tip_window, quantile=q)
    radii = np.asarray([r_neck, r_body, r_tip], dtype=float)
    fallback_radius = float(np.nanmedian(distances)) if np.any(np.isfinite(distances)) else 0.0
    radii = np.where(np.isfinite(radii), radii, fallback_radius)
    radii = np.maximum(radii, 1e-8)
    r_neck, r_body, r_tip = map(float, radii)

    predicted = np.maximum(quadratic_radius(s, r_neck, r_body, r_tip), 1e-8)
    residuals = distances - predicted
    summary = _residual_summary(residuals)

    _, _, length = polyline_lengths(centerline)
    straight = float(np.linalg.norm(centerline[-1] - centerline[0]))
    bend_angles = bend_angles_for_polyline(centerline)
    bend_angle = float(np.nanmax(bend_angles)) if bend_angles else 0.0
    derived = {
        "length": float(length),
        "straight_distance": straight,
        "tortuosity": float(length / straight) if straight > 1e-12 else float("nan"),
        "bend_angle": bend_angle,
        "bend_angles": bend_angles,
        "constriction_ratio": float(r_neck / r_body) if r_body > 1e-12 else float("nan"),
        "taper_ratio": float(r_tip / r_body) if r_body > 1e-12 else float("nan"),
    }
    return PrimitiveFit(
        primitive_type="tapered_capped_tube",
        parameters={
            "centerline_points": centerline,
            "path_node_ids": list(path_node_ids or []),
            "r_neck": r_neck,
            "r_body": r_body,
            "r_tip": r_tip,
            "radius_quantile": q,
            "radius_profile": "quadratic_3_radius",
            "cap": "distal_rounded_from_r_tip",
            "neck_window": neck_window,
            "body_window": body_window,
            "tip_window": tip_window,
        },
        fit_error=summary["rmse"],
        residuals=summary,
        derived_parameters=derived,
        metadata={
            "fit_method": "point_distances_to_piecewise_linear_centerline",
            "n_points": int(pts.shape[0]),
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
    **fit_kwargs,
) -> dict[str, PrimitiveAttachment]:
    """Fit tapered capped tubes to crypt components.

    ``crypt_components`` can be keyed by crypt id, by tip node id, or by an
    explicit tuple/list of path node ids.  Values can be vertex indices into
    ``vertices`` or direct ``(N, 3)`` point arrays.
    """
    out = {}
    for key, component in crypt_components.items():
        points = component_points(vertices, component)
        for attachment_id, path in _resolve_crypt_paths(graph, key):
            centerline = np.vstack([graph.node(node_id).position for node_id in path])
            fit = fit_crypt_tube_to_points(
                points,
                centerline,
                path_node_ids=path,
                metadata={"component": "crypt", "component_key": str(key)},
                **fit_kwargs,
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

"""Geometry utilities and derived measurements for skeleton graphs."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import numpy as np

from organograph.skeleton.datatypes import SkeletonEdge, SkeletonGraph, SkeletonNode


def as_points(points) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected an (N, 3) point array, got shape {arr.shape}")
    return arr


def as_vertex_indices(indices) -> np.ndarray:
    arr = np.asarray(list(indices) if isinstance(indices, set) else indices, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError("Vertex indices must be one-dimensional")
    return arr


def centroid(points) -> np.ndarray:
    pts = as_points(points)
    if pts.shape[0] == 0:
        raise ValueError("Cannot compute centroid of an empty point set")
    return np.mean(pts, axis=0)


def face_areas(vertices, faces) -> np.ndarray:
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )


def surface_area_centroid(vertices, faces) -> np.ndarray:
    """Area-weighted centroid of a triangle surface."""
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return centroid(vertices)
    areas = face_areas(vertices, faces)
    total = float(np.sum(areas))
    if total <= 0.0:
        return centroid(vertices)
    return (vertices[faces].mean(axis=1) * areas[:, None]).sum(axis=0) / total


def chord_midpoint(neck: Iterable[float], tip: Iterable[float]) -> np.ndarray:
    return 0.5 * (np.asarray(neck, dtype=float) + np.asarray(tip, dtype=float))


def crypt_centroid_midsection(
    vertices,
    crypt_vertices,
    neck_position,
    tip_position,
    *,
    lo: float = 0.4,
    hi: float = 0.6,
) -> np.ndarray | None:
    """Estimate a bend point from crypt-region vertices near the chord middle.

    Points are projected onto the straight neck-tip chord.  The centroid of
    points whose projection lies in the middle interval is used as a simple,
    robust placeholder bend estimate.
    """
    vertices = as_points(vertices)
    idx = as_vertex_indices(crypt_vertices)
    if idx.size == 0:
        return None

    neck = np.asarray(neck_position, dtype=float)
    tip = np.asarray(tip_position, dtype=float)
    chord = tip - neck
    denom = float(np.dot(chord, chord))
    if denom <= 1e-12:
        return None

    pts = vertices[idx]
    t = ((pts - neck) @ chord) / denom
    mask = (t >= float(lo)) & (t <= float(hi))
    if not np.any(mask):
        return None
    return centroid(pts[mask])


def estimate_bend_position(
    vertices,
    crypt_vertices,
    neck_position,
    tip_position,
    *,
    strategy: str = "none",
) -> np.ndarray | None:
    strategy = str(strategy)
    if strategy == "none":
        return None
    if strategy == "midpoint":
        return chord_midpoint(neck_position, tip_position)
    if strategy == "crypt_centroid":
        idx = as_vertex_indices(crypt_vertices)
        if idx.size:
            return centroid(as_points(vertices)[idx])
        return chord_midpoint(neck_position, tip_position)
    if strategy == "crypt_centroid_midsection":
        bend = crypt_centroid_midsection(
            vertices,
            crypt_vertices,
            neck_position,
            tip_position,
        )
        if bend is not None:
            return bend
        return chord_midpoint(neck_position, tip_position)
    raise ValueError(
        "bend_strategy must be one of 'none', 'midpoint', "
        "'crypt_centroid', or 'crypt_centroid_midsection'"
    )


def transform_points_body_relative(
    points,
    *,
    body_center,
    orientation=None,
    scale: float = 1.0,
) -> np.ndarray:
    """Convert raw coordinates into a body-relative frame.

    The body center becomes the origin, `orientation` maps raw centered points
    into the canonical frame, and `scale` normalizes size.  This is a utility
    hook for later VAE-style parameterization.
    """
    pts = as_points(points)
    center = np.asarray(body_center, dtype=float)
    if center.shape != (3,):
        raise ValueError("body_center must be a 3-vector")
    if scale is None:
        scale = 1.0
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    out = pts - center[None, :]
    if orientation is not None:
        R = np.asarray(orientation, dtype=float)
        if R.shape != (3, 3):
            raise ValueError("orientation must be a (3, 3) matrix")
        out = out @ R.T
    return out / scale


def skeleton_to_body_relative(
    graph: SkeletonGraph,
    *,
    orientation=None,
    scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Return node positions in a body-relative coordinate frame."""
    body = graph.body_node()
    node_ids = list(graph.nodes)
    points = np.vstack([graph.nodes[node_id].position for node_id in node_ids])
    rel = transform_points_body_relative(
        points,
        body_center=body.position,
        orientation=orientation,
        scale=scale,
    )
    return {node_id: rel[i] for i, node_id in enumerate(node_ids)}


def edge_length(graph: SkeletonGraph, edge_id: str) -> float:
    edge = graph.edge(edge_id)
    p0 = graph.node(edge.source).position
    p1 = graph.node(edge.target).position
    return float(np.linalg.norm(p1 - p0))


def _crypt_adjacency(graph: SkeletonGraph, crypt_id, *, include_body_edge: bool = False):
    children = defaultdict(list)
    for edge in graph.edges_for_crypt(crypt_id, include_body_edge=include_body_edge):
        children[edge.source].append(edge.target)
    return children


def _neck_nodes(graph: SkeletonGraph, crypt_id) -> list[SkeletonNode]:
    return [
        node
        for node in graph.nodes_for_crypt(crypt_id)
        if node.node_type in {"neck", "attachment"}
    ]


def _root_neck_nodes(graph: SkeletonGraph, crypt_id) -> list[SkeletonNode]:
    """Neck nodes with no same-crypt non-body parent.

    Split crypts can contain daughter necks downstream of a branch node.  For
    path descriptors and attachment direction, the root neck is the crypt's
    attachment to the villus/body.
    """
    necks = _neck_nodes(graph, crypt_id)
    if len(necks) <= 1:
        return necks

    incoming_nonbody = set()
    for edge in graph.edges_for_crypt(crypt_id, include_body_edge=False):
        if graph.node(edge.target).node_type in {"neck", "attachment"}:
            incoming_nonbody.add(edge.target)
    roots = [neck for neck in necks if neck.node_id not in incoming_nonbody]
    return roots or necks


def _tip_nodes(graph: SkeletonGraph, crypt_id) -> list[SkeletonNode]:
    return graph.nodes_for_crypt(crypt_id, node_type="tip")


def _paths_from_neck_to_tips(graph: SkeletonGraph, crypt_id) -> list[list[str]]:
    necks = _root_neck_nodes(graph, crypt_id)
    tips = {node.node_id for node in _tip_nodes(graph, crypt_id)}
    if not necks or not tips:
        return []

    children = _crypt_adjacency(graph, crypt_id, include_body_edge=False)
    paths = []
    for neck in necks:
        q = deque([(neck.node_id, [neck.node_id])])
        while q:
            node_id, path = q.popleft()
            if node_id in tips:
                paths.append(path)
                continue
            for child in children.get(node_id, []):
                if child not in path:
                    q.append((child, path + [child]))
    return paths


def _path_length(graph: SkeletonGraph, path: list[str]) -> float:
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        total += float(np.linalg.norm(graph.node(b).position - graph.node(a).position))
    return total


def _longest_crypt_path(graph: SkeletonGraph, crypt_id) -> list[str]:
    paths = _paths_from_neck_to_tips(graph, crypt_id)
    if not paths:
        return []
    return max(paths, key=lambda p: _path_length(graph, p))


def crypt_path_length(graph: SkeletonGraph, crypt_id) -> float:
    """Length of the longest neck-to-tip skeleton path for a crypt."""
    path = _longest_crypt_path(graph, crypt_id)
    if not path:
        return float("nan")
    return _path_length(graph, path)


def crypt_straight_distance(graph: SkeletonGraph, crypt_id) -> float:
    """Euclidean neck-to-tip distance for the longest crypt path."""
    path = _longest_crypt_path(graph, crypt_id)
    if len(path) < 2:
        return float("nan")
    return float(np.linalg.norm(graph.node(path[-1]).position - graph.node(path[0]).position))


def crypt_tortuosity(graph: SkeletonGraph, crypt_id) -> float:
    length = crypt_path_length(graph, crypt_id)
    straight = crypt_straight_distance(graph, crypt_id)
    if not np.isfinite(length) or not np.isfinite(straight) or straight <= 1e-12:
        return float("nan")
    return float(length / straight)


def _angle_between(v0: np.ndarray, v1: np.ndarray) -> float:
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v1))
    if n0 <= 1e-12 or n1 <= 1e-12:
        return float("nan")
    cosang = float(np.dot(v0, v1) / (n0 * n1))
    return float(np.arccos(np.clip(cosang, -1.0, 1.0)))


def crypt_bend_angle(graph: SkeletonGraph, crypt_id) -> float:
    """Angle in radians between the two straight segments around a bend node."""
    path = _longest_crypt_path(graph, crypt_id)
    bend_ids = [node_id for node_id in path if graph.node(node_id).node_type == "bend"]
    if not bend_ids:
        return 0.0
    bend_id = bend_ids[0]
    i = path.index(bend_id)
    if i == 0 or i == len(path) - 1:
        return float("nan")
    p_prev = graph.node(path[i - 1]).position
    p_bend = graph.node(path[i]).position
    p_next = graph.node(path[i + 1]).position
    return _angle_between(p_bend - p_prev, p_next - p_bend)


def crypt_attachment_direction(graph: SkeletonGraph, crypt_id) -> np.ndarray:
    """Unit vector from body center to the crypt neck."""
    body = graph.body_node()
    necks = _root_neck_nodes(graph, crypt_id)
    if not necks:
        return np.full(3, np.nan)
    vec = necks[0].position - body.position
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.full(3, np.nan)
    return vec / norm


def number_of_crypts(graph: SkeletonGraph) -> int:
    return len(graph.crypt_ids())


def number_of_split_crypts(graph: SkeletonGraph) -> int:
    count = 0
    for crypt_id in graph.crypt_ids():
        has_branch = bool(graph.nodes_for_crypt(crypt_id, node_type="branch"))
        n_tips = len(graph.nodes_for_crypt(crypt_id, node_type="tip"))
        if has_branch or n_tips > 1:
            count += 1
    return count

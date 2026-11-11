"""Extract original mesh triangles and require an orientable topological disk."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .datatypes import SurfacePatch


class PatchTopologyError(ValueError):
    """A patch cannot be mapped to one cap without changing its topology."""


def triangle_areas(vertices, faces):
    p = np.asarray(vertices, float)[np.asarray(faces, int)]
    return np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2


def vertex_areas(vertices, faces):
    return np.bincount(
        np.asarray(faces).ravel(), weights=np.repeat(triangle_areas(vertices, faces) / 3, 3),
        minlength=len(vertices),
    )


def surface_moments(vertices, faces):
    """Exact area, centroid and RMS radius of the piecewise planar surface."""
    p = np.asarray(vertices, float)[faces]
    areas = triangle_areas(vertices, faces)
    area = float(areas.sum())
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Surface must have positive finite area")
    center = np.sum(areas[:, None] * p.mean(axis=1), axis=0) / area
    d = p - center
    second = (np.sum(d * d, axis=(1, 2)) + np.sum(d.sum(axis=1) ** 2, axis=1)) / 12
    radius = float(np.sqrt(areas @ second / area))
    return area, center, radius


def extract_patch(vertices, faces, vertex_ids=None):
    """Keep faces whose three vertices are selected; never fill holes or cut handles.

    Isolated selected vertices unused by any retained face are reported separately.
    Vertex coordinates remain in the caller's units and face winding is preserved.
    """
    vertices = np.asarray(vertices, float)
    faces_input = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be a finite (N, 3) array")
    if faces_input.ndim != 2 or faces_input.shape[1] != 3 or not np.issubdtype(faces_input.dtype, np.integer):
        raise ValueError("faces must be an integer (F, 3) array")
    faces = faces_input.astype(np.int64)
    if faces.size == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Invalid or empty face indices")
    ids = np.arange(len(vertices)) if vertex_ids is None else np.asarray(list(vertex_ids))
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("vertex_ids must be a nonempty integer vector")
    ids = np.unique(ids.astype(np.int64))
    if not ids.size or ids.min() < 0 or ids.max() >= len(vertices):
        raise ValueError("Invalid or empty selected vertex indices")
    selected = np.zeros(len(vertices), bool)
    selected[ids] = True
    face_ids = np.flatnonzero(selected[faces].all(axis=1))
    if not len(face_ids):
        raise PatchTopologyError("Selected vertices contain no complete triangles")
    src = np.unique(faces[face_ids])
    remap = np.full(len(vertices), -1, int)
    remap[src] = np.arange(len(src))
    f = remap[faces[face_ids]]
    v = vertices[src].copy()
    if len(np.unique(np.sort(f, axis=1), axis=0)) != len(f):
        raise PatchTopologyError("Duplicate triangles")
    areas = triangle_areas(v, f)
    extent = np.linalg.norm(np.ptp(v, axis=0))
    if np.any(areas <= np.finfo(float).eps * extent ** 2):
        raise PatchTopologyError("Degenerate or near-zero-area triangles")
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    edges, inv, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise PatchTopologyError("Nonmanifold edges")
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    winding = np.bincount(inv, weights=signs)
    if np.any(winding[counts == 2] != 0):
        raise PatchTopologyError("Inconsistent triangle winding; orient the source mesh first")
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(v), len(v)))
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise PatchTopologyError("Disconnected patch")
    b_edges = directed[counts[inv] == 1]
    if len(b_edges) < 3:
        raise PatchTopologyError("Patch has no open boundary")
    b = np.unique(b_edges)
    if np.any(np.bincount(b_edges.ravel(), minlength=len(v))[b] != 2):
        raise PatchTopologyError("Pinched/nonmanifold boundary")
    if len(v) - len(edges) + len(f) != 1:
        raise PatchTopologyError("Patch is not a disk (holes or handles)")
    next_vertex = dict(b_edges.tolist())
    if len(next_vertex) != len(b):
        raise PatchTopologyError("Inconsistent boundary winding")
    boundary = [int(b.min())]
    while True:
        nxt = next_vertex[boundary[-1]]
        if nxt == boundary[0]:
            break
        if nxt in boundary or len(boundary) >= len(b):
            raise PatchTopologyError("Invalid boundary loop")
        boundary.append(nxt)
    if len(boundary) != len(b):
        raise PatchTopologyError("Multiple boundary loops")
    # A manifold vertex link is a connected path (boundary) or cycle (interior).
    links = [[] for _ in v]
    for a, c, d in f:
        links[a].append((c, d))
        links[c].append((d, a))
        links[d].append((a, c))
    for link in links:
        adjacency = {}
        for a, c in link:
            adjacency.setdefault(a, []).append(c)
            adjacency.setdefault(c, []).append(a)
        visited, queue = set(), [next(iter(adjacency))]
        while queue:
            a = queue.pop()
            if a not in visited:
                visited.add(a)
                queue.extend(adjacency[a])
        if len(visited) != len(adjacency):
            raise PatchTopologyError("Nonmanifold vertex link")
    return SurfacePatch(v, f, src, face_ids, np.asarray(boundary), np.setdiff1d(ids, src))

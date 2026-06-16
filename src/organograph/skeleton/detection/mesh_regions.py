"""Mesh-region utilities used by skeleton detection and graph building."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np

from organograph.skeleton.detection.common import _coerce_patch, _first_present
from organograph.skeleton.geometry import as_points, centroid

def _mesh_like(vertices, faces):
    """Small mesh object for segmentation helpers that only need v, f, areas."""
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)

    def vertex_areas(from_mass_matrix: bool = False):
        tri = vertices[faces]
        face_areas = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
            axis=1,
        )
        areas = np.zeros(vertices.shape[0], dtype=float)
        for i in range(3):
            np.add.at(areas, faces[:, i], face_areas / 3.0)
        return areas

    return SimpleNamespace(v=vertices, f=faces, vertex_areas=vertex_areas)

def _low_pass_smoothed_mesh_for_detection(
    mesh,
    *,
    lmax: int = 5,
    recompute_eigen: bool = True,
    eigen_k: int | None = None,
):
    """Return a shallow mesh copy with low-pass reconstructed coordinates."""
    lmax = int(lmax)
    if lmax <= 0:
        raise ValueError("smooth_lmax must be positive")

    mesh.compute_spectral_coefficients(lmax=lmax)
    vertices_smoothed = mesh.reconstruct_from_coeffs(mesh.coeffs_v, lmax=lmax)

    smoothed = copy.copy(mesh)
    smoothed.v = np.asarray(vertices_smoothed, dtype=float)
    if recompute_eigen:
        V = int(smoothed.v.shape[0])
        if eigen_k is None:
            eigen_k = len(mesh.eigvals) if getattr(mesh, "eigvals", None) is not None else lmax**2
        eigen_k = max(2, min(int(eigen_k), max(V - 2, 2)))
        smoothed.laplacian = None
        smoothed.mass_matrix = None
        smoothed.eigvals = None
        smoothed.eigvecs = None
        smoothed.coeffs_v = None
        smoothed.lmax = None
        smoothed._eig_decomp(k=eigen_k)
    return smoothed

def _mesh_edges_from_faces(faces) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)

def _weighted_vertex_adjacency_from_edges(
    vertices,
    n_vertices: int,
    edges: np.ndarray,
) -> list[dict[int, float]]:
    vertices = as_points(vertices)
    adjacency = [dict() for _ in range(int(n_vertices))]
    for a, b in np.asarray(edges, dtype=np.int64):
        a = int(a)
        b = int(b)
        length = float(np.linalg.norm(vertices[b] - vertices[a]))
        adjacency[a][b] = length
        adjacency[b][a] = length
    return adjacency

def _boundary_edges_for_region(edges: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    inside_a = region_mask[edges[:, 0]]
    inside_b = region_mask[edges[:, 1]]
    return edges[inside_a != inside_b]

def _boundary_length_and_center(vertices, boundary_edges) -> tuple[float, np.ndarray | None]:
    vertices = as_points(vertices)
    boundary_edges = np.asarray(boundary_edges, dtype=np.int64)
    if boundary_edges.size == 0:
        return 0.0, None
    p0 = vertices[boundary_edges[:, 0]]
    p1 = vertices[boundary_edges[:, 1]]
    lengths = np.linalg.norm(p1 - p0, axis=1)
    total = float(np.sum(lengths))
    if total <= 1e-12:
        return total, centroid(0.5 * (p0 + p1))
    midpoints = 0.5 * (p0 + p1)
    center = np.sum(midpoints * lengths[:, None], axis=0) / total
    return total, center

def _frontier_for_region(adjacency: list[dict[int, float]], region: set[int], region_mask: np.ndarray) -> set[int]:
    frontier = set()
    for v in region:
        frontier.update(n for n in adjacency[v] if not region_mask[n])
    return frontier

def _best_frontier_addition(
    adjacency: list[dict[int, float]],
    region_mask: np.ndarray,
    frontier: set[int],
    current_length: float,
) -> tuple[int | None, float | None, float | None]:
    best_vertex = None
    best_length = None
    best_delta = None
    for candidate in frontier:
        delta = 0.0
        for neighbor, edge_length in adjacency[candidate].items():
            delta += -edge_length if region_mask[neighbor] else edge_length
        predicted = float(current_length) + delta
        if (
            best_length is None
            or predicted < best_length
            or (np.isclose(predicted, best_length) and delta < best_delta)
        ):
            best_vertex = int(candidate)
            best_length = float(predicted)
            best_delta = float(delta)
    return best_vertex, best_length, best_delta

def _add_vertex_to_region(
    vertex: int,
    adjacency: list[dict[int, float]],
    region: set[int],
    region_mask: np.ndarray,
    frontier: set[int],
) -> None:
    vertex = int(vertex)
    region.add(vertex)
    region_mask[vertex] = True
    frontier.discard(vertex)
    frontier.update(n for n in adjacency[vertex] if not region_mask[n])
    frontier.difference_update(region)

def _radial_distances_to_axis(points, origin, direction) -> np.ndarray:
    """Return perpendicular distances to an infinite 3D axis."""
    points = as_points(points)
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if points.size == 0 or norm <= 1e-12:
        return np.empty(0, dtype=float)
    unit = direction / norm
    offsets = points - origin[None, :]
    axial = offsets @ unit
    radial = offsets - axial[:, None] * unit[None, :]
    return np.linalg.norm(radial, axis=1)

def _boundary_vertices_from_patch(faces, patch_vertices) -> np.ndarray:
    patch_vertices = _coerce_patch(patch_vertices)
    if patch_vertices.size == 0:
        return patch_vertices
    faces = np.asarray(faces, dtype=np.int64)
    keep = np.isin(faces, patch_vertices).all(axis=1)
    patch_faces = faces[keep]
    if patch_faces.size == 0:
        return patch_vertices
    edges = np.vstack(
        [
            patch_faces[:, [0, 1]],
            patch_faces[:, [1, 2]],
            patch_faces[:, [2, 0]],
        ]
    )
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if boundary_edges.size == 0:
        return patch_vertices
    return np.unique(boundary_edges.reshape(-1))

def _crypt_side_region(detection: dict[str, Any]) -> np.ndarray:
    """Vertices on the crypt side of the root neck boundary."""
    region = _coerce_patch(
        _first_present(
            detection,
            (
                "attachment_region_vertices",
                "neck_region_vertices",
                "neck_side_vertices",
                "root_region_vertices",
            ),
        )
    )
    if region.size:
        return region

    dfield = _first_present(detection, ("d_crypt", "distance_field", "dnorm", "dnorm_vertices"))
    if dfield is not None:
        dfield = np.asarray(dfield, dtype=float).reshape(-1)
        level = float(
            detection.get(
                "attachment_level",
                detection.get("neck_level", 1.0),
            )
        )
        region = np.where(np.isfinite(dfield) & (dfield <= level))[0].astype(np.int64)
        if region.size:
            return region

    return _coerce_patch(
        _first_present(
            detection,
            ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
        )
    )

def _body_center_from_root_regions(vertices, detections: list[dict[str, Any]]) -> np.ndarray | None:
    vertices = as_points(vertices)
    excluded: set[int] = set()
    for detection in detections:
        region = _crypt_side_region(detection)
        if region.size:
            excluded.update(int(v) for v in region)

    if not excluded:
        return None
    body_vertices = np.setdiff1d(np.arange(vertices.shape[0], dtype=np.int64), np.fromiter(excluded, dtype=np.int64))
    if body_vertices.size == 0:
        return None
    return centroid(vertices[body_vertices])

def _branch_region_vertices(detection: dict[str, Any], daughters: list[dict[str, Any]]) -> np.ndarray:
    parent_region = _crypt_side_region(detection)
    if parent_region.size == 0:
        parent_region = _coerce_patch(_first_present(detection, ("stem_vertices", "trunk_vertices")))
    if parent_region.size == 0:
        return parent_region

    remove: set[int] = set()
    for daughter in daughters:
        daughter_region = _crypt_side_region(daughter)
        if daughter_region.size:
            remove.update(int(v) for v in daughter_region)
    if not remove:
        stem = _coerce_patch(_first_present(detection, ("stem_vertices", "trunk_vertices")))
        return stem if stem.size else parent_region

    branch_region = np.asarray([int(v) for v in parent_region if int(v) not in remove], dtype=np.int64)
    if branch_region.size:
        return branch_region
    stem = _coerce_patch(_first_present(detection, ("stem_vertices", "trunk_vertices")))
    return stem

def _branch_position_from_regions(vertices, detection: dict[str, Any], daughters: list[dict[str, Any]]) -> tuple[np.ndarray | None, np.ndarray]:
    branch_region = _branch_region_vertices(detection, daughters)
    if branch_region.size == 0:
        return None, branch_region
    return centroid(as_points(vertices)[branch_region]), branch_region


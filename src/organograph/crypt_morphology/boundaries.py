"""Prescribed families of openings on an unchanged parent membrane.

No anatomical boundary objective, detector bottom, or detector axis is used.
Distances are edge-graph geodesic approximations; cuts interpolate the resulting
piecewise linear scalar field across source triangles. New vertices retain exact
barycentric provenance on the original piecewise planar surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve

from .datatypes import CapMapping, SurfacePatch
from .parameterization import disk_to_cap
from .patches import extract_patch, triangle_areas


@dataclass(frozen=True)
class OpeningShift:
    name: str
    offset: float = 0.0
    cosine: tuple[float, ...] = ()
    sine: tuple[float, ...] = ()

    @property
    def bound(self):
        n = max(len(self.cosine), len(self.sine))
        a = np.pad(self.cosine, (0, n - len(self.cosine)))
        b = np.pad(self.sine, (0, n - len(self.sine)))
        return float(abs(self.offset) + np.hypot(a, b).sum())


@dataclass
class ClippedPatch:
    patch: SurfacePatch
    interpolation: csr_matrix  # (new vertices, original parent vertices)

    def transfer(self, parent_values):
        """Interpolate values without changing their parent-mesh coordinates."""
        return self.interpolation @ np.asarray(parent_values)


@dataclass
class BoundaryFamily:
    vertices: np.ndarray
    faces: np.ndarray
    original: SurfacePatch
    signed_distance: np.ndarray
    angular_fields: np.ndarray  # cos(phase), sin(phase), cos(2 phase), ...
    max_displacement: float
    collar_width: float
    forbidden_faces: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def level_set(self, shift: OpeningShift):
        if not np.isfinite([shift.offset, *shift.cosine, *shift.sine]).all():
            raise ValueError("Displacements must be finite")
        if shift.bound > self.max_displacement * (1 + 1e-12):
            raise ValueError("Shift exceeds the prescribed displacement band")
        if max(len(shift.cosine), len(shift.sine)) * 2 > self.angular_fields.shape[1]:
            raise ValueError("Requested angular mode was not prepared")
        displacement = np.full(len(self.vertices), shift.offset)
        for j, a in enumerate(shift.cosine):
            displacement += a * self.angular_fields[:, 2 * j]
        for j, b in enumerate(shift.sine):
            displacement += b * self.angular_fields[:, 2 * j + 1]
        return self.signed_distance - displacement

    def cut(self, shift: OpeningShift):
        cut = clip_level_set(self.vertices, self.faces, self.level_set(shift))
        if np.any(self.forbidden_faces[cut.patch.source_face_ids]):
            raise ValueError("Opening enters a triangle protected for a neighboring crypt")
        return cut

    def padded_patch(self):
        # Computational support may include neighboring geometry. Such faces are
        # explicitly excluded from soft-fit weights; actual cuts may not enter it.
        return clip_level_set(self.vertices, self.faces, self.signed_distance - self.collar_width)

    def with_collar_width(self, width):
        """Change computational support while keeping every opening field fixed.

        Re-preparing a family with a different collar also changes the harmonic
        extension's band. Use this method to isolate support/chart effects from
        changes to the prescribed contours. Shared geometry/fields are read-only
        by convention; this operation does not mutate the original family.
        """
        if not np.isfinite(width) or width <= self.max_displacement:
            raise ValueError("Support collar must exceed the displacement bound")
        return replace(self, collar_width=float(width),
                       diagnostics=dict(self.diagnostics, collar_width=float(width)))


def opening_shifts(radius, *, fractions=(0.05, 0.10), n_directions=4, include_second=True):
    """Baseline, signed offsets, tilt-like modes, and optional second harmonic.

    Amplitudes are fractions of a *fixed original* radius, not radii recalculated
    from the perturbed patches. These settings are sensitivity scales, not a
    calibrated distribution of anatomical boundary uncertainty.
    """
    if not np.isfinite(radius) or radius <= 0 or n_directions < 2:
        raise ValueError("Positive radius and at least two directions required")
    fractions = tuple(float(f) for f in fractions)
    if not fractions or any(not np.isfinite(f) or f <= 0 for f in fractions):
        raise ValueError("fractions must be finite and positive")
    shifts = [OpeningShift("baseline")]
    for fraction in fractions:
        amplitude = fraction * radius
        shifts.extend([OpeningShift(f"offset_{sign * fraction:+.3f}", sign * amplitude) for sign in (-1, 1)])
        for j in range(n_directions):
            phase = 2 * np.pi * j / n_directions
            shifts.append(OpeningShift(f"tilt_{fraction:.3f}_{j}", cosine=(amplitude * np.cos(phase),),
                                       sine=(amplitude * np.sin(phase),)))
    if include_second:
        amplitude = max(fractions) * radius
        for j in range(n_directions):
            phase = 2 * np.pi * j / n_directions
            shifts.append(OpeningShift(f"second_{max(fractions):.3f}_{j}",
                                       cosine=(0, amplitude * np.cos(phase)),
                                       sine=(0, amplitude * np.sin(phase))))
    return shifts


def prepare_boundary_family(vertices, faces, vertex_ids, *, max_displacement,
                            collar_width, max_harmonic=2, forbidden_vertex_ids=()):
    """Prepare signed distances and smooth angular extensions from the rim.

    Positive outside / negative inside. Rim Fourier values are extended through
    a narrow band by a positive inverse-edge-length graph Laplacian with fixed
    values at the original rim and natural boundaries at the band edges. This
    removes nearest-rim-label jumps. Distances and extension remain discretization
    dependent, so mesh-refinement controls are part of validation.
    """
    if (not np.isfinite([max_displacement, collar_width]).all() or
            not 0 < max_displacement < collar_width or max_harmonic < 1):
        raise ValueError("Require 0 < max_displacement < collar_width and positive harmonic count")
    v, f = np.asarray(vertices, float), np.asarray(faces, int)
    patch = extract_patch(v, f, vertex_ids)
    edges = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Parent mesh contains zero-length edges")
    row = np.r_[edges[:, 0], edges[:, 1]]
    col = np.r_[edges[:, 1], edges[:, 0]]
    graph = coo_matrix((np.tile(lengths, 2), (row, col)), shape=(len(v), len(v))).tocsr()
    boundary = patch.source_vertex_ids[patch.boundary]
    distance = dijkstra(graph, indices=boundary, directed=False, min_only=True)
    inside = np.zeros(len(v), bool)
    inside[patch.source_vertex_ids] = True
    signed = np.where(inside, -distance, distance)
    signed[boundary] = 0
    band = np.flatnonzero(distance <= collar_width + lengths.max())
    local = np.full(len(v), -1, int)
    local[band] = np.arange(len(band))
    keep = (local[edges] >= 0).all(axis=1)
    a, b = local[edges[keep]].T
    w = 1 / lengths[keep]
    adjacency = coo_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(len(band), len(band))).tocsr()
    lap = diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    boundary_lengths = np.linalg.norm(v[np.roll(boundary, -1)] - v[boundary], axis=1)
    phase = 2 * np.pi * np.r_[0, np.cumsum(boundary_lengths[:-1])] / boundary_lengths.sum()
    values = np.column_stack([g(m * phase) for m in range(1, max_harmonic + 1) for g in (np.cos, np.sin)])
    local_b = local[boundary]
    free = np.setdiff1d(np.arange(len(band)), local_b)
    extended = np.zeros((len(band), 2 * max_harmonic))
    extended[local_b] = values
    if len(free):
        extended[free] = spsolve(lap[free][:, free], -lap[free][:, local_b] @ values)
    if not np.isfinite(extended).all():
        raise ValueError("Angular extension is singular/disconnected")
    fields = np.zeros((len(v), 2 * max_harmonic))
    fields[band] = extended
    forbidden = np.zeros(len(v), bool)
    forbidden_ids = np.asarray(list(forbidden_vertex_ids), int)
    if np.any(forbidden_ids < 0) or np.any(forbidden_ids >= len(v)):
        raise ValueError("Invalid protected vertex IDs")
    forbidden[forbidden_ids] = True
    forbidden_faces = forbidden[f].any(axis=1)
    if np.any(forbidden_faces[patch.source_face_ids]):
        raise ValueError("Original crypt overlaps protected neighboring vertices")
    diagnostics = {"distance_method": "edge_graph_dijkstra", "extension_method": "inverse_edge_graph_harmonic",
                   "median_edge_length": float(np.median(lengths)),
                   "rim_perimeter": float(boundary_lengths.sum()),
                   "max_displacement": float(max_displacement), "collar_width": float(collar_width),
                   "max_displacement_in_median_edges": float(max_displacement / np.median(lengths))}
    return BoundaryFamily(v, f, patch, signed, fields, float(max_displacement), float(collar_width),
                          forbidden_faces, diagnostics)


def clip_level_set(vertices, faces, level):
    """Retain f<=0 using exact clipping of a linearly interpolated triangle field.

    Edge intersection vertices are shared between adjacent faces. Every retained
    triangle has the same winding and lies exactly in one source triangle. No
    cap, geometric smoothing, or anatomical repair is performed.
    """
    v, f, level = np.asarray(vertices, float), np.asarray(faces, int), np.asarray(level, float)
    if level.shape != (len(v),) or np.isnan(level).any():
        raise ValueError("One non-NaN level-set value is required per parent vertex")
    if f.ndim != 2 or f.shape[1] != 3 or f.min() < 0 or f.max() >= len(v):
        raise ValueError("Invalid parent faces")
    active = np.flatnonzero(np.min(level[f], axis=1) <= 0)
    if not len(active):
        raise ValueError("The opening removed the whole patch")
    points, provenance, old_ids, triangles, parents = [], [], [], [], []
    vertex_cache, edge_cache = {}, {}

    def vertex(i):
        if i not in vertex_cache:
            vertex_cache[i] = len(points)
            points.append(v[i]); provenance.append(((int(i), 1.0),)); old_ids.append(i)
        return vertex_cache[i]

    def intersection(a, b):
        if level[a] == 0:
            return vertex(a)
        if level[b] == 0:
            return vertex(b)
        edge = tuple(sorted((int(a), int(b))))
        if edge not in edge_cache:
            a, b = edge
            t = level[a] / (level[a] - level[b])
            if not np.isfinite(t):
                raise ValueError("Nonfinite values occur in a cut triangle")
            if t < 1e-12:
                return vertex(a)
            if t > 1 - 1e-12:
                return vertex(b)
            edge_cache[edge] = len(points)
            points.append((1-t) * v[a] + t * v[b])
            provenance.append(((a, float(1-t)), (b, float(t))))
            old_ids.append(-1)
        return edge_cache[edge]

    for face_id in active:
        face = f[face_id]
        polygon = []
        for j in range(3):
            a, b = int(face[j]), int(face[(j+1) % 3])
            if level[a] <= 0:
                polygon.append(vertex(a))
            if (level[a] < 0 and level[b] > 0) or (level[a] > 0 and level[b] < 0):
                polygon.append(intersection(a, b))
        polygon = list(dict.fromkeys(polygon))
        for j in range(1, len(polygon)-1):
            triangles.append((polygon[0], polygon[j], polygon[j+1])); parents.append(face_id)
    if not triangles:
        raise ValueError("The opening has no nonzero-area surface")
    points, triangles = np.asarray(points), np.asarray(triangles, int)
    valid = triangle_areas(points, triangles) > 0
    triangles, parents = triangles[valid], np.asarray(parents, int)[valid]
    checked = extract_patch(points, triangles)
    kept = checked.source_vertex_ids.copy()
    rows, cols, data = [], [], []
    for new, old in enumerate(kept):
        for source, weight in provenance[old]:
            rows.append(new); cols.append(source); data.append(weight)
    matrix = csr_matrix((data, (rows, cols)), shape=(len(kept), len(v)))
    checked.source_face_ids = parents[checked.source_face_ids]
    checked.source_vertex_ids = np.asarray(old_ids, int)[kept]
    return ClippedPatch(checked, matrix)


def restrict_mapping(cut, support, mapping):
    """Pull the fixed support chart onto a clipped subset using source triangles.

    Each clipped vertex is a point of its original parent triangle. Solving for
    barycentric coordinates in that triangle of the support preserves the chart
    without reparameterizing the new opening.
    """
    parent_to_support = {}
    for local_face, parent_face in enumerate(support.patch.source_face_ids):
        parent_to_support.setdefault(int(parent_face), []).append(local_face)
    # Most vertices are interpolations of original vertices present in support.
    # Handle these in one sparse operation; only support-edge cases need solves.
    parent_uv = np.zeros((support.interpolation.shape[1], 2))
    present = np.zeros(len(parent_uv))
    original = support.patch.source_vertex_ids >= 0
    ids = support.patch.source_vertex_ids[original]
    parent_uv[ids] = mapping.uv[original]
    present[ids] = 1
    uv = cut.interpolation @ parent_uv
    uv[np.asarray(cut.interpolation @ present).ravel() < 1 - 1e-12] = np.nan
    for face, parent in zip(cut.patch.faces, cut.patch.source_face_ids):
        face = face[~np.isfinite(uv[face]).all(axis=1)]
        if not len(face):
            continue
        for local_face in parent_to_support.get(int(parent), []):
            sf = support.patch.faces[local_face]
            xyz = support.patch.vertices[sf]
            vectors = (xyz[1:] - xyz[0]).T
            ab = np.linalg.lstsq(vectors, (cut.patch.vertices[face] - xyz[0]).T, rcond=None)[0].T
            bary = np.column_stack([1-ab.sum(axis=1), ab])
            valid = (bary >= -1e-7).all(axis=1)
            uv[face[valid]] = bary[valid] @ mapping.uv[sf]
    if not np.isfinite(uv).all():
        raise ValueError("Clipped patch is not contained in the fixed support")
    return CapMapping(uv, disk_to_cap(uv, mapping.half_angle), mapping.half_angle,
                      dict(mapping.diagnostics, method="restricted_fixed_support_chart"))

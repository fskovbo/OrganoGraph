"""Finite-resolution geometric checks for decoded, open triangle surfaces.

Intersections use separating axes and a NumPy bounding-box sweep. No additional
spatial-index package is needed. This is a numerical mesh diagnostic, not a proof about the
continuous cap expansion or anatomy. Contact at shared topology is excluded by
a tiny contraction for incident triangle pairs; see ``mesh_intersections``.
"""
import numpy as np


def _mesh(vertices, faces):
    v, f = np.asarray(vertices, float), np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError("vertices must be finite (N, 3)")
    if (f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer)
            or not len(f) or f.min() < 0 or f.max() >= len(v)):
        raise ValueError("faces must be nonempty valid integer triangles")
    return v, f.astype(np.int64, copy=False)


def _overlapping_face_pairs(bounds, face_ids):
    """Find all inclusive AABB overlaps by sweeping one coordinate axis.

    Sort lower bounds, then compare each box only with following boxes starting
    before its upper bound. The other axes reject remaining disjoint pairs.
    Touching and zero-width boxes are retained for the triangle test. Choose an
    axis with large center spread relative to box width to reduce candidates.
    """
    if len(face_ids) < 2:
        return np.empty((0, 2), int)
    selected = bounds[face_ids]
    lower, upper = selected[:, :3], selected[:, 3:]
    spread = np.ptp((lower+upper)/2, axis=0)
    axis = int(np.argmax(spread / np.maximum((upper-lower).mean(axis=0), np.finfo(float).eps)))
    order = np.argsort(lower[:, axis], kind="stable")
    lower, upper, ids = lower[order], upper[order], np.asarray(face_ids)[order]
    stops = np.searchsorted(lower[:, axis], upper[:, axis], side="right")
    pairs = []
    for i, stop in enumerate(stops):
        candidates = ids[i+1:stop]
        if not len(candidates):
            continue
        overlap = ((lower[i+1:stop] <= upper[i]).all(axis=1)
                   & (upper[i+1:stop] >= lower[i]).all(axis=1))
        other = candidates[overlap]
        if len(other):
            pairs.append(np.column_stack([np.minimum(ids[i], other), np.maximum(ids[i], other)]))
    if not pairs:
        return np.empty((0, 2), int)
    result = np.concatenate(pairs)
    return result[np.lexsort((result[:, 1], result[:, 0]))]


def mesh_intersections(vertices, faces, *, incident_contraction=1e-7):
    """Return intersecting face pairs, including overlap of incident triangles.

    A NumPy AABB sweep supplies candidate pairs. Triangle normals and all edge cross products
    give the separating axes; in-plane edge normals also handle coplanar pairs.
    Nonincident contact is included. Pairs sharing vertices/edges are
    contracted toward their respective centroids by ``incident_contraction``
    before testing. This excludes expected topological contact while
    retaining overlap beyond it; overlap confined to that tiny removed strip
    can be missed. Duplicate faces are always reported. Degenerate triangles
    are skipped here and must be checked separately. Every candidate pair is
    centered and scaled independently; interval tolerance is 1e-12 of pair
    extent. This avoids scale-dependent false contacts on small triangles.

    All pairs are tested; counts are not truncated. Neither this predicate nor
    finite triangulation certifies the underlying continuous surface.
    """
    v, f = _mesh(vertices, faces)
    if not np.isfinite(incident_contraction) or not 0 < incident_contraction < 1e-3:
        raise ValueError("incident_contraction must be between 0 and 1e-3")
    scale = np.linalg.norm(np.ptp(v, axis=0))
    if scale == 0:
        return np.empty((0, 2), int)
    triangles = ((v-v.mean(axis=0))/scale)[f]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    valid = np.linalg.norm(normals, axis=1) > np.finfo(float).eps
    if not valid.any():
        return np.empty((0, 2), int)
    bounds = np.column_stack([triangles.min(axis=1), triangles.max(axis=1)])
    pairs = _overlapping_face_pairs(bounds, np.flatnonzero(valid))
    if not len(pairs):
        return np.empty((0, 2), int)
    shared = (f[pairs[:, 0], :, None] == f[pairs[:, 1], None, :]).any(axis=2).sum(axis=1)
    a, b = triangles[pairs[:, 0]].copy(), triangles[pairs[:, 1]].copy()
    incident = (shared > 0) & (shared < 3)
    for tri in (a, b):
        tri[incident] += incident_contraction*(tri[incident].mean(axis=1)[:, None, :]-tri[incident])
    hits = []
    # Work in chunks to bound temporary arrays if many AABBs overlap.
    for start in range(0, len(pairs), 8192):
        stop = start+8192
        aa, bb = a[start:stop].copy(), b[start:stop].copy()
        both = np.concatenate([aa, bb], axis=1)
        center = both.mean(axis=1)[:, None]
        extent = np.linalg.norm(np.ptp(both, axis=1), axis=1)[:, None, None]
        aa, bb = (aa-center)/extent, (bb-center)/extent
        ea, eb = np.roll(aa, -1, axis=1)-aa, np.roll(bb, -1, axis=1)-bb
        na, nb = np.cross(ea[:, 0], ea[:, 1]), np.cross(eb[:, 0], eb[:, 1])
        axes = np.concatenate([na[:, None], nb[:, None],
                               np.cross(ea[:, :, None], eb[:, None, :]).reshape(-1, 9, 3),
                               np.cross(na[:, None], ea), np.cross(nb[:, None], eb)], axis=1)
        lengths = np.linalg.norm(axes, axis=2)
        usable = lengths > 1e-14
        axes = np.divide(axes, lengths[:, :, None], out=np.zeros_like(axes), where=usable[:, :, None])
        pa = np.einsum("pvd,pad->pav", aa, axes)
        pb = np.einsum("pvd,pad->pav", bb, axes)
        separated = (((pa.max(axis=2) < pb.min(axis=2)-1e-12)
                      | (pb.max(axis=2) < pa.min(axis=2)-1e-12)) & usable).any(axis=1)
        hits.extend(pairs[start:stop][~separated | (shared[start:stop] == 3)])
    return np.asarray(hits, int).reshape(-1, 2)


def mesh_quality_metrics(vertices, faces, *, reference_vertices=None, radius=None,
                         check_intersections=True):
    """Areas, edge stretches, normal changes, and numerical intersections.

    A >90° normal turn relative to a reference is a warning, not a proof of a
    fold: matched parameter samples may move substantially on a valid surface.
    ``geometry_warning`` means at least one collapsed face, intersection, or
    such normal turn was observed. No automatic biological acceptance threshold
    is imposed on continuous displacement or stretch values.
    """
    from .patches import surface_moments
    v, f = _mesh(vertices, faces)
    if radius is None:
        radius = surface_moments(v, f)[2]
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    tri = v[f]
    normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    areas = np.linalg.norm(normals, axis=1)/2
    collapsed = areas <= 1e-12*radius**2
    edge_squared = np.sum((np.roll(tri, -1, axis=1)-tri)**2, axis=2)
    quality = np.divide(4*np.sqrt(3)*areas, edge_squared.sum(axis=1),
                        out=np.zeros(len(f)), where=edge_squared.sum(axis=1) > 0)
    metrics = dict(n_vertices=len(v), n_faces=len(f), collapsed_faces=int(collapsed.sum()),
                   min_triangle_quality=float(quality.min()),
                   intersection_check="triangle_separating_axes" if check_intersections else "not_run",
                   incident_contraction=1e-7)
    pairs = mesh_intersections(v, f) if check_intersections else None
    metrics["intersection_pairs"] = len(pairs) if pairs is not None else None
    metrics["intersecting_faces"] = len(np.unique(pairs)) if pairs is not None else None
    turns = 0
    if reference_vertices is not None:
        ref = np.asarray(reference_vertices, float)
        if ref.shape != v.shape or not np.isfinite(ref).all():
            raise ValueError("Reference vertices must match decoded vertices")
        rt = ref[f]
        rn = np.cross(rt[:, 1]-rt[:, 0], rt[:, 2]-rt[:, 0])
        ra = np.linalg.norm(rn, axis=1)/2
        if np.any(ra <= 1e-12*radius**2):
            raise ValueError("Reference has collapsed triangles; relative checks are undefined")
        turns = int(np.count_nonzero(np.einsum("ij,ij->i", normals, rn) < 0))
        re = np.linalg.norm(np.roll(rt, -1, axis=1)-rt, axis=2)
        metrics.update(normal_turn_faces=turns, min_face_area_ratio=float((areas/ra).min()),
                       max_edge_stretch=float((np.sqrt(edge_squared)/re).max()),
                       max_displacement_relative=float(np.linalg.norm(v-ref, axis=1).max()/radius))
    metrics["geometry_warning"] = bool(collapsed.any() or (pairs is not None and len(pairs)) or turns)
    return metrics

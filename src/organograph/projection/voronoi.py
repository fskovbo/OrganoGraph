import numpy as np
import heapq


def voronoi_on_mesh_dijkstra(
    mesh,
    sources,
):
    """
    Build edge adjacency (undirected, Euclidean edge lengths) from a triangle mesh
    and run multi-source Dijkstra (geodesic Voronoi on the edge graph).

    Parameters
    ----------
    mesh : OrganoidMesh
        Contains vertices and faces stored as mesh.v and mesh.f 
    sources : (N_cells,) int
        Projected vertex id per cell. Rejected cells should have -1.

    Returns
    -------
    owner_cell : (V,) int64
        For each mesh vertex, the owning *cell index* (0..N_cells-1).
        -1 means no owner (e.g. all sources rejected).
    dist : (V,) float64
        Distance to the nearest source (edge-graph geodesic).
        np.inf if owner_cell == -1.
    """
    # -----------------------------
    # Build undirected adjacency
    # -----------------------------

    if mesh.v.ndim != 2 or mesh.v.shape[0] == 0:
        raise ValueError("mesh.v must be a non-empty array of shape (V, D).")
    if mesh.f.ndim != 2 or mesh.f.shape[1] != 3:
        raise ValueError("mesh.f must be an array of shape (F, 3) of triangle indices.")
    if np.any(mesh.f < 0) or np.any(mesh.f >= mesh.v.shape[0]):
        raise ValueError("mesh.f contains out-of-bounds vertex indices.")

    # Directed edges from faces (a->b, b->c, c->a) plus reverse to make undirected
    e01 = mesh.f[:, [0, 1]]
    e12 = mesh.f[:, [1, 2]]
    e20 = mesh.f[:, [2, 0]]
    E = np.vstack([e01, e12, e20])
    E = np.vstack([E, E[:, ::-1]])  # add reverse edges

    I = E[:, 0]
    J = E[:, 1]

    # Edge lengths
    W = np.linalg.norm(mesh.v[I] - mesh.v[J], axis=1)

    # Sort by (I, J) so duplicates can be collapsed
    order = np.lexsort((J, I))
    I = I[order]
    J = J[order]
    W = W[order]

    # Collapse duplicate (I,J) edges by taking min weight
    same = (I[1:] == I[:-1]) & (J[1:] == J[:-1])
    keep = np.ones(len(I), dtype=bool)
    keep[1:][same] = False  # keep first occurrence of each group
    group_starts = np.r_[0, np.where(~same)[0] + 1]
    Wmin = np.minimum.reduceat(W, group_starts)

    Iu = I[keep]
    Ju = J[keep]
    Wu = Wmin  # aligned with kept first-of-group edges

    Vn = mesh.v.shape[0]
    counts = np.bincount(Iu, minlength=Vn)
    offsets = np.cumsum(np.r_[0, counts])

    nbr_all = np.empty(len(Iu), dtype=np.int64)
    wt_all  = np.empty(len(Iu), dtype=np.float64)

    cursor = offsets[:-1].copy()
    for a, b, w in zip(Iu, Ju, Wu):
        k = cursor[a]
        nbr_all[k] = b
        wt_all[k] = w
        cursor[a] += 1

    nbrs = [None] * Vn
    wts  = [None] * Vn
    for i in range(Vn):
        a, b = offsets[i], offsets[i + 1]
        nbrs[i] = nbr_all[a:b]
        wts[i]  = wt_all[a:b]

    # -----------------------------
    # Multi-source Dijkstra
    # -----------------------------
    sources = np.asarray(sources, dtype=np.int64)
    V = Vn

    dist = np.full(V, np.inf, dtype=np.float64)
    owner_cell = np.full(V, -1, dtype=np.int64)

    valid_cells = np.flatnonzero(sources >= 0)  # indices into original cells
    if valid_cells.size == 0:
        return owner_cell, dist

    valid_source_vertices = sources[valid_cells]
    if np.any(valid_source_vertices >= V) or np.any(valid_source_vertices < 0):
        raise ValueError("sources contains out-of-bounds vertex ids (or negatives besides -1).")

    heap = []

    # Seed heap; if multiple cells map to same vertex, first one in valid_cells wins
    for cell_idx, s_idx in zip(valid_cells, valid_source_vertices):
        if dist[s_idx] > 0.0:
            dist[s_idx] = 0.0
            owner_cell[s_idx] = cell_idx
            heapq.heappush(heap, (0.0, s_idx))

    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        ou = owner_cell[u]
        for vv, ww in zip(nbrs[u], wts[u]):
            nd = d + ww
            if nd < dist[vv]:
                dist[vv] = nd
                owner_cell[vv] = ou
                heapq.heappush(heap, (nd, vv))

    return owner_cell, dist

import numpy as np
from organograph.projection.voronoi import voronoi_on_mesh_dijkstra

def assign_vertices_to_nuclei(mesh, nuclei_xyz, max_dist=None):
    """
    Assign each mesh vertex to a nuclei by:
    1. Project nuclei to closest mesh vertex.
    2. Assign each vertex according to closest (by Dijkstra) projected nucleus 

    Parameters
    ----------
    mesh : OrganoidMesh
        Contains vertices and faces as mesh.v and mesh.f
    nuclei_xyz : (N_cells, 3)
        Nucleus positions.
    max_dist : float (optional)
        Reject nuclei at further distance from mesh

    Returns
    -------
    vertex_owner : (V,) int64
        Assigned nucleus index for each mesh vertex
    proj_vertex_ids : (N_cells,) int64
        Vertex index of projected nuclei 
    """

    proj_vertex_ids, proj_points = project_nuclei_to_mesh(
        nuclei_xyz,
        mesh,
        resolve_duplicates=True, # if multiple nuclei are projected to the same vertex, shift them slightly
        max_dist=max_dist,
    )

    vertex_owner, dist = voronoi_on_mesh_dijkstra(mesh, proj_vertex_ids)

    return vertex_owner, proj_vertex_ids


def compute_face_normals_and_centroids(v, f):
    """
    v: (V,3), f: (F,3) int
    returns (normals: (F,3), centroids: (F,3))
    """
    tri = v[f]  # (F,3,3)
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals /= norm
    centroids = (v0 + v1 + v2) / 3.0
    return normals, centroids


def project_nuclei_to_mesh(
    nuclei_xyz,
    mesh,
    resolve_duplicates=False,
    dtype=np.float32,
    max_dist=None,
):
    """
    Project nuclei onto the mesh by nearest vertex (exact), with optional
    duplicate-resolution using neighboring vertices.

    Parameters
    ----------
    nuclei_xyz : (N_cells, 3)
        Nucleus positions.
    mesh : OrganoidMesh
        Contains vertices and faces as mesh.v and mesh.f
    resolve_duplicates : bool
        If True, detect nuclei that project to the same vertex and
        shift all but one to neighboring vertices
    dtype : np.dtype
        Computation dtype for nearest-vertex search (float32 recommended).
    max_dist : float
        Reject nuclei at distance further than max_dist from mesh 
        (proj_vertex_id flagged as -1 and proj_points as NaN)

    Returns
    -------
    proj_vertex_ids : (N_cells,) int64
        Vertex index used as geodesic source for each cell.
    proj_points : (N_cells, 3)
        Projected points on the mesh surface (at vertices).
    """
    # --- Input normalization (do NOT overwrite caller's arrays)
    nuclei_xyz_in = np.asarray(nuclei_xyz)
    mesh_v_in = np.asarray(mesh.v)
    mesh_f_in = np.asarray(mesh.f, dtype=np.int64)

    if nuclei_xyz_in.ndim != 2 or nuclei_xyz_in.shape[1] != 3:
        raise ValueError("nuclei_xyz must have shape (N_cells, 3)")
    if mesh_v_in.ndim != 2 or mesh_v_in.shape[1] != 3:
        raise ValueError("mesh_v must have shape (V, 3)")
    if mesh_f_in.ndim != 2 or mesh_f_in.shape[1] != 3:
        raise ValueError("mesh_f must have shape (F, 3)")

    N_cells = nuclei_xyz_in.shape[0]
    proj_vertex_ids = np.empty(N_cells, dtype=np.int64)
    proj_points = np.empty_like(nuclei_xyz_in)

    # ------------------------------------------------------------------
    # 1) Basic projection: exact nearest vertex for each nucleus (NumPy-only)
    #    Uses: ||x - v||^2 = ||x||^2 + ||v||^2 - 2 x·v
    #    Fast because x @ v.T uses optimized BLAS when contiguous.
    # ------------------------------------------------------------------
    # Working arrays for fast math (copy ONLY if needed: non-contiguous and/or dtype mismatch)
    x = np.ascontiguousarray(nuclei_xyz_in, dtype=dtype)  # (N_cells, 3)
    v = np.ascontiguousarray(mesh_v_in, dtype=dtype)      # (V, 3)

    # Precompute squared norms
    v_norm2 = np.einsum("ij,ij->i", v, v)  # (V,)
    x_norm2 = np.einsum("ij,ij->i", x, x)  # (N_cells,)

    # Dot products (N_cells, V)
    dots = x @ v.T

    # Squared distances
    dist2 = x_norm2[:, None] + v_norm2[None, :] - 2.0 * dots

    # Nearest vertex indices
    vid = np.argmin(dist2, axis=1).astype(np.int64)               # (N_cells,)
    min_dist2 = dist2[np.arange(dist2.shape[0]), vid]             # (N_cells,)

    # Optional rejection threshold
    if max_dist is not None:
        max_dist2 = (dtype.type(max_dist) ** 2) if hasattr(dtype, "type") else (max_dist ** 2)
        mask = min_dist2 <= max_dist2
    else:
        mask = np.ones_like(vid, dtype=bool)

    # Fill outputs
    proj_vertex_ids[:] = -1
    proj_points[:] = np.nan

    proj_vertex_ids[mask] = vid[mask]
    proj_points[mask] = mesh_v_in[vid[mask]]


    # ------------------------------------------------------------------
    # 2) Optional: handle duplicate projections using tangential direction
    # ------------------------------------------------------------------
    if not resolve_duplicates:
        return proj_vertex_ids, proj_points

    # Find vertices with >1 projected cell (IGNORE rejected nuclei: -1)
    valid = proj_vertex_ids >= 0
    if not np.any(valid):
        print("Duplicate projections: none (all nuclei rejected).")
        return proj_vertex_ids, proj_points

    unique_vids, counts = np.unique(proj_vertex_ids[valid], return_counts=True)
    dup_vids = unique_vids[counts > 1]
    dup_counts = counts[counts > 1]

    if dup_vids.size == 0:
        print("Duplicate projections: none.")
        return proj_vertex_ids, proj_points

    total_pairs = int(np.sum(dup_counts * (dup_counts - 1) // 2))
    print(
        f"Duplicate projections: {len(dup_vids)} vertices with duplicates, "
        f"{total_pairs} duplicate cell pairs."
    )

    # ---- Build vertex adjacency (1-ring neighbors)
    V = mesh_v_in.shape[0]
    neighbors = [[] for _ in range(V)]
    for a, b, c in mesh_f_in:
        neighbors[a].extend([b, c])
        neighbors[b].extend([a, c])
        neighbors[c].extend([a, b])
    neighbors = [list(set(nl)) for nl in neighbors]

    # ---- Precompute vertex normals (for tangential projection)
    #     Normal = average of adjacent face normals
    # NOTE: This assumes compute_face_normals_and_centroids uses mesh_v coords correctly.
    face_normals, _ = compute_face_normals_and_centroids(mesh_v_in, mesh_f_in)

    vertex_normals = np.zeros_like(mesh_v_in, dtype=np.float64)
    for fi, (a, b, c) in enumerate(mesh_f_in):
        n = face_normals[fi]
        vertex_normals[a] += n
        vertex_normals[b] += n
        vertex_normals[c] += n

    # normalize
    vn_norm = np.linalg.norm(vertex_normals, axis=1)
    vn_norm[vn_norm == 0] = 1.0
    vertex_normals = (vertex_normals / vn_norm[:, None]).astype(mesh_v_in.dtype, copy=False)

    # ---- Process each duplicated vertex
    for v_id, n_here in zip(dup_vids, dup_counts):
        if v_id < 0:
            continue

        idxs_here = np.where(proj_vertex_ids == v_id)[0]
        if n_here <= 1:
            continue

        neighs = neighbors[v_id]
        if len(neighs) == 0:
            continue

        base_pos = mesh_v_in[v_id]
        normal = vertex_normals[v_id]

        # Assign one cell to stay at v_id
        # keep_idx = idxs_here[0]  # (unused, but kept conceptually)

        # Remaining cells
        rest = idxs_here[1:]
        if rest.size == 0:
            continue

        # For each cell: compute tangential displacement direction
        nucleus_block = nuclei_xyz_in[idxs_here]
        center = nucleus_block.mean(axis=0)

        tangential_vectors = []
        for cell_idx in rest:
            d = nuclei_xyz_in[cell_idx] - center
            d_perp = np.dot(d, normal) * normal
            d_tan = d - d_perp
            nrm = np.linalg.norm(d_tan)
            if nrm < 1e-12:
                d_tan = np.random.randn(3)
                nrm = np.linalg.norm(d_tan)
                if nrm < 1e-12:
                    continue
            tangential_vectors.append((cell_idx, d_tan / nrm))

        if not tangential_vectors:
            continue

        # Determine directions from v_id to neighbors (projected to tangent plane)
        neigh_dirs = []
        for u in neighs:
            d = mesh_v_in[u] - base_pos
            d_perp = np.dot(d, normal) * normal
            d_tan = d - d_perp
            nrm = np.linalg.norm(d_tan)
            if nrm < 1e-12:
                continue
            neigh_dirs.append((u, d_tan / nrm))

        if not neigh_dirs:
            continue

        # Assign each cell to best matching neighbor based on cosine similarity
        used = set()
        for cell_idx, d_tan in tangential_vectors:
            scores = [(float(np.dot(d_tan, ndir)), u) for (u, ndir) in neigh_dirs if u not in used]
            if not scores:
                continue
            _, best_u = max(scores, key=lambda x: x[0])
            used.add(best_u)

            proj_vertex_ids[cell_idx] = best_u
            proj_points[cell_idx] = mesh_v_in[best_u]

    return proj_vertex_ids, proj_points
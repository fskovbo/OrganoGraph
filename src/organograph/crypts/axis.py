import numpy as np
import heapq
from scipy.signal import savgol_filter
from organograph.crypts.analysis import crypt_circumference


# =============================================================================
# Crypt boundary and bottom 
# =============================================================================

def _get_patch_faces(mesh, patch_vertices):
    """
    Return faces fully contained in a vertex patch.

    Parameters
    ----------
    mesh
        Mesh object with:
          - mesh.f : (F,3) int array of triangle vertex indices (global vertex IDs).
    patch_vertices
        1D iterable/array of int
        Global vertex IDs belonging to the patch.

    Returns
    -------
    patch_faces : (Fp,3) int64 ndarray
        Subset of mesh.f where all three vertices are in patch_vertices.
        Uses global vertex indices (no remapping).
    """
    f = np.asarray(mesh.f, dtype=np.int64)
    pv = np.asarray(patch_vertices, dtype=np.int64)

    if pv.ndim != 1:
        raise ValueError("patch_vertices must be 1D.")
    if pv.size == 0:
        return np.empty((0, 3), dtype=np.int64)

    keep = np.isin(f, pv).all(axis=1)
    return f[keep]


def _get_boundary_vertices(
    mesh,
    *,
    patch_vertices=None,
    patch_faces=None,
):
    """
    Compute boundary vertices of a patch.

    The patch can be specified either by a set of vertices or by its faces.

    Parameters
    ----------
    mesh
        Mesh object with attribute:
            mesh.f : (F,3) int array
                Triangle vertex indices.

    patch_vertices : 1D iterable[int], optional
        Global vertex indices belonging to the patch.
        If provided, patch faces will be computed internally.

    patch_faces : (Fp,3) int array, optional
        Faces fully contained in the patch (global vertex indices).
        If provided, this is used directly and avoids recomputing patch faces.

    Returns
    -------
    boundary_vertices : (B,) int64 ndarray
        Global vertex IDs lying on the boundary of the patch.
    """

    if (patch_vertices is None) == (patch_faces is None):
        raise ValueError("Provide exactly one of patch_vertices or patch_faces.")

    if patch_faces is None:
        # compute patch faces from vertices
        f = np.asarray(mesh.f, dtype=np.int64)
        pv = np.asarray(patch_vertices, dtype=np.int64)

        if pv.ndim != 1:
            raise ValueError("patch_vertices must be 1D.")
        if pv.size == 0:
            return np.empty((0,), dtype=np.int64)

        keep = np.isin(f, pv).all(axis=1)
        patch_faces = f[keep]

    else:
        patch_faces = np.asarray(patch_faces, dtype=np.int64)

    if patch_faces.size == 0:
        return np.empty((0,), dtype=np.int64)

    # build edge list
    edges = np.vstack([
        patch_faces[:, [0, 1]],
        patch_faces[:, [1, 2]],
        patch_faces[:, [2, 0]],
    ])
    edges = np.sort(edges, axis=1)

    # boundary edges occur only once
    uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = uniq_edges[counts == 1]

    return np.unique(boundary_edges.reshape(-1))


def _sample_boundary(boundary: np.ndarray, n_samples: int, rng: np.random.Generator | None) -> np.ndarray:
    """
    Very fast sampling: uniform random without replacement.
    """
    boundary = np.asarray(boundary, dtype=np.int64)
    B = boundary.size
    if n_samples is None or n_samples <= 0 or n_samples >= B:
        return boundary
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice(boundary, size=int(n_samples), replace=False)


def _crypt_bottom_single_patch_multisource_dijkstra(
    mesh,
    patch: set[int],
    *,
    n_boundary_samples: int = 256,
    rng: np.random.Generator | None = None,
):
    """
    Returns:
        bottom_vertex : int (global vertex index)
        sources       : (S,) sampled boundary vertex indices (global)
        dist_patch    : (P,) distances for patch vertices (local order = patch_vertices)
        patch_vertices: (P,) patch vertices in local order (global indices)
    """
    V = mesh.v.shape[0]
    patch_vertices = np.fromiter(patch, dtype=np.int64)
    if patch_vertices.size == 0:
        raise ValueError("Patch is empty.")

    patch_faces = _get_patch_faces(mesh, patch_vertices)
    if patch_faces.size == 0:
        # Degenerate patch with no faces; pick arbitrary vertex
        bottom = int(patch_vertices[0])
        sources = np.empty((0,), dtype=np.int64)
        dist_patch = np.zeros(patch_vertices.size, dtype=np.float64)
        return bottom, sources, dist_patch, patch_vertices

    boundary = _get_boundary_vertices(mesh, patch_faces=patch_faces)
    if boundary.size == 0:
        raise ValueError("Patch boundary is empty (patch appears closed); cannot define distance-to-boundary.")
    sources = _sample_boundary(boundary, n_boundary_samples, rng)

    # Compact local indexing for patch vertices
    P = patch_vertices.size
    local_id = np.full(V, -1, dtype=np.int64)
    local_id[patch_vertices] = np.arange(P, dtype=np.int64)

    # Unique patch edges from patch faces
    f = patch_faces
    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    # Convert to local indices
    ei = local_id[edges[:, 0]]
    ej = local_id[edges[:, 1]]

    # Edge lengths
    vpos = np.asarray(mesh.v, float)
    w = np.linalg.norm(vpos[edges[:, 0]] - vpos[edges[:, 1]], axis=1).astype(np.float64)

    # Adjacency list on local vertices
    nbrs = [[] for _ in range(P)]
    for a, b, ww in zip(ei, ej, w):
        if a >= 0 and b >= 0:
            a = int(a); b = int(b); ww = float(ww)
            nbrs[a].append((b, ww))
            nbrs[b].append((a, ww))

    # Multi-source Dijkstra
    dist = np.full(P, np.inf, dtype=np.float64)
    pq = []

    src_local = local_id[sources]
    src_local = src_local[src_local >= 0]  # safety

    for s in src_local:
        dist[int(s)] = 0.0
        pq.append((0.0, int(s)))
    heapq.heapify(pq)

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, ww in nbrs[u]:
            nd = d + ww
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    finite = np.isfinite(dist)
    if not np.any(finite):
        bottom_local = 0
    else:
        bottom_local = int(np.argmax(np.where(finite, dist, -np.inf)))

    bottom_vertex = int(patch_vertices[bottom_local])
    return bottom_vertex, sources, dist, patch_vertices


def compute_crypt_bottoms(
    mesh,
    crypt_patches: list[set[int]],
    *,
    n_boundary_samples: int = 256,
    seed: int | None = None,
    return_details: bool = False,
):
    """
    Compute crypt bottoms for ALL patches of one organoid.

    Args:
        mesh: object with mesh.v (V,3) and mesh.f (F,3)
        crypt_patches: list[set[int]] patch vertex sets
        n_boundary_samples: number of boundary vertices to sample per patch
        seed: set for reproducible sampling (same seed -> same samples)
        return_details: if True, also return sampled sources per patch

    Returns:
        bottoms: list[int] bottom vertex (global index) per patch, in same order
        (optional) details: list[dict] with sources, dist stats, etc.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    bottoms = []
    details = []

    for p in crypt_patches:
        btm, srcs, dist_patch, patch_vertices = _crypt_bottom_single_patch_multisource_dijkstra(
            mesh, p, n_boundary_samples=n_boundary_samples, rng=rng
        )
        bottoms.append(int(btm))

        if return_details:
            # lightweight diagnostics
            finite = dist_patch[np.isfinite(dist_patch)]
            details.append({
                "bottom": int(btm),
                "n_patch_vertices": int(len(p)),
                "n_sources": int(len(srcs)),
                "sources": srcs,
                "dist_max": float(np.max(finite)) if finite.size else float("nan"),
                "dist_mean": float(np.mean(finite)) if finite.size else float("nan"),
            })

    return (bottoms, details) if return_details else bottoms



# =============================================================================
# Crypt axis and rescaling
# =============================================================================


def compute_crypt_axis(
    mesh,
    crypt_patches,
    geodesic_fn,
    geodesic_kwargs=None,
):
    """
    Compute per-feature (crypt) distance fields from feature bottoms.

    Inputs
    ------
    mesh : OrganoidMesh
        Surface mesh with attributes:
        - v : (V,3) vertices
        - f : (F,3) faces
    crypt_patches : list[set[int]]
        Cell-id sets defining each feature (crypt).
    boundary_patches : list[set[int]]
        Cell-id sets defining boundary regions used to normalize length.
    geodesic_fn : callable
        Geodesic routine called as geodesic_fn(mesh, sources=[...], **kwargs)
        Must return (S,V) or (V,) distances.
    geodesic_kwargs : dict or None
        Extra keyword args forwarded to geodesic_fn.

    Returns
    -------
    dnorm_all : (K, V) float
        Normalized vertex distances (divided by boundary mean length).
        NaN for invalid features.
    L_crypt_all : (K,) float
        Mean boundary distance per feature (normalization length).
        NaN for invalid features.
    bottom_vertex_ids : (K,) int
        Bottom projected vertex id per feature (-1 if invalid).
    """
    if geodesic_kwargs is None:
        geodesic_kwargs = {}

    K = len(crypt_patches)
    V = mesh.v.shape[0]

    bottom_vertex_ids = compute_crypt_bottoms(mesh, crypt_patches)
    bottom_vertex_ids = np.asarray(bottom_vertex_ids)
    valid_js = np.where(bottom_vertex_ids >= 0)[0]

    dnorm_all = np.full((K, V), np.nan, dtype=float)
    L_crypt_all = np.full(K, np.nan, dtype=float)

    if len(valid_js) == 0:
        return dnorm_all, L_crypt_all, bottom_vertex_ids

    # --- 2) geodesics from all valid bottoms ---
    sources = [int(s) for s in bottom_vertex_ids[valid_js]]
    D_multi = np.asarray(geodesic_fn(mesh, sources=sources, **geodesic_kwargs))
    if D_multi.ndim == 1:
        D_multi = D_multi[None, :]

    # --- 3) normalize per feature ---
    for r, j in enumerate(valid_js):
        dist_vertices = D_multi[r]  # (V,)

        patch_vertices = np.fromiter(crypt_patches[r], dtype=np.int64)
        boundary_vertex_ids = _get_boundary_vertices(mesh, patch_vertices=patch_vertices)


        L_mean = float(np.mean(dist_vertices[boundary_vertex_ids]))
        L_mean = max(L_mean, 1e-12)

        dnorm_all[j] = dist_vertices / L_mean
        L_crypt_all[j] = L_mean

    return dnorm_all, L_crypt_all, bottom_vertex_ids



def normalize_crypt_axis_to_neckline(
    mesh,
    dnorm_vertices,
    bin_centers,
    search_interval=(0.75, 1.25),
    L_crypt=None,
    window_length=9,
    polyorder=3,
    min_prominence=0.05,
):
    """
    Rescale feature axis so that the narrowest circumference occurs at s=1,
    without recomputing circumference on the mesh after rescaling.

    Inputs
    ------
    mesh : OrganoidMesh
        Mesh used by crypt_circumference().
    dnorm_vertices : (V,) or (K,V) array
        Normalized distances at mesh vertices (per feature).
    bin_centers : (B,) array
        Canonical axis levels (used elsewhere; only axis exposed to caller).
    search_interval : (lo, hi)
        Interval for s_star search. Also defines internal axis stretching via hi.
    L_crypt : optional, scalar or (K,) array
        Crypt length proxy in physical units (per feature).
    window_length, polyorder, min_prominence :
        Used internally for smoothing and minimum selection.

    Returns (ALWAYS BATCHED)
    -----------------------
    CC_rescaled : (K, B) float
        Circumference curves sampled on bin_centers after axis rescaling.
    dnorm_vertices_rescaled : (K, V) float
        Rescaled vertex distances: dnorm_vertices / s_star.
    L_crypt_rescaled : (K,) float or None
        Rescaled length proxy: L_crypt * s_star (or None if L_crypt not given).
    """
    s = np.asarray(bin_centers, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("bin_centers must be a non-empty 1D array")

    lo, hi = float(search_interval[0]), float(search_interval[1])
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        raise ValueError("search_interval must be (lo, hi) with 0 < lo < hi")

    dv = np.asarray(dnorm_vertices, dtype=float)
    if dv.ndim == 1:
        dv = dv[None, :]  # (1, V)
    elif dv.ndim != 2:
        raise ValueError("dnorm_vertices must be shape (V,) or (K,V)")

    K, V = dv.shape

    L = None
    if L_crypt is not None:
        L = np.asarray(L_crypt, dtype=float)
        if L.ndim == 0:
            L = np.full(K, float(L), dtype=float)
        if L.ndim != 1 or L.shape[0] != K:
            raise ValueError("L_crypt must be scalar or shape (K,) matching dnorm_vertices' K")

    # wide axis to avoid extrapolation for s_query = s * s_star
    s_wide = np.linspace(0.0, float(s.max()) * hi, s.size)

    CC_rescaled = np.full((K, s.size), np.nan, dtype=float)
    s_star = np.ones(K, dtype=float)

    for k in range(K):
        CC0 = np.asarray(
            crypt_circumference(mesh=mesh, crypt_dist=dv[k], levels=s_wide),
            dtype=float,
        )

        # -------- inline: norm_dist_to_neckline(s_wide, CC0, ...) --------
        sw = np.asarray(s_wide, dtype=float)
        Cw = np.asarray(CC0, dtype=float)

        ss = 1.0
        if sw.ndim == 1 and Cw.ndim == 1 and sw.size == Cw.size and sw.size >= 7:
            # sort by s
            order = np.argsort(sw)
            s0 = sw[order]
            C0 = Cw[order]

            # fill NaNs in C
            m = np.isfinite(C0)
            if np.sum(m) >= 7:
                Cfill = np.interp(s0, s0[m], C0[m])

                # Savitzky–Golay smoothing (good for derivatives)
                wl = int(window_length)
                if wl % 2 == 0:
                    wl += 1
                wl = min(wl, s0.size if s0.size % 2 == 1 else s0.size - 1)
                wl = max(wl, 5)
                po = int(polyorder)
                po = min(po, wl - 2)
                Cs = savgol_filter(Cfill, window_length=wl, polyorder=po, mode="interp")

                # derivatives
                d1 = np.gradient(Cs, s0)
                d2 = np.gradient(d1, s0)

                win = np.where((s0 >= lo) & (s0 <= hi))[0]
                if win.size >= 3:
                    idx = win[(win > 0) & (win < len(Cs) - 1)]
                    minima = idx[(Cs[idx - 1] > Cs[idx]) & (Cs[idx] < Cs[idx + 1])]

                    # optional prominence filter
                    if minima.size and min_prominence > 0:
                        keep = []
                        for i in minima:
                            left = win[win <= i]
                            right = win[win >= i]
                            if left.size == 0 or right.size == 0:
                                continue
                            mref = max(np.max(Cs[left]), np.max(Cs[right]), 1e-12)
                            prom = (mref - Cs[i]) / mref
                            if prom >= float(min_prominence):
                                keep.append(i)
                        minima = np.asarray(keep, dtype=np.int64)

                    if minima.size:
                        i_star = minima[np.argmin(Cs[minima])]  # deepest minimum
                        s_tmp = float(s0[i_star])
                        if np.isfinite(s_tmp) and s_tmp > 0:
                            ss = s_tmp
                    else:
                        # fallback: inflection = local max of d2 with d1>0
                        cand = idx[d1[idx] > 0]
                        infl = cand[(d2[cand - 1] < d2[cand]) & (d2[cand] > d2[cand + 1])]
                        if infl.size:
                            i_star = infl[np.argmax(d2[infl])]
                            s_tmp = float(s0[i_star])
                            if np.isfinite(s_tmp) and s_tmp > 0:
                                ss = s_tmp
        # ----------------------------------------------------------------

        if not (np.isfinite(ss) and ss > 0):
            ss = 1.0
        s_star[k] = ss

        s_query = np.clip(s * ss, s_wide[0], s_wide[-1])
        CC_rescaled[k] = np.interp(s_query, s_wide, CC0)

    dv_rescaled = dv / np.maximum(s_star[:, None], 1e-12)
    L_rescaled = (L * s_star) if L is not None else None

    return CC_rescaled, dv_rescaled, L_rescaled



def assign_features_by_distance(dnorm_per_feature, s_thresh=1.0):
    """
    Assign each item (cell, vertex, etc.) to at most one feature using
    a normalized-distance threshold and nearest-feature rule.

    Rule
    ----
    An item i is eligible for feature k if:
        dnorm_per_feature[k, i] < s_thresh

    If multiple features qualify, the item is assigned to the feature
    with the smallest distance.

    This works identically whether “items” are:
      - cells  → distances at cell centers
      - vertices → distances at mesh vertices
      - any other indexed objects

    Parameters
    ----------
    dnorm_per_feature : array, shape (K, N_items)
        Normalized distances from each feature k to each item i.
        Example:
            K = number of crypts/features
            N_items = number of cells OR vertices
        Distances should already include any axis rescaling (e.g. / s_star).
    s_thresh : float
        Threshold for membership (default 1.0).

    Returns
    -------
    feature_patches : list[set[int]]
        Disjoint sets of assigned item indices, one set per feature (length K).
    best_feature : (N_items,) int
        Assigned feature index per item, or -1 if unassigned.
    best_dist : (N_items,) float
        Winning (smallest) distance per item, or +inf if unassigned.
    """
    D = np.asarray(dnorm_per_feature, dtype=float)
    if D.ndim != 2:
        raise ValueError("dnorm_per_feature must have shape (K, N_items)")

    K, N_items = D.shape

    best_dist = np.full(N_items, np.inf, dtype=float)
    best_feature = np.full(N_items, -1, dtype=int)

    for k in range(K):
        dk = D[k]
        mask = np.isfinite(dk) & (dk < float(s_thresh)) & (dk < best_dist)
        best_dist[mask] = dk[mask]
        best_feature[mask] = k

    feature_patches = [
        set(np.where(best_feature == k)[0].tolist())
        for k in range(K)
    ]

    return feature_patches, best_feature, best_dist


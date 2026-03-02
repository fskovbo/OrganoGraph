import numpy as np

from organograph.crypts.features import (
    seed_features_by_vocab,
    grow_crypts_toward_necks,
    assign_features_by_distance,
)
from organograph.crypts.axis import calc_crypt_axis
from organograph.crypts.analysis import crypt_circumference  # uses your iso-contour length routine
from organograph.mesh.geodesics import compute_geodesics_dijkstra
from organograph.graph.access import graph_get


# ============================================================
# FAR-SCAN neckline detection helpers (geometry-driven, fast)
# ============================================================

class _UnionFind:
    """Tiny Union-Find for connected-components on the iso-contour intersection graph."""
    __slots__ = ("p", "r")

    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int32)
        self.r = np.zeros(n, dtype=np.int8)

    def find(self, a: int) -> int:
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        r = self.r
        p = self.p
        if r[ra] < r[rb]:
            p[ra] = rb
            return rb
        if r[ra] > r[rb]:
            p[rb] = ra
            return ra
        p[rb] = ra
        r[ra] += 1
        return ra


def organoid_length_from_dist(crypt_dist_v, q: float = 95.0, eps: float = 1e-12) -> float:
    """
    Robust proxy for "organoid length" measured on the surface from a crypt bottom.

    This is intentionally simple and fast: use a high percentile of the bottom-distance field
    instead of max(), which is sensitive to outliers (spikes / stray triangles).

    Parameters
    ----------
    crypt_dist_v : (V,) array_like
        Geodesic distances from the crypt bottom to each mesh vertex (physical units).
    q : float
        Percentile to use (typical: 90–99).
    eps : float
        Minimum returned length.

    Returns
    -------
    L_org : float
        Robust organoid length proxy (physical units).
    """
    d = np.asarray(crypt_dist_v, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float(eps)
    return max(float(np.percentile(d, q)), float(eps))


def iso_contour_metrics(
    mesh,
    scalar_v,
    levels,
    *,
    vertex_score=None,
    score_thresh: float = 0.5,
    strict_crossing: bool = True,
):
    """
    Compute per-level iso-contour metrics of a scalar field on a triangle mesh.

    This is used to (a) measure circumference (total iso-contour length) and (b) estimate
    whether the iso-contour is "ring-like" via a connected-components count.

    Implementation detail
    ---------------------
    For a fixed iso-level t, each mesh edge can be intersected at most once (ignoring degeneracies).
    We represent each intersection by its *edge key* (min(i,j), max(i,j)) which is unique.
    Each active triangle contributes either 0 or 1 segment connecting two such edge-intersections.
    We then compute:
      - total contour length: sum of segment lengths
      - number of components: CC count of the segment graph via Union-Find

    Optional neck support
    ---------------------
    If `vertex_score` is provided (e.g. neck confidence per vertex in [0,1]),
    we also compute:
      - support_mean: length-weighted mean score on the contour
      - support_frac_above: length-weighted fraction of contour with score >= score_thresh

    Parameters
    ----------
    mesh : OrganoidMesh-like
        Must provide `mesh.v` (V,3) vertices and `mesh.f` (F,3) faces.
    scalar_v : (V,) array_like
        Scalar field on vertices (e.g. crypt bottom distance in physical units).
    levels : (K,) array_like
        Iso-levels at which to evaluate the contour.
    vertex_score : (V,) array_like or None
        Optional per-vertex score (e.g. neck confidence). If None, support outputs are NaN.
    score_thresh : float
        Threshold used to compute support_frac_above.
    strict_crossing : bool
        If True, only count strict sign changes across an edge to avoid degeneracies
        when scalar equals the level on vertices.

    Returns
    -------
    metrics : dict[str, np.ndarray]
        Keys:
          - "length": (K,) total contour length
          - "n_components": (K,) number of connected components
          - "support_mean": (K,) NaN if vertex_score None, else length-weighted mean on contour
          - "support_frac_above": (K,) NaN if vertex_score None, else length-weighted frac above thresh
    """
    v = np.asarray(mesh.v, dtype=float)
    f = np.asarray(mesh.f, dtype=np.int64)
    s = np.asarray(scalar_v, dtype=float)
    lev = np.asarray(levels, dtype=float)

    V = v.shape[0]
    vs = None
    if vertex_score is not None:
        vs = np.asarray(vertex_score, dtype=float)
        if vs.shape[0] != V:
            raise ValueError("vertex_score must have shape (V,)")

    tri_s = s[f]  # (F,3)
    tri_min = np.min(tri_s, axis=1)
    tri_max = np.max(tri_s, axis=1)

    out_len = np.zeros(len(lev), dtype=float)
    out_comp = np.zeros(len(lev), dtype=np.int32)
    out_sup = np.full(len(lev), np.nan, dtype=float)
    out_frac = np.full(len(lev), np.nan, dtype=float)

    def _edge_key(i, j):
        i = int(i); j = int(j)
        return (i, j) if i < j else (j, i)

    for li, t in enumerate(lev):
        active = (tri_min <= t) & (tri_max >= t)
        if not np.any(active):
            continue

        tri_a = f[active]
        tri_s_a = tri_s[active]

        # edge-intersection nodes for this level
        edge_to_node = {}
        node_points = []
        node_scores = []
        segs = []  # (u, v, seg_len, seg_score_mean, seg_frac_mean)

        def _cross(si, sj):
            if strict_crossing:
                return (si < t and sj > t) or (si > t and sj < t)
            return (si <= t and sj >= t) or (si >= t and sj <= t)

        def _add_node(i, j, wi):
            k = _edge_key(i, j)
            idx = edge_to_node.get(k, None)
            if idx is not None:
                return idx
            pi = v[i]
            pj = v[j]
            p = pi + wi * (pj - pi)
            edge_to_node[k] = len(node_points)
            node_points.append(p)
            if vs is None:
                node_scores.append(np.nan)
            else:
                node_scores.append(float(vs[i] + wi * (vs[j] - vs[i])))
            return edge_to_node[k]

        for verts, sval in zip(tri_a, tri_s_a):
            a, b, c = int(verts[0]), int(verts[1]), int(verts[2])
            sa, sb, sc = float(sval[0]), float(sval[1]), float(sval[2])

            hits = []
            if _cross(sa, sb):
                w = (t - sa) / (sb - sa)
                hits.append(_add_node(a, b, float(w)))
            if _cross(sb, sc):
                w = (t - sb) / (sc - sb)
                hits.append(_add_node(b, c, float(w)))
            if _cross(sc, sa):
                w = (t - sc) / (sa - sc)
                hits.append(_add_node(c, a, float(w)))

            if len(hits) != 2:
                continue

            u, widx = hits[0], hits[1]
            pu = node_points[u]
            pw = node_points[widx]
            seg_len = float(np.linalg.norm(pw - pu))

            if vs is None:
                seg_score = np.nan
                seg_frac = np.nan
            else:
                su = node_scores[u]
                sw = node_scores[widx]
                seg_score = 0.5 * (su + sw)
                seg_frac = 0.5 * (float(su >= score_thresh) + float(sw >= score_thresh))

            segs.append((u, widx, seg_len, seg_score, seg_frac))

        if not segs:
            continue

        total_len = float(sum(seg[2] for seg in segs))
        out_len[li] = total_len

        # components of the intersection graph
        uf = _UnionFind(len(node_points))
        for (u, widx, _, _, _) in segs:
            uf.union(u, widx)
        roots = np.array([uf.find(i) for i in range(len(node_points))], dtype=np.int32)
        out_comp[li] = int(np.unique(roots).size)

        if vs is not None:
            wsum = max(total_len, 1e-12)
            out_sup[li] = float(sum(seg_len * seg_score for (_, _, seg_len, seg_score, _) in segs) / wsum)
            out_frac[li] = float(sum(seg_len * seg_frac for (_, _, seg_len, _, seg_frac) in segs) / wsum)

    return {
        "length": out_len,
        "n_components": out_comp,
        "support_mean": out_sup,
        "support_frac_above": out_frac,
    }


def _local_minima_1d(y):
    """Return indices i where y[i-1] > y[i] < y[i+1]."""
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return np.array([], dtype=np.int64)
    i = np.arange(1, y.size - 1)
    return i[(y[i - 1] > y[i]) & (y[i] < y[i + 1])]


def pick_neckline_far_scan(
    mesh,
    crypt_dist_v,
    *,
    max_frac: float = 0.5,
    min_frac: float = 0.05,
    n_levels: int = 160,
    length_percentile: float = 95.0,
    neck_score_v=None,
    neck_score_thresh: float = 0.4,
    w_len: float = 1.0,
    w_comp: float = 0.35,
    w_neck: float = 0.0,
):
    """
    Pick a neckline distance t* (physical units) by scanning a large range from the crypt bottom.

    This implements the "scan in organoid-length units" idea:
      - compute L_org = percentile(dist) from bottom
      - scan t in [min_frac*L_org, max_frac*L_org]
      - compute circumference C(t) and number of contour components n_comp(t)
      - pick among local minima of C(t) the one with best score

    Score
    -----
      score(t) = - w_len * C_norm(t) - w_comp * (n_comp(t) - 1) + w_neck * neck_support(t)

    Notes
    -----
    - By default `w_neck=0` because neck signal can be sparse; you can enable it later.
    - `n_comp` penalizes fragmented iso-contours; 1 component is preferred.

    Parameters
    ----------
    mesh : OrganoidMesh-like
        Must provide `mesh.v` and `mesh.f`.
    crypt_dist_v : (V,) array_like
        Bottom-distance field in physical units.
    max_frac, min_frac : float
        Scan window as fractions of L_org.
    n_levels : int
        Number of iso-level samples in the scan window.
    length_percentile : float
        Percentile used for L_org.
    neck_score_v : (V,) array_like or None
        Optional per-vertex neck confidence in [0,1].
    neck_score_thresh : float
        Threshold for support_frac_above (only relevant if neck_score_v is given).
    w_len, w_comp, w_neck : float
        Weights for score components.

    Returns
    -------
    t_star : float
        Selected neckline distance (physical units).
    info : dict
        Debug information (levels, C curve, component counts, etc.).
    """
    d = np.asarray(crypt_dist_v, dtype=float)
    L_org = organoid_length_from_dist(d, q=length_percentile)

    t_lo = float(min_frac * L_org)
    t_hi = float(max_frac * L_org)
    if not (np.isfinite(t_lo) and np.isfinite(t_hi) and t_hi > t_lo):
        # fallback: scan full finite range
        finite = d[np.isfinite(d)]
        t_lo = float(np.min(finite))
        t_hi = float(np.max(finite))

    levels = np.linspace(t_lo, t_hi, int(n_levels), dtype=float)

    met = iso_contour_metrics(
        mesh=mesh,
        scalar_v=d,
        levels=levels,
        vertex_score=neck_score_v,
        score_thresh=neck_score_thresh,
        strict_crossing=True,
    )

    C = np.asarray(met["length"], dtype=float)
    ncomp = np.asarray(met["n_components"], dtype=float)
    sup = np.asarray(met["support_mean"], dtype=float)
    sup = np.where(np.isfinite(sup), sup, 0.0)

    # local minima candidates; fallback to global min
    cand = _local_minima_1d(C)
    if cand.size == 0:
        cand = np.array([int(np.nanargmin(C))], dtype=np.int64)

    # normalize C for scoring
    m = np.isfinite(C)
    if np.sum(m) >= 2:
        cmin = float(np.nanmin(C[m]))
        cmax = float(np.nanmax(C[m]))
        denom = max(cmax - cmin, 1e-12)
        Cn = (C - cmin) / denom
    else:
        Cn = np.zeros_like(C)

    comp_pen = np.maximum(ncomp - 1.0, 0.0)
    scores = -float(w_len) * Cn - float(w_comp) * comp_pen + float(w_neck) * sup

    best_idx = int(cand[np.argmax(scores[cand])])
    t_star = float(levels[best_idx])

    info = {
        "L_org": float(L_org),
        "levels": levels,
        "C": C,
        "n_components": met["n_components"],
        "support_mean": met["support_mean"],
        "support_frac_above": met["support_frac_above"],
        "scores": scores,
        "candidates": cand,
        "best_idx": best_idx,
        "t_lo": t_lo,
        "t_hi": t_hi,
    }
    return t_star, info


# ============================================================
# New segmentation function (no plateau adjustment)
# ============================================================

def segment_crypts_organoid(
    G,
    mesh,
    bin_centers,
    crypt_vocab_idx,
    neck_vocab_idx=None,
    *,
    crypt_seed_thresh: float = 0.2,
    neck_seed_thresh: float = 0.5,
    min_crypt_seed_size: int = 10,
    grow_steps: int = 2,
    geodesic_fn=compute_geodesics_dijkstra,
    geodesic_kwargs=None,
    # --- far-scan neckline params ---
    far_scan_max_frac: float = 0.5,
    far_scan_min_frac: float = 0.05,
    far_scan_levels: int = 160,
    far_scan_length_percentile: float = 95.0,
    far_scan_w_len: float = 1.0,
    far_scan_w_comp: float = 0.35,
    # optional use of neck support during far-scan (default off)
    far_scan_w_neck: float = 0.0,
    far_scan_neck_score_thresh: float = 0.4,
    # --- safety ---
    no_shrink_margin: float = 1.01,
    debug: bool = False,
):
    """
    Segment crypts for a single organoid, and (re)normalize each crypt axis such that
    the detected neckline sits at normalized distance s=1.

    This version uses a *far-scan neckline* strategy intended to be robust to large/irregular
    crypts and buds:
      1) compute a bottom-to-surface distance field for each crypt (geodesics)
      2) compute a robust organoid length proxy (percentile of distances)
      3) scan up to `far_scan_max_frac * L_org` for candidate necklines
      4) choose a neckline via circumference minima + component penalty
      5) rescale distance field so neckline maps to s=1
      6) assign cells to crypts by nearest feature within s<=1

    Important safety rule (prevents rare "shrinking crypts")
    --------------------------------------------------------
    After selecting a neckline distance t*, we clamp:
        t* >= no_shrink_margin * max_distance_of_baseline_crypt_cells
    where the baseline crypt cells are those produced by the vocab seeding (+ optional growth).

    Inputs
    ------
    G : networkx.Graph
        Cell adjacency graph with nodes 0..N-1. Required node attributes:
          - "proj_vertex" : int mesh vertex id for each cell
          - "vocab_encoding" : 1D array for seeding (used by seed_features_by_vocab)
    mesh : OrganoidMesh-like
        Must provide mesh.v (V,3) vertices and mesh.f (F,3) faces.
    bin_centers : (B,) array_like
        Normalized axis points at which circumference curves are returned/aligned.
        Example: np.linspace(0, 2, 201)
    crypt_vocab_idx : iterable[int]
        Indices into vocab_encoding indicating crypt-like bins.
    neck_vocab_idx : iterable[int] or None
        Indices indicating neck-like bins. Used only for seeding neck regions and (optionally)
        to build a weak neck_score_v for far-scan. If None, neck seeding is disabled.

    Keyword Parameters
    ------------------
    crypt_seed_thresh, neck_seed_thresh : float
        Thresholds on max(vocab_encoding[idx]) to seed regions.
    min_crypt_seed_size : int
        Minimum connected component size to keep crypt seeds (and neck seeds if enabled).
    grow_steps : int
        Steps for grow_crypts_toward_necks (0 disables).
    geodesic_fn, geodesic_kwargs
        Geodesic backend used by calc_crypt_axis.

    far_scan_* : parameters controlling neckline scan
        - far_scan_max_frac : scan upper bound as fraction of L_org
        - far_scan_min_frac : scan lower bound as fraction of L_org
        - far_scan_levels : number of iso-level samples in scan
        - far_scan_length_percentile : percentile for L_org
        - far_scan_w_len : weight for circumference term
        - far_scan_w_comp : weight for component penalty
        - far_scan_w_neck : optional weight for neck support (default 0)
        - far_scan_neck_score_thresh : threshold for support_frac_above

    no_shrink_margin : float
        Multiplier for baseline crypt extent clamp (e.g. 1.01).
    debug : bool
        If True, also return a dbg dict with intermediate outputs.

    Returns
    -------
    crypts_final : list[set[int]]
        Final crypt patches (cell ids).
    villi_final : list[set[int]]
        Villus cells as a single patch (all cells not assigned to any crypt).
    d_norm_v : (K, V) float
        Rescaled normalized distance fields at mesh vertices, where neckline is at s=1.
        K is the number of crypts (features).
    L_scale : (K,) float
        Physical scale used for each crypt after rescaling (the selected neckline distance t*).
    Circ : (K, B) float
        Circumference curves evaluated at physical levels = bin_centers * L_scale[k].
    dbg : dict
        Only returned if debug=True.

    Notes
    -----
    - This function intentionally does *not* include the optional “plateau adjustment” stage.
      That can be applied later as a refinement pass if desired.
    """
    if geodesic_kwargs is None:
        geodesic_kwargs = {}

    # --- 1) seed regions from vocab encoding ---
    crypts_seed, necks_seed, villi_seed = seed_features_by_vocab(
        G,
        crypt_vocab_idx=crypt_vocab_idx,
        crypt_thresh=crypt_seed_thresh,
        min_crypt_region_size=min_crypt_seed_size,
        neck_vocab_idx=neck_vocab_idx,
        neck_thresh=neck_seed_thresh,
    )

    # --- 2) optionally grow crypts toward necks (keeps your existing behavior) ---
    if grow_steps and grow_steps > 0:
        crypts_grow, necks_grow, villi_grow = grow_crypts_toward_necks(
            G,
            crypt_regions=crypts_seed,
            neck_regions=necks_seed,
            N=grow_steps,
            min_villus_region_size=1,
        )
    else:
        crypts_grow, necks_grow, villi_grow = crypts_seed, necks_seed, villi_seed

    boundary_patches = villi_grow + necks_grow

    # --- 3) compute bottom distance fields (normalized by mean boundary distance) ---
    d_norm0_v, L0, bottom_vertex_ids = calc_crypt_axis(
        G=G,
        mesh=mesh,
        crypt_patches=crypts_grow,
        boundary_patches=boundary_patches,
        geodesic_fn=geodesic_fn,
        geodesic_kwargs=geodesic_kwargs,
    )
    # d_norm0_v: (K,V) and L0: (K,)
    K, V = d_norm0_v.shape

    # --- per-cell -> vertex projection (used for final assignment and no-shrink clamp) ---
    proj_vertex_ids = graph_get(G, "proj_vertex", dtype=np.int32)  # (N_cells,)

    # --- optional neck_score on vertices (default only used if far_scan_w_neck>0) ---
    neck_score_v = None
    if neck_vocab_idx is not None and far_scan_w_neck != 0.0:
        nodes = sorted(G.nodes())
        enc = np.stack([np.asarray(G.nodes[n]["vocab_encoding"], float) for n in nodes], axis=0)
        neck_idx = np.asarray(list(neck_vocab_idx), dtype=int)
        neck_score_c = enc[:, neck_idx].max(axis=1)  # (N_cells,)

        neck_score_v = np.zeros(V, dtype=float)
        np.maximum.at(neck_score_v, proj_vertex_ids, neck_score_c)

    # --- 4) far-scan neckline per crypt, then rescale so neckline is at s=1 ---
    bin_centers = np.asarray(bin_centers, dtype=float)
    Circ = np.full((K, bin_centers.size), np.nan, dtype=float)
    d_norm_v = np.full_like(d_norm0_v, np.nan, dtype=float)
    L_scale = np.full(K, np.nan, dtype=float)  # this will be the physical neckline distance t*
    far_scan_info = [None] * K
    base_max_dist_arr = np.full(K, np.nan, dtype=float)

    for k in range(K):
        if not np.isfinite(L0[k]) or L0[k] <= 0:
            continue

        # physical distance field from crypt bottom
        dist_phys = d_norm0_v[k] * float(L0[k])  # (V,)

        # pick neckline by far scan
        t_star, info = pick_neckline_far_scan(
            mesh=mesh,
            crypt_dist_v=dist_phys,
            max_frac=far_scan_max_frac,
            min_frac=far_scan_min_frac,
            n_levels=far_scan_levels,
            length_percentile=far_scan_length_percentile,
            neck_score_v=neck_score_v,
            neck_score_thresh=far_scan_neck_score_thresh,
            w_len=far_scan_w_len,
            w_comp=far_scan_w_comp,
            w_neck=far_scan_w_neck,
        )
        far_scan_info[k] = info

        # --- no-shrink clamp: new neckline must not be closer than current baseline crypt extent ---
        base_cells = list(crypts_grow[k]) if k < len(crypts_grow) else []
        base_max_dist = 0.0
        if len(base_cells) > 0:
            base_verts = proj_vertex_ids[base_cells]
            base_max_dist = float(np.nanmax(dist_phys[base_verts]))
        base_max_dist_arr[k] = base_max_dist

        t_star = max(float(t_star), float(no_shrink_margin) * float(base_max_dist), 1e-12)

        # store scale and rescaled distances
        L_scale[k] = float(t_star)
        d_norm_v[k] = dist_phys / float(t_star)

        # aligned circumference curve at bin_centers (physical levels = bin_centers * t_star)
        levels_phys = bin_centers * float(t_star)
        Circ[k] = crypt_circumference(mesh, dist_phys, levels_phys)

    # --- 5) map vertex distances to cell distances ---
    d_norm_c = d_norm_v[:, proj_vertex_ids]  # (K, N_cells)

    # --- 6) finalize crypt membership by neckline (s_thresh=1.0 inside assign_features_by_distance) ---
    crypts_final, best_feature, best_dist = assign_features_by_distance(d_norm_c)

    # --- 7) villus = all cells not in any crypt (single patch) ---
    N_cells = G.number_of_nodes()
    crypt_union = set().union(*crypts_final) if crypts_final else set()
    villus_cells = set(range(N_cells)) - crypt_union
    villi_final = [villus_cells] if villus_cells else []

    if not debug:
        return crypts_final, villi_final, d_norm_v, L_scale, Circ

    dbg = {
        "crypts_seed": crypts_seed,
        "necks_seed": necks_seed,
        "villi_seed": villi_seed,
        "crypts_grow": crypts_grow,
        "necks_grow": necks_grow,
        "villi_grow": villi_grow,
        "boundary_patches": boundary_patches,
        "bottom_vertex_ids": bottom_vertex_ids,
        "L0_boundarymean": L0,                 # original boundary-mean scale from calc_crypt_axis
        "L_scale_neckline": L_scale,           # final scale (physical neckline distance)
        "base_max_dist": base_max_dist_arr,    # baseline extent used for no-shrink clamp
        "far_scan_info": far_scan_info,        # per-crypt scan curves/metrics
        "best_feature": best_feature,          # (N_cells,)
        "best_dist": best_dist,                # (N_cells,)
    }

    return crypts_final, villi_final, d_norm_v, L_scale, Circ, dbg



def segment_crypts_organoid_legacy(
    G,                     # networkx cell-graph
    mesh,                  # OrganoidMesh 
    bin_centers,           # (B,) axis used for circumference curves + rescaling
    crypt_vocab_idx,       # iterable[int] indices into vocab_encoding indicating "crypt" vocab bins
    neck_vocab_idx=None,   # iterable[int] or None; indices indicating "neck" vocab bins
    crypt_seed_thresh=0.2, # threshold on max(vocab_encoding[crypt_vocab_idx]) to seed crypt
    neck_seed_thresh=0.5,  # threshold on max(vocab_encoding[neck_vocab_idx]) to seed neck
    min_crypt_seed_size=10,# minimum connected-component size for crypt (and neck if enabled)
    grow_steps=2,          # iterations for grow_crypts_toward_necks; set 0 to disable
    geodesic_fn=compute_geodesics_dijkstra,  # geodesics(mesh, sources=[...], **geodesic_kwargs)
    geodesic_kwargs=None,  # dict of kwargs forwarded to geodesic_fn
    search_interval=(0.75, 1.25),  # allowed stretch range for finding min circumference
    window_length=9,       # smoothing window for adjust_cryptlength_by_circumference
    polyorder=3,           # polynomial order for adjust_cryptlength_by_circumference smoothing
    min_prominence=0.05,   # peak prominence threshold for adjust_cryptlength_by_circumference
    debug=False,           # if True, also return intermediate regions + internals
):
    """
    One-stop organoid segmentation + crypt-axis computation.

    Pipeline
    --------
    1) seed crypt/neck/villus regions from vocab_encoding
    2) optionally grow crypts toward necks
    3) compute crypt axis (raw + normalized) using geodesics from crypt bottoms
    4) rescale axis by circumference so minimum aligns with s=1 on bin_centers
    5) map vertex distances to cell-center distances
    6) finalize crypt membership by neckline (s_thresh=1.0)
    7) villus_final = all cells not assigned to any crypt

    Returns
    -------
    crypts_final : list[set[int]]
    villi_final  : list[set[int]]          # single patch: all non-crypt cells
    d_norm       : (K, V) float            # rescaled normalized distances at vertices
    L_crypt      : (K,) float              # mean boundary distance (rescaled consistently with dnorm_v)
    C            : (K, B) float            # circumference curves aligned to bin_centers
    dbg          : dict (only if debug=True)
    """
    if geodesic_kwargs is None:
        geodesic_kwargs = {}

    # --- 1) seed regions ---
    crypts_seed, necks_seed, villi_seed = seed_features_by_vocab(
        G,
        crypt_vocab_idx=crypt_vocab_idx,
        crypt_thresh=crypt_seed_thresh,
        min_crypt_region_size=min_crypt_seed_size,
        neck_vocab_idx=neck_vocab_idx,
        neck_thresh=neck_seed_thresh,
    )

    # --- 2) optionally grow/refine ---
    if grow_steps and grow_steps > 0:
        crypts_grow, necks_grow, villi_grow = grow_crypts_toward_necks(
            G,
            crypt_regions=crypts_seed,
            neck_regions=necks_seed,
            N=grow_steps,
            min_villus_region_size=1,  # fixed default
        )
    else:
        crypts_grow, necks_grow, villi_grow = crypts_seed, necks_seed, villi_seed

    boundary_patches = villi_grow + necks_grow

    # --- 3) compute crypt axis (raw + normalized) ---
    d_norm, L_crypt, bottom_vertex_ids = calc_crypt_axis(
        G=G,
        mesh=mesh,
        crypt_patches=crypts_grow,
        boundary_patches=boundary_patches,
        geodesic_fn=geodesic_fn,
        geodesic_kwargs=geodesic_kwargs,
    )

    # --- 4) rescale by circumference (always returns (K,B), (K,V), (K,)) ---
    Circ, d_norm, L_crypt = normalize_crypt_axis_to_neckline(
        mesh=mesh,
        dnorm_vertices=d_norm,
        bin_centers=bin_centers,
        search_interval=search_interval,
        L_crypt=L_crypt,
        window_length=window_length,
        polyorder=polyorder,
        min_prominence=min_prominence,
    )

    # --- 5) vertex -> cell center distances ---
    proj_vertex_ids = graph_get(G, "proj_vertex", dtype=np.int32)
    dnorm_c = d_norm[:, proj_vertex_ids]  # (K, N_cells)

    # --- 6) finalize crypt membership by neckline (default s_thresh=1.0) ---
    crypts_final, best_feature, best_dist = assign_features_by_distance(dnorm_c)

    # --- 7) villus = all cells not in any crypt (single patch) ---
    N_cells = G.number_of_nodes()
    crypt_union = set().union(*crypts_final) if crypts_final else set()
    villus_cells = set(range(N_cells)) - crypt_union
    villi_final = [villus_cells] if villus_cells else []

    if not debug:
        return crypts_final, villi_final, d_norm, L_crypt, Circ

    dbg = {
        "crypts_seed": crypts_seed,
        "necks_seed": necks_seed,
        "villi_seed": villi_seed,
        "crypts_grow": crypts_grow,
        "necks_grow": necks_grow,
        "villi_grow": villi_grow,
        "boundary_patches": boundary_patches,
        "bottom_vertex_ids": bottom_vertex_ids,
        "best_feature": best_feature,  # (N_cells,)
        "best_dist": best_dist,        # (N_cells,)
    }

    return crypts_final, villi_final, d_norm, L_crypt, Circ, dbg
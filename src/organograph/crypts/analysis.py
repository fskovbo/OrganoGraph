import numpy as np


# ============================================================
# Circumference from iso-contours
# ============================================================

def crypt_circumference(
    mesh,           # mesh with v (V,3), f (F,3)
    crypt_dist,     # (V,) distance to crypt bottom for each vertex
    levels,         # (K,) array, iso-contour levels
):
    """
    Compute iso-contour lengths (circumference) of a scalar field on the mesh.

    Output
    ------
    C : (K,) array
    """
    v = np.asarray(mesh.v, dtype=float)
    f = np.asarray(mesh.f, dtype=np.int64)
    s = np.asarray(crypt_dist, dtype=float)
    levels = np.asarray(levels, dtype=float)

    tri_pos = v[f]
    tri_s = s[f]
    tri_min = np.min(tri_s, axis=1)
    tri_max = np.max(tri_s, axis=1)

    C = np.zeros(len(levels), dtype=float)

    def tri_isoseg(p, sval, t):
        inter = []
        a, b, c = sval

        if (a < t and b > t) or (a > t and b < t):
            w = (t - a) / (b - a)
            inter.append(p[0] + w * (p[1] - p[0]))
        if (b < t and c > t) or (b > t and c < t):
            w = (t - b) / (c - b)
            inter.append(p[1] + w * (p[2] - p[1]))
        if (c < t and a > t) or (c > t and a < t):
            w = (t - c) / (a - c)
            inter.append(p[2] + w * (p[0] - p[2]))

        return inter if len(inter) == 2 else None

    for li, t in enumerate(levels):
        active = (tri_min <= t) & (tri_max >= t)
        total = 0.0
        for p, sval in zip(tri_pos[active], tri_s[active]):
            seg = tri_isoseg(p, sval, t)
            if seg is not None:
                total += np.linalg.norm(seg[1] - seg[0])
        C[li] = total

    return C


# ============================================================
# Marker binning
# ============================================================

def bin_marker_positivity(
    markers,   # (N_cells, M) array, marker positivity per cell
    distance,  # (N_cells,) array, distance coordinate per cell
    bin_edges, # (B+1,) array, bin edges
):
    """
    Bin marker positivity vs distance.

    Output
    ------
    counts_pos : (M, B) array
    counts_total : (B,) array
    """
    markers = np.asarray(markers)
    distance = np.asarray(distance)
    bin_edges = np.asarray(bin_edges)

    N, M = markers.shape
    B = len(bin_edges) - 1
    bin_ids = np.digitize(distance, bin_edges) - 1

    counts_pos = np.zeros((M, B), dtype=int)
    counts_total = np.zeros(B, dtype=int)

    for b in range(B):
        mask = (bin_ids == b)
        if np.any(mask):
            counts_total[b] = mask.sum()
            counts_pos[:, b] = markers[mask].astype(bool).sum(axis=0)

    return counts_pos, counts_total


def field_stats_along_crypt(
    mesh_field,        # (V,) field on mesh vertices
    s_vertices,        # (V,) normalized distance per vertex (e.g., dnorm_vertices)
    bin_edges,         # (B+1,) shared bin edges
    weights=None,      # (V,) optional per-vertex weights (e.g., vertex areas)
):
    """
    Compute mean and std of a vertex-defined field as a function of s using shared bins.

    For each bin b spanning (bin_edges[b], bin_edges[b+1]]:
      - collect vertices whose s falls in that bin
      - compute (weighted) mean and std of mesh_field over those vertices

    Parameters
    ----------
    mesh_field : array-like, shape (V,)
        Scalar field defined on mesh vertices.
    s_vertices : array-like, shape (V,)
        Normalized distances for vertices (same length as mesh_field).
    bin_edges : array-like, shape (B+1,)
        Bin edges along s (shared across crypts/meshes).
    weights : array-like, shape (V,), optional
        Nonnegative weights per vertex. If provided, computes weighted mean/std.

    Returns
    -------
    mean : (B,) float
    std : (B,) float
    count : (B,) int
        Number of vertices used per bin (after finite-value masking).
    """
    x = np.asarray(mesh_field, dtype=float)
    s = np.asarray(s_vertices, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)

    if x.shape != s.shape:
        raise ValueError("mesh_field and s_vertices must have the same shape.")
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must be 1D with length >= 2.")

    B = edges.size - 1
    mean = np.full(B, np.nan, dtype=float)
    std  = np.full(B, np.nan, dtype=float)
    count = np.zeros(B, dtype=int)

    if weights is None:
        w = None
        m = np.isfinite(x) & np.isfinite(s)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != x.shape:
            raise ValueError("weights must have the same shape as mesh_field.")
        m = np.isfinite(x) & np.isfinite(s) & np.isfinite(w) & (w >= 0)

    if not np.any(m):
        return mean, std, count

    x = x[m]
    s = s[m]
    if w is not None:
        w = w[m]

    # Bin assignment: b in [0, B-1]
    # Use right-open bins [edge_i, edge_{i+1}) except include the last edge in last bin.
    b = np.searchsorted(edges, s, side="right") - 1
    valid = (b >= 0) & (b < B)
    x = x[valid]
    b = b[valid]
    if w is not None:
        w = w[valid]

    # Counts
    count = np.bincount(b, minlength=B).astype(int)

    if w is None:
        # sums and sums of squares per bin
        sx = np.bincount(b, weights=x, minlength=B)
        sx2 = np.bincount(b, weights=x*x, minlength=B)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sx / count
            var = sx2 / count - mean*mean
            var = np.maximum(var, 0.0)
            std = np.sqrt(var)
        mean[count == 0] = np.nan
        std[count == 0] = np.nan
        return mean, std, count

    # Weighted
    sw = np.bincount(b, weights=w, minlength=B)
    sxw = np.bincount(b, weights=w*x, minlength=B)
    sx2w = np.bincount(b, weights=w*x*x, minlength=B)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sxw / sw
        var = sx2w / sw - mean*mean
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)

    mean[sw == 0] = np.nan
    std[sw == 0] = np.nan
    return mean, std, count


def budding_index(
    s,        # (B,) bin centers (or s-sampling)
    C,        # (B,) circumference profile C(s)
    s_max=1.0,
    eps=1e-12,
):
    """
    Compute budding index (BI) for a single crypt:

        BI = 1 - C(s=1) / max_{s<=1} C(s)

    Returns
    -------
    BI : float in [0,1]

    Notes
    -----
    - Uses only bins with s <= s_max and finite C.
    - If s does not contain exactly 1.0, uses the last valid value
      (same behavior as your current code).
    - If profile is invalid, returns 0.
    """
    s = np.asarray(s, dtype=float)
    C = np.asarray(C, dtype=float)

    if s.ndim != 1 or C.ndim != 1 or len(s) != len(C):
        raise ValueError("s and C must be 1D arrays of equal length.")

    m = (s <= float(s_max)) & np.isfinite(C)
    if np.sum(m) < 2:
        return 0.0

    sC = s[m]
    CC = C[m]

    # circumference at s≈1 (use last valid bin ≤ s_max)
    C1 = CC[-1]
    Cmax = np.nanmax(CC)

    if not (np.isfinite(C1) and np.isfinite(Cmax) and Cmax > eps):
        return 0.0

    BI = 1.0 - float(C1) / float(Cmax)
    return float(np.clip(BI, 0.0, 1.0))
import numpy as np
from organograph.mesh.hks import compute_hks



def compute_gaussian_curvature(
    mesh,
    tau_ref=None,                # dimensionless reference times used for HKS evaluation
    early_min_points=3,          # minimum number of points in the early linear fit
    early_r2_drop=0.02,          # maximum allowed drop in R^2 when extending the early fit
    early_min_r2=0.985,          # minimum acceptable R^2 for the early fit
    late_t_max=10.0,             # only times t < late_t_max are considered for the curvature fit
    late_min_points=3,           # minimum number of points in the late linear fit
    late_r2_drop=0.02,           # maximum allowed drop in R^2 when extending the late fit to shorter times
    late_min_r2=0.985,           # minimum acceptable R^2 for the late fit
    defect_zmax=5.0,             # threshold on early-slope z-score for marking defects
    positive_defects_only=True,  # if True, only positive-curvature outliers are marked as defects
    defect_dilation_steps=2,     # number of 1-ring dilation steps used to expand the defect mask
    inpaint_iters=30,            # number of neighbor-averaging iterations used to fill masked regions
    return_debug=False,          # if True, also return a dict with fit/debug quantities
):
    """
    Estimate Gaussian curvature from HKS using two automatically selected
    linear regimes in

        y(t) = 4 * pi * t * HKS(x, t) - 1.

    The method works in three stages:
      1. Fit an early linear regime at short times and use its slope to detect
         defect-like regions.
      2. Fit a later linear regime before large-time deviations set in and use
         its slope to estimate Gaussian curvature via K = 3 * slope.
      3. Mask detected defects and inpaint them from neighboring vertices.

    Returns
    -------
    curvature : (V,) ndarray
        Final curvature field after defect masking and inpainting.
    debug : dict, optional
        Returned only if `return_debug=True`. Contains quantities useful for
        inspecting the fit and defect detection.
    """

    if tau_ref is None:
        tau_ref = np.geomspace(5e-5, 3e-2, 24)

    V = len(mesh.v)
    ts_mesh, L_mesh = _rescale_times_from_tau(mesh, tau_ref)
    ts_mesh = np.asarray(ts_mesh, float)

    hks_full = np.asarray(compute_hks(mesh, ts_mesh, coeffs=False), float)
    y_full = 4.0 * np.pi * hks_full * ts_mesh[None, :] - 1.0

    # Early regime: prefix fits
    early_end_idx, early_slopes_all, early_intercepts_all, early_r2_all = _choose_early_window(
        ts_mesh,
        y_full,
        min_points=early_min_points,
        r2_drop=early_r2_drop,
        min_r2=early_min_r2,
    )

    early_slope = _gather_by_index(early_slopes_all, early_end_idx)
    early_intercept = _gather_by_index(early_intercepts_all, early_end_idx)
    early_r2 = _gather_by_index(early_r2_all, early_end_idx)

    # Late regime: suffix fits on times below late_t_max
    late_start_idx, late_slopes_all_sub, late_intercepts_all_sub, late_r2_all_sub, late_valid_idx = _choose_late_window(
        ts_mesh,
        y_full,
        t_max=late_t_max,
        min_points=late_min_points,
        r2_drop=late_r2_drop,
        min_r2=late_min_r2,
    )

    late_local_map = {g: j for j, g in enumerate(late_valid_idx)}
    late_start_idx_local = np.array([late_local_map[g] for g in late_start_idx], dtype=np.int64)

    late_slope = _gather_by_index(late_slopes_all_sub, late_start_idx_local)
    late_intercept = _gather_by_index(late_intercepts_all_sub, late_start_idx_local)
    late_r2 = _gather_by_index(late_r2_all_sub, late_start_idx_local)

    # Raw curvature from the late slope
    curvature_raw = 3.0 * late_slope

    # Defect detection from the early slope
    defect_score = _spatial_zscore(early_slope)
    defect_mask = defect_score > float(defect_zmax)

    if positive_defects_only:
        defect_mask &= np.isfinite(curvature_raw) & (curvature_raw > 0)

    neighbors = _mesh_neighbors_from_faces(mesh)

    if defect_dilation_steps > 0:
        defect_mask = _dilate_vertex_mask(defect_mask, neighbors, n_steps=defect_dilation_steps)

    # Masked curvature
    curvature_masked = curvature_raw.copy()
    curvature_masked[defect_mask] = np.nan

    # Final curvature after inpainting
    curvature = _nanmean_neighbors(curvature_masked, neighbors, n_iter=inpaint_iters)

    if not return_debug:
        return curvature

    debug = {
        "ts_mesh": ts_mesh,
        "L_mesh": L_mesh,
        "hks": hks_full,
        "fit_signal": y_full,
        "early_end_idx": early_end_idx,
        "late_start_idx": late_start_idx,
        "early_slope": early_slope,
        "early_intercept": early_intercept,
        "early_r2": early_r2,
        "late_slope": late_slope,
        "late_intercept": late_intercept,
        "late_r2": late_r2,
        "defect_score": defect_score,
        "defect_mask": defect_mask,
        "curvature_raw": curvature_raw,
        "curvature_masked": curvature_masked,
    }

    return curvature, debug


# =====================================================================
# Mesh helpers
# =====================================================================


def _mesh_neighbors_from_faces(mesh):
    """
    Build 1-ring vertex neighbors from triangular faces.
    """
    V = len(mesh.v)
    neigh = [set() for _ in range(V)]
    faces = np.asarray(mesh.f, dtype=np.int64)

    for tri in faces:
        i, j, k = tri
        neigh[i].update([j, k])
        neigh[j].update([i, k])
        neigh[k].update([i, j])

    return [np.fromiter(s, dtype=np.int64) for s in neigh]


def _compute_length_scale(mesh):
    """
    Characteristic organoid size based on total surface area.
    """
    vertex_areas = np.asarray(mesh.vertex_areas(), float)
    return float(np.sqrt(np.sum(vertex_areas)))


def _rescale_times_from_tau(mesh, tau_ref):
    """
    Convert dimensionless reference times tau_ref into mesh times.
    """
    tau_ref = np.asarray(tau_ref, float)
    L_mesh = _compute_length_scale(mesh)
    ts_mesh = tau_ref * (L_mesh ** 2)
    return ts_mesh, L_mesh


def _spatial_zscore(x):
    """
    Robust spatial z-score using median / MAD.
    """
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    denom = 1.4826 * mad
    if denom <= 0:
        denom = np.nanstd(x)
    if denom <= 0:
        denom = 1.0
    return (x - med) / denom


def _nanmean_neighbors(values, neighbors, n_iter=20):
    """
    Inpaint NaNs by iterative neighbor averaging.
    """
    out = np.asarray(values, float).copy()
    valid = np.isfinite(out)

    for _ in range(n_iter):
        changed = False
        out_new = out.copy()

        for i in range(len(out)):
            if valid[i]:
                continue

            nb = neighbors[i]
            if nb.size == 0:
                continue

            vals = out[nb]
            vals = vals[np.isfinite(vals)]

            if vals.size > 0:
                out_new[i] = np.mean(vals)
                changed = True

        out = out_new
        valid = np.isfinite(out)

        if not changed:
            break

    return out


def _dilate_vertex_mask(mask, neighbors, n_steps=1):
    """
    Expand a boolean vertex mask by repeated 1-ring dilation.
    """
    mask = np.asarray(mask, dtype=bool).copy()

    for _ in range(int(n_steps)):
        new_mask = mask.copy()
        idx = np.flatnonzero(mask)
        for i in idx:
            nb = neighbors[i]
            if nb.size > 0:
                new_mask[nb] = True
        mask = new_mask

    return mask


# =====================================================================
# Fitting helpers
# =====================================================================


def _fit_line_prefixes(t, Y):
    """
    Vectorized linear fits for prefixes of the time axis.

    Parameters
    ----------
    t : (T,) array
    Y : (V, T) array

    Returns
    -------
    slopes, intercepts, r2, sse : each of shape (V, T)
        Entry [:, j] uses points t[0:j+1].
        Entries for j < 1 are not meaningful.
    """
    t = np.asarray(t, float)
    Y = np.asarray(Y, float)

    V, T = Y.shape
    n = np.arange(1, T + 1, dtype=float)

    c_t = np.cumsum(t)
    c_t2 = np.cumsum(t**2)

    c_y = np.cumsum(Y, axis=1)
    c_y2 = np.cumsum(Y**2, axis=1)
    c_ty = np.cumsum(Y * t[None, :], axis=1)

    den = n * c_t2 - c_t**2
    den = np.where(np.abs(den) < 1e-15, np.nan, den)

    slopes = (n[None, :] * c_ty - c_t[None, :] * c_y) / den[None, :]
    intercepts = (c_y - slopes * c_t[None, :]) / n[None, :]

    sse = (
        c_y2
        + slopes**2 * c_t2[None, :]
        + n[None, :] * intercepts**2
        + 2.0 * slopes * intercepts * c_t[None, :]
        - 2.0 * slopes * c_ty
        - 2.0 * intercepts * c_y
    )

    mean_y = c_y / n[None, :]
    sst = c_y2 - n[None, :] * mean_y**2
    r2 = 1.0 - sse / np.where(sst > 1e-15, sst, np.nan)
    r2 = np.where(sst <= 1e-15, 1.0, r2)

    return slopes, intercepts, r2, sse


def _fit_line_suffixes(t, Y):
    """
    Vectorized linear fits for suffixes of the time axis.

    Entry [:, j] uses points t[j:].
    """
    t = np.asarray(t, float)
    Y = np.asarray(Y, float)

    t_rev = t[::-1]
    Y_rev = Y[:, ::-1]

    slopes_r, intercepts_r, r2_r, sse_r = _fit_line_prefixes(t_rev, Y_rev)

    slopes = slopes_r[:, ::-1]
    intercepts = intercepts_r[:, ::-1]
    r2 = r2_r[:, ::-1]
    sse = sse_r[:, ::-1]

    return slopes, intercepts, r2, sse


def _choose_early_window(t, Y, min_points=3, r2_drop=0.02, min_r2=0.985):
    """
    Start from the first `min_points` indices and keep extending until the fit
    becomes noticeably worse.

    Returns
    -------
    end_idx : (V,) integer array
        Inclusive end index of the chosen early window.
    slopes, intercepts, r2 : full prefix fit tables
    """
    slopes, intercepts, r2, sse = _fit_line_prefixes(t, Y)
    V, T = Y.shape

    end_idx = np.full(V, min_points - 1, dtype=np.int64)

    for j in range(min_points, T):
        prev = r2[:, j - 1]
        cur = r2[:, j]

        good = np.isfinite(cur) & (cur >= min_r2) & ((prev - cur) <= r2_drop)
        end_idx[good] = j

    return end_idx, slopes, intercepts, r2


def _choose_late_window(
    t,
    Y,
    t_max=10.0,
    min_points=3,
    r2_drop=0.02,
    min_r2=0.985,
):
    """
    Choose the curvature window by:
      1. restricting to t < t_max
      2. starting from the first `min_points` points in that restricted set
      3. adding progressively shorter-time points until fit gets worse

    This is implemented as a suffix fit on the restricted time grid.

    Returns
    -------
    start_idx_global : (V,) integer array
        Start index in the full time array for the chosen late window.
    slopes_local, intercepts_local, r2_local : suffix fit tables on restricted grid
    valid_idx : indices of timepoints used in the late-window search
    """
    valid_idx = np.flatnonzero(t < t_max)
    if valid_idx.size < min_points:
        raise ValueError("Not enough timepoints with t < t_max for late fit")

    t_sub = t[valid_idx]
    Y_sub = Y[:, valid_idx]

    slopes_suf, intercepts_suf, r2_suf, sse_suf = _fit_line_suffixes(t_sub, Y_sub)

    V, Tsub = Y_sub.shape
    max_start = Tsub - min_points
    start_idx_local = np.full(V, max_start, dtype=np.int64)

    for j in range(max_start - 1, -1, -1):
        prev = r2_suf[:, j + 1]
        cur = r2_suf[:, j]

        good = np.isfinite(cur) & (cur >= min_r2) & ((prev - cur) <= r2_drop)
        start_idx_local[good] = j

    start_idx_global = valid_idx[start_idx_local]

    return start_idx_global, slopes_suf, intercepts_suf, r2_suf, valid_idx


def _gather_by_index(arr2d, idx):
    """
    Gather arr2d[i, idx[i]] for each row i.
    """
    idx = np.asarray(idx, dtype=np.int64)
    rows = np.arange(arr2d.shape[0], dtype=np.int64)
    return arr2d[rows, idx]


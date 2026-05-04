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



def integrate_gaussian_curvature(
    mesh,
    curvature,
    mask=None,
    use_vertex_areas=True,
    from_mass_matrix=True,
    return_details=False,
):
    """
    Integrate a vertex-wise Gaussian curvature field over the mesh surface.

    Parameters
    ----------
    mesh : OrganoidMesh-like object
        Must provide:
          - mesh.v
          - mesh.f
          - mesh.vertex_areas(...)
        and optionally mesh.face_areas() if use_vertex_areas=False.
    curvature : (V,) array_like
        Gaussian curvature values defined at vertices.
    mask : (V,) bool array_like, optional
        If given, only integrate over vertices where mask is True.
        Useful for checking the integral on the "good" region before inpainting.
    use_vertex_areas : bool, default=True
        If True, compute integral as sum_i K_i * A_i using per-vertex areas.
        This is the recommended choice for your curvature field.
        If False, curvature is first averaged to faces and integrated face-wise.
    from_mass_matrix : bool, default=True
        Passed to mesh.vertex_areas(). Set False if you want purely geometric
        barycentric areas and do not want to rely on a precomputed mass matrix.
    return_details : bool, default=False
        If True, also return a dict with useful diagnostics.

    Returns
    -------
    total_curvature : float
        Approximation to ∫ K dA.
    details : dict, optional
        Returned only if return_details=True.
    """
    K = np.asarray(curvature, dtype=float)
    V = len(mesh.v)

    if K.shape != (V,):
        raise ValueError(f"`curvature` must have shape ({V},), got {K.shape}")

    finite = np.isfinite(K)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (V,):
            raise ValueError(f"`mask` must have shape ({V},), got {mask.shape}")
        finite &= mask

    if use_vertex_areas:
        # Best choice for a vertex-defined curvature field
        A = np.asarray(mesh.vertex_areas(from_mass_matrix=from_mass_matrix), dtype=float)
        if A.shape != (V,):
            raise ValueError(f"vertex areas must have shape ({V},), got {A.shape}")

        total_curvature = float(np.sum(K[finite] * A[finite]))
        integrated_area = float(np.sum(A[finite]))

    else:
        # Optional face-based integration by averaging vertex curvature to faces
        F = np.asarray(mesh.f, dtype=np.int64)
        face_areas = np.asarray(mesh.face_areas(), dtype=float)

        K_face = np.mean(K[F], axis=1)
        face_valid = np.all(finite[F], axis=1)

        total_curvature = float(np.sum(K_face[face_valid] * face_areas[face_valid]))
        integrated_area = float(np.sum(face_areas[face_valid]))

    if not return_details:
        return total_curvature

    details = {
        "target_sphere": 4.0 * np.pi,
        "error_to_4pi": total_curvature - 4.0 * np.pi,
        "relative_error_to_4pi": (total_curvature - 4.0 * np.pi) / (4.0 * np.pi),
        "estimated_euler_characteristic": total_curvature / (2.0 * np.pi),
        "integrated_area": integrated_area,
        "n_vertices_total": V,
        "n_vertices_used": int(np.sum(finite)),
    }
    return total_curvature, details



import numpy as np
from organograph.mesh.hks import compute_hks


def compute_gaussian_curvature_fixed_time(
    mesh,
    c_h2=(4.0, 6.0, 8.0),        # times t = c * h^2; can be float or sequence
    tau_ref=None,                # optional extra HKS times; merged with c_h2 times
    defect_tau_ref=None,         # optional separate times for defect detection
    early_min_points=3,
    early_r2_drop=0.02,
    early_min_r2=0.985,
    defect_zmax=5.0,
    positive_defects_only=True,
    defect_dilation_steps=2,
    inpaint_iters=30,
    aggregate="median",          # how to combine multi-time estimates: "mean" or "median"
    return_debug=False,
):
    """
    Estimate Gaussian curvature from HKS using one or more fixed times scaled by
    mesh resolution, while retaining defect detection + inpainting.

    Parameters
    ----------
    mesh : mesh object
        Must provide:
          - mesh.v
          - mesh.f
    c_h2 : float or sequence of float, default=(4.0, 6.0, 8.0)
        Resolution-scaled times are chosen as t = c * h^2, where h is the mean
        unique edge length of the mesh. Using several values and aggregating
        tends to be more stable than using a single one.
    tau_ref : sequence of float, optional
        Additional dimensionless times, interpreted as t = tau_ref * L_mesh^2,
        where L_mesh is the characteristic size from _compute_length_scale(mesh).
        These are merged with the c_h2 times for HKS evaluation.
    defect_tau_ref : sequence of float, optional
        Separate times used only for defect detection. If None, defaults to the
        same early-time grid as in the original function.
    early_min_points, early_r2_drop, early_min_r2
        Same meaning as in compute_gaussian_curvature(); used for defect detection.
    defect_zmax : float
        Threshold on the robust z-score of the early slope for defect detection.
    positive_defects_only : bool
        If True, only positive-curvature outliers are marked as defects.
    defect_dilation_steps : int
        Number of 1-ring dilation steps on the defect mask.
    inpaint_iters : int
        Number of neighbor-averaging iterations for filling masked values.
    aggregate : {"mean", "median"}
        How to combine multiple fixed-time curvature estimates.
    return_debug : bool
        If True, also return debug information.

    Returns
    -------
    curvature : (V,) ndarray
        Final curvature field after defect masking and inpainting.
    debug : dict, optional
        Returned only if return_debug=True.
    """

    # ------------------------------------------------------------------
    # Helpers local to this estimator
    # ------------------------------------------------------------------

    def _mean_edge_length(mesh):
        v = np.asarray(mesh.v, dtype=float)
        f = np.asarray(mesh.f, dtype=np.int64)

        edges = np.vstack([
            f[:, [0, 1]],
            f[:, [1, 2]],
            f[:, [2, 0]],
        ])
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
        return float(np.mean(lengths))

    def _as_1d_array(x):
        if np.isscalar(x):
            return np.array([float(x)], dtype=float)
        return np.asarray(x, dtype=float).ravel()

    # ------------------------------------------------------------------
    # Build time grids
    # ------------------------------------------------------------------

    if defect_tau_ref is None:
        defect_tau_ref = np.geomspace(5e-5, 3e-2, 24)

    h_mean = _mean_edge_length(mesh)
    L_mesh = _compute_length_scale(mesh)

    c_h2 = _as_1d_array(c_h2)
    ts_fixed = c_h2 * (h_mean ** 2)

    ts_extra = np.array([], dtype=float)
    if tau_ref is not None:
        tau_ref = np.asarray(tau_ref, dtype=float)
        ts_extra = tau_ref * (L_mesh ** 2)

    ts_defect = np.asarray(defect_tau_ref, dtype=float) * (L_mesh ** 2)

    # merged HKS evaluation grid
    ts_all = np.unique(np.concatenate([ts_fixed, ts_extra, ts_defect]))
    ts_all = np.sort(ts_all.astype(float))

    # ------------------------------------------------------------------
    # Evaluate HKS and transformed signal
    # ------------------------------------------------------------------

    hks_all = np.asarray(compute_hks(mesh, ts_all, coeffs=False), float)
    y_all = 4.0 * np.pi * hks_all * ts_all[None, :] - 1.0

    # ------------------------------------------------------------------
    # Fixed-time curvature estimate
    # ------------------------------------------------------------------

    fixed_idx = np.array([np.where(np.isclose(ts_all, t))[0][0] for t in ts_fixed], dtype=np.int64)
    y_fixed = y_all[:, fixed_idx]
    t_fixed = ts_all[fixed_idx]

    # K_t(x) = 3/t * (4 pi t H - 1)
    K_candidates = 3.0 * y_fixed / t_fixed[None, :]

    if aggregate == "median":
        curvature_raw = np.nanmedian(K_candidates, axis=1)
    elif aggregate == "mean":
        curvature_raw = np.nanmean(K_candidates, axis=1)
    else:
        raise ValueError("aggregate must be 'mean' or 'median'")

    # ------------------------------------------------------------------
    # Defect detection from early slope, same spirit as original pipeline
    # ------------------------------------------------------------------

    defect_idx = np.array([np.where(np.isclose(ts_all, t))[0][0] for t in ts_defect], dtype=np.int64)
    ts_defect_eval = ts_all[defect_idx]
    y_defect = y_all[:, defect_idx]

    early_end_idx, early_slopes_all, early_intercepts_all, early_r2_all = _choose_early_window(
        ts_defect_eval,
        y_defect,
        min_points=early_min_points,
        r2_drop=early_r2_drop,
        min_r2=early_min_r2,
    )

    early_slope = _gather_by_index(early_slopes_all, early_end_idx)
    early_intercept = _gather_by_index(early_intercepts_all, early_end_idx)
    early_r2 = _gather_by_index(early_r2_all, early_end_idx)

    defect_score = _spatial_zscore(early_slope)
    defect_mask = defect_score > float(defect_zmax)

    if positive_defects_only:
        defect_mask &= np.isfinite(curvature_raw) & (curvature_raw > 0)

    neighbors = _mesh_neighbors_from_faces(mesh)

    if defect_dilation_steps > 0:
        defect_mask = _dilate_vertex_mask(defect_mask, neighbors, n_steps=defect_dilation_steps)

    curvature_masked = curvature_raw.copy()
    curvature_masked[defect_mask] = np.nan

    curvature = _nanmean_neighbors(curvature_masked, neighbors, n_iter=inpaint_iters)

    if not return_debug:
        return curvature

    debug = {
        "h_mean": h_mean,
        "L_mesh": L_mesh,
        "c_h2": c_h2,
        "ts_fixed": ts_fixed,
        "ts_defect": ts_defect_eval,
        "ts_all": ts_all,
        "hks": hks_all,
        "fit_signal": y_all,
        "fixed_idx": fixed_idx,
        "K_candidates": K_candidates,
        "aggregate": aggregate,
        "early_end_idx": early_end_idx,
        "early_slope": early_slope,
        "early_intercept": early_intercept,
        "early_r2": early_r2,
        "defect_score": defect_score,
        "defect_mask": defect_mask,
        "curvature_raw": curvature_raw,
        "curvature_masked": curvature_masked,
    }
    return curvature, debug



def compute_gaussian_curvature_adaptive(
    mesh,
    c_h2_candidates=None,
    threshold_factor=1.02,
    aggregate="median",
    defect_tau_ref=None,
    early_min_points=3,
    early_r2_drop=0.02,
    early_min_r2=0.985,
    defect_zmax=5.0,
    positive_defects_only=True,
    defect_dilation_steps=2,
    inpaint_iters=30,
    integration_kwargs=None,
    renormalize=True,          # <-- NEW
    return_debug=False,
):
    """
    Adaptive fixed-time Gaussian curvature estimator with optional
    Gauss–Bonnet normalization (∫K dA = 4π).

    See previous version for full description.
    """

    if c_h2_candidates is None:
        c_h2_candidates = [
            (5.0, 15.0, 25.0),
            (10.0, 20.0, 30.0),
            (15.0, 25.0, 35.0),
            (20.0, 30.0, 40.0),
            (25.0, 35.0, 45.0),
            (30.0, 40.0, 50.0),
            (35.0, 45.0, 55.0),
            (40.0, 50.0, 60.0),
        ]

    if integration_kwargs is None:
        integration_kwargs = {}

    target = threshold_factor * (4.0 * np.pi)

    tried = []
    selected_idx = None
    selected_curvature = None
    selected_curvature_debug = None
    selected_total = None
    selected_info = None

    # ------------------------------------------------------------
    # Sweep candidates
    # ------------------------------------------------------------
    for i, c_h2 in enumerate(c_h2_candidates):
        curv_out = compute_gaussian_curvature_fixed_time(
            mesh,
            c_h2=c_h2,
            defect_tau_ref=defect_tau_ref,
            early_min_points=early_min_points,
            early_r2_drop=early_r2_drop,
            early_min_r2=early_min_r2,
            defect_zmax=defect_zmax,
            positive_defects_only=positive_defects_only,
            defect_dilation_steps=defect_dilation_steps,
            inpaint_iters=inpaint_iters,
            aggregate=aggregate,
            return_debug=return_debug,
        )

        if return_debug:
            curvature_i, curvature_debug_i = curv_out
        else:
            curvature_i = curv_out
            curvature_debug_i = None

        total_i, info_i = integrate_gaussian_curvature(
            mesh,
            curvature_i,
            return_details=True,
            **integration_kwargs,
        )

        tried.append({
            "candidate_index": i,
            "c_h2": tuple(float(x) for x in c_h2),
            "total_curvature": float(total_i),
            "error_to_4pi": float(info_i["error_to_4pi"]),
            "relative_error_to_4pi": float(info_i["relative_error_to_4pi"]),
            "chi_est": float(info_i["estimated_euler_characteristic"],
            ),
            "selected": False,
        })

        if total_i >= target and selected_idx is None:
            selected_idx = i
            selected_curvature = curvature_i
            selected_curvature_debug = curvature_debug_i
            selected_total = total_i
            selected_info = info_i
            tried[-1]["selected"] = True
            break

    # ------------------------------------------------------------
    # Fallback if nothing crossed threshold
    # ------------------------------------------------------------
    if selected_idx is None:
        selected_idx = len(c_h2_candidates) - 1
        c_h2 = c_h2_candidates[selected_idx]

        curv_out = compute_gaussian_curvature_fixed_time(
            mesh,
            c_h2=c_h2,
            defect_tau_ref=defect_tau_ref,
            early_min_points=early_min_points,
            early_r2_drop=early_r2_drop,
            early_min_r2=early_min_r2,
            defect_zmax=defect_zmax,
            positive_defects_only=positive_defects_only,
            defect_dilation_steps=defect_dilation_steps,
            inpaint_iters=inpaint_iters,
            aggregate=aggregate,
            return_debug=return_debug,
        )

        if return_debug:
            selected_curvature, selected_curvature_debug = curv_out
        else:
            selected_curvature = curv_out
            selected_curvature_debug = None

        selected_total, selected_info = integrate_gaussian_curvature(
            mesh,
            selected_curvature,
            return_details=True,
            **integration_kwargs,
        )

        tried[-1]["selected"] = True

    # ------------------------------------------------------------
    # NEW: Gauss–Bonnet normalization
    # ------------------------------------------------------------
    if renormalize:
        if np.isfinite(selected_total) and selected_total != 0.0:
            scale = (4.0 * np.pi) / selected_total
        else:
            scale = 1.0

        curvature_final = selected_curvature * scale

        total_after, info_after = integrate_gaussian_curvature(
            mesh,
            curvature_final,
            return_details=True,
            **integration_kwargs,
        )
    else:
        scale = 1.0
        curvature_final = selected_curvature
        total_after = selected_total
        info_after = selected_info

    if not return_debug:
        return curvature_final

    debug = {
        "selection_target": target,
        "threshold_factor": threshold_factor,
        "selected_index": selected_idx,
        "selected_c_h2": tuple(float(x) for x in c_h2_candidates[selected_idx]),
        "selected_total_before": float(selected_total),
        "selected_total_after": float(total_after),
        "scale_factor": float(scale),
        "tried_candidates": tried,
        "curvature_debug": selected_curvature_debug,
        "renormalized": renormalize,
        "final_info": info_after,
    }

    return curvature_final, debug



def compute_gaussian_curvature_smoothed_hks(
    mesh,
    t_smooth=(0.75, 1.0, 1.25),
    smooth_aggregate="median",
    defect_tau_ref=None,
    early_min_points=3,
    early_r2_drop=0.02,
    early_min_r2=0.985,
    defect_zmax=5.0,
    positive_defects_only=True,
    defect_positive_gate="smooth",   # "smooth", "fit", or "none"
    fit_t_max=10.0,
    fit_min_points=3,
    fit_r2_drop=0.02,
    fit_min_r2=0.985,
    defect_dilation_steps=2,
    inpaint_iters=30,
    renormalize=True,
    integration_kwargs=None,
    return_debug=False,
):
    """
    Compute a smoothed Gaussian curvature field from HKS at one or more
    user-chosen smooth times, while retaining early-time defect detection
    and inpainting from the original pipeline.
    """

    if integration_kwargs is None:
        integration_kwargs = {}

    def _as_1d_array(x):
        if np.isscalar(x):
            return np.array([float(x)], dtype=float)
        return np.asarray(x, dtype=float).ravel()

    if defect_tau_ref is None:
        defect_tau_ref = np.geomspace(5e-5, 3e-2, 24)

    # ------------------------------------------------------------------
    # Build evaluation times
    # ------------------------------------------------------------------

    t_smooth = _as_1d_array(t_smooth)
    if np.any(t_smooth <= 0):
        raise ValueError("All entries in t_smooth must be > 0")

    ts_defect, L_mesh = _rescale_times_from_tau(mesh, defect_tau_ref)
    ts_defect = np.asarray(ts_defect, dtype=float)

    ts_all = np.unique(np.concatenate([t_smooth, ts_defect]))
    ts_all = np.sort(ts_all.astype(float))

    # ------------------------------------------------------------------
    # Evaluate HKS and transformed signal
    # ------------------------------------------------------------------

    hks_all = np.asarray(compute_hks(mesh, ts_all, coeffs=False), float)
    y_all = 4.0 * np.pi * hks_all * ts_all[None, :] - 1.0

    # ------------------------------------------------------------------
    # Final smoothed curvature from selected smooth times
    # ------------------------------------------------------------------

    smooth_idx = np.array(
        [np.where(np.isclose(ts_all, t))[0][0] for t in t_smooth],
        dtype=np.int64,
    )
    y_smooth = y_all[:, smooth_idx]
    t_s_eval = ts_all[smooth_idx]

    K_candidates = 3.0 * y_smooth / t_s_eval[None, :]

    if smooth_aggregate == "median":
        curvature_raw = np.nanmedian(K_candidates, axis=1)
    elif smooth_aggregate == "mean":
        curvature_raw = np.nanmean(K_candidates, axis=1)
    else:
        raise ValueError("smooth_aggregate must be 'mean' or 'median'")

    # ------------------------------------------------------------------
    # Early-time defect detection from linear fit
    # ------------------------------------------------------------------

    defect_idx = np.array(
        [np.where(np.isclose(ts_all, t))[0][0] for t in ts_defect],
        dtype=np.int64,
    )
    ts_defect_eval = ts_all[defect_idx]
    y_defect = y_all[:, defect_idx]

    early_end_idx, early_slopes_all, early_intercepts_all, early_r2_all = _choose_early_window(
        ts_defect_eval,
        y_defect,
        min_points=early_min_points,
        r2_drop=early_r2_drop,
        min_r2=early_min_r2,
    )

    early_slope = _gather_by_index(early_slopes_all, early_end_idx)
    early_intercept = _gather_by_index(early_intercepts_all, early_end_idx)
    early_r2 = _gather_by_index(early_r2_all, early_end_idx)

    defect_score = _spatial_zscore(early_slope)
    defect_mask_base = defect_score > float(defect_zmax)

    # Optional positivity gate
    curvature_gate = None
    late_slope_gate = None

    if positive_defects_only:
        if defect_positive_gate == "smooth":
            curvature_gate = curvature_raw

        elif defect_positive_gate == "fit":
            late_start_idx, late_slopes_all_sub, late_intercepts_all_sub, late_r2_all_sub, late_valid_idx = _choose_late_window(
                ts_defect_eval,
                y_defect,
                t_max=fit_t_max,
                min_points=fit_min_points,
                r2_drop=fit_r2_drop,
                min_r2=fit_min_r2,
            )
            late_local_map = {g: j for j, g in enumerate(late_valid_idx)}
            late_start_idx_local = np.array(
                [late_local_map[g] for g in late_start_idx],
                dtype=np.int64,
            )
            late_slope_gate = _gather_by_index(late_slopes_all_sub, late_start_idx_local)
            curvature_gate = 3.0 * late_slope_gate

        elif defect_positive_gate == "none":
            curvature_gate = None

        else:
            raise ValueError("defect_positive_gate must be 'smooth', 'fit', or 'none'")

    defect_mask = defect_mask_base.copy()
    if positive_defects_only and curvature_gate is not None:
        defect_mask &= np.isfinite(curvature_gate) & (curvature_gate > 0)

    neighbors = _mesh_neighbors_from_faces(mesh)

    defect_mask_dilated = defect_mask.copy()
    if defect_dilation_steps > 0:
        defect_mask_dilated = _dilate_vertex_mask(
            defect_mask_dilated,
            neighbors,
            n_steps=defect_dilation_steps,
        )

    # ------------------------------------------------------------------
    # Mask + inpaint
    # ------------------------------------------------------------------

    curvature_masked = curvature_raw.copy()
    curvature_masked[defect_mask_dilated] = np.nan

    curvature = _nanmean_neighbors(curvature_masked, neighbors, n_iter=inpaint_iters)

    # ------------------------------------------------------------------
    # Optional Gauss-Bonnet renormalization
    # ------------------------------------------------------------------

    total_before, info_before = integrate_gaussian_curvature(
        mesh,
        curvature,
        return_details=True,
        **integration_kwargs,
    )

    if renormalize and np.isfinite(total_before) and total_before != 0.0:
        scale = (4.0 * np.pi) / total_before
    else:
        scale = 1.0

    curvature_final = curvature * scale

    total_after, info_after = integrate_gaussian_curvature(
        mesh,
        curvature_final,
        return_details=True,
        **integration_kwargs,
    )

    if not return_debug:
        return curvature_final

    debug = {
        # core plotting/debug fields
        "defect_score": defect_score,
        "defect_mask": defect_mask_dilated,

        # extra mask diagnostics
        "defect_mask_base": defect_mask_base,
        "defect_mask_pre_dilation": defect_mask,
        "defect_mask_dilated": defect_mask_dilated,
        "defect_zmax": float(defect_zmax),
        "defect_positive_gate": defect_positive_gate,
        "positive_defects_only": bool(positive_defects_only),

        # time grids / HKS
        "L_mesh": L_mesh,
        "t_smooth": t_smooth,
        "ts_defect": ts_defect_eval,
        "ts_all": ts_all,
        "hks": hks_all,
        "fit_signal": y_all,

        # smooth curvature estimate
        "smooth_idx": smooth_idx,
        "K_candidates": K_candidates,
        "smooth_aggregate": smooth_aggregate,
        "curvature_raw": curvature_raw,
        "curvature_masked": curvature_masked,
        "curvature_final": curvature_final,

        # early fit used for defect detection
        "early_end_idx": early_end_idx,
        "early_slope": early_slope,
        "early_intercept": early_intercept,
        "early_r2": early_r2,

        # optional fit-based positivity gate
        "curvature_gate": curvature_gate,
        "late_slope_gate": late_slope_gate,

        # renormalization info
        "scale_factor": float(scale),
        "total_before": float(total_before),
        "total_after": float(total_after),
        "info_before": info_before,
        "info_after": info_after,
        "renormalized": bool(renormalize),
    }

    return curvature_final, debug
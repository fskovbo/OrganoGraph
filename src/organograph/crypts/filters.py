import numpy as np 
from organograph.mesh.hks import compute_hks

def filter_crypt_by_hks_percent(
    patches,                      # list[set[int]] candidate patches
    *,
    mesh,                         # mesh (used if HKS not provided)
    seg_vars,                     # dict from segmentation (may include "hks", "ts_mesh")
    min_percent_greater: float,   # keep patch if mean(HKS_patch) >= (1 + X%) * mean(HKS_background)
    t_min=None,                   # optional min time (in ts_mesh units); None => no lower bound
    t_max=None,                   # optional max time (in ts_mesh units); None => no upper bound
):
    """
    Filter patches using *raw* HKS mean compared to background, restricted to a time range.

    Returns
    -------
    kept_patches : list[set[int]]
    info : dict with percent_greater, bg_mean, patch_means, time_mask summary, etc.
    """
    # --- get HKS and ts_mesh ---
    hks = seg_vars.get("hks", None)
    ts_mesh = seg_vars.get("ts_mesh", None)

    # --- if missing, compute on a default / provided time axis ---
    if hks is None or ts_mesh is None:
        t0 = 1.0 if t_min is None else float(t_min)
        t1 = 20.0 if t_max is None else float(t_max)

        if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
            raise ValueError(f"Invalid time bounds for synthetic ts_mesh: t_min={t0}, t_max={t1}")

        ts_mesh = np.linspace(t0, t1, 5, dtype=float)
        hks = compute_hks(mesh, ts_mesh, coeffs=False)


    ts_mesh = np.asarray(ts_mesh, float)
    V, T = hks.shape
    if ts_mesh.shape != (T,):
        raise ValueError(f"ts_mesh must have shape (T,), got {ts_mesh.shape} for HKS with T={T}")

    # --- build time mask from [t_min, t_max] ---
    time_mask = np.ones(T, dtype=bool)
    if t_min is not None:
        time_mask &= (ts_mesh >= float(t_min))
    if t_max is not None:
        time_mask &= (ts_mesh <= float(t_max))

    if not np.any(time_mask):
        raise ValueError("Time range selects no HKS times (empty mask). Check t_min/t_max vs ts_mesh.")

    # --- union of crypt vertices to define background ---
    crypt_union = np.zeros(V, dtype=bool)
    for p in patches:
        if p:
            crypt_union[np.fromiter(p, dtype=np.int64)] = True

    noncrypt_idx = np.where(~crypt_union)[0]
    if noncrypt_idx.size == 0:
        bg_mean = float(np.mean(hks[:, time_mask])) if V > 0 else 0.0
        return list(patches), {
            "name": "mean_hks_percent_timerange",
            "kept": len(patches),
            "removed": 0,
            "bg_mean": bg_mean,
            "percent_greater": [],
            "patch_means": [],
            "t_min": t_min,
            "t_max": t_max,
            "n_times_used": int(np.sum(time_mask)),
        }

    bg_mean = float(np.mean(hks[noncrypt_idx][:, time_mask]))

    kept = []
    percent_greater = []
    patch_means = []

    for p in patches:
        idx = np.fromiter(p, dtype=np.int64)
        pm = float(np.mean(hks[idx][:, time_mask])) if idx.size else 0.0
        patch_means.append(pm)

        pct = 100.0 * (pm / (bg_mean + 1e-12) - 1.0)
        percent_greater.append(float(pct))

        if pct >= float(min_percent_greater):
            kept.append(p)

    info = {
        "name": "mean_hks_percent_timerange",
        "kept": int(len(kept)),
        "removed": int(len(patches) - len(kept)),
        "bg_mean": float(bg_mean),
        "percent_greater": percent_greater,
        "patch_means": patch_means,
        "t_min": t_min,
        "t_max": t_max,
        "n_times_used": int(np.sum(time_mask)),
    }
    return kept, info



def filter_crypts_by_markers(
    G,
    crypt_cells,
    pos_markers=None,     # list[int]
    neg_markers=None,     # list[int]
    pos_min=1,            # threshold: count OR fraction (see mode)
    neg_min=1,            # threshold: count OR fraction (see mode)
    roi_frac=None,        # e.g. 0.3 -> use cells with dist_bottom <= 0.3
    dist_bottom=None,     # (N_cells,) or (K, N_cells) normalized distances
    require_all_pos=True,
    mode="count",         # "count" (default) or "frac"/"percent"
):
    """
    Filter crypt(s) by marker content.

    Supports either:
      - mode="count": thresholds are absolute numbers of ROI cells
      - mode="frac" / "percent": thresholds are fractions of ROI cells (0..1)

    Inputs
    ------
    G : networkx.Graph
        Node attribute "markers_bin" must be array-like of length M per cell.
    crypt_cells : set[int] OR list[set[int]]
        Cell-id indices belonging to one crypt or many crypts.
    pos_markers, neg_markers : list[int] or None
        Marker indices. Positive markers must be present in enough ROI cells;
        negative markers reject if present in enough ROI cells.
    pos_min : float or int
        If mode="count": minimum number of ROI cells positive for each pos marker.
        If mode="frac": minimum fraction of ROI cells positive for each pos marker.
        (Better name: pos_min_count / pos_min_frac)
    neg_min : float or int
        If mode="count": reject if >= this many ROI cells positive for each neg marker.
        If mode="frac": reject if >= this fraction of ROI cells positive for each neg marker.
        (Better name: neg_min_count / neg_min_frac)
    roi_frac : float or None
        If given, restrict ROI to crypt cells with dist_bottom <= roi_frac.
    dist_bottom : array or None
        Normalized distance(s) used for ROI selection.
        - single crypt: (N_cells,)
        - many crypts: (K, N_cells) matching crypt order
    require_all_pos : bool
        If True: all pos_markers must pass. If False: at least one pos_marker must pass.
    mode : str
        "count" or "frac"/"percent"

    Returns
    -------
    keep : bool OR np.ndarray of bool shape (K,)
        Whether each crypt passes the filter.
    """
    mode = str(mode).lower()
    if mode in ("fraction", "fractions", "frac", "percent", "percentage"):
        mode = "frac"
    elif mode != "count":
        raise ValueError("mode must be 'count' or 'frac'/'percent'")

    # ---- normalize crypt input ----
    single = isinstance(crypt_cells, set)
    crypt_list = [crypt_cells] if single else (list(crypt_cells) if crypt_cells is not None else [])
    K = len(crypt_list)

    if K == 0:
        return False if single else np.zeros(0, dtype=bool)

    pos_markers = [] if pos_markers is None else [int(k) for k in pos_markers]
    neg_markers = [] if neg_markers is None else [int(k) for k in neg_markers]

    # Nothing to filter on => keep non-empty crypts
    if not pos_markers and not neg_markers:
        out = np.array([len(p) > 0 for p in crypt_list], dtype=bool)
        return bool(out[0]) if single else out

    if mode == "frac":
        # interpret thresholds as fractions
        if not (0.0 <= float(pos_min) <= 1.0) or not (0.0 <= float(neg_min) <= 1.0):
            raise ValueError("In mode='frac', pos_min and neg_min must be fractions in [0,1].")

    # Pull markers once: (N_cells, M)
    N = G.number_of_nodes()
    markers = np.asarray([G.nodes[i]["markers_bin"] for i in range(N)])
    # If markers are bool/0-1, sum(axis=0) gives counts.

    # dist_bottom normalization: allow (N,) or (K,N)
    Db = None
    if roi_frac is not None:
        if dist_bottom is None:
            raise ValueError("dist_bottom required when roi_frac is used.")
        Db = np.asarray(dist_bottom, float)
        if Db.ndim == 1:
            Db = np.broadcast_to(Db[None, :], (K, Db.shape[0]))
        if Db.ndim != 2 or Db.shape[0] != K or Db.shape[1] != N:
            raise ValueError("dist_bottom must be shape (N_cells,) or (K, N_cells) matching crypt_cells.")

    keep = np.zeros(K, dtype=bool)

    for j, patch in enumerate(crypt_list):
        if not patch:
            continue

        idx = np.fromiter(patch, dtype=np.int64)
        if idx.size == 0:
            continue

        # ROI selection
        if roi_frac is not None:
            dj = Db[j, idx]
            roi = idx[np.isfinite(dj) & (dj <= float(roi_frac))]
            if roi.size == 0:
                continue
        else:
            roi = idx

        n_roi = int(roi.size)

        # counts per marker index among ROI cells
        counts = markers[roi].sum(axis=0)  # (M,)

        # Convert thresholds depending on mode
        if mode == "count":
            pos_thr = float(pos_min)
            neg_thr = float(neg_min)
        else:
            # fraction thresholds -> convert to counts for consistent comparisons
            pos_thr = float(pos_min) * n_roi
            neg_thr = float(neg_min) * n_roi

        # Positive marker rule
        if pos_markers:
            ok = [(counts[k] >= pos_thr) for k in pos_markers]
            if require_all_pos:
                if not all(ok):
                    continue
            else:
                if not any(ok):
                    continue

        # Negative marker rule
        if any(counts[k] >= neg_thr for k in neg_markers):
            continue

        keep[j] = True

    return bool(keep[0]) if single else keep
import numpy as np 

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
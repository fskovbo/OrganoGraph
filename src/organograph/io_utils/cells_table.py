import numpy as np
import warnings


def make_nuclei_extractor(
    cells_df,                     # pandas DataFrame (ideally indexed by label_uid)
    *,
    label_col="label_uid",        # name of label_uid column (used if index not set)
    xyz_cols=("0.x_pos_pix", "0.y_pos_pix", "0.z_pos_pix_scaled"),
    marker_cols=None,             # list[str] of columns to extract as markers
    marker_binarize_fn=None,      # callable(markers_raw)->markers_bin ; default: >0
    marker_postprocess_fn=None,   # callable(markers_bin, marker_names)->markers_bin
):
    """
    Build a callable extractor(label_uid) -> (nuclei_xyz_raw, markers_bin, marker_names).

    Guarantees:
      - nuclei_xyz are RAW coordinates from the table (no transforms here)
      - fast lookup if cells_df is indexed by label_uid
      - marker extraction is customizable and can include dataset-specific postprocessing
    """
    if marker_cols is None:
        marker_cols = []
    marker_cols = list(marker_cols)

    if marker_binarize_fn is None:
        def marker_binarize_fn(x):
            return np.asarray(x) > 0.0

    # --- validate columns once (fail early) ---
    missing_xyz = [c for c in xyz_cols if c not in cells_df.columns]
    if missing_xyz:
        raise ValueError(f"Missing xyz_cols in cells_df: {missing_xyz}")

    missing_markers = [c for c in marker_cols if c not in cells_df.columns]
    if missing_markers:
        raise ValueError(f"Missing marker_cols in cells_df: {missing_markers}")

    # --- decide lookup mode once ---
    use_index = getattr(cells_df.index, "name", None) == label_col

    def _get_rows(label_uid):
        if use_index:
            df_org = cells_df.loc[label_uid]
            # normalize Series -> DataFrame
            if not hasattr(df_org, "columns"):
                df_org = df_org.to_frame().T
            return df_org
        # fallback: boolean mask (slower)
        return cells_df[cells_df[label_col] == label_uid]

    def extract(label_uid):
        df_org = _get_rows(label_uid)
        if df_org is None or len(df_org) == 0:
            raise ValueError(f"No nuclei rows found for label_uid={label_uid}")

        nuclei_xyz = df_org.loc[:, list(xyz_cols)].to_numpy(dtype=float, copy=False)

        if marker_cols:
            markers_raw = df_org.loc[:, marker_cols].to_numpy(copy=False)
            markers_bin = np.asarray(marker_binarize_fn(markers_raw), dtype=np.int8)
            if markers_bin.shape[0] != nuclei_xyz.shape[0]:
                raise ValueError("marker_binarize_fn returned wrong N (rows)")

            if marker_postprocess_fn is not None:
                markers_bin = marker_postprocess_fn(markers_bin, marker_cols)
                markers_bin = np.asarray(markers_bin, dtype=np.int8)

        else:
            markers_bin = np.zeros((nuclei_xyz.shape[0], 0), dtype=np.int8)

        return nuclei_xyz, markers_bin, marker_cols

    return extract


def prepare_cells_table(cells_df, label_col="label_uid", sort_index=True):
    """
    Prepare a nuclei/cell table for fast per-organoid lookup.

    What it does
    ------------
    - Ensures `label_col` exists.
    - Sets the DataFrame index to `label_col` (if not already).
      This enables fast lookup: df.loc[label_uid]
    - Warns if label_uid values are duplicated (still supported).

    Notes on performance / copying
    ------------------------------
    Passing DataFrames is by reference. Setting an index returns a new DataFrame
    object, but it does not necessarily copy all underlying column data in a way
    that matters for your use case. The main goal is to avoid repeated boolean
    masking over a large table.

    Returns
    -------
    df_idx : pandas.DataFrame
        Same data, indexed by `label_col`.
    """
    if label_col not in cells_df.columns and getattr(cells_df.index, "name", None) != label_col:
        raise ValueError(f"cells_df must contain a '{label_col}' column or be indexed by it.")

    # Already indexed correctly
    if getattr(cells_df.index, "name", None) == label_col:
        df_idx = cells_df
    else:
        # Keep the column as well (drop=False) because it's handy for debugging/exports
        df_idx = cells_df.set_index(label_col, drop=False)

    # Optional sorting (can speed up some index ops; slight upfront cost)
    if sort_index:
        df_idx = df_idx.sort_index()

    # Warn on duplicates (not an error: multiple rows per label_uid is normal for cells)
    try:
        dup = df_idx.index.duplicated().any()
    except Exception:
        dup = False

    if dup:
        warnings.warn(
            f"prepare_cells_table: '{label_col}' index contains duplicates. "
            "This is usually expected (multiple cells per organoid). "
            "df.loc[label_uid] will return multiple rows.",
            RuntimeWarning,
        )

    return df_idx


def suppress_marker_if_coexpressed(
    markers_bin,           # (N, K) Binary marker matrix (0/1 or bool)
    marker_names,          # list[str] Names of marker columns (length K)
    *,
    exclusive_marker,      # e.g. "LGR.bin"
    forbidden_markers,     # e.g. ("LYZ.bin","MUC.bin",...)
    copy=True,             # If True, return a modified copy. If False, modify in place.
    ignore_missing=True,   # If False, raise if any required marker is missing, otherwise ignore.
):
    """
    Suppress one marker (set it to 0) in cells that coexpress any forbidden markers.

    This is meant to be simple and chainable: call it multiple times for multiple
    exclusive markers.

    Returns
    -------
    out : (N, K) ndarray
        Updated marker matrix.
    """
    if marker_names is None or len(marker_names) == 0:
        raise ValueError("marker_names must be a non-empty list")

    X = np.asarray(markers_bin)
    if X.ndim != 2:
        raise ValueError(f"markers_bin must be 2D (N,K); got shape {X.shape}")

    name_to_idx = {n: i for i, n in enumerate(marker_names)}

    if exclusive_marker not in name_to_idx:
        if ignore_missing:
            return X.copy() if copy else X
        raise ValueError(
            f"exclusive_marker '{exclusive_marker}' not found. "
            f"Available: {sorted(marker_names)}"
        )

    excl_idx = name_to_idx[exclusive_marker]

    forb_idx = []
    for m in forbidden_markers:
        if m in name_to_idx:
            forb_idx.append(name_to_idx[m])
        elif not ignore_missing:
            raise ValueError(
                f"forbidden marker '{m}' not found. Available: {sorted(marker_names)}"
            )

    # If none of the forbidden markers exist, nothing to do
    if len(forb_idx) == 0:
        return X.copy() if copy else X

    out = X.copy() if copy else X

    excl_pos = out[:, excl_idx].astype(bool, copy=False)
    forb_pos = out[:, forb_idx].astype(bool, copy=False).any(axis=1)

    conflict = excl_pos & forb_pos
    if np.any(conflict):
        out[conflict, excl_idx] = 0

    return out

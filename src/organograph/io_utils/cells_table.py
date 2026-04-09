import numpy as np
import warnings



def make_nuclei_extractor(
    cells_df,                     # pandas DataFrame (ideally indexed by label_uid)
    *,
    label_col="label_uid",        # name of label_uid column (used if index not set)
    xyz_cols=("0.x_pos_pix", "0.y_pos_pix", "0.z_pos_pix_scaled"),
    marker_cols=None,             # list[str] of columns to extract as markers
    marker_alias=None,            # optional list[str] returned instead of marker_cols
    marker_binarize_fn=None,      # callable(markers_raw)->markers_bin ; default: >0
    marker_postprocess_fn=None,   # callable(markers_bin, marker_names)->markers_bin OR (markers_bin, marker_names)
):
    """
    Build a callable extractor(label_uid) -> (nuclei_xyz_raw, markers_bin, marker_names).

    Guarantees
    ----------
    - nuclei_xyz are RAW coordinates from the table
    - fast lookup if cells_df is indexed by label_uid
    - marker extraction customizable
    - marker_alias allows returning semantic names instead of dataframe column names
    - marker_postprocess_fn may optionally update both the marker matrix and marker names

    Returns
    -------
    extract : callable
        Function extract(label_uid) -> (nuclei_xyz, markers_bin, marker_names)
    """
    if marker_cols is None:
        marker_cols = []
    marker_cols = list(marker_cols)

    if marker_alias is not None:
        marker_alias = list(marker_alias)
        if len(marker_alias) != len(marker_cols):
            raise ValueError("marker_alias must have the same length as marker_cols")
    else:
        marker_alias = list(marker_cols)

    if marker_binarize_fn is None:
        def marker_binarize_fn(x):
            return np.asarray(x) > 0.0

    # --- validate columns once ---
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
            marker_names = list(marker_alias)

            # raw marker values from table
            markers_raw = df_org.loc[:, marker_cols].to_numpy(copy=False)

            # binarize
            markers_bin = np.asarray(marker_binarize_fn(markers_raw), dtype=np.int8)

            if markers_bin.ndim != 2:
                raise ValueError(
                    f"marker_binarize_fn must return a 2D array; got shape {markers_bin.shape}"
                )
            if markers_bin.shape[0] != nuclei_xyz.shape[0]:
                raise ValueError(
                    "marker_binarize_fn returned wrong number of rows "
                    f"(got {markers_bin.shape[0]}, expected {nuclei_xyz.shape[0]})"
                )
            if markers_bin.shape[1] != len(marker_names):
                raise ValueError(
                    "marker_binarize_fn returned wrong number of columns "
                    f"(got {markers_bin.shape[1]}, expected {len(marker_names)})"
                )

            # optional postprocessing
            if marker_postprocess_fn is not None:
                out = marker_postprocess_fn(markers_bin, marker_names)

                # Accept either:
                #   markers_bin_out
                # or
                #   (markers_bin_out, marker_names_out)
                if isinstance(out, tuple):
                    if len(out) != 2:
                        raise ValueError(
                            "marker_postprocess_fn must return either markers_bin "
                            "or (markers_bin, marker_names)"
                        )
                    markers_bin, marker_names = out
                else:
                    markers_bin = out

                markers_bin = np.asarray(markers_bin, dtype=np.int8)
                marker_names = list(marker_names)

                if markers_bin.ndim != 2:
                    raise ValueError(
                        f"marker_postprocess_fn must return a 2D marker matrix; got shape {markers_bin.shape}"
                    )
                if markers_bin.shape[0] != nuclei_xyz.shape[0]:
                    raise ValueError(
                        "marker_postprocess_fn returned wrong number of rows "
                        f"(got {markers_bin.shape[0]}, expected {nuclei_xyz.shape[0]})"
                    )
                if markers_bin.shape[1] != len(marker_names):
                    raise ValueError(
                        "marker_postprocess_fn returned inconsistent marker matrix and marker names "
                        f"(matrix has {markers_bin.shape[1]} columns, marker_names has length {len(marker_names)})"
                    )

        else:
            markers_bin = np.zeros((nuclei_xyz.shape[0], 0), dtype=np.int8)
            marker_names = []

        return nuclei_xyz, markers_bin, marker_names

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


def enforce_marker_exclusivity(
    markers_bin,         # (N, K) binary marker matrix
    marker_names,        # list[str] marker names corresponding to columns of markers_bin
    *,
    exclusivity_rules,   # dict: marker_name -> list of marker_names that exclude it
    copy=True,           # if True, return a modified copy; else modify in place
    ignore_missing=True, # if False, raise if a marker in the rules is missing
):
    """
    Enforce marker exclusivity by repeatedly calling suppress_marker_if_coexpressed().

    For each entry
        marker_name -> forbidden_markers
    this applies:

        marker_name := 0  if any forbidden marker is positive

    Parameters
    ----------
    markers_bin : (N, K) array_like
        Binary marker matrix (0/1 or bool).
    marker_names : list[str]
        Marker names corresponding to the K columns of markers_bin.
    exclusivity_rules : dict
        Mapping:
            marker_name -> list of markers that exclude it
        Example:
            {
                "LGR5":   ["CHROMA", "MUC", "ALDOB", "GLUC", "AGR", "SERO", "LYZ"],
                "CHROMA": ["MUC", "GLUC", "SERO", "LYZ"],
                ...
            }
    copy : bool
        If True, return a modified copy. If False, modify in place.
    ignore_missing : bool
        If True, missing markers referenced in the rules are ignored.
        If False, missing markers raise a ValueError.

    Returns
    -------
    out : (N, K) ndarray
        Updated marker matrix with exclusivity enforced.

    Notes
    -----
    - The output has the same shape and column order as the input.
    - No new marker categories are created.
    - Rules are applied sequentially in dictionary order.
    """
    if marker_names is None or len(marker_names) == 0:
        raise ValueError("marker_names must be a non-empty list")

    if not isinstance(exclusivity_rules, dict):
        raise ValueError("exclusivity_rules must be a dict")

    out = np.asarray(markers_bin)
    if out.ndim != 2:
        raise ValueError(f"markers_bin must be 2D (N,K); got shape {out.shape}")

    out = out.copy() if copy else out

    for marker_name, forbidden_markers in exclusivity_rules.items():
        out = suppress_marker_if_coexpressed(
            out,
            marker_names,
            exclusive_marker=marker_name,
            forbidden_markers=forbidden_markers,
            copy=False,                  # already copied above if needed
            ignore_missing=ignore_missing,
        )

    return out


def harmonize_markers(
    markers_bin,          # (N, K) binary marker matrix
    marker_names,         # list[str], length K
    *,
    marker_rules,         # dict: new_name -> old_name or list/tuple of old_names
    keep_unmapped=True,   # if True, keep markers not consumed by marker_rules
    dtype=np.int8,
):
    """
    Rename and/or combine markers into a harmonized marker set.

    Rules
    -----
    - new_name: "old_name"
        rename one marker
    - new_name: ["old1", "old2", ...]
        combine several markers by logical OR

    Important behavior
    ------------------
    - Any source marker mentioned in marker_rules is *consumed* and therefore
      removed from the unmapped output.
    - Harmonized markers are appended at the END of the output.
    - Output column order always matches output marker_names.

    Parameters
    ----------
    markers_bin : (N, K) array_like
        Binary marker matrix.
    marker_names : list[str]
        Marker names corresponding to columns of markers_bin.
    marker_rules : dict
        Mapping:
            new_name -> old_name or iterable of old_names

        Example:
            {
                "TA": ["Cyclin A", "Cyclin D", "KI67"],
                "LGR5": "STEMCELL",
            }

    keep_unmapped : bool
        If True, markers not consumed by marker_rules are kept unchanged.
        If False, only harmonized markers are returned.
    dtype : numpy dtype
        Output dtype.

    Returns
    -------
    markers_out : (N, K_out) ndarray
        Harmonized marker matrix.
    marker_names_out : list[str]
        Harmonized marker names.

    Notes
    -----
    - Missing source markers are ignored.
    - If none of the source markers for a harmonized marker are present,
      the harmonized marker is omitted.
    """
    X = np.asarray(markers_bin)
    if X.ndim != 2:
        raise ValueError(f"markers_bin must be 2D (N,K); got shape {X.shape}")

    marker_names = list(marker_names)
    name_to_idx = {n: i for i, n in enumerate(marker_names)}
    N = X.shape[0]

    # Normalize rules so each new_name maps to a list of source names
    normalized_rules = {}
    consumed_sources = set()

    for new_name, src in marker_rules.items():
        if isinstance(src, str):
            src_list = [src]
        else:
            src_list = list(src)

        normalized_rules[new_name] = src_list
        consumed_sources.update(src_list)

    out_cols = []
    out_names = []

    # 1) keep unmapped markers first, in original order
    if keep_unmapped:
        for name in marker_names:
            if name not in consumed_sources:
                out_cols.append(np.asarray(X[:, name_to_idx[name]], dtype=dtype))
                out_names.append(name)

    # 2) append harmonized markers at the end
    for new_name, src_list in normalized_rules.items():
        src_idx = [name_to_idx[s] for s in src_list if s in name_to_idx]

        # If none of the source markers are present, skip this harmonized marker
        if len(src_idx) == 0:
            continue

        if len(src_idx) == 1:
            col = np.asarray(X[:, src_idx[0]], dtype=dtype)
        else:
            col = np.any(X[:, src_idx].astype(bool, copy=False), axis=1).astype(dtype)

        out_cols.append(col)
        out_names.append(new_name)

    if len(out_cols) == 0:
        return np.zeros((N, 0), dtype=dtype), []

    markers_out = np.column_stack(out_cols)
    return markers_out, out_names
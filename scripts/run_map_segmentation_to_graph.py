#!/usr/bin/env python3
"""
Project mesh-based crypt segmentations onto organoid cell graphs.

Behavior
--------
For each mesh discovered from mesh_config.json:
1. Resolve the corresponding mesh-based crypt segmentation saved by
   run_crypt_segmentation.py.
2. Try to load the corresponding precomputed cell graph saved by
   run_graph_preprocess.py, using graph index.csv files when available.
3. If the graph does not exist, optionally build it on the fly from:
      - the organoid mesh
      - the nuclei/cell CSV
4. Project mesh crypt patches to graph nodes using each node's "proj_vertex".
5. Optionally filter projected crypts by a minimum number of cells.
6. Save the projected graph crypt patches as crypts_ll (list of node-id lists).

Primary storage choice
----------------------
By default, projected crypts are stored as:

    crypts_ll : list[list[int]]

where each inner list contains the graph node ids belonging to one crypt.

Optionally, a per-node crypt index vector can also be saved:

    crypt_index_by_node : (N_nodes,) int
        -1 means "not in a crypt", otherwise the crypt index.

This vector is useful for fast node-level lookup, but the patch list remains the
more convenient primary representation for retrieving whole crypts.
"""

import os
import time
import numpy as np
import pandas as pd

from organograph.mesh.OrganoidMesh import OrganoidMesh

from organograph.io_utils.cells_table import (
    prepare_cells_table,
    make_nuclei_extractor,
)
from organograph.io_utils.dataset_config import load_mesh_dataset_config, load_cell_table_config
from organograph.io_utils.path_parsing import parse_mesh_path, discover_mesh_paths
from organograph.io_utils.blacklist import default_discard_labels_path, load_optional_blacklist
from organograph.io_utils.run_metadata import write_run_settings
from organograph.graph.build import build_organoid_graph, assign_mesh_patches_to_graph
from organograph.io_utils.segmentation_io import load_mesh_crypt_segmentation
from organograph.graph.io import load_cell_graph, save_cell_graph
from organograph.graph.access import graph_get
from organograph.graph.marker_postprocess import (
    ablate_lysozyme_not_agr2_in_clusters,
    copy_graph_marker_fields,
    suppress_graph_marker_if_coexpressed,
)

# =============================================================================
# DATASET PATHS (EDIT THESE)
# =============================================================================

DATASET         = "20251201" # "20251201"  20250929

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(_SCRIPT_DIR)

DATASET_ROOT    = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET)
MESH_DATA_DIR   = os.path.join(DATASET_ROOT, "fractal_output")

# Input: mesh-based segmentation results from run_crypt_segmentation.py
SEG_MESH_DIR    = os.path.join(DATASET_ROOT, "crypt_segmentations_mesh")

# Input: nuclei/cell table used only when BUILD_GRAPHS_IF_MISSING=True
CELLS_CSV       = os.path.join(DATASET_ROOT, "feature_tables", "cell_features_class.csv")

# Optional existing graph directory from run_graph_preprocess.py
GRAPHS_DIR      = os.path.join(DATASET_ROOT, "graphs_preprocessed")

# Output: graph-based crypt projections
GRAPH_SEG_DIR   = os.path.join(DATASET_ROOT, "crypt_segmentations_graph")

# config files with data structure
MESH_CONFIG_PATH= os.path.join(DATASET_ROOT, "mesh_config.json")
CELL_CONFIG_PATH= os.path.join(DATASET_ROOT, "cell_table_config.json")
BLACKLIST_PATH  = default_discard_labels_path(DATASET_ROOT)



# =============================================================================
# OPTIONAL FILTERING / BEHAVIOR
# =============================================================================

# Optional override. If None, use all timepoints from mesh_config.json
TIMEPOINTS = None # ['day3', 'day3p5', 'day4', 'day4p5', 'day4p5-more']

OVERWRITE = True
VERBOSE = True
DRY_RUN = False
MAX_MESHES = None

# Minimum number of graph nodes (cells) for a projected crypt to be kept
MIN_CELLS_PER_CRYPT = 10   # e.g. 5, or None to disable

# Whether to save a per-node crypt index vector in addition to crypts_ll
SAVE_CRYPT_INDEX_VECTOR = False

# If a graph is missing and we build it on the fly, save it to GRAPHS_DIR
BUILD_GRAPHS_IF_MISSING = False
SAVE_BUILT_GRAPHS = False


# =============================================================================
# LOAD CONFIG
# =============================================================================

mesh_cfg = load_mesh_dataset_config(MESH_CONFIG_PATH)
cell_cfg = load_cell_table_config(CELL_CONFIG_PATH)

ZARR_NAME_BY_TP = mesh_cfg["zarr_name_by_tp"]
ROUND_BY_TP = mesh_cfg["round_by_tp"]
MESHNAME_BY_TP = mesh_cfg["meshname_by_tp"]
WELLS_BY_TP = mesh_cfg.get("wells_by_tp", {})


COORD_COLS = tuple(cell_cfg["coord_cols"])
MARKER_COLS = list(cell_cfg["marker_cols"])
MARKER_ALIAS = list(cell_cfg["marker_names"])


def marker_config_name_to_alias(name):
    """Resolve a marker config entry to the names stored on graph nodes."""
    if name in MARKER_ALIAS:
        return name
    if name in MARKER_COLS:
        return MARKER_ALIAS[MARKER_COLS.index(name)]
    raise ValueError(
        f"Marker '{name}' is not in marker_cols or marker_names. "
        f"Available marker_cols={MARKER_COLS}; marker_names={MARKER_ALIAS}"
    )


def optional_marker_config_name_to_alias(name):
    """Resolve a marker name if present in this dataset marker panel."""
    try:
        return marker_config_name_to_alias(name)
    except ValueError:
        return None


LGR5_MARKER = marker_config_name_to_alias(cell_cfg["lgr5_marker"])
COEXP_MARKERS = tuple(marker_config_name_to_alias(m) for m in cell_cfg["coexp_markers"])

# Graph-level marker postprocessing for graphs built on the fly. Existing graphs
# loaded from GRAPHS_DIR are assumed to have been created by run_graph_preprocess.py.
STORE_RAW_GRAPH_MARKERS = True  # Preserve original per-node markers in markers_int_raw and markers_bin_raw.
ENABLE_LYSOZYME_AGR2_ABLATION = True  # Remove Lysozyme from clustered Lysozyme+ cells unless they are Agr2+.
LYSOZYME_MARKER = marker_config_name_to_alias("Lysozyme")
AGR2_MARKER = marker_config_name_to_alias("Agr2")
LYSOZYME_ABLATION_MIN_CLUSTER_SIZE = 2  # Only process connected Lysozyme+ components with at least this many cells.
ENABLE_MUCIN2_AGR2_ABLATION = True  # Remove Mucin 2 from clustered Mucin 2+ cells unless they are Agr2+.
MUCIN2_MARKER = optional_marker_config_name_to_alias("Mucin 2")
MUCIN2_ABLATION_MIN_CLUSTER_SIZE = 2  # Only process connected Mucin 2+ components with at least this many cells.
ENABLE_GRAPH_COEXPRESSION_SUPPRESSION = True  # Apply the LGR5-vs-forbidden-marker rule after cluster cleanup.


# =============================================================================
# HELPERS
# =============================================================================

def marker_postprocess(markers_bin, marker_names):
    """Keep table-derived markers raw until graph-level cleanup can use adjacency."""
    return markers_bin


def graph_marker_postprocess(G):
    """Apply graph-dependent marker cleanup after cell adjacency is available."""
    steps = []

    if STORE_RAW_GRAPH_MARKERS:
        copy_graph_marker_fields(G)
        steps.append("copy_marker_fields_raw")

    if ENABLE_LYSOZYME_AGR2_ABLATION:
        ablate_lysozyme_not_agr2_in_clusters(
            G,
            lysozyme_marker=LYSOZYME_MARKER,
            agr2_marker=AGR2_MARKER,
            min_cluster_size=LYSOZYME_ABLATION_MIN_CLUSTER_SIZE,
        )
        steps.append("ablate_lysozyme_not_agr2_in_clusters")

    if ENABLE_MUCIN2_AGR2_ABLATION and MUCIN2_MARKER is not None:
        ablate_lysozyme_not_agr2_in_clusters(
            G,
            lysozyme_marker=MUCIN2_MARKER,
            agr2_marker=AGR2_MARKER,
            min_cluster_size=MUCIN2_ABLATION_MIN_CLUSTER_SIZE,
        )
        steps.append("ablate_mucin2_not_agr2_in_clusters")

    if ENABLE_GRAPH_COEXPRESSION_SUPPRESSION:
        suppress_graph_marker_if_coexpressed(
            G,
            exclusive_marker=LGR5_MARKER,
            forbidden_markers=COEXP_MARKERS,
            copy=True,
            ignore_missing=False,
        )
        steps.append("suppress_marker_if_coexpressed")

    G.graph["marker_postprocess_steps"] = steps
    return G


def enabled_marker_postprocessing_functions():
    steps = []
    if STORE_RAW_GRAPH_MARKERS:
        steps.append("copy_marker_fields_raw")
    if ENABLE_LYSOZYME_AGR2_ABLATION:
        steps.append("ablate_lysozyme_not_agr2_in_clusters")
    if ENABLE_MUCIN2_AGR2_ABLATION and MUCIN2_MARKER is not None:
        steps.append("ablate_mucin2_not_agr2_in_clusters")
    if ENABLE_GRAPH_COEXPRESSION_SUPPRESSION:
        steps.append("suppress_marker_if_coexpressed")
    return steps


def resolve_timepoints(timepoints_override, zarr_names, rounds, meshes):
    """Return the configured timepoints unless a manual override is supplied."""
    if timepoints_override is not None:
        return list(timepoints_override)
    return [tp for tp in zarr_names if tp in rounds and tp in meshes]


def build_graph_index(graphs_dir, timepoints):
    """Read graph index.csv files and map both label_uid and parsed_label_uid to graph paths."""
    index = {}
    for tp in timepoints:
        idx_path = os.path.join(graphs_dir, tp, "index.csv")
        if not os.path.exists(idx_path):
            continue
        try:
            df = pd.read_csv(idx_path)
        except Exception as exc:
            if VERBOSE:
                print(f"[warn] could not read graph index {idx_path}: {exc}")
            continue
        for _, row in df.iterrows():
            graph_path = row.get("graph_path", None)
            if pd.isna(graph_path):
                continue
            graph_path = str(graph_path)
            if not os.path.isabs(graph_path):
                graph_path = os.path.join(graphs_dir, tp, os.path.basename(graph_path))
            keys = []
            for col in ("label_uid", "parsed_label_uid"):
                val = row.get(col, None)
                if pd.notna(val):
                    keys.append(str(val))
            for key in keys:
                index[(str(tp), key)] = graph_path
    return index


def patches_to_ll(patches):
    """list[set[int]] -> list[list[int]] for npz saving."""
    return [sorted(list(p)) for p in (patches or [])]


def graph_path_for(tp, label_uid):
    return os.path.join(GRAPHS_DIR, tp, f"{label_uid}.gpickle")


def mesh_seg_path_for(tp, label_uid):
    return os.path.join(SEG_MESH_DIR, tp, f"{label_uid}.npz")


def resolve_graph_path(tp, label_uid, parsed_label_uid=None, graph_index=None, seg=None):
    """Resolve a graph path using direct layout, graph index.csv, or segmentation metadata."""
    candidates = []
    for key in (label_uid, parsed_label_uid):
        if key:
            candidates.append(graph_path_for(tp, key))
            if graph_index is not None:
                indexed = graph_index.get((str(tp), str(key)))
                if indexed:
                    candidates.append(indexed)

    if seg is not None:
        seg_graph_path = seg.get("graph_path", None)
        if seg_graph_path is not None:
            candidates.append(str(seg_graph_path))

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else graph_path_for(tp, label_uid)


def output_path_for(tp, label_uid):
    return os.path.join(GRAPH_SEG_DIR, tp, f"{label_uid}.npz")



def filter_graph_patches_by_min_cells(graph_patches, min_cells_per_crypt):
    """
    Filter projected graph crypt patches by minimum number of cells.

    Returns
    -------
    graph_patches_kept : list[set[int]]
        Filtered graph patches.
    keep_idx : (N_kept,) ndarray
        Indices of the original graph_patches that survived.
    """
    if min_cells_per_crypt is None:
        keep_idx = np.arange(len(graph_patches), dtype=np.int64)
        return graph_patches, keep_idx

    keep_mask = np.array(
        [len(p) >= int(min_cells_per_crypt) for p in graph_patches],
        dtype=bool,
    )
    keep_idx = np.where(keep_mask)[0]
    graph_patches_kept = [graph_patches[i] for i in keep_idx]
    return graph_patches_kept, keep_idx


def subset_per_crypt_seg_vars(seg, keep_idx):
    """
    Subset per-crypt arrays in a loaded mesh segmentation dict.

    Only keys whose first dimension matches the number of crypts are subset.
    Shared arrays such as d_discretized are left unchanged.

    Parameters
    ----------
    seg : dict
        Output of load_mesh_crypt_segmentation(...)
    keep_idx : array_like
        Indices of crypts that survived graph-side filtering.

    Returns
    -------
    seg_sub : dict
        Copy of seg with aligned per-crypt quantities subset.
    """
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    seg_sub = dict(seg)

    # Number of crypts in the original mesh segmentation
    n0 = len(seg.get("crypts_mesh", []))

    per_crypt_keys = (
        "crypts_mesh",
        "bottom_vertex_ids",
        "L_crypts",
        "d_crypts",
        "crypt_constrictions",
        "crypt_elongations",
    )

    for k in per_crypt_keys:
        if k not in seg_sub:
            continue

        val = seg_sub[k]

        if isinstance(val, list):
            if len(val) == n0:
                seg_sub[k] = [val[i] for i in keep_idx]

        else:
            arr = np.asarray(val)
            if arr.ndim >= 1 and arr.shape[0] == n0:
                seg_sub[k] = arr[keep_idx]

    # circumference_crypts is usually (K, B), but handle both (K,B) and (B,K)
    if "circumference_crypts" in seg_sub:
        arr = np.asarray(seg_sub["circumference_crypts"])
        if arr.ndim == 2:
            if arr.shape[0] == n0:
                seg_sub["circumference_crypts"] = arr[keep_idx]
            elif arr.shape[1] == n0:
                seg_sub["circumference_crypts"] = arr[:, keep_idx]

    return seg_sub


def remap_mesh_to_graph_index(mesh_to_graph_index, keep_idx_graph):
    """
    Update mesh_to_graph_index after graph-patch filtering.

    Parameters
    ----------
    mesh_to_graph_index : array_like
        Original mapping from mesh crypt index -> graph patch index (or -1).
    keep_idx_graph : array_like
        Indices of graph patches that survived filtering.

    Returns
    -------
    new_map : ndarray
        Updated mapping from mesh crypt index -> filtered graph patch index (or -1).
    """
    keep_idx_graph = np.asarray(keep_idx_graph, dtype=np.int64)
    old_to_new = {int(old): i for i, old in enumerate(keep_idx_graph)}

    out = []
    for x in mesh_to_graph_index:
        x = int(x)
        out.append(old_to_new.get(x, -1) if x >= 0 else -1)
    return np.asarray(out, dtype=np.int64)


def make_crypt_index_by_node(G, graph_patches):
    """
    Build a per-node crypt index vector aligned to node ids 0..N-1.
    -1 means "not in any crypt".
    """
    n = G.number_of_nodes()
    arr = np.full(n, -1, dtype=np.int64)
    for k, patch in enumerate(graph_patches):
        idx = np.fromiter(patch, dtype=np.int64)
        arr[idx] = k
    return arr


def build_graph_for_organoid(mesh_path, label_uid, extractor):
    """
    Build one organoid graph from mesh + nuclei table.
    """
    mesh = OrganoidMesh(str(mesh_path))
    mesh.normalize_inplace()
    mesh.label_uid = label_uid
    G, _aux = build_organoid_graph(mesh=mesh, extract_fn=extractor)
    return graph_marker_postprocess(G)


def project_mesh_field_to_graph(G, field_mesh, *, proj_field="proj_vertex"):
    """
    Project a mesh-based field to graph nodes via each node's projected mesh vertex.

    Parameters
    ----------
    G
        Cell graph whose nodes contain a mesh vertex index in `proj_field`.
    field_mesh : array_like
        Mesh-based field to project. Supported shapes:
          - (V_mesh,)      : one value per mesh vertex
          - (K, V_mesh)    : K values per mesh vertex (e.g. one row per crypt)
    proj_field : str
        Node attribute containing the projected mesh vertex index.

    Returns
    -------
    field_graph : ndarray
        Graph-level field evaluated at each node's projected mesh vertex.
        Shapes:
          - (N_nodes,)      if input shape is (V_mesh,)
          - (K, N_nodes)    if input shape is (K, V_mesh)

        Nodes with invalid proj_vertex get NaN.
    """
    F = np.asarray(field_mesh)

    if F.ndim not in (1, 2):
        raise ValueError(
            f"field_mesh must have shape (V_mesh,) or (K, V_mesh), got {F.shape}"
        )

    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        raise ValueError("Graph has no nodes")

    proj_vertex = graph_get(G, proj_field, dtype=np.int64)   # (N_nodes,)
    if proj_vertex.ndim != 1 or proj_vertex.size != n_nodes:
        raise ValueError("proj_vertex must be a 1D array of length N_nodes")

    V_mesh = F.shape[-1]
    valid = (proj_vertex >= 0) & (proj_vertex < V_mesh)

    if F.ndim == 1:
        out = np.full(n_nodes, np.nan, dtype=float)
        if np.any(valid):
            out[valid] = F[proj_vertex[valid]]
        return out

    # F.ndim == 2
    out = np.full((F.shape[0], n_nodes), np.nan, dtype=float)
    if np.any(valid):
        out[:, valid] = F[:, proj_vertex[valid]]
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.perf_counter()
    stats = {
        "mesh_files_found": 0,
        "segmentations_found": 0,
        "projected_saved": 0,
        "graphs_built_on_the_fly": 0,
        "graphs_loaded": 0,
        "skipped_blacklist": 0,
        "skipped_existing": 0,
        "skipped_missing_segmentation": 0,
        "skipped_missing_graph": 0,
        "failed": 0,
        "dry_run": bool(DRY_RUN),
    }

    if BUILD_GRAPHS_IF_MISSING and not os.path.exists(CELLS_CSV):
        raise FileNotFoundError(f"CELLS_CSV not found: {CELLS_CSV}")

    extractor = None
    if BUILD_GRAPHS_IF_MISSING:
        # load + prepare nuclei table once
        cells_df = pd.read_csv(CELLS_CSV)
        cells_df = prepare_cells_table(cells_df, label_col="label_uid")

        extractor = make_nuclei_extractor(
            cells_df,
            label_col="label_uid",
            xyz_cols=COORD_COLS,
            marker_cols=MARKER_COLS,
            marker_alias=MARKER_ALIAS,
            marker_postprocess_fn=marker_postprocess,
            return_marker_intensity=True,
        )

    search_timepoints = resolve_timepoints(
        TIMEPOINTS,
        ZARR_NAME_BY_TP,
        ROUND_BY_TP,
        MESHNAME_BY_TP,
    )
    blacklist = load_optional_blacklist(BLACKLIST_PATH, label="blacklist", verbose=VERBOSE)
    graph_index = build_graph_index(GRAPHS_DIR, search_timepoints)

    mesh_paths = discover_mesh_paths(
        data_dir=MESH_DATA_DIR,
        timepoints=search_timepoints,
        zarr_names=ZARR_NAME_BY_TP,
        rounds=ROUND_BY_TP,
        meshes=MESHNAME_BY_TP,
        wells=WELLS_BY_TP,
    )

    if VERBOSE:
        print(f"[graph-proj] searching timepoints: {search_timepoints}")
        print(f"[graph-proj] found {len(mesh_paths)} mesh files (pre-filter)")
        print(f"[graph-proj] loaded graph index entries: {len(graph_index)}")
    stats["mesh_files_found"] = int(len(mesh_paths))

    n_done = 0

    for mesh_path in mesh_paths:
        try:
            rec = parse_mesh_path(mesh_path)
        except Exception as e:
            if VERBOSE:
                print(f"[skip] cannot parse mesh path: {mesh_path} ({e})")
            stats["failed"] += 1
            continue

        tp = rec.get("timepoint", None)
        parsed_label_uid = rec.get("label_uid", None)

        if not tp or not parsed_label_uid:
            if VERBOSE:
                print(f"[skip] missing timepoint/label_uid for: {mesh_path}")
            continue

        if parsed_label_uid in blacklist:
            if VERBOSE:
                print(f"[skip] {parsed_label_uid} is blacklisted")
            stats["skipped_blacklist"] += 1
            continue

        seg_path = mesh_seg_path_for(tp, parsed_label_uid)
        if not os.path.exists(seg_path):
            if VERBOSE:
                print(f"[skip] missing mesh segmentation for {tp}/{parsed_label_uid}: {seg_path}")
            stats["skipped_missing_segmentation"] += 1
            continue
        stats["segmentations_found"] += 1

        try:
            seg = load_mesh_crypt_segmentation(seg_path)
        except Exception as e:
            if VERBOSE:
                print(f"[skip] failed loading segmentation {seg_path}: {e}")
            stats["failed"] += 1
            continue

        tp = str(seg.get("timepoint", tp))
        mesh_seg_label_uid = str(seg.get("label_uid", parsed_label_uid))
        mesh_path = str(seg.get("mesh_path", mesh_path))
        crypts_mesh = seg["crypts_mesh"]


        if not tp or not mesh_seg_label_uid or not mesh_path:
            if VERBOSE:
                print(f"[skip] missing timepoint/label_uid/mesh_path in {seg_path}")
            continue

        gpath = resolve_graph_path(
            tp,
            mesh_seg_label_uid,
            parsed_label_uid=parsed_label_uid,
            graph_index=graph_index,
            seg=seg,
        )
        graph_label_uid = os.path.splitext(os.path.basename(gpath))[0] if gpath else mesh_seg_label_uid

        if graph_label_uid in blacklist and graph_label_uid != parsed_label_uid:
            if VERBOSE:
                print(f"[skip] {graph_label_uid} is blacklisted")
            stats["skipped_blacklist"] += 1
            continue

        out_path = output_path_for(tp, graph_label_uid)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if (not OVERWRITE) and os.path.exists(out_path):
            if VERBOSE:
                print(f"[skip] exists: {out_path}")
            stats["skipped_existing"] += 1
            continue

        if DRY_RUN:
            print(f"[DRY_RUN] would project: tp={tp} parsed_label_uid={parsed_label_uid} graph_label_uid={graph_label_uid}")
            print(f"          seg  : {seg_path}")
            print(f"          mesh : {mesh_path}")
            print(f"          graph: {gpath}")
            print(f"          out  : {out_path}")
            n_done += 1
            if MAX_MESHES is not None and n_done >= int(MAX_MESHES):
                break
            continue

        # load existing graph if present; otherwise build it
        G = None
        if os.path.exists(gpath):
            try:
                G = load_cell_graph(gpath)
                stats["graphs_loaded"] += 1
                if VERBOSE:
                    print(f"[graph-proj] loaded existing graph for {tp}/{graph_label_uid}")
            except Exception as e:
                if VERBOSE:
                    print(f"[warn] failed loading graph {gpath}: {e}")
                G = None

        if G is None and BUILD_GRAPHS_IF_MISSING:
            try:
                G = build_graph_for_organoid(mesh_path, graph_label_uid, extractor)
                stats["graphs_built_on_the_fly"] += 1
                if VERBOSE:
                    print(f"[graph-proj] built graph on the fly for {tp}/{graph_label_uid}")
            except Exception as e:
                if VERBOSE:
                    print(f"[{tp}] graph build failed for {graph_label_uid}: {e}")
                stats["failed"] += 1
                continue

            if SAVE_BUILT_GRAPHS:
                try:
                    os.makedirs(os.path.dirname(gpath), exist_ok=True)
                    save_cell_graph(gpath, G)
                except Exception as e:
                    if VERBOSE:
                        print(f"[warn] could not save built graph for {tp}/{graph_label_uid}: {e}")
        
        # --- skip if still missing ---
        if G is None:
            if VERBOSE:
                print(f"[skip] Graph missing for {tp}/{mesh_seg_label_uid} (parsed={parsed_label_uid}, resolved={gpath})")
            stats["skipped_missing_graph"] += 1
            continue

        # project mesh patches -> graph patches (actual node ids)
        try:
            graph_patches, info = assign_mesh_patches_to_graph(
                G,
                crypts_mesh,
                proj_field="proj_vertex",
                drop_empty=True,
                return_node_ids=True,
            )
        except Exception as e:
            if VERBOSE:
                print(f"[{tp}] graph projection failed for {graph_label_uid}: {e}")
            stats["failed"] += 1
            continue

        # optional filter by number of cells
        graph_patches, keep_idx_graph = filter_graph_patches_by_min_cells(
            graph_patches,
            MIN_CELLS_PER_CRYPT,
        )

        # realign mesh-seg quantities to the surviving projected crypts
        seg = subset_per_crypt_seg_vars(seg, keep_idx_graph)

        # project per-crypt mesh distance fields to graph nodes
        if "d_crypts" in seg:
            d_crypts_graph = project_mesh_field_to_graph(G, seg["d_crypts"], proj_field="proj_vertex")
        else:
            d_crypts_graph = None
            if VERBOSE:
                print(f"[warn] 'd_crypts' not in segmentation {seg_path}")

        if "curvature_gauss" in seg:
            curvature_gauss_graph = project_mesh_field_to_graph(G, seg["curvature_gauss"], proj_field="proj_vertex")
        else:
            curvature_gauss_graph = None
            if VERBOSE:
                print(f"[warn] 'curvature_gauss' not in segmentation {seg_path}")
        
        # recompute graph patch sizes after filtering
        graph_patch_sizes = np.array([len(p) for p in graph_patches], dtype=np.int64)

        save_dict = {
            "label_uid": str(graph_label_uid),
            "graph_label_uid": str(graph_label_uid),
            "mesh_seg_label_uid": str(mesh_seg_label_uid),
            "parsed_label_uid": str(parsed_label_uid),
            "timepoint": str(tp),
            "mesh_seg_path": str(seg_path),
            "mesh_path": str(mesh_path),
            "graph_path": str(gpath),
            "crypts_ll": np.array(patches_to_ll(graph_patches), dtype=object),
            "d_crypts_graph": d_crypts_graph,
            "curvature_gauss_graph": curvature_gauss_graph,
            "n_crypts": int(len(graph_patches)),
            "keep_idx_graph": np.asarray(keep_idx_graph, dtype=np.int64),
            "mesh_to_graph_index": remap_mesh_to_graph_index(info["mesh_to_graph_index"], keep_idx_graph,),
            "graph_patch_sizes": graph_patch_sizes,
            "mesh_patch_sizes": np.asarray(info["mesh_patch_sizes"], dtype=np.int64),
        }

        # carry through selected mesh-seg variables, already re-aligned above
        for k in (
            "bottom_vertex_ids",
            "L_crypts",
            "circumference_crypts",
            "d_discretized",
            "crypt_constrictions",
            "crypt_elongations",
        ):
            if k in seg:
                save_dict[k] = seg[k]


        if SAVE_CRYPT_INDEX_VECTOR:
            save_dict["crypt_index_by_node"] = make_crypt_index_by_node(G, graph_patches)

        np.savez_compressed(out_path, **save_dict)

        if VERBOSE:
            print(f"[graph-proj] saved {tp}/{graph_label_uid} -> {out_path} (n_crypts={len(graph_patches)})")
        stats["projected_saved"] += 1

        n_done += 1
        if MAX_MESHES is not None and n_done >= int(MAX_MESHES):
            break

    elapsed_s = time.perf_counter() - t_start
    write_map_run_settings(elapsed_s=elapsed_s, stats=stats)
    if VERBOSE:
        print(f"[graph-proj] done. processed={n_done} DRY_RUN={DRY_RUN} elapsed={elapsed_s:.2f}s ({elapsed_s/60.0:.2f} min)")


def write_map_run_settings(*, elapsed_s, stats):
    timepoints = resolve_timepoints(
        TIMEPOINTS,
        ZARR_NAME_BY_TP,
        ROUND_BY_TP,
        MESHNAME_BY_TP,
    )
    write_run_settings(
        GRAPH_SEG_DIR,
        script_name=os.path.basename(__file__),
        payload={
            "dataset": DATASET,
            "timepoints": timepoints,
            "paths": {
                "dataset_root": DATASET_ROOT,
                "mesh_data_dir": MESH_DATA_DIR,
                "seg_mesh_dir": SEG_MESH_DIR,
                "cells_csv": CELLS_CSV,
                "graphs_dir": GRAPHS_DIR,
                "graph_seg_dir": GRAPH_SEG_DIR,
                "mesh_config_path": MESH_CONFIG_PATH,
                "cell_config_path": CELL_CONFIG_PATH,
                "blacklist_path": BLACKLIST_PATH,
            },
            "parameters": {
                "overwrite": OVERWRITE,
                "dry_run": DRY_RUN,
                "max_meshes": MAX_MESHES,
                "min_cells_per_crypt": MIN_CELLS_PER_CRYPT,
                "save_crypt_index_vector": SAVE_CRYPT_INDEX_VECTOR,
                "build_graphs_if_missing": BUILD_GRAPHS_IF_MISSING,
                "save_built_graphs": SAVE_BUILT_GRAPHS,
            },
            "marker_fields": ["markers_int", "markers_bin"],
            "marker_positive_rule": "markers_bin = markers_int > 0",
            "postprocessing_functions": {
                "enabled": enabled_marker_postprocessing_functions(),
                "store_raw_graph_markers": STORE_RAW_GRAPH_MARKERS,
                "enable_lysozyme_agr2_ablation": ENABLE_LYSOZYME_AGR2_ABLATION,
                "enable_mucin2_agr2_ablation": ENABLE_MUCIN2_AGR2_ABLATION,
                "mucin2_marker": MUCIN2_MARKER,
                "enable_graph_coexpression_suppression": ENABLE_GRAPH_COEXPRESSION_SUPPRESSION,
            },
            "stats": stats,
            "elapsed_s": float(elapsed_s),
        },
        verbose=VERBOSE,
    )



if __name__ == "__main__":
    main()

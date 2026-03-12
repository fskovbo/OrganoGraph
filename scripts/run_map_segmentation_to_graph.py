#!/usr/bin/env python3
"""
Project mesh-based crypt segmentations onto organoid cell graphs.

Behavior
--------
For each mesh-based crypt segmentation (.npz):
1. Load the saved crypt patches on the mesh.
2. Try to load the corresponding precomputed cell graph.
3. If the graph does not exist, build it on the fly from:
      - the organoid mesh
      - the nuclei/cell CSV
4. Project mesh crypt patches to graph nodes using each node's "proj_vertex".
5. Optionally filter projected crypts by a minimum number of cells.
6. Save the projected graph crypt patches as graph_crypts_ll (list of node-id lists).

Primary storage choice
----------------------
By default, projected crypts are stored as:

    graph_crypts_ll : list[list[int]]

where each inner list contains the graph node ids belonging to one crypt.

Optionally, a per-node crypt index vector can also be saved:

    crypt_index_by_node : (N_nodes,) int
        -1 means "not in a crypt", otherwise the crypt index.

This vector is useful for fast node-level lookup, but the patch list remains the
more convenient primary representation for retrieving whole crypts.
"""

import os
import glob
import time
import numpy as np
import pandas as pd

from organograph.mesh.OrganoidMesh import OrganoidMesh

from organograph.io_utils.cells_table import (
    prepare_cells_table,
    make_nuclei_extractor,
    suppress_marker_if_coexpressed,
)
from organograph.io_utils.dataset_config import load_mesh_dataset_config, load_cell_table_config
from organograph.graph.build import build_organoid_graph, assign_mesh_patches_to_graph
from organograph.io_utils.segmentation_io import load_mesh_crypt_segmentation
from organograph.graph.io import load_cell_graph, save_cell_graph
from organograph.graph.access import graph_get

# =============================================================================
# DATASET PATHS (EDIT THESE)
# =============================================================================

DATASET         = "20250929" # "20251201" 

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(_SCRIPT_DIR)

# Input: mesh-based segmentation results
SEG_MESH_DIR    = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "crypt_segmentations_mesh")

# Input: nuclei/cell table used to build graphs if needed
CELLS_CSV       = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "cell_types_class.csv") # cell_types_class # cell_features_class

# Optional existing graph directory; if graph missing here, it will be built on the fly
GRAPHS_DIR      = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "graphs_preprocessed")

# Output: graph-based crypt projections
GRAPH_SEG_DIR   = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "crypt_segmentations_graph")

# config files with data structure
MESH_CONFIG_PATH= os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "mesh_config.json")
CELL_CONFIG_PATH= os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "cell_table_config.json")



# =============================================================================
# OPTIONAL FILTERING / BEHAVIOR
# =============================================================================

TIMEPOINTS = ['day3p5', 'day4', 'day4p5', 'day4p5-more']   # or None for all timepoints found under SEG_MESH_DIR

OVERWRITE = True
VERBOSE = True
DRY_RUN = False
MAX_ORGANOIDS = None

# Minimum number of graph nodes (cells) for a projected crypt to be kept
MIN_CELLS_PER_CRYPT = 10   # e.g. 5, or None to disable

# Whether to save a per-node crypt index vector in addition to graph_crypts_ll
SAVE_CRYPT_INDEX_VECTOR = False

# If a graph is missing and we build it on the fly, save it to GRAPHS_DIR
BUILD_GRAPHS_IF_MISSING = False
SAVE_BUILT_GRAPHS = False


# =============================================================================
# LOAD CONFIG
# =============================================================================

mesh_cfg = load_mesh_dataset_config(MESH_CONFIG_PATH)
cell_cfg = load_cell_table_config(CELL_CONFIG_PATH)


COORD_COLS = tuple(cell_cfg["coord_cols"])
MARKER_COLS = list(cell_cfg["marker_cols"])
LGR5_MARKER = cell_cfg["lgr5_marker"]
COEXP_MARKERS = tuple(cell_cfg["coexp_markers"])


# =============================================================================
# HELPERS
# =============================================================================

def marker_postprocess(markers_bin, marker_names):
    return suppress_marker_if_coexpressed(
        markers_bin,
        marker_names,
        exclusive_marker=LGR5_MARKER,
        forbidden_markers=COEXP_MARKERS,
        copy=True,
        ignore_missing=True,
    )


def patches_to_ll(patches):
    """list[set[int]] -> list[list[int]] for npz saving."""
    return [sorted(list(p)) for p in (patches or [])]


def graph_path_for(tp, label_uid):
    return os.path.join(GRAPHS_DIR, tp, f"{label_uid}.gpickle")


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
    return G


def project_d_crypts_to_graph(G, d_crypts_mesh, *, proj_field="proj_vertex"):
    """
    Project per-crypt mesh distances to graph nodes via each node's projected mesh vertex.

    Returns
    -------
    d_crypts_graph : ndarray, shape (K, N_nodes)
        Per-crypt distances evaluated at each graph node's projected mesh vertex.
        Nodes with invalid proj_vertex get NaN.
    """
    D = np.asarray(d_crypts_mesh, dtype=float)
    if D.ndim != 2:
        raise ValueError(f"d_crypts_mesh must have shape (K, V_mesh), got {D.shape}")

    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        raise ValueError("Graph has no nodes")

    proj_vertex = graph_get(G, proj_field, dtype=np.int64)   # (N_nodes,)
    if proj_vertex.ndim != 1 or proj_vertex.size != n_nodes:
        raise ValueError("proj_vertex must be a 1D array of length N_nodes")

    valid = (proj_vertex >= 0) & (proj_vertex < D.shape[1])

    out = np.full((D.shape[0], n_nodes), np.nan, dtype=float)
    if np.any(valid):
        out[:, valid] = D[:, proj_vertex[valid]]

    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.perf_counter()

    if not os.path.exists(CELLS_CSV):
        raise FileNotFoundError(f"CELLS_CSV not found: {CELLS_CSV}")

    # load + prepare nuclei table once
    cells_df = pd.read_csv(CELLS_CSV)
    cells_df = prepare_cells_table(cells_df, label_col="label_uid")

    extractor = make_nuclei_extractor(
        cells_df,
        label_col="label_uid",
        xyz_cols=COORD_COLS,
        marker_cols=MARKER_COLS,
        marker_postprocess_fn=marker_postprocess,
    )

    # discover segmentation files
    if TIMEPOINTS is None:
        seg_paths = sorted(glob.glob(os.path.join(SEG_MESH_DIR, "*", "*.npz")))
    else:
        seg_paths = []
        for tp in TIMEPOINTS:
            seg_paths.extend(sorted(glob.glob(os.path.join(SEG_MESH_DIR, tp, "*.npz"))))

    if VERBOSE:
        print(f"[graph-proj] found {len(seg_paths)} mesh segmentations")

    n_done = 0

    for seg_path in seg_paths:
        try:
            seg = load_mesh_crypt_segmentation(seg_path)
        except Exception as e:
            if VERBOSE:
                print(f"[skip] failed loading segmentation {seg_path}: {e}")
            continue

        tp = seg["timepoint"]
        label_uid = seg["label_uid"]
        mesh_path = seg["mesh_path"]
        crypts_mesh = seg["crypts_mesh"]


        if not tp or not label_uid or not mesh_path:
            if VERBOSE:
                print(f"[skip] missing timepoint/label_uid/mesh_path in {seg_path}")
            continue

        out_path = output_path_for(tp, label_uid)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if (not OVERWRITE) and os.path.exists(out_path):
            if VERBOSE:
                print(f"[skip] exists: {out_path}")
            continue

        gpath = graph_path_for(tp, label_uid)

        if DRY_RUN:
            print(f"[DRY_RUN] would project: tp={tp} label_uid={label_uid}")
            print(f"          seg  : {seg_path}")
            print(f"          mesh : {mesh_path}")
            print(f"          graph: {gpath}")
            print(f"          out  : {out_path}")
            n_done += 1
            if MAX_ORGANOIDS is not None and n_done >= int(MAX_ORGANOIDS):
                break
            continue

        # load existing graph if present; otherwise build it
        G = None
        if os.path.exists(gpath):
            try:
                G = load_cell_graph(gpath)
                if VERBOSE:
                    print(f"[graph-proj] loaded existing graph for {tp}/{label_uid}")
            except Exception as e:
                if VERBOSE:
                    print(f"[warn] failed loading graph {gpath}: {e}")
                G = None

        if G is None and BUILD_GRAPHS_IF_MISSING:
            try:
                G = build_graph_for_organoid(mesh_path, label_uid, extractor)
                if VERBOSE:
                    print(f"[graph-proj] built graph on the fly for {tp}/{label_uid}")
            except Exception as e:
                if VERBOSE:
                    print(f"[{tp}] graph build failed for {label_uid}: {e}")
                continue

            if SAVE_BUILT_GRAPHS:
                try:
                    os.makedirs(os.path.dirname(gpath), exist_ok=True)
                    save_cell_graph(gpath, G)
                except Exception as e:
                    if VERBOSE:
                        print(f"[warn] could not save built graph for {tp}/{label_uid}: {e}")
        
        # --- skip if still missing ---
        if G is None:
            if VERBOSE:
                print(f"[skip] Graph missing for {tp}/{label_uid}")
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
                print(f"[{tp}] graph projection failed for {label_uid}: {e}")
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
            d_crypts_graph = project_d_crypts_to_graph(G, seg["d_crypts"], proj_field="proj_vertex")
        else:
            d_crypts_graph = None
        
        # recompute graph patch sizes after filtering
        graph_patch_sizes = np.array([len(p) for p in graph_patches], dtype=np.int64)

        save_dict = {
            "label_uid": str(label_uid),
            "timepoint": str(tp),
            "mesh_seg_path": str(seg_path),
            "mesh_path": str(mesh_path),
            "graph_path": str(gpath),
            "crypts_ll": np.array(patches_to_ll(graph_patches), dtype=object),
            "d_crypts_graph": d_crypts_graph,
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
            print(f"[graph-proj] saved {tp}/{label_uid} -> {out_path} (n_crypts={len(graph_patches)})")

        n_done += 1
        if MAX_ORGANOIDS is not None and n_done >= int(MAX_ORGANOIDS):
            break

    elapsed_s = time.perf_counter() - t_start
    if VERBOSE:
        print(f"[graph-proj] done. processed={n_done} DRY_RUN={DRY_RUN} elapsed={elapsed_s:.2f}s ({elapsed_s/60.0:.2f} min)")



if __name__ == "__main__":
    main()
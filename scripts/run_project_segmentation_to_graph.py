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
from organograph.io_utils.blacklist import load_blacklist
from organograph.graph.build import build_organoid_graph, assign_mesh_patches_to_graph
from organograph.io_utils.segmentation_io import load_mesh_crypt_segmentation
from organograph.graph.io import load_cell_graph, save_cell_graph


# =============================================================================
# DATASET PATHS (EDIT THESE)
# =============================================================================

DATASET         = "20250929"

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(_SCRIPT_DIR)

# Input: mesh-based segmentation results
SEG_MESH_DIR    = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "crypt_segmentations_mesh")

# Input: nuclei/cell table used to build graphs if needed
CELLS_CSV       = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "cell_types_class.csv")

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

TIMEPOINTS = None   # or None for all timepoints found under SEG_MESH_DIR
OVERWRITE = True
VERBOSE = True
DRY_RUN = False
MAX_ORGANOIDS = None

# Minimum number of graph nodes (cells) for a projected crypt to be kept
MIN_CELLS_PER_CRYPT = 10   # e.g. 5, or None to disable

# Whether to save a per-node crypt index vector in addition to graph_crypts_ll
SAVE_CRYPT_INDEX_VECTOR = False

# If a graph is missing and we build it on the fly, save it to GRAPHS_DIR
SAVE_BUILT_GRAPHS = True


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
    if min_cells_per_crypt is None:
        return graph_patches
    return [p for p in graph_patches if len(p) >= int(min_cells_per_crypt)]


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

        if G is None:
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
        graph_patches = filter_graph_patches_by_min_cells(graph_patches, MIN_CELLS_PER_CRYPT)

        # save
        save_dict = {
            "label_uid": str(label_uid),
            "timepoint": str(tp),
            "mesh_seg_path": str(seg_path),
            "mesh_path": str(mesh_path),
            "graph_path": str(gpath),
            "crypts_ll": np.array(patches_to_ll(graph_patches), dtype=object),
            "n_crypts": int(len(graph_patches)),
            "mesh_to_graph_index": np.asarray(info["mesh_to_graph_index"], dtype=np.int64),
            "graph_patch_sizes": np.asarray(info["graph_patch_sizes"], dtype=np.int64),
            "mesh_patch_sizes": np.asarray(info["mesh_patch_sizes"], dtype=np.int64),
        }

        # carry through selected mesh-seg variables if present
        for k in ("bottom_vertex_ids", "L_crypts", "circumference_crypts", "d_discretized"):
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
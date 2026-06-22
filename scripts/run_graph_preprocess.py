#!/usr/bin/env python3
"""
Preprocessing script: build and save cell graphs from organoid meshes.

Expected folder structure
-------------------------
This script expects meshes at:

    {data_dir}/{timepoint}/{zarr_name}/{well_letter}/{well_field}/{round_name}/meshes/{mesh_name}/{organoid_id}.vtp

The only "dataset-specific" code is in io_utils/path_parsing.py:
  - discover_mesh_paths(...)
  - parse_mesh_path(mesh_path)

Outputs
-------
Graphs:
    {OUT_GRAPHS_DIR}/{timepoint}/{label_uid}.gpickle

Index:
    {OUT_GRAPHS_DIR}/{timepoint}/index.csv
"""

import os
import csv
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import organograph
from organograph.mesh.OrganoidMesh import OrganoidMesh

from organograph.io_utils.cells_table import prepare_cells_table, make_nuclei_extractor, suppress_marker_if_coexpressed, enforce_marker_exclusivity, harmonize_markers
from organograph.io_utils.path_parsing import parse_mesh_path, discover_mesh_paths
from organograph.io_utils.dataset_config import load_mesh_dataset_config, load_cell_table_config
from organograph.io_utils.blacklist import default_discard_labels_path, load_optional_blacklist
from organograph.io_utils.run_metadata import write_run_settings
from organograph.graph.build import build_organoid_graph, add_vertex_field_to_graph
from organograph.graph.io import save_cell_graph
from organograph.graph.marker_postprocess import (
    ablate_lysozyme_not_agr2_in_clusters,
    copy_graph_marker_fields,
    suppress_graph_marker_if_coexpressed,
)

from organograph.mesh.hks import compute_hks
from organograph.crypts.vocab import compute_vocabulary_encoding


# =============================================================================
# DATASET CONFIG
# =============================================================================

DATASET         = "20250929" # "20251201" 20250929

# Absolute path to this script file
_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))

# Project root = parent of the "scripts" folder
PROJECT_ROOT    = os.path.dirname(_SCRIPT_DIR)

MESH_DATA_DIR   = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "fractal_output")
CELLS_CSV       = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "feature_tables", "cell_features_class.csv") # cell_types_class # cell_features_class
OUT_GRAPHS_DIR  = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "graphs_preprocessed")
DATASET_ROOT    = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET)

MESH_CONFIG_PATH= os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "mesh_config.json")
CELL_CONFIG_PATH= os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "cell_table_config.json")
BLACKLIST_PATH  = default_discard_labels_path(DATASET_ROOT)


# Optional override. 
timepoints = None  


MAX_PROJ_DIST = 2.0  # max accepted distance between nuclei and membrane for projection. If None, use all distances


# Dev/UX options
OVERWRITE = True
VERBOSE = True
DRY_RUN = False       # If True: do not load meshes, do not write outputs; just print what would happen
MAX_MESHES = None     # e.g. 10 for quick testing; None means no limit


# =============================================================================
# LOAD DATA STRUCTURE 
# =============================================================================

mesh_cfg    = load_mesh_dataset_config(MESH_CONFIG_PATH)
zarr_names  = mesh_cfg["zarr_name_by_tp"]
rounds      = mesh_cfg["round_by_tp"]
meshes      = mesh_cfg["meshname_by_tp"]
wells       = mesh_cfg["wells_by_tp"]



cell_cfg    = load_cell_table_config(CELL_CONFIG_PATH)
COORD_COLS  = tuple(cell_cfg["coord_cols"])
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

# Graph-level marker postprocessing. These steps run after graph construction,
# so they can use graph adjacency before co-expression suppression is applied.
STORE_RAW_GRAPH_MARKERS = True  # Preserve original per-node markers in markers_int_raw and markers_bin_raw.
ENABLE_LYSOZYME_AGR2_ABLATION = True  # Remove Lysozyme from clustered Lysozyme+ cells unless they are Agr2+.
LYSOZYME_MARKER = marker_config_name_to_alias("Lysozyme")
AGR2_MARKER = marker_config_name_to_alias("Agr2")
LYSOZYME_ABLATION_MIN_CLUSTER_SIZE = 2  # Only process connected Lysozyme+ components with at least this many cells.
ENABLE_MUCIN2_AGR2_ABLATION = True  # Remove Mucin 2 from clustered Mucin 2+ cells unless they are Agr2+.
MUCIN2_MARKER = optional_marker_config_name_to_alias("Mucin 2")
MUCIN2_ABLATION_MIN_CLUSTER_SIZE = 2  # Only process connected Mucin 2+ components with at least this many cells.
ENABLE_GRAPH_COEXPRESSION_SUPPRESSION = True  # Apply the LGR5-vs-forbidden-marker rule after cluster cleanup.


# EXCLUSIVITY_RULES = {
#     "LGR5":     ["Chroma", "Mucin 2", "AldoB", "Glucagon", "Agr2", "Serotonin", "Lysozyme"],
#     "Chroma":   ["Mucin 2", "Glucagon", "Serotonin", "Lysozyme"],
#     "Mucin 2":  ["Chroma", "Glucagon", "Serotonin", "Lysozyme"],
#     "AldoB":    ["Chroma", "Mucin 2", "Glucagon", "Agr2", "Serotonin", "Lysozyme"],
#     "Glucagon": ["Serotonin"],
#     "Agr2":     ["Chroma", "Mucin 2", "Glucagon", "Serotonin", "Lysozyme"],
#     "Serotonin":[],
#     "Lysozyme": ["Chroma", "Glucagon", "Serotonin"],
#     "Cyclin D": ["LGR5", "Chroma", "Mucin 2", "AldoB", "Glucagon", "Agr2", "Serotonin", "Lysozyme"],
#     "Cyclin A": ["LGR5", "Chroma", "Mucin 2", "AldoB", "Glucagon", "Agr2", "Serotonin", "Lysozyme"],
#     "KI67":     ["LGR5", "Chroma", "Mucin 2", "AldoB", "Glucagon", "Agr2", "Serotonin", "Lysozyme"],
# }

# HARMONIZATION_RULES = {
#     "TA": ["Cyclin A", "Cyclin D", "KI67"],
# }


# =============================================================================
# MARKER POSTPROCESS
# =============================================================================

def resolve_timepoints(timepoints_override, zarr_names, rounds, meshes):
    """Return the configured timepoints unless a manual override is supplied."""
    if timepoints_override is not None:
        return list(timepoints_override)
    return [tp for tp in zarr_names if tp in rounds and tp in meshes]


def marker_postprocess(markers_bin, marker_names):
    """Keep the raw binarized marker calls unchanged."""
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

# def marker_postprocess(markers_bin, marker_names):
#     return suppress_marker_if_coexpressed(
#         markers_bin,
#         marker_names,
#         exclusive_marker=LGR5_MARKER,
#         forbidden_markers=COEXP_MARKERS,
#         copy=True,
#         ignore_missing=False,
#     )

# def marker_postprocess(markers_bin, marker_names):
#     # Step 1: enforce exclusivity on the original marker space
#     markers_bin = enforce_marker_exclusivity(
#         markers_bin,
#         marker_names,
#         exclusivity_rules=EXCLUSIVITY_RULES,
#         copy=True,
#         ignore_missing=True,
#     )

#     # Step 2: harmonize markers (this may change both matrix and names)
#     markers_bin, marker_names = harmonize_markers(
#         markers_bin,
#         marker_names,
#         marker_rules=HARMONIZATION_RULES,
#         keep_unmapped=True,
#     )

#     return markers_bin, marker_names


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.perf_counter()
    stats = build_graphs_for_dataset(
        overwrite=OVERWRITE,
        verbose=VERBOSE,
        blacklist_path=BLACKLIST_PATH,
    )
    elapsed_s = time.perf_counter() - t_start
    write_graph_run_settings(elapsed_s=elapsed_s, stats=stats)
    if VERBOSE:
        print(f"[graphs] done. DRY_RUN={DRY_RUN} elapsed={elapsed_s:.2f}s ({elapsed_s/60.0:.2f} min)")


def build_graphs_for_dataset(overwrite=False, verbose=True, blacklist_path=None):
    blacklist = load_optional_blacklist(blacklist_path, label="blacklist", verbose=verbose)
    search_timepoints = resolve_timepoints(timepoints, zarr_names, rounds, meshes)
    tp_allow = set(search_timepoints)
    stats = {
        "mesh_files_found": 0,
        "planned_or_done": 0,
        "graphs_saved": 0,
        "vertex_owner_sidecars_saved": 0,
        "skipped_blacklist": 0,
        "skipped_existing": 0,
        "failed": 0,
        "dry_run": bool(DRY_RUN),
    }

    # --- load & index cells table once ---
    if not os.path.exists(CELLS_CSV):
        raise FileNotFoundError(f"CELLS_CSV not found: {CELLS_CSV}")

    cells_df = pd.read_csv(CELLS_CSV)
    cells_df = prepare_cells_table(cells_df, label_col="label_uid")

    # --- build extractor once: extractor(label_uid)->(xyz_raw, markers_bin, marker_names) ---
    extractor = make_nuclei_extractor(
        cells_df,
        label_col="label_uid",
        xyz_cols=COORD_COLS,
        marker_cols=MARKER_COLS,
        marker_alias=MARKER_ALIAS,
        marker_postprocess_fn=marker_postprocess,
        return_marker_intensity=True,
    )

    # --- discover mesh paths (restrictive glob based on config) ---
    mesh_paths = discover_mesh_paths(
        data_dir=MESH_DATA_DIR,
        timepoints=search_timepoints,
        zarr_names=zarr_names,
        rounds=rounds,
        meshes=meshes,
        wells=wells,
    )

    if verbose:
        print(f"[graphs] searching timepoints: {search_timepoints}")
        print(f"[graphs] found {len(mesh_paths)} mesh files (pre-filter)")
    stats["mesh_files_found"] = int(len(mesh_paths))

    index_rows = {}  # timepoint -> list of dicts
    it = tqdm(mesh_paths, desc="build graphs") if verbose else mesh_paths

    n_planned_or_done = 0

    for mesh_path in it:
        # ---- parse identifiers ----
        try:
            rec = parse_mesh_path(mesh_path)
        except Exception as e:
            if verbose:
                print(f"[skip] cannot parse mesh path: {mesh_path} ({e})")
            stats["failed"] += 1
            continue

        tp = rec.get("timepoint", None)
        label_uid = rec.get("label_uid", None)
        well = rec.get("well", None)

        # Robustness: timepoint is REQUIRED for output layout + indexing
        if tp is None or tp == "":
            if verbose:
                print(f"[skip] parse_mesh_path did not return timepoint for: {mesh_path}")
            continue

        if label_uid is None or label_uid == "":
            if verbose:
                print(f"[skip] parse_mesh_path did not return label_uid for: {mesh_path}")
            continue

        if tp_allow is not None and tp not in tp_allow:
            continue

        if label_uid in blacklist:
            if verbose:
                print(f"[skip] {label_uid} is blacklisted")
            stats["skipped_blacklist"] += 1
            continue

        out_dir = os.path.join(OUT_GRAPHS_DIR, tp)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{label_uid}.gpickle")
        vertex_owner_path = os.path.join(out_dir, f"{label_uid}.vertex_owner.npz")

        if (not overwrite) and os.path.exists(out_path):
            if verbose:
                print(f"[skip] exists: {out_path}")
            stats["skipped_existing"] += 1
            continue

        # DRY_RUN / MAX_MESHES logic (after all filters)
        n_planned_or_done += 1
        stats["planned_or_done"] = int(n_planned_or_done)
        if DRY_RUN:
            if verbose:
                print(f"[dry-run] would build: tp={tp} well={well} label_uid={label_uid} -> {out_path}")
            if MAX_MESHES is not None and n_planned_or_done >= int(MAX_MESHES):
                break
            continue

        # --- load mesh ---
        try:
            mesh = OrganoidMesh(mesh_path)
        except Exception as e:
            if verbose:
                print(f"[{tp}] mesh load failed: {mesh_path} ({e})")
            stats["failed"] += 1
            continue

        # --- normalize mesh coordinates ---
        mesh.normalize_inplace()
        mesh.label_uid = label_uid

        # --- build graph (extractor does the table lookup) ---
        try:
            G, aux = build_organoid_graph(mesh=mesh, extract_fn=extractor, max_dist=MAX_PROJ_DIST)
        except Exception as e:
            if verbose:
                print(f"[{tp}] graph build failed for {label_uid}: {e}")
            stats["failed"] += 1
            continue

        try:
            G = graph_marker_postprocess(G)
        except Exception as e:
            if verbose:
                print(f"[{tp}] graph marker postprocess failed for {label_uid}: {e}")
            stats["failed"] += 1
            continue

        # --- save graph + index ---
        G.graph["mesh_path"] = str(mesh_path)
        G.graph["vertex_owner_path"] = str(vertex_owner_path)
        save_cell_graph(out_path, G)
        stats["graphs_saved"] += 1

        np.savez_compressed(
            vertex_owner_path,
            label_uid=str(label_uid),
            timepoint=str(tp),
            mesh_path=str(mesh_path),
            graph_path=str(out_path),
            vertex_owner=np.asarray(aux["vertex_owner"], dtype=np.int64),
            proj_vertex_ids=np.asarray(aux["proj_vertex_ids"], dtype=np.int64),
        )
        stats["vertex_owner_sidecars_saved"] += 1
        index_rows.setdefault(tp, []).append(
            {
                "label_uid": label_uid,
                "well": well,
                "mesh_path": mesh_path,
                "graph_path": out_path,
                "vertex_owner_path": vertex_owner_path,
                "N_cells": int(G.number_of_nodes()),
                "N_edges": int(G.number_of_edges()),
                "max_proj_dist": MAX_PROJ_DIST,
            }
        )

        if MAX_MESHES is not None and n_planned_or_done >= int(MAX_MESHES):
            break

    # --- write index.csv per timepoint ---
    for tp, rows in index_rows.items():
        if not rows:
            continue
        idx_path = os.path.join(OUT_GRAPHS_DIR, tp, "index.csv")
        with open(idx_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        if verbose:
            print(f"[graphs] wrote {idx_path} ({len(rows)} rows)")

    return stats


def write_graph_run_settings(*, elapsed_s, stats):
    write_run_settings(
        OUT_GRAPHS_DIR,
        script_name=os.path.basename(__file__),
        payload={
            "dataset": DATASET,
            "timepoints": resolve_timepoints(timepoints, zarr_names, rounds, meshes),
            "paths": {
                "dataset_root": DATASET_ROOT,
                "mesh_data_dir": MESH_DATA_DIR,
                "cells_csv": CELLS_CSV,
                "out_graphs_dir": OUT_GRAPHS_DIR,
                "mesh_config_path": MESH_CONFIG_PATH,
                "cell_config_path": CELL_CONFIG_PATH,
                "blacklist_path": BLACKLIST_PATH,
            },
            "parameters": {
                "max_proj_dist": MAX_PROJ_DIST,
                "overwrite": OVERWRITE,
                "dry_run": DRY_RUN,
                "max_meshes": MAX_MESHES,
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
            "outputs": {
                "graph_pickle": "{OUT_GRAPHS_DIR}/{timepoint}/{label_uid}.gpickle",
                "vertex_owner_sidecar": "{OUT_GRAPHS_DIR}/{timepoint}/{label_uid}.vertex_owner.npz",
            },
            "stats": stats,
            "elapsed_s": float(elapsed_s),
        },
        verbose=VERBOSE,
    )


if __name__ == "__main__":
    main()

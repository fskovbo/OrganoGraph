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

from organograph.io_utils.cells_table import prepare_cells_table, make_nuclei_extractor, suppress_marker_if_coexpressed
from organograph.io_utils.path_parsing import parse_mesh_path, discover_mesh_paths
from organograph.io_utils.dataset_config import load_mesh_dataset_config, load_cell_table_config
from organograph.io_utils.blacklist import load_blacklist
from organograph.graph.build import build_organoid_graph, add_vertex_field_to_graph
from organograph.graph.io import save_cell_graph

from organograph.mesh.hks import compute_hks
from organograph.crypts.vocab import compute_vocabulary_encoding


# =============================================================================
# DATASET CONFIG
# =============================================================================

DATASET         = "20251201" # "20250929"

# Absolute path to this script file
_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))

# Project root = parent of the "scripts" folder
PROJECT_ROOT    = os.path.dirname(_SCRIPT_DIR)

MESH_DATA_DIR   = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "fractal_output")
CELLS_CSV       = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "cell_features_class.csv") # cell_types_class
OUT_GRAPHS_DIR  = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "graphs_preprocessed")

MESH_CONFIG_PATH= os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "mesh_config.json")
CELL_CONFIG_PATH= os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "cell_table_config.json")
BLACKLIST_PATH  = None # os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "combined_labels_to_discard.npy")


# Optional override. If None, use all timepoints from mesh_config.json
timepoints = ['day4p5'] # ['day3p5', 'day4', 'day4p5', 'day4p5-more']   


HKS_TIMES = [1.0, 2.0, 4.0, 8.0, 25.0]
MAX_PROJ_DIST = 2.5  # max accepted distance between nuclei and membrane for projection. If None, use all distances


VOCAB_PATH = "./sim/vocab_with_meta.npz"
# If you know specific arrays that MUST be present in vocab.npz, list them here.
REQUIRED_VOCAB_KEYS = []

# Dev/UX options
OVERWRITE = False
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
LGR5_MARKER = cell_cfg["lgr5_marker"]
COEXP_MARKERS = tuple(cell_cfg["coexp_markers"])

# =============================================================================
# MARKER POSTPROCESS
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


def validate_vocab_npz(vocab):
    # Basic sanity checks to fail early with a helpful message
    files = getattr(vocab, "files", None)
    if files is None or len(files) == 0:
        raise ValueError(
            f"Vocab file '{VOCAB_PATH}' loaded, but contained no arrays. "
            "Expected an .npz with at least one array."
        )

    missing = [k for k in REQUIRED_VOCAB_KEYS if k not in files]
    if missing:
        raise ValueError(
            f"Vocab file '{VOCAB_PATH}' is missing required keys: {missing}. "
            f"Available keys: {list(files)}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.perf_counter()
    build_graphs_for_dataset(
        overwrite=OVERWRITE,
        verbose=VERBOSE,
        blacklist_path=BLACKLIST_PATH,
    )
    elapsed_s = time.perf_counter() - t_start
    if VERBOSE:
        print(f"[graphs] done. DRY_RUN={DRY_RUN} elapsed={elapsed_s:.2f}s ({elapsed_s/60.0:.2f} min)")


def build_graphs_for_dataset(overwrite=False, verbose=True, blacklist_path=None):
    blacklist = load_blacklist(blacklist_path) if blacklist_path else set()
    tp_allow = set(timepoints) if timepoints else None

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
        marker_postprocess_fn=marker_postprocess,
    )

    # --- load + validate vocab once ---
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"VOCAB_PATH not found: {VOCAB_PATH}")

    vocab = np.load(VOCAB_PATH, allow_pickle=True)
    validate_vocab_npz(vocab)

    # --- discover mesh paths (restrictive glob based on config) ---
    mesh_paths = discover_mesh_paths(
        data_dir=MESH_DATA_DIR,
        timepoints=timepoints,
        zarr_names=zarr_names,
        rounds=rounds,
        meshes=meshes,
        wells=wells,
    )

    if verbose:
        print(f"[graphs] found {len(mesh_paths)} mesh files (pre-filter)")

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
            continue

        out_dir = os.path.join(OUT_GRAPHS_DIR, tp)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{label_uid}.gpickle")

        if (not overwrite) and os.path.exists(out_path):
            if verbose:
                print(f"[skip] exists: {out_path}")
            continue

        # DRY_RUN / MAX_MESHES logic (after all filters)
        n_planned_or_done += 1
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
            continue

        # --- compute + attach HKS (required) ---
        try:
            mesh._eig_decomp()  # Laplace eigenpairs (required for HKS)
            hks = compute_hks(mesh, t=HKS_TIMES, coeffs=False)  # (V, T)
            add_vertex_field_to_graph(G, hks, "hks")
            G.graph["hks_times"] = np.asarray(HKS_TIMES, float)
        except Exception as e:
            if verbose:
                print(f"[{tp}] HKS failed for {label_uid}: {e}")
            # Requirement: do not save graphs without HKS
            continue

        # --- compute + attach vocab encoding ---
        try:
            encoding = compute_vocabulary_encoding(vocab, mesh)[0]
            add_vertex_field_to_graph(G, encoding, "vocab_encoding")
            G.graph["vocab_path"] = VOCAB_PATH
        except Exception as e:
            if verbose:
                print(f"[{tp}] vocab encoding failed for {label_uid}: {e}")
            # Keep going: you may still want graphs with HKS only

        # --- save graph + index ---
        save_cell_graph(out_path, G)
        index_rows.setdefault(tp, []).append(
            {
                "label_uid": label_uid,
                "well": well,
                "mesh_path": mesh_path,
                "graph_path": out_path,
                "N_cells": int(G.number_of_nodes()),
                "N_edges": int(G.number_of_edges()),
                "has_hks": True,
                "max_proj_dist": MAX_PROJ_DIST,
                "has_vocab_encoding": ("vocab_encoding" in next(iter(G.nodes(data=True)))[1]) if G.number_of_nodes() > 0 else False,
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


if __name__ == "__main__":
    main()
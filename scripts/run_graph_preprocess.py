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

import numpy as np
import pandas as pd
from tqdm import tqdm

import organograph
from organograph.mesh.OrganoidMesh import OrganoidMesh

from organograph.io_utils.cells_table import prepare_cells_table, make_nuclei_extractor, suppress_marker_if_coexpressed
from organograph.io_utils.path_parsing import parse_mesh_path, discover_mesh_paths
from organograph.graph.build import build_organoid_graph, add_vertex_field_to_graph
from organograph.graph.io import save_cell_graph

from organograph.mesh.hks import compute_hks
from organograph.crypts.vocab import compute_vocabulary_encoding


# # =============================================================================
# # DATASET CONFIG
# # =============================================================================

# # Absolute path to this script file
# _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # Project root = parent of the "scripts" folder
# PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# MESH_DATA_DIR   = os.path.join(PROJECT_ROOT, "..", "NicoleData", "20251201", "fractal_output")
# CELLS_CSV       = os.path.join(PROJECT_ROOT, "..", "NicoleData", "20251201", "cell_features_class.csv") 
# OUT_GRAPHS_DIR  = os.path.join(PROJECT_ROOT, "..", "NicoleData", "20251201", "graphs_preprocessed")

# timepoints = ["day4p5"]
# zarr_names = {tp: "251130R0.zarr" for tp in timepoints}
# rounds     = {tp: "2_zillum_registered" for tp in timepoints}
# meshes     = {tp: "nnorg_corrected_annotated_by_projection" for tp in timepoints}

# wells = {
#     "day4p5": ["B02", "B03", "B04", "B05"],
# }


# COORD_COLS = ("0.x_pos_pix", "0.y_pos_pix", "0.z_pos_pix_scaled")

# MARKER_COLS = [
#     "0.C02.percentile99_class",  # LGR5
#     "0.C03.percentile99_class",  # chroma
#     "0.C04.percentile99_class",  # aldoB
#     "1.C02.percentile99_class",  # Sero
#     "1.C03.percentile99_class",  # Lyz
#     "1.C04.percentile99_class",  # Agr2
#     "2.C04.percentile99_class",  # ki67
# ]

# LGR5_MARKER = "0.C02.percentile99_class"
# COEXP_MARKERS = (
#     "1.C03.percentile99_class",  # Lyz
#     "1.C04.percentile99_class",  # Agr2
#     "1.C02.percentile99_class",  # Sero
#     "0.C03.percentile99_class",  # chroma
# )

# HKS_TIMES = [1.0, 2.0, 4.0, 8.0, 25.0]
# VOCAB_PATH = "./sim/vocab.npz"
# # If you know specific arrays that MUST be present in vocab.npz, list them here.
# REQUIRED_VOCAB_KEYS = []

# # Dev/UX options
# DRY_RUN = False       # If True: do not load meshes, do not write outputs; just print what would happen
# MAX_MESHES = None     # e.g. 10 for quick testing; None means no limit




# =============================================================================
# DATASET CONFIG
# =============================================================================

# Absolute path to this script file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root = parent of the "scripts" folder
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

MESH_DATA_DIR   = os.path.join(PROJECT_ROOT, "..", "NicoleData", "20250929", "fractal_output")
CELLS_CSV       = os.path.join(PROJECT_ROOT, "..", "NicoleData", "20250929", "cell_types_class.csv") 
OUT_GRAPHS_DIR  = os.path.join(PROJECT_ROOT, "..", "NicoleData", "20250929", "graphs_preprocessed")

# timepoints = ['day2', 'day2p5', 'day3', 'day3p5', 'day4', 'day4p5', 'day4p5-more']   # extend as needed: ['day3', 'day3p5', ...]
timepoints = ['day3p5',  'day4p5']   # extend as needed: ['day3', 'day3p5', ...]

zarr_names = {tp: 'r0.zarr' for tp in timepoints}
rounds     = {tp: '0_fused_zillum_registered' for tp in timepoints}
meshes     = {tp: 'nnorg_linked_multi_annotated_class' for tp in timepoints}

wells = {
    'day1p5': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06'],
    'day2': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06'],
    'day2p5': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06'],
    'day3': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'B02', 'B03'],
    'day3p5': ['A01', 'A02', 'A03', 'A04', 'B03'],
    'day4': ['A02', 'A03', 'A04', 'A05', 'A06', 'B01', 'B02'],
    'day4p5': ['A06', 'B06'],
    'day4p5-more': ['C01', 'C02', 'C03', 'C04', 'C05', 'C06'],
}




COORD_COLS = ("0.x_pos_pix", "0.y_pos_pix", "0.z_pos_pix_scaled")

MARKER_COLS = [
"LGR.bin", "CHROMA.bin", "CYCD.bin", "MUC.bin", "ALDOB.bin",
    "GLUC.bin", "CYCA.bin", "AGR.bin", "SERO.bin", "LYZ.bin",
]

LGR5_MARKER = "LGR.bin"
COEXP_MARKERS = (
    "LYZ.bin",     # Lysozyme → Paneth
    "MUC.bin",     # Mucin 2 → Goblet
    "AGR.bin",     # Agr2 → Goblet/Paneth
    "SERO.bin",    # Serotonin → Enterochromaffin
    "GLUC.bin",    # Glucagon → Enteroendocrine
    "CHROMA.bin",  # Chromogranin A → Enteroendocrine
)

HKS_TIMES = [1.0, 2.0, 4.0, 8.0, 25.0]
VOCAB_PATH = "./sim/vocab.npz"
# If you know specific arrays that MUST be present in vocab.npz, list them here.
REQUIRED_VOCAB_KEYS = []

# Dev/UX options
DRY_RUN = False       # If True: do not load meshes, do not write outputs; just print what would happen
MAX_MESHES = None     # e.g. 10 for quick testing; None means no limit


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
    build_graphs_for_dataset(
        overwrite=False,
        verbose=True,
        blacklist=[],
    )


def build_graphs_for_dataset(overwrite=False, verbose=True, blacklist=None):
    blacklist = set([] if blacklist is None else blacklist)
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
            continue

        out_dir = os.path.join(OUT_GRAPHS_DIR, tp)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{label_uid}.gpickle")

        if (not overwrite) and os.path.exists(out_path):
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
            G, aux = build_organoid_graph(mesh=mesh, extract_fn=extractor)
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
            encoding, _, _, _ = compute_vocabulary_encoding(vocab, mesh)
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
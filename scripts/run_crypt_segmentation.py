#!/usr/bin/env python3
"""
Segment crypts for a whole dataset of precomputed cell graphs.

Expected inputs
---------------
Graphs directory with per-timepoint subfolders:

    {GRAPHS_DIR}/{timepoint}/*.gpickle
    {GRAPHS_DIR}/{timepoint}/index.csv    (recommended; produced by graph preprocessing)

Each index.csv should at least contain:
    label_uid, mesh_path, graph_path

Outputs
-------
Segmentation files:

    {SEG_DIR}/{timepoint}/{label_uid}.npz

Optional compact features:

    {FEATURES_DIR}/{timepoint}/{label_uid}.npz


Quickstart
----------
1) Set GRAPHS_DIR, SEG_DIR, FEATURES_DIR.
2) Set TIMEPOINTS (or leave None to auto-discover from GRAPHS_DIR).
3) Set BIN_CENTERS and CRYPT_VOCAB_IDX (required).
4) Run:
       python run_crypt_segmentation.py

Notes
-----
- If index.csv is missing for a timepoint, we fall back to globbing .gpickle files,
  but then we may not know the mesh path. In that fallback mode, we REQUIRE that
  the graph has G.graph["mesh_path"] set, or we skip that graph with a clear message.
"""

import os
import glob
import csv

import numpy as np
from tqdm import tqdm

from organograph.mesh.OrganoidMesh import OrganoidMesh
from organograph.mesh.transform import ensure_mesh_graph_aligned
from organograph.graph.io import load_cell_graph
from organograph.graph.access import graph_get

from organograph.io_utils.segmentation_io import save_segmentation_npz, patches_to_ll
from organograph.io_utils.features_io import save_features_npz
from organograph.crypts.segment import segment_crypts_organoid


# =============================================================================
# CONFIG (edit these)
# =============================================================================

# Absolute path to this script file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root = parent of the "scripts" folder
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Where the graph preprocessing wrote graphs + per-timepoint index.csv
GRAPHS_DIR  = os.path.join(PROJECT_ROOT, "..", "NicoleData", "20251201", "graphs_preprocessed")

# Where to write segmentation outputs (.npz)
SEG_DIR  = os.path.join(PROJECT_ROOT, "..", "NicoleData", "20251201", "crypt_segmentations")

# Optional: where to write compact per-organoid arrays (.npz). Set to None to disable.
FEATURES_DIR = None

# Which timepoints to process. If None, auto-discover subfolders under GRAPHS_DIR.
TIMEPOINTS = ["day4p5"]

# Skip these label_uids
BLACKLIST = []

# Required: canonical axis used for circumference curves + rescaling
# (Fill in with your downstream canonical axis.)
S = 1.5
n_bins = 80
bin_edges   = np.linspace(0.0, S, n_bins + 1)
BIN_CENTERS = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Required: indices into node attribute "vocab_encoding" used for crypt seeding
CRYPT_VOCAB_IDX = [1, 2, 7]

# Optional: indices used for neck seeding (or None)
NECK_VOCAB_IDX = [0, 3, 4]

# Extra kwargs forwarded to segment_crypts_organoid(...)
SEGMENT_KWARGS = dict(
    # Example knobs (uncomment/tune as needed):
    # crypt_seed_thresh=0.2,
    # neck_seed_thresh=0.5,
    # min_crypt_seed_size=10,
    # grow_steps=2,
)

# Save compact features alongside segmentations?
SAVE_FEATURES = False

# Include extra debug info inside the segmentation npz? (Can be large.)
DEBUG = False

# Behavior knobs
OVERWRITE = False
VERBOSE = True
DRY_RUN = False
MAX_GRAPHS = None  # e.g. 10


# =============================================================================
# Helpers
# =============================================================================

def _discover_timepoints(graphs_dir):
    tps = []
    if not os.path.isdir(graphs_dir):
        return tps
    for name in sorted(os.listdir(graphs_dir)):
        p = os.path.join(graphs_dir, name)
        if os.path.isdir(p):
            # treat any subfolder as a timepoint if it contains .gpickle or index.csv
            if glob.glob(os.path.join(p, "*.gpickle")) or os.path.exists(os.path.join(p, "index.csv")):
                tps.append(name)
    return tps


def _iter_records_from_index_csv(graphs_dir, timepoint):
    """
    Yield dict records with at least: label_uid, mesh_path, graph_path.
    """
    idx_path = os.path.join(graphs_dir, timepoint, "index.csv")
    if not os.path.exists(idx_path):
        return None  # sentinel: caller will fall back

    rows = []
    with open(idx_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # normalize keys we rely on
            label_uid = row.get("label_uid", None)
            mesh_path = row.get("mesh_path", None)
            graph_path = row.get("graph_path", None)
            if not label_uid or not graph_path:
                continue
            rows.append({"timepoint": timepoint, "label_uid": label_uid, "mesh_path": mesh_path, "graph_path": graph_path})
    return rows


def _iter_records_fallback_glob(graphs_dir, timepoint):
    """
    Fallback if index.csv isn't available.
    We can find graph_path, but mesh_path must come from G.graph["mesh_path"].
    """
    gpaths = sorted(glob.glob(os.path.join(graphs_dir, timepoint, "*.gpickle")))
    rows = []
    for gpath in gpaths:
        rows.append({"timepoint": timepoint, "label_uid": None, "mesh_path": None, "graph_path": gpath})
    return rows


def _require_array(name, x):
    if x is None:
        raise ValueError(f"Provide {name} in the CONFIG section at the top of the file.")


# =============================================================================
# Main pipeline
# =============================================================================

def main():
    run_segmentation_dataset()


def run_segmentation_dataset():
    # --- validate config ---
    _require_array("BIN_CENTERS", BIN_CENTERS)
    _require_array("CRYPT_VOCAB_IDX", CRYPT_VOCAB_IDX)

    blacklist = set(BLACKLIST or [])
    bin_centers = np.asarray(BIN_CENTERS, float)

    if TIMEPOINTS is None:
        timepoints = _discover_timepoints(GRAPHS_DIR)
        if VERBOSE:
            print(f"[segment] discovered timepoints: {timepoints}")
    else:
        timepoints = list(TIMEPOINTS)

    # gather records
    recs = []
    for tp in timepoints:
        rows = _iter_records_from_index_csv(GRAPHS_DIR, tp)
        if rows is None:
            if VERBOSE:
                print(f"[segment] no index.csv for {tp}; falling back to globbing graphs")
            rows = _iter_records_fallback_glob(GRAPHS_DIR, tp)
        recs.extend(rows)

    if VERBOSE:
        print(f"[segment] found {len(recs)} graph records (pre-filter)")

    # iterate
    it = tqdm(recs, desc="segment") if VERBOSE else recs
    n_done = 0

    for rec in it:
        tp = rec["timepoint"]
        gpath = rec["graph_path"]

        # 1) load graph
        # 1) load graph (with fallback if index path is stale)
        G = None

        try:
            G = load_cell_graph(gpath)
        except Exception as e:
            if VERBOSE:
                print(f"[{tp}] graph load failed via index: {gpath} ({e})")

            # --- fallback: try to reconstruct path from GRAPHS_DIR ---
            label_uid = rec.get("label_uid", None)
            if label_uid is not None:
                fallback_path = os.path.join(GRAPHS_DIR, tp, f"{label_uid}.gpickle")

                if os.path.exists(fallback_path):
                    if VERBOSE:
                        print(f"[{tp}] retrying via fallback path: {fallback_path}")
                    try:
                        G = load_cell_graph(fallback_path)
                        gpath = fallback_path  # update for downstream saving
                    except Exception as e2:
                        if VERBOSE:
                            print(f"[{tp}] fallback load failed: {fallback_path} ({e2})")

            if G is None:
                continue


        label_uid = G.graph.get("label_uid", None) or rec.get("label_uid", None)
        if label_uid is None:
            if VERBOSE:
                print(f"[{tp}] missing label_uid: {gpath}")
            continue
        if label_uid in blacklist:
            continue

        # 2) resolve mesh path
        mesh_path = rec.get("mesh_path", None) or G.graph.get("mesh_path", None)
        if not mesh_path:
            if VERBOSE:
                print(f"[{tp}] missing mesh_path for {label_uid}. "
                      f"Fix by ensuring {os.path.join(GRAPHS_DIR, tp, 'index.csv')} has mesh_path "
                      f"or storing G.graph['mesh_path'] during graph preprocessing.")
            continue
        if not os.path.exists(mesh_path):
            if VERBOSE:
                print(f"[{tp}] missing mesh for {label_uid}: {mesh_path}")
            continue

        # 3) output paths
        tp_seg_dir = os.path.join(SEG_DIR, tp)
        seg_path = os.path.join(tp_seg_dir, f"{label_uid}.npz")

        if (not OVERWRITE) and os.path.exists(seg_path):
            continue

        if DRY_RUN:
            if VERBOSE:
                print(f"[dry-run] would segment {tp}/{label_uid}")
                print(f"         graph: {gpath}")
                print(f"         mesh : {mesh_path}")
                print(f"         out  : {seg_path}")
            n_done += 1
            if MAX_GRAPHS is not None and n_done >= int(MAX_GRAPHS):
                break
            continue

        # 4) load mesh
        try:
            mesh = OrganoidMesh(mesh_path)
        except Exception as e:
            if VERBOSE:
                print(f"[{tp}] mesh load failed for {label_uid}: {e}")
            continue

        # ensure mesh / graph alignment before segmentation
        try:
            ensure_mesh_graph_aligned(mesh, G)
        except Exception as e:
            if VERBOSE:
                print(f"[{tp}] mesh/graph alignment failed for {label_uid}: {e}")
            continue

        # 5) run segmentation
        try:
            out = segment_crypts_organoid(
                G=G,
                mesh=mesh,
                bin_centers=bin_centers,
                crypt_vocab_idx=list(CRYPT_VOCAB_IDX),
                neck_vocab_idx=None if NECK_VOCAB_IDX is None else list(NECK_VOCAB_IDX),
                debug=DEBUG,
                **(SEGMENT_KWARGS or {}),
            )
        except Exception as e:
            if VERBOSE:
                print(f"[{tp}] segmentation failed for {label_uid}: {e}")
            continue

        if DEBUG:
            crypts, villi, d_norm, L_crypt, Circ, dbg = out
        else:
            crypts, villi, d_norm, L_crypt, Circ = out
            dbg = None

        # 6) save segmentation
        os.makedirs(tp_seg_dir, exist_ok=True)
        save_segmentation_npz(
            seg_path,
            label_uid=label_uid,
            crypts_ll=patches_to_ll(crypts),
            villi_ll=patches_to_ll(villi),
            bin_centers=bin_centers,
            d_norm=d_norm,
            L_crypt=L_crypt,
            Circ=Circ,
            extra={"debug": dbg, "graph_path": gpath, "mesh_path": mesh_path} if dbg is not None else
                  {"graph_path": gpath, "mesh_path": mesh_path},
        )

        # 7) optional: save compact features
        if SAVE_FEATURES and FEATURES_DIR is not None:
            tp_feat_dir = os.path.join(FEATURES_DIR, tp)
            os.makedirs(tp_feat_dir, exist_ok=True)
            feat_path = os.path.join(tp_feat_dir, f"{label_uid}.npz")

            if OVERWRITE or (not os.path.exists(feat_path)):
                try:
                    proj_vertex_ids = graph_get(G, "proj_vertex", dtype=np.int32)
                except Exception:
                    proj_vertex_ids = None
                try:
                    markers_bin = graph_get(G, "markers_bin")
                except Exception:
                    markers_bin = None
                try:
                    centroids = graph_get(G, "centroid", dtype=float)
                except Exception:
                    centroids = None

                save_features_npz(
                    feat_path,
                    label_uid=label_uid,
                    markers_bin=markers_bin,
                    proj_vertex_ids=proj_vertex_ids,
                    centroids=centroids,
                    extra={"graph_path": gpath, "mesh_path": mesh_path},
                )

        n_done += 1
        if MAX_GRAPHS is not None and n_done >= int(MAX_GRAPHS):
            break

    if VERBOSE:
        print(f"[segment] done. processed={n_done}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Run coarse shape complexity and symmetry analysis on organoid meshes.

This script follows the dataset-loop style used by run_crypt_segmentation.py.
For each discovered organoid mesh it computes:

    - raw mesh volume and surface area
    - number of cells from the dataset cell CSV
    - LB reconstruction complexity on the normalized mesh
    - reflection, C2, and C3 symmetry scores at one chosen LB level

Outputs
-------
Two CSV files are written:

    {OUT_DIR}/shape_complexity_symmetry_summary.csv
        One row per organoid, convenient for downstream analysis.

    {OUT_DIR}/shape_complexity_symmetry_symmetry_long.csv
        One row per organoid and symmetry family.

Complexity interpretation
-------------------------
``complexity_error_at_l`` is the area-weighted relative reconstruction error at
``ANALYSIS_L``.  Lower values mean the chosen number of LB modes reconstructs
the shape more accurately.  The optional error curve summary reports the first
level where reconstruction error falls below ``COMPLEXITY_THRESHOLD`` and a
smooth AUC-style error summary across ``COMPLEXITY_L_VALUES``.
"""

import csv
import os
import time

import numpy as np
import pandas as pd

from organograph.mesh.OrganoidMesh import OrganoidMesh
from organograph.mesh.complexity import (
    area_weighted_relative_reconstruction_error,
    reconstruction_error_curve,
    summarize_reconstruction_complexity,
)
from organograph.mesh.symmetry import score_all_symmetry_candidates_at_level

from organograph.io_utils.blacklist import load_blacklist
from organograph.io_utils.cells_table import prepare_cells_table
from organograph.io_utils.dataset_config import load_mesh_dataset_config
from organograph.io_utils.path_parsing import discover_mesh_paths, parse_mesh_path


# =============================================================================
# CONFIG: paths + dataset layout (EDIT THESE)
# =============================================================================

DATASET = "20251201"  # "20251201" or "20250929"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

MESH_DATA_DIR = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "fractal_output")
CELLS_CSV = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "cell_features_class.csv") # cell_types_class # cell_features_class
MESH_CONFIG_PATH = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "mesh_config.json")

OUT_DIR = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "shape_complexity_symmetry")

BLACKLIST_PATH = None # os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "blacklist_labels.csv")
WHITELIST_PATH = None  # e.g. os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "day3p5_goodmeshes.csv")

# Optional override. If None, use all timepoints from mesh_config.json.
TIMEPOINTS = None


# =============================================================================
# Analysis parameters
# =============================================================================

# Symmetry and single-level complexity are reported at this LB reconstruction
# level.  The number of coordinate modes is ANALYSIS_L**2.
ANALYSIS_L = 8

# Extra levels for the reconstruction error curve.  Include ANALYSIS_L so the
# single-level error and the curve summary are computed from the same basis.
COMPLEXITY_L_VALUES = [2, 3, 5, 8, 10, 12, 15]
COMPLEXITY_THRESHOLD = 0.05

N_SYMMETRY_SAMPLES = 8000
TRIM_FRACTION = 0.95
CLOSE_THRESHOLD = 0.05
RANDOM_SEED = 0

NORMALIZE_MESH = True
NORMALIZE_SCALE = 10.0

OVERWRITE = True
VERBOSE = True
DRY_RUN = False
MAX_MESHES = None


# =============================================================================
# LOAD DATA STRUCTURE
# =============================================================================

mesh_cfg = load_mesh_dataset_config(MESH_CONFIG_PATH)

ZARR_NAME_BY_TP = mesh_cfg["zarr_name_by_tp"]
ROUND_BY_TP = mesh_cfg["round_by_tp"]
MESHNAME_BY_TP = mesh_cfg["meshname_by_tp"]
WELLS_BY_TP = mesh_cfg.get("wells_by_tp", {})


# =============================================================================
# Helpers
# =============================================================================

def _surface_area_from_faces(mesh):
    """Total mesh surface area in the mesh's current coordinate system."""
    return float(np.sum(mesh.face_areas()))


def _count_cells_by_label(cells_df, label_uid):
    """Count rows in the prepared cells table for one organoid label."""
    if label_uid not in cells_df.index:
        return 0
    rows = cells_df.loc[label_uid]
    if isinstance(rows, pd.Series):
        return 1
    return int(len(rows))


def _symmetry_rows_to_wide(symmetry_rows):
    """Flatten the three symmetry-family rows into one wide dict."""
    out = {}
    for score in symmetry_rows:
        prefix = f"symmetry_{score.symmetry}"
        out[f"{prefix}_best_axis"] = score.axis_name
        out[f"{prefix}_axis_x"] = float(score.axis[0])
        out[f"{prefix}_axis_y"] = float(score.axis[1])
        out[f"{prefix}_axis_z"] = float(score.axis[2])
        out[f"{prefix}_trimmed_rms"] = float(score.trimmed_rms)
        out[f"{prefix}_normalized_rms"] = float(score.normalized_rms)
        out[f"{prefix}_median"] = float(score.median)
        out[f"{prefix}_matched_fraction"] = float(score.matched_fraction)
    return out


def _write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    t_start = time.perf_counter()

    if not os.path.exists(CELLS_CSV):
        raise FileNotFoundError(f"CELLS_CSV not found: {CELLS_CSV}")
    cells_df = pd.read_csv(CELLS_CSV)
    cells_df = prepare_cells_table(cells_df, label_col="label_uid")

    blacklist = load_blacklist(BLACKLIST_PATH) if BLACKLIST_PATH else set()
    whitelist = load_blacklist(WHITELIST_PATH) if WHITELIST_PATH else None

    timepoints = list(TIMEPOINTS) if TIMEPOINTS is not None else list(ZARR_NAME_BY_TP.keys())

    mesh_paths = discover_mesh_paths(
        data_dir=MESH_DATA_DIR,
        timepoints=timepoints,
        zarr_names=ZARR_NAME_BY_TP,
        rounds=ROUND_BY_TP,
        meshes=MESHNAME_BY_TP,
        wells=WELLS_BY_TP,
    )

    if VERBOSE:
        print(f"[shape] found {len(mesh_paths)} mesh files")
        print(f"[shape] analysis_l={ANALYSIS_L} n_modes={ANALYSIS_L**2}")

    os.makedirs(OUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUT_DIR, "shape_complexity_symmetry_summary.csv")
    symmetry_long_path = os.path.join(OUT_DIR, "shape_complexity_symmetry_symmetry_long.csv")

    if (not OVERWRITE) and (os.path.exists(summary_path) or os.path.exists(symmetry_long_path)):
        raise FileExistsError(
            "Output CSV exists and OVERWRITE=False. "
            f"summary={summary_path} symmetry_long={symmetry_long_path}"
        )

    summary_rows = []
    symmetry_long_rows = []
    failed_rows = []

    n_done = 0
    max_l = max(max(COMPLEXITY_L_VALUES), int(ANALYSIS_L))

    for mesh_path in mesh_paths:
        try:
            rec = parse_mesh_path(mesh_path)
        except Exception as e:
            if VERBOSE:
                print(f"[skip] cannot parse mesh path: {mesh_path} ({e})")
            continue

        tp = rec.get("timepoint", None)
        label_uid = rec.get("label_uid", None)
        well = rec.get("well", None)
        organoid_id = rec.get("organoid_id", None)

        if not tp or not label_uid:
            if VERBOSE:
                print(f"[skip] missing timepoint/label_uid for: {mesh_path}")
            continue

        if label_uid in blacklist:
            if VERBOSE:
                print(f"[skip] {label_uid} is blacklisted")
            continue

        if whitelist is not None and label_uid not in whitelist:
            if VERBOSE:
                print(f"[skip] {label_uid} not in whitelist")
            continue

        n_planned = n_done + 1
        if DRY_RUN:
            print(f"[DRY_RUN] would analyze: tp={tp} label_uid={label_uid}")
            print(f"          mesh: {mesh_path}")
            if MAX_MESHES is not None and n_planned >= int(MAX_MESHES):
                break
            n_done = n_planned
            continue

        if VERBOSE:
            print(f"[shape] analyzing {tp}/{label_uid}")

        try:
            mesh = OrganoidMesh(str(mesh_path))

            raw_volume = float(mesh.volume())
            raw_surface_area = _surface_area_from_faces(mesh)

            if NORMALIZE_MESH:
                mesh.normalize_inplace(scale=NORMALIZE_SCALE)

            normalized_volume = float(mesh.volume())
            normalized_surface_area = _surface_area_from_faces(mesh)

            # Compute the largest requested eigenspace once; all lower l values
            # reuse the stored coefficients/reconstruction path.
            mesh.compute_spectral_coefficients(lmax=max_l)

            error_curve = reconstruction_error_curve(mesh, COMPLEXITY_L_VALUES)
            complexity_summary = summarize_reconstruction_complexity(
                error_curve,
                threshold=COMPLEXITY_THRESHOLD,
            )
            complexity_error_at_l = area_weighted_relative_reconstruction_error(mesh, ANALYSIS_L)

            all_symmetry_scores = score_all_symmetry_candidates_at_level(
                mesh,
                ANALYSIS_L,
                n_samples=N_SYMMETRY_SAMPLES,
                trim_fraction=TRIM_FRACTION,
                close_threshold=CLOSE_THRESHOLD,
                rng=RANDOM_SEED,
            )
            symmetry_scores = [
                min(
                    [score for score in all_symmetry_scores if score.symmetry == symmetry],
                    key=lambda score: score.trimmed_rms,
                )
                for symmetry in ("reflection", "C2", "C3")
            ]

            n_cells = _count_cells_by_label(cells_df, label_uid)

        except Exception as e:
            failed_rows.append(
                {
                    "timepoint": tp,
                    "well": well,
                    "organoid_id": organoid_id,
                    "label_uid": label_uid,
                    "mesh_path": str(mesh_path),
                    "error": repr(e),
                }
            )
            if VERBOSE:
                print(f"[shape] failed for {tp}/{label_uid}: {e!r}")
            continue

        base_row = {
            "dataset": DATASET,
            "timepoint": tp,
            "well": well,
            "organoid_id": organoid_id,
            "label_uid": label_uid,
            "mesh_path": str(mesh_path),
            "n_cells": int(n_cells),
            "n_vertices": int(mesh.v.shape[0]),
            "n_faces": int(mesh.f.shape[0]),
            "raw_volume": raw_volume,
            "raw_surface_area": raw_surface_area,
            "normalized_volume": normalized_volume,
            "normalized_surface_area": normalized_surface_area,
            "normalized": bool(NORMALIZE_MESH),
            "normalization_scale": float(NORMALIZE_SCALE),
            "analysis_l": int(ANALYSIS_L),
            "analysis_n_modes": int(ANALYSIS_L**2),
            "complexity_error_at_l": float(complexity_error_at_l),
            "complexity_l_values": ";".join(str(int(l)) for l in COMPLEXITY_L_VALUES),
            **complexity_summary,
        }
        base_row.update(_symmetry_rows_to_wide(symmetry_scores))
        summary_rows.append(base_row)

        for score in all_symmetry_scores:
            score_row = score.to_record(organoid_id=label_uid)
            score_row.update(
                {
                    "dataset": DATASET,
                    "timepoint": tp,
                    "well": well,
                    "organoid_id_raw": organoid_id,
                    "mesh_path": str(mesh_path),
                }
            )
            symmetry_long_rows.append(score_row)

        if VERBOSE:
            print(
                f"[shape] done {tp}/{label_uid}: "
                f"cells={n_cells} complexity_error_l{ANALYSIS_L}={complexity_error_at_l:.4f}"
            )

        n_done += 1
        if MAX_MESHES is not None and n_done >= int(MAX_MESHES):
            break

    _write_csv(summary_path, summary_rows)
    _write_csv(symmetry_long_path, symmetry_long_rows)

    if failed_rows:
        failed_path = os.path.join(OUT_DIR, "shape_complexity_symmetry_failed.csv")
        _write_csv(failed_path, failed_rows)
        if VERBOSE:
            print(f"[shape] wrote failures: {failed_path} ({len(failed_rows)} rows)")

    elapsed_s = time.perf_counter() - t_start
    if VERBOSE:
        print(f"[shape] wrote summary      : {summary_path} ({len(summary_rows)} rows)")
        print(f"[shape] wrote symmetry long: {symmetry_long_path} ({len(symmetry_long_rows)} rows)")
        print(f"[shape] done. processed={n_done} DRY_RUN={DRY_RUN} elapsed={elapsed_s:.2f}s ({elapsed_s/60.0:.2f} min)")


if __name__ == "__main__":
    main()

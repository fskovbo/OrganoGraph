#!/usr/bin/env python3
"""
Run crypt segmentation on organoid meshes.

This script performs automated segmentation of intestinal crypts on a dataset
of organoid surface meshes. It iterates over all meshes discovered in the
specified data directory, runs the crypt segmentation pipeline on each mesh,
and saves the resulting segmentation outputs to disk.

Pipeline overview
-----------------
For each mesh:
1. The mesh is loaded and normalized (centered and rescaled).
2. The Laplace–Beltrami operator is diagonalized to enable spectral
   computations used by the Heat Kernel Signature (HKS).
3. The mesh is encoded using a precomputed vocabulary of HKS features
   (the "bag of features").
4. Candidate crypt regions are detected based on similarity of the
   encoded HKS to crypt-like vocabulary features.
5. Optional filtering steps are applied to remove false positives
   (for example using raw HKS statistics).
6. Detected regions can optionally be refined by subdividing large patches.
7. A geodesic distance field is computed from each crypt bottom to
   estimate the crypt axis and characteristic crypt length.
8. Crypt regions may be extended along this axis to obtain the final
   segmentation.

Outputs
-------
For each processed organoid, the script saves a compressed `.npz` file
containing the segmentation results, including:

    - crypts_ll              : list of vertex indices for each crypt
    - bottom_vertex_ids      : vertex index of the crypt bottom
    - L_crypts               : estimated crypt lengths
    - circumference_crypts   : circumference profile along the crypt axis
    - d_discretized          : discretized distance coordinate along the axis

Optional intermediate variables from the segmentation pipeline can also
be stored depending on the configuration.

Customization
-------------
Users can easily customize the behavior of the pipeline by editing the
configuration section at the top of this file, including:

    - dataset paths
    - timepoints and wells to process
    - segmentation parameters
    - geodesic distance algorithm
    - filtering functions applied to candidate crypts
    - which intermediate variables are saved

The script also supports a DRY_RUN mode, which prints the meshes that
would be processed without actually running the segmentation.

Typical usage
-------------
Run the script directly from the command line:

    python run_crypt_segmentation.py

The script will discover meshes, process them sequentially, and write the
segmentation results to the configured output directory.
"""

import os
import time 
import numpy as np

from organograph.mesh.OrganoidMesh import OrganoidMesh
from organograph.mesh.hks import compute_hks
from organograph.mesh.curvature import compute_gaussian_curvature

from organograph.io_utils.path_parsing import discover_mesh_paths, parse_mesh_path
from organograph.io_utils.blacklist import load_blacklist
from organograph.io_utils.dataset_config import load_mesh_dataset_config

from organograph.crypts.segment import segment_crypts_organoid
from organograph.crypts.filters import filter_crypts_by_hks_percent, filter_crypts_by_size


# =============================================================================
# CONFIG: paths + dataset layout (EDIT THESE)
# =============================================================================

DATASET         = "20250929" # 20250929 20251201

_SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(_SCRIPT_DIR)

MESH_DATA_DIR   = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "fractal_output")
SEG_DIR         = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "crypt_segmentations_mesh_3p5selected")
VOCAB_PATH      = os.path.join(PROJECT_ROOT, "sim", "vocab_with_meta.npz")
MESH_CONFIG_PATH= os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "mesh_config.json")

BLACKLIST_PATH  = None # os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "blacklist_labels.csv")
WHITELIST_PATH  = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET, "day3p5_goodmeshes.csv")


# Optional override. If None, use all timepoints from mesh_config.json
TIMEPOINTS      = ['day3p5']   


OVERWRITE = True
VERBOSE = True
MAX_MESHES = None
DRY_RUN = False  


# =============================================================================
# LOAD DATA STRUCTURE
# =============================================================================

mesh_cfg = load_mesh_dataset_config(MESH_CONFIG_PATH)

ZARR_NAME_BY_TP = mesh_cfg["zarr_name_by_tp"]
ROUND_BY_TP = mesh_cfg["round_by_tp"]
MESHNAME_BY_TP = mesh_cfg["meshname_by_tp"]
WELLS_BY_TP = mesh_cfg.get("wells_by_tp", {})



# =============================================================================
# USER-EDITABLE: choose geodesic function + filters (CUSTOMIZE THESE IF NEEDED)
# =============================================================================

from organograph.mesh.geodesics import compute_geodesics_dijkstra
GEODESIC_FN = compute_geodesics_dijkstra
GEODESIC_KWARGS = None  # or dict(...)

FILTERS = [
    lambda patches, **kw: filter_crypts_by_hks_percent(
        patches,
        min_percent_greater=4.0, # 1.5, 3.0
        t_max=10,
        **kw
    ),
    lambda patches, **kw: filter_crypts_by_size(
        patches,
        min_patch_verts=25,
        min_patch_area=5,
        **kw
    ),
]

CALC_GAUSSIAN_CURV = True # if True, compute and store Gaussian curvature field for full organoid
TAU_GAUSSIAN_CURV = None # optional dimensionless timescales used for computing Gaussian curvature

# =============================================================================
# Segmentation parameters (ALL user-tunable args live here)
# =============================================================================
# Note: L_ref/crypt_vocab_idx can be left as None if part of vocab.
SEGMENT_KWARGS = dict(
    # --- vocab / encoding defaults ---
    L_ref=None,
    crypt_vocab_idx=None,

    # --- detection ---
    threshold=0.5,

    # --- refinement ---
    refine_crypts=False, # True
    refine_threshold=0.0,
    refine_only_if_area_at_least=5.0,
    min_refined_frac_of_parent=0.05,

    # --- neck extension ---
    extend_max=1.5, # 2.0
    disc_resolution=150,
    remove_nested_features=True,
)

# =============================================================================
# Which seg_vars fields to store (user-editable)
# =============================================================================
# You can specify either:
#   - a list of keys (store those keys if present), or
#   - a dict {key: True/False} to toggle fields explicitly.
#
# Keep this minimal if file size matters.
SAVE_SEG_VARS = [
    # "ts_mesh",
    # "ts_vocab",
    # "hks_segment",
    # "normalised_hks_segment",
    # "encoding",
    # "filter_info",
]

# =============================================================================
# Helpers
# =============================================================================

def patches_to_ll(patches):
    """list[set[int]] -> list[list[int]] for npz saving."""
    return [sorted(list(p)) for p in (patches or [])]

def normalize_save_spec(spec):
    """Return a list of keys to save."""
    if spec is None:
        return []
    if isinstance(spec, dict):
        return [k for k, v in spec.items() if bool(v)]
    return list(spec)


# =============================================================================
# Main
# =============================================================================

def main():
    t_start = time.perf_counter()

    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"VOCAB_PATH not found: {VOCAB_PATH}")
    vocab = np.load(VOCAB_PATH, allow_pickle=True)

    blacklist = load_blacklist(BLACKLIST_PATH) if BLACKLIST_PATH else set()
    whitelist = load_blacklist(WHITELIST_PATH) if WHITELIST_PATH else None

    if VERBOSE and blacklist:
        print(f"[mesh-seg] loaded blacklist with {len(blacklist)} entries")
    if VERBOSE and whitelist is not None:
        print(f"[mesh-seg] loaded whitelist with {len(whitelist)} entries")

    # discover mesh files
    timepoints = list(TIMEPOINTS)

    mesh_paths = discover_mesh_paths(
        data_dir=MESH_DATA_DIR,
        timepoints=timepoints,
        zarr_names=ZARR_NAME_BY_TP,
        rounds=ROUND_BY_TP,
        meshes=MESHNAME_BY_TP,
        wells=WELLS_BY_TP,
    )

    if VERBOSE:
        print(f"[mesh-seg] found {len(mesh_paths)} mesh files")

    keys_to_save = normalize_save_spec(SAVE_SEG_VARS)

    n_done = 0
    for mesh_path in mesh_paths:
        # parse path metadata
        try:
            rec = parse_mesh_path(mesh_path)
        except Exception as e:
            if VERBOSE:
                print(f"[skip] cannot parse mesh path: {mesh_path} ({e})")
            continue

        tp = rec.get("timepoint", None)
        label_uid = rec.get("label_uid", None)

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

        out_dir = os.path.join(SEG_DIR, tp)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{label_uid}.npz")

        if (not OVERWRITE) and os.path.exists(out_path):
            if VERBOSE:
                print(f"[skip] exists: {out_path}")
            continue

        if DRY_RUN:
            print(f"[DRY_RUN] would segment: tp={tp} label_uid={label_uid}")
            print(f"          mesh: {mesh_path}")
            print(f"          out : {out_path}")
            n_done += 1
            if MAX_MESHES is not None and n_done >= int(MAX_MESHES):
                break
            continue

        # load mesh
        try:
            mesh = OrganoidMesh(str(mesh_path))
        except Exception as e:
            if VERBOSE:
                print(f"[{tp}] mesh load failed for {label_uid}: {e}")
            continue

        # normalize + id
        mesh.normalize_inplace()
        mesh.label_uid = label_uid

        # optional eigendecomp (keep if your HKS needs it precomputed)
        try:
            mesh._eig_decomp()
        except Exception as e:
            if VERBOSE:
                print(f"[{tp}] eigendecomp failed for {label_uid}: {e}")
            continue

        # run segmentation
        try:
            t0 = time.perf_counter()
            crypts, seg_vars = segment_crypts_organoid(
                mesh,
                vocab,
                geodesic_fn=GEODESIC_FN,
                geodesic_kwargs=GEODESIC_KWARGS,
                filter_fn_list=FILTERS,
                return_vars=True,
                **SEGMENT_KWARGS,
            )
            dt = time.perf_counter() - t0
            if VERBOSE:
                print(f"[mesh-seg] segmented {tp}/{label_uid} in {dt:.2f}s")
        except Exception as e:
            if VERBOSE:
                print(f"[{tp}] segmentation failed for {label_uid}: {e}")
            continue

        # --- main results ---
        save_dict = {
            "label_uid": str(label_uid),
            "timepoint": str(tp),
            "mesh_path": str(mesh_path),
            "crypts_ll": np.array(patches_to_ll(crypts), dtype=object),
        }

        # Always store these key outputs if present
        for k in ("bottom_vertex_ids", "L_crypts", "d_crypts", "circumference_crypts", "d_discretized", "crypt_constrictions", "crypt_elongations"):
            if k in seg_vars:
                save_dict[k] = seg_vars[k]
            else:
                if VERBOSE:
                    print(f"[warn] seg_vars missing '{k}' for {tp}/{label_uid}")

        # --- optional additional seg_vars ---
        for k in keys_to_save:
            if k in save_dict:
                continue  # already stored above
            if k in seg_vars:
                save_dict[k] = seg_vars[k]

        # --- optional computations ---
        if CALC_GAUSSIAN_CURV:
            curvature_gauss = compute_gaussian_curvature(mesh, TAU_GAUSSIAN_CURV)
            save_dict["curvature_gauss"] = curvature_gauss

        np.savez_compressed(out_path, **save_dict)

        if VERBOSE:
            print(f"[mesh-seg] saved {tp}/{label_uid} -> {out_path} (n_crypts={len(crypts)})")

        n_done += 1
        if MAX_MESHES is not None and n_done >= int(MAX_MESHES):
            break

    elapsed_s = time.perf_counter() - t_start
    if VERBOSE:
        print(f"[mesh-seg] done. processed={n_done} DRY_RUN={DRY_RUN} elapsed={elapsed_s:.2f}s ({elapsed_s/60.0:.2f} min)")

    


if __name__ == "__main__":
    main()
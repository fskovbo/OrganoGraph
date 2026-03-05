#!/usr/bin/env python3
"""
Run crypt segmentation directly on meshes (no graphs required).
"""

import os
import time 
import numpy as np

from organograph.mesh.OrganoidMesh import OrganoidMesh
from organograph.io_utils.path_parsing import discover_mesh_paths, parse_mesh_path

from organograph.crypts.segment import segment_crypts_organoid
from organograph.crypts.filters import filter_crypts_by_hks_percent

# =============================================================================
# CONFIG: paths + dataset layout (EDIT THESE)
# =============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

MESH_DATA_DIR = os.path.join(PROJECT_ROOT, "..", "NicoleData", "20251201", "fractal_output")
SEG_DIR       = os.path.join(PROJECT_ROOT, "..", "NicoleData", "20251201", "crypt_segmentations_mesh")
VOCAB_PATH    = os.path.join(PROJECT_ROOT, "sim", "vocab_with_meta.npz")

TIMEPOINTS = ["day4p5"] 

ZARR_NAME_BY_TP = {tp: "251130R0.zarr" for tp in TIMEPOINTS}
ROUND_BY_TP     = {tp: "2_zillum_registered" for tp in TIMEPOINTS}
MESHNAME_BY_TP  = {tp: "nnorg_corrected_annotated_by_projection" for tp in TIMEPOINTS}

WELLS_BY_TP = {"day4p5": ["B02", "B03", "B04", "B05"],}

OVERWRITE = True
VERBOSE = True
MAX_MESHES = None
DRY_RUN = False  

# =============================================================================
# USER-EDITABLE: choose geodesic function + filters (CUSTOMIZE THESE IF NEEDED)
# =============================================================================

from organograph.mesh.geodesics import compute_geodesics_dijkstra
GEODESIC_FN = compute_geodesics_dijkstra
GEODESIC_KWARGS = None  # or dict(...)

FILTERS = [
    lambda patches, **kw: filter_crypts_by_hks_percent(
        patches,
        min_percent_greater=3.0,
        t_max=10,
        **kw
    ),
]

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
    min_patch_verts=25,
    min_patch_area=None,

    # --- refinement ---
    refine_crypts=True,
    refine_threshold=0.0,
    refine_only_if_area_at_least=5.0,

    # --- neck extension ---
    extend_max=2.5,
    disc_resolution=100,
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
    # "hks",
    # "normalised_hks",
    # "encoding",
    "d_crypts",
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
        for k in ("bottom_vertex_ids", "L_crypts", "circumference_crypts", "d_discretized"):
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
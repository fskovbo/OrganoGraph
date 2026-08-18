#!/usr/bin/env python3
"""
Build and export biology-aware skeletons plus fitted primitives for a dataset.

The output is intended as a portable raw interchange format for a separate VAE
project.  Each organoid gets a directory containing:

    shape.json
    shape_nodes.csv
    shape_edges.csv
    shape_primitives.csv
    shape_arrays.npz

The JSON payload preserves the full skeleton graph, primitive attachments,
configs, detections, metadata, and compact component summaries.  The CSV/NPZ
files are easier to load from lightweight downstream analysis code.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from organograph.crypts.filters import filter_crypts_by_hks_percent, filter_crypts_by_size
from organograph.io_utils.blacklist import (
    default_discard_labels_path,
    load_blacklist,
    load_optional_blacklist,
)
from organograph.io_utils.dataset_config import load_mesh_dataset_config
from organograph.io_utils.path_parsing import discover_mesh_paths, parse_mesh_path
from organograph.io_utils.run_metadata import write_run_settings
from organograph.skeleton import (
    BarrierConfig,
    BodyTransitionConfig,
    BranchValidationConfig,
    CandidateDetectionConfig,
    CryptOverlapConfig,
    DetectionConfig,
    GraphConfig,
    MeshPreparationConfig,
    NeckProfileConfig,
    PrimitiveFitConfig,
    SkeletonizationConfig,
    fit_primitives_for_skeletonization_result,
    save_shape_export,
    skeletonize_organoid,
    write_export_readme,
)


# =============================================================================
# DEFAULT CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET = "20250929" # "20251201"
DATA_ROOT = (PROJECT_ROOT.parent / "NicoleData").resolve()
DATASET_ROOT = DATA_ROOT / DATASET
MESH_DATA_DIR = DATASET_ROOT / "fractal_output"
MESH_CONFIG_PATH = DATASET_ROOT / "mesh_config.json"
VOCAB_PATH = PROJECT_ROOT / "sim" / "vocab_with_meta.npz"
OUTPUT_ROOT = DATASET_ROOT / "skeleton_primitive_exports"
BLACKLIST_PATH = default_discard_labels_path(str(DATASET_ROOT))
WHITELIST_PATH = None

# Set to None to process all configured timepoints.
TIMEPOINTS = ["day3p5", "day4", "day4p5", "day4p5-more"]

OVERWRITE = False
VERBOSE = True
DRY_RUN = False
STRICT = False
MAX_MESHES = None

NORMALIZE_MESH = True
NORMALIZE_SCALE = 10.0
EIGEN_K = 225
SMOOTH_MESH = True
SMOOTH_LMAX = 12
SMOOTH_EIGEN_K = None

FILTER_KWARGS = {
    "use_hks_filter": True,
    "min_percent_greater": 2.0,
    "hks_t_min": None,
    "hks_t_max": 10.0,
    "use_size_filter": True,
    "min_patch_verts": 25,
    "min_patch_area": 5.0,
}

FILTER_NAMES = [
    "filter_crypts_by_hks_percent",
    "filter_crypts_by_size",
]

def make_filter_list(**kw):
    filters = []
    if kw.get("use_hks_filter", True):
        filters.append(
            lambda patches, **inner: filter_crypts_by_hks_percent(
                patches,
                min_percent_greater=kw["min_percent_greater"],
                t_min=kw.get("hks_t_min"),
                t_max=kw.get("hks_t_max"),
                **inner,
            )
        )
    if kw.get("use_size_filter", True):
        filters.append(
            lambda patches, **inner: filter_crypts_by_size(
                patches,
                min_patch_verts=kw["min_patch_verts"],
                min_patch_area=kw.get("min_patch_area"),
                **inner,
            )
        )
    return filters or None


FILTERS = make_filter_list(**FILTER_KWARGS)

SKELETONIZATION_CONFIG = SkeletonizationConfig(
    detection=DetectionConfig(
        candidates=CandidateDetectionConfig(
            threshold=0.5,
            filters=FILTERS,
            refine_threshold=0.0,
            refine_min_area=5.0,
            min_child_fraction=0.05,
            final_tip_hks_time=1.0,
            final_tip_bottom_fraction=0.6,
            final_tip_min_hks_percent_increase=5.0,
        ),
        necks=NeckProfileConfig(
            max_axis_level=2.0,
            resolution=200,
            search_interval=(0.8, 2.0),
            min_prominence=0.05,
            min_length=0.05,
        ),
        branches=BranchValidationConfig(
            min_confidence=0.85,
            max_neck_to_body_radius_ratio=0.70,
            max_growth_size_factor=3.0,
            max_mesh_fraction=0.40,
        ),
        body_transition=BodyTransitionConfig(enabled=True),
        barriers=BarrierConfig(),
        mesh=MeshPreparationConfig(smooth=False),
    ),
    graph=GraphConfig(
        max_dimensionless_curvature=0.50,
        curvature_penalty=8.0,
    ),
)

PRIMITIVE_FIT_CONFIG = PrimitiveFitConfig(
    refine_host_primitives=False,
    component_kwargs={},
    body_branch_neck_kwargs={
        "radius_quantile": 0.5,
        "expansion_factor": 1.35,
        "max_extent_fraction": 0.25,
        "min_extent_radius_fraction": 0.35,
    },
    body_kwargs={
        "primitive_type": "asymmetric_superellipsoid",
        "add_attachment_cap_support": True,
        "cap_support_points_per_attachment": 2 * 64,
        "cap_support_radius_fraction": 0.5,
    },
    branch_kwargs={"primitive_type": "asymmetric_superellipsoid"},
    crypt_tube_kwargs={
        "smooth_centerline": True,
        "smooth_bulged_centerlines": False,
        "centerline_n_bands": 7,
        "centerline_n_samples": 64,
        "centerline_constriction_weight": 4.0,
        "update_crypt_nodes": True,
        "radius_quantile": 0.5,
        "optimize_radius_profile": True,
        "initial_body_position": 0.5,
        "initial_taper_position": 0.85,
        "body_position_bounds": (0.2, 0.7),
        "min_taper_gap": 0.1,
        "max_taper_position": 0.9,
    },
    crypt_overlap=CryptOverlapConfig(
        enabled=True,
        threshold=0.30,
        samples=32768,
        random_seed=0,
        max_passes=3,
        max_host_attachment_angle=np.pi / 3,
    ),
)


# =============================================================================
# HELPERS
# =============================================================================

def parse_timepoints(value: str | None):
    if value is None:
        return TIMEPOINTS
    value = value.strip()
    if value.lower() in {"", "none", "all"}:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_paths(args) -> dict[str, Path | str | None]:
    data_root = Path(args.data_root).resolve()
    dataset_root = data_root / args.dataset
    return {
        "data_root": data_root,
        "dataset_root": dataset_root,
        "mesh_data_dir": Path(args.mesh_data_dir).resolve()
        if args.mesh_data_dir
        else dataset_root / "fractal_output",
        "mesh_config_path": Path(args.mesh_config).resolve()
        if args.mesh_config
        else dataset_root / "mesh_config.json",
        "vocab_path": Path(args.vocab_path).resolve(),
        "output_root": Path(args.output_root).resolve()
        if args.output_root
        else dataset_root / "skeleton_primitive_exports",
        "blacklist_path": args.blacklist_path,
        "whitelist_path": args.whitelist_path,
    }


def _clamped_eigen_k(mesh, requested_k):
    n_vertices = int(np.asarray(mesh.v).shape[0])
    return max(2, min(int(requested_k), n_vertices - 2))


def _reset_spectral_state(mesh):
    mesh.laplacian = None
    mesh.mass_matrix = None
    mesh.eigvals = None
    mesh.eigvecs = None
    mesh.coeffs_v = None
    mesh.lmax = None


def _ensure_mesh_eigendecomposition(mesh, requested_k):
    k = _clamped_eigen_k(mesh, requested_k)
    if mesh.eigvals is None or mesh.eigvecs is None or mesh.eigvecs.shape[1] < k:
        _reset_spectral_state(mesh)
        mesh._eig_decomp(k=k)
    return k


def _smooth_mesh_low_pass(mesh):
    lmax = int(SMOOTH_LMAX)
    if lmax < 1:
        raise ValueError("SMOOTH_LMAX must be at least 1")

    coeff_k = int(lmax**2)
    _ensure_mesh_eigendecomposition(mesh, max(EIGEN_K, coeff_k))
    mesh.compute_spectral_coefficients(lmax=lmax)
    mesh.v = np.asarray(
        mesh.reconstruct_from_coeffs(mesh.coeffs_v, lmax=lmax),
        dtype=float,
    )

    _reset_spectral_state(mesh)
    _ensure_mesh_eigendecomposition(mesh, SMOOTH_EIGEN_K or EIGEN_K)
    return mesh


def prepare_mesh_for_skeleton_export(mesh):
    """Match the mesh preparation cell in notebooks/tutorial_skeleton.ipynb."""
    if NORMALIZE_MESH:
        mesh.normalize_inplace(scale=NORMALIZE_SCALE, center="mean")

    if SMOOTH_MESH:
        _smooth_mesh_low_pass(mesh)
    else:
        _ensure_mesh_eigendecomposition(mesh, EIGEN_K)
    return mesh


def output_exists(organoid_dir: Path) -> bool:
    return (organoid_dir / "shape.json").exists()


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_failure(path: Path, *, label_uid: str, mesh_path: str, error: BaseException) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"label_uid={label_uid}\n")
        handle.write(f"mesh_path={mesh_path}\n")
        handle.write(f"error={type(error).__name__}: {error}\n")
        handle.write(traceback.format_exc())
        handle.write("\n" + "=" * 80 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and export skeletons plus fitted primitives."
    )
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--mesh-data-dir", default=None)
    parser.add_argument("--mesh-config", default=None)
    parser.add_argument("--vocab-path", default=str(VOCAB_PATH))
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--timepoints",
        default=None,
        help="Comma-separated timepoints. Use 'all' to process all configured timepoints.",
    )
    parser.add_argument("--blacklist-path", default=BLACKLIST_PATH)
    parser.add_argument("--whitelist-path", default=WHITELIST_PATH)
    parser.add_argument("--max-meshes", type=int, default=MAX_MESHES)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--strict", action="store_true", default=STRICT)
    parser.add_argument(
        "--include-intermediates",
        action="store_true",
        help="Include full skeleton detection intermediates in shape.json.",
    )
    parser.add_argument(
        "--include-components",
        action="store_true",
        help="Include full primitive component dictionaries in shape.json.",
    )
    return parser


# =============================================================================
# MAIN
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    verbose = bool(VERBOSE and not args.quiet)
    timepoints = parse_timepoints(args.timepoints)
    paths = resolve_paths(args)

    output_root = Path(paths["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    failure_log = output_root / "failures.log"
    if failure_log.exists() and args.overwrite:
        failure_log.unlink()

    if not Path(paths["vocab_path"]).exists():
        raise FileNotFoundError(f"VOCAB_PATH not found: {paths['vocab_path']}")
    if not Path(paths["mesh_config_path"]).exists():
        raise FileNotFoundError(f"Mesh config not found: {paths['mesh_config_path']}")

    mesh_cfg = load_mesh_dataset_config(str(paths["mesh_config_path"]))
    zarr_names = mesh_cfg["zarr_name_by_tp"]
    rounds = mesh_cfg["round_by_tp"]
    meshes = mesh_cfg["meshname_by_tp"]
    wells = mesh_cfg.get("wells_by_tp", {})

    blacklist = load_optional_blacklist(
        paths["blacklist_path"],
        label="blacklist",
        verbose=verbose,
    )
    whitelist = (
        load_blacklist(paths["whitelist_path"])
        if paths["whitelist_path"]
        else None
    )
    vocab = np.load(paths["vocab_path"], allow_pickle=True)

    mesh_paths = discover_mesh_paths(
        data_dir=str(paths["mesh_data_dir"]),
        timepoints=timepoints,
        zarr_names=zarr_names,
        rounds=rounds,
        meshes=meshes,
        wells=wells,
    )
    if args.max_meshes is not None:
        mesh_paths = mesh_paths[: int(args.max_meshes)]

    OrganoidMesh = None
    compute_geodesics_dijkstra = None
    if not args.dry_run:
        from organograph.mesh.OrganoidMesh import OrganoidMesh
        from organograph.mesh.geodesics import compute_geodesics_dijkstra

    if verbose:
        print(f"[skeleton-export] found {len(mesh_paths)} mesh files")
        print(f"[skeleton-export] output root: {output_root}")

    stats = {
        "mesh_files_found": int(len(mesh_paths)),
        "exported": 0,
        "skipped_blacklist": 0,
        "skipped_whitelist": 0,
        "skipped_existing": 0,
        "failed": 0,
        "dry_run": bool(args.dry_run),
    }
    manifest_rows: list[dict] = []
    t_start = time.perf_counter()

    for mesh_path in mesh_paths:
        label_uid = str(mesh_path)
        try:
            rec = parse_mesh_path(mesh_path)
            label_uid = rec["label_uid"]
            timepoint = rec["timepoint"]

            if label_uid in blacklist:
                stats["skipped_blacklist"] += 1
                if verbose:
                    print(f"[skip] {label_uid} is blacklisted")
                continue
            if whitelist is not None and label_uid not in whitelist:
                stats["skipped_whitelist"] += 1
                if verbose:
                    print(f"[skip] {label_uid} not in whitelist")
                continue

            organoid_dir = output_root / timepoint / label_uid
            if output_exists(organoid_dir) and not args.overwrite:
                stats["skipped_existing"] += 1
                if verbose:
                    print(f"[skip] exists: {organoid_dir}")
                continue

            if args.dry_run:
                print(f"[DRY_RUN] would export {label_uid}")
                print(f"          mesh: {mesh_path}")
                print(f"          out : {organoid_dir}")
                continue

            mesh = OrganoidMesh(str(mesh_path))
            mesh.label_uid = label_uid
            prepare_mesh_for_skeleton_export(mesh)

            metadata = {
                "dataset": args.dataset,
                "timepoint": timepoint,
                "well": rec.get("well"),
                "organoid_id": rec.get("organoid_id"),
                "label_uid": label_uid,
                "mesh_path": str(mesh_path),
                "vocab_path": str(paths["vocab_path"]),
            }
            t0 = time.perf_counter()
            skeleton_result = skeletonize_organoid(
                mesh,
                vocab,
                geodesic_fn=compute_geodesics_dijkstra,
                config=SKELETONIZATION_CONFIG,
                metadata={"record": dict(metadata)},
            )
            primitive_result = fit_primitives_for_skeletonization_result(
                skeleton_result,
                config=PRIMITIVE_FIT_CONFIG,
            )
            export_paths = save_shape_export(
                primitive_result,
                organoid_dir,
                metadata=metadata,
                include_detections=True,
                include_intermediates=args.include_intermediates,
                include_components=args.include_components,
            )
            dt = time.perf_counter() - t0
            summary = primitive_result.graph.to_dict()
            n_nodes = len(summary.get("nodes", []))
            n_edges = len(summary.get("edges", []))
            n_primitives = len(summary.get("primitive_attachments", []))

            manifest_rows.append(
                {
                    "dataset": args.dataset,
                    "timepoint": timepoint,
                    "well": rec.get("well"),
                    "organoid_id": rec.get("organoid_id"),
                    "label_uid": label_uid,
                    "mesh_path": str(mesh_path),
                    "output_dir": str(organoid_dir),
                    "json_path": export_paths.get("json", ""),
                    "arrays_npz_path": export_paths.get("arrays_npz", ""),
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "n_primitives": n_primitives,
                    "elapsed_s": f"{dt:.3f}",
                }
            )
            stats["exported"] += 1
            if verbose:
                print(
                    f"[skeleton-export] {label_uid}: nodes={n_nodes} "
                    f"edges={n_edges} primitives={n_primitives} in {dt:.2f}s"
                )

        except Exception as exc:
            stats["failed"] += 1
            append_failure(failure_log, label_uid=label_uid, mesh_path=str(mesh_path), error=exc)
            if verbose:
                print(f"[failed] {label_uid}: {type(exc).__name__}: {exc}")
            if args.strict:
                raise

    write_manifest(output_root / "manifest.csv", manifest_rows)
    write_export_readme(output_root / "README.md", dataset=args.dataset)
    elapsed_s = time.perf_counter() - t_start
    write_run_settings(
        output_root,
        script_name=os.path.basename(__file__),
        payload={
            "dataset": args.dataset,
            "timepoints": timepoints,
            "paths": {
                "data_root": str(paths["data_root"]),
                "dataset_root": str(paths["dataset_root"]),
                "mesh_data_dir": str(paths["mesh_data_dir"]),
                "mesh_config_path": str(paths["mesh_config_path"]),
                "vocab_path": str(paths["vocab_path"]),
                "output_root": str(output_root),
                "blacklist_path": paths["blacklist_path"],
                "whitelist_path": paths["whitelist_path"],
            },
            "parameters": {
                "overwrite": bool(args.overwrite),
                "dry_run": bool(args.dry_run),
                "strict": bool(args.strict),
                "max_meshes": args.max_meshes,
                "include_intermediates": bool(args.include_intermediates),
                "include_components": bool(args.include_components),
                "mesh_preparation": {
                    "normalize_mesh": NORMALIZE_MESH,
                    "normalize_scale": NORMALIZE_SCALE,
                    "eigen_k": EIGEN_K,
                    "smooth_mesh": SMOOTH_MESH,
                    "smooth_lmax": SMOOTH_LMAX,
                    "smooth_eigen_k": SMOOTH_EIGEN_K,
                },
                "geodesic_fn": "compute_geodesics_dijkstra",
                "skeletonization_config": SKELETONIZATION_CONFIG.to_dict(),
                "primitive_fit_config": PRIMITIVE_FIT_CONFIG.to_dict(),
                "filter_kwargs": FILTER_KWARGS,
                "filters": FILTER_NAMES,
            },
            "stats": stats,
            "elapsed_s": float(elapsed_s),
        },
        verbose=verbose,
    )
    if verbose:
        print(
            f"[skeleton-export] done exported={stats['exported']} "
            f"failed={stats['failed']} elapsed={elapsed_s:.2f}s"
        )
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

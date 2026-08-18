#!/usr/bin/env python3
"""
Build and export biology-aware skeletons plus fitted primitives for one or more
mesh datasets.

The output is a compact reconstructive format for a separate VAE project. Each
organoid gets a directory containing:

    shape.json

The JSON payload preserves the final skeleton, fitted primitive geometry, and
the reversible transform to original mesh coordinates. Fitting diagnostics and
segmentation arrays are intentionally omitted.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback
from pathlib import Path

import numpy as np

from organograph.io_utils.blacklist import (
    default_discard_labels_path,
    load_blacklist,
    load_optional_blacklist,
)
from organograph.io_utils.dataset_config import load_mesh_dataset_config
from organograph.io_utils.path_parsing import discover_mesh_paths, parse_mesh_path
from organograph.io_utils.run_metadata import write_run_settings
from organograph.skeleton import (
    SHAPE_EXPORT_SCHEMA_VERSION,
    definitive_filter_options,
    definitive_mesh_preparation,
    definitive_primitive_fit_config,
    definitive_skeletonization_config,
    fit_primitives_for_skeletonization_result,
    graph_summary,
    save_shape_export,
    skeletonize_organoid,
    write_export_readme,
)


# =============================================================================
# DEFAULT CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_ROOT = (PROJECT_ROOT.parent / "NicoleData").resolve()
DATASET_TIMEPOINTS = {
    "20250929": ["day3p5", "day4", "day4p5", "day4p5-more"],
    "20251201": ["day4p5"],
}

# All configured datasets are written into this one VAE-ready export dataset.
# Dataset names remain part of each sample path to avoid label collisions.
EXPORT_ROOT = (DATA_ROOT / "combined_skeleton_primitive_exports_v2").resolve()

VOCAB_PATH = PROJECT_ROOT / "sim" / "vocab_with_meta.npz"
WHITELIST_PATH = None

OVERWRITE = False
VERBOSE = True
DRY_RUN = True
STRICT = False
MAX_MESHES = None

MESH_PREPARATION = definitive_mesh_preparation()
NORMALIZE_MESH = MESH_PREPARATION["normalize_mesh"]
NORMALIZE_SCALE = MESH_PREPARATION["normalize_scale"]
EIGEN_K = MESH_PREPARATION["eigen_k"]
SMOOTH_MESH = MESH_PREPARATION["smooth_mesh"]
SMOOTH_LMAX = MESH_PREPARATION["smooth_lmax"]
SMOOTH_EIGEN_K = MESH_PREPARATION["smooth_eigen_k"]

FILTER_KWARGS = definitive_filter_options()
FILTER_NAMES = ["filter_crypts_by_hks_percent", "filter_crypts_by_size"]
SKELETONIZATION_CONFIG = definitive_skeletonization_config()
PRIMITIVE_FIT_CONFIG = definitive_primitive_fit_config()


# =============================================================================
# HELPERS
# =============================================================================

def parse_csv_values(value: str | None):
    if value is None:
        return None
    value = value.strip()
    if value.lower() in {"", "none", "all"}:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def selected_datasets(args) -> list[str]:
    requested = parse_csv_values(args.datasets)
    return requested if requested is not None else list(DATASET_TIMEPOINTS)


def resolve_shared_paths(args) -> dict[str, Path]:
    data_root = Path(args.data_root).resolve()
    return {
        "data_root": data_root,
        "vocab_path": Path(args.vocab_path).resolve(),
        "output_root": Path(args.output_root).resolve(),
    }


def resolve_dataset_paths(
    dataset: str,
    *,
    data_root: Path,
    args,
    allow_single_dataset_overrides: bool,
) -> dict[str, Path | str | None]:
    dataset_root = data_root / dataset
    return {
        "dataset_root": dataset_root,
        "mesh_data_dir": (
            Path(args.mesh_data_dir).resolve()
            if allow_single_dataset_overrides and args.mesh_data_dir
            else dataset_root / "fractal_output"
        ),
        "mesh_config_path": (
            Path(args.mesh_config).resolve()
            if allow_single_dataset_overrides and args.mesh_config
            else dataset_root / "mesh_config.json"
        ),
        "blacklist_path": (
            args.blacklist_path
            if allow_single_dataset_overrides and args.blacklist_path
            else default_discard_labels_path(str(dataset_root))
        ),
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
        description="Build and export a combined skeleton-plus-primitive dataset."
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "Comma-separated datasets. By default, process every dataset in "
            "DATASET_TIMEPOINTS using its configured timepoints."
        ),
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument(
        "--mesh-data-dir",
        default=None,
        help="Override the mesh root when processing exactly one dataset.",
    )
    parser.add_argument(
        "--mesh-config",
        default=None,
        help="Override mesh_config.json when processing exactly one dataset.",
    )
    parser.add_argument("--vocab-path", default=str(VOCAB_PATH))
    parser.add_argument("--output-root", default=str(EXPORT_ROOT))
    parser.add_argument(
        "--timepoints",
        default=None,
        help=(
            "Override timepoints for every selected dataset. By default, use "
            "DATASET_TIMEPOINTS; use 'all' for all timepoints in each mesh config."
        ),
    )
    parser.add_argument(
        "--blacklist-path",
        default=None,
        help="Override the blacklist when processing exactly one dataset.",
    )
    parser.add_argument("--whitelist-path", default=WHITELIST_PATH)
    parser.add_argument("--max-meshes", type=int, default=MAX_MESHES)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--strict", action="store_true", default=STRICT)
    parser.add_argument(
        "--unbranched-only",
        action="store_true",
        help=(
            "Skip organoids containing branch nodes; all exports are still "
            "marked for eligibility."
        ),
    )
    return parser


# =============================================================================
# MAIN
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    verbose = bool(VERBOSE and not args.quiet)
    datasets = selected_datasets(args)
    if not datasets:
        parser.error("At least one dataset must be selected")
    if len(datasets) > 1 and any(
        (args.mesh_data_dir, args.mesh_config, args.blacklist_path)
    ):
        parser.error(
            "--mesh-data-dir, --mesh-config, and --blacklist-path can only be "
            "used when processing one dataset"
        )

    timepoint_override = parse_csv_values(args.timepoints)
    shared_paths = resolve_shared_paths(args)
    output_root = shared_paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    failure_log = output_root / "failures.log"
    if failure_log.exists() and args.overwrite:
        failure_log.unlink()

    if not shared_paths["vocab_path"].exists():
        raise FileNotFoundError(f"VOCAB_PATH not found: {shared_paths['vocab_path']}")
    whitelist = (
        load_blacklist(args.whitelist_path)
        if args.whitelist_path
        else None
    )
    vocab = np.load(shared_paths["vocab_path"], allow_pickle=True)

    dataset_runs: dict[str, dict] = {}
    mesh_records: list[dict] = []
    for dataset in datasets:
        paths = resolve_dataset_paths(
            dataset,
            data_root=shared_paths["data_root"],
            args=args,
            allow_single_dataset_overrides=len(datasets) == 1,
        )
        mesh_config_path = Path(paths["mesh_config_path"])
        if not mesh_config_path.exists():
            raise FileNotFoundError(
                f"Mesh config for dataset {dataset!r} not found: {mesh_config_path}"
            )
        mesh_cfg = load_mesh_dataset_config(str(mesh_config_path))
        timepoints = (
            timepoint_override
            if args.timepoints is not None
            else DATASET_TIMEPOINTS.get(dataset)
        )
        dataset_mesh_paths = discover_mesh_paths(
            data_dir=str(paths["mesh_data_dir"]),
            timepoints=timepoints,
            zarr_names=mesh_cfg["zarr_name_by_tp"],
            rounds=mesh_cfg["round_by_tp"],
            meshes=mesh_cfg["meshname_by_tp"],
            wells=mesh_cfg.get("wells_by_tp", {}),
        )
        blacklist = load_optional_blacklist(
            paths["blacklist_path"],
            label=f"{dataset} blacklist",
            verbose=verbose,
        )
        dataset_runs[dataset] = {
            "timepoints": timepoints,
            "paths": paths,
            "blacklist": blacklist,
            "mesh_files_found": len(dataset_mesh_paths),
        }
        mesh_records.extend(
            {"dataset": dataset, "mesh_path": mesh_path}
            for mesh_path in dataset_mesh_paths
        )

    if args.max_meshes is not None:
        mesh_records = mesh_records[: int(args.max_meshes)]

    OrganoidMesh = None
    compute_geodesics_dijkstra = None
    if not args.dry_run:
        from organograph.mesh.OrganoidMesh import OrganoidMesh
        from organograph.mesh.geodesics import compute_geodesics_dijkstra

    if verbose:
        for dataset, run in dataset_runs.items():
            print(
                f"[skeleton-export] {dataset}: found "
                f"{run['mesh_files_found']} mesh files"
            )
        print(f"[skeleton-export] combined total: {len(mesh_records)} mesh files")
        print(f"[skeleton-export] output root: {output_root}")

    stats = {
        "mesh_files_found": int(len(mesh_records)),
        "mesh_files_found_by_dataset": {
            dataset: int(run["mesh_files_found"])
            for dataset, run in dataset_runs.items()
        },
        "exported": 0,
        "skipped_blacklist": 0,
        "skipped_whitelist": 0,
        "skipped_existing": 0,
        "skipped_branched": 0,
        "failed": 0,
        "dry_run": bool(args.dry_run),
    }
    manifest_rows: list[dict] = []
    t_start = time.perf_counter()

    for mesh_record in mesh_records:
        dataset = mesh_record["dataset"]
        mesh_path = mesh_record["mesh_path"]
        dataset_run = dataset_runs[dataset]
        blacklist = dataset_run["blacklist"]
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

            organoid_dir = output_root / dataset / timepoint / label_uid
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
                "dataset": dataset,
                "timepoint": timepoint,
                "well": rec.get("well"),
                "organoid_id": rec.get("organoid_id"),
                "label_uid": label_uid,
                "mesh_path": str(mesh_path),
                "vocab_path": str(shared_paths["vocab_path"]),
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
            summary = graph_summary(primitive_result.graph)
            has_branches = summary["n_branches"] > 0
            if args.unbranched_only and has_branches:
                stats["skipped_branched"] += 1
                if verbose:
                    print(f"[skip branched] {label_uid}")
                continue
            export_paths = save_shape_export(
                primitive_result,
                organoid_dir,
                metadata=metadata,
            )
            dt = time.perf_counter() - t0

            manifest_rows.append(
                {
                    "schema_version": SHAPE_EXPORT_SCHEMA_VERSION,
                    "dataset": dataset,
                    "timepoint": timepoint,
                    "well": rec.get("well"),
                    "organoid_id": rec.get("organoid_id"),
                    "label_uid": label_uid,
                    "mesh_path": str(mesh_path),
                    "output_dir": str(organoid_dir),
                    "json_path": export_paths.get("json", ""),
                    "has_branches": has_branches,
                    "vae_eligible": not has_branches,
                    **summary,
                    "elapsed_s": f"{dt:.3f}",
                }
            )
            stats["exported"] += 1
            if verbose:
                print(
                    f"[skeleton-export] {label_uid}: nodes={summary['n_nodes']} "
                    f"edges={summary['n_edges']} primitives={summary['n_primitives']} "
                    f"in {dt:.2f}s"
                )

        except Exception as exc:
            stats["failed"] += 1
            append_failure(failure_log, label_uid=label_uid, mesh_path=str(mesh_path), error=exc)
            if verbose:
                print(f"[failed] {label_uid}: {type(exc).__name__}: {exc}")
            if args.strict:
                raise

    write_manifest(output_root / "manifest.csv", manifest_rows)
    write_export_readme(output_root / "README.md", dataset=", ".join(datasets))
    elapsed_s = time.perf_counter() - t_start
    write_run_settings(
        output_root,
        script_name=os.path.basename(__file__),
        payload={
            "shape_export_schema": SHAPE_EXPORT_SCHEMA_VERSION,
            "datasets": {
                dataset: {
                    "timepoints": run["timepoints"],
                    "mesh_files_found": int(run["mesh_files_found"]),
                    "paths": {
                        key: str(value) if value is not None else None
                        for key, value in run["paths"].items()
                    },
                }
                for dataset, run in dataset_runs.items()
            },
            "paths": {
                "data_root": str(shared_paths["data_root"]),
                "vocab_path": str(shared_paths["vocab_path"]),
                "output_root": str(output_root),
                "whitelist_path": args.whitelist_path,
            },
            "parameters": {
                "overwrite": bool(args.overwrite),
                "dry_run": bool(args.dry_run),
                "strict": bool(args.strict),
                "max_meshes": args.max_meshes,
                "unbranched_only": bool(args.unbranched_only),
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

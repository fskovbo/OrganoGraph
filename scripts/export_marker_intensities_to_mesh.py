#!/usr/bin/env python3
"""
Export processed cell marker intensities onto organoid meshes.

Inputs
------
This script consumes the outputs of scripts/run_graph_preprocess.py:

    {GRAPHS_DIR}/{timepoint}/{label_uid}.gpickle
    {GRAPHS_DIR}/{timepoint}/{label_uid}.vertex_owner.npz
    {GRAPHS_DIR}/{timepoint}/index.csv

Outputs
-------
For each organoid:

    {OUT_DIR}/vtp/{timepoint}/{label_uid}.vtp

The VTP contains the original mesh geometry plus point-data fields:

    - cell_owner_table_index
    - one scalar marker intensity field per marker, in marker_names.csv order

Summary files:

    {OUT_DIR}/marker_occurrence.csv
    {OUT_DIR}/marker_names.csv
    {OUT_DIR}/marker_names.json
    {OUT_DIR}/README.md

The occurrence CSV counts processed marker-positive cells from graph
``markers_bin``. The VTP intensity fields use processed ``markers_int``.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

from organograph.graph.access import graph_get
from organograph.graph.io import load_cell_graph
from organograph.mesh.OrganoidMesh import OrganoidMesh
from organograph.io_utils.run_metadata import write_run_settings


# =============================================================================
# CONFIG
# =============================================================================

DATASET = "20251201"  # "20250929" or "20251201"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_ROOT = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET)

GRAPHS_DIR = os.path.join(DATASET_ROOT, "graphs_preprocessed")
OUT_DIR = os.path.join(DATASET_ROOT, "marker_intensities_mesh")

# If None, use all timepoint directories found in GRAPHS_DIR.
TIMEPOINTS = None

OVERWRITE = True
VERBOSE = True
DRY_RUN = False
MAX_ORGANOIDS = None

# VTP field naming. Marker names themselves are written to marker_names.csv/json.
MARKER_FIELD_SUFFIX = "_intensity"
OWNER_FIELD_NAME = "cell_owner_table_index"
UNASSIGNED_OWNER = -1


# =============================================================================
# VTP WRITER
# =============================================================================

def write_mesh_vtp(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    point_fields: dict[str, np.ndarray],
) -> None:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))

    packed_faces = np.column_stack(
        [np.full(len(faces), 3, dtype=np.int64), faces]
    ).reshape(-1)
    cells = vtk.vtkCellArray()
    cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(packed_faces, deep=True))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    for name, values in point_fields.items():
        values = np.asarray(values)
        if values.shape[0] != len(vertices):
            raise ValueError(
                f"VTP point field '{name}' has {values.shape[0]} rows; expected {len(vertices)}"
            )
        vtk_array = numpy_to_vtk(values, deep=True)
        vtk_array.SetName(str(name))
        polydata.GetPointData().AddArray(vtk_array)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        raise OSError(f"VTK failed to write {path}")


# =============================================================================
# HELPERS
# =============================================================================

def sanitize_field_name(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    out = re.sub(r"_+", "_", out)
    if not out:
        out = "marker"
    if out[0].isdigit():
        out = f"marker_{out}"
    return out


def marker_field_names(marker_names: list[str]) -> list[str]:
    fields = []
    used = set()
    for i, marker in enumerate(marker_names):
        base = f"{sanitize_field_name(marker)}{MARKER_FIELD_SUFFIX}"
        field = base
        if field in used:
            field = f"{base}_{i}"
        used.add(field)
        fields.append(field)
    return fields


def selected_timepoints() -> list[str]:
    if TIMEPOINTS is not None:
        return list(TIMEPOINTS)
    if not os.path.isdir(GRAPHS_DIR):
        return []
    return sorted(
        name
        for name in os.listdir(GRAPHS_DIR)
        if os.path.isdir(os.path.join(GRAPHS_DIR, name))
    )


def iter_graph_index_rows():
    for tp in selected_timepoints():
        index_path = os.path.join(GRAPHS_DIR, tp, "index.csv")
        if not os.path.exists(index_path):
            if VERBOSE:
                print(f"[warn] missing graph index: {index_path}")
            continue

        df = pd.read_csv(index_path)
        for _, row in df.iterrows():
            rec = {str(k): row[k] for k in df.columns}
            rec["timepoint"] = str(tp)
            rec["index_path"] = index_path
            yield rec


def infer_vertex_owner_path(row: dict) -> str:
    path = row.get("vertex_owner_path", None)
    if isinstance(path, str) and path and os.path.exists(path):
        return path

    graph_path = str(row["graph_path"])
    stem = os.path.splitext(os.path.basename(graph_path))[0]
    return os.path.join(os.path.dirname(graph_path), f"{stem}.vertex_owner.npz")


def remap_owner_to_node(owner_table, G):
    """Map sidecar table-row ownership to graph node ids for marker lookup."""
    old_to_node = {
        int(G.nodes[node]["cell_index"]): int(node)
        for node in G.nodes
        if "cell_index" in G.nodes[node]
    }
    owner_table = np.asarray(owner_table, dtype=np.int64).reshape(-1)
    owner_node = np.full(owner_table.shape, UNASSIGNED_OWNER, dtype=np.int64)
    valid = owner_table >= 0
    if np.any(valid):
        valid_idx = np.flatnonzero(valid)
        for old_idx in np.unique(owner_table[valid]):
            node = old_to_node.get(int(old_idx), UNASSIGNED_OWNER)
            if node >= 0:
                owner_node[valid_idx[owner_table[valid] == old_idx]] = node
    return owner_node


def project_markers_to_vertices(owner_node, markers_int):
    markers_int = np.asarray(markers_int, dtype=np.float32)
    owner_node = np.asarray(owner_node, dtype=np.int64).reshape(-1)
    out = np.zeros((owner_node.size, markers_int.shape[1]), dtype=np.float32)
    valid = (owner_node >= 0) & (owner_node < markers_int.shape[0])
    out[valid] = markers_int[owner_node[valid]]
    return out


def write_marker_name_files(out_dir: str, marker_names: list[str], vtp_fields: list[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = [
        {"marker_index": i, "marker_name": marker, "vtp_field": field}
        for i, (marker, field) in enumerate(zip(marker_names, vtp_fields))
    ]
    csv_path = os.path.join(out_dir, "marker_names.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["marker_index", "marker_name", "vtp_field"])
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(out_dir, "marker_names.json"), "w") as f:
        json.dump(rows, f, indent=2)


def write_occurrence_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(out_dir: str, marker_names: list[str], vtp_fields: list[str]) -> None:
    marker_rows = "\n".join(
        f"| {i} | `{marker}` | `{field}` |"
        for i, (marker, field) in enumerate(zip(marker_names, vtp_fields))
    )
    text = f"""# Marker Intensity Mesh Export

This folder contains processed cell marker intensities projected onto the original organoid surface meshes.

## Files

- `vtp/<timepoint>/<label_uid>.vtp`: one VTP mesh per organoid.
- `marker_occurrence.csv`: one row per organoid with counts of marker-positive cells.
- `marker_names.csv`: marker order and the matching VTP point-data field names.
- `marker_names.json`: JSON version of `marker_names.csv`.
- `export_status.csv`: export status for each attempted organoid.
- `run_settings_*.json`: script settings and run summary.

## VTP Point Data

Each VTP stores the original mesh geometry plus point-data arrays:

- `{OWNER_FIELD_NAME}`: integer owner of each mesh vertex, using the per-organoid cell-table index from the graph preprocessing Voronoi assignment. `-1` means no owner.
- Marker intensity fields: one scalar array per marker. Each mesh vertex receives the processed `markers_int` value of its owning cell. Vertices with owner `-1` or owners not present in the processed graph receive intensity `0`.

The marker intensity fields are ordered as follows:

| Index | Marker | VTP field |
| ---: | --- | --- |
{marker_rows}

## Marker Occurrence Table

`marker_occurrence.csv` contains:

- `dataset`, `timepoint`, `label_uid`: organoid identifiers.
- `mesh_path`, `graph_path`, `vertex_owner_path`, `vtp_path`: source and output paths used for the export.
- `n_cells`: number of processed graph cells.
- `n_mesh_vertices`: number of vertices in the mesh.
- `n_owned_mesh_vertices`: number of vertices with a non-negative `{OWNER_FIELD_NAME}`.
- `<marker>_n_pos`: number of processed graph cells positive for that marker.

Marker positivity is defined by processed graph `markers_bin`, where a cell is positive if its processed marker intensity is greater than zero after graph-level postprocessing.

## Notes

The VTP fields contain processed marker intensities, not raw unfiltered table values. Graph-level postprocessing may set selected marker intensities to zero, for example when coexpression suppression is enabled in `run_graph_preprocess.py`.
"""
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(text)


def process_one(row: dict, expected_marker_names: list[str] | None):
    label_uid = str(row["label_uid"])
    tp = str(row["timepoint"])
    graph_path = str(row["graph_path"])
    mesh_path = str(row["mesh_path"])
    vertex_owner_path = infer_vertex_owner_path(row)

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph not found: {graph_path}")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    if not os.path.exists(vertex_owner_path):
        raise FileNotFoundError(f"Vertex-owner sidecar not found: {vertex_owner_path}")

    G = load_cell_graph(graph_path)
    marker_names = [str(x) for x in G.graph.get("marker_names", [])]
    if expected_marker_names is not None and marker_names != expected_marker_names:
        raise ValueError(
            f"Marker names differ for {label_uid}. "
            f"Expected {expected_marker_names}, got {marker_names}"
        )

    markers_int = graph_get(G, "markers_int", dtype=float)
    markers_bin = graph_get(G, "markers_bin") > 0
    if markers_int.shape != markers_bin.shape:
        raise ValueError(
            f"markers_int shape {markers_int.shape} != markers_bin shape {markers_bin.shape}"
        )
    if markers_int.shape[1] != len(marker_names):
        raise ValueError(
            f"Marker matrix has {markers_int.shape[1]} columns but marker_names has {len(marker_names)}"
        )

    mesh = OrganoidMesh(mesh_path)
    side = np.load(vertex_owner_path, allow_pickle=True)
    owner_table_raw = np.asarray(side["vertex_owner"], dtype=np.int64)
    if owner_table_raw.shape[0] != mesh.v.shape[0]:
        raise ValueError(
            f"vertex_owner length {owner_table_raw.shape[0]} != mesh vertices {mesh.v.shape[0]} "
            f"for {label_uid}"
        )

    owner_table = owner_table_raw.astype(np.int64, copy=False).reshape(-1)
    owner_node = remap_owner_to_node(owner_table_raw, G)
    vertex_markers = project_markers_to_vertices(owner_node, markers_int)
    vtp_fields = marker_field_names(marker_names)

    point_fields = {OWNER_FIELD_NAME: owner_table.astype(np.int64)}
    for j, field in enumerate(vtp_fields):
        point_fields[field] = vertex_markers[:, j]

    out_dir = os.path.join(OUT_DIR, "vtp", tp)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label_uid}.vtp")

    if (not OVERWRITE) and os.path.exists(out_path):
        return marker_names, vtp_fields, None, {
            "label_uid": label_uid,
            "timepoint": tp,
            "status": "skipped_existing",
        }

    if not DRY_RUN:
        write_mesh_vtp(Path(out_path), mesh.v, mesh.f, point_fields)

    counts = markers_bin.sum(axis=0).astype(int)
    occurrence = {
        "dataset": DATASET,
        "timepoint": tp,
        "label_uid": label_uid,
        "mesh_path": mesh_path,
        "graph_path": graph_path,
        "vertex_owner_path": vertex_owner_path,
        "vtp_path": out_path,
        "n_cells": int(G.number_of_nodes()),
        "n_mesh_vertices": int(mesh.v.shape[0]),
        "n_owned_mesh_vertices": int(np.sum(owner_table >= 0)),
    }
    for marker, count in zip(marker_names, counts):
        occurrence[f"{marker}_n_pos"] = int(count)

    return marker_names, vtp_fields, occurrence, {
        "label_uid": label_uid,
        "timepoint": tp,
        "status": "exported" if not DRY_RUN else "dry_run",
    }


def main():
    t_start = time.perf_counter()
    os.makedirs(OUT_DIR, exist_ok=True)

    occurrence_rows = []
    status_rows = []
    marker_names_ref = None
    vtp_fields_ref = None
    stats = {
        "planned": 0,
        "exported": 0,
        "skipped_existing": 0,
        "failed": 0,
        "dry_run": bool(DRY_RUN),
    }

    for row in iter_graph_index_rows():
        stats["planned"] += 1
        if MAX_ORGANOIDS is not None and stats["planned"] > int(MAX_ORGANOIDS):
            break

        try:
            marker_names, vtp_fields, occurrence, status = process_one(row, marker_names_ref)
            if marker_names_ref is None:
                marker_names_ref = marker_names
                vtp_fields_ref = vtp_fields
            status_rows.append(status)
            if occurrence is not None:
                occurrence_rows.append(occurrence)
                stats["exported"] += 1
                if VERBOSE:
                    print(f"[ok] {status['timepoint']}/{status['label_uid']}")
            else:
                stats["skipped_existing"] += 1
                if VERBOSE:
                    print(f"[skip exists] {status['timepoint']}/{status['label_uid']}")
        except Exception as e:
            stats["failed"] += 1
            status_rows.append(
                {
                    "label_uid": str(row.get("label_uid", "")),
                    "timepoint": str(row.get("timepoint", "")),
                    "status": "failed",
                    "error": str(e),
                }
            )
            if VERBOSE:
                print(f"[failed] {row.get('timepoint', '')}/{row.get('label_uid', '')}: {e}")

    if marker_names_ref is None:
        raise RuntimeError("No organoids were exported; no marker names found.")

    write_marker_name_files(OUT_DIR, marker_names_ref, vtp_fields_ref)
    write_readme(OUT_DIR, marker_names_ref, vtp_fields_ref)
    write_occurrence_csv(os.path.join(OUT_DIR, "marker_occurrence.csv"), occurrence_rows)
    write_occurrence_csv(os.path.join(OUT_DIR, "export_status.csv"), status_rows)

    elapsed_s = time.perf_counter() - t_start
    write_run_settings(
        OUT_DIR,
        script_name=os.path.basename(__file__),
        payload={
            "dataset": DATASET,
            "timepoints": selected_timepoints(),
            "paths": {
                "graphs_dir": GRAPHS_DIR,
                "out_dir": OUT_DIR,
            },
            "parameters": {
                "overwrite": OVERWRITE,
                "dry_run": DRY_RUN,
                "max_organoids": MAX_ORGANOIDS,
                "owner_field_name": OWNER_FIELD_NAME,
                "marker_field_suffix": MARKER_FIELD_SUFFIX,
            },
            "marker_names": marker_names_ref,
            "vtp_marker_fields": vtp_fields_ref,
            "stats": stats,
            "elapsed_s": float(elapsed_s),
        },
        verbose=VERBOSE,
    )

    if VERBOSE:
        print(f"[done] exported={stats['exported']} failed={stats['failed']} elapsed={elapsed_s:.2f}s")
        print(f"[done] output: {OUT_DIR}")


if __name__ == "__main__":
    main()

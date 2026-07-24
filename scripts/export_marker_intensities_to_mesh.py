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
    {OUT_DIR}/cell_features_class_with_projection.csv
    {OUT_DIR}/marker_names.csv
    {OUT_DIR}/marker_names.json
    {OUT_DIR}/README.md

The occurrence CSV counts marker-positive cells in the exported marker space.
By default, the VTP intensity fields use processed graph ``markers_int``.
Optionally, collaborator-style mutually exclusive cell-type intensities can be
derived from the graph marker intensities before export.
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
from organograph.io_utils.cells_table import prepare_cells_table
from organograph.io_utils.dataset_config import load_cell_table_config
from organograph.io_utils.run_metadata import write_run_settings


# =============================================================================
# CONFIG
# =============================================================================

DATASET = "20251201"  # "20250929" or "20251201"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_ROOT = os.path.join(PROJECT_ROOT, "..", "NicoleData", DATASET)

GRAPHS_DIR = os.path.join(DATASET_ROOT, "graphs_preprocessed")
OUT_DIR = os.path.join(DATASET_ROOT, "marker_intensities_mesh_exclusive")
CELLS_CSV = os.path.join(DATASET_ROOT, "feature_tables", "cell_features_class.csv")
CELL_CONFIG_PATH = os.path.join(DATASET_ROOT, "cell_table_config.json")

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
PROJECTED_COL = "projected_to_mesh"
PROJECTED_VERTEX_COL = "proj_vertex"
ENRICHED_CELL_TABLE_NAME = "cell_features_class_with_projection.csv"

# Optional collaborator-style marker exclusivity. Graphs loaded from
# ``graphs_preprocessed`` should use the processed ``markers_int`` field. Keep
# the raw-marker option disabled unless deliberately auditing the stored raw
# provenance fields.
APPLY_COLLABORATOR_EXCLUSIVITY = True
COLLABORATOR_EXCLUSIVITY_USE_RAW_GRAPH_MARKERS = False
# TA markers differ between panels. In "auto" mode, TA uses Cyclin D/A when
# available and falls back to KI67 when the panel has no Cyclin D/A columns.
COLLABORATOR_TA_MARKER_MODE = "auto"  # "auto", "cyclins", "ki67", or "all"
COLLABORATOR_INCLUDE_KI67_AS_TA = True
RAW_MARKERS_INT_FIELD = "markers_int_raw"

COLLABORATOR_CLASSES = {
    "LGR": "0.C02",
    "CHROMA": "0.C03",
    "CYCD": "0.C04",
    "MUC": "1.C03",
    "ALDOB": "1.C04",
    "GLUC": "2.C02",
    "CYCA": "2.C03",
    "AGR": "2.C04",
    "SERO": "3.C02",
    "LYZ": "3.C03",
}

COLLABORATOR_MARKER_PREFIXES = {
    # Prefixes are only fallbacks; marker-name aliases are preferred. Some
    # channel prefixes differ between panels, and some are ambiguous across
    # panels, so the graph/cell-table marker names remain the primary signal.
    "LGR": ["0.C02"],
    "CHROMA": ["0.C03"],
    "CYCD": ["0.C04"],
    "MUC": ["1.C03"],
    "ALDOB": ["1.C04", "0.C04"],
    "GLUC": ["2.C02"],
    "CYCA": ["2.C03"],
    "AGR": ["2.C04", "1.C04"],
    "SERO": ["3.C02", "1.C02"],
    "LYZ": ["3.C03", "1.C03"],
    "KI67": ["2.C04"],
}

COLLABORATOR_MARKER_ALIASES = {
    "LGR": ["LGR5", "LGR"],
    "CHROMA": ["Chroma", "CHROMA"],
    "CYCD": ["Cyclin D", "CYCD"],
    "MUC": ["Mucin 2", "MUC"],
    "ALDOB": ["AldoB", "ALDOB"],
    "GLUC": ["Glucagon", "GLUC"],
    "CYCA": ["Cyclin A", "CYCA"],
    "AGR": ["Agr2", "AGR"],
    "SERO": ["Serotonin", "SERO"],
    "LYZ": ["Lysozyme", "LYZ"],
    "KI67": ["KI67"],
}

COLLABORATOR_MUTUALLY_INCLUDE = {
    "STEM": ["LGR"],
    "EEPROG": ["CHROMA"],
    "GOBLET": ["MUC"],
    "ABS": ["ALDOB"],
    "EE": ["GLUC"],
    "SECPROG": ["AGR"],
    "EC": ["SERO"],
    "PANETH": ["LYZ"],
    "TA": ["CYCD", "CYCA"],
}

COLLABORATOR_MUTUALLY_EXCLUDE = {
    "STEM": ["CHROMA", "MUC", "ALDOB", "GLUC", "AGR", "SERO", "LYZ"],
    "EEPROG": ["MUC", "GLUC", "SERO", "LYZ"],
    "GOBLET": ["CHROMA", "GLUC", "SERO", "LYZ"],
    "ABS": ["CHROMA", "MUC", "GLUC", "AGR", "SERO", "LYZ"],
    "EE": ["SERO"],
    "SECPROG": ["CHROMA", "MUC", "GLUC", "SERO", "LYZ"],
    "EC": [],
    "PANETH": ["CHROMA", "GLUC", "SERO"],
    "TA": ["LGR", "CHROMA", "MUC", "ALDOB", "GLUC", "AGR", "SERO", "LYZ"],
}


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


def graph_has_node_field(G, field: str) -> bool:
    return G.number_of_nodes() > 0 and field in G.nodes[0]


def graph_dir_is_preprocessed(graphs_dir: str) -> bool:
    return os.path.basename(os.path.normpath(str(graphs_dir))) == "graphs_preprocessed"


def collaborator_ta_include_markers(
    marker_index_by_code: dict[str, list[int]] | None = None,
) -> list[str]:
    mode = str(COLLABORATOR_TA_MARKER_MODE).strip().lower()
    if mode not in {"auto", "cyclins", "ki67", "all"}:
        raise ValueError(
            "COLLABORATOR_TA_MARKER_MODE must be one of "
            "'auto', 'cyclins', 'ki67', or 'all'"
        )

    if mode == "cyclins":
        markers = ["CYCD", "CYCA"]
    elif mode == "ki67":
        markers = ["KI67"]
    elif mode == "all":
        markers = ["CYCD", "CYCA", "KI67"]
    else:
        marker_index_by_code = marker_index_by_code or {}
        has_cyclins = bool(marker_index_by_code.get("CYCD")) or bool(marker_index_by_code.get("CYCA"))
        has_ki67 = bool(marker_index_by_code.get("KI67"))
        if has_cyclins:
            markers = [
                code
                for code in ("CYCD", "CYCA")
                if bool(marker_index_by_code.get(code))
            ]
            if COLLABORATOR_INCLUDE_KI67_AS_TA and has_ki67:
                markers.append("KI67")
        elif has_ki67:
            markers = ["KI67"]
        else:
            markers = ["CYCD", "CYCA"]

    if COLLABORATOR_INCLUDE_KI67_AS_TA and mode == "cyclins":
        markers = list(markers) + ["KI67"]
    return list(dict.fromkeys(markers))


def collaborator_include_rules(
    marker_index_by_code: dict[str, list[int]] | None = None,
) -> dict[str, list[str]]:
    rules = {
        cell_type: list(markers)
        for cell_type, markers in COLLABORATOR_MUTUALLY_INCLUDE.items()
    }
    rules["TA"] = collaborator_ta_include_markers(marker_index_by_code)
    return rules


def resolve_collaborator_marker_indices(
    marker_names: list[str],
    marker_cols: list[str],
) -> dict[str, list[int]]:
    """Resolve collaborator marker codes to graph marker columns."""
    norm_name_to_idx = {
        str(name).strip().lower(): i
        for i, name in enumerate(marker_names)
    }
    alias_owner = {}
    for alias_code, aliases in COLLABORATOR_MARKER_ALIASES.items():
        for alias in aliases:
            alias_owner[str(alias).strip().lower()] = alias_code
    out = {}

    all_codes = list(dict.fromkeys([
        *COLLABORATOR_CLASSES,
        *COLLABORATOR_MARKER_PREFIXES,
        *COLLABORATOR_MARKER_ALIASES,
    ]))
    for code in all_codes:
        indices = []
        for alias in COLLABORATOR_MARKER_ALIASES.get(code, [code]):
            idx = norm_name_to_idx.get(str(alias).strip().lower())
            if idx is not None:
                indices.append(idx)

        if not indices:
            prefixes = COLLABORATOR_MARKER_PREFIXES.get(
                code,
                [COLLABORATOR_CLASSES[code]] if code in COLLABORATOR_CLASSES else [],
            )
            for i, col in enumerate(marker_cols):
                col = str(col)
                if not any(col == prefix or col.startswith(f"{prefix}.") for prefix in prefixes):
                    continue
                name_owner = alias_owner.get(str(marker_names[i]).strip().lower())
                if name_owner is None or name_owner == code:
                    indices.append(i)
        out[code] = list(dict.fromkeys(indices))

    return out


def collaborator_marker_codes_to_indices(
    codes: list[str],
    marker_index_by_code: dict[str, list[int]],
) -> list[int]:
    indices = []
    for code in codes:
        indices.extend(marker_index_by_code.get(code, []))
    return list(dict.fromkeys(indices))


def apply_collaborator_exclusivity_to_intensities(
    markers_int: np.ndarray,
    marker_names: list[str],
    marker_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Convert marker intensities to collaborator-defined cell-type intensities.

    For each cell type, a cell is included when any include marker has intensity
    > 0 and all available exclude markers have intensity == 0. The exported
    value is the mean intensity across the available include-marker columns.
    """
    X = np.asarray(markers_int, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"markers_int must be 2D (N,K); got shape {X.shape}")
    if X.shape[1] != len(marker_names) or len(marker_cols) != len(marker_names):
        raise ValueError(
            "markers_int, marker_names, and marker_cols must describe the same marker columns"
        )

    marker_index_by_code = resolve_collaborator_marker_indices(marker_names, marker_cols)
    include_rules = collaborator_include_rules(marker_index_by_code)
    output_names = list(include_rules)
    output = np.zeros((X.shape[0], len(output_names)), dtype=float)

    for out_idx, cell_type in enumerate(output_names):
        include_idx = collaborator_marker_codes_to_indices(
            include_rules[cell_type],
            marker_index_by_code,
        )
        exclude_idx = collaborator_marker_codes_to_indices(
            COLLABORATOR_MUTUALLY_EXCLUDE.get(cell_type, []),
            marker_index_by_code,
        )

        if include_idx:
            include_values = X[:, include_idx]
            include_mask = np.any(include_values > 0, axis=1)
        else:
            include_values = np.zeros((X.shape[0], 0), dtype=float)
            include_mask = np.zeros(X.shape[0], dtype=bool)

        if exclude_idx:
            exclude_mask = np.all(X[:, exclude_idx] == 0, axis=1)
        else:
            exclude_mask = np.ones(X.shape[0], dtype=bool)

        final_mask = include_mask & exclude_mask
        if np.any(final_mask):
            output[final_mask, out_idx] = np.mean(include_values[final_mask], axis=1)

    return output, output > 0, output_names, output_names


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


def filtered_marker_col(marker_col: str) -> str:
    marker_col = str(marker_col)
    suffix = "percentile99_class"
    if marker_col.endswith(suffix):
        return f"{marker_col}_filtered"
    return f"{marker_col}_filtered"


def build_cell_projection_updates(G, markers_int, marker_cols: list[str]) -> dict[tuple[str, int], dict]:
    updates = {}
    label_uid = str(G.graph.get("label_uid", ""))
    markers_int = np.asarray(markers_int, dtype=float)
    proj_vertices = graph_get(G, "proj_vertex", dtype=np.int64)

    for pos, node in enumerate(range(G.number_of_nodes())):
        node_data = G.nodes[node]
        cell_index = int(node_data.get("cell_index", node))
        update = {
            PROJECTED_COL: True,
            PROJECTED_VERTEX_COL: int(proj_vertices[pos]),
        }
        for j, marker_col in enumerate(marker_cols):
            update[filtered_marker_col(marker_col)] = float(markers_int[pos, j])
        updates[(label_uid, cell_index)] = update
    return updates


def write_enriched_cell_table(
    *,
    path: str,
    source_csv: str,
    updates: dict[tuple[str, int], dict],
    marker_cols: list[str],
    label_col: str = "label_uid",
) -> int:
    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"Cell feature table not found: {source_csv}")

    df = pd.read_csv(source_csv)
    if label_col not in df.columns:
        raise KeyError(f"Cell feature table missing required column '{label_col}': {source_csv}")

    df = df.copy()
    df["_original_row_index"] = np.arange(len(df), dtype=np.int64)
    df_prepared = prepare_cells_table(df, label_col=label_col)
    df_prepared = df_prepared.copy()
    df_prepared["_organoid_cell_index"] = df_prepared.groupby(level=0).cumcount()
    cell_index_by_original_row = df_prepared.set_index("_original_row_index")["_organoid_cell_index"]
    df["_organoid_cell_index"] = df["_original_row_index"].map(cell_index_by_original_row).astype(int)

    filtered_cols = [filtered_marker_col(c) for c in marker_cols]
    df[PROJECTED_COL] = False
    df[PROJECTED_VERTEX_COL] = np.int64(UNASSIGNED_OWNER)
    for col in filtered_cols:
        df[col] = np.nan

    for row_idx, label_uid, cell_index in zip(
        df.index,
        df[label_col].astype(str),
        df["_organoid_cell_index"].astype(int),
    ):
        update = updates.get((label_uid, int(cell_index)))
        if update is None:
            continue
        for col, value in update.items():
            df.at[row_idx, col] = value

    n_projected = int(df[PROJECTED_COL].sum())
    df = df.drop(columns=["_organoid_cell_index", "_original_row_index"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return n_projected


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
    if APPLY_COLLABORATOR_EXCLUSIVITY:
        marker_source_text = (
            "The VTP fields contain collaborator-exclusivity cell-type intensities "
            "derived from graph marker intensities. A cell type is positive when any "
            "included marker is > 0 and all available excluded markers are == 0; the "
            "exported value is the mean of the available included-marker intensities."
        )
        marker_positive_text = (
            "Marker positivity is defined by the exported class intensity being greater than zero."
        )
        enriched_marker_text = (
            f"- `<cell-type>_filtered`: exported collaborator-exclusivity class intensity "
            f"for each projected cell."
        )
    else:
        marker_source_text = (
            "The VTP fields contain processed marker intensities, not raw unfiltered table values. "
            "Graph-level postprocessing may set selected marker intensities to zero, for example "
            "when coexpression suppression is enabled in `run_graph_preprocess.py`."
        )
        marker_positive_text = (
            "Marker positivity is defined by processed graph `markers_bin`, where a cell is "
            "positive if its processed marker intensity is greater than zero after graph-level postprocessing."
        )
        enriched_marker_text = (
            f"- `<marker-prefix>.percentile99_class_filtered`: processed marker intensity after "
            f"graph preprocessing and filtering. These columns use the same marker column prefixes "
            f"as `cell_table_config.json`."
        )
    text = f"""# Marker Intensity Mesh Export

This folder contains processed cell marker intensities projected onto the original organoid surface meshes.

## Files

- `vtp/<timepoint>/<label_uid>.vtp`: one VTP mesh per organoid.
- `marker_occurrence.csv`: one row per organoid with counts of marker-positive cells.
- `{ENRICHED_CELL_TABLE_NAME}`: copy of the source cell feature table with projection and processed-marker columns appended.
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

{marker_positive_text}

## Enriched Cell Feature Table

`{ENRICHED_CELL_TABLE_NAME}` is a copy of the source cell feature table with extra columns appended:

- `{PROJECTED_COL}`: `True` when the cell was retained in the processed graph and projected to the mesh, otherwise `False`.
- `{PROJECTED_VERTEX_COL}`: mesh vertex id that the cell/nucleus was projected to, or `-1` when not projected.
{enriched_marker_text}

Rows for cells that were filtered out during projection/graph construction keep `{PROJECTED_COL}=False`, `{PROJECTED_VERTEX_COL}=-1`, and filtered marker intensity columns as missing values.

## Notes

{marker_source_text}
"""
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(text)


def process_one(
    row: dict,
    expected_marker_names: list[str] | None,
    marker_cols: list[str],
    config_marker_names: list[str],
):
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
    graph_marker_names = [str(x) for x in G.graph.get("marker_names", [])]

    markers_int = graph_get(G, "markers_int", dtype=float)
    markers_bin = graph_get(G, "markers_bin") > 0
    if markers_int.shape != markers_bin.shape:
        raise ValueError(
            f"markers_int shape {markers_int.shape} != markers_bin shape {markers_bin.shape}"
        )
    if markers_int.shape[1] != len(graph_marker_names):
        raise ValueError(
            f"Marker matrix has {markers_int.shape[1]} columns but marker_names has {len(graph_marker_names)}"
        )
    if len(marker_cols) != len(graph_marker_names):
        raise ValueError(
            f"cell_table_config marker_cols has {len(marker_cols)} entries but graph marker_names has {len(graph_marker_names)}"
        )
    if [str(x) for x in config_marker_names] != graph_marker_names:
        raise ValueError(
            f"cell_table_config marker_names order does not match graph marker_names for {label_uid}. "
            f"Config={config_marker_names}; graph={graph_marker_names}"
        )

    marker_source_field = "markers_int"
    output_marker_cols = marker_cols
    marker_names = graph_marker_names
    if APPLY_COLLABORATOR_EXCLUSIVITY:
        if COLLABORATOR_EXCLUSIVITY_USE_RAW_GRAPH_MARKERS:
            if graph_dir_is_preprocessed(GRAPHS_DIR):
                raise ValueError(
                    "Refusing to use raw marker fields from graphs_preprocessed. "
                    "Set COLLABORATOR_EXCLUSIVITY_USE_RAW_GRAPH_MARKERS = False."
                )
            if graph_has_node_field(G, RAW_MARKERS_INT_FIELD):
                markers_int = graph_get(G, RAW_MARKERS_INT_FIELD, dtype=float)
                marker_source_field = RAW_MARKERS_INT_FIELD
        markers_int, markers_bin, marker_names, output_marker_cols = apply_collaborator_exclusivity_to_intensities(
            markers_int,
            graph_marker_names,
            marker_cols,
        )

    if expected_marker_names is not None and marker_names != expected_marker_names:
        raise ValueError(
            f"Exported marker names differ for {label_uid}. "
            f"Expected {expected_marker_names}, got {marker_names}"
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

    cell_updates = build_cell_projection_updates(G, markers_int, output_marker_cols)

    vtp_status = "exported"
    if (not OVERWRITE) and os.path.exists(out_path):
        vtp_status = "skipped_existing"
    elif DRY_RUN:
        vtp_status = "dry_run"
    else:
        write_mesh_vtp(Path(out_path), mesh.v, mesh.f, point_fields)

    return marker_names, vtp_fields, output_marker_cols, occurrence, cell_updates, {
        "label_uid": label_uid,
        "timepoint": tp,
        "status": vtp_status,
        "marker_source_field": marker_source_field,
    }


def main():
    t_start = time.perf_counter()
    os.makedirs(OUT_DIR, exist_ok=True)
    cell_cfg = load_cell_table_config(CELL_CONFIG_PATH)
    marker_cols = list(cell_cfg["marker_cols"])
    config_marker_names = list(cell_cfg.get("marker_names", marker_cols))

    occurrence_rows = []
    cell_projection_updates = {}
    status_rows = []
    marker_names_ref = None
    vtp_fields_ref = None
    output_marker_cols_ref = None
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
            marker_names, vtp_fields, output_marker_cols, occurrence, cell_updates, status = process_one(
                row,
                marker_names_ref,
                marker_cols,
                config_marker_names,
            )
            if marker_names_ref is None:
                marker_names_ref = marker_names
                vtp_fields_ref = vtp_fields
                output_marker_cols_ref = output_marker_cols
            status_rows.append(status)
            occurrence_rows.append(occurrence)
            cell_projection_updates.update(cell_updates)

            if status["status"] in ("exported", "dry_run"):
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
    if output_marker_cols_ref is None:
        raise RuntimeError("No organoids were exported; no output marker columns found.")

    write_marker_name_files(OUT_DIR, marker_names_ref, vtp_fields_ref)
    write_readme(OUT_DIR, marker_names_ref, vtp_fields_ref)
    write_occurrence_csv(os.path.join(OUT_DIR, "marker_occurrence.csv"), occurrence_rows)
    n_projected_cells = write_enriched_cell_table(
        path=os.path.join(OUT_DIR, ENRICHED_CELL_TABLE_NAME),
        source_csv=CELLS_CSV,
        updates=cell_projection_updates,
        marker_cols=output_marker_cols_ref,
    )
    write_occurrence_csv(os.path.join(OUT_DIR, "export_status.csv"), status_rows)
    stats["occurrence_rows_written"] = int(len(occurrence_rows))
    stats["cell_projection_updates"] = int(len(cell_projection_updates))
    stats["projected_cells_in_enriched_table"] = int(n_projected_cells)

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
                "cells_csv": CELLS_CSV,
                "cell_config_path": CELL_CONFIG_PATH,
            },
            "parameters": {
                "overwrite": OVERWRITE,
                "dry_run": DRY_RUN,
                "max_organoids": MAX_ORGANOIDS,
                "owner_field_name": OWNER_FIELD_NAME,
                "marker_field_suffix": MARKER_FIELD_SUFFIX,
                "apply_collaborator_exclusivity": APPLY_COLLABORATOR_EXCLUSIVITY,
                "collaborator_exclusivity_use_raw_graph_markers": COLLABORATOR_EXCLUSIVITY_USE_RAW_GRAPH_MARKERS,
                "collaborator_include_ki67_as_ta": COLLABORATOR_INCLUDE_KI67_AS_TA,
            },
            "marker_names": marker_names_ref,
            "vtp_marker_fields": vtp_fields_ref,
            "output_marker_cols": output_marker_cols_ref,
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

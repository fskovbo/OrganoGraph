#!/usr/bin/env python3
"""
Rebuild a unified GNN dataset directly from preprocessed organoid graphs.

This script bypasses the mesh segmentation pipeline and re-exports organoids in
Project-B format from already existing graph files plus the linked membrane mesh.

What it does
------------
0. Infer the common marker space across all selected datasets/timepoints and
   optionally apply marker harmonization rules so datasets with slightly
   different marker panels remain compatible.
1. Iterate over preprocessed graphs.
2. Resolve the corresponding membrane mesh from the graph metadata.
3. Check graph/mesh alignment via `ensure_mesh_graph_aligned(mesh, G)`.
4. Recompute Gaussian and mean curvature on the membrane mesh.
5. Recompute the Voronoi tessellation on the membrane from projected graph-cell
   centers and assign each graph node:
      - patch-averaged Gaussian curvature
      - patch-averaged mean curvature
      - patch surface area
6. Resolve the matching crypt-segmentation output (if available), load `d_crypts`,
   and project it from the membrane mesh to graph nodes while excluding cells
   with zero Voronoi area.
7. Export Project-B compatible `.npz` files + sidecars + a manifest CSV.

Expected exported format per organoid
-------------------------------------
- organoid_<label_uid>.npz          : x, y, edges, N, M, where y[:, 0] = Gaussian curvature and y[:, 1] = mean curvature
- organoid_<label_uid>_markers.json : common marker names
- organoid_<label_uid>_aux.json     : metadata sidecar

Important assumptions
---------------------
- Graph nodes contain:
    * markers_bin
    * proj_vertex
- Graph metadata contains:
    * mesh_path  (preferred)
      OR it can be recovered from the per-timepoint index.csv
- The following project utilities already exist in your codebase:
    * ensure_mesh_graph_aligned
    * voronoi_on_mesh_dijkstra
    * compute_gaussian_curvature
    * load_cell_graph
    * graph_get

Adapt only the import lines if your module layout differs.
"""

from __future__ import annotations

import os
import re
import json
import glob
import warnings
from pathlib import Path
from functools import lru_cache
from collections.abc import Iterable

import numpy as np
import pandas as pd
import networkx as nx

# -----------------------------------------------------------------------------
# Project imports: adapt these only if your module layout changed
# -----------------------------------------------------------------------------
from organograph.mesh.OrganoidMesh import OrganoidMesh
from organograph.graph.io import load_cell_graph
from organograph.graph.access import graph_get
from organograph.mesh.curvature_mean import compute_organoid_curvatures, integrate_curvature

from organograph.mesh.transform import ensure_mesh_graph_aligned
from organograph.projection.voronoi import voronoi_on_mesh_dijkstra


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Project directory that contains this scripts/ folder.
DATA_ROOT = os.path.join(PROJECT_ROOT, "..", "NicoleData")  # Root folder containing the dataset batches.

DEFAULT_GRAPH_SUBDIR = "graphs_preprocessed"  # Per-dataset folder containing preprocessed graph .gpickle files.
DEFAULT_INDEX_CSV_NAME = "index.csv"  # Per-timepoint graph index file used to recover mesh paths.
DEFAULT_TSNE_CSV_NAME = "tsne_results.csv"  # Optional per-dataset table with shape complexity metadata.
DEFAULT_SEGMENTATION_SUBDIR = "crypt_segmentations_mesh"  # Per-dataset folder with mesh crypt segmentation .npz files.

OUTPUT_DATASET_DIR_NAME = "GNN_training_data_curvature_patchavg_smooth"  # Name of the exported GNN dataset folder.
EXPORT_FILE_PREFIX = "organoid"  # Prefix for each exported organoid .npz and sidecar file.
MANIFEST_CSV_NAME = "manifest.csv"  # Name of the exported manifest table.
COMMON_MARKERS_JSON_NAME = "common_markers.json"  # Name of the exported common marker list.
MARKER_HARMONIZATION_JSON_NAME = "marker_harmonization.json"  # Name of the exported marker harmonization metadata.
FAILURES_LOG_NAME = "failures.log"  # Name of the log file listing organoids that failed to export.
AUX_JSON_SUFFIX = "_aux.json"  # Suffix for per-organoid metadata sidecars.
MARKERS_JSON_SUFFIX = "_markers.json"  # Suffix for per-organoid marker-name sidecars.

GRAPH_MARKERS_FIELD = "markers_bin"  # Node field containing the binary marker matrix.
GRAPH_PROJ_VERTEX_FIELD = "proj_vertex"  # Node field containing the projected mesh vertex id.
GRAPH_LABEL_UID_KEY = "label_uid"  # Graph metadata key storing the organoid/cell-table label.
GRAPH_MESH_PATH_KEY = "mesh_path"  # Graph metadata key storing the source mesh path.
GRAPH_MARKER_NAME_KEYS = ["marker_names", "markers", "marker_cols"]  # Metadata keys checked for marker names, in priority order.

TSNE_LABEL_COL = "label_uid"  # Column in the complexity table identifying organoids.
TSNE_COMPLEXITY_COL = "complexity"  # Column in the complexity table containing shape complexity values.
INDEX_LABEL_COL = "label_uid"  # Column in graph index.csv identifying organoids.
INDEX_MESH_PATH_COL = "mesh_path"  # Column in graph index.csv storing the source mesh path.
SEGMENTATION_DISTANCE_FIELD = "d_crypts"  # Mesh-segmentation field projected to graph nodes when available.

CURVATURE_LMAX = 12  # Maximum Laplace-Beltrami mode used for smoothed curvature estimation.
CURVATURE_DIFFUSION_SMOOTHEN_TIME = 0.8  # Diffusion smoothing time passed to curvature computation.
PRECOMPUTE_MESH_EIGENDECOMP = True  # Whether to compute the mesh eigendecomposition before curvature estimation.
CURVATURE_Y_COLUMNS = ["gaussian_curvature", "mean_curvature"]  # Column names for y[:, 0] and y[:, 1].
GAUSSIAN_CURVATURE_GRAPH_KEY = "curvature_gauss_graph"  # Graph metadata key for exported Gaussian curvature.
MEAN_CURVATURE_GRAPH_KEY = "curvature_mean_graph"  # Graph metadata key for exported mean curvature.
CURVATURE_Y_COLUMNS_GRAPH_KEY = "curvature_graph_y_columns"  # Graph metadata key storing y column names.
CELL_PATCH_AREA_KEY = "cell_patch_area"  # Graph/npz key storing Voronoi patch area per exported cell.
ZERO_AREA_THRESHOLD = 0.0  # Cells with patch area <= this threshold are removed before export.

DATA_SPECS = [
    {
        "dataset": "20250929",
        "graph_subdir": DEFAULT_GRAPH_SUBDIR,
        "timepoints": ["day3p5", "day4", "day4p5", "day4p5-more"],
        "label": "batch 1",
        "index_csv_name": DEFAULT_INDEX_CSV_NAME,
        # optional per-dataset complexity sidecar
        "tsne_csv": DEFAULT_TSNE_CSV_NAME,
        "segmentation_subdir": DEFAULT_SEGMENTATION_SUBDIR,
    },
    {
        "dataset": "20251201",
        "graph_subdir": DEFAULT_GRAPH_SUBDIR,
        "timepoints": ["day4p5"],
        "label": "batch 2",
        "index_csv_name": DEFAULT_INDEX_CSV_NAME,
        "tsne_csv": DEFAULT_TSNE_CSV_NAME,
        "segmentation_subdir": DEFAULT_SEGMENTATION_SUBDIR,
    },
]

OUT_DIR = os.path.join(DATA_ROOT, OUTPUT_DATASET_DIR_NAME)
OVERWRITE = True        # Replace existing organoid exports when True.
VERBOSE = True          # Print progress and skip/failure messages when True.
STRICT = False          # Raise on first organoid failure when True.
MIN_NODES = 50          # Skip organoids with fewer graph nodes than this.

# Optional marker-combination rules.
# Same idea as your current setup_GNN_dataset.py: derive a canonical marker when
# datasets differ in how they encode proliferation markers.
ENABLE_COMBINED_MARKERS = True  # Enable derived marker columns before finding the common marker set.
COMBINE_MARKER_RULES = [
    {
        "new_name": "KI67",
        "source_markers": ["Cyclin A", "Cyclin D"],
        "mode": "any",
    },
]


# -----------------------------------------------------------------------------
# EXPORT HELPERS (kept compatible with your current setup_GNN_dataset.py)
# -----------------------------------------------------------------------------
def _edges_from_graph(organoid_graph: nx.Graph, N_expected: int) -> np.ndarray:
    """Convert NetworkX graph to an undirected edge array (E,2) with u < v."""
    nodes = sorted(organoid_graph.nodes())
    if len(nodes) != N_expected:
        id_map = {old: new for new, old in enumerate(nodes)}
        N = len(nodes)
        warnings.warn(
            f"Graph node count {len(nodes)} != N={N_expected}; reindexing nodes 0..{N-1}."
        )
    else:
        id_map = None
        N = N_expected

    edges_uplow = set()
    for u, v in organoid_graph.edges():
        uu = id_map[u] if id_map is not None else u
        vv = id_map[v] if id_map is not None else v
        if uu == vv:
            continue
        a, b = (uu, vv) if uu < vv else (vv, uu)
        edges_uplow.add((int(a), int(b)))

    if not edges_uplow:
        return np.zeros((0, 2), dtype=np.int64)

    arr = np.array(sorted(edges_uplow), dtype=np.int64)
    if (arr[:, 0] < 0).any() or (arr[:, 1] >= N).any():
        raise ValueError("Edge indices out of range after reindex.")
    return arr



def save_aux_metadata(sidecar_path_no_ext: str, organoid_id: str, aux: dict) -> None:
    if not aux:
        return
    payload = {"organoid_id": str(organoid_id), **aux}
    json_path = f"{sidecar_path_no_ext}{AUX_JSON_SUFFIX}"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)



def export_organoid_npz(
    out_dir: str,
    organoid_id: str,
    y: np.ndarray,
    x_bin: np.ndarray,
    organoid_graph: nx.Graph,
    marker_names: list[str] | None = None,
    aux_meta: dict | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    X = np.asarray(x_bin, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"x must be 2-D (N,M); got {X.shape}")
    if not np.isin(X, [0.0, 1.0]).all():
        X = (X > 0).astype(np.float32)

    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError(f"y must be 1-D or 2-D with shape (N, C); got {y.shape}")

    N, M = X.shape
    if y.shape[0] != N:
        raise ValueError(f"y rows {y.shape[0]} != N={N}")
    if N <= 1:
        raise ValueError("N must be > 1 (need at least 2 cells).")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN/Inf.")

    edge_array = _edges_from_graph(organoid_graph, N_expected=N)

    file_stem = f"{EXPORT_FILE_PREFIX}_{organoid_id}"
    out_path = os.path.join(out_dir, f"{file_stem}.npz")
    payload = {
        "x": X,
        "y": y,
        "edges": edge_array,
        "N": np.int64(N),
        "M": np.int64(M),
    }
    if extra_arrays:
        for key, value in extra_arrays.items():
            payload[str(key)] = np.asarray(value)

    np.savez_compressed(out_path, **payload)

    if marker_names is not None:
        with open(os.path.join(out_dir, f"{file_stem}{MARKERS_JSON_SUFFIX}"), "w") as f:
            json.dump(list(marker_names), f, indent=2)
    if aux_meta:
        save_aux_metadata(os.path.join(out_dir, file_stem), organoid_id, aux_meta)

    return out_path


# -----------------------------------------------------------------------------
# SMALL UTILS
# -----------------------------------------------------------------------------
def _safe_array(x):
    if x is None:
        return None
    return np.asarray(x)



def _as_python(v):
    if isinstance(v, np.generic):
        return v.item()
    return v



def sanitize_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")



def patches_to_ll(patches: Iterable[Iterable[int]]) -> list[list[int]]:
    return [sorted(list(p)) for p in (patches or [])]

def _remap_vertex_owner_to_positions(vertex_owner: np.ndarray, G: nx.Graph) -> np.ndarray:
    """Convert Voronoi ownership to contiguous node positions [0, N)."""
    vertex_owner = np.asarray(vertex_owner, dtype=np.int64).reshape(-1)
    node_ids_sorted = np.array(sorted(G.nodes()), dtype=np.int64)
    node_id_to_pos = {int(nid): i for i, nid in enumerate(node_ids_sorted)}

    unique_valid = np.unique(vertex_owner[vertex_owner >= 0])
    if np.all(np.isin(unique_valid, node_ids_sorted)):
        owner_pos = np.full_like(vertex_owner, fill_value=-1)
        valid = vertex_owner >= 0
        owner_pos[valid] = np.array([node_id_to_pos[int(x)] for x in vertex_owner[valid]], dtype=np.int64)
        return owner_pos

    return vertex_owner.copy()


def prune_zero_area_nodes(
    G: nx.Graph,
    X: np.ndarray,
    y: np.ndarray,
    patch_area: np.ndarray,
    proj_vertex_ids: np.ndarray,
    vertex_owner: np.ndarray,
):
    """
    Drop cells with zero owned mesh area and return a reindexed graph plus mapping arrays.

    Returns a dict containing filtered graph/data and explicit old<->new mappings so
    downstream code can reconcile the exported PyG graph with the original graph.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    patch_area = np.asarray(patch_area, dtype=np.float64).reshape(-1)
    proj_vertex_ids = np.asarray(proj_vertex_ids, dtype=np.int64).reshape(-1)
    vertex_owner = np.asarray(vertex_owner, dtype=np.int64).reshape(-1)

    n_nodes = G.number_of_nodes()
    if X.shape[0] != n_nodes:
        raise ValueError(f"X rows {X.shape[0]} != graph nodes {n_nodes}")
    if y.shape[0] != n_nodes:
        raise ValueError(f"y rows {y.shape[0]} != graph nodes {n_nodes}")
    if patch_area.shape[0] != n_nodes:
        raise ValueError(f"patch_area length {patch_area.shape[0]} != graph nodes {n_nodes}")
    if proj_vertex_ids.shape[0] != n_nodes:
        raise ValueError(f"proj_vertex_ids length {proj_vertex_ids.shape[0]} != graph nodes {n_nodes}")

    original_node_ids = np.array(sorted(G.nodes()), dtype=np.int64)
    keep_mask = patch_area > ZERO_AREA_THRESHOLD
    delete_mask = ~keep_mask

    kept_node_ids = original_node_ids[keep_mask]
    deleted_node_ids = original_node_ids[delete_mask]

    old_to_new = np.full(n_nodes, -1, dtype=np.int64)
    old_to_new[np.flatnonzero(keep_mask)] = np.arange(int(keep_mask.sum()), dtype=np.int64)
    new_to_old = np.flatnonzero(keep_mask).astype(np.int64)

    G_kept = G.subgraph(kept_node_ids.tolist()).copy()
    relabel_map = {int(old_id): int(new_idx) for new_idx, old_id in enumerate(kept_node_ids.tolist())}
    G_pruned = nx.relabel_nodes(G_kept, relabel_map, copy=True)

    vertex_owner_pos = _remap_vertex_owner_to_positions(vertex_owner, G)
    vertex_owner_pruned = np.full_like(vertex_owner_pos, fill_value=-1)
    valid = (vertex_owner_pos >= 0) & (vertex_owner_pos < n_nodes)
    kept_valid = valid & keep_mask[vertex_owner_pos]
    vertex_owner_pruned[kept_valid] = old_to_new[vertex_owner_pos[kept_valid]]

    return {
        "graph": G_pruned,
        "x": np.asarray(X[keep_mask]),
        "y": y[keep_mask],
        "patch_area": patch_area[keep_mask],
        "proj_vertex_ids": proj_vertex_ids[keep_mask],
        "vertex_owner": vertex_owner_pruned,
        "keep_mask": keep_mask.astype(bool),
        "delete_mask": delete_mask.astype(bool),
        "kept_node_ids": kept_node_ids,
        "deleted_node_ids": deleted_node_ids,
        "old_to_new_index": old_to_new,
        "new_to_old_index": new_to_old,
        "num_deleted": np.int64(delete_mask.sum()),
        "original_num_nodes": np.int64(n_nodes),
        "filtered_num_nodes": np.int64(keep_mask.sum()),
    }



# -----------------------------------------------------------------------------
# PATH / DISCOVERY HELPERS
# -----------------------------------------------------------------------------
def build_dataset_paths(data_spec: dict, data_root: str) -> tuple[str, str]:
    dataset = data_spec["dataset"]
    dataset_root = os.path.join(data_root, dataset)
    graphs_dir = os.path.join(dataset_root, data_spec.get("graph_subdir", DEFAULT_GRAPH_SUBDIR))
    return dataset_root, graphs_dir



def iter_selected_graph_paths(data_specs: list[dict], data_root: str, verbose: bool = True):
    for ds in data_specs:
        dataset_root, graphs_dir = build_dataset_paths(ds, data_root)
        tps = ds.get("timepoints") or []

        for tp in tps:
            tp_dir = os.path.join(graphs_dir, tp)
            graph_paths = sorted(glob.glob(os.path.join(tp_dir, "*.gpickle")))

            if verbose and len(graph_paths) == 0:
                print(f"[warn] no graph files found in: {tp_dir}")

            for graph_path in graph_paths:
                yield {
                    "dataset": ds["dataset"],
                    "dataset_root": dataset_root,
                    "timepoint": tp,
                    "label": ds.get("label", f"{ds['dataset']}:{tp}"),
                    "graph_path": graph_path,
                    "graphs_dir": graphs_dir,
                    "index_csv_name": ds.get("index_csv_name", DEFAULT_INDEX_CSV_NAME),
                    "tsne_csv": ds.get("tsne_csv", DEFAULT_TSNE_CSV_NAME),
                    "segmentation_subdir": ds.get("segmentation_subdir"),
                }


# -----------------------------------------------------------------------------
# GRAPH / INDEX / METADATA HELPERS
# -----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def load_graph_cached(graph_path: str) -> nx.Graph:
    return load_cell_graph(graph_path)


@lru_cache(maxsize=None)
def load_tsne_table(tsne_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(tsne_csv_path):
        raise FileNotFoundError(f"{DEFAULT_TSNE_CSV_NAME} not found: {tsne_csv_path}")
    df = pd.read_csv(tsne_csv_path)
    df.columns = [str(c) for c in df.columns]
    return df


@lru_cache(maxsize=None)
def load_index_table(index_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(index_csv_path):
        raise FileNotFoundError(f"{DEFAULT_INDEX_CSV_NAME} not found: {index_csv_path}")
    df = pd.read_csv(index_csv_path)
    df.columns = [str(c) for c in df.columns]
    return df



def build_complexity_lookup(tsne_csv_path: str) -> dict[str, float]:
    df = load_tsne_table(tsne_csv_path)

    required_cols = {TSNE_LABEL_COL, TSNE_COMPLEXITY_COL}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in {tsne_csv_path}: {missing}. "
            f"Expected columns: {sorted(required_cols)}"
        )

    out = {}
    for _, row in df[[TSNE_LABEL_COL, TSNE_COMPLEXITY_COL]].dropna(subset=[TSNE_LABEL_COL]).iterrows():
        out[str(row[TSNE_LABEL_COL])] = (
            float(row[TSNE_COMPLEXITY_COL]) if pd.notna(row[TSNE_COMPLEXITY_COL]) else np.nan
        )
    return out



def _infer_label_uid(G: nx.Graph, fallback=None):
    return G.graph.get(GRAPH_LABEL_UID_KEY, None) or fallback



def _infer_marker_names(G: nx.Graph) -> list[str]:
    for k in GRAPH_MARKER_NAME_KEYS:
        v = G.graph.get(k, None)
        if v is not None:
            return [str(x) for x in v]

    markers_bin = extract_marker_matrix(G)
    n_markers = markers_bin.shape[1]
    return [f"marker_{i}" for i in range(n_markers)]



def resolve_mesh_path(G: nx.Graph, graph_path: str, dataset_root: str, timepoint: str, index_csv_name: str) -> str:
    # 1) prefer graph metadata
    mesh_path = G.graph.get(GRAPH_MESH_PATH_KEY, None)
    if mesh_path and os.path.exists(mesh_path):
        return str(mesh_path)

    # 2) recover from timepoint index.csv
    label_uid = _infer_label_uid(G, fallback=os.path.splitext(os.path.basename(graph_path))[0])
    index_csv_path = os.path.join(os.path.dirname(graph_path), index_csv_name)
    if not os.path.exists(index_csv_path):
        index_csv_path = os.path.join(dataset_root, os.path.basename(os.path.dirname(graph_path)), index_csv_name)

    df = load_index_table(index_csv_path)
    if INDEX_LABEL_COL not in df.columns or INDEX_MESH_PATH_COL not in df.columns:
        raise KeyError(
            f"{DEFAULT_INDEX_CSV_NAME} missing required columns "
            f"{[INDEX_LABEL_COL, INDEX_MESH_PATH_COL]}: {index_csv_path}"
        )

    hit = df.loc[df[INDEX_LABEL_COL].astype(str) == str(label_uid)]
    if len(hit) == 0:
        raise KeyError(f"Could not resolve mesh_path for label_uid={label_uid} from {index_csv_path}")

    mesh_path = str(hit.iloc[0][INDEX_MESH_PATH_COL])
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Resolved mesh_path does not exist: {mesh_path}")
    return mesh_path


def resolve_segmentation_dir(item: dict, dataset_root: str) -> str | None:
    seg_subdir = item.get("segmentation_subdir", None)
    if not seg_subdir:
        return None
    return os.path.join(dataset_root, seg_subdir)


def resolve_segmentation_path(
    *,
    segmentation_dir: str | None,
    timepoint: str,
    label_uid: str,
    mesh_path: str | None = None,
) -> str | None:
    """
    Resolve the saved crypt-segmentation `.npz` for a given organoid.

    Search order:
      1. <segmentation_dir>/<timepoint>/<label_uid>.npz
      2. <segmentation_dir>/<label_uid>.npz
      3. files that match the mesh stem under the timepoint directory
    """
    if not segmentation_dir:
        return None

    candidates: list[str] = []
    if timepoint:
        candidates.append(os.path.join(segmentation_dir, str(timepoint), f"{label_uid}.npz"))
    candidates.append(os.path.join(segmentation_dir, f"{label_uid}.npz"))

    mesh_stem = Path(mesh_path).stem if mesh_path else None
    if mesh_stem:
        if timepoint:
            candidates.extend(sorted(glob.glob(os.path.join(segmentation_dir, str(timepoint), f"{mesh_stem}*.npz"))))
        candidates.extend(sorted(glob.glob(os.path.join(segmentation_dir, f"{mesh_stem}*.npz"))))

    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return None


def load_segmentation_field(
    segmentation_path: str | None,
    field_name: str = SEGMENTATION_DISTANCE_FIELD,
) -> np.ndarray | None:
    if segmentation_path is None:
        return None

    with np.load(segmentation_path, allow_pickle=True) as seg:
        if field_name not in seg:
            return None
        field = np.asarray(seg[field_name])

    if field.ndim not in (1, 2):
        raise ValueError(
            f"Segmentation field '{field_name}' must have shape (V_mesh,) or (K, V_mesh); got {field.shape}"
        )
    return field


def project_mesh_field_to_graph(G, field_mesh, *, proj_field=GRAPH_PROJ_VERTEX_FIELD):
    """
    Project a mesh-based field to graph nodes via each node's projected mesh vertex.

    Parameters
    ----------
    G
        Cell graph whose nodes contain a mesh vertex index in `proj_field`.
    field_mesh : array_like
        Mesh-based field to project. Supported shapes:
          - (V_mesh,)      : one value per mesh vertex
          - (K, V_mesh)    : K values per mesh vertex (e.g. one row per crypt)
    proj_field : str
        Node attribute containing the projected mesh vertex index.

    Returns
    -------
    field_graph : ndarray
        Graph-level field evaluated at each node's projected mesh vertex.
        Shapes:
          - (N_nodes,)      if input shape is (V_mesh,)
          - (K, N_nodes)    if input shape is (K, V_mesh)

        Nodes with invalid proj_vertex get NaN.
    """
    F = np.asarray(field_mesh)

    if F.ndim not in (1, 2):
        raise ValueError(
            f"field_mesh must have shape (V_mesh,) or (K, V_mesh), got {F.shape}"
        )

    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        raise ValueError("Graph has no nodes")

    proj_vertex = graph_get(G, proj_field, dtype=np.int64)
    if proj_vertex.ndim != 1 or proj_vertex.size != n_nodes:
        raise ValueError("proj_vertex must be a 1D array of length N_nodes")

    V_mesh = F.shape[-1]
    valid = (proj_vertex >= 0) & (proj_vertex < V_mesh)

    if F.ndim == 1:
        out = np.full(n_nodes, np.nan, dtype=float)
        if np.any(valid):
            out[valid] = F[proj_vertex[valid]]
        return out

    out = np.full((F.shape[0], n_nodes), np.nan, dtype=float)
    if np.any(valid):
        out[:, valid] = F[:, proj_vertex[valid]]
    return out


def prune_projected_mesh_field(field_graph: np.ndarray | None, keep_mask: np.ndarray) -> np.ndarray | None:
    if field_graph is None:
        return None

    F = np.asarray(field_graph)
    keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)

    if F.ndim == 1:
        if F.shape[0] != keep_mask.shape[0]:
            raise ValueError(f"Projected field length {F.shape[0]} != keep_mask length {keep_mask.shape[0]}")
        return np.asarray(F[keep_mask], dtype=np.float64)

    if F.ndim == 2:
        if F.shape[1] != keep_mask.shape[0]:
            raise ValueError(
                f"Projected field shape {F.shape} incompatible with keep_mask length {keep_mask.shape[0]}"
            )
        return np.asarray(F[:, keep_mask], dtype=np.float64)

    raise ValueError(f"Projected field must be 1-D or 2-D, got {F.shape}")


# -----------------------------------------------------------------------------
# MARKER EXTRACTION / HARMONIZATION
# -----------------------------------------------------------------------------
def extract_marker_matrix(G: nx.Graph) -> np.ndarray:
    X = np.asarray(graph_get(G, GRAPH_MARKERS_FIELD))
    if X.ndim != 2:
        raise ValueError(f"{GRAPH_MARKERS_FIELD} must be 2-D (N, M); got shape {X.shape}")
    return (X > 0).astype(np.float32)



def build_effective_marker_representation(
    X: np.ndarray,
    marker_names: list[str],
    *,
    combine_rules: list[dict] | None = None,
    enable_combined_markers: bool = True,
) -> tuple[np.ndarray, list[str]]:
    X_eff = np.asarray(X, dtype=np.float32)
    names_eff = [str(m) for m in marker_names]

    if not enable_combined_markers or not combine_rules:
        return X_eff, names_eff

    for rule in combine_rules:
        new_name = str(rule["new_name"])
        src = [str(x) for x in rule.get("source_markers", [])]
        mode = str(rule.get("mode", "any"))

        name_to_idx = {name: i for i, name in enumerate(names_eff)}
        if not all(s in name_to_idx for s in src):
            continue

        src_idx = [name_to_idx[s] for s in src]
        X_src = X_eff[:, src_idx]

        if mode == "any":
            new_col = (X_src > 0).any(axis=1).astype(np.float32)
        else:
            raise NotImplementedError(f"Unsupported combine mode: {mode}")

        keep_mask = np.array([name not in src for name in names_eff], dtype=bool)
        X_eff = X_eff[:, keep_mask]
        names_eff = [n for n in names_eff if n not in src]

        if new_name in names_eff:
            j = names_eff.index(new_name)
            X_eff[:, j] = np.maximum(X_eff[:, j], new_col)
        else:
            X_eff = np.column_stack([X_eff, new_col])
            names_eff.append(new_name)

    return X_eff.astype(np.float32), names_eff



def subset_marker_matrix(X: np.ndarray, marker_names: list[str], keep_markers: Iterable[str]) -> np.ndarray:
    keep_markers = list(keep_markers)
    idx = {str(name): i for i, name in enumerate(marker_names)}
    missing = [m for m in keep_markers if m not in idx]
    if missing:
        raise KeyError(f"Requested markers missing from graph: {missing}")
    cols = [idx[m] for m in keep_markers]
    return np.asarray(X[:, cols], dtype=np.float32)



def infer_common_markers(data_specs: list[dict], data_root: str, verbose: bool = True) -> list[str]:
    marker_sets = []

    for item in iter_selected_graph_paths(data_specs, data_root, verbose=verbose):
        G = load_graph_cached(item["graph_path"])
        if G.number_of_nodes() < MIN_NODES:
            continue
        X = extract_marker_matrix(G)
        marker_names = _infer_marker_names(G)
        X, marker_names = build_effective_marker_representation(
            X,
            marker_names,
            combine_rules=COMBINE_MARKER_RULES,
            enable_combined_markers=ENABLE_COMBINED_MARKERS,
        )
        marker_sets.append(set(marker_names))

    if len(marker_sets) == 0:
        return []

    common = set.intersection(*marker_sets)
    return sorted(common)


# -----------------------------------------------------------------------------
# CURVATURE / PATCH ASSIGNMENT
# -----------------------------------------------------------------------------
def ensure_mesh_mass_matrix(mesh: OrganoidMesh) -> None:
    if getattr(mesh, "mass_matrix", None) is None:
        L, M = mesh.build_cotangent_laplacian_and_mass(mesh.v, mesh.f)
        mesh.laplacian = L
        mesh.mass_matrix = M



def compute_total_surface_area(mesh: OrganoidMesh) -> float:
    ensure_mesh_mass_matrix(mesh)
    vertex_areas = np.asarray(mesh.vertex_areas(from_mass_matrix=True), dtype=np.float64)
    return float(np.sum(vertex_areas))


def assign_patch_curvatures_and_area(
    mesh: OrganoidMesh,
    G: nx.Graph,
    curvature_gauss: np.ndarray,
    curvature_mean: np.ndarray,
):
    """
    Recompute Voronoi patches from projected cell centers on the mesh and assign
    each cell:
      - patch-averaged Gaussian curvature
      - patch-averaged mean curvature
      - patch area

    Returns
    -------
    node_curvatures : (N, 2) ndarray
        Area-weighted average curvatures over each cell patch.
        Column 0 is Gaussian curvature K, column 1 is mean curvature H.
    node_patch_area : (N,) ndarray
        Total patch area per cell.
    vertex_owner : (V_mesh,) ndarray
        Voronoi ownership on the mesh.
    dist_mat : any
        Raw distance matrix returned by voronoi_on_mesh_dijkstra.
    proj_vertex_ids : (N,) ndarray
        Mesh vertex id used as Voronoi seed for each graph node.
    """
    ensure_mesh_mass_matrix(mesh)
    vertex_areas = np.asarray(mesh.vertex_areas(from_mass_matrix=True), dtype=np.float64)
    curvature_gauss = np.asarray(curvature_gauss, dtype=np.float64).reshape(-1)
    curvature_mean = np.asarray(curvature_mean, dtype=np.float64).reshape(-1)

    n_vertices = mesh.v.shape[0]
    if curvature_gauss.shape[0] != n_vertices:
        raise ValueError(
            f"curvature_gauss length {curvature_gauss.shape[0]} != number of mesh vertices {n_vertices}"
        )
    if curvature_mean.shape[0] != n_vertices:
        raise ValueError(
            f"curvature_mean length {curvature_mean.shape[0]} != number of mesh vertices {n_vertices}"
        )

    proj_vertex_ids = np.asarray(graph_get(G, GRAPH_PROJ_VERTEX_FIELD), dtype=np.int64)
    n_nodes = G.number_of_nodes()
    if proj_vertex_ids.ndim != 1 or proj_vertex_ids.shape[0] != n_nodes:
        raise ValueError(f"{GRAPH_PROJ_VERTEX_FIELD} must be a 1D array of length N_nodes")

    if (proj_vertex_ids < 0).any() or (proj_vertex_ids >= n_vertices).any():
        raise ValueError(f"{GRAPH_PROJ_VERTEX_FIELD} contains invalid mesh vertex ids")

    vertex_owner, dist_mat = voronoi_on_mesh_dijkstra(mesh, proj_vertex_ids)
    vertex_owner = np.asarray(vertex_owner, dtype=np.int64).reshape(-1)
    if vertex_owner.shape[0] != n_vertices:
        raise ValueError("vertex_owner must have length V_mesh")

    owner_pos = _remap_vertex_owner_to_positions(vertex_owner, G)
    valid_owner = (owner_pos >= 0) & (owner_pos < n_nodes)

    patch_area = np.bincount(
        owner_pos[valid_owner],
        weights=vertex_areas[valid_owner],
        minlength=n_nodes,
    ).astype(np.float64)

    curvatures = np.column_stack([curvature_gauss, curvature_mean])
    node_curvatures = np.full((n_nodes, 2), np.nan, dtype=np.float64)
    nz = patch_area > 0

    for c in range(2):
        patch_curv_integral = np.bincount(
            owner_pos[valid_owner],
            weights=vertex_areas[valid_owner] * curvatures[valid_owner, c],
            minlength=n_nodes,
        ).astype(np.float64)
        node_curvatures[nz, c] = patch_curv_integral[nz] / patch_area[nz]

    return node_curvatures, patch_area, vertex_owner, dist_mat, proj_vertex_ids


# Backwards-compatible alias for old callers. New code should use
# assign_patch_curvatures_and_area().
def assign_patch_curvature_and_area(mesh: OrganoidMesh, G: nx.Graph, curvature_gauss: np.ndarray):
    zeros_mean = np.zeros_like(np.asarray(curvature_gauss, dtype=np.float64))
    node_curvatures, patch_area, vertex_owner, dist_mat, proj_vertex_ids = assign_patch_curvatures_and_area(
        mesh=mesh,
        G=G,
        curvature_gauss=curvature_gauss,
        curvature_mean=zeros_mean,
    )
    return node_curvatures[:, 0], patch_area, vertex_owner, dist_mat, proj_vertex_ids


# -----------------------------------------------------------------------------
# MAIN PER-ORGANOID PROCESSING
# -----------------------------------------------------------------------------
def process_one_organoid(item: dict, common_markers: list[str], out_dir: str) -> dict | None:
    graph_path = item["graph_path"]
    dataset = item["dataset"]
    dataset_root = item["dataset_root"]
    timepoint = item["timepoint"]

    G = load_graph_cached(graph_path)
    if G.number_of_nodes() < MIN_NODES:
        return None

    label_uid = _infer_label_uid(G, fallback=os.path.splitext(os.path.basename(graph_path))[0])

    mesh_path = resolve_mesh_path(
        G=G,
        graph_path=graph_path,
        dataset_root=dataset_root,
        timepoint=timepoint,
        index_csv_name=item["index_csv_name"],
    )

    mesh = OrganoidMesh(str(mesh_path))
    # mesh.normalize_inplace() # should be done automaticaally when checking if aligned
    mesh.label_uid = label_uid

    aligned = ensure_mesh_graph_aligned(mesh, G)
    if not aligned:
        raise ValueError(
            f"Mesh and graph are not aligned:\n"
            f"graph_path={graph_path}\nmesh_path={mesh_path}"
        )
    
    if PRECOMPUTE_MESH_EIGENDECOMP:
        mesh._eig_decomp()

    curvature_gauss, curvature_mean = compute_organoid_curvatures(
        mesh,
        lmax=CURVATURE_LMAX,
        diffusion_smoothen_time=CURVATURE_DIFFUSION_SMOOTHEN_TIME,
    )

    y, cell_patch_area, vertex_owner, _dist_mat, proj_vertex_ids = assign_patch_curvatures_and_area(
        mesh=mesh,
        G=G,
        curvature_gauss=curvature_gauss,
        curvature_mean=curvature_mean,
    )

    if not np.isfinite(cell_patch_area).all():
        raise ValueError(f"Patch areas contain NaN/Inf for {graph_path}")

    segmentation_dir = resolve_segmentation_dir(item, dataset_root)
    segmentation_path = resolve_segmentation_path(
        segmentation_dir=segmentation_dir,
        timepoint=timepoint,
        label_uid=str(label_uid),
        mesh_path=mesh_path,
    )
    d_crypts_mesh = load_segmentation_field(segmentation_path, field_name=SEGMENTATION_DISTANCE_FIELD)
    d_crypts_graph_full = project_mesh_field_to_graph(G, d_crypts_mesh) if d_crypts_mesh is not None else None

    marker_names = _infer_marker_names(G)
    X_all = extract_marker_matrix(G)
    X_all, marker_names = build_effective_marker_representation(
        X_all,
        marker_names,
        combine_rules=COMBINE_MARKER_RULES,
        enable_combined_markers=ENABLE_COMBINED_MARKERS,
    )
    X_common = subset_marker_matrix(X_all, marker_names, common_markers)

    pruned = prune_zero_area_nodes(
        G=G,
        X=X_common,
        y=y,
        patch_area=cell_patch_area,
        proj_vertex_ids=proj_vertex_ids,
        vertex_owner=vertex_owner,
    )

    G = pruned["graph"]
    X_common = pruned["x"]
    y = pruned["y"]
    cell_patch_area = pruned["patch_area"]
    proj_vertex_ids = pruned["proj_vertex_ids"]
    vertex_owner = pruned["vertex_owner"]
    d_crypts_graph = prune_projected_mesh_field(d_crypts_graph_full, pruned["keep_mask"])

    if G.number_of_nodes() < MIN_NODES:
        return None
    if not np.isfinite(y).all():
        raise ValueError(f"Patch-averaged curvatures contain NaN/Inf for {graph_path}")

    tsne_csv_path = os.path.join(dataset_root, item["tsne_csv"])
    complexity_lookup = build_complexity_lookup(tsne_csv_path) if os.path.exists(tsne_csv_path) else {}
    complexity = complexity_lookup.get(str(label_uid), np.nan)

    total_surface_area = compute_total_surface_area(mesh)
    total_volume = float(mesh.volume())
    total_curvature_discrete = float(np.sum(np.asarray(y, dtype=np.float64)[:, 0] * np.asarray(cell_patch_area, dtype=np.float64)))
    total_mean_curvature_discrete = float(np.sum(np.asarray(y, dtype=np.float64)[:, 1] * np.asarray(cell_patch_area, dtype=np.float64)))

    G.graph[GAUSSIAN_CURVATURE_GRAPH_KEY] = y[:, 0]
    G.graph[MEAN_CURVATURE_GRAPH_KEY] = y[:, 1]
    G.graph[CURVATURE_Y_COLUMNS_GRAPH_KEY] = CURVATURE_Y_COLUMNS
    G.graph[CELL_PATCH_AREA_KEY] = cell_patch_area
    G.graph[GRAPH_PROJ_VERTEX_FIELD] = proj_vertex_ids
    G.graph[f"{SEGMENTATION_DISTANCE_FIELD}_graph"] = (
        None if d_crypts_graph is None else np.asarray(d_crypts_graph, dtype=np.float64).tolist()
    )
    G.graph[f"{SEGMENTATION_DISTANCE_FIELD}_mesh_shape"] = (
        None if d_crypts_mesh is None else list(np.asarray(d_crypts_mesh).shape)
    )
    G.graph["segmentation_path"] = None if segmentation_path is None else str(segmentation_path)
    G.graph["complexity"] = _as_python(complexity)
    G.graph["timepoint"] = str(timepoint)
    G.graph["dataset"] = str(dataset)
    G.graph["num_nodes"] = int(G.number_of_nodes())
    G.graph["original_num_nodes"] = int(pruned["original_num_nodes"])
    G.graph["num_deleted_nodes"] = int(pruned["num_deleted"])
    G.graph["kept_node_ids"] = pruned["kept_node_ids"].tolist()
    G.graph["deleted_node_ids"] = pruned["deleted_node_ids"].tolist()
    G.graph["old_to_new_index"] = pruned["old_to_new_index"].tolist()
    G.graph["new_to_old_index"] = pruned["new_to_old_index"].tolist()
    G.graph[GRAPH_LABEL_UID_KEY] = str(label_uid)
    G.graph["common_marker_names"] = list(common_markers)
    G.graph[GRAPH_MESH_PATH_KEY] = str(mesh_path)
    G.graph["graph_path"] = str(graph_path)
    G.graph["total_surface_area"] = total_surface_area
    G.graph["total_volume"] = total_volume

    aux_meta = {
        "label_uid": str(label_uid),
        "dataset": str(dataset),
        "timepoint": str(timepoint),
        "num_nodes": int(G.number_of_nodes()),
        "original_num_nodes": int(pruned["original_num_nodes"]),
        "num_deleted_nodes": int(pruned["num_deleted"]),
        "deleted_node_ids": pruned["deleted_node_ids"].tolist(),
        "kept_node_ids": pruned["kept_node_ids"].tolist(),
        "old_to_new_index": pruned["old_to_new_index"].tolist(),
        "new_to_old_index": pruned["new_to_old_index"].tolist(),
        "complexity": _as_python(complexity) if pd.notna(complexity) else None,
        "graph_path": str(graph_path),
        "mesh_path": str(mesh_path),
        "segmentation_path": None if segmentation_path is None else str(segmentation_path),
        f"{SEGMENTATION_DISTANCE_FIELD}_graph": (
            None if d_crypts_graph is None else np.asarray(d_crypts_graph, dtype=np.float64).tolist()
        ),
        f"{SEGMENTATION_DISTANCE_FIELD}_mesh_shape": (
            None if d_crypts_mesh is None else list(np.asarray(d_crypts_mesh).shape)
        ),
        "total_surface_area": float(total_surface_area),
        "total_volume": float(total_volume),
        "cell_patch_area": np.asarray(cell_patch_area, dtype=np.float64).tolist(),
        "proj_vertex_ids": np.asarray(proj_vertex_ids, dtype=np.int64).tolist(),
        "total_curvature_discrete": float(total_curvature_discrete),
        "total_mean_curvature_discrete": float(total_mean_curvature_discrete),
        "mean_gaussian_curvature": float(np.mean(y[:, 0])),
        "mean_mean_curvature": float(np.mean(y[:, 1])),
        "total_mean_curvature_discrete": float(total_mean_curvature_discrete),
        "y_columns": CURVATURE_Y_COLUMNS,
        "voronoi_vertex_owner": np.asarray(vertex_owner, dtype=np.int64).tolist(),
    }

    extra_arrays = {
        CELL_PATCH_AREA_KEY: np.asarray(cell_patch_area, dtype=np.float32),
        "proj_vertex_ids": np.asarray(proj_vertex_ids, dtype=np.int64),
        "deleted_mask": np.asarray(pruned["delete_mask"], dtype=bool),
        "kept_mask": np.asarray(pruned["keep_mask"], dtype=bool),
        "deleted_node_ids": np.asarray(pruned["deleted_node_ids"], dtype=np.int64),
        "kept_node_ids": np.asarray(pruned["kept_node_ids"], dtype=np.int64),
        "old_to_new_index": np.asarray(pruned["old_to_new_index"], dtype=np.int64),
        "new_to_old_index": np.asarray(pruned["new_to_old_index"], dtype=np.int64),
        "original_num_nodes": np.asarray(pruned["original_num_nodes"], dtype=np.int64),
        GAUSSIAN_CURVATURE_GRAPH_KEY: np.asarray(y[:, 0], dtype=np.float32),
        MEAN_CURVATURE_GRAPH_KEY: np.asarray(y[:, 1], dtype=np.float32),
    }
    if d_crypts_graph is not None:
        extra_arrays[f"{SEGMENTATION_DISTANCE_FIELD}_graph"] = np.asarray(d_crypts_graph, dtype=np.float32)

    organoid_id = sanitize_filename(label_uid)
    out_path = export_organoid_npz(
        out_dir=out_dir,
        organoid_id=organoid_id,
        y=y,
        x_bin=X_common,
        organoid_graph=G,
        marker_names=common_markers,
        aux_meta=aux_meta,
        extra_arrays=extra_arrays,
    )

    return {
        "label_uid": str(label_uid),
        "dataset": str(dataset),
        "timepoint": str(timepoint),
        "num_nodes": int(G.number_of_nodes()),
        "original_num_nodes": int(pruned["original_num_nodes"]),
        "num_deleted_nodes": int(pruned["num_deleted"]),
        "n_common_markers": int(len(common_markers)),
        "complexity": _as_python(complexity) if pd.notna(complexity) else np.nan,
        "graph_path": str(graph_path),
        "mesh_path": str(mesh_path),
        "segmentation_path": None if segmentation_path is None else str(segmentation_path),
        f"has_{SEGMENTATION_DISTANCE_FIELD}": bool(d_crypts_graph is not None),
        "out_path": str(out_path),
        "total_surface_area": float(total_surface_area),
        "total_volume": float(total_volume),
        "mean_patch_area": float(np.mean(cell_patch_area)),
        "sum_patch_area": float(np.sum(cell_patch_area)),
        "total_curvature_discrete": float(total_curvature_discrete),
        "total_mean_curvature_discrete": float(total_mean_curvature_discrete),
        "mean_gaussian_curvature": float(np.mean(y[:, 0])),
        "mean_mean_curvature": float(np.mean(y[:, 1])),
        "total_mean_curvature_discrete": float(total_mean_curvature_discrete),
        "y_columns": CURVATURE_Y_COLUMNS,
    }


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    common_markers = infer_common_markers(DATA_SPECS, DATA_ROOT, verbose=VERBOSE)
    if len(common_markers) == 0:
        raise RuntimeError("No common markers found across the selected datasets.")

    if VERBOSE:
        print(f"[common markers] {len(common_markers)} found")
        print(common_markers)

    manifest_rows = []
    failures = []

    for item in iter_selected_graph_paths(DATA_SPECS, DATA_ROOT, verbose=VERBOSE):
        try:
            G_preview = load_graph_cached(item["graph_path"])
            label_uid_preview = _infer_label_uid(
                G_preview,
                fallback=os.path.splitext(os.path.basename(item["graph_path"]))[0],
            )
            out_name = f"{EXPORT_FILE_PREFIX}_{sanitize_filename(label_uid_preview)}.npz"
            out_path = os.path.join(OUT_DIR, out_name)

            if (not OVERWRITE) and os.path.exists(out_path):
                if VERBOSE:
                    print(f"[skip exists] {out_path}")
                continue

            row = process_one_organoid(item, common_markers, OUT_DIR)
            if row is not None:
                manifest_rows.append(row)
                if VERBOSE:
                    print(
                        f"[ok] dataset={row['dataset']} tp={row['timepoint']} "
                        f"label_uid={row['label_uid']} N={row['num_nodes']} -> {row['out_path']}"
                    )
            else:
                if VERBOSE:
                    print(f"[skip small] {item['graph_path']}")
        except Exception as e:
            msg = f"[failed] {item['graph_path']}: {e}"
            failures.append(msg)
            if VERBOSE:
                print(msg)
            if STRICT:
                raise

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(OUT_DIR, MANIFEST_CSV_NAME)
    manifest.to_csv(manifest_path, index=False)

    with open(os.path.join(OUT_DIR, COMMON_MARKERS_JSON_NAME), "w") as f:
        json.dump(common_markers, f, indent=2)

    with open(os.path.join(OUT_DIR, MARKER_HARMONIZATION_JSON_NAME), "w") as f:
        json.dump(
            {
                "enable_combined_markers": ENABLE_COMBINED_MARKERS,
                "combine_marker_rules": COMBINE_MARKER_RULES,
            },
            f,
            indent=2,
        )

    with open(os.path.join(OUT_DIR, FAILURES_LOG_NAME), "w") as f:
        for line in failures:
            f.write(line + "\n")

    print("\n=== DONE ===")
    print(f"Export dir      : {OUT_DIR}")
    print(f"Organoids saved : {len(manifest_rows)}")
    print(f"Failures        : {len(failures)}")
    print(f"Manifest        : {manifest_path}")
    print(f"Common markers  : {os.path.join(OUT_DIR, COMMON_MARKERS_JSON_NAME)}")


if __name__ == "__main__":
    main()

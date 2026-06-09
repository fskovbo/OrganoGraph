#!/usr/bin/env python3
"""
Export smoothed meshes, curvature fields, and curvature-annotated cell graphs
for the real organoids selected in notebooks/symmetry_real_organoids.ipynb.

For each organoid, this script exports:

1. The graph-aligned surface reconstructed from the first ``lmax**2``
   Laplace--Beltrami modes.
2. Gaussian and mean curvature on that smoothed surface.
3. The cell graph with area-weighted patch-average Gaussian and mean curvature,
   patch area, and explicit node positions.

Output layout
-------------
{OUTPUT_ROOT}/lmax_{lmax}/{label_uid}/
    mesh_lmax_{lmax}.vtp
    mesh_lmax_{lmax}.npz
    graph_lmax_{lmax}.gpickle
    graph_lmax_{lmax}.npz
    metadata_lmax_{lmax}.json

The mesh NPZ stores vertices, faces, vertex curvature, vertex areas, and Voronoi
patch ownership. The graph NPZ stores node IDs, edges, nucleus positions,
projected positions on the source and smoothed meshes, patch-average curvature,
patch area, and marker calls when available.
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path

import networkx as nx
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

from organograph.graph.io import load_cell_graph, save_cell_graph
from organograph.io_utils.dataset_config import load_mesh_dataset_config
from organograph.mesh.OrganoidMesh import OrganoidMesh
from organograph.mesh.curvature_mean import compute_organoid_curvatures
from organograph.mesh.transform import ensure_mesh_graph_aligned
from organograph.projection.voronoi import voronoi_on_mesh_dijkstra


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = (PROJECT_ROOT.parent / "NicoleData").resolve()

DATASET = "20251201"
TIMEPOINT = "day4p5"
GRAPH_SUBDIR = "graphs_preprocessed"
MESH_SUBDIR = "fractal_output"
MESH_CONFIG_NAME = "mesh_config.json"

# These are the organoids listed in notebooks/symmetry_real_organoids.ipynb.
ORGANOIDS = [
    {"well": "B03", "org_id": "144"},
    {"well": "B02", "org_id": "124"},
    {"well": "B04", "org_id": "4"},
    {"well": "B05", "org_id": "54"},
    {"well": "B02", "org_id": "115"},
    {"well": "B02", "org_id": "100"},
    {"well": "B02", "org_id": "31"},
    {"well": "B02", "org_id": "53"},
]

LMAX = 12  # Keep the first LMAX**2 Laplace--Beltrami coordinate modes.
CURVATURE_DIFFUSION_SMOOTHEN_TIME = 0.8  # Final scalar-field diffusion time for K and H.
CURVATURE_DEFECT_DETECTION = "hks"  # HKS-based debris detection used by setup_GNN_dataset.py.
CURVATURE_DEFECT_DETECTION_MESH = "raw"  # Detect defects on the aligned source mesh.

# Match setup_GNN_dataset.py: use source-mesh geodesic patches and source-mesh
# vertex areas when averaging curvature fields that share the same topology.
PATCH_GEOMETRY = "source"

OUTPUT_ROOT = DATA_ROOT / DATASET / "curvature_graphs_exports"
OVERWRITE = True  # Replace existing files for an organoid/lmax when True.
VERBOSE = True
STRICT = False  # Raise immediately on an organoid failure when True.
WRITE_VTP = True  # Write a ParaView/PyVista-compatible smoothed surface.
WRITE_GRAPH_GPICKLE = True  # Preserve the complete NetworkX graph and node attributes.
MAX_ORGANOIDS = None  # Limit processing for smoke tests; None exports the full notebook list.

MANIFEST_NAME = "manifest.csv"
FAILURES_NAME = "failures.log"
README_NAME = "README.md"

NODE_GAUSS_ATTR = "curvature_gauss_patch_mean"
NODE_MEAN_ATTR = "curvature_mean_patch_mean"
NODE_PATCH_AREA_ATTR = "cell_patch_area"
NODE_PATCH_NVERTS_ATTR = "cell_patch_n_vertices"
NODE_SMOOTHED_PROJ_ATTR = "smoothed_proj_point"


# =============================================================================
# PATHS / GRAPH ARRAYS
# =============================================================================

def organoid_key(record: dict) -> str:
    return f"{record['well']}_{record['org_id']}"


def label_uid_for(record: dict) -> str:
    return f"{TIMEPOINT}_{organoid_key(record)}"


def load_mesh_config() -> dict:
    config_path = DATA_ROOT / DATASET / MESH_CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(f"Mesh config not found: {config_path}")
    return load_mesh_dataset_config(str(config_path))


def mesh_path_for(record: dict, mesh_config: dict) -> Path:
    well = str(record["well"])
    org_id = str(record["org_id"])

    zarr_name = mesh_config["zarr_name_by_tp"][TIMEPOINT]
    round_name = mesh_config["round_by_tp"][TIMEPOINT]
    mesh_name = mesh_config["meshname_by_tp"][TIMEPOINT]

    return (
        DATA_ROOT
        / DATASET
        / MESH_SUBDIR
        / TIMEPOINT
        / zarr_name
        / well[0]
        / well[1:]
        / round_name
        / "meshes"
        / mesh_name
        / f"{org_id}.vtp"
    )


def graph_path_for(record: dict) -> Path:
    label_uid = label_uid_for(record)
    return DATA_ROOT / DATASET / GRAPH_SUBDIR / TIMEPOINT / f"{label_uid}.gpickle"


def sorted_node_ids(G: nx.Graph) -> np.ndarray:
    nodes = np.asarray(sorted(G.nodes()), dtype=np.int64)
    if nodes.size != G.number_of_nodes():
        raise ValueError("Could not construct a complete sorted node order")
    return nodes


def node_array(G: nx.Graph, node_ids: np.ndarray, field: str, dtype=None) -> np.ndarray:
    missing = [int(node) for node in node_ids if field not in G.nodes[int(node)]]
    if missing:
        raise KeyError(f"Node field '{field}' is missing for {len(missing)} nodes")
    arr = np.asarray([G.nodes[int(node)][field] for node in node_ids])
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def edge_array(G: nx.Graph, node_ids: np.ndarray) -> np.ndarray:
    node_to_pos = {int(node): i for i, node in enumerate(node_ids)}
    edges = {
        tuple(sorted((node_to_pos[int(u)], node_to_pos[int(v)])))
        for u, v in G.edges()
        if u != v
    }
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(edges), dtype=np.int64)


def projected_positions_from_graph(
    G: nx.Graph,
    node_ids: np.ndarray,
    source_mesh: OrganoidMesh,
    proj_vertex_ids: np.ndarray,
) -> np.ndarray:
    if all(G.nodes[int(node)].get("proj_point") is not None for node in node_ids):
        positions = node_array(G, node_ids, "proj_point", dtype=np.float64)
        if positions.shape == (len(node_ids), 3) and np.isfinite(positions).all():
            return positions
    return np.asarray(source_mesh.v[proj_vertex_ids], dtype=np.float64)


# =============================================================================
# PATCH AVERAGING
# =============================================================================

def ensure_mass_matrix(mesh: OrganoidMesh) -> None:
    if getattr(mesh, "mass_matrix", None) is None:
        mesh.laplacian, mesh.mass_matrix = mesh.build_cotangent_laplacian_and_mass(
            mesh.v, mesh.f
        )


def average_curvature_over_cell_patches(
    patch_mesh: OrganoidMesh,
    proj_vertex_ids: np.ndarray,
    curvature_gauss: np.ndarray,
    curvature_mean: np.ndarray,
) -> dict[str, np.ndarray]:
    """Area-average vertex curvature fields over geodesic Voronoi cell patches."""
    ensure_mass_matrix(patch_mesh)

    proj_vertex_ids = np.asarray(proj_vertex_ids, dtype=np.int64).reshape(-1)
    n_nodes = proj_vertex_ids.size
    n_vertices = len(patch_mesh.v)
    if np.any(proj_vertex_ids < 0) or np.any(proj_vertex_ids >= n_vertices):
        raise ValueError("proj_vertex contains indices outside the mesh")

    curvature_gauss = np.asarray(curvature_gauss, dtype=np.float64).reshape(-1)
    curvature_mean = np.asarray(curvature_mean, dtype=np.float64).reshape(-1)
    if curvature_gauss.size != n_vertices or curvature_mean.size != n_vertices:
        raise ValueError("Curvature fields must have one value per mesh vertex")

    vertex_owner, geodesic_distance = voronoi_on_mesh_dijkstra(
        patch_mesh, proj_vertex_ids
    )
    vertex_owner = np.asarray(vertex_owner, dtype=np.int64).reshape(-1)
    vertex_areas = np.asarray(
        patch_mesh.vertex_areas(from_mass_matrix=True), dtype=np.float64
    ).reshape(-1)

    valid = (
        (vertex_owner >= 0)
        & (vertex_owner < n_nodes)
        & np.isfinite(vertex_areas)
        & np.isfinite(curvature_gauss)
        & np.isfinite(curvature_mean)
    )

    patch_area = np.bincount(
        vertex_owner[valid],
        weights=vertex_areas[valid],
        minlength=n_nodes,
    ).astype(np.float64)
    patch_n_vertices = np.bincount(
        vertex_owner[valid],
        minlength=n_nodes,
    ).astype(np.int64)

    curvatures = np.column_stack([curvature_gauss, curvature_mean])
    patch_curvature = np.full((n_nodes, 2), np.nan, dtype=np.float64)
    nonzero = patch_area > 0
    for column in range(2):
        integral = np.bincount(
            vertex_owner[valid],
            weights=vertex_areas[valid] * curvatures[valid, column],
            minlength=n_nodes,
        ).astype(np.float64)
        patch_curvature[nonzero, column] = integral[nonzero] / patch_area[nonzero]

    return {
        "curvature": patch_curvature,
        "patch_area": patch_area,
        "patch_n_vertices": patch_n_vertices,
        "vertex_owner": vertex_owner,
        "vertex_areas": vertex_areas,
        "geodesic_distance_to_seed": np.asarray(geodesic_distance, dtype=np.float64),
    }


def attach_patch_fields_to_graph(
    G: nx.Graph,
    node_ids: np.ndarray,
    patch_result: dict[str, np.ndarray],
    smoothed_projected_positions: np.ndarray,
    *,
    lmax: int,
    source_mesh_path: Path,
    source_graph_path: Path,
) -> nx.Graph:
    patch_curvature = patch_result["curvature"]
    patch_area = patch_result["patch_area"]
    patch_n_vertices = patch_result["patch_n_vertices"]

    for position, node in enumerate(node_ids):
        attrs = G.nodes[int(node)]
        attrs[NODE_GAUSS_ATTR] = float(patch_curvature[position, 0])
        attrs[NODE_MEAN_ATTR] = float(patch_curvature[position, 1])
        attrs[NODE_PATCH_AREA_ATTR] = float(patch_area[position])
        attrs[NODE_PATCH_NVERTS_ATTR] = int(patch_n_vertices[position])
        attrs[NODE_SMOOTHED_PROJ_ATTR] = (
            np.asarray(smoothed_projected_positions[position], dtype=float).tolist()
        )

    G.graph[NODE_GAUSS_ATTR] = np.asarray(patch_curvature[:, 0], dtype=np.float32)
    G.graph[NODE_MEAN_ATTR] = np.asarray(patch_curvature[:, 1], dtype=np.float32)
    G.graph[NODE_PATCH_AREA_ATTR] = np.asarray(patch_area, dtype=np.float32)
    G.graph[NODE_PATCH_NVERTS_ATTR] = np.asarray(
        patch_n_vertices, dtype=np.int64
    )
    G.graph["node_position_field"] = "centroid"
    G.graph["smoothed_projected_position_field"] = NODE_SMOOTHED_PROJ_ATTR
    G.graph["curvature_lmax"] = int(lmax)
    G.graph["curvature_n_modes"] = int(lmax**2)
    G.graph["curvature_patch_geometry"] = PATCH_GEOMETRY
    G.graph["curvature_patch_tessellation"] = "voronoi_on_mesh_dijkstra"
    G.graph["source_mesh_path"] = str(source_mesh_path)
    G.graph["source_graph_path"] = str(source_graph_path)
    return G


# =============================================================================
# EXPORT
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
    cells.SetCells(
        len(faces),
        numpy_to_vtkIdTypeArray(packed_faces, deep=True),
    )

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


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size <= 64:
            return value.tolist()
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


def compact_curvature_diagnostics(diag: dict) -> dict:
    return {
        key: json_safe(value)
        for key, value in diag.items()
        if key != "smoothed_mesh"
    }


def save_graph_npz(
    path: Path,
    G: nx.Graph,
    node_ids: np.ndarray,
    node_positions: np.ndarray,
    source_projected_positions: np.ndarray,
    smoothed_projected_positions: np.ndarray,
    proj_vertex_ids: np.ndarray,
    patch_result: dict[str, np.ndarray],
) -> None:
    payload = {
        "node_ids": np.asarray(node_ids, dtype=np.int64),
        "edges": edge_array(G, node_ids),
        "node_positions": np.asarray(node_positions, dtype=np.float32),
        "source_projected_positions": np.asarray(
            source_projected_positions, dtype=np.float32
        ),
        "smoothed_projected_positions": np.asarray(
            smoothed_projected_positions, dtype=np.float32
        ),
        "proj_vertex_ids": np.asarray(proj_vertex_ids, dtype=np.int64),
        NODE_GAUSS_ATTR: np.asarray(
            patch_result["curvature"][:, 0], dtype=np.float32
        ),
        NODE_MEAN_ATTR: np.asarray(
            patch_result["curvature"][:, 1], dtype=np.float32
        ),
        NODE_PATCH_AREA_ATTR: np.asarray(
            patch_result["patch_area"], dtype=np.float32
        ),
        NODE_PATCH_NVERTS_ATTR: np.asarray(
            patch_result["patch_n_vertices"], dtype=np.int64
        ),
    }

    if all("markers_bin" in G.nodes[int(node)] for node in node_ids):
        payload["markers_bin"] = node_array(
            G, node_ids, "markers_bin", dtype=np.int8
        )
        payload["marker_names"] = np.asarray(
            list(G.graph.get("marker_names", [])), dtype=str
        )
    np.savez_compressed(path, **payload)


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_export_readme(path: Path, *, lmax: int) -> None:
    """Write a compact data dictionary for collaborators receiving the export."""
    text = f"""# Organoid curvature and cell-graph export

This directory contains smoothed outer-membrane meshes and cell graphs for selected
intestinal organoids from dataset `{DATASET}`, timepoint `{TIMEPOINT}`.

## Processing summary

- The membrane mesh is represented in the coordinate frame used by the cell graph.
- Mesh coordinates are spectrally smoothed by retaining the first `{lmax**2}`
  Laplace-Beltrami modes (`lmax = {lmax}`).
- Gaussian curvature `K` and signed mean curvature `H` are computed on this
  smoothed mesh.
- Curvature is transferred to cells by an area-weighted average over each cell's
  geodesic Voronoi patch on the membrane.
- Patch ownership and patch area use the `{PATCH_GEOMETRY}` mesh geometry.

Curvature units depend on the coordinate units: mean curvature has units `1/length`,
Gaussian curvature has units `1/length^2`, and patch area has units `length^2`.

## Directory layout

Each organoid has a directory named by its `label_uid`, containing:

- `mesh_lmax_{lmax}.vtp`: smoothed triangular mesh for visualization in ParaView,
  PyVista, or another VTK-compatible program.
- `mesh_lmax_{lmax}.npz`: smoothed mesh geometry and vertex-level arrays.
- `graph_lmax_{lmax}.npz`: portable cell-graph arrays.
- `graph_lmax_{lmax}.gpickle`: complete NetworkX graph, including node attributes.
- `metadata_lmax_{lmax}.json`: provenance, dimensions, marker names, coordinate
  transform, and curvature diagnostics.

The top-level `manifest.csv` lists the organoids and paths to all files.
`failures.log` is created only if an organoid could not be exported.

## Mesh NPZ variables

| Variable | Shape | Description |
|---|---:|---|
| `vertices` | `(V, 3)` | Smoothed mesh vertex coordinates. |
| `faces` | `(F, 3)` | Zero-based triangle vertex indices. |
| `curvature_gauss` | `(V,)` | Gaussian curvature `K` at each smoothed-mesh vertex. |
| `curvature_mean` | `(V,)` | Signed mean curvature `H` at each smoothed-mesh vertex. |
| `smoothed_vertex_areas` | `(V,)` | Vertex areas on the smoothed mesh. |
| `patch_vertex_areas` | `(V,)` | Vertex areas used for cell-patch averaging. |
| `voronoi_vertex_owner` | `(V,)` | Owning cell position for each vertex; indexes `node_ids` in the graph NPZ. |
| `geodesic_distance_to_cell_seed` | `(V,)` | Mesh-edge geodesic distance to the owning cell seed. |
| `lmax` | scalar | Laplace-Beltrami reconstruction level. |
| `n_modes` | scalar | Number of retained coordinate modes, equal to `lmax^2`. |

The VTP file contains the same mesh geometry and point-data arrays
`curvature_gauss`, `curvature_mean`, and `voronoi_owner`.

## Graph NPZ variables

| Variable | Shape | Description |
|---|---:|---|
| `node_ids` | `(N,)` | Original NetworkX node identifiers, in array row order. |
| `edges` | `(E, 2)` | Undirected edges indexing rows of `node_ids`. |
| `node_positions` | `(N, 3)` | Cell-nucleus positions (`centroid`) in the graph-aligned coordinate frame. |
| `source_projected_positions` | `(N, 3)` | Cell positions projected onto the aligned source mesh. |
| `smoothed_projected_positions` | `(N, 3)` | Corresponding projected vertices on the smoothed mesh. |
| `proj_vertex_ids` | `(N,)` | Mesh vertex index used as the Voronoi seed for each cell. |
| `{NODE_GAUSS_ATTR}` | `(N,)` | Area-weighted mean Gaussian curvature over each cell patch. |
| `{NODE_MEAN_ATTR}` | `(N,)` | Area-weighted mean signed mean curvature over each cell patch. |
| `{NODE_PATCH_AREA_ATTR}` | `(N,)` | Surface area of each cell's membrane patch. |
| `{NODE_PATCH_NVERTS_ATTR}` | `(N,)` | Number of mesh vertices assigned to each cell patch. |
| `markers_bin` | `(N, M)` | Binary marker calls, when available. Columns follow `marker_names`. |
| `marker_names` | `(M,)` | Marker names corresponding to columns of `markers_bin`. |

The gpickle contains the same curvature and area values as node attributes, while
preserving all original graph metadata and marker attributes. It requires Python
and NetworkX to load; the NPZ files are recommended for language-independent use.

## Minimal Python example

```python
import numpy as np

mesh = np.load("LABEL_UID/mesh_lmax_{lmax}.npz")
graph = np.load("LABEL_UID/graph_lmax_{lmax}.npz")

vertices = mesh["vertices"]
faces = mesh["faces"]
gaussian_curvature = mesh["curvature_gauss"]

node_positions = graph["node_positions"]
edges = graph["edges"]
cell_gaussian_curvature = graph["{NODE_GAUSS_ATTR}"]
```
"""
    path.write_text(text)


# =============================================================================
# PROCESSING
# =============================================================================

def process_organoid(
    record: dict,
    mesh_config: dict,
    *,
    lmax: int,
    output_dir: Path,
) -> dict:
    label_uid = label_uid_for(record)
    mesh_path = mesh_path_for(record, mesh_config)
    graph_path = graph_path_for(record)

    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    organoid_dir = output_dir / label_uid
    organoid_dir.mkdir(parents=True, exist_ok=True)
    stem = f"lmax_{lmax}"
    mesh_npz_path = organoid_dir / f"mesh_{stem}.npz"
    mesh_vtp_path = organoid_dir / f"mesh_{stem}.vtp"
    graph_npz_path = organoid_dir / f"graph_{stem}.npz"
    graph_pickle_path = organoid_dir / f"graph_{stem}.gpickle"
    metadata_path = organoid_dir / f"metadata_{stem}.json"

    expected = [mesh_npz_path, graph_npz_path, metadata_path]
    if WRITE_VTP:
        expected.append(mesh_vtp_path)
    if WRITE_GRAPH_GPICKLE:
        expected.append(graph_pickle_path)
    if not OVERWRITE and all(path.exists() for path in expected):
        return {
            "label_uid": label_uid,
            "status": "skipped_exists",
            "lmax": int(lmax),
            "n_modes": int(lmax**2),
            "mesh_path": str(mesh_path),
            "graph_path": str(graph_path),
            "mesh_npz": str(mesh_npz_path),
            "mesh_vtp": str(mesh_vtp_path) if WRITE_VTP else "",
            "graph_npz": str(graph_npz_path),
            "graph_gpickle": str(graph_pickle_path) if WRITE_GRAPH_GPICKLE else "",
            "metadata_json": str(metadata_path),
            "n_vertices": "",
            "n_faces": "",
            "n_nodes": "",
            "n_edges": "",
            "n_zero_area_patches": "",
        }

    G = load_cell_graph(str(graph_path))
    mesh = OrganoidMesh(str(mesh_path))
    mesh.label_uid = label_uid

    # Align the source mesh to the coordinate transform already stored in the graph.
    ensure_mesh_graph_aligned(mesh, G)

    node_ids = sorted_node_ids(G)
    node_positions = node_array(G, node_ids, "centroid", dtype=np.float64)
    if node_positions.shape != (len(node_ids), 3):
        raise ValueError(f"centroid must have shape (N, 3), got {node_positions.shape}")

    proj_vertex_ids = node_array(
        G, node_ids, "proj_vertex", dtype=np.int64
    ).reshape(-1)
    if np.any(proj_vertex_ids < 0) or np.any(proj_vertex_ids >= len(mesh.v)):
        raise ValueError("Graph proj_vertex values are outside the source mesh")

    curvature_gauss, curvature_mean, curvature_diag = compute_organoid_curvatures(
        mesh,
        lmax=int(lmax),
        defect_detection=CURVATURE_DEFECT_DETECTION,
        defect_detection_mesh=CURVATURE_DEFECT_DETECTION_MESH,
        diffusion_smoothen_time=CURVATURE_DIFFUSION_SMOOTHEN_TIME,
        return_diag=True,
    )
    smoothed_mesh = curvature_diag["smoothed_mesh"]

    patch_mesh = mesh if PATCH_GEOMETRY == "source" else smoothed_mesh
    if PATCH_GEOMETRY not in {"source", "smoothed"}:
        raise ValueError("PATCH_GEOMETRY must be 'source' or 'smoothed'")

    patch_result = average_curvature_over_cell_patches(
        patch_mesh=patch_mesh,
        proj_vertex_ids=proj_vertex_ids,
        curvature_gauss=curvature_gauss,
        curvature_mean=curvature_mean,
    )

    source_projected_positions = projected_positions_from_graph(
        G, node_ids, mesh, proj_vertex_ids
    )
    smoothed_projected_positions = np.asarray(
        smoothed_mesh.v[proj_vertex_ids], dtype=np.float64
    )

    attach_patch_fields_to_graph(
        G,
        node_ids,
        patch_result,
        smoothed_projected_positions,
        lmax=lmax,
        source_mesh_path=mesh_path,
        source_graph_path=graph_path,
    )

    mesh_payload = {
        "vertices": np.asarray(smoothed_mesh.v, dtype=np.float32),
        "faces": np.asarray(smoothed_mesh.f, dtype=np.int64),
        "curvature_gauss": np.asarray(curvature_gauss, dtype=np.float32),
        "curvature_mean": np.asarray(curvature_mean, dtype=np.float32),
        "smoothed_vertex_areas": np.asarray(
            smoothed_mesh.vertex_areas(from_mass_matrix=True), dtype=np.float32
        ),
        "patch_vertex_areas": np.asarray(
            patch_result["vertex_areas"], dtype=np.float32
        ),
        "voronoi_vertex_owner": np.asarray(
            patch_result["vertex_owner"], dtype=np.int64
        ),
        "geodesic_distance_to_cell_seed": np.asarray(
            patch_result["geodesic_distance_to_seed"], dtype=np.float32
        ),
        "lmax": np.int64(lmax),
        "n_modes": np.int64(lmax**2),
    }
    np.savez_compressed(mesh_npz_path, **mesh_payload)

    if WRITE_VTP:
        write_mesh_vtp(
            mesh_vtp_path,
            smoothed_mesh.v,
            smoothed_mesh.f,
            {
                "curvature_gauss": np.asarray(curvature_gauss, dtype=np.float32),
                "curvature_mean": np.asarray(curvature_mean, dtype=np.float32),
                "voronoi_owner": np.asarray(
                    patch_result["vertex_owner"], dtype=np.int64
                ),
            },
        )

    save_graph_npz(
        graph_npz_path,
        G,
        node_ids,
        node_positions,
        source_projected_positions,
        smoothed_projected_positions,
        proj_vertex_ids,
        patch_result,
    )
    if WRITE_GRAPH_GPICKLE:
        save_cell_graph(str(graph_pickle_path), G)

    n_zero_area = int(np.sum(patch_result["patch_area"] <= 0))
    metadata = {
        "label_uid": label_uid,
        "dataset": DATASET,
        "timepoint": TIMEPOINT,
        "well": str(record["well"]),
        "organoid_id": str(record["org_id"]),
        "lmax": int(lmax),
        "n_modes": int(lmax**2),
        "coordinate_frame": "graph_aligned",
        "patch_geometry": PATCH_GEOMETRY,
        "source_mesh_path": str(mesh_path),
        "source_graph_path": str(graph_path),
        "mesh_npz": str(mesh_npz_path),
        "mesh_vtp": str(mesh_vtp_path) if WRITE_VTP else None,
        "graph_npz": str(graph_npz_path),
        "graph_gpickle": str(graph_pickle_path) if WRITE_GRAPH_GPICKLE else None,
        "node_position_field": "centroid",
        "source_projected_position_field": "proj_point",
        "smoothed_projected_position_field": NODE_SMOOTHED_PROJ_ATTR,
        "marker_names": list(G.graph.get("marker_names", [])),
        "n_vertices": int(len(smoothed_mesh.v)),
        "n_faces": int(len(smoothed_mesh.f)),
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "n_zero_area_patches": n_zero_area,
        "curvature_diagnostics": compact_curvature_diagnostics(curvature_diag),
        "coord_transform": json_safe(getattr(mesh, "coord_transform", None)),
    }
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "label_uid": label_uid,
        "status": "ok",
        "lmax": int(lmax),
        "n_modes": int(lmax**2),
        "mesh_path": str(mesh_path),
        "graph_path": str(graph_path),
        "mesh_npz": str(mesh_npz_path),
        "mesh_vtp": str(mesh_vtp_path) if WRITE_VTP else "",
        "graph_npz": str(graph_npz_path),
        "graph_gpickle": str(graph_pickle_path) if WRITE_GRAPH_GPICKLE else "",
        "metadata_json": str(metadata_path),
        "n_vertices": int(len(smoothed_mesh.v)),
        "n_faces": int(len(smoothed_mesh.f)),
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "n_zero_area_patches": n_zero_area,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lmax",
        type=int,
        default=LMAX,
        help=f"LB reconstruction level; retains lmax**2 modes (default: {LMAX}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Root export directory (default: {OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=STRICT,
        help="Stop immediately if one organoid fails.",
    )
    parser.add_argument(
        "--max-organoids",
        type=int,
        default=MAX_ORGANOIDS,
        help="Process only the first N configured organoids (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lmax <= 0:
        raise ValueError("--lmax must be positive")

    mesh_config = load_mesh_config()
    output_dir = args.output_root.resolve() / f"lmax_{args.lmax}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    records = ORGANOIDS
    if args.max_organoids is not None:
        if args.max_organoids <= 0:
            raise ValueError("--max-organoids must be positive")
        records = ORGANOIDS[: args.max_organoids]

    for index, record in enumerate(records, start=1):
        label_uid = label_uid_for(record)
        if VERBOSE:
            print(f"[{index}/{len(records)}] {label_uid}")
        try:
            row = process_organoid(
                record,
                mesh_config,
                lmax=args.lmax,
                output_dir=output_dir,
            )
            rows.append(row)
            if VERBOSE:
                print(
                    f"  {row['status']}: nodes={row['n_nodes']} "
                    f"vertices={row['n_vertices']} zero_area={row['n_zero_area_patches']}"
                )
        except Exception as exc:
            message = f"{label_uid}: {type(exc).__name__}: {exc}"
            failures.append(f"{message}\n{traceback.format_exc()}")
            print(f"  FAILED: {message}")
            if args.strict:
                raise

    write_manifest(output_dir / MANIFEST_NAME, rows)
    write_export_readme(output_dir / README_NAME, lmax=args.lmax)
    failures_path = output_dir / FAILURES_NAME
    if failures:
        failures_path.write_text("\n\n".join(failures))
    elif failures_path.exists():
        failures_path.unlink()

    print(f"\nExported {sum(row['status'] == 'ok' for row in rows)} organoids")
    print(f"Manifest: {output_dir / MANIFEST_NAME}")
    print(f"README: {output_dir / README_NAME}")
    if failures:
        print(f"Failures: {len(failures)} ({failures_path})")


if __name__ == "__main__":
    main()

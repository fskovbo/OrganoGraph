"""Portable exports for skeleton and primitive fitting results.

The functions in this module deliberately keep two views of the same shape:

1. a faithful JSON payload built from the graph/result containers; and
2. flat CSV/NPZ tables that are easy to consume in a separate VAE project.

The export schema avoids hard-coding individual primitive parameters.  New
node fields, edge fields, primitive families, or metadata can be carried
through JSON columns without changing the table layout.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from organograph.skeleton.datatypes import SkeletonGraph
from organograph.skeleton.results import (
    OrganoidShapeResult,
    PrimitiveFitResult,
    SkeletonizationResult,
)


SHAPE_EXPORT_SCHEMA_VERSION = "organograph_skeleton_primitives_v1"


def json_safe(value: Any) -> Any:
    """Convert common scientific Python values into JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if callable(value):
        return getattr(value, "__name__", str(value))
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    return str(value)


def _json_string(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class _CoercedShape:
    graph: SkeletonGraph
    skeleton: SkeletonizationResult | None = None
    primitives: PrimitiveFitResult | None = None


def _coerce_shape_result(
    result: SkeletonGraph | SkeletonizationResult | PrimitiveFitResult | OrganoidShapeResult,
    *,
    primitive_result: PrimitiveFitResult | None = None,
) -> _CoercedShape:
    if isinstance(result, PrimitiveFitResult):
        primitives = result
        skeleton = result.skeleton
        graph = result.graph
    elif isinstance(result, SkeletonizationResult):
        skeleton = result
        primitives = primitive_result
        graph = primitives.graph if primitives is not None else result.graph
    elif isinstance(result, OrganoidShapeResult):
        skeleton = result.skeleton
        primitives = primitive_result if primitive_result is not None else result.primitives
        graph = primitives.graph if primitives is not None else result.graph
    elif isinstance(result, SkeletonGraph):
        skeleton = None
        primitives = primitive_result
        graph = primitives.graph if primitives is not None else result
    else:
        raise TypeError(
            "result must be a SkeletonGraph, SkeletonizationResult, "
            "PrimitiveFitResult, or OrganoidShapeResult"
        )
    return _CoercedShape(graph=graph, skeleton=skeleton, primitives=primitives)


def graph_summary(graph: SkeletonGraph) -> dict[str, Any]:
    """Return small, stable counts useful for manifests and quick filtering."""
    node_type_counts: dict[str, int] = {}
    for node in graph.nodes.values():
        node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1
    primitive_type_counts: dict[str, int] = {}
    for attachment in graph.primitive_attachments.values():
        primitive_type_counts[attachment.primitive_type] = (
            primitive_type_counts.get(attachment.primitive_type, 0) + 1
        )
    return {
        "n_nodes": int(len(graph.nodes)),
        "n_edges": int(len(graph.edges)),
        "n_crypts": int(len(graph.crypt_ids())),
        "n_primitive_attachments": int(len(graph.primitive_attachments)),
        "node_type_counts": node_type_counts,
        "primitive_type_counts": primitive_type_counts,
    }


def node_records(graph: SkeletonGraph) -> list[dict[str, Any]]:
    """Return node rows with generic JSON columns for future extensibility."""
    rows = []
    for node in graph.nodes.values():
        attachment = node.primitive_attachment
        rows.append(
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "crypt_id": "" if node.crypt_id is None else str(node.crypt_id),
                "x": float(node.position[0]),
                "y": float(node.position[1]),
                "z": float(node.position[2]),
                "metadata_json": _json_string(node.metadata),
                "primitive_attachment_json": _json_string(
                    None if attachment is None else attachment.to_dict()
                ),
            }
        )
    return rows


def edge_records(graph: SkeletonGraph) -> list[dict[str, Any]]:
    """Return edge rows with generic JSON columns for future extensibility."""
    rows = []
    for edge in graph.edges.values():
        attachment = edge.primitive_attachment
        rows.append(
            {
                "edge_id": edge.edge_id,
                "source": edge.source,
                "target": edge.target,
                "edge_type": edge.edge_type,
                "crypt_id": "" if edge.crypt_id is None else str(edge.crypt_id),
                "metadata_json": _json_string(edge.metadata),
                "primitive_attachment_json": _json_string(
                    None if attachment is None else attachment.to_dict()
                ),
            }
        )
    return rows


def primitive_records(graph: SkeletonGraph) -> list[dict[str, Any]]:
    """Return graph-level primitive attachment rows.

    Individual primitive parameters stay in JSON columns so new primitive
    families do not require schema changes.
    """
    rows = []
    for attachment_id, attachment in graph.primitive_attachments.items():
        rows.append(
            {
                "attachment_id": str(attachment_id),
                "attachment_type": "" if attachment.attachment_type is None else str(attachment.attachment_type),
                "primitive_type": str(attachment.primitive_type),
                "target_ids_json": _json_string(attachment.target_ids),
                "fit_error": "" if attachment.fit_error is None else float(attachment.fit_error),
                "parameters_json": _json_string(attachment.parameters),
                "derived_parameters_json": _json_string(attachment.derived_parameters),
                "residuals_json": _json_string(attachment.residuals),
                "metadata_json": _json_string(attachment.metadata),
            }
        )
    return rows


def component_summary(value: Any) -> Any:
    """Compact component/intermediate arrays without exporting large meshes."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): component_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if all(isinstance(item, (set, list, tuple, np.ndarray)) for item in value):
            lengths = []
            for item in value:
                try:
                    lengths.append(int(len(item)))
                except TypeError:
                    lengths.append(None)
            return {"n_items": len(value), "item_lengths": lengths}
        return [component_summary(item) for item in value]
    if isinstance(value, set):
        return {"n_items": len(value)}
    if isinstance(value, (str, int, float, bool, np.generic)):
        return json_safe(value)
    return str(type(value).__name__)


def shape_export_payload(
    result: SkeletonGraph | SkeletonizationResult | PrimitiveFitResult | OrganoidShapeResult,
    *,
    primitive_result: PrimitiveFitResult | None = None,
    metadata: dict[str, Any] | None = None,
    include_detections: bool = True,
    include_intermediates: bool = False,
    include_components: bool = False,
    schema_version: str = SHAPE_EXPORT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build a portable JSON payload for one skeleton/primitive result."""
    coerced = _coerce_shape_result(result, primitive_result=primitive_result)
    graph = coerced.graph
    skeleton = coerced.skeleton
    primitives = coerced.primitives

    merged_metadata: dict[str, Any] = {}
    if skeleton is not None:
        merged_metadata.update(skeleton.metadata)
    if primitives is not None:
        merged_metadata.update(primitives.metadata)
    if metadata:
        merged_metadata.update(metadata)

    skeleton_payload: dict[str, Any] | None = None
    if skeleton is not None:
        skeleton_payload = {
            "config": skeleton.config.to_dict(),
            "metadata": dict(skeleton.metadata),
        }
        if include_detections:
            skeleton_payload["detections"] = skeleton.detections
        if include_intermediates:
            skeleton_payload["intermediates"] = skeleton.intermediates

    primitive_payload: dict[str, Any] | None = None
    if primitives is not None:
        primitive_payload = {
            "config": primitives.config.to_dict(),
            "metadata": dict(primitives.metadata),
            "attachments": primitives.attachments,
        }
        if include_components:
            primitive_payload["components"] = primitives.components
        else:
            primitive_payload["component_summary"] = component_summary(
                primitives.components
            )

    return json_safe(
        {
            "schema_version": schema_version,
            "created_at_utc": _now_utc(),
            "metadata": merged_metadata,
            "summary": graph_summary(graph),
            "graph": graph.to_dict(),
            "tables": {
                "nodes": node_records(graph),
                "edges": edge_records(graph),
                "primitives": primitive_records(graph),
            },
            "skeletonization": skeleton_payload,
            "primitive_fit": primitive_payload,
        }
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
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


def graph_arrays(graph: SkeletonGraph) -> dict[str, np.ndarray]:
    """Return portable arrays for quick loading from NumPy."""
    node_ids = list(graph.nodes)
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}
    edge_ids = list(graph.edges)
    primitive_ids = list(graph.primitive_attachments)

    edge_index = np.full((len(edge_ids), 2), -1, dtype=np.int64)
    for i, edge_id in enumerate(edge_ids):
        edge = graph.edge(edge_id)
        edge_index[i, 0] = node_index.get(edge.source, -1)
        edge_index[i, 1] = node_index.get(edge.target, -1)

    fit_error = np.full(len(primitive_ids), np.nan, dtype=np.float64)
    for i, attachment_id in enumerate(primitive_ids):
        error = graph.primitive_attachments[attachment_id].fit_error
        if error is not None:
            fit_error[i] = float(error)

    return {
        "node_ids": np.asarray(node_ids, dtype=str),
        "node_types": np.asarray(
            [graph.node(node_id).node_type for node_id in node_ids],
            dtype=str,
        ),
        "node_crypt_ids": np.asarray(
            [
                "" if graph.node(node_id).crypt_id is None else str(graph.node(node_id).crypt_id)
                for node_id in node_ids
            ],
            dtype=str,
        ),
        "node_positions": np.asarray(
            [graph.node(node_id).position for node_id in node_ids],
            dtype=np.float64,
        ).reshape(len(node_ids), 3),
        "edge_ids": np.asarray(edge_ids, dtype=str),
        "edge_sources": np.asarray(
            [graph.edge(edge_id).source for edge_id in edge_ids],
            dtype=str,
        ),
        "edge_targets": np.asarray(
            [graph.edge(edge_id).target for edge_id in edge_ids],
            dtype=str,
        ),
        "edge_types": np.asarray(
            [graph.edge(edge_id).edge_type for edge_id in edge_ids],
            dtype=str,
        ),
        "edge_crypt_ids": np.asarray(
            [
                "" if graph.edge(edge_id).crypt_id is None else str(graph.edge(edge_id).crypt_id)
                for edge_id in edge_ids
            ],
            dtype=str,
        ),
        "edge_index": edge_index,
        "primitive_attachment_ids": np.asarray(primitive_ids, dtype=str),
        "primitive_types": np.asarray(
            [
                graph.primitive_attachments[attachment_id].primitive_type
                for attachment_id in primitive_ids
            ],
            dtype=str,
        ),
        "primitive_attachment_types": np.asarray(
            [
                ""
                if graph.primitive_attachments[attachment_id].attachment_type is None
                else str(graph.primitive_attachments[attachment_id].attachment_type)
                for attachment_id in primitive_ids
            ],
            dtype=str,
        ),
        "primitive_target_ids_json": np.asarray(
            [
                _json_string(graph.primitive_attachments[attachment_id].target_ids)
                for attachment_id in primitive_ids
            ],
            dtype=str,
        ),
        "primitive_parameters_json": np.asarray(
            [
                _json_string(graph.primitive_attachments[attachment_id].parameters)
                for attachment_id in primitive_ids
            ],
            dtype=str,
        ),
        "primitive_derived_parameters_json": np.asarray(
            [
                _json_string(
                    graph.primitive_attachments[attachment_id].derived_parameters
                )
                for attachment_id in primitive_ids
            ],
            dtype=str,
        ),
        "primitive_residuals_json": np.asarray(
            [
                _json_string(graph.primitive_attachments[attachment_id].residuals)
                for attachment_id in primitive_ids
            ],
            dtype=str,
        ),
        "primitive_fit_error": fit_error,
    }


def save_shape_export(
    result: SkeletonGraph | SkeletonizationResult | PrimitiveFitResult | OrganoidShapeResult,
    output_dir,
    *,
    primitive_result: PrimitiveFitResult | None = None,
    metadata: dict[str, Any] | None = None,
    prefix: str = "shape",
    include_detections: bool = True,
    include_intermediates: bool = False,
    include_components: bool = False,
    write_json: bool = True,
    write_tables: bool = True,
    write_npz: bool = True,
) -> dict[str, str]:
    """Write one portable skeleton/primitives export directory.

    Returns a dictionary of output paths keyed by artifact name.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coerced = _coerce_shape_result(result, primitive_result=primitive_result)

    payload = shape_export_payload(
        result,
        primitive_result=primitive_result,
        metadata=metadata,
        include_detections=include_detections,
        include_intermediates=include_intermediates,
        include_components=include_components,
    )
    paths: dict[str, str] = {}
    if write_json:
        path = output_dir / f"{prefix}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        paths["json"] = str(path)
    if write_tables:
        nodes_path = output_dir / f"{prefix}_nodes.csv"
        edges_path = output_dir / f"{prefix}_edges.csv"
        primitives_path = output_dir / f"{prefix}_primitives.csv"
        _write_csv(nodes_path, payload["tables"]["nodes"])
        _write_csv(edges_path, payload["tables"]["edges"])
        _write_csv(primitives_path, payload["tables"]["primitives"])
        paths.update(
            {
                "nodes_csv": str(nodes_path),
                "edges_csv": str(edges_path),
                "primitives_csv": str(primitives_path),
            }
        )
    if write_npz:
        path = output_dir / f"{prefix}_arrays.npz"
        np.savez_compressed(path, **graph_arrays(coerced.graph))
        paths["arrays_npz"] = str(path)
    return paths


def load_shape_export_json(path) -> dict[str, Any]:
    """Load a JSON payload written by :func:`save_shape_export`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_export_readme(path, *, dataset: str | None = None) -> None:
    """Write a compact data dictionary for a skeleton/primitives export root."""
    dataset_line = f"Dataset: `{dataset}`.\n\n" if dataset else ""
    text = f"""# Skeleton and Primitive Export

{dataset_line}This directory contains portable exports of organoid skeleton graphs and fitted
primitive attachments for downstream shape modeling, including VAE work in a
separate project.

## Per-organoid layout

Each organoid directory contains:

- `shape.json`: complete JSON payload with metadata, graph nodes/edges,
  primitive attachments, configs, detections, and compact component summaries.
- `shape_nodes.csv`: node table with `node_id`, `node_type`, `crypt_id`, `x`,
  `y`, `z`, and JSON metadata columns.
- `shape_edges.csv`: edge table with `edge_id`, `source`, `target`,
  `edge_type`, `crypt_id`, and JSON metadata columns.
- `shape_primitives.csv`: one row per graph-level primitive attachment.  The
  `parameters_json`, `derived_parameters_json`, `residuals_json`, and
  `metadata_json` columns intentionally keep primitive-specific content generic.
- `shape_arrays.npz`: NumPy arrays for quick loading of node positions, edge
  indices, node/edge types, and primitive JSON strings.

## Schema principle

The export is designed to survive changes in skeleton nodes, primitive
families, and metadata.  Stable columns identify biological graph structure;
rapidly evolving details are stored as JSON dictionaries.  Downstream projects
should version their own packed VAE tensors separately from this raw portable
export.

## Minimal loading example

```python
import json
import numpy as np

with open("LABEL_UID/shape.json") as f:
    payload = json.load(f)

arrays = np.load("LABEL_UID/shape_arrays.npz", allow_pickle=False)
node_positions = arrays["node_positions"]
edge_index = arrays["edge_index"]
primitive_parameters = arrays["primitive_parameters_json"]
```

## Recommended downstream use

Use this export as the raw interchange format.  Build a separate, versioned VAE
packing step that selects and normalizes `T` (topology), `S` (core skeleton
morphology), `P` (primitive details), and `C` (context/metadata) for a specific
model experiment.
"""
    Path(path).write_text(text, encoding="utf-8")

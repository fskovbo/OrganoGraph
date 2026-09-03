"""Minimal, reconstructive exports for final skeleton and primitive fits.

The v5 export contains only identity/context, reversible coordinate transforms,
graph topology, node positions, and the primitive degrees of freedom required
to recreate the final visualization. Detection arrays, component ownership,
residuals, and fitting objectives are intentionally excluded.

Crypt list order has no biological meaning. Consumers must use ``crypt_id`` and
graph connectivity, or a permutation-invariant model, rather than list index.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from organograph.skeleton.datatypes import SkeletonGraph
from organograph.skeleton.legacy_curves import (
    sample_circular_arc,
    sample_cubic_bezier,
    sample_quadratic_bezier,
    sample_sinusoidal_bend,
)
from organograph.skeleton.primitive.crypt_geometry import sample_tangent_hermite
from organograph.skeleton.primitives import PrimitiveAttachment
from organograph.skeleton.results import (
    OrganoidShapeResult,
    PrimitiveFitResult,
    SkeletonizationResult,
)


SHAPE_EXPORT_SCHEMA_VERSION = "organograph_shape_v5"
LEGACY_SHAPE_EXPORT_SCHEMA_VERSIONS = {
    "organograph_shape_v2",
    "organograph_shape_v3",
    "organograph_shape_v4",
}
SHAPE_QUALITY_SCHEMA_VERSION = "organograph_shape_quality_v4"
_SAMPLE_FIELDS = (
    "dataset",
    "timepoint",
    "well",
    "organoid_id",
    "label_uid",
    "condition",
    "replicate",
    "age",
    "cell_count",
    "mesh_path",
    "source_units",
)
_TUBE_TYPE = "tapered_capped_tube"
_NECK_TYPE = "straight_cylinder"


@dataclass
class _CoercedShape:
    graph: SkeletonGraph
    skeleton: SkeletonizationResult | None
    primitives: PrimitiveFitResult | None
    metadata: dict[str, Any]


def _coerce_shape_result(
    result: SkeletonGraph | SkeletonizationResult | PrimitiveFitResult | OrganoidShapeResult,
    *,
    primitive_result: PrimitiveFitResult | None = None,
) -> _CoercedShape:
    result_metadata: dict[str, Any] = {}
    if isinstance(result, PrimitiveFitResult):
        primitives = result
        skeleton = result.skeleton
        graph = result.graph
        result_metadata.update(result.metadata)
    elif isinstance(result, SkeletonizationResult):
        skeleton = result
        primitives = primitive_result
        graph = primitives.graph if primitives is not None else result.graph
        result_metadata.update(result.metadata)
    elif isinstance(result, OrganoidShapeResult):
        skeleton = result.skeleton
        primitives = primitive_result if primitive_result is not None else result.primitives
        graph = primitives.graph if primitives is not None else result.graph
        result_metadata.update(result.metadata)
    elif isinstance(result, SkeletonGraph):
        skeleton = None
        primitives = primitive_result
        graph = primitives.graph if primitives is not None else result
    else:
        raise TypeError(
            "result must be a SkeletonGraph, SkeletonizationResult, "
            "PrimitiveFitResult, or OrganoidShapeResult"
        )
    if skeleton is not None:
        result_metadata.update(skeleton.metadata)
    if primitives is not None:
        result_metadata.update(primitives.metadata)
    return _CoercedShape(graph, skeleton, primitives, result_metadata)


def _finite_json(value: Any, *, path: str = "root") -> Any:
    """Return strict JSON values and reject every NaN or infinite number."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value at {path}: {value!r}")
        return value
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist(), path=path)
    if isinstance(value, dict):
        return {
            str(key): _finite_json(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, set):
        value = sorted(value, key=str)
    if isinstance(value, (list, tuple)):
        return [
            _finite_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"Unsupported export value at {path}: {type(value).__name__}")


def _nullable_quality_json(value: Any) -> Any:
    """Convert diagnostic values to JSON, replacing non-finite values by null."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _nullable_quality_json(value.tolist())
    if isinstance(value, dict):
        return {str(key): _nullable_quality_json(item) for key, item in value.items()}
    if isinstance(value, set):
        value = sorted(value, key=str)
    if isinstance(value, (list, tuple)):
        return [_nullable_quality_json(item) for item in value]
    return str(value)


def _metadata_sources(coerced: _CoercedShape, metadata: dict[str, Any] | None):
    sources = [coerced.metadata]
    for source in list(sources):
        record = source.get("record") if isinstance(source, dict) else None
        if isinstance(record, dict):
            sources.insert(0, record)
        skeleton_metadata = source.get("skeleton_metadata") if isinstance(source, dict) else None
        if isinstance(skeleton_metadata, dict):
            nested_record = skeleton_metadata.get("record")
            if isinstance(nested_record, dict):
                sources.insert(0, nested_record)
    if metadata:
        sources.append(metadata)
    return sources


def _sample_record(
    coerced: _CoercedShape,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    sample: dict[str, Any] = {}
    for source in _metadata_sources(coerced, metadata):
        for key in _SAMPLE_FIELDS:
            if key in source and source[key] is not None:
                sample[key] = source[key]
    n_branches = sum(node.node_type == "branch" for node in coerced.graph.nodes.values())
    sample.update(
        {
            "has_branches": bool(n_branches),
            "vae_eligible": not bool(n_branches),
        }
    )
    return _finite_json(sample, path="sample")


def _mesh_from_shape(coerced: _CoercedShape):
    if coerced.primitives is not None and coerced.primitives.mesh is not None:
        return coerced.primitives.mesh
    if coerced.skeleton is not None:
        return coerced.skeleton.mesh
    return None


def _coordinate_transform(coerced: _CoercedShape, sample: dict[str, Any]) -> dict[str, Any]:
    mesh = _mesh_from_shape(coerced)
    transform = getattr(mesh, "coord_transform", None) if mesh is not None else None
    transform = dict(transform or {})

    center = np.asarray(transform.get("center", np.zeros(3)), dtype=float)
    center_applied = center.shape == (3,) and np.all(np.isfinite(center))
    if not center_applied:
        center = np.zeros(3, dtype=float)
    scale = float(transform.get("scale", 1.0)) if center_applied else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Mesh coordinate transform scale must be finite and positive")
    rotation = np.asarray(transform.get("rotation", np.eye(3)), dtype=float)
    rotation_applied = rotation.shape == (3, 3) and np.all(np.isfinite(rotation))
    if not rotation_applied:
        rotation = np.eye(3, dtype=float)
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-7):
        raise ValueError("Mesh coordinate transform rotation must be orthonormal")

    source_to_fitted = np.eye(4, dtype=float)
    source_to_fitted[:3, :3] = rotation / scale
    source_to_fitted[:3, 3] = -(rotation @ center) / scale
    fitted_to_source = np.linalg.inv(source_to_fitted)
    source_units = str(sample.get("source_units", "source_mesh_units"))
    return _finite_json(
        {
            "source_coordinate_system": "original_mesh",
            "fitted_coordinate_system": "prepared_mesh",
            "center": center,
            "scale": scale,
            "rotation": rotation,
            "center_applied": bool(center_applied),
            "rotation_applied": bool(rotation_applied),
            "source_to_fitted": source_to_fitted,
            "fitted_to_source": fitted_to_source,
            "source_units": source_units,
            "fitted_units": "prepared_mesh_units",
        },
        path="coordinate_transform",
    )


def _skeleton_record(graph: SkeletonGraph) -> dict[str, Any]:
    nodes = [
        {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "crypt_id": node.crypt_id,
            "position": node.position,
        }
        for node in graph.nodes.values()
    ]
    edges = [
        {
            "edge_id": edge.edge_id,
            "source": edge.source,
            "target": edge.target,
            "edge_type": edge.edge_type,
            "crypt_id": edge.crypt_id,
        }
        for edge in graph.edges.values()
    ]
    return _finite_json({"nodes": nodes, "edges": edges}, path="skeleton")


def _all_attachments(graph: SkeletonGraph):
    for node_id, node in graph.nodes.items():
        if node.primitive_attachment is not None:
            yield "node", node_id, node.primitive_attachment
    for edge_id, edge in graph.edges.items():
        if edge.primitive_attachment is not None:
            yield "edge", edge_id, edge.primitive_attachment
    for attachment_id, attachment in graph.primitive_attachments.items():
        yield "graph", attachment_id, attachment


def _curve_record(attachment: PrimitiveAttachment) -> dict[str, Any]:
    parameters = attachment.parameters
    metadata = attachment.metadata
    centerline = np.asarray(parameters.get("centerline_points"), dtype=float)
    controls = np.asarray(
        metadata.get("centerline_control_points", centerline),
        dtype=float,
    )
    method = str(metadata.get("centerline_method", ""))
    if "tangent_constrained_hermite" in method:
        curve_type = "tangent_hermite"
        controls = centerline[[0, -1]]
    elif "circular_arc" in method:
        curve_type = "circular_arc"
        controls = centerline[[0, -1]]
    elif "sinusoidal_bend" in method:
        curve_type = "sinusoidal_bend"
    elif "cubic_bezier" in method or controls.shape[0] == 4:
        curve_type = "cubic_bezier"
    elif "quadratic_bezier" in method or controls.shape[0] == 3:
        curve_type = "quadratic_bezier"
    elif controls.shape[0] == 2:
        curve_type = "line"
    else:
        curve_type = "polyline"
    if controls.ndim != 2 or controls.shape[1] != 3 or controls.shape[0] < 2:
        raise ValueError(
            f"Primitive {attachment.attachment_id!r} has invalid centerline controls"
        )
    n_samples = int(centerline.shape[0]) if centerline.ndim == 2 else 64
    record = {
        "centerline_type": curve_type,
        "centerline_control_points": controls,
        "centerline_samples": max(2, n_samples),
    }
    if curve_type == "sinusoidal_bend":
        bend = np.asarray(metadata.get("centerline_bend_vector"), dtype=float)
        if bend.shape != (3,):
            raise ValueError(
                f"Primitive {attachment.attachment_id!r} has an invalid bend vector"
            )
        record["centerline_bend_vector"] = bend
    elif curve_type == "circular_arc":
        sagitta = np.asarray(metadata.get("centerline_sagitta_vector"), dtype=float)
        if sagitta.shape != (3,):
            raise ValueError(
                f"Primitive {attachment.attachment_id!r} has an invalid arc sagitta"
            )
        record["centerline_sagitta_vector"] = sagitta
    elif curve_type == "tangent_hermite":
        for key in ("centerline_start_tangent", "centerline_end_tangent"):
            tangent = np.asarray(metadata.get(key), dtype=float)
            if tangent.shape != (3,):
                raise ValueError(
                    f"Primitive {attachment.attachment_id!r} has an invalid {key}"
                )
            record[key] = tangent
    return record


def _compact_parameters(attachment: PrimitiveAttachment) -> dict[str, Any]:
    primitive_type = str(attachment.primitive_type)
    source = attachment.parameters
    if primitive_type in {"ellipsoid", "superellipsoid"}:
        out = {
            "center": source["center"],
            "orientation": source["orientation"],
            "axis_lengths": source["axis_lengths"],
        }
        if primitive_type == "superellipsoid":
            out["epsilon_1"] = source["epsilon_1"]
            out["epsilon_2"] = source["epsilon_2"]
        return out
    if primitive_type == "asymmetric_superellipsoid":
        return {
            "center": source["center"],
            "orientation": source["orientation"],
            "axis_lengths_negative": source["axis_lengths_negative"],
            "axis_lengths_positive": source["axis_lengths_positive"],
            "epsilon_1": source["epsilon_1"],
            "epsilon_2": source["epsilon_2"],
        }
    if primitive_type == _TUBE_TYPE:
        r_distal = source.get(
            "r_distal", source.get("r_taper", source.get("r_tip"))
        )
        output = {
            **_curve_record(attachment),
            "r_attachment": (
                source["r_attachment"] if "r_attachment" in source else source["r_neck"]
            ),
            "r_center": source["r_center"] if "r_center" in source else source["r_body"],
            "r_distal": r_distal,
            "s_center": source.get("s_center", source.get("s_body", 0.5)),
            "s_taper": source.get("s_taper", source.get("distal_taper_start", 0.85)),
            "r_constriction": source.get("r_constriction"),
            "s_constriction": source.get("s_constriction"),
            "radius_profile": "semantic_landmarks_squared_radius_v2",
        }
        if source.get("opening_normal") is not None:
            output["opening_normal"] = source["opening_normal"]
            output["opening_frame_blend_fraction"] = source.get(
                "opening_frame_blend_fraction", 0.15
            )
        return output
    if primitive_type == _NECK_TYPE:
        centerline = np.asarray(source["centerline_points"], dtype=float)
        return {
            "centerline_type": "line" if centerline.shape[0] == 2 else "polyline",
            "centerline_control_points": centerline,
            "radius": source["radius"],
        }
    raise ValueError(
        f"Unsupported reconstructive primitive type {primitive_type!r}; "
        "add an explicit parameter adapter before exporting it"
    )


def _primitive_role(graph: SkeletonGraph, scope: str, owner_id: str, primitive_type: str) -> str:
    if scope == "node" and owner_id in graph.nodes:
        node_type = graph.node(owner_id).node_type
        if node_type in {"body", "branch"}:
            return node_type
    if primitive_type == _TUBE_TYPE:
        return "crypt"
    if primitive_type == _NECK_TYPE:
        return "body_branch_neck"
    return "other"


def _primitive_records(graph: SkeletonGraph) -> list[dict[str, Any]]:
    records = []
    for scope, owner_id, attachment in _all_attachments(graph):
        primitive_id = attachment.attachment_id or str(owner_id)
        records.append(
            {
                "primitive_id": str(primitive_id),
                "primitive_type": str(attachment.primitive_type),
                "role": _primitive_role(
                    graph,
                    scope,
                    str(owner_id),
                    str(attachment.primitive_type),
                ),
                "attachment_scope": scope,
                "owner_id": str(owner_id),
                "target_node_ids": [str(item) for item in attachment.target_ids],
                "parameters": _compact_parameters(attachment),
            }
        )
    return _finite_json(records, path="primitives")


def graph_summary(graph: SkeletonGraph) -> dict[str, int]:
    """Return stable counts for manifests and eligibility checks."""
    return {
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "n_crypts": sum(node.node_type == "tip" for node in graph.nodes.values()),
        "n_branches": sum(node.node_type == "branch" for node in graph.nodes.values()),
        "n_primitives": sum(1 for _ in _all_attachments(graph)),
    }


def shape_export_payload(
    result: SkeletonGraph | SkeletonizationResult | PrimitiveFitResult | OrganoidShapeResult,
    *,
    primitive_result: PrimitiveFitResult | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one compact v5 payload from the final fitted primitive graph."""
    coerced = _coerce_shape_result(result, primitive_result=primitive_result)
    sample = _sample_record(coerced, metadata)
    payload = {
        "schema_version": SHAPE_EXPORT_SCHEMA_VERSION,
        "sample": sample,
        "coordinate_transform": _coordinate_transform(coerced, sample),
        "summary": graph_summary(coerced.graph),
        "skeleton": _skeleton_record(coerced.graph),
        "primitives": _primitive_records(coerced.graph),
    }
    payload = _finite_json(payload)
    validate_shape_export_payload(payload)
    return payload


def validate_shape_export_payload(payload: dict[str, Any]) -> None:
    """Validate topology, transforms, primitive targets, and finite geometry."""
    _finite_json(payload)
    schema = payload.get("schema_version")
    if schema not in {SHAPE_EXPORT_SCHEMA_VERSION, *LEGACY_SHAPE_EXPORT_SCHEMA_VERSIONS}:
        raise ValueError(f"Unsupported shape export schema {schema!r}")
    skeleton = payload.get("skeleton") or {}
    nodes = skeleton.get("nodes") or []
    edges = skeleton.get("edges") or []
    node_ids = [str(node["node_id"]) for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Skeleton export contains duplicate node IDs")
    if sum(node.get("node_type") == "body" for node in nodes) != 1:
        raise ValueError("Skeleton export must contain exactly one body node")
    node_id_set = set(node_ids)
    for node in nodes:
        position = np.asarray(node["position"], dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"Node {node['node_id']!r} has an invalid position")
    for edge in edges:
        if str(edge["source"]) not in node_id_set or str(edge["target"]) not in node_id_set:
            raise ValueError(f"Edge {edge['edge_id']!r} references a missing node")
    primitives = payload.get("primitives") or []
    primitive_ids = [str(item["primitive_id"]) for item in primitives]
    if len(primitive_ids) != len(set(primitive_ids)):
        raise ValueError("Shape export contains duplicate primitive IDs")
    body_primitives = [item for item in primitives if item.get("role") == "body"]
    if len(body_primitives) != 1:
        raise ValueError("Final shape export must contain exactly one body primitive")
    crypt_primitives = [item for item in primitives if item.get("role") == "crypt"]
    tip_ids = {
        str(node["node_id"])
        for node in nodes
        if node.get("node_type") == "tip"
    }
    targeted_tips = {
        str(target)
        for primitive in crypt_primitives
        for target in primitive.get("target_node_ids") or []
        if str(target) in tip_ids
    }
    if targeted_tips != tip_ids:
        raise ValueError(
            "Every crypt tip must be targeted by exactly one reconstructive crypt primitive"
        )
    if len(crypt_primitives) != len(tip_ids):
        raise ValueError("Final shape export must contain one crypt primitive per tip")
    for primitive in crypt_primitives:
        primitive_tips = tip_ids.intersection(
            map(str, primitive.get("target_node_ids") or [])
        )
        if len(primitive_tips) != 1:
            raise ValueError(
                f"Crypt primitive {primitive['primitive_id']!r} must target one tip"
            )
    for primitive in primitives:
        missing = set(map(str, primitive.get("target_node_ids") or [])) - node_id_set
        if missing:
            raise ValueError(
                f"Primitive {primitive['primitive_id']!r} targets missing nodes {sorted(missing)}"
            )
        parameters = primitive.get("parameters") or {}
        primitive_type = primitive.get("primitive_type")
        if primitive_type in {"ellipsoid", "superellipsoid"}:
            _validate_blob_parameters(primitive, parameters, asymmetric=False)
        elif primitive_type == "asymmetric_superellipsoid":
            _validate_blob_parameters(primitive, parameters, asymmetric=True)
        elif primitive_type == _TUBE_TYPE:
            _validate_tube_parameters(primitive, parameters)
        elif primitive_type == _NECK_TYPE:
            _validate_neck_parameters(primitive, parameters)
        else:
            raise ValueError(
                f"Unsupported primitive type in shape payload: {primitive_type!r}"
            )
    transform = payload.get("coordinate_transform") or {}
    forward = np.asarray(transform.get("source_to_fitted"), dtype=float)
    inverse = np.asarray(transform.get("fitted_to_source"), dtype=float)
    if forward.shape != (4, 4) or inverse.shape != (4, 4):
        raise ValueError("Coordinate transforms must be 4x4 matrices")
    if not np.allclose(forward @ inverse, np.eye(4), atol=1e-8):
        raise ValueError("Coordinate transform matrices are not inverses")


def _validate_blob_parameters(
    primitive: dict[str, Any],
    parameters: dict[str, Any],
    *,
    asymmetric: bool,
) -> None:
    primitive_id = primitive["primitive_id"]
    center = np.asarray(parameters.get("center"), dtype=float)
    orientation = np.asarray(parameters.get("orientation"), dtype=float)
    if center.shape != (3,):
        raise ValueError(f"Blob {primitive_id!r} center must be a 3-vector")
    if orientation.shape != (3, 3) or not np.allclose(
        orientation @ orientation.T, np.eye(3), atol=1e-6
    ):
        raise ValueError(f"Blob {primitive_id!r} orientation must be orthonormal")
    axis_keys = (
        ("axis_lengths_negative", "axis_lengths_positive")
        if asymmetric
        else ("axis_lengths",)
    )
    for key in axis_keys:
        axes = np.asarray(parameters.get(key), dtype=float)
        if axes.shape != (3,) or np.any(axes <= 0.0):
            raise ValueError(f"Blob {primitive_id!r} {key} must contain three positive values")
    for key in ("epsilon_1", "epsilon_2"):
        if key in parameters and float(parameters[key]) <= 0.0:
            raise ValueError(f"Blob {primitive_id!r} {key} must be positive")


def _validate_curve(primitive_id: str, parameters: dict[str, Any]) -> None:
    controls = np.asarray(parameters.get("centerline_control_points"), dtype=float)
    curve_type = parameters.get("centerline_type")
    expected = {
        "line": 2,
        "quadratic_bezier": 3,
        "cubic_bezier": 4,
        "sinusoidal_bend": 2,
        "circular_arc": 2,
        "tangent_hermite": 2,
    }
    if controls.ndim != 2 or controls.shape[1:] != (3,) or controls.shape[0] < 2:
        raise ValueError(f"Primitive {primitive_id!r} has invalid centerline controls")
    if curve_type in expected and controls.shape[0] != expected[curve_type]:
        raise ValueError(
            f"Primitive {primitive_id!r} {curve_type} requires "
            f"{expected[curve_type]} controls"
        )
    if curve_type not in {*expected, "polyline"}:
        raise ValueError(f"Primitive {primitive_id!r} has unknown centerline type")
    if curve_type == "sinusoidal_bend":
        bend = np.asarray(parameters.get("centerline_bend_vector"), dtype=float)
        if bend.shape != (3,):
            raise ValueError(
                f"Primitive {primitive_id!r} sinusoidal bend requires a 3-vector"
            )
        chord = controls[1] - controls[0]
        tolerance = 1e-7 * max(float(np.linalg.norm(chord)), 1.0)
        if abs(float(np.dot(bend, chord))) > tolerance:
            raise ValueError(
                f"Primitive {primitive_id!r} bend vector must be transverse"
            )
    if curve_type == "circular_arc":
        sagitta = np.asarray(parameters.get("centerline_sagitta_vector"), dtype=float)
        if sagitta.shape != (3,):
            raise ValueError(
                f"Primitive {primitive_id!r} circular arc requires a 3-vector sagitta"
            )
    if curve_type == "tangent_hermite":
        for key in ("centerline_start_tangent", "centerline_end_tangent"):
            tangent = np.asarray(parameters.get(key), dtype=float)
            if tangent.shape != (3,) or np.linalg.norm(tangent) <= 1e-12:
                raise ValueError(
                    f"Primitive {primitive_id!r} tangent Hermite curve requires "
                    f"a non-zero {key}"
                )


def _validate_tube_parameters(
    primitive: dict[str, Any],
    parameters: dict[str, Any],
) -> None:
    primitive_id = primitive["primitive_id"]
    _validate_curve(primitive_id, parameters)
    is_v3 = "r_attachment" in parameters
    radius_keys = (
        ("r_attachment", "r_center", "r_distal")
        if is_v3
        else ("r_neck", "r_body", "r_tip")
    )
    for key in radius_keys:
        if float(parameters.get(key, 0.0)) <= 0.0:
            raise ValueError(f"Crypt tube {primitive_id!r} {key} must be positive")
    s_body = float(parameters.get("s_center", parameters.get("s_body", -1.0)))
    s_taper = float(parameters.get("s_taper", -1.0))
    if not 0.0 < s_body < s_taper < 1.0:
        raise ValueError(
            f"Crypt tube {primitive_id!r} requires 0 < s_center < s_taper < 1"
        )
    r_constriction = parameters.get("r_constriction")
    s_constriction = parameters.get("s_constriction")
    if (r_constriction is None) != (s_constriction is None):
        raise ValueError(
            f"Crypt tube {primitive_id!r} constriction radius and position must "
            "both be present or both be null"
        )
    if r_constriction is not None:
        if float(r_constriction) <= 0.0 or not 0.0 <= float(s_constriction) <= 1.0:
            raise ValueError(f"Crypt tube {primitive_id!r} has an invalid constriction")
    opening_normal = parameters.get("opening_normal")
    if opening_normal is not None:
        opening_normal = np.asarray(opening_normal, dtype=float)
        if opening_normal.shape != (3,) or np.linalg.norm(opening_normal) <= 1e-12:
            raise ValueError(f"Crypt tube {primitive_id!r} has an invalid opening normal")
        blend = float(parameters.get("opening_frame_blend_fraction", 0.15))
        if not 0.0 < blend <= 1.0:
            raise ValueError(
                f"Crypt tube {primitive_id!r} has an invalid opening-frame blend"
            )


def _validate_neck_parameters(
    primitive: dict[str, Any],
    parameters: dict[str, Any],
) -> None:
    primitive_id = primitive["primitive_id"]
    _validate_curve(primitive_id, parameters)
    if float(parameters.get("radius", 0.0)) <= 0.0:
        raise ValueError(f"Neck cylinder {primitive_id!r} radius must be positive")


def _expanded_parameters(record: dict[str, Any]) -> dict[str, Any]:
    primitive_type = str(record["primitive_type"])
    compact = dict(record["parameters"])
    if primitive_type not in {_TUBE_TYPE, _NECK_TYPE}:
        return compact
    controls = np.asarray(compact["centerline_control_points"], dtype=float)
    curve_type = compact["centerline_type"]
    n_samples = int(compact.get("centerline_samples", 64))
    if curve_type == "quadratic_bezier":
        centerline = sample_quadratic_bezier(*controls, n_samples=n_samples)
    elif curve_type == "cubic_bezier":
        centerline = sample_cubic_bezier(*controls, n_samples=n_samples)
    elif curve_type == "sinusoidal_bend":
        centerline = sample_sinusoidal_bend(
            controls[0],
            controls[1],
            compact["centerline_bend_vector"],
            n_samples=n_samples,
        )
    elif curve_type == "circular_arc":
        centerline = sample_circular_arc(
            controls[0],
            controls[1],
            compact["centerline_sagitta_vector"],
            n_samples=n_samples,
        )
    elif curve_type == "tangent_hermite":
        centerline = sample_tangent_hermite(
            controls[0],
            controls[1],
            compact["centerline_start_tangent"],
            compact["centerline_end_tangent"],
            n_samples=n_samples,
        )
    else:
        centerline = controls
    expanded = {**compact, "centerline_points": centerline}
    if primitive_type == _TUBE_TYPE:
        r_attachment = compact.get("r_attachment", compact.get("r_neck"))
        r_center = compact.get("r_center", compact.get("r_body"))
        r_distal = compact.get("r_distal", compact.get("r_tip"))
        s_center = compact.get("s_center", compact.get("s_body", 0.5))
        expanded.update(
            {
                "r_attachment": r_attachment,
                "r_neck": r_attachment,
                "r_center": r_center,
                "r_body": r_center,
                "r_distal": r_distal,
                "r_tip": r_distal,
                "r_taper": r_distal,
                "s_center": s_center,
                "s_body": s_center,
                "distal_taper_start": compact["s_taper"],
            }
        )
    return expanded


def graph_from_shape_export_payload(
    payload: dict[str, Any],
    *,
    coordinate_system: str = "fitted",
) -> SkeletonGraph:
    """Reconstruct a final graph from a v5 or legacy v2-v4 payload."""
    validate_shape_export_payload(payload)
    graph = SkeletonGraph(
        metadata={"sample": dict(payload.get("sample") or {})},
        coordinate_frame={"kind": "prepared_mesh"},
    )
    for node in payload["skeleton"]["nodes"]:
        graph.add_node(
            node["node_id"],
            node["node_type"],
            node["position"],
            crypt_id=node.get("crypt_id"),
        )
    for edge in payload["skeleton"]["edges"]:
        graph.add_edge(
            edge["edge_id"],
            edge["source"],
            edge["target"],
            edge_type=edge.get("edge_type", "skeleton"),
            crypt_id=edge.get("crypt_id"),
        )
    for record in payload.get("primitives") or []:
        attachment = PrimitiveAttachment(
            primitive_type=record["primitive_type"],
            parameters=_expanded_parameters(record),
            attachment_type=(
                "path" if record["attachment_scope"] == "graph" else record["attachment_scope"]
            ),
            attachment_id=record["primitive_id"],
            target_ids=list(record.get("target_node_ids") or []),
            metadata={"role": record.get("role")},
        )
        scope = record["attachment_scope"]
        owner_id = str(record["owner_id"])
        if scope == "node":
            graph.node(owner_id).primitive_attachment = attachment
        elif scope == "edge":
            graph.edge(owner_id).primitive_attachment = attachment
        elif scope == "graph":
            graph.add_primitive_attachment(owner_id, attachment)
        else:
            raise ValueError(f"Unknown primitive attachment scope {scope!r}")
    if coordinate_system == "source":
        _transform_graph_to_source(graph, payload["coordinate_transform"])
    elif coordinate_system != "fitted":
        raise ValueError("coordinate_system must be 'fitted' or 'source'")
    return graph


def _transform_points(points, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _transform_graph_to_source(graph: SkeletonGraph, transform: dict[str, Any]) -> None:
    matrix = np.asarray(transform["fitted_to_source"], dtype=float)
    linear = matrix[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    scale = float(np.mean(singular_values))
    if not np.allclose(singular_values, scale, rtol=1e-7, atol=1e-9):
        raise ValueError("Only uniform fitted-to-source scaling is supported")
    rotation = linear / scale
    for node in graph.nodes.values():
        node.position = _transform_points(node.position[None, :], matrix)[0]

    attachments = [item[2] for item in _all_attachments(graph)]
    for attachment in attachments:
        parameters = attachment.parameters
        for key in ("center", "neck_center"):
            if key in parameters and parameters[key] is not None:
                parameters[key] = _transform_points(
                    np.asarray(parameters[key], dtype=float)[None, :], matrix
                )[0]
        for key in ("centerline_points", "centerline_control_points"):
            if key in parameters and parameters[key] is not None:
                parameters[key] = _transform_points(parameters[key], matrix)
        if parameters.get("opening_normal") is not None:
            opening_normal = rotation @ np.asarray(
                parameters["opening_normal"], dtype=float
            )
            parameters["opening_normal"] = opening_normal / max(
                float(np.linalg.norm(opening_normal)), 1e-12
            )
        if parameters.get("centerline_bend_vector") is not None:
            parameters["centerline_bend_vector"] = linear @ np.asarray(
                parameters["centerline_bend_vector"], dtype=float
            )
        if parameters.get("centerline_sagitta_vector") is not None:
            parameters["centerline_sagitta_vector"] = linear @ np.asarray(
                parameters["centerline_sagitta_vector"], dtype=float
            )
        for key in ("centerline_start_tangent", "centerline_end_tangent"):
            if parameters.get(key) is not None:
                parameters[key] = linear @ np.asarray(parameters[key], dtype=float)
        if "orientation" in parameters:
            parameters["orientation"] = rotation @ np.asarray(
                parameters["orientation"], dtype=float
            )
        for key in (
            "axis_lengths",
            "axis_lengths_negative",
            "axis_lengths_positive",
            "r_neck",
            "r_attachment",
            "r_body",
            "r_center",
            "r_tip",
            "r_distal",
            "r_taper",
            "r_constriction",
            "radius",
        ):
            if key in parameters and parameters[key] is not None:
                parameters[key] = np.asarray(parameters[key], dtype=float) * scale
                if np.ndim(parameters[key]) == 0:
                    parameters[key] = float(parameters[key])
    graph.coordinate_frame = {"kind": "original_mesh"}


def shape_quality_payload(
    result: SkeletonGraph | SkeletonizationResult | PrimitiveFitResult | OrganoidShapeResult,
    *,
    primitive_result: PrimitiveFitResult | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a non-VAE sidecar containing crypt fitting diagnostics.

    Unlike ``shape.json``, this payload is explicitly operational: optimizer
    state, residuals, support counts, and constraint activation are useful for
    filtering and fitting experiments but are not reconstructive shape degrees
    of freedom.
    """
    coerced = _coerce_shape_result(result, primitive_result=primitive_result)
    sample = _sample_record(coerced, metadata)
    crypt_records = []
    for scope, owner_id, attachment in _all_attachments(coerced.graph):
        if attachment.primitive_type != _TUBE_TYPE:
            continue
        crypt_records.append(
            {
                "primitive_id": str(attachment.attachment_id or owner_id),
                "attachment_scope": scope,
                "owner_id": str(owner_id),
                "target_node_ids": list(attachment.target_ids),
                "fit_error": attachment.fit_error,
                "residuals": dict(attachment.residuals),
                "n_points": attachment.metadata.get("n_points"),
                "fit_method": attachment.metadata.get("fit_method"),
                "centerline_method": attachment.metadata.get("centerline_method"),
                "tip_source": attachment.metadata.get("tip_source"),
                "tip_vertex_id": attachment.metadata.get("tip_vertex_id"),
                "centerline_kind": attachment.metadata.get("centerline_kind"),
                "centerline_sagitta_vector": attachment.metadata.get(
                    "centerline_sagitta_vector"
                ),
                "centerline_start_tangent": attachment.metadata.get(
                    "centerline_start_tangent"
                ),
                "centerline_end_tangent": attachment.metadata.get(
                    "centerline_end_tangent"
                ),
                "centerline_tangent_length": attachment.metadata.get(
                    "centerline_tangent_length"
                ),
                "centerline_fit_rmse": attachment.metadata.get(
                    "centerline_fit_rmse"
                ),
                "opening_normal_source": attachment.metadata.get(
                    "opening_normal_source"
                ),
                "tip_normal": attachment.metadata.get("tip_normal"),
                "ratio_contours": dict(
                    attachment.metadata.get("ratio_contours") or {}
                ),
                "profile_optimization": dict(
                    attachment.metadata.get("profile_optimization") or {}
                ),
                "profile_observations": dict(
                    attachment.metadata.get("profile_observations") or {}
                ),
                "volume_center": dict(
                    attachment.metadata.get("volume_center") or {}
                ),
            }
        )
    return _nullable_quality_json(
        {
            "schema_version": SHAPE_QUALITY_SCHEMA_VERSION,
            "sample": sample,
            "crypt_primitives": crypt_records,
        }
    )


def save_shape_export(
    result: SkeletonGraph | SkeletonizationResult | PrimitiveFitResult | OrganoidShapeResult,
    output_dir,
    *,
    primitive_result: PrimitiveFitResult | None = None,
    metadata: dict[str, Any] | None = None,
    prefix: str = "shape",
    save_quality: bool = True,
) -> dict[str, str]:
    """Write compact shape JSON and an optional non-VAE quality sidecar."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = shape_export_payload(
        result,
        primitive_result=primitive_result,
        metadata=metadata,
    )
    path = output_dir / f"{prefix}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    paths = {"json": str(path)}
    if save_quality:
        quality_path = output_dir / "quality.json"
        quality_path.write_text(
            json.dumps(
                shape_quality_payload(
                    result,
                    primitive_result=primitive_result,
                    metadata=metadata,
                ),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        paths["quality_json"] = str(quality_path)
    return paths


def load_shape_export_json(path) -> dict[str, Any]:
    """Load and validate a compact v5 or legacy v2-v4 shape payload."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_shape_export_payload(payload)
    return payload


def load_shape_export_graph(path, *, coordinate_system: str = "fitted") -> SkeletonGraph:
    """Load an export directly as a fitted- or original-coordinate graph."""
    return graph_from_shape_export_payload(
        load_shape_export_json(path),
        coordinate_system=coordinate_system,
    )


def write_export_readme(path, *, dataset: str | None = None) -> None:
    """Write the compact batch-export data dictionary."""
    dataset_line = f"Dataset: `{dataset}`.\n\n" if dataset else ""
    text = f"""# OrganoGraph Shape Export v5

{dataset_line}Each organoid directory contains `shape.json` with the final
skeleton and fitted primitives shown by `notebooks/tutorial_skeleton.ipynb`.
New exports also contain `quality.json` with crypt optimizer, support, and
residual diagnostics. This sidecar is for filtering and fitting audits; it is
not part of the VAE shape representation.

Detection arrays and component masks are not included. Fit errors and residuals
remain excluded from `shape.json`.

The file contains sample identity and VAE eligibility, reversible original-mesh
coordinate transforms, graph nodes and edges, and reconstructive primitive
parameters. Crypt array order is explicitly non-semantic; use `crypt_id` and
graph connectivity or permutation-invariant matching.

Crypt tubes use endpoint-normal Hermite centerlines and semantic
attachment/center/distal radii. Their cap onset is a fixed reconstruction
setting (`s_taper=0.85`), not a VAE degree of freedom. The graph crypt node is
placed at the fitted tube volume center computed from `r(s)^2`.

Use `organograph.skeleton.load_shape_export_graph(path)` to reconstruct the
prepared-mesh graph, or pass `coordinate_system="source"` to restore positions,
scales, and orientations to the original mesh coordinate system.
"""
    Path(path).write_text(text, encoding="utf-8")

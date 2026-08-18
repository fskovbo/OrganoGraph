"""Build SkeletonGraph objects from normalized crypt detections."""

from __future__ import annotations

from typing import Any

import numpy as np

from organograph.skeleton.datatypes import SkeletonGraph
from organograph.skeleton.detection.common import (
    _coerce_patch,
    _first_present,
    _json_safe_metadata,
    _point_from_keys,
    _point_from_vertex,
)
from organograph.skeleton.detection.mesh_regions import (
    _radial_distances_to_axis,
)
from organograph.skeleton.detection.neck_profiles import _neck_position
from organograph.skeleton.geometry import as_points, centroid

def _tip_position(vertices, detection: dict[str, Any]) -> np.ndarray:
    explicit = _point_from_keys(
        vertices,
        detection,
        ("tip_position", "tip_center", "tip", "bottom_position", "crypt_tip", "p_tip"),
    )
    if explicit is not None:
        return explicit

    vertex_id = _first_present(
        detection,
        ("tip_vertex_id", "bottom_vertex_id", "bottom", "bottom_vertex"),
    )
    by_vertex = _point_from_vertex(vertices, vertex_id)
    if by_vertex is not None:
        return by_vertex

    patch = _coerce_patch(
        _first_present(
            detection,
            ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
        )
    )
    if patch.size:
        return centroid(as_points(vertices)[patch])
    raise ValueError("Crypt detection is missing a tip/bottom position or vertex id.")

def _crypt_position(vertices, detection: dict[str, Any]) -> np.ndarray:
    explicit = _point_from_keys(
        vertices,
        detection,
        ("crypt_position", "crypt_center", "crypt_centroid", "p_crypt"),
    )
    if explicit is not None:
        return explicit

    patch = _coerce_patch(
        _first_present(
            detection,
            ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
        )
    )
    if patch.size:
        return centroid(as_points(vertices)[patch])
    return _tip_position(vertices, detection)

def _penalize_short_crypt_bending(
    vertices,
    crypt_vertices,
    source_position,
    intermediate_position,
    tip_position,
    *,
    max_dimensionless_curvature: float | None,
    penalty_strength: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Softly pull a crypt waypoint toward its chord when bending is excessive."""
    source = np.asarray(source_position, dtype=float)
    candidate = np.asarray(intermediate_position, dtype=float)
    tip = np.asarray(tip_position, dtype=float)
    limit = (
        None
        if max_dimensionless_curvature is None
        else float(max_dimensionless_curvature)
    )
    diagnostics = {
        "applied": False,
        "max_dimensionless_curvature": limit,
        "penalty_strength": float(penalty_strength),
        "original_dimensionless_curvature": None,
        "final_dimensionless_curvature": None,
        "waypoint_lateral_scale": 1.0,
    }
    chord = tip - source
    chord_length = float(np.linalg.norm(chord))
    patch = _coerce_patch(crypt_vertices)
    if (
        limit is None
        or limit <= 0.0
        or float(penalty_strength) <= 0.0
        or chord_length <= 1e-12
        or patch.size < 3
    ):
        diagnostics["reason"] = "disabled_or_insufficient_geometry"
        return candidate, diagnostics

    unit = chord / chord_length
    longitudinal = float(
        np.clip(
            np.dot(candidate - source, unit),
            0.1 * chord_length,
            0.9 * chord_length,
        )
    )
    projection = source + longitudinal * unit
    radial = _radial_distances_to_axis(
        as_points(vertices)[patch],
        source,
        chord,
    )
    radial = radial[np.isfinite(radial)]
    if radial.size < 3:
        diagnostics["reason"] = "insufficient_radius_samples"
        return candidate, diagnostics
    crypt_radius = float(np.median(radial))
    if crypt_radius <= 1e-12:
        diagnostics["reason"] = "degenerate_crypt_radius"
        return candidate, diagnostics

    def curvature(point: np.ndarray) -> tuple[float, float, float]:
        first = point - source
        second = tip - point
        n_first = float(np.linalg.norm(first))
        n_second = float(np.linalg.norm(second))
        path_length = n_first + n_second
        if n_first <= 1e-12 or n_second <= 1e-12 or path_length <= 1e-12:
            return 0.0, 0.0, path_length
        cosine = float(np.clip(np.dot(first, second) / (n_first * n_second), -1.0, 1.0))
        angle = float(np.arccos(cosine))
        return angle * crypt_radius / path_length, angle, path_length

    original_curvature, original_angle, original_length = curvature(candidate)
    diagnostics.update(
        {
            "crypt_radius": crypt_radius,
            "original_dimensionless_curvature": original_curvature,
            "original_bend_angle": original_angle,
            "original_path_length": original_length,
        }
    )
    if original_curvature <= limit:
        diagnostics.update(
            {
                "reason": "within_curvature_limit",
                "final_dimensionless_curvature": original_curvature,
            }
        )
        return candidate, diagnostics

    lateral = candidate - projection
    alphas = np.linspace(0.0, 1.0, 101)
    objectives = []
    for alpha in alphas:
        trial = projection + float(alpha) * lateral
        trial_curvature, _, _ = curvature(trial)
        excess = max(trial_curvature / limit - 1.0, 0.0)
        objectives.append((1.0 - float(alpha)) ** 2 + float(penalty_strength) * excess**2)
    best_index = int(np.argmin(objectives))
    best_alpha = float(alphas[best_index])
    refined = projection + best_alpha * lateral
    final_curvature, final_angle, final_length = curvature(refined)
    diagnostics.update(
        {
            "applied": best_alpha < 1.0 - 1e-12,
            "reason": "curvature_penalty_applied",
            "waypoint_lateral_scale": best_alpha,
            "final_dimensionless_curvature": final_curvature,
            "final_bend_angle": final_angle,
            "final_path_length": final_length,
        }
    )
    return refined, diagnostics

def _add_crypt_tip_path(
    graph: SkeletonGraph,
    vertices,
    *,
    path_prefix: str,
    source_id: str,
    tip_id: str,
    crypt_id,
    detection: dict[str, Any],
    crypt_vertices: np.ndarray,
    source_position: np.ndarray,
    tip_position: np.ndarray,
    metadata: dict[str, Any],
    bend_max_dimensionless_curvature: float | None,
    bend_curvature_penalty: float,
    crypt_role: str = "crypt_centroid",
) -> None:
    """Connect an attachment or constriction to a tip through its crypt center."""
    intermediate_position = _crypt_position(vertices, detection)
    intermediate_type = "crypt"

    intermediate_position, bend_diagnostics = _penalize_short_crypt_bending(
        vertices,
        crypt_vertices,
        source_position,
        intermediate_position,
        tip_position,
        max_dimensionless_curvature=bend_max_dimensionless_curvature,
        penalty_strength=bend_curvature_penalty,
    )
    intermediate_id = f"{path_prefix}_{intermediate_type}"
    graph.add_node(
        intermediate_id,
        intermediate_type,
        intermediate_position,
        crypt_id=crypt_id,
        metadata={
            **metadata,
            "role": crypt_role,
            "bend_validation": _json_safe_metadata(bend_diagnostics),
        },
    )
    source_type = graph.node(source_id).node_type
    graph.add_edge(
        f"{path_prefix}_{source_type}_to_{intermediate_type}",
        source_id,
        intermediate_id,
        edge_type=f"{source_type}_to_{intermediate_type}",
        crypt_id=crypt_id,
    )
    graph.add_edge(
        f"{path_prefix}_{intermediate_type}_to_tip",
        intermediate_id,
        tip_id,
        edge_type=f"{intermediate_type}_to_tip",
        crypt_id=crypt_id,
    )

def _add_attachment_path(
    graph: SkeletonGraph,
    vertices,
    faces,
    *,
    path_prefix: str,
    host_id: str,
    crypt_id,
    detection: dict[str, Any],
    metadata: dict[str, Any],
    host_edge_prefix: str,
) -> tuple[str, np.ndarray]:
    """Add a branch neck or a terminal attachment/constriction sequence."""
    profile = detection.get("neck_profile")
    if not isinstance(profile, dict):
        neck = _neck_position(vertices, faces, detection)
        neck_id = f"{path_prefix}_neck"
        graph.add_node(
            neck_id,
            "neck",
            neck,
            crypt_id=crypt_id,
            metadata=metadata,
        )
        graph.add_edge(
            f"{path_prefix}_{host_edge_prefix}_to_neck",
            host_id,
            neck_id,
            edge_type=f"{host_edge_prefix}_to_neck",
            crypt_id=crypt_id,
        )
        return neck_id, neck

    attachment = _point_from_keys(
        vertices,
        detection,
        ("attachment_position", "attachment_center", "p_attachment"),
    )
    if attachment is None:
        attachment = _neck_position(vertices, faces, detection)
    attachment_id = f"{path_prefix}_attachment"
    junction_meta = {
        **metadata,
        "neck_profile": _json_safe_metadata(profile),
        "attachment_level": float(detection.get("attachment_level", 1.0)),
    }
    graph.add_node(
        attachment_id,
        "attachment",
        attachment,
        crypt_id=crypt_id,
        metadata={**junction_meta, "role": "component_attachment"},
    )
    graph.add_edge(
        f"{path_prefix}_{host_edge_prefix}_to_attachment",
        host_id,
        attachment_id,
        edge_type=f"{host_edge_prefix}_to_attachment",
        crypt_id=crypt_id,
    )
    if profile.get("kind") != "constriction":
        return attachment_id, attachment

    constriction = _point_from_keys(
        vertices,
        detection,
        ("constriction_position", "constriction_center", "neck_position"),
    )
    if constriction is None:
        return attachment_id, attachment
    constriction_id = f"{path_prefix}_constriction"
    graph.add_node(
        constriction_id,
        "constriction",
        constriction,
        crypt_id=crypt_id,
        metadata={
            **junction_meta,
            "role": "narrowest_constriction",
            "constriction_level": profile.get("constriction_level"),
            "distal_boundary_level": profile.get("distal_boundary_level"),
            "c_min": profile.get("c_min"),
            "c_half": profile.get("c_half"),
        },
    )
    graph.add_edge(
        f"{path_prefix}_attachment_to_constriction",
        attachment_id,
        constriction_id,
        edge_type="attachment_to_constriction",
        crypt_id=crypt_id,
    )
    return constriction_id, constriction

def _branch_center_override(
    branch_center_overrides: dict[Any, Any] | None,
    *,
    branch_node_id: str,
    crypt_id,
) -> np.ndarray | None:
    """Resolve an externally fitted center for one branch node."""
    if not branch_center_overrides:
        return None
    candidates = (branch_node_id, crypt_id, str(crypt_id))
    for key in candidates:
        if key not in branch_center_overrides:
            continue
        center = np.asarray(branch_center_overrides[key], dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError(
                f"Branch center override for {branch_node_id!r} must be a finite 3-vector"
            )
        return center
    return None

def _daughter_detections(detection: dict[str, Any]) -> list[dict[str, Any]]:
    daughters = _first_present(detection, ("daughters", "daughter_tips", "branches", "children"))
    if daughters is None:
        return []
    out = []
    for daughter in daughters:
        if isinstance(daughter, dict):
            out.append(dict(daughter))
        else:
            arr = np.asarray(daughter)
            if arr.shape == (3,):
                out.append({"tip_position": arr})
            else:
                out.append({"tip_vertex_id": int(arr)})
    return out

def build_skeleton_graph(
    vertices,
    faces,
    crypt_detections,
    *,
    body_center,
    branch_centers: dict[Any, Any] | None = None,
    bend_max_dimensionless_curvature: float | None = 0.5,
    bend_curvature_penalty: float = 8.0,
    metadata: dict[str, Any] | None = None,
) -> SkeletonGraph:
    """Build the fixed biology-aware topology from barrier-bounded detections.

    Body and branch centers are supplied by their barrier primitives. Every
    crypt path contains one centroid waypoint, so bending remains represented
    by straight graph edges while the later tube fit may use a smooth midline.
    """
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    if crypt_detections is None:
        detections = []
    elif not all(isinstance(item, dict) for item in crypt_detections):
        raise TypeError("crypt_detections must be a sequence of detection dictionaries")
    else:
        detections = [dict(item) for item in crypt_detections]

    graph = SkeletonGraph(
        metadata=_json_safe_metadata(metadata),
        coordinate_frame={
            "kind": "raw",
            "body_center_node": "body",
            "description": "Raw mesh/world coordinates; edges are straight segments.",
        },
    )
    body_position = np.asarray(body_center, dtype=float)
    if body_position.shape != (3,) or not np.all(np.isfinite(body_position)):
        raise ValueError("body_center must be a finite 3-vector from the body barrier")

    graph.add_node(
        "body",
        "body",
        body_position,
        metadata={
            "role": "villus_body_center",
            "center_source": "body_barrier_primitive",
        },
    )

    for i, detection in enumerate(detections):
        crypt_id = detection.get("crypt_id", i)
        crypt_prefix = f"crypt_{crypt_id}"
        crypt_vertices = _coerce_patch(
            _first_present(
                detection,
                ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
            )
        )
        common_meta = _json_safe_metadata(
            {
                "source_detection_index": i,
                "n_crypt_vertices": int(crypt_vertices.size),
                **dict(detection.get("metadata", {})),
            }
        )

        root_source_id, root_source_position = _add_attachment_path(
            graph,
            vertices,
            faces,
            path_prefix=crypt_prefix,
            host_id="body",
            crypt_id=crypt_id,
            detection=detection,
            metadata=common_meta,
            host_edge_prefix="body",
        )

        daughters = _daughter_detections(detection)
        if daughters:
            daughter_tips = [_tip_position(vertices, daughter) for daughter in daughters]
            branch_id = f"{crypt_prefix}_branch"
            branch = _branch_center_override(
                branch_centers,
                branch_node_id=branch_id,
                crypt_id=crypt_id,
            )
            center_source = "branch_barrier_primitive"
            if branch is None:
                branch = _point_from_keys(
                    vertices,
                    detection,
                    ("branch_position", "branch_center", "split_position"),
                )
                center_source = "explicit_detection_position"
            if branch is None:
                raise ValueError(f"Accepted branch {branch_id!r} has no center")
            graph.add_node(
                branch_id,
                "branch",
                branch,
                crypt_id=crypt_id,
                metadata={
                    **common_meta,
                    "n_daughters": len(daughters),
                    "center_source": center_source,
                },
            )
            graph.add_edge(
                f"{crypt_prefix}_{graph.node(root_source_id).node_type}_to_branch",
                root_source_id,
                branch_id,
                edge_type=f"{graph.node(root_source_id).node_type}_to_branch",
                crypt_id=crypt_id,
            )
            for j, daughter in enumerate(daughters):
                daughter_meta = _json_safe_metadata(
                    {
                        **common_meta,
                        "daughter_index": j,
                        **dict(daughter.get("metadata", {})),
                    }
                )
                daughter_source_id, daughter_source_position = _add_attachment_path(
                    graph,
                    vertices,
                    faces,
                    path_prefix=f"{crypt_prefix}_daughter_{j}",
                    host_id=branch_id,
                    crypt_id=crypt_id,
                    detection=daughter,
                    metadata={
                        **daughter_meta,
                        "role": "daughter_junction",
                    },
                    host_edge_prefix="branch",
                )

                tip_id = f"{crypt_prefix}_tip_{j}"
                graph.add_node(
                    tip_id,
                    "tip",
                    daughter_tips[j],
                    crypt_id=crypt_id,
                    metadata=daughter_meta,
                )
                daughter_vertices = _coerce_patch(
                    _first_present(
                        daughter,
                        ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
                    )
                )
                _add_crypt_tip_path(
                    graph,
                    vertices,
                    path_prefix=f"{crypt_prefix}_daughter_{j}",
                    source_id=daughter_source_id,
                    tip_id=tip_id,
                    crypt_id=crypt_id,
                    detection=daughter,
                    crypt_vertices=daughter_vertices,
                    source_position=daughter_source_position,
                    tip_position=daughter_tips[j],
                    metadata=daughter_meta,
                    bend_max_dimensionless_curvature=bend_max_dimensionless_curvature,
                    bend_curvature_penalty=bend_curvature_penalty,
                    crypt_role="daughter_crypt_centroid",
                )
            continue

        tip = _tip_position(vertices, detection)
        tip_id = f"{crypt_prefix}_tip"
        graph.add_node(
            tip_id,
            "tip",
            tip,
            crypt_id=crypt_id,
            metadata=common_meta,
        )
        _add_crypt_tip_path(
            graph,
            vertices,
            path_prefix=crypt_prefix,
            source_id=root_source_id,
            tip_id=tip_id,
            crypt_id=crypt_id,
            detection=detection,
            crypt_vertices=crypt_vertices,
            source_position=root_source_position,
            tip_position=tip,
            metadata=common_meta,
            bend_max_dimensionless_curvature=bend_max_dimensionless_curvature,
            bend_curvature_penalty=bend_curvature_penalty,
        )

    return graph

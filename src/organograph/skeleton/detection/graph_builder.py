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
    _body_center_from_root_regions,
    _branch_position_from_regions,
    _crypt_side_region,
    _radial_distances_to_axis,
)
from organograph.skeleton.detection.neck_profiles import _neck_position, _add_neck_profile_geometry
from organograph.skeleton.geometry import as_points, centroid, estimate_bend_position, surface_area_centroid

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
    bend_strategy: str,
    bend_max_dimensionless_curvature: float | None,
    bend_curvature_penalty: float,
    crypt_role: str = "crypt_centroid",
) -> None:
    """Connect a crypt neck-like source to a tip, optionally through a waypoint."""
    strategy = str(bend_strategy).lower()
    intermediate_position = None
    intermediate_type = None
    intermediate_role = None

    if strategy != "none":
        if strategy == "crypt_centroid":
            intermediate_position = _crypt_position(vertices, detection)
            intermediate_type = "crypt"
            intermediate_role = crypt_role
        else:
            intermediate_position = _point_from_keys(
                vertices,
                detection,
                ("bend_position", "bend_center", "bend", "p_bend"),
            )
            if intermediate_position is None:
                intermediate_position = estimate_bend_position(
                    vertices,
                    crypt_vertices,
                    source_position,
                    tip_position,
                    strategy=strategy,
                )
            if intermediate_position is not None:
                intermediate_type = "bend"
                intermediate_role = "bend"

    if intermediate_position is None:
        source_type = graph.node(source_id).node_type
        graph.add_edge(
            f"{path_prefix}_{source_type}_to_tip",
            source_id,
            tip_id,
            edge_type=f"{source_type}_to_tip",
            crypt_id=crypt_id,
        )
        return

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
            "role": intermediate_role,
            "bend_strategy": strategy,
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
    """Add legacy neck or explicit attachment/constriction nodes."""
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

def _branch_position(vertices, detection: dict[str, Any], neck, daughter_tips) -> np.ndarray:
    explicit = _point_from_keys(
        vertices,
        detection,
        ("branch_position", "branch_center", "branch", "split_position", "split_center"),
    )
    if explicit is not None:
        return explicit

    vertex_id = _first_present(detection, ("branch_vertex_id", "split_vertex_id"))
    by_vertex = _point_from_vertex(vertices, vertex_id)
    if by_vertex is not None:
        return by_vertex

    stem_vertices = _coerce_patch(_first_present(detection, ("stem_vertices", "trunk_vertices")))
    if stem_vertices.size:
        return centroid(as_points(vertices)[stem_vertices])

    daughter_mean = centroid(np.vstack(daughter_tips))
    return 0.5 * (np.asarray(neck, dtype=float) + daughter_mean)


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

def normalize_crypt_detections(crypt_detections) -> list[dict[str, Any]]:
    """Normalize common segmentation outputs to a list of detection dicts.

    Accepted inputs include:
    - list of dicts with explicit neck/tip fields;
    - list of vertex-index patches;
    - segmentation dictionaries containing `crypts_mesh`, `crypts_ll`, or
      `crypts`, optionally with per-crypt arrays such as `bottom_vertex_ids`
      and `d_crypts`.
    """
    if crypt_detections is None:
        return []

    if isinstance(crypt_detections, dict):
        patches = _first_present(
            crypt_detections,
            ("crypt_detections", "crypts_mesh", "crypts_ll", "crypts", "patches"),
        )
        if patches is not None and not isinstance(patches, dict):
            if all(isinstance(patch, dict) for patch in patches):
                return [dict(patch, crypt_id=patch.get("crypt_id", i)) for i, patch in enumerate(patches)]
            detections = []
            for i, patch in enumerate(patches):
                det = {"crypt_id": i, "crypt_vertices": patch}
                for src_key, dst_key in (
                    ("bottom_vertex_ids", "bottom_vertex_id"),
                    ("tip_vertex_ids", "tip_vertex_id"),
                    ("d_crypts", "d_crypt"),
                    ("L_crypts", "L_crypt"),
                    ("circumference_crypts", "circumference"),
                    ("crypt_constrictions", "constriction"),
                    ("crypt_elongations", "elongation"),
                ):
                    if src_key in crypt_detections:
                        values = crypt_detections[src_key]
                        if len(values) > i:
                            det[dst_key] = values[i]
                detections.append(det)
            return detections
        return [dict(crypt_detections)]

    detections = []
    for i, item in enumerate(crypt_detections):
        if isinstance(item, dict):
            det = dict(item)
            det.setdefault("crypt_id", i)
        else:
            det = {"crypt_id": i, "crypt_vertices": item}
        detections.append(det)
    return detections

def _body_center(vertices, faces, body_vertices, body_faces, body_center) -> np.ndarray:
    if body_center is not None:
        center = np.asarray(body_center, dtype=float)
        if center.shape != (3,):
            raise ValueError("body_center must be a 3-vector")
        return center
    vertices = as_points(vertices)
    if body_vertices is not None:
        idx = _coerce_patch(body_vertices)
        if idx.size:
            return centroid(vertices[idx])
    if body_faces is not None:
        return surface_area_centroid(vertices, np.asarray(body_faces, dtype=np.int64))
    if faces is not None:
        return surface_area_centroid(vertices, faces)
    return centroid(vertices)

def build_skeleton_from_crypt_detections(
    vertices,
    faces,
    crypt_detections,
    body_vertices=None,
    body_faces=None,
    body_center=None,
    branch_center_overrides: dict[Any, Any] | None = None,
    bend_strategy: str = "none",
    bend_max_dimensionless_curvature: float | None = 0.5,
    bend_curvature_penalty: float = 5.0,
    refine_body_center_from_necks: bool = True,
    refine_branch_centers_from_necks: bool = True,
    metadata: dict[str, Any] | None = None,
) -> SkeletonGraph:
    """Build a straight-edge organoid skeleton from crypt detections.

    Each non-split crypt is represented as `body -> neck -> tip` by default.
    When an intermediate waypoint is requested through ``bend_strategy``, the
    crypt path becomes either `neck -> bend -> tip` or, for
    ``bend_strategy="crypt_centroid"``, `neck -> crypt -> tip`.  Split
    detections with daughters use the same daughter-neck to daughter-tip rule.
    Optional crypt waypoints are softly straightened when their bend is too
    sharp for the path length and estimated crypt radius.

    An explicit ``body_center`` and entries in ``branch_center_overrides`` take
    precedence over region-derived centers. This allows an upstream component
    primitive, such as a soft-barrier ellipsoid, to define the corresponding
    skeleton node center.

    When enabled, body and branch node positions are otherwise refined from mesh regions:
    root necks bound crypt-side regions that are excluded from the villus body,
    and split branches are placed at the centroid of the parent region after
    subtracting daughter crypt-side regions.
    """
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    detections = normalize_crypt_detections(crypt_detections)

    graph = SkeletonGraph(
        metadata=_json_safe_metadata(metadata),
        coordinate_frame={
            "kind": "raw",
            "body_center_node": "body",
            "description": "Raw mesh/world coordinates; edges are straight segments.",
        },
    )
    body_position = _body_center(vertices, faces, body_vertices, body_faces, body_center)
    body_refined = False
    if body_center is None and body_vertices is None and body_faces is None and refine_body_center_from_necks:
        refined_body = _body_center_from_root_regions(vertices, detections)
        if refined_body is not None:
            body_position = refined_body
            body_refined = True

    graph.add_node(
        "body",
        "body",
        body_position,
        metadata={
            "role": "villus_body_center",
            "center_refined_from_neck_regions": body_refined,
            "center_source": (
                "explicit_override"
                if body_center is not None
                else "neck_bounded_region"
                if body_refined
                else "mesh_centroid"
            ),
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
            branch_region = np.empty(0, dtype=np.int64)
            branch = _branch_center_override(
                branch_center_overrides,
                branch_node_id=branch_id,
                crypt_id=crypt_id,
            )
            branch_center_overridden = branch is not None
            if branch is None and refine_branch_centers_from_necks:
                branch, branch_region = _branch_position_from_regions(vertices, detection, daughters)
            if branch is None:
                branch = _branch_position(
                    vertices,
                    detection,
                    root_source_position,
                    daughter_tips,
                )
            graph.add_node(
                branch_id,
                "branch",
                branch,
                crypt_id=crypt_id,
                metadata={
                    **common_meta,
                    "n_daughters": len(daughters),
                    "center_refined_from_neck_regions": bool(branch_region.size),
                    "center_source": (
                        "explicit_override"
                        if branch_center_overridden
                        else "neck_bounded_region"
                        if branch_region.size
                        else "detection_or_geometric_fallback"
                    ),
                    "n_branch_region_vertices": int(branch_region.size),
                    "branch_region_vertices": branch_region.tolist(),
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
                    bend_strategy=bend_strategy,
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
            bend_strategy=bend_strategy,
            bend_max_dimensionless_curvature=bend_max_dimensionless_curvature,
            bend_curvature_penalty=bend_curvature_penalty,
        )

    return graph

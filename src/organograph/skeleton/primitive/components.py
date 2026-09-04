"""Component extraction for primitive fitting."""

from __future__ import annotations

from typing import Any

import numpy as np

from organograph.skeleton.datatypes import SkeletonGraph
from organograph.skeleton.geometry import as_points
from organograph.skeleton.primitive.common import _coerce_indices, _first_detection_value, _region_boundary_vertices

def component_region_from_detection(
    detection: dict[str, Any],
    n_vertices: int,
    *,
    region_keys: tuple[str, ...] = ("crypt_vertices",),
) -> np.ndarray:
    """Return the host-trimmed HKS component used by definitive fitting."""
    region = _coerce_indices(_first_detection_value(detection, region_keys))
    if region.size:
        return np.unique(region)
    legacy_region = _coerce_indices(
        _first_detection_value(
            detection,
            (
                "attachment_region_vertices",
                "neck_region_vertices",
                "neck_side_vertices",
                "root_region_vertices",
            ),
        )
    )
    if legacy_region.size:
        return np.unique(legacy_region)

    dfield = _first_detection_value(
        detection,
        ("d_crypt", "distance_field", "dnorm", "dnorm_vertices"),
    )
    if dfield is not None:
        dfield = np.asarray(dfield, dtype=float).reshape(-1)
        if dfield.size == int(n_vertices):
            level = float(
                detection.get(
                    "attachment_level",
                    detection.get("neck_level", 1.0),
                )
            )
            region = np.where(np.isfinite(dfield) & (dfield <= level))[0].astype(np.int64)
            if region.size:
                return region

    return np.unique(
        _coerce_indices(
            _first_detection_value(
                detection,
                ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
            )
        )
    )

def primitive_components_from_crypt_detections(
    vertices,
    crypt_detections: list[dict[str, Any]],
    graph: SkeletonGraph | None = None,
    faces=None,
) -> dict[str, Any]:
    """Build neck-cut body, branch, and crypt component vertex sets.

    Body vertices are the mesh vertices left after detaching every root
    appendage at its body-side neckline.  Branch vertices are parent
    neck-side regions with daughter crypt-side regions removed.  Crypt tube
    components are the terminal regions after their final neckline.
    """
    vertices = as_points(vertices)
    n_vertices = int(vertices.shape[0])
    all_vertices = set(range(n_vertices))
    body_excluded: set[int] = set()
    branches: dict[str, list[int]] = {}
    crypts: dict[Any, list[int]] = {}
    crypt_centerlines: dict[Any, dict[str, Any]] = {}
    body_branch_necks: dict[str, dict[str, Any]] = {}

    for detection in crypt_detections:
        crypt_id = detection.get("crypt_id")
        daughters = detection.get("daughters") or []
        parent_region = component_region_from_detection(detection, n_vertices)

        if daughters:
            if parent_region.size:
                body_excluded.update(map(int, parent_region.tolist()))

            daughter_regions = []
            for j, daughter in enumerate(daughters):
                daughter_region = component_region_from_detection(daughter, n_vertices)
                daughter_regions.append(daughter_region)
                if daughter_region.size:
                    tip_node_id = f"crypt_{crypt_id}_tip_{j}"
                    crypts[tip_node_id] = sorted(map(int, daughter_region.tolist()))
                    crypt_centerlines[tip_node_id] = {
                        "vertex_indices": crypts[tip_node_id],
                        "boundary_tip_vertex_id": daughter.get(
                            "boundary_distance_bottom_vertex_id"
                        ),
                        "hks_tip_vertex_id": daughter.get("bottom_vertex_id"),
                        "distance_field": _first_detection_value(
                            daughter,
                            ("d_crypt", "distance_field", "dnorm", "dnorm_vertices"),
                        ),
                        "neck_level": float(
                            daughter.get(
                                "attachment_level",
                                daughter.get("neck_level", 1.0),
                            )
                        ),
                        "neck_profile": daughter.get("neck_profile"),
                        "attachment_surface_normal": daughter.get(
                            "attachment_surface_normal"
                        ),
                        "attachment_normal_source": daughter.get(
                            "attachment_normal_source"
                        ),
                        "candidate_boundary_vertices": daughter.get(
                            "candidate_boundary_vertices"
                        ),
                        "constriction_position": daughter.get(
                            "constriction_position"
                        ),
                    }

            remove = set()
            for daughter_region in daughter_regions:
                remove.update(map(int, daughter_region.tolist()))
            branch_region = [int(v) for v in parent_region.tolist() if int(v) not in remove]
            if not branch_region:
                stem = _coerce_indices(
                    _first_detection_value(detection, ("stem_vertices", "trunk_vertices"))
                )
                branch_region = sorted(map(int, stem.tolist()))

            branch_node_id = f"crypt_{crypt_id}_branch"
            if branch_region and (graph is None or branch_node_id in graph.nodes):
                branches[branch_node_id] = sorted(set(branch_region))
            neck_node_id = f"crypt_{crypt_id}_neck"
            if (
                faces is not None
                and graph is not None
                and neck_node_id in graph.nodes
                and branch_node_id in graph.nodes
            ):
                boundary_vertices = _region_boundary_vertices(
                    faces,
                    parent_region,
                    n_vertices,
                )
                if boundary_vertices.size:
                    attachment_id = f"{neck_node_id}_cylinder"
                    body_branch_necks[attachment_id] = {
                        "neck_node_id": neck_node_id,
                        "body_node_id": graph.body_node().node_id,
                        "branch_node_id": branch_node_id,
                        "boundary_vertices": boundary_vertices.tolist(),
                    }
            continue

        region = component_region_from_detection(detection, n_vertices)
        if region.size:
            body_excluded.update(map(int, region.tolist()))
            crypts[crypt_id] = sorted(map(int, region.tolist()))
            crypt_centerlines[crypt_id] = {
                "vertex_indices": crypts[crypt_id],
                "boundary_tip_vertex_id": detection.get(
                    "boundary_distance_bottom_vertex_id"
                ),
                "hks_tip_vertex_id": detection.get("bottom_vertex_id"),
                "distance_field": _first_detection_value(
                    detection,
                    ("d_crypt", "distance_field", "dnorm", "dnorm_vertices"),
                ),
                "neck_level": float(
                    detection.get(
                        "attachment_level",
                        detection.get("neck_level", 1.0),
                    )
                ),
                "neck_profile": detection.get("neck_profile"),
                "attachment_surface_normal": detection.get(
                    "attachment_surface_normal"
                ),
                "attachment_normal_source": detection.get(
                    "attachment_normal_source"
                ),
                "candidate_boundary_vertices": detection.get(
                    "candidate_boundary_vertices"
                ),
                "constriction_position": detection.get("constriction_position"),
            }

    body = sorted(all_vertices.difference(body_excluded))
    if len(body) < 3:
        body = sorted(all_vertices)

    return {
        "body": body,
        "branches": branches,
        "crypts": crypts,
        "crypt_centerlines": crypt_centerlines,
        "body_branch_necks": body_branch_necks,
        "metadata": {
            "n_body_vertices": len(body),
            "n_body_excluded_vertices": len(body_excluded),
            "component_source": "neck_cut_crypt_detections",
        },
    }

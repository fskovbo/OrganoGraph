"""Post-fit overlap validation and topology merging for crypt tubes."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import qmc

from organograph.skeleton.config import CryptOverlapConfig
from organograph.skeleton.primitive_geometry import (
    polyline_lengths,
    project_points_to_polyline,
    tube_radius_from_parameters,
)


@dataclass(frozen=True)
class TerminalCryptReference:
    """Link one fitted terminal tube back to its detection hierarchy entry."""

    attachment_id: str
    tip_node_id: str
    host_id: str
    top_index: int
    daughter_index: int | None
    component_key: Any
    support_size: int
    path_length: float
    proximal_node_id: str | None = None


@dataclass
class CryptOverlapAssessment:
    """Pairwise overlap measurements and connected merge groups."""

    pairs: list[dict[str, Any]] = field(default_factory=list)
    groups: list[list[TerminalCryptReference]] = field(default_factory=list)
    threshold: float = 0.30

    @property
    def requires_merge(self) -> bool:
        return bool(self.groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "requires_merge": self.requires_merge,
            "pairs": list(self.pairs),
            "groups": [
                [reference.attachment_id for reference in group]
                for group in self.groups
            ],
        }


@dataclass
class CryptDetectionMergeResult:
    """Detection hierarchy after applying primitive-overlap groups."""

    detections: list[dict[str, Any]]
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.records)


def _tube_radius(parameters, s) -> np.ndarray:
    return tube_radius_from_parameters(parameters, np.asarray(s, dtype=float))


def _tube_volume(parameters, *, samples: int = 512) -> float:
    centerline = np.asarray(parameters["centerline_points"], dtype=float)
    _, _, length = polyline_lengths(centerline)
    if length <= 1e-12:
        return 0.0
    s = np.linspace(0.0, 1.0, max(16, int(samples)))
    radius = _tube_radius(parameters, s)
    return float(length * np.trapezoid(np.pi * radius**2, s))


def _tube_bounds(parameters) -> tuple[np.ndarray, np.ndarray]:
    centerline = np.asarray(parameters["centerline_points"], dtype=float)
    s = np.linspace(0.0, 1.0, 256)
    radius = float(np.max(_tube_radius(parameters, s)))
    return np.min(centerline, axis=0) - radius, np.max(centerline, axis=0) + radius


def _points_inside_tube(points, parameters) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    centerline = np.asarray(parameters["centerline_points"], dtype=float)
    projection = project_points_to_polyline(points, centerline)
    radius = _tube_radius(parameters, projection["s"])
    inside = projection["distances"] <= radius + 1e-10

    start_tangent = centerline[1] - centerline[0]
    end_tangent = centerline[-1] - centerline[-2]
    start_norm = float(np.linalg.norm(start_tangent))
    end_norm = float(np.linalg.norm(end_tangent))
    if start_norm > 1e-12:
        inside &= (points - centerline[0]) @ (start_tangent / start_norm) >= -1e-10
    if end_norm > 1e-12:
        inside &= (points - centerline[-1]) @ (end_tangent / end_norm) <= 1e-10
    return inside


def tube_overlap_fraction(
    first,
    second,
    *,
    samples: int = 32768,
    random_seed: int | None = 0,
) -> dict[str, float]:
    """Estimate intersection volume relative to the smaller tube volume."""
    first_parameters = first.parameters if hasattr(first, "parameters") else first
    second_parameters = second.parameters if hasattr(second, "parameters") else second
    first_volume = _tube_volume(first_parameters)
    second_volume = _tube_volume(second_parameters)
    first_lo, first_hi = _tube_bounds(first_parameters)
    second_lo, second_hi = _tube_bounds(second_parameters)
    lower = np.maximum(first_lo, second_lo)
    upper = np.minimum(first_hi, second_hi)
    span = upper - lower
    if np.any(span <= 0.0) or min(first_volume, second_volume) <= 1e-12:
        intersection = 0.0
    else:
        count = max(256, int(samples))
        exponent = int(np.ceil(np.log2(count)))
        unit = qmc.Sobol(d=3, scramble=True, seed=random_seed).random_base2(exponent)
        points = lower + unit[:count] * span
        intersection = float(
            np.prod(span)
            * np.mean(
                _points_inside_tube(points, first_parameters)
                & _points_inside_tube(points, second_parameters)
            )
        )
    fraction = float(
        np.clip(intersection / max(min(first_volume, second_volume), 1e-12), 0.0, 1.0)
    )
    return {
        "fraction_of_smaller": fraction,
        "intersection_volume": intersection,
        "first_volume": first_volume,
        "second_volume": second_volume,
    }


def _terminal_references(primitive_result) -> list[TerminalCryptReference]:
    graph = primitive_result.graph
    attachments = dict(primitive_result.attachments.get("crypts", {}))
    by_tip = {
        str(attachment.target_ids[-1]): (str(attachment_id), attachment)
        for attachment_id, attachment in attachments.items()
        if attachment.target_ids
    }
    components = primitive_result.components.get("crypts", {})
    references = []
    for top_index, detection in enumerate(primitive_result.skeleton.detections):
        crypt_id = detection.get("crypt_id", top_index)
        daughters = detection.get("daughters") or []
        entries = (
            [
                (daughter, daughter_index, f"crypt_{crypt_id}_tip_{daughter_index}")
                for daughter_index, daughter in enumerate(daughters)
            ]
            if daughters
            else [(detection, None, f"crypt_{crypt_id}_tip")]
        )
        host_id = f"crypt_{crypt_id}_branch" if daughters else "body"
        for terminal, daughter_index, tip_node_id in entries:
            matched = by_tip.get(tip_node_id)
            if matched is None:
                continue
            attachment_id, attachment = matched
            component_key = tip_node_id if daughters else crypt_id
            support = components.get(component_key, terminal.get("crypt_vertices", []))
            references.append(
                TerminalCryptReference(
                    attachment_id=attachment_id,
                    tip_node_id=tip_node_id,
                    host_id=host_id,
                    top_index=top_index,
                    daughter_index=daughter_index,
                    component_key=component_key,
                    support_size=int(_indices(support).size),
                    path_length=float(attachment.derived_parameters.get("length", 0.0)),
                    proximal_node_id=(
                        str(attachment.target_ids[0])
                        if attachment.target_ids
                        else None
                    ),
                )
            )
    return references


def assess_crypt_primitive_overlaps(
    primitive_result,
    config: CryptOverlapConfig,
) -> CryptOverlapAssessment:
    """Measure same-host terminal tube overlaps and build merge components."""
    graph = primitive_result.graph
    references = _terminal_references(primitive_result)
    attachments = dict(primitive_result.attachments.get("crypts", {}))
    threshold = float(np.clip(config.threshold, 0.0, 1.0))
    adjacency = {reference.attachment_id: set() for reference in references}
    pairs = []
    pair_index = 0

    def host_attachment_angle(first, second):
        if (
            first.host_id not in graph.nodes
            or first.proximal_node_id not in graph.nodes
            or second.proximal_node_id not in graph.nodes
        ):
            return None
        host = graph.node(first.host_id).position
        first_direction = graph.node(first.proximal_node_id).position - host
        second_direction = graph.node(second.proximal_node_id).position - host
        first_norm = float(np.linalg.norm(first_direction))
        second_norm = float(np.linalg.norm(second_direction))
        if first_norm <= 1e-12 or second_norm <= 1e-12:
            return None
        cosine = float(
            np.clip(
                np.dot(first_direction, second_direction) / (first_norm * second_norm),
                -1.0,
                1.0,
            )
        )
        return float(np.arccos(cosine))

    for i, first in enumerate(references):
        for second in references[i + 1 :]:
            if first.host_id != second.host_id:
                continue
            angle = host_attachment_angle(first, second)
            max_angle = config.max_host_attachment_angle
            if angle is not None and max_angle is not None and angle > float(max_angle):
                pairs.append(
                    {
                        "first_attachment_id": first.attachment_id,
                        "second_attachment_id": second.attachment_id,
                        "host_id": first.host_id,
                        "attachment_angle": angle,
                        "skipped": True,
                        "skip_reason": "host_attachment_angle_exceeds_limit",
                        "merge": False,
                    }
                )
                pair_index += 1
                continue
            overlap = tube_overlap_fraction(
                attachments[first.attachment_id],
                attachments[second.attachment_id],
                samples=config.samples,
                random_seed=(
                    None
                    if config.random_seed is None
                    else int(config.random_seed) + pair_index
                ),
            )
            pair_index += 1
            merge = overlap["fraction_of_smaller"] >= threshold
            pairs.append(
                {
                    "first_attachment_id": first.attachment_id,
                    "second_attachment_id": second.attachment_id,
                    "host_id": first.host_id,
                    "attachment_angle": angle,
                    "skipped": False,
                    "merge": bool(merge),
                    **overlap,
                }
            )
            if merge:
                adjacency[first.attachment_id].add(second.attachment_id)
                adjacency[second.attachment_id].add(first.attachment_id)

    by_id = {reference.attachment_id: reference for reference in references}
    visited = set()
    groups = []
    for attachment_id in adjacency:
        if attachment_id in visited or not adjacency[attachment_id]:
            continue
        stack = [attachment_id]
        component = []
        visited.add(attachment_id)
        while stack:
            current = stack.pop()
            component.append(by_id[current])
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(component) > 1:
            groups.append(sorted(component, key=lambda item: item.attachment_id))
    return CryptOverlapAssessment(pairs=pairs, groups=groups, threshold=threshold)


_COMPONENT_REGION_KEYS = (
    "crypt_vertices",
    "attachment_region_vertices",
    "root_region_vertices",
    "candidate_crypt_vertices",
)


def _indices(value) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.int64)
    if isinstance(value, set):
        value = list(value)
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _merge_records(records, references, pair_records):
    ranked = sorted(
        zip(records, references),
        key=lambda item: (
            -item[1].support_size,
            -item[1].path_length,
            item[1].attachment_id,
        ),
    )
    representative, representative_reference = ranked[0]
    merged = copy.deepcopy(representative)
    for key in _COMPONENT_REGION_KEYS:
        values = [_indices(record.get(key)) for record in records if record.get(key) is not None]
        if values:
            merged[key] = np.unique(np.concatenate(values)).astype(np.int64)
    original_attachment_ids = []
    original_crypt_ids = []
    previous_pair_overlaps = []
    for record, reference in zip(records, references):
        previous = dict(record.get("metadata") or {}).get(
            "crypt_primitive_overlap_merge",
            {},
        )
        original_attachment_ids.extend(
            previous.get("merged_attachment_ids", [reference.attachment_id])
        )
        original_crypt_ids.extend(
            previous.get(
                "original_crypt_ids",
                [record.get("crypt_id")] if record.get("crypt_id") is not None else [],
            )
        )
        previous_pair_overlaps.extend(previous.get("pair_overlaps", []))

    def unique(values):
        return list(dict.fromkeys(values))

    metadata = dict(merged.get("metadata") or {})
    metadata["crypt_primitive_overlap_merge"] = {
        "representative_attachment_id": representative_reference.attachment_id,
        "merged_attachment_ids": unique(original_attachment_ids),
        "original_crypt_ids": unique(original_crypt_ids),
        "pair_overlaps": previous_pair_overlaps + pair_records,
    }
    merged["metadata"] = metadata
    return merged, representative_reference


def _distance_level_at_position(vertices, distance_field, position, fallback):
    field = np.asarray(distance_field, dtype=float).reshape(-1)
    point = np.asarray(position, dtype=float) if position is not None else None
    if (
        point is None
        or point.shape != (3,)
        or not np.all(np.isfinite(point))
        or field.size != vertices.shape[0]
        or not np.any(np.isfinite(field))
    ):
        return float(fallback)
    nearest = int(np.argmin(np.linalg.norm(vertices - point[None, :], axis=1)))
    value = float(field[nearest])
    return value if np.isfinite(value) else float(fallback)


def _collapse_single_daughter(parent, daughter, vertices):
    promoted = copy.deepcopy(daughter)
    parent_region = np.unique(
        np.concatenate(
            [_indices(parent.get(key)) for key in _COMPONENT_REGION_KEYS]
            + [_indices(daughter.get("crypt_vertices"))]
        )
    ).astype(np.int64)
    parent_attachment = parent.get("attachment_position", parent.get("neck_position"))
    daughter_attachment = daughter.get("attachment_position", daughter.get("neck_position"))
    promoted["crypt_id"] = parent.get("crypt_id")
    for key in _COMPONENT_REGION_KEYS[:3]:
        promoted[key] = parent_region
    if parent_attachment is None:
        parent_attachment = daughter_attachment
    if parent_attachment is not None:
        promoted["attachment_position"] = np.asarray(parent_attachment, dtype=float)
    field = promoted.get("d_crypt")
    attachment_level = _distance_level_at_position(
        vertices,
        field,
        parent_attachment,
        parent.get("attachment_level", 1.0),
    )
    promoted["attachment_level"] = attachment_level
    if daughter_attachment is not None:
        constriction_level = _distance_level_at_position(
            vertices,
            field,
            daughter_attachment,
            daughter.get("attachment_level", 0.5 * attachment_level),
        )
    else:
        constriction_level = attachment_level
    if (
        daughter_attachment is not None
        and constriction_level < attachment_level - 1e-6
    ):
        promoted["constriction_position"] = np.asarray(daughter_attachment, dtype=float)
        promoted["neck_position"] = np.asarray(daughter_attachment, dtype=float)
        promoted["neck_profile"] = {
            "kind": "constriction",
            "attachment_level": attachment_level,
            "constriction_level": constriction_level,
            "reason": "collapsed_overlapping_split_daughters",
        }
    else:
        if parent_attachment is not None:
            promoted["neck_position"] = np.asarray(parent_attachment, dtype=float)
        promoted["neck_profile"] = {
            "kind": "transition",
            "attachment_level": attachment_level,
            "reason": "collapsed_overlapping_split_daughters",
        }
        promoted.pop("constriction_position", None)
    for key in ("daughters", "stem_vertices", "branch_position", "branch_center"):
        promoted.pop(key, None)
    metadata = dict(promoted.get("metadata") or {})
    metadata["collapsed_split_parent_crypt_id"] = parent.get("crypt_id")
    promoted["metadata"] = metadata
    return promoted


def merge_overlapping_crypt_detections(
    detections,
    assessment: CryptOverlapAssessment,
    vertices,
) -> CryptDetectionMergeResult:
    """Merge overlap groups in detection hierarchy and collapse one-child branches."""
    out = copy.deepcopy(detections)
    vertices = np.asarray(vertices, dtype=float)
    pair_lookup = {
        frozenset((pair["first_attachment_id"], pair["second_attachment_id"])): pair
        for pair in assessment.pairs
    }
    records = []

    top_groups = [group for group in assessment.groups if group[0].host_id == "body"]
    remove_top = set()
    replace_top = {}
    for group in top_groups:
        group_records = [out[reference.top_index] for reference in group]
        pair_records = [
            pair_lookup[key]
            for i, first in enumerate(group)
            for second in group[i + 1 :]
            if (key := frozenset((first.attachment_id, second.attachment_id))) in pair_lookup
        ]
        merged, representative = _merge_records(group_records, group, pair_records)
        replace_top[representative.top_index] = merged
        remove_top.update(
            reference.top_index
            for reference in group
            if reference.top_index != representative.top_index
        )
        records.append(
            {
                "host_id": "body",
                "representative_attachment_id": representative.attachment_id,
                "merged_attachment_ids": [reference.attachment_id for reference in group],
                "max_overlap_fraction": max(
                    (pair["fraction_of_smaller"] for pair in pair_records),
                    default=0.0,
                ),
                "collapsed_branch": False,
            }
        )

    daughter_groups = [group for group in assessment.groups if group[0].host_id != "body"]
    groups_by_parent: dict[int, list[list[TerminalCryptReference]]] = {}
    for group in daughter_groups:
        parent_indices = {reference.top_index for reference in group}
        if len(parent_indices) != 1:
            raise ValueError("A crypt-overlap merge group must have exactly one host branch")
        groups_by_parent.setdefault(parent_indices.pop(), []).append(group)

    affected_parents = set()
    for parent_index, parent_groups in groups_by_parent.items():
        original_daughters = list(out[parent_index].get("daughters") or [])
        replacements = {}
        removed = set()
        parent_records = []
        for group in parent_groups:
            daughter_indices = [int(reference.daughter_index) for reference in group]
            group_records = [original_daughters[index] for index in daughter_indices]
            pair_records = [
                pair_lookup[key]
                for i, first in enumerate(group)
                for second in group[i + 1 :]
                if (key := frozenset((first.attachment_id, second.attachment_id)))
                in pair_lookup
            ]
            merged, representative = _merge_records(group_records, group, pair_records)
            representative_index = int(representative.daughter_index)
            replacements[representative_index] = merged
            removed.update(index for index in daughter_indices if index != representative_index)
            parent_records.append(
                {
                    "host_id": group[0].host_id,
                    "representative_attachment_id": representative.attachment_id,
                    "merged_attachment_ids": [
                        reference.attachment_id for reference in group
                    ],
                    "max_overlap_fraction": max(
                        (pair["fraction_of_smaller"] for pair in pair_records),
                        default=0.0,
                    ),
                }
            )

        out[parent_index]["daughters"] = [
            replacements.get(index, daughter)
            for index, daughter in enumerate(original_daughters)
            if index not in removed
        ]
        affected_parents.add(parent_index)
        collapsed = len(out[parent_index]["daughters"]) == 1
        for record in parent_records:
            records.append({**record, "collapsed_branch": collapsed})

    for top_index, detection in enumerate(out):
        if top_index in replace_top:
            out[top_index] = replace_top[top_index]
            detection = out[top_index]
        daughters = detection.get("daughters") or []
        if top_index in affected_parents and len(daughters) == 1:
            out[top_index] = _collapse_single_daughter(detection, daughters[0], vertices)

    out = [detection for index, detection in enumerate(out) if index not in remove_top]
    return CryptDetectionMergeResult(detections=out, records=records)


_STALE_GEOMETRY_KEYS = (
    "boundary_distance_bottom_vertex_id",
    "bottom_vertex_id",
    "tip_position",
    "tip_center",
    "bottom_position",
    "crypt_tip",
    "p_tip",
    "crypt_position",
    "crypt_center",
    "crypt_centroid",
    "p_crypt",
    "d_crypt",
    "L_crypt",
    "neck_profile",
    "circumference_levels",
    "circumference",
    "attachment_level",
    "attachment_position",
    "attachment_surface_normal",
    "attachment_normal_source",
    "candidate_boundary_vertices",
    "neck_position",
    "constriction_position",
    "distal_neck_boundary_position",
    "neck_region_vertices",
    "neck_vertices",
    "attachment_region_vertices",
    "root_region_vertices",
)


def recompute_merged_crypt_geometry(
    mesh,
    detections,
    *,
    detection_config,
    barriers,
    diagnostics: dict[str, Any] | None = None,
    geodesic_fn=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute a merged terminal crypt from its union mesh region.

    A merge initially retains one record only to provide a stable crypt ID.
    This function discards that record's positional geometry and reruns the
    definitive boundary-distance tip, circumference normalization, HKS tip,
    neck-profile, and projected host-opening stages on the union region.
    """
    from organograph.mesh.geodesics import compute_geodesics_dijkstra
    from organograph.crypts.axis import (
        _get_boundary_vertices,
        normalize_crypt_axis_to_neckline,
    )
    from organograph.skeleton.detection.attachments import (
        assign_crypt_attachments,
        attachment_projection_diagnostics,
    )
    from organograph.skeleton.detection.neck_profiles import (
        _add_neck_profile_geometry,
    )
    from organograph.skeleton.detection.pipeline import (
        _profile_crypt_patches,
        _profile_search_interval,
    )

    if barriers is None:
        raise ValueError("Barrier fits are required to recompute merged crypt geometry")
    diagnostics = {} if diagnostics is None else diagnostics
    geodesic_fn = geodesic_fn or compute_geodesics_dijkstra
    candidate_config = detection_config.candidates
    neck_config = detection_config.necks
    barrier_config = detection_config.barriers
    profile_search_interval = _profile_search_interval(
        neck_config,
        barrier_config,
    )
    levels = np.linspace(
        0.01,
        float(neck_config.max_axis_level),
        int(neck_config.resolution),
    )
    hks = diagnostics.get("hks")
    hks_times = diagnostics.get("ts_mesh")
    records = []

    def axis_from_final_tip(patch, tip_vertex_id):
        distances = np.asarray(
            geodesic_fn(
                mesh,
                sources=[int(tip_vertex_id)],
                **dict(candidate_config.geodesic_kwargs),
            ),
            dtype=float,
        )
        if distances.ndim == 2:
            distances = distances[0]
        distances = distances.reshape(-1)
        if distances.size != mesh.v.shape[0]:
            raise ValueError("geodesic function returned an invalid distance field")
        boundary = _get_boundary_vertices(mesh, patch_vertices=patch)
        boundary_distances = distances[boundary]
        boundary_distances = boundary_distances[np.isfinite(boundary_distances)]
        if boundary_distances.size == 0:
            raise ValueError("merged crypt region has no finite boundary distances")
        length = max(float(np.mean(boundary_distances)), 1e-12)
        normalized = distances / length
        circumference, normalized, lengths = normalize_crypt_axis_to_neckline(
            mesh,
            normalized,
            levels,
            search_interval=profile_search_interval,
            L_crypt=length,
            window_length=9,
            polyorder=3,
            min_prominence=neck_config.min_prominence,
        )
        return circumference[0], normalized[0], float(lengths[0])

    def refresh(detection, *, relation: str):
        current = copy.deepcopy(detection)
        metadata = dict(current.get("metadata") or {})
        merge_info = metadata.get("crypt_primitive_overlap_merge")
        if isinstance(merge_info, dict):
            merged_ids = list(merge_info.get("merged_attachment_ids", []))
            previous = metadata.get("merged_geometry_recomputation", {})
            if previous.get("merged_attachment_ids") != merged_ids:
                patch = np.unique(_indices(current.get("crypt_vertices")))
                record = {
                    "crypt_id": current.get("crypt_id"),
                    "relation": relation,
                    "merged_attachment_ids": merged_ids,
                    "n_union_vertices": int(patch.size),
                    "success": False,
                }
                if patch.size < 3:
                    record["reason"] = "insufficient_union_vertices"
                else:
                    try:
                        profile = _profile_crypt_patches(
                            mesh,
                            [set(map(int, patch.tolist()))],
                            geodesic_fn=geodesic_fn,
                            geodesic_kwargs=dict(candidate_config.geodesic_kwargs),
                            d_levels=levels,
                            neck_config=neck_config,
                            candidate_config=candidate_config,
                            hks=hks,
                            hks_times=hks_times,
                            search_interval=profile_search_interval,
                        )
                        final_tip = int(profile["final_tips"][0])
                        circumference, distance_field, crypt_length = (
                            axis_from_final_tip(patch, final_tip)
                        )
                        for key in _STALE_GEOMETRY_KEYS:
                            current.pop(key, None)
                        current["crypt_vertices"] = patch
                        current["candidate_crypt_vertices"] = patch
                        current["boundary_distance_bottom_vertex_id"] = int(
                            profile["initial_tips"][0]
                        )
                        current["bottom_vertex_id"] = final_tip
                        current["d_crypt"] = distance_field
                        current["L_crypt"] = crypt_length
                        metadata = dict(current.get("metadata") or {})
                        metadata.update(
                            {
                                "boundary_distance_bottom_selection": profile[
                                    "initial_tip_info"
                                ][0],
                                "final_tip_selection": profile["final_tip_info"][0],
                            }
                        )
                        current["metadata"] = metadata
                        current = _add_neck_profile_geometry(
                            mesh.v,
                            mesh.f,
                            current,
                            levels,
                            circumference,
                            relation=relation,
                            window_length=9,
                            polyorder=3,
                            min_prominence=neck_config.min_prominence,
                            min_neck_length=neck_config.min_length,
                        )
                        record.update(
                            {
                                "success": True,
                                "reason": "recomputed_from_union_region",
                                "boundary_distance_tip_vertex_id": int(
                                    profile["initial_tips"][0]
                                ),
                                "final_tip_vertex_id": int(profile["final_tips"][0]),
                                "distance_field_source": "final_hks_tip",
                            }
                        )
                    except (RuntimeError, TypeError, ValueError) as exc:
                        record["reason"] = f"recomputation_failed: {exc}"
                metadata = dict(current.get("metadata") or {})
                metadata["merged_geometry_recomputation"] = dict(record)
                current["metadata"] = metadata
                records.append(record)

        daughters = current.get("daughters") or []
        if daughters:
            current["daughters"] = [
                refresh(daughter, relation="branch_crypt") for daughter in daughters
            ]
        return current

    refreshed = [refresh(detection, relation="body_crypt") for detection in detections]
    refreshed = assign_crypt_attachments(
        mesh.v,
        mesh.f,
        refreshed,
        barriers.body_fit,
        grid_resolution=detection_config.barriers.opening_grid_resolution,
        boundary_refinement_max_mesh_fraction=(
            detection_config.barriers.boundary_refinement_max_mesh_fraction
        ),
        strategy=detection_config.barriers.attachment_strategy,
        assign_body_roots=True,
        assign_branch_daughters=False,
    )
    refreshed = assign_crypt_attachments(
        mesh.v,
        mesh.f,
        refreshed,
        barriers.body_fit,
        branch_fits=barriers.branch_fits,
        grid_resolution=detection_config.barriers.opening_grid_resolution,
        boundary_refinement_max_mesh_fraction=(
            detection_config.barriers.boundary_refinement_max_mesh_fraction
        ),
        strategy=detection_config.barriers.attachment_strategy,
        assign_body_roots=False,
        assign_branch_daughters=True,
    )
    projection_records = attachment_projection_diagnostics(refreshed)
    diagnostics["attachment_projections"] = projection_records
    diagnostics["attachment_projection_failures"] = [
        record for record in projection_records if not record.get("found", False)
    ]
    return refreshed, records


__all__ = [
    "CryptDetectionMergeResult",
    "CryptOverlapAssessment",
    "TerminalCryptReference",
    "assess_crypt_primitive_overlaps",
    "merge_overlapping_crypt_detections",
    "recompute_merged_crypt_geometry",
    "tube_overlap_fraction",
]

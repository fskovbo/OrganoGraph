"""Create visualization-only crypt attachment extensions."""

from __future__ import annotations

from typing import Any

import numpy as np

from organograph.skeleton.blending.base import BlendAttachment
from organograph.skeleton.blending.geometry import (
    local_blob_radius_at_point,
    sample_quadratic_through_midpoint,
    unit_vector,
)
from organograph.skeleton.config import BlendConfig
from organograph.skeleton.primitive_geometry import tube_radius_from_parameters


def _incoming_edges(graph, node_id: str):
    return [edge for edge in graph.edges.values() if edge.target == str(node_id)]


def _find_host_node_id(graph, node_id: str) -> str | None:
    """Walk upstream from a crypt path start until a body or branch is found."""
    seen = set()
    stack = [str(node_id)]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        node = graph.node(current)
        if node.node_type in {"body", "branch"}:
            return current
        for edge in _incoming_edges(graph, current):
            stack.append(edge.source)
    return None


def _host_primitive_attachment(graph, host_node_id: str):
    node = graph.node(host_node_id)
    if node.primitive_attachment is not None:
        return node.primitive_attachment
    if host_node_id == graph.body_node().node_id:
        return graph.body_node().primitive_attachment
    return None


def _host_local_radius(host_attachment, point, axis_direction, *, fallback: float) -> float:
    if host_attachment is None:
        return float(fallback)
    return local_blob_radius_at_point(
        host_attachment,
        point,
        axis_direction,
        default_radius=fallback,
    )


def _crypt_start_radius(tube_attachment) -> float:
    radius = float(
        tube_radius_from_parameters(
            tube_attachment.parameters, np.asarray([0.0], dtype=float)
        )[0]
    )
    return max(radius, 1e-8)


def _create_crypt_extension_blends(graph, config: BlendConfig) -> dict[str, BlendAttachment]:
    blends: dict[str, BlendAttachment] = {}
    for tube_id, tube in graph.primitive_attachments.items():
        if tube.primitive_type != "tapered_capped_tube":
            continue
        path = [str(node_id) for node_id in tube.target_ids]
        if len(path) < 2:
            continue
        host_id = _find_host_node_id(graph, path[0])
        if host_id is None:
            continue
        host_attachment = _host_primitive_attachment(graph, host_id)

        tube_centerline = np.asarray(tube.parameters["centerline_points"], dtype=float)
        if tube_centerline.ndim != 2 or tube_centerline.shape[0] < 2:
            continue

        attachment = graph.node(path[0]).position
        next_node = graph.node(path[1]).position
        crypt_axis = tube_centerline[1] - tube_centerline[0]
        crypt_distance = float(np.linalg.norm(crypt_axis))
        if crypt_distance <= 1e-12:
            crypt_axis = next_node - attachment
            crypt_distance = float(np.linalg.norm(crypt_axis))
        if crypt_distance <= 1e-12:
            continue
        crypt_direction = unit_vector(crypt_axis)
        host_distance = float(np.linalg.norm(attachment - graph.node(host_id).position))
        length = float(config.extension_length_fraction) * host_distance
        if length <= 1e-12:
            continue
        r_crypt = _crypt_start_radius(tube)
        host_end = attachment - length * crypt_direction
        r_host = _host_local_radius(
            host_attachment,
            host_end,
            crypt_direction,
            fallback=r_crypt,
        )
        centerline = np.vstack([host_end, attachment])
        blend_id = f"blend_{tube_id}"
        blends[blend_id] = BlendAttachment(
            blend_type="tapered_attachment_extension_tube",
            attachment_id=blend_id,
            target_ids=[host_id, *path],
            parameters={
                "centerline_points": centerline,
                "r_host": r_host,
                "r_crypt": r_crypt,
                "radius_profile": "linear_host_local_to_attachment",
                "extension_length_fraction": float(config.extension_length_fraction),
                "crypt_tube_attachment_id": tube_id,
            },
            diagnostics={
                "length": length,
                "host_node_id": host_id,
                "host_primitive_type": (
                    None if host_attachment is None else host_attachment.primitive_type
                ),
                "attachment_node_id": path[0],
                "next_crypt_node_id": path[1],
                "attachment_radius": r_crypt,
                "host_radius": r_host,
                "host_radius_source": "endpoint_disk_expanded_to_host_primitive",
                "attachment_to_host_node_distance": host_distance,
                "centerline_tangent_sample_distance": crypt_distance,
                "direction": crypt_direction,
            },
            metadata={
                "stage": "visual_blending",
                "vae_parameter": False,
                "strategy": "tapered_crypt_centerline_tangent_extension",
                **dict(config.metadata),
            },
        )
    return blends


def _create_body_branch_neck_replacement_blends(graph, config: BlendConfig) -> dict[str, BlendAttachment]:
    blends: dict[str, BlendAttachment] = {}
    for primitive_id, primitive in graph.primitive_attachments.items():
        if primitive.primitive_type != "straight_cylinder":
            continue
        if primitive.metadata.get("component") != "body_branch_neck":
            continue
        targets = [str(target) for target in primitive.target_ids]
        if len(targets) < 3:
            continue
        body_id, neck_id, branch_id = targets[:3]
        if body_id not in graph.nodes or neck_id not in graph.nodes or branch_id not in graph.nodes:
            continue
        body = graph.node(body_id).position
        neck = graph.node(neck_id).position
        branch = graph.node(branch_id).position
        body_end = 0.5 * (body + neck)
        branch_end = 0.5 * (branch + neck)
        centerline = sample_quadratic_through_midpoint(
            body_end,
            neck,
            branch_end,
            n_samples=config.n_samples,
        )
        neck_radius = float(primitive.parameters.get("radius", 0.0))
        if not np.isfinite(neck_radius) or neck_radius <= 0.0:
            continue
        body_axis = unit_vector(neck - body)
        branch_axis = unit_vector(neck - branch)
        body_attachment = _host_primitive_attachment(graph, body_id)
        branch_attachment = _host_primitive_attachment(graph, branch_id)
        body_radius = _host_local_radius(
            body_attachment,
            body_end,
            body_axis,
            fallback=neck_radius,
        )
        branch_radius = _host_local_radius(
            branch_attachment,
            branch_end,
            branch_axis,
            fallback=neck_radius,
        )
        blend_id = f"blend_{primitive_id}"
        blends[blend_id] = BlendAttachment(
            blend_type="body_branch_neck_replacement_tube",
            attachment_id=blend_id,
            target_ids=[body_id, neck_id, branch_id],
            parameters={
                "centerline_points": centerline,
                "r_body": body_radius,
                "r_neck": neck_radius,
                "r_branch": branch_radius,
                "radius_profile": "linear_body_neck_branch",
                "replaced_primitive_attachment_id": primitive_id,
            },
            diagnostics={
                "body_radius": body_radius,
                "neck_radius": neck_radius,
                "branch_radius": branch_radius,
                "body_radius_source": "endpoint_disk_expanded_to_host_primitive",
                "branch_radius_source": "endpoint_disk_expanded_to_host_primitive",
                "body_end": body_end,
                "branch_end": branch_end,
            },
            metadata={
                "stage": "visual_blending",
                "vae_parameter": False,
                "strategy": "curved_body_branch_neck_replacement",
                "replaces_primitive_attachment_id": primitive_id,
                **dict(config.metadata),
            },
        )
    return blends


def create_attachment_blends(
    graph,
    *,
    vertices=None,
    config: BlendConfig | dict[str, Any] | None = None,
) -> dict[str, BlendAttachment]:
    """Create simple straight crypt-extension tubes.

    Blends are intentionally not graph primitives.  They should be recomputed
    after decoding or sampling VAE-facing skeleton/primitive parameters.

    Each fitted crypt tube is extended from its first skeleton node away from
    the crypt along the tangent of the fitted curved crypt centerline at the
    attachment.  The radius linearly tapers from the crypt attachment radius to
    the disk radius at the body/branch endpoint when expanded until it touches
    the host primitive.  This avoids fitting additional VAE-facing degrees of
    freedom.
    """
    if not isinstance(config, BlendConfig):
        config = BlendConfig.from_dict(config)
    if not config.enabled:
        return {}

    blends: dict[str, BlendAttachment] = {}
    blends.update(_create_crypt_extension_blends(graph, config))
    blends.update(_create_body_branch_neck_replacement_blends(graph, config))
    return blends

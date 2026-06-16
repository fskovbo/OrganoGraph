"""Primitive fitting for body-branch neck connector cylinders."""

from __future__ import annotations

from typing import Any

import numpy as np

from organograph.skeleton.datatypes import SkeletonGraph
from organograph.skeleton.geometry import as_points
from organograph.skeleton.primitive.common import _coerce_indices, _residual_summary
from organograph.skeleton.primitives import PrimitiveFit

def fit_straight_neck_cylinder(
    vertices,
    boundary_vertices,
    body_center,
    neck_center,
    branch_center,
    *,
    radius_quantile: float = 0.5,
    expansion_factor: float = 1.35,
    max_extent_fraction: float = 0.25,
    min_extent_radius_fraction: float = 0.35,
    n_axial_bins: int = 24,
) -> PrimitiveFit:
    """Fit a local straight cylinder at a stored body-branch cut ring.

    The axis is fixed by the body and branch centers and passes through the
    existing neck node. Radius comes from the cut-ring vertices. Cylinder
    endpoints are the first local axial bins whose robust surface radius has
    expanded by ``expansion_factor``, bounded to remain near the neck.
    """
    vertices = as_points(vertices)
    boundary = vertices[_coerce_indices(boundary_vertices)]
    body = np.asarray(body_center, dtype=float)
    neck = np.asarray(neck_center, dtype=float)
    branch = np.asarray(branch_center, dtype=float)
    axis = branch - body
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        axis = branch - neck
        axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        raise ValueError("Body and branch centers do not define a neck axis")
    axis /= axis_norm
    if float(np.dot(branch - neck, axis)) < 0.0:
        axis *= -1.0

    boundary_delta = boundary - neck[None, :]
    boundary_axial = boundary_delta @ axis
    boundary_radial = np.linalg.norm(
        boundary_delta - boundary_axial[:, None] * axis[None, :],
        axis=1,
    )
    finite_radius = boundary_radial[np.isfinite(boundary_radial)]
    if finite_radius.size < 3:
        raise ValueError("Too few finite body-branch boundary vertices")
    radius = float(np.quantile(finite_radius, float(radius_quantile)))
    radius = max(radius, 1e-6)

    body_limit = max(
        float(np.linalg.norm(neck - body)) * float(max_extent_fraction),
        radius * float(min_extent_radius_fraction),
    )
    branch_limit = max(
        float(np.linalg.norm(branch - neck)) * float(max_extent_fraction),
        radius * float(min_extent_radius_fraction),
    )
    delta = vertices - neck[None, :]
    axial = delta @ axis
    radial = np.linalg.norm(delta - axial[:, None] * axis[None, :], axis=1)
    local = (
        (axial >= -body_limit)
        & (axial <= branch_limit)
        & (radial <= max(3.0 * radius, radius + 1e-6))
    )
    threshold = max(float(expansion_factor), 1.0) * radius

    def first_expansion(sign: float, limit: float) -> float:
        edges = np.linspace(0.0, limit, max(4, int(n_axial_bins)) + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            coord = sign * axial
            mask = local & (coord >= lo) & (coord < hi)
            if np.count_nonzero(mask) < 3:
                continue
            local_radius = float(np.median(radial[mask]))
            if local_radius >= threshold:
                return max(float(lo), radius * float(min_extent_radius_fraction))
        return float(limit)

    body_extent = min(first_expansion(-1.0, body_limit), body_limit)
    branch_extent = min(first_expansion(1.0, branch_limit), branch_limit)
    start = neck - body_extent * axis
    end = neck + branch_extent * axis
    supported = (
        local
        & (axial >= -body_extent)
        & (axial <= branch_extent)
        & (radial <= threshold)
    )
    residuals = radial[supported] - radius
    summary = _residual_summary(residuals)
    return PrimitiveFit(
        primitive_type="straight_cylinder",
        parameters={
            "centerline_points": np.vstack([start, end]),
            "radius": radius,
            "axis": axis,
            "body_extent": body_extent,
            "branch_extent": branch_extent,
            "neck_center": neck,
            "expansion_factor": float(expansion_factor),
        },
        fit_error=summary["rmse"],
        residuals=summary,
        derived_parameters={
            "length": float(body_extent + branch_extent),
            "diameter": float(2.0 * radius),
            "aspect_ratio": float((body_extent + branch_extent) / (2.0 * radius)),
        },
        metadata={
            "fit_method": "fixed_axis_cut_ring_radius_local_expansion",
            "n_boundary_vertices": int(boundary.shape[0]),
            "n_supported_vertices": int(np.count_nonzero(supported)),
        },
    )

def attach_body_branch_neck_primitives(
    graph: SkeletonGraph,
    vertices,
    neck_components: dict[str, dict[str, Any]],
    *,
    body_component,
    branch_components: dict[str, Any],
    trim_radius_factor: float = 1.5,
    **fit_kwargs,
) -> dict[str, Any]:
    """Fit body-branch neck cylinders first and trim later blob components."""
    vertices = as_points(vertices)
    body_indices = _coerce_indices(body_component)
    trimmed_branches = {
        str(key): _coerce_indices(value)
        for key, value in branch_components.items()
    }
    attachments = {}

    for attachment_id, spec in neck_components.items():
        body_node_id = str(spec["body_node_id"])
        neck_node_id = str(spec["neck_node_id"])
        branch_node_id = str(spec["branch_node_id"])
        fit = fit_straight_neck_cylinder(
            vertices,
            spec["boundary_vertices"],
            graph.node(body_node_id).position,
            graph.node(neck_node_id).position,
            graph.node(branch_node_id).position,
            **fit_kwargs,
        )
        attachment = fit.to_attachment(
            attachment_type="path",
            attachment_id=str(attachment_id),
            target_ids=[body_node_id, neck_node_id, branch_node_id],
            metadata={"component": "body_branch_neck"},
        )
        graph.add_primitive_attachment(str(attachment_id), attachment)
        attachments[str(attachment_id)] = attachment

        axis = np.asarray(fit.parameters["axis"], dtype=float)
        neck = np.asarray(fit.parameters["neck_center"], dtype=float)
        radius = float(fit.parameters["radius"]) * float(trim_radius_factor)
        body_extent = float(fit.parameters["body_extent"])
        branch_extent = float(fit.parameters["branch_extent"])

        def trim(indices, *, side: str):
            if indices.size == 0:
                return indices
            delta = vertices[indices] - neck[None, :]
            axial = delta @ axis
            radial = np.linalg.norm(
                delta - axial[:, None] * axis[None, :],
                axis=1,
            )
            if side == "body":
                remove = (axial > -body_extent) & (radial <= radius)
            else:
                remove = (axial < branch_extent) & (radial <= radius)
            kept = indices[~remove]
            return kept if kept.size >= 3 else indices

        body_indices = trim(body_indices, side="body")
        if branch_node_id in trimmed_branches:
            trimmed_branches[branch_node_id] = trim(
                trimmed_branches[branch_node_id],
                side="branch",
            )

    return {
        "attachments": attachments,
        "body": body_indices.tolist(),
        "branches": {
            key: value.tolist()
            for key, value in trimmed_branches.items()
        },
    }


"""High-level workflow helpers for batch skeletonization."""

from __future__ import annotations

from typing import Any

from organograph.skeleton.config import PrimitiveFitConfig, SkeletonizationConfig
from organograph.skeleton.detection.graph_builder import build_skeleton_from_crypt_detections
from organograph.skeleton.detection.pipeline import detect_crypts_for_skeleton
from organograph.skeleton.primitive import (
    attach_body_branch_neck_primitives,
    attach_body_primitive,
    attach_branch_primitives,
    attach_crypt_tube_primitives,
    primitive_components_from_crypt_detections,
)
from organograph.skeleton.results import PrimitiveFitResult, SkeletonizationResult


def skeletonize_organoid(
    mesh,
    vocab,
    *,
    geodesic_fn,
    config: SkeletonizationConfig | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> SkeletonizationResult:
    """Run crypt detection and build a skeleton result object for one mesh."""
    if not isinstance(config, SkeletonizationConfig):
        config = SkeletonizationConfig.from_dict(config)

    detection_kwargs = dict(config.detection_kwargs)
    build_kwargs = dict(config.build_kwargs)
    detection_kwargs.pop("return_intermediates", None)
    detections, intermediates = detect_crypts_for_skeleton(
        mesh,
        vocab,
        geodesic_fn=geodesic_fn,
        return_intermediates=True,
        **detection_kwargs,
    )
    graph = build_skeleton_from_crypt_detections(
        vertices=mesh.v,
        faces=mesh.f,
        crypt_detections=detections,
        **build_kwargs,
    )
    return SkeletonizationResult(
        graph=graph,
        detections=detections,
        intermediates=intermediates,
        config=config,
        metadata=dict(metadata or {}),
        mesh=mesh,
    )


def fit_primitives_for_skeletonization_result(
    result: SkeletonizationResult,
    *,
    config: PrimitiveFitConfig | dict[str, Any] | None = None,
) -> PrimitiveFitResult:
    """Fit body, branch, neck, and crypt primitives for one skeleton result."""
    if result.mesh is None:
        raise ValueError("SkeletonizationResult.mesh is required for primitive fitting")
    if not isinstance(config, PrimitiveFitConfig):
        config = PrimitiveFitConfig.from_dict(config)

    mesh = result.mesh
    graph = result.graph
    graph.primitive_attachments.clear()
    components = primitive_components_from_crypt_detections(
        mesh.v,
        result.detections,
        graph=graph,
        faces=mesh.f,
        **dict(config.component_kwargs),
    )

    blob_components = {
        "body": components["body"],
        "branches": components["branches"],
    }
    neck_attachments = {}
    if components["body_branch_necks"]:
        blob_components = attach_body_branch_neck_primitives(
            graph,
            mesh.v,
            components["body_branch_necks"],
            body_component=components["body"],
            branch_components=components["branches"],
            **dict(config.body_branch_neck_kwargs),
        )
        neck_attachments = blob_components.get("attachments", {})

    body_attachment = attach_body_primitive(
        graph,
        mesh.v,
        blob_components["body"],
        **dict(config.body_kwargs),
    )
    branch_attachments = {}
    if blob_components["branches"]:
        branch_attachments = attach_branch_primitives(
            graph,
            mesh.v,
            blob_components["branches"],
            **dict(config.branch_kwargs),
        )
    crypt_attachments = {}
    if components["crypts"]:
        crypt_attachments = attach_crypt_tube_primitives(
            graph,
            mesh.v,
            components["crypts"],
            centerline_data=components["crypt_centerlines"],
            **dict(config.crypt_tube_kwargs),
        )

    return PrimitiveFitResult(
        graph=graph,
        components=components,
        attachments={
            "body": body_attachment,
            "branches": branch_attachments,
            "body_branch_necks": neck_attachments,
            "crypts": crypt_attachments,
        },
        config=config,
        metadata={"skeleton_metadata": dict(result.metadata)},
        skeleton=result,
    )

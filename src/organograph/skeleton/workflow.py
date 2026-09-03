"""High-level workflow helpers for batch skeletonization."""

from __future__ import annotations

from typing import Any

from organograph.skeleton.blending import create_attachment_blends
from organograph.skeleton.config import BlendConfig, PrimitiveFitConfig, SkeletonizationConfig
from organograph.skeleton.detection.graph_builder import build_skeleton_graph
from organograph.skeleton.detection.pipeline import detect_crypts_for_skeleton
from organograph.skeleton.primitive import (
    attach_body_branch_neck_primitives,
    attach_body_primitive,
    attach_branch_primitives,
    attach_crypt_tube_primitives,
    primitive_components_from_crypt_detections,
)
from organograph.skeleton.primitive.overlap import (
    assess_crypt_primitive_overlaps,
    merge_overlapping_crypt_detections,
    recompute_merged_crypt_geometry,
)
from organograph.skeleton.results import BlendResult, PrimitiveFitResult, SkeletonizationResult
from organograph.skeleton.primitives import PrimitiveAttachment


def _barrier_attachment(fit, *, attachment_id: str, component: str) -> PrimitiveAttachment:
    """Represent a detection-stage barrier as the definitive host primitive."""
    return PrimitiveAttachment(
        primitive_type=fit.primitive_type,
        parameters=fit.to_primitive_parameters(),
        fit_error=fit.objective,
        residuals={"objective": fit.objective},
        metadata={
            "component": component,
            "source": "detection_barrier",
            "fit_success": bool(fit.success),
            "fit_message": fit.message,
        },
        attachment_type="node",
        attachment_id=attachment_id,
        target_ids=[attachment_id],
    )


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

    detection = detect_crypts_for_skeleton(
        mesh,
        vocab,
        geodesic_fn=geodesic_fn,
        config=config.detection,
    )
    working_mesh = detection.detection_mesh
    branch_centers = {
        branch_id: fit.center for branch_id, fit in detection.barriers.branch_fits.items()
    }
    graph = build_skeleton_graph(
        vertices=working_mesh.v,
        faces=working_mesh.f,
        crypt_detections=detection.detections,
        body_center=detection.barriers.body_fit.center,
        branch_centers=branch_centers,
        metadata=metadata,
    )
    return SkeletonizationResult(
        graph=graph,
        detections=detection.detections,
        barriers=detection.barriers,
        intermediates=detection.diagnostics,
        config=config,
        metadata=dict(metadata or {}),
        mesh=working_mesh,
        geodesic_fn=geodesic_fn,
    )


def _fit_primitives_once(
    result: SkeletonizationResult,
    *,
    config: PrimitiveFitConfig,
) -> PrimitiveFitResult:
    """Fit all primitives once without post-fit topology validation."""
    if result.mesh is None:
        raise ValueError("SkeletonizationResult.mesh is required for primitive fitting")

    mesh = result.mesh
    graph = result.graph
    graph.primitive_attachments.clear()
    for node in graph.nodes.values():
        node.primitive_attachment = None
    for edge in graph.edges.values():
        edge.primitive_attachment = None
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

    if config.refine_host_primitives:
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
    else:
        if result.barriers is None:
            raise ValueError("Barrier fits are required when host refinement is disabled")
        body_attachment = _barrier_attachment(
            result.barriers.body_fit,
            attachment_id="body",
            component="body",
        )
        graph.node("body").primitive_attachment = body_attachment
        branch_attachments = {}
        for branch_id, fit in result.barriers.branch_fits.items():
            if branch_id not in graph.nodes:
                continue
            attachment = _barrier_attachment(
                fit,
                attachment_id=branch_id,
                component="branch",
            )
            graph.node(branch_id).primitive_attachment = attachment
            branch_attachments[branch_id] = attachment
    crypt_attachments = {}
    if components["crypts"]:
        crypt_attachments = attach_crypt_tube_primitives(
            graph,
            mesh.v,
            components["crypts"],
            centerline_data=components["crypt_centerlines"],
            mesh=mesh,
            geodesic_fn=result.geodesic_fn,
            geodesic_kwargs=dict(
                result.config.detection.candidates.geodesic_kwargs
            ),
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


def _rebuild_graph_after_crypt_merge(
    result: SkeletonizationResult,
    detections: list[dict[str, Any]],
) -> None:
    """Replace a result's graph with topology rebuilt from merged detections."""
    branch_centers = {}
    if result.barriers is not None:
        branch_centers = {
            branch_id: fit.center
            for branch_id, fit in result.barriers.branch_fits.items()
        }
    result.detections = detections
    result.graph = build_skeleton_graph(
        vertices=result.mesh.v,
        faces=result.mesh.f,
        crypt_detections=detections,
        body_center=(
            result.barriers.body_fit.center
            if result.barriers is not None
            else result.graph.body_node().position
        ),
        branch_centers=branch_centers,
        metadata=result.metadata,
    )


def fit_primitives_for_skeletonization_result(
    result: SkeletonizationResult,
    *,
    config: PrimitiveFitConfig | dict[str, Any] | None = None,
) -> PrimitiveFitResult:
    """Fit primitives, merge strongly overlapping crypts, and refit.

    Crypt overlap is measured only between terminal tubes attached to the same
    host component. Connected overlap groups are merged in the detection
    hierarchy, the affected skeleton topology is rebuilt, and all primitives
    are fitted again. This keeps the final graph and primitive representation
    consistent for downstream encoding.
    """
    if not isinstance(config, PrimitiveFitConfig):
        config = PrimitiveFitConfig.from_dict(config)

    primitive_result = _fit_primitives_once(result, config=config)
    overlap_config = config.crypt_overlap
    overlap_metadata = {
        "enabled": bool(overlap_config.enabled),
        "threshold": float(overlap_config.threshold),
        "passes": [],
        "merge_records": [],
        "geometry_recomputations": [],
        "n_merge_groups": 0,
        "converged": True,
    }
    if not overlap_config.enabled:
        primitive_result.metadata["crypt_overlap_merge"] = overlap_metadata
        result.metadata["crypt_overlap_merge"] = overlap_metadata
        result.graph.metadata["crypt_overlap_merge"] = overlap_metadata
        return primitive_result

    max_passes = max(1, int(overlap_config.max_passes))
    last_assessment = None
    for pass_index in range(max_passes):
        assessment = assess_crypt_primitive_overlaps(
            primitive_result,
            overlap_config,
        )
        assessment_data = assessment.to_dict()
        assessment_data["pass_index"] = pass_index
        overlap_metadata["passes"].append(assessment_data)
        last_assessment = assessment
        if not assessment.requires_merge:
            break

        merge_result = merge_overlapping_crypt_detections(
            result.detections,
            assessment,
            result.mesh.v,
        )
        if not merge_result.changed:
            overlap_metadata["converged"] = False
            break
        overlap_metadata["merge_records"].extend(merge_result.records)
        merged_detections, geometry_records = recompute_merged_crypt_geometry(
            result.mesh,
            merge_result.detections,
            detection_config=result.config.detection,
            barriers=result.barriers,
            diagnostics=result.intermediates,
            geodesic_fn=result.geodesic_fn,
        )
        overlap_metadata["geometry_recomputations"].extend(geometry_records)
        _rebuild_graph_after_crypt_merge(result, merged_detections)
        primitive_result = _fit_primitives_once(result, config=config)
    else:
        if last_assessment is not None and last_assessment.requires_merge:
            final_assessment = assess_crypt_primitive_overlaps(
                primitive_result,
                overlap_config,
            )
            final_data = final_assessment.to_dict()
            final_data["pass_index"] = max_passes
            final_data["validation_only"] = True
            overlap_metadata["passes"].append(final_data)
            overlap_metadata["converged"] = not final_assessment.requires_merge

    overlap_metadata["n_merge_groups"] = len(overlap_metadata["merge_records"])
    primitive_result.metadata["crypt_overlap_merge"] = overlap_metadata
    result.metadata["crypt_overlap_merge"] = overlap_metadata
    result.graph.metadata["crypt_overlap_merge"] = overlap_metadata
    return primitive_result


def blend_primitives_for_visualization(
    primitive_result: PrimitiveFitResult,
    *,
    config: BlendConfig | dict[str, Any] | None = None,
) -> BlendResult:
    """Create visualization-only blends from a primitive fit result."""
    if not isinstance(config, BlendConfig):
        config = BlendConfig.from_dict(config)
    mesh = primitive_result.mesh
    blend_attachments = create_attachment_blends(
        primitive_result.graph,
        vertices=None if mesh is None else mesh.v,
        config=config,
    )
    return BlendResult(
        graph=primitive_result.graph,
        blend_attachments=blend_attachments,
        config=config,
        primitive_result=primitive_result,
        metadata={"stage": "visual_blending"},
    )

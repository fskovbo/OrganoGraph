"""Barrier-aware, biology-aware organoid skeletons and shape primitives.

The public workflow has three explicit stages: skeletonization, interpretable
primitive fitting, and visualization-only blending. Skeletonization always
fits body/branch barriers and places appendix attachments at their boundaries.
"""

from organograph.skeleton.config import (
    BarrierConfig,
    BlendConfig,
    BodyTransitionConfig,
    BranchValidationConfig,
    CandidateDetectionConfig,
    CryptOverlapConfig,
    DetectionConfig,
    GraphConfig,
    MeshPreparationConfig,
    NeckProfileConfig,
    PrimitiveFitConfig,
    SkeletonizationConfig,
)
from organograph.skeleton.datatypes import NODE_TYPES, SkeletonEdge, SkeletonGraph, SkeletonNode
from organograph.skeleton.export import load_shape_export_json, save_shape_export, write_export_readme
from organograph.skeleton.io import load_skeleton_json, save_skeleton_json
from organograph.skeleton.primitives import Primitive, PrimitiveAttachment, PrimitiveFit
from organograph.skeleton.results import (
    BarrierStageResult,
    BlendResult,
    DetectionResult,
    OrganoidShapeResult,
    PrimitiveFitResult,
    SkeletonizationResult,
)
from organograph.skeleton.workflow import (
    blend_primitives_for_visualization,
    fit_primitives_for_skeletonization_result,
    skeletonize_organoid,
)

__all__ = [
    "BarrierConfig",
    "BarrierStageResult",
    "BlendConfig",
    "BlendResult",
    "BodyTransitionConfig",
    "BranchValidationConfig",
    "CandidateDetectionConfig",
    "CryptOverlapConfig",
    "DetectionConfig",
    "DetectionResult",
    "GraphConfig",
    "MeshPreparationConfig",
    "NODE_TYPES",
    "NeckProfileConfig",
    "OrganoidShapeResult",
    "Primitive",
    "PrimitiveAttachment",
    "PrimitiveFit",
    "PrimitiveFitConfig",
    "PrimitiveFitResult",
    "SkeletonEdge",
    "SkeletonGraph",
    "SkeletonNode",
    "SkeletonizationConfig",
    "SkeletonizationResult",
    "blend_primitives_for_visualization",
    "fit_primitives_for_skeletonization_result",
    "load_shape_export_json",
    "load_skeleton_json",
    "save_shape_export",
    "save_skeleton_json",
    "skeletonize_organoid",
    "write_export_readme",
]

"""Barrier-aware, biology-aware organoid skeletons and shape primitives.

The public workflow has three explicit stages: skeletonization, interpretable
primitive fitting, and visualization-only blending. Skeletonization always
fits body/branch barriers and places appendix attachments at their boundaries.
"""

from organograph.skeleton.config import (
    BarrierConfig,
    BlendConfig,
    BranchValidationConfig,
    CandidateDetectionConfig,
    CryptOverlapConfig,
    DetectionConfig,
    MeshPreparationConfig,
    NeckProfileConfig,
    PrimitiveFitConfig,
    SkeletonizationConfig,
)
from organograph.skeleton.datatypes import (
    NODE_TYPES,
    SkeletonEdge,
    SkeletonGraph,
    SkeletonNode,
)
from organograph.skeleton.export import (
    SHAPE_QUALITY_SCHEMA_VERSION,
    SHAPE_EXPORT_SCHEMA_VERSION,
    graph_from_shape_export_payload,
    graph_summary,
    load_shape_export_graph,
    load_shape_export_json,
    save_shape_export,
    shape_quality_payload,
    shape_export_payload,
    validate_shape_export_payload,
    write_export_readme,
)
from organograph.skeleton.io import load_skeleton_json, save_skeleton_json
from organograph.skeleton.profiles import (
    definitive_filter_options,
    definitive_mesh_preparation,
    definitive_primitive_fit_config,
    definitive_skeletonization_config,
)
from organograph.skeleton.primitives import (
    Primitive,
    PrimitiveAttachment,
    PrimitiveFit,
)
from organograph.skeleton.primitive.crypt_geometry import (
    CryptGeometryFit,
    HermiteCenterlineFit,
    boundary_tip_ratio_field,
    centerline_radius_observations,
    fit_crypt_geometry,
    hermite_curvature_diagnostics,
    minimum_contour_radius,
    monotonic_project_points_to_polyline,
    sample_tangent_hermite,
)
from organograph.skeleton.primitive_qc import (
    CryptPrimitiveQCConfig,
    QC_FLAG_COLUMNS,
    crypt_primitive_qc_records,
    discover_shape_exports,
    load_crypt_primitive_qc,
    pair_crypt_primitive_qc,
)
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
    "BranchValidationConfig",
    "CandidateDetectionConfig",
    "CryptOverlapConfig",
    "CryptGeometryFit",
    "HermiteCenterlineFit",
    "CryptPrimitiveQCConfig",
    "DetectionConfig",
    "DetectionResult",
    "MeshPreparationConfig",
    "NODE_TYPES",
    "NeckProfileConfig",
    "OrganoidShapeResult",
    "Primitive",
    "PrimitiveAttachment",
    "PrimitiveFit",
    "PrimitiveFitConfig",
    "PrimitiveFitResult",
    "QC_FLAG_COLUMNS",
    "SHAPE_EXPORT_SCHEMA_VERSION",
    "SHAPE_QUALITY_SCHEMA_VERSION",
    "SkeletonEdge",
    "SkeletonGraph",
    "SkeletonNode",
    "SkeletonizationConfig",
    "SkeletonizationResult",
    "blend_primitives_for_visualization",
    "boundary_tip_ratio_field",
    "centerline_radius_observations",
    "crypt_primitive_qc_records",
    "fit_primitives_for_skeletonization_result",
    "fit_crypt_geometry",
    "hermite_curvature_diagnostics",
    "definitive_filter_options",
    "definitive_mesh_preparation",
    "definitive_primitive_fit_config",
    "definitive_skeletonization_config",
    "graph_from_shape_export_payload",
    "graph_summary",
    "discover_shape_exports",
    "load_crypt_primitive_qc",
    "load_shape_export_graph",
    "load_shape_export_json",
    "load_skeleton_json",
    "minimum_contour_radius",
    "monotonic_project_points_to_polyline",
    "save_shape_export",
    "pair_crypt_primitive_qc",
    "save_skeleton_json",
    "sample_tangent_hermite",
    "skeletonize_organoid",
    "shape_export_payload",
    "shape_quality_payload",
    "validate_shape_export_payload",
    "write_export_readme",
]

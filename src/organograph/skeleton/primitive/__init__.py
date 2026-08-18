"""Interpretable primitive fitting used after skeletonization."""

from organograph.skeleton.primitive.barriers import (
    BarrierPrimitiveConfig,
    BarrierPrimitiveFit,
    barrier_primitive_level,
    barrier_primitive_vertices_like_mesh,
    fit_barrier_primitive,
    fit_barrier_primitive_sampled,
    relative_height_field,
)
from organograph.skeleton.primitive.blobs import (
    fit_asymmetric_superellipsoid_to_points,
    fit_blob_primitive_to_points,
    fit_ellipsoid_to_points,
)
from organograph.skeleton.primitive.components import primitive_components_from_crypt_detections
from organograph.skeleton.primitive.necks import attach_body_branch_neck_primitives, fit_straight_neck_cylinder
from organograph.skeleton.primitive.overlap import (
    CryptDetectionMergeResult,
    CryptOverlapAssessment,
    TerminalCryptReference,
    assess_crypt_primitive_overlaps,
    merge_overlapping_crypt_detections,
    recompute_merged_crypt_geometry,
    tube_overlap_fraction,
)
from organograph.skeleton.primitive.tubes import (
    attach_body_primitive,
    attach_branch_primitives,
    attach_crypt_tube_primitives,
    crypt_terminal_paths,
    fit_crypt_tube_to_points,
    primitive_attachments_to_dataframe,
)

__all__ = [
    "BarrierPrimitiveConfig",
    "BarrierPrimitiveFit",
    "CryptDetectionMergeResult",
    "CryptOverlapAssessment",
    "TerminalCryptReference",
    "assess_crypt_primitive_overlaps",
    "attach_body_branch_neck_primitives",
    "attach_body_primitive",
    "attach_branch_primitives",
    "attach_crypt_tube_primitives",
    "barrier_primitive_level",
    "barrier_primitive_vertices_like_mesh",
    "crypt_terminal_paths",
    "fit_asymmetric_superellipsoid_to_points",
    "fit_barrier_primitive",
    "fit_barrier_primitive_sampled",
    "fit_blob_primitive_to_points",
    "fit_crypt_tube_to_points",
    "fit_ellipsoid_to_points",
    "fit_straight_neck_cylinder",
    "merge_overlapping_crypt_detections",
    "recompute_merged_crypt_geometry",
    "primitive_attachments_to_dataframe",
    "primitive_components_from_crypt_detections",
    "relative_height_field",
    "tube_overlap_fraction",
]

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
from organograph.skeleton.primitive.radius_support import (
    CryptRadiusSupportResult,
    grow_crypt_radius_support_regions,
)
from organograph.skeleton.primitive.radius_profiles import (
    RadiusProfileFitResult,
    RadiusProfileObservations,
    fit_fixed_grid_radius_profile,
    fitted_radius_volume_center,
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
    "CryptGeometryFit",
    "HermiteCenterlineFit",
    "CryptOverlapAssessment",
    "CryptRadiusSupportResult",
    "RadiusProfileFitResult",
    "RadiusProfileObservations",
    "TerminalCryptReference",
    "assess_crypt_primitive_overlaps",
    "attach_body_branch_neck_primitives",
    "attach_body_primitive",
    "attach_branch_primitives",
    "attach_crypt_tube_primitives",
    "boundary_tip_ratio_field",
    "centerline_radius_observations",
    "barrier_primitive_level",
    "barrier_primitive_vertices_like_mesh",
    "crypt_terminal_paths",
    "fit_asymmetric_superellipsoid_to_points",
    "fit_barrier_primitive",
    "fit_barrier_primitive_sampled",
    "fit_blob_primitive_to_points",
    "fit_crypt_tube_to_points",
    "fit_crypt_geometry",
    "fit_fixed_grid_radius_profile",
    "fitted_radius_volume_center",
    "hermite_curvature_diagnostics",
    "fit_ellipsoid_to_points",
    "fit_straight_neck_cylinder",
    "merge_overlapping_crypt_detections",
    "grow_crypt_radius_support_regions",
    "minimum_contour_radius",
    "monotonic_project_points_to_polyline",
    "recompute_merged_crypt_geometry",
    "primitive_attachments_to_dataframe",
    "primitive_components_from_crypt_detections",
    "relative_height_field",
    "sample_tangent_hermite",
    "tube_overlap_fraction",
]

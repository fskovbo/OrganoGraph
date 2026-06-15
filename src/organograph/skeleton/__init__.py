"""Biology-aware skeleton graphs for organoid meshes.

The public API builds compact straight-edge skeletons from crypt detections.
It is intentionally separate from primitive fitting and from generic medial
axis extraction.
"""

from organograph.skeleton.build import (
    build_skeleton_from_crypt_detections,
    build_skeleton_from_segmentation_parameters,
    detect_crypts_for_skeleton,
    normalize_crypt_detections,
)
from organograph.skeleton.datatypes import (
    NODE_TYPES,
    SkeletonEdge,
    SkeletonGraph,
    SkeletonNode,
)
from organograph.skeleton.geometry import (
    crypt_attachment_direction,
    crypt_bend_angle,
    crypt_path_length,
    crypt_straight_distance,
    crypt_tortuosity,
    edge_length,
    number_of_crypts,
    number_of_split_crypts,
    skeleton_to_body_relative,
    transform_points_body_relative,
)
from organograph.skeleton.io import load_skeleton_json, save_skeleton_json
from organograph.skeleton.primitive_geometry import (
    estimate_smooth_crypt_centerline,
    sample_quadratic_bezier,
)
from organograph.skeleton.primitive_fitting import (
    attach_body_primitive,
    attach_branch_primitives,
    attach_crypt_tube_primitives,
    crypt_terminal_paths,
    fit_blob_primitive_to_points,
    fit_crypt_tube_to_points,
    fit_ellipsoid_to_points,
    primitive_components_from_crypt_detections,
    primitive_attachments_to_dataframe,
)
from organograph.skeleton.primitives import (
    Primitive,
    PrimitiveAttachment,
    PrimitiveFit,
)

__all__ = [
    "NODE_TYPES",
    "Primitive",
    "PrimitiveAttachment",
    "PrimitiveFit",
    "SkeletonEdge",
    "SkeletonGraph",
    "SkeletonNode",
    "attach_body_primitive",
    "attach_branch_primitives",
    "attach_crypt_tube_primitives",
    "build_skeleton_from_crypt_detections",
    "build_skeleton_from_segmentation_parameters",
    "crypt_attachment_direction",
    "crypt_bend_angle",
    "crypt_path_length",
    "crypt_straight_distance",
    "crypt_tortuosity",
    "crypt_terminal_paths",
    "detect_crypts_for_skeleton",
    "edge_length",
    "estimate_smooth_crypt_centerline",
    "fit_blob_primitive_to_points",
    "fit_crypt_tube_to_points",
    "fit_ellipsoid_to_points",
    "load_skeleton_json",
    "normalize_crypt_detections",
    "number_of_crypts",
    "number_of_split_crypts",
    "primitive_components_from_crypt_detections",
    "primitive_attachments_to_dataframe",
    "save_skeleton_json",
    "sample_quadratic_bezier",
    "skeleton_to_body_relative",
    "transform_points_body_relative",
]

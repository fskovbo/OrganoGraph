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
from organograph.skeleton.primitives import PrimitiveAttachment

__all__ = [
    "NODE_TYPES",
    "PrimitiveAttachment",
    "SkeletonEdge",
    "SkeletonGraph",
    "SkeletonNode",
    "build_skeleton_from_crypt_detections",
    "build_skeleton_from_segmentation_parameters",
    "crypt_attachment_direction",
    "crypt_bend_angle",
    "crypt_path_length",
    "crypt_straight_distance",
    "crypt_tortuosity",
    "detect_crypts_for_skeleton",
    "edge_length",
    "load_skeleton_json",
    "normalize_crypt_detections",
    "number_of_crypts",
    "number_of_split_crypts",
    "save_skeleton_json",
    "skeleton_to_body_relative",
    "transform_points_body_relative",
]

"""Compatibility facade for organoid skeletonization.

The implementation is split across :mod:`organograph.skeleton.detection` so the
pipeline can evolve in smaller pieces.  Import from the submodules for new code;
this module re-exports the historical names used by tests and notebooks.
"""

from organograph.skeleton.detection.branch_validation import (
    _grow_parent_patch_to_neck,
    _validate_split_branch_geometry,
)
from organograph.skeleton.detection.graph_builder import (
    _penalize_short_crypt_bending,
    build_skeleton_from_crypt_detections,
    normalize_crypt_detections,
)
from organograph.skeleton.detection.neck_profiles import analyze_neck_circumference_profile
from organograph.skeleton.detection.pipeline import (
    build_skeleton_from_segmentation_parameters,
    detect_crypts_for_skeleton,
)
from organograph.skeleton.detection.region_refinement import (
    _earlier_second_derivative_transition_level,
    _refine_body_transition_width_outliers,
    _refine_broad_transition_opening,
)
from organograph.skeleton.detection.tips import _select_hks_tips_from_axis

__all__ = [
    "_earlier_second_derivative_transition_level",
    "_grow_parent_patch_to_neck",
    "_penalize_short_crypt_bending",
    "_refine_body_transition_width_outliers",
    "_refine_broad_transition_opening",
    "_select_hks_tips_from_axis",
    "_validate_split_branch_geometry",
    "analyze_neck_circumference_profile",
    "build_skeleton_from_crypt_detections",
    "build_skeleton_from_segmentation_parameters",
    "detect_crypts_for_skeleton",
    "normalize_crypt_detections",
]

"""Compatibility facade for primitive fitting.

Primitive fitting is implemented in :mod:`organograph.skeleton.primitive` modules.
This file re-exports the historical names used by notebooks and tests.
"""

from organograph.skeleton.primitive.blobs import (
    fit_asymmetric_superellipsoid_to_points,
    fit_blob_primitive_to_points,
    fit_ellipsoid_to_points,
)
from organograph.skeleton.primitive.components import (
    component_region_from_detection,
    primitive_components_from_crypt_detections,
)
from organograph.skeleton.primitive.necks import (
    attach_body_branch_neck_primitives,
    fit_straight_neck_cylinder,
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
    "attach_body_primitive",
    "attach_body_branch_neck_primitives",
    "attach_branch_primitives",
    "attach_crypt_tube_primitives",
    "component_region_from_detection",
    "crypt_terminal_paths",
    "fit_asymmetric_superellipsoid_to_points",
    "fit_blob_primitive_to_points",
    "fit_crypt_tube_to_points",
    "fit_ellipsoid_to_points",
    "fit_straight_neck_cylinder",
    "primitive_components_from_crypt_detections",
    "primitive_attachments_to_dataframe",
]

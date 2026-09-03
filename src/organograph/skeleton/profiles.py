"""Maintained parameter profiles for the definitive shape pipeline.

The tutorial and batch exporter use these factories so an exported shape is the
same skeleton and primitive fit shown in ``notebooks/tutorial_skeleton.ipynb``.
Callers may still construct the public config dataclasses directly for parameter
experiments.
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial

import numpy as np

from organograph.crypts.filters import (
    filter_crypts_by_hks_percent,
    filter_crypts_by_size,
)
from organograph.skeleton.config import (
    BarrierConfig,
    BranchValidationConfig,
    CandidateDetectionConfig,
    CryptOverlapConfig,
    DetectionConfig,
    MeshPreparationConfig,
    NeckProfileConfig,
    PrimitiveFitConfig,
    SkeletonizationConfig,
)


DEFINITIVE_MESH_PREPARATION = {
    "normalize_mesh": True,
    "normalize_scale": 10.0,
    "eigen_k": 225,
    "smooth_mesh": True,
    "smooth_lmax": 12,
    "smooth_eigen_k": None,
}

DEFINITIVE_FILTER_OPTIONS = {
    "use_hks_filter": True,
    "min_percent_greater": 1.0,
    "hks_t_min": None,
    "hks_t_max": None,
    "use_size_filter": True,
    "min_patch_area": 5.0,
}


def definitive_mesh_preparation() -> dict:
    """Return an independent copy of the maintained mesh preparation settings."""
    return deepcopy(DEFINITIVE_MESH_PREPARATION)


def definitive_filter_options() -> dict:
    """Return an independent copy of the maintained crypt-filter settings."""
    return deepcopy(DEFINITIVE_FILTER_OPTIONS)


def make_definitive_filters(options: dict | None = None) -> list | None:
    """Build crypt filters used by the maintained tutorial/export profile."""
    settings = definitive_filter_options()
    settings.update(dict(options or {}))
    filters = []
    if settings["use_hks_filter"]:
        filters.append(
            partial(
                filter_crypts_by_hks_percent,
                min_percent_greater=settings["min_percent_greater"],
                t_min=settings.get("hks_t_min"),
                t_max=settings.get("hks_t_max"),
            )
        )
    if settings["use_size_filter"]:
        filters.append(
            partial(
                filter_crypts_by_size,
                min_patch_area=settings.get("min_patch_area"),
            )
        )
    return filters or None


def definitive_skeletonization_config(
    *,
    filter_options: dict | None = None,
) -> SkeletonizationConfig:
    """Return the maintained barrier-aware skeletonization configuration."""
    body_barrier_options = {
        "primitive_type": "superellipsoid",
        "barrier_weight": 2.5,
        "underfill_weight": 0.02,
        "center_regularization": 0.01,
        "anisotropy_regularization": 0.1,
        "center_shift_limit_frac": 0.55,
        "initial_radius_quantile": 0.6,
        "initial_epsilon_1": 0.9,
        "epsilon_1_bounds": (0.35, 1.0),
        "epsilon_1_regularization": 0.01,
        "epsilon_2": 1.0,
        "maxiter": 2000,
    }
    branch_barrier_options = {
        **body_barrier_options,
        "primitive_type": "ellipsoid",
        "anisotropy_regularization": 1.0,
        "underfill_weight": 0.5,
        "barrier_weight": 5.0,
    }
    return SkeletonizationConfig(
        detection=DetectionConfig(
            candidates=CandidateDetectionConfig(
                threshold=0.5,
                filters=make_definitive_filters(filter_options),
                refine_threshold=0.0,
                refine_min_area=5.0,
                min_child_fraction=0.05,
                final_tip_hks_time=1.0,
                final_tip_bottom_fraction=0.6,
                final_tip_min_hks_percent_increase=5.0,
            ),
            necks=NeckProfileConfig(
                max_axis_level=2.0,
                resolution=200,
                search_interval=(0.8, 2.0),
                min_prominence=0.05,
                min_length=0.05,
            ),
            branches=BranchValidationConfig(
                min_confidence=0.85,
                max_neck_to_body_radius_ratio=0.70,
                max_growth_size_factor=3.0,
                max_mesh_fraction=0.40,
            ),
            barriers=BarrierConfig(
                body_fit_options=body_barrier_options,
                branch_fit_options=branch_barrier_options,
                body_ownership_level=1.2,
                branch_ownership_level=1.1,
                sample_fraction=1.0,
                sample_seed=0,
            ),
            mesh=MeshPreparationConfig(smooth=False),
        ),
    )


def definitive_primitive_fit_config() -> PrimitiveFitConfig:
    """Return the maintained primitive fit shown in the skeleton tutorial."""
    return PrimitiveFitConfig(
        refine_host_primitives=False,
        body_branch_neck_kwargs={
            "radius_quantile": 0.25,
            "expansion_factor": 1.35,
            "max_extent_fraction": 0.25,
            "min_extent_radius_fraction": 0.35,
        },
        crypt_tube_kwargs={
            "centerline_n_contours": 10,
            "centerline_n_samples": 64,
            "opening_frame_blend_fraction": 0.15,
            "update_crypt_nodes": True,
            "radius_quantile": 0.5,
            "optimize_radius_profile": True,
            "fixed_taper_position": 0.85,
            "outside_volume_weight": 2.0,
            "profile_n_bins": 20,
            "profile_min_points_per_bin": 2,
            "profile_min_supported_bins": 6,
            "max_constriction_to_neighbor_fraction": 0.98,
            "tip_projection_tolerance": 1e-6,
        },
        crypt_overlap=CryptOverlapConfig(
            enabled=True,
            threshold=0.30,
            samples=8192,
            random_seed=0,
            max_passes=3,
            max_host_attachment_angle=np.pi / 3,
        ),
    )

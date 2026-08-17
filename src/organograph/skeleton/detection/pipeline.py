"""High-level skeletonization pipeline from crypt detection to graph inputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from organograph.skeleton.datatypes import SkeletonGraph
from organograph.skeleton.detection.branch_validation import _grow_parent_patch_to_neck, _validate_split_branch_geometry
from organograph.skeleton.detection.common import _coerce_patch
from organograph.skeleton.detection.graph_builder import build_skeleton_from_crypt_detections
from organograph.skeleton.detection.mesh_regions import _boundary_edges_for_region, _low_pass_smoothed_mesh_for_detection, _mesh_edges_from_faces
from organograph.skeleton.detection.neck_profiles import _add_neck_profile_geometry
from organograph.skeleton.detection.region_refinement import _refine_body_transition_width_outliers, _refine_broad_transition_opening
from organograph.skeleton.detection.tips import _select_hks_tips_from_axis
from organograph.skeleton.barrier_ellipsoid import (
    fit_branch_barrier_primitives,
    fit_soft_barrier_primitive_sampled,
    protect_patches_from_mask,
    villus_mask_from_barrier_primitive,
)
from organograph.skeleton.detection.barrier_crossings import (
    assign_crypt_attachments_from_barrier_crossings,
)

def detect_crypts_for_skeleton(
    mesh,
    vocab,
    *,
    geodesic_fn,
    L_ref=None,
    crypt_vocab_idx=None,
    threshold=0.5,
    filter_fn_list=None,
    refine_crypts=True,
    refine_threshold=0.0,
    refine_only_if_area_at_least=5.0,
    min_refined_frac_of_parent=0.1,
    geodesic_kwargs=None,
    final_tip_hks_time: float = 1.0,
    final_tip_bottom_fraction: float = 0.5,
    final_tip_min_hks_percent_increase: float = 0.0,
    extend_max=2.0,
    disc_resolution=200,
    neck_search_interval=(0.8, 2.0),
    neck_window_length: int = 9,
    neck_polyorder: int = 3,
    neck_min_prominence: float = 0.05,
    neck_min_length: float = 0.05,
    validate_split_stems: bool = True,
    validate_branch_geometry: bool = True,
    branch_min_confidence: float = 0.6,
    branch_max_neck_to_body_radius_ratio: float = 0.8,
    split_growth_max_size_factor: float = 2.0,
    split_growth_max_mesh_fraction: float = 0.35,
    split_growth_smooth_perimeter: bool = True,
    split_growth_smoothing_tolerance: float = 0.0,
    split_growth_min_decrease_fraction: float = 0.0,
    split_growth_min_prominence_fraction: float = 0.01,
    split_growth_robust_window: int = 1,
    refine_broad_crypt_openings: bool = True,
    max_opening_to_crypt_body_ratio: float = 0.85,
    branch_max_opening_to_crypt_body_ratio: float = 0.95,
    broad_opening_min_linear_profile_r2: float = 0.985,
    broad_opening_max_linear_profile_deviation: float = 0.08,
    broad_opening_min_attachment_level: float = 0.35,
    refine_body_transition_width_outliers: bool = True,
    body_transition_max_crypt_to_host_width_ratio: float = 0.8,
    body_transition_host_width_quantile: float = 0.75,
    body_transition_min_second_derivative_score: float = 0.6,
    body_transition_min_attachment_level: float = 0.25,
    body_barrier_ellipsoid: bool = False,
    body_barrier_config: dict[str, Any] | None = None,
    body_barrier_relative_height_threshold: float = 1.05,
    body_barrier_min_candidate_vertices: int = 4,
    body_barrier_sample_fraction: float = 1.0,
    body_barrier_sample_seed: int | None = 0,
    barrier_boundary_attachments: bool = True,
    barrier_crossing_kwargs: dict[str, Any] | None = None,
    branch_barrier_config: dict[str, Any] | None = None,
    branch_barrier_relative_height_threshold: float = 1.05,
    branch_barrier_min_candidate_vertices: int = 20,
    smooth_mesh: bool = False,
    smooth_lmax: int = 5,
    smooth_recompute_eigen: bool = True,
    smooth_eigen_k: int | None = None,
    return_intermediates=False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run crypt detection pieces needed for skeleton construction.

    With body-barrier fitting enabled, the body primitive and ownership field
    are computed before the fresh HKS candidate screen. Parent candidate patches
    are then kept as skeleton crypt trunks; if local refinement splits a parent
    into multiple child patches, the output marks that parent as a split crypt
    with daughter tips.

    The geodesic axis and neckline are computed from the original
    boundary-distance bottom.  The skeleton tip is then updated to the max-HKS
    vertex near ``final_tip_hks_time`` in the bottom fraction of the refined
    crypt axis, provided the HKS increase over the initial tip clears
    ``final_tip_min_hks_percent_increase``. Circumference profiles classify
    transitions and genuine constrictions. For barrier-aware runs, final
    host-side boundaries come from the first persistent geodesic-ring crossing
    of the body or branch primitive; the resulting tip-connected regions then
    drive graph and primitive construction.
    """
    from organograph.crypts.axis import compute_crypt_axis, normalize_crypt_axis_to_neckline
    from organograph.crypts.filters import apply_filters
    from organograph.crypts.vocab import detect_crypts_by_encoding, subdivide_crypts_by_encoding

    if geodesic_kwargs is None:
        geodesic_kwargs = {}

    detection_mesh = (
        _low_pass_smoothed_mesh_for_detection(
            mesh,
            lmax=smooth_lmax,
            recompute_eigen=smooth_recompute_eigen,
            eigen_k=smooth_eigen_k,
        )
        if smooth_mesh
        else mesh
    )

    # The body estimate is deliberately independent of HKS candidates. Fitting
    # it first avoids candidate-dependent circularity and lets the resulting
    # ownership field constrain every subsequent segmentation stage.
    body_barrier_info = None
    if body_barrier_ellipsoid:
        body_barrier_fit = fit_soft_barrier_primitive_sampled(
            detection_mesh.v,
            detection_mesh.f,
            config=body_barrier_config,
            require_inside_center=True,
            sample_fraction=body_barrier_sample_fraction,
            random_seed=body_barrier_sample_seed,
        )
        body_barrier_mask = villus_mask_from_barrier_primitive(
            detection_mesh.v,
            body_barrier_fit,
            relative_height_threshold=body_barrier_relative_height_threshold,
        )
        body_barrier_info = {
            "enabled": True,
            "fit_stage": "before_hks_candidate_detection",
            "fit": body_barrier_fit,
            "mask": body_barrier_mask,
            "relative_height_threshold": float(body_barrier_relative_height_threshold),
            "min_candidate_vertices": int(body_barrier_min_candidate_vertices),
            "sample_fraction": float(body_barrier_sample_fraction),
            "sample_seed": body_barrier_sample_seed,
        }

    parents, enc_vars = detect_crypts_by_encoding(
        vocab,
        detection_mesh,
        L_ref=L_ref,
        crypt_vocab_idx=crypt_vocab_idx,
        threshold=threshold,
        return_intermediates=True,
    )

    seg_vars = {
        "encoding": enc_vars.get("encoding"),
        "ts_mesh": enc_vars.get("ts_mesh"),
        "ts_vocab": enc_vars.get("ts_vocab"),
        "hks": enc_vars.get("hks"),
        "norm_hks": enc_vars.get("norm_hks"),
        "hks_segment": enc_vars.get("hks"),
        "normalised_hks_segment": enc_vars.get("norm_hks"),
        "vertex_areas": np.asarray(detection_mesh.vertex_areas(), float),
    }
    if body_barrier_info is not None:
        seg_vars["body_barrier_ellipsoid"] = body_barrier_info
    if filter_fn_list is not None:
        parents, filter_info, keep_idx = apply_filters(
            parents,
            filters=filter_fn_list,
            mesh=detection_mesh,
            seg_vars=seg_vars,
        )
        seg_vars["filter_info_initial"] = filter_info
        seg_vars["keep_idx_initial"] = keep_idx

    if body_barrier_info is not None:
        raw_candidate_sizes = [int(len(patch)) for patch in parents]
        parents, body_barrier_patch_info = protect_patches_from_mask(
            parents,
            body_barrier_info["mask"],
            min_vertices=body_barrier_min_candidate_vertices,
        )
        body_barrier_info.update(
            {
                "raw_candidate_sizes": raw_candidate_sizes,
                "patch_filter_info": body_barrier_patch_info,
            }
        )

    if len(parents) == 0:
        intermediates = {
            "initial_patches": parents,
            "refined_by_parent": [],
            "encoding": enc_vars,
            "detection_mesh_smoothed": bool(smooth_mesh),
            **seg_vars,
        }
        if body_barrier_info is not None:
            intermediates["body_barrier_ellipsoid"] = body_barrier_info
        if return_intermediates:
            return [], intermediates
        return []

    dnorm_parent, L_parent, bottom_parent = compute_crypt_axis(
        detection_mesh,
        parents,
        geodesic_fn,
        geodesic_kwargs=geodesic_kwargs,
    )
    bottom_parent = np.asarray(bottom_parent, dtype=np.int64)
    bottom_info_parent = [
        {
            "strategy": "boundary_distance",
            "bottom_vertex_id": int(bottom),
            "n_patch_vertices": int(len(patch)),
        }
        for bottom, patch in zip(bottom_parent, parents)
    ]
    d_levels = np.linspace(0.01, float(extend_max), int(disc_resolution))
    circumference_parent, dnorm_parent, L_parent = normalize_crypt_axis_to_neckline(
        detection_mesh,
        dnorm_parent,
        d_levels,
        search_interval=neck_search_interval,
        L_crypt=L_parent,
        window_length=neck_window_length,
        polyorder=neck_polyorder,
        min_prominence=neck_min_prominence,
    )
    final_bottom_parent, final_bottom_info_parent = _select_hks_tips_from_axis(
        detection_mesh.v,
        parents,
        dnorm_parent,
        seg_vars.get("hks"),
        seg_vars.get("ts_mesh"),
        bottom_parent,
        hks_time=final_tip_hks_time,
        bottom_fraction=final_tip_bottom_fraction,
        min_hks_percent_increase=final_tip_min_hks_percent_increase,
    )

    detections = []
    refined_by_parent = []
    split_validations = []
    for i, parent in enumerate(parents):
        daughters = []
        if refine_crypts:
            refined = subdivide_crypts_by_encoding(
                vocab,
                detection_mesh,
                L_ref=L_ref,
                crypt_vocab_idx=crypt_vocab_idx,
                patches=[parent],
                threshold=refine_threshold,
                refine_only_if_area_at_least=refine_only_if_area_at_least,
                min_refined_frac_of_parent=min_refined_frac_of_parent,
            )
        else:
            refined = [parent]
        refined_by_parent.append(refined)

        if len(refined) > 1:
            dnorm_child, L_child, bottom_child = compute_crypt_axis(
                detection_mesh,
                refined,
                geodesic_fn,
                geodesic_kwargs=geodesic_kwargs,
            )
            bottom_child = np.asarray(bottom_child, dtype=np.int64)
            bottom_info_child = [
                {
                    "strategy": "boundary_distance",
                    "bottom_vertex_id": int(bottom),
                    "n_patch_vertices": int(len(patch)),
                }
                for bottom, patch in zip(bottom_child, refined)
            ]
            circumference_child, dnorm_child, L_child = normalize_crypt_axis_to_neckline(
                detection_mesh,
                dnorm_child,
                d_levels,
                search_interval=neck_search_interval,
                L_crypt=L_child,
                window_length=neck_window_length,
                polyorder=neck_polyorder,
                min_prominence=neck_min_prominence,
            )
            final_bottom_child, final_bottom_info_child = _select_hks_tips_from_axis(
                detection_mesh.v,
                refined,
                dnorm_child,
                seg_vars.get("hks"),
                seg_vars.get("ts_mesh"),
                bottom_child,
                hks_time=final_tip_hks_time,
                bottom_fraction=final_tip_bottom_fraction,
                min_hks_percent_increase=final_tip_min_hks_percent_increase,
            )
            daughter_union = set().union(*[set(child) for child in refined])
            stem_vertices = sorted(set(parent).difference(daughter_union))
            for j, child in enumerate(refined):
                daughter_detection = {
                        "crypt_id": f"{i}.{j}",
                        "crypt_vertices": child,
                        "boundary_distance_bottom_vertex_id": int(bottom_child[j]),
                        "bottom_vertex_id": int(final_bottom_child[j]),
                        "d_crypt": dnorm_child[j],
                        "L_crypt": float(L_child[j]),
                        "metadata": {
                            "boundary_distance_bottom_selection": bottom_info_child[j],
                            "final_tip_selection": final_bottom_info_child[j],
                        },
                    }
                daughter_detection = _add_neck_profile_geometry(
                    detection_mesh.v,
                    detection_mesh.f,
                    daughter_detection,
                    d_levels,
                    circumference_child[j],
                    relation="branch_crypt",
                    window_length=neck_window_length,
                    polyorder=neck_polyorder,
                    min_prominence=neck_min_prominence,
                    min_neck_length=neck_min_length,
                )
                if refine_broad_crypt_openings:
                    daughter_detection = _refine_broad_transition_opening(
                        detection_mesh,
                        daughter_detection,
                        d_levels,
                        geodesic_fn=geodesic_fn,
                        geodesic_kwargs=geodesic_kwargs,
                        max_opening_to_crypt_body_ratio=max_opening_to_crypt_body_ratio,
                        branch_max_opening_to_crypt_body_ratio=branch_max_opening_to_crypt_body_ratio,
                        min_linear_profile_r2=broad_opening_min_linear_profile_r2,
                        max_linear_profile_deviation=broad_opening_max_linear_profile_deviation,
                        min_attachment_level=broad_opening_min_attachment_level,
                        window_length=neck_window_length,
                        polyorder=neck_polyorder,
                    )
                daughters.append(daughter_detection)
        else:
            stem_vertices = []

        split_validation = {
            "kept_as_split": bool(len(daughters) > 0),
            "reason": "not_refined_split" if not daughters else "split_validation_disabled",
            "neck_position": None,
            "neck_region_vertices": [],
            "final_region_vertices": [],
            "smoothed_region_vertices": [],
            "raw_initial_size": int(len(parent)) if parent is not None else 0,
            "neck_region_size": None,
            "final_region_size": None,
            "smoothed_initial_size": None,
            "initial_size": int(len(parent)) if parent is not None else 0,
            "max_allowed_size": None,
            "max_mesh_fraction": float(split_growth_max_mesh_fraction),
            "mesh_fraction_size_limit": None,
            "perimeter_smoothed": bool(split_growth_smooth_perimeter),
            "perimeter_smoothing_added_vertices": [],
            "perimeter_smoothing_n_added": 0,
            "raw_initial_boundary_length": None,
            "smoothed_initial_boundary_length": None,
            "initial_boundary_length": None,
            "neck_boundary_length": None,
            "boundary_lengths": [],
            "region_sizes": [],
            "minimum_index": None,
            "minimum_boundary_length": None,
            "final_boundary_length": None,
            "min_decrease_fraction": float(split_growth_min_decrease_fraction),
            "min_prominence_fraction": float(split_growth_min_prominence_fraction),
            "robust_window": int(split_growth_robust_window),
        }
        if daughters and validate_split_stems:
            split_validation = _grow_parent_patch_to_neck(
                detection_mesh.v,
                detection_mesh.f,
                parent,
                max_size_factor=split_growth_max_size_factor,
                max_mesh_fraction=split_growth_max_mesh_fraction,
                smooth_perimeter=split_growth_smooth_perimeter,
                smoothing_tolerance=split_growth_smoothing_tolerance,
                min_decrease_fraction=split_growth_min_decrease_fraction,
                min_prominence_fraction=split_growth_min_prominence_fraction,
                robust_window=split_growth_robust_window,
            )
        if daughters and validate_branch_geometry:
            split_validation = _validate_split_branch_geometry(
                detection_mesh.v,
                detection_mesh.f,
                parent,
                daughters,
                split_validation,
                min_confidence=branch_min_confidence,
                max_neck_to_body_radius_ratio=branch_max_neck_to_body_radius_ratio,
            )
        split_validations.append(split_validation)

        if daughters and not split_validation.get("kept_as_split", False):
            for j, daughter in enumerate(daughters):
                daughter_meta = dict(daughter.get("metadata", {}))
                daughter_meta["split_validation"] = {
                    **split_validation,
                    "flattened_from_parent_crypt_id": i,
                    "daughter_index": j,
                }
                flattened = dict(daughter)
                flattened["crypt_id"] = f"{i}.{j}"
                flattened["metadata"] = daughter_meta
                detections.append(flattened)
            continue

        det = {
            "crypt_id": i,
            "crypt_vertices": parent,
            "boundary_distance_bottom_vertex_id": int(bottom_parent[i]),
            "bottom_vertex_id": int(final_bottom_parent[i]),
            "d_crypt": dnorm_parent[i],
            "L_crypt": float(L_parent[i]),
            "metadata": {
                "detection_stage": "fresh_initial_candidate",
                "split_validation": split_validation,
                "boundary_distance_bottom_selection": bottom_info_parent[i],
                "final_tip_selection": final_bottom_info_parent[i],
            },
        }
        if daughters:
            det["daughters"] = daughters
            det["stem_vertices"] = stem_vertices
            validated_position = split_validation.get("neck_position")
            if validated_position is not None:
                det["neck_position"] = validated_position
                validated_region = split_validation.get(
                    "neck_region_vertices",
                    [],
                )
                det["neck_vertices"] = validated_region
                det["neck_region_vertices"] = validated_region
                region_mask = np.zeros(detection_mesh.v.shape[0], dtype=bool)
                region_mask[np.asarray(validated_region, dtype=np.int64)] = True
                boundary_edges = _boundary_edges_for_region(
                    _mesh_edges_from_faces(detection_mesh.f),
                    region_mask,
                )
                boundary_vertices = (
                    np.unique(boundary_edges)
                    if boundary_edges.size
                    else np.empty(0, dtype=np.int64)
                )
                boundary_levels = dnorm_parent[i, boundary_vertices]
                boundary_levels = boundary_levels[np.isfinite(boundary_levels)]
                if boundary_levels.size:
                    current_neck_level = float(np.median(boundary_levels))
                else:
                    nearest_vertex = int(
                        np.argmin(
                            np.linalg.norm(
                                detection_mesh.v
                                - np.asarray(validated_position, dtype=float)[None, :],
                                axis=1,
                            )
                        )
                    )
                    current_neck_level = float(dnorm_parent[i, nearest_vertex])
                if not np.isfinite(current_neck_level):
                    current_neck_level = 1.0
                neck_source = "validated_parent_patch_boundary"
            else:
                current_neck_level = 1.0
                neck_source = "normalized_parent_axis_neck"
            det["body_branch_circumference_levels"] = (
                np.asarray(d_levels, dtype=float) - current_neck_level
            )
            det["body_branch_circumference"] = np.asarray(
                circumference_parent[i],
                dtype=float,
            )
            det["body_branch_current_neck_level"] = current_neck_level
            det["body_branch_neck_position_source"] = neck_source
            det["body_branch_neck_logic"] = "legacy_single_neck"
        else:
            det = _add_neck_profile_geometry(
                detection_mesh.v,
                detection_mesh.f,
                det,
                d_levels,
                circumference_parent[i],
                relation="body_crypt",
                window_length=neck_window_length,
                polyorder=neck_polyorder,
                min_prominence=neck_min_prominence,
                min_neck_length=neck_min_length,
            )
            if refine_broad_crypt_openings:
                det = _refine_broad_transition_opening(
                    detection_mesh,
                    det,
                    d_levels,
                    geodesic_fn=geodesic_fn,
                    geodesic_kwargs=geodesic_kwargs,
                    max_opening_to_crypt_body_ratio=max_opening_to_crypt_body_ratio,
                    branch_max_opening_to_crypt_body_ratio=branch_max_opening_to_crypt_body_ratio,
                    min_linear_profile_r2=broad_opening_min_linear_profile_r2,
                    max_linear_profile_deviation=broad_opening_max_linear_profile_deviation,
                    min_attachment_level=broad_opening_min_attachment_level,
                    window_length=neck_window_length,
                    polyorder=neck_polyorder,
                )
        detections.append(det)

    if refine_body_transition_width_outliers:
        detections = _refine_body_transition_width_outliers(
            detection_mesh,
            detections,
            max_crypt_to_host_width_ratio=body_transition_max_crypt_to_host_width_ratio,
            host_width_quantile=body_transition_host_width_quantile,
            min_second_derivative_score=body_transition_min_second_derivative_score,
            min_attachment_level=body_transition_min_attachment_level,
            window_length=neck_window_length,
            polyorder=neck_polyorder,
        )

    if body_barrier_info is not None:
        crossing_config = dict(barrier_crossing_kwargs or {})
        crossing_config.setdefault("max_axis_level", float(extend_max))
        if barrier_boundary_attachments:
            detections = assign_crypt_attachments_from_barrier_crossings(
                detection_mesh.v,
                detection_mesh.f,
                detections,
                body_barrier_info["fit"],
                crossing_kwargs=crossing_config,
                assign_body_roots=True,
                assign_branch_daughters=False,
            )

        branch_fits, branch_masks, branch_fit_info = fit_branch_barrier_primitives(
            detection_mesh.v,
            detections,
            body_barrier_info["mask"],
            config=branch_barrier_config,
            relative_height_threshold=branch_barrier_relative_height_threshold,
            min_vertices=branch_barrier_min_candidate_vertices,
            sample_fraction=body_barrier_sample_fraction,
            random_seed=body_barrier_sample_seed,
        )
        protected_mask = np.asarray(body_barrier_info["mask"], dtype=bool).copy()
        for branch_mask in branch_masks.values():
            protected_mask |= np.asarray(branch_mask, dtype=bool)

        if barrier_boundary_attachments:
            detections = assign_crypt_attachments_from_barrier_crossings(
                detection_mesh.v,
                detection_mesh.f,
                detections,
                body_barrier_info["fit"],
                branch_fits=branch_fits,
                crossing_kwargs=crossing_config,
                assign_body_roots=False,
                assign_branch_daughters=True,
            )
        body_barrier_info.update(
            {
                "branch_fits": branch_fits,
                "branch_masks": branch_masks,
                "branch_fit_info": branch_fit_info,
                "branch_relative_height_threshold": float(
                    branch_barrier_relative_height_threshold
                ),
                "protected_mask": protected_mask,
                "boundary_attachments_enabled": bool(barrier_boundary_attachments),
                "crossing_config": crossing_config,
            }
        )

    intermediates = {
        "initial_patches": parents,
        "refined_by_parent": refined_by_parent,
        "encoding": enc_vars,
        "d_levels": d_levels,
        "dnorm_parent": dnorm_parent,
        "circumference_parent": circumference_parent,
        "L_parent": L_parent,
        "bottom_parent": bottom_parent,
        "final_bottom_parent": final_bottom_parent,
        "bottom_info_parent": bottom_info_parent,
        "final_bottom_info_parent": final_bottom_info_parent,
        "split_validations": split_validations,
        "detection_mesh_smoothed": bool(smooth_mesh),
        **seg_vars,
    }
    if return_intermediates:
        return detections, intermediates
    return detections

def build_skeleton_from_segmentation_parameters(
    mesh,
    vocab,
    *,
    geodesic_fn,
    build_kwargs: dict[str, Any] | None = None,
    detection_kwargs: dict[str, Any] | None = None,
) -> SkeletonGraph:
    """Convenience wrapper that reruns detection and builds a skeleton."""
    detection_kwargs = dict(detection_kwargs or {})
    build_kwargs = dict(build_kwargs or {})
    detection_kwargs.pop("return_intermediates", None)
    detections = detect_crypts_for_skeleton(
        mesh,
        vocab,
        geodesic_fn=geodesic_fn,
        **detection_kwargs,
    )
    return build_skeleton_from_crypt_detections(
        vertices=mesh.v,
        faces=mesh.f,
        crypt_detections=detections,
        **build_kwargs,
    )

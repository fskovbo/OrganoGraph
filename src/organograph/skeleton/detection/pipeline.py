"""High-level skeletonization pipeline from crypt detection to graph inputs."""

from __future__ import annotations

import numpy as np

from organograph.skeleton.config import DetectionConfig
from organograph.skeleton.detection.branch_validation import _grow_parent_patch_to_neck, _validate_split_branch_geometry
from organograph.skeleton.detection.mesh_regions import _boundary_edges_for_region, _low_pass_smoothed_mesh_for_detection, _mesh_edges_from_faces
from organograph.skeleton.detection.neck_profiles import _add_neck_profile_geometry
from organograph.skeleton.detection.region_refinement import _refine_body_transition_width_outliers
from organograph.skeleton.detection.tips import _select_hks_tips_from_axis
from organograph.skeleton.primitive.barriers import (
    fit_branch_barrier_primitives,
    fit_barrier_primitive_sampled,
    exclude_host_vertices_from_patches,
    host_mask_from_barrier,
)
from organograph.skeleton.detection.attachments import (
    attachment_crossing_diagnostics,
    assign_crypt_attachments_from_barrier_crossings,
)
from organograph.skeleton.results import BarrierStageResult, DetectionResult


def _profile_crypt_patches(
    mesh,
    patches,
    *,
    geodesic_fn,
    geodesic_kwargs,
    d_levels,
    neck_config,
    candidate_config,
    hks,
    hks_times,
):
    """Compute initial tips, circumference-normalized axes, and final HKS tips."""
    from organograph.crypts.axis import compute_crypt_axis, normalize_crypt_axis_to_neckline

    distance_fields, lengths, initial_tips = compute_crypt_axis(
        mesh,
        patches,
        geodesic_fn,
        geodesic_kwargs=geodesic_kwargs,
    )
    initial_tips = np.asarray(initial_tips, dtype=np.int64)
    circumference, distance_fields, lengths = normalize_crypt_axis_to_neckline(
        mesh,
        distance_fields,
        d_levels,
        search_interval=neck_config.search_interval,
        L_crypt=lengths,
        window_length=9,
        polyorder=3,
        min_prominence=neck_config.min_prominence,
    )
    final_tips, final_tip_info = _select_hks_tips_from_axis(
        mesh.v,
        patches,
        distance_fields,
        hks,
        hks_times,
        initial_tips,
        hks_time=candidate_config.final_tip_hks_time,
        bottom_fraction=candidate_config.final_tip_bottom_fraction,
        min_hks_percent_increase=candidate_config.final_tip_min_hks_percent_increase,
    )
    initial_tip_info = [
        {
            "strategy": "boundary_distance",
            "bottom_vertex_id": int(tip),
            "n_patch_vertices": int(len(patch)),
        }
        for tip, patch in zip(initial_tips, patches)
    ]
    return {
        "distance_fields": distance_fields,
        "lengths": lengths,
        "initial_tips": initial_tips,
        "initial_tip_info": initial_tip_info,
        "final_tips": final_tips,
        "final_tip_info": final_tip_info,
        "circumference": circumference,
    }


def detect_crypts_for_skeleton(
    mesh,
    vocab,
    *,
    geodesic_fn,
    config: DetectionConfig | dict | None = None,
) -> DetectionResult:
    """Detect barrier-bounded crypt and branch components.

    The body barrier is fitted before HKS detection. Circumference profiles are
    used only to classify transition versus constriction; host-side component
    boundaries always come from persistent crossings of fitted body or branch
    barriers. This is the sole supported skeletonization path.
    """
    from organograph.crypts.filters import apply_filters
    from organograph.crypts.vocab import detect_crypts_by_encoding, subdivide_crypts_by_encoding

    if not isinstance(config, DetectionConfig):
        config = DetectionConfig(**dict(config or {}))
    candidate_config = config.candidates
    neck_config = config.necks
    branch_config = config.branches
    transition_config = config.body_transition
    barrier_config = config.barriers
    mesh_config = config.mesh

    L_ref = candidate_config.vocab_reference
    crypt_vocab_idx = candidate_config.crypt_vocab_indices
    threshold = candidate_config.threshold
    filter_fn_list = candidate_config.filters
    refine_threshold = candidate_config.refine_threshold
    refine_only_if_area_at_least = candidate_config.refine_min_area
    min_refined_frac_of_parent = candidate_config.min_child_fraction
    geodesic_kwargs = dict(candidate_config.geodesic_kwargs)
    extend_max = neck_config.max_axis_level
    disc_resolution = neck_config.resolution
    neck_window_length = 9
    neck_polyorder = 3
    neck_min_prominence = neck_config.min_prominence
    neck_min_length = neck_config.min_length

    split_growth_smooth_perimeter = True
    split_growth_smoothing_tolerance = 0.0
    split_growth_min_decrease_fraction = 0.0
    split_growth_robust_window = 1

    detection_mesh = (
        _low_pass_smoothed_mesh_for_detection(
            mesh,
            lmax=mesh_config.spectral_lmax,
            recompute_eigen=mesh_config.recompute_eigen,
            eigen_k=mesh_config.eigen_k,
        )
        if mesh_config.smooth
        else mesh
    )

    # The body estimate is deliberately independent of HKS candidates. Fitting
    # it first avoids candidate-dependent circularity and lets the resulting
    # ownership field constrain every subsequent segmentation stage.
    body_barrier_fit = fit_barrier_primitive_sampled(
        detection_mesh.v,
        detection_mesh.f,
        config=barrier_config.body_fit_options,
        require_inside_center=True,
        sample_fraction=barrier_config.sample_fraction,
        random_seed=barrier_config.sample_seed,
    )
    body_barrier_mask = host_mask_from_barrier(
        detection_mesh.v,
        body_barrier_fit,
        relative_height_threshold=barrier_config.body_ownership_level,
    )
    body_barrier_info = {
        "fit_stage": "before_hks_candidate_detection",
        "fit": body_barrier_fit,
        "mask": body_barrier_mask,
        "relative_height_threshold": float(barrier_config.body_ownership_level),
        "min_candidate_vertices": int(barrier_config.min_candidate_vertices),
        "sample_fraction": float(barrier_config.sample_fraction),
        "sample_seed": barrier_config.sample_seed,
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
    seg_vars["body_barriers"] = body_barrier_info
    if filter_fn_list is not None:
        parents, filter_info, keep_idx = apply_filters(
            parents,
            filters=filter_fn_list,
            mesh=detection_mesh,
            seg_vars=seg_vars,
        )
        seg_vars["filter_info_initial"] = filter_info
        seg_vars["keep_idx_initial"] = keep_idx

    raw_candidate_sizes = [int(len(patch)) for patch in parents]
    parents, body_barrier_patch_info = exclude_host_vertices_from_patches(
        parents,
        body_barrier_info["mask"],
        min_vertices=barrier_config.min_candidate_vertices,
    )
    body_barrier_info.update(
        {
            "raw_candidate_sizes": raw_candidate_sizes,
            "patch_filter_info": body_barrier_patch_info,
        }
    )

    if len(parents) == 0:
        diagnostics = {
            "initial_patches": parents,
            "refined_by_parent": [],
            "encoding": enc_vars,
            "detection_mesh_smoothed": bool(mesh_config.smooth),
            **seg_vars,
        }
        barriers = BarrierStageResult(
            body_fit=body_barrier_fit,
            body_mask=body_barrier_mask,
        )
        return DetectionResult(
            detections=[],
            barriers=barriers,
            config=config,
            diagnostics=diagnostics,
            detection_mesh=detection_mesh,
        )

    d_levels = np.linspace(0.01, float(extend_max), int(disc_resolution))
    parent_profiles = _profile_crypt_patches(
        detection_mesh,
        parents,
        geodesic_fn=geodesic_fn,
        geodesic_kwargs=geodesic_kwargs,
        d_levels=d_levels,
        neck_config=neck_config,
        candidate_config=candidate_config,
        hks=seg_vars.get("hks"),
        hks_times=seg_vars.get("ts_mesh"),
    )
    dnorm_parent = parent_profiles["distance_fields"]
    L_parent = parent_profiles["lengths"]
    bottom_parent = parent_profiles["initial_tips"]
    bottom_info_parent = parent_profiles["initial_tip_info"]
    final_bottom_parent = parent_profiles["final_tips"]
    final_bottom_info_parent = parent_profiles["final_tip_info"]
    circumference_parent = parent_profiles["circumference"]

    detections = []
    refined_by_parent = []
    split_validations = []
    for i, parent in enumerate(parents):
        daughters = []
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
        refined_by_parent.append(refined)

        if len(refined) > 1:
            child_profiles = _profile_crypt_patches(
                detection_mesh,
                refined,
                geodesic_kwargs=geodesic_kwargs,
                geodesic_fn=geodesic_fn,
                d_levels=d_levels,
                neck_config=neck_config,
                candidate_config=candidate_config,
                hks=seg_vars.get("hks"),
                hks_times=seg_vars.get("ts_mesh"),
            )
            dnorm_child = child_profiles["distance_fields"]
            L_child = child_profiles["lengths"]
            bottom_child = child_profiles["initial_tips"]
            bottom_info_child = child_profiles["initial_tip_info"]
            final_bottom_child = child_profiles["final_tips"]
            final_bottom_info_child = child_profiles["final_tip_info"]
            circumference_child = child_profiles["circumference"]
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
            "max_mesh_fraction": float(branch_config.max_mesh_fraction),
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
            "min_prominence_fraction": float(branch_config.min_perimeter_prominence_fraction),
            "robust_window": int(split_growth_robust_window),
        }
        if daughters:
            split_validation = _grow_parent_patch_to_neck(
                detection_mesh.v,
                detection_mesh.f,
                parent,
                max_size_factor=branch_config.max_growth_size_factor,
                max_mesh_fraction=branch_config.max_mesh_fraction,
                smooth_perimeter=split_growth_smooth_perimeter,
                smoothing_tolerance=split_growth_smoothing_tolerance,
                min_decrease_fraction=split_growth_min_decrease_fraction,
                min_prominence_fraction=branch_config.min_perimeter_prominence_fraction,
                robust_window=split_growth_robust_window,
            )
        if daughters:
            split_validation = _validate_split_branch_geometry(
                detection_mesh.v,
                detection_mesh.f,
                parent,
                daughters,
                split_validation,
                min_confidence=branch_config.min_confidence,
                max_neck_to_body_radius_ratio=branch_config.max_neck_to_body_radius_ratio,
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
            det["body_branch_neck_logic"] = "validated_branch_boundary"
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
        detections.append(det)

    if transition_config.enabled:
        detections = _refine_body_transition_width_outliers(
            detection_mesh,
            detections,
            max_crypt_to_host_width_ratio=transition_config.max_crypt_to_host_width_ratio,
            host_width_quantile=transition_config.host_width_quantile,
            min_second_derivative_score=transition_config.min_second_derivative_score,
            min_attachment_level=transition_config.min_attachment_level,
            window_length=neck_window_length,
            polyorder=neck_polyorder,
        )

    crossing_config = barrier_config.crossing_kwargs()
    detections = assign_crypt_attachments_from_barrier_crossings(
        detection_mesh.v,
        detection_mesh.f,
        detections,
        body_barrier_fit,
        crossing_kwargs=crossing_config,
        assign_body_roots=True,
        assign_branch_daughters=False,
    )

    branch_fits, branch_masks, branch_fit_info = fit_branch_barrier_primitives(
        detection_mesh.v,
        detections,
        body_barrier_mask,
        config=barrier_config.branch_fit_options,
        relative_height_threshold=barrier_config.branch_ownership_level,
        min_vertices=barrier_config.min_branch_vertices,
        sample_fraction=barrier_config.sample_fraction,
        random_seed=barrier_config.sample_seed,
    )
    protected_mask = np.asarray(body_barrier_mask, dtype=bool).copy()
    for branch_mask in branch_masks.values():
        protected_mask |= np.asarray(branch_mask, dtype=bool)

    detections = assign_crypt_attachments_from_barrier_crossings(
        detection_mesh.v,
        detection_mesh.f,
        detections,
        body_barrier_fit,
        branch_fits=branch_fits,
        crossing_kwargs=crossing_config,
        assign_body_roots=False,
        assign_branch_daughters=True,
    )
    boundary_detections = detections
    crossing_diagnostics = attachment_crossing_diagnostics(boundary_detections)
    body_barrier_info.update(
        {
            "branch_fits": branch_fits,
            "branch_masks": branch_masks,
            "branch_fit_info": branch_fit_info,
            "branch_relative_height_threshold": float(
                barrier_config.branch_ownership_level
            ),
            "protected_mask": protected_mask,
            "crossing_config": crossing_config,
            "post_crossing_host_vertex_exclusion": False,
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
        "detection_mesh_smoothed": bool(mesh_config.smooth),
        "boundary_detections": boundary_detections,
        "attachment_crossings": crossing_diagnostics,
        "attachment_crossing_failures": [
            record for record in crossing_diagnostics if not record.get("found", False)
        ],
        **seg_vars,
    }
    barriers = BarrierStageResult(
        body_fit=body_barrier_fit,
        body_mask=body_barrier_mask,
        branch_fits=branch_fits,
        branch_masks=branch_masks,
        protected_mask=protected_mask,
        branch_diagnostics=branch_fit_info,
    )
    return DetectionResult(
        detections=detections,
        barriers=barriers,
        config=config,
        diagnostics=intermediates,
        detection_mesh=detection_mesh,
    )

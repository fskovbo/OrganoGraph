import numpy as np

from organograph.crypts.vocab import (
    detect_crypts_by_encoding,
    subdivide_crypts_by_encoding,
)
from organograph.crypts.axis import (
    compute_crypt_axis,
    normalize_crypt_axis_to_neckline,
    assign_features_by_distance
)
from organograph.crypts.filters import apply_filters, subset_per_crypt_vars
from organograph.crypts.analysis import calc_crypt_constriction, calc_crypt_elongation

def segment_crypts_organoid(
    mesh,                               # organoid surface mesh (mesh.v, mesh.f, mesh.vertex_areas())
    vocab,                              # bag_of_features dict (stores defaults for L_ref + vocab indices)
    *,
    # Crypt seeding variables
    L_ref=None,                         # OPTIONAL override (vocab may already store a default)
    crypt_vocab_idx=None,               # OPTIONAL override (vocab functions may already have defaults)
    threshold=0.5,                      # vertex-score threshold for crypt detection
    # Crypt refinement/subdivision variables
    refine_crypts=True,                 # if False, skip refinement and use round-1 patches as final patches
    refine_threshold=0.0,               # threshold used during subdivision step
    refine_only_if_area_at_least=5.0,   # skip refinement if patch area smaller than this (None disables)
    min_refined_frac_of_parent=0.1,     # keep only refined patches with number of verts above a fraction of its parent
    # Neck search variables
    geodesic_fn=None,                   # callable to compute distances (e.g. compute_geodesics_dijkstra)
    geodesic_kwargs=None,               # kwargs forwarded to geodesic_fn
    extend_max=2.0,                     # max normalized axis for neck search
    disc_resolution=200,                # discretization of crypt axis for circumference calculation
    remove_nested_features=True,        # removes crypts nested inside other crypts
    # Filters
    filter_fn_list=None,                # list of filter callables to apply
    return_vars=False,                  # if True, also return intermediates for debugging/plotting
):
    if geodesic_fn is None:
        raise ValueError("geodesic_fn must be provided (e.g. compute_geodesics_dijkstra).")
    if geodesic_kwargs is None:
        geodesic_kwargs = {}

    seg_vars = {}

    # Cache vertex areas for filters
    seg_vars["vertex_areas"] = np.asarray(mesh.vertex_areas(), float)

    # --- Round 1 detection ---
    crypts, enc_vars = detect_crypts_by_encoding(
        vocab,
        mesh,
        L_ref=L_ref,
        crypt_vocab_idx=crypt_vocab_idx,
        threshold=threshold,
        return_intermediates=True,
    )

    seg_vars["encoding"] = enc_vars["encoding"]
    seg_vars["ts_mesh"] = enc_vars["ts_mesh"]
    seg_vars["ts_vocab"] = enc_vars["ts_vocab"]
    seg_vars["hks_segment"] = enc_vars["hks"]
    seg_vars["normalised_hks_segment"] = enc_vars["norm_hks"]

    # --- Filters after detection ---
    crypts, filt_info_initial, _keep_idx_initial = apply_filters(
        crypts,
        filters=filter_fn_list,
        mesh=mesh,
        seg_vars=seg_vars,
    )
    seg_vars["filter_info_initial"] = filt_info_initial

    # --- Optional refinement ---
    if refine_crypts:
        sub_crypts = subdivide_crypts_by_encoding(
            vocab,
            mesh,
            L_ref=L_ref,
            crypt_vocab_idx=crypt_vocab_idx,
            patches=crypts,
            threshold=refine_threshold,
            refine_only_if_area_at_least=refine_only_if_area_at_least,
            min_refined_frac_of_parent=min_refined_frac_of_parent,
        )
    else:
        sub_crypts = crypts

    # --- Axis + neckline normalize + extend ---
    dnorm_all, L_crypt_all, bottom_vertex_ids = compute_crypt_axis(
        mesh, sub_crypts, geodesic_fn, geodesic_kwargs=geodesic_kwargs
    )

    d_discretized = np.linspace(0.01, float(extend_max), int(disc_resolution))
    search_interval = (0.8, d_discretized[-1])

    CC_all, dnorm_all, L_crypt_all = normalize_crypt_axis_to_neckline(
        mesh,
        dnorm_all,
        d_discretized,
        search_interval=search_interval,
        L_crypt=L_crypt_all,
    )

    # --- Final crypt assignment / extension ---
    # Small crypt candidates whose center lies within the extent of a larger
    # crypt are removed before growing.
    crypts_extended, best_feature, best_dist, keep_idx_merge = assign_features_by_distance(
        dnorm_all,
        remove_nested_features=remove_nested_features,
    )

    # --- Coompute some lightweight descriptors of crypt shape
    constrictions = calc_crypt_constriction(d_discretized, CC_all)
    elongations = calc_crypt_elongation(d_discretized, CC_all, L_crypt_all)

    # keep_idx_global always refers to rows of the current per-crypt arrays
    keep_idx_global = np.asarray(keep_idx_merge, dtype=np.int64)

    # --- Filters after final growth ---
    crypts_extended, filt_info_final, keep_idx_final = apply_filters(
        crypts_extended,
        filters=filter_fn_list,
        mesh=mesh,
        seg_vars=seg_vars,
    )
    seg_vars["filter_info_final"] = filt_info_final

    keep_idx_global = keep_idx_global[np.asarray(keep_idx_final, dtype=np.int64)]

    # --- Align all per-crypt arrays once, at the end ---
    aligned = subset_per_crypt_vars(
        keep_idx_global,
        bottom_vertex_ids=bottom_vertex_ids,
        dnorm_all=dnorm_all,
        L_crypt_all=L_crypt_all,
        CC_all=CC_all,
        constrictions=constrictions,
        elongations=elongations,
    )

    if not return_vars:
        return crypts_extended

    seg_vars.update({
        "bottom_vertex_ids": aligned["bottom_vertex_ids"],
        "d_crypts": aligned["dnorm_all"],
        "L_crypts": aligned["L_crypt_all"],
        "circumference_crypts": aligned["CC_all"],
        "d_discretized": d_discretized,
        "keep_idx_final": keep_idx_global,
        "crypt_constrictions": aligned["constrictions"],
        "crypt_elongations": aligned["elongations"],
    })
    return crypts_extended, seg_vars
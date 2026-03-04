import numpy as np

from organograph.crypts.vocab import (
    detect_crypts_by_encoding,
    subdivide_crypts_by_encoding,
)
from organograph.crypts.axis import (
    compute_crypt_axis,
    normalize_crypt_axis_to_neckline,
)
from organograph.crypts.features import assign_features_by_distance


def apply_filters(patches, *, filters, mesh, seg_vars):
    infos = []
    out = patches
    for f in (filters or []):
        out, info = f(out, mesh=mesh, seg_vars=seg_vars)
        infos.append(info)
    return out, infos


def segment_crypts_organoid(
    mesh,                               # organoid surface mesh (mesh.v, mesh.f, mesh.vertex_areas())
    vocab,                              # bag_of_features dict (stores defaults for L_ref + vocab indices)
    *,
    # Crypt seeding variables
    L_ref=None,                         # OPTIONAL override (vocab may already store a default)
    crypt_vocab_idx=None,               # OPTIONAL override (vocab functions may already have defaults)
    threshold=0.5,                      # vertex-score threshold for crypt detection
    min_patch_verts=25,                 # min vertices for detected (and refined) patches
    min_patch_area=None,                # optional area filter for detection/refinement
    # Crypt refinement/subdivision variables
    refine_crypts=True,                 # if False, skip refinement and use round-1 patches as final patches
    refine_threshold=0.0,               # threshold used during subdivision step
    refine_only_if_verts_at_least=0,    # skip refinement if patch has fewer verts
    refine_only_if_area_at_least=5.0,   # skip refinement if patch area smaller than this (None disables)
    # Neck search variables
    geodesic_fn=None,                   # callable to compute distances (e.g. compute_geodesics_dijkstra)
    geodesic_kwargs=None,               # kwargs forwarded to geodesic_fn
    extend_max=2.5,                     # max normalized axis for neck search
    # Filters
    filter_fn_list=None,                # list of filter callables to apply 
    return_vars=False,                  # if True, also return intermediates for debugging/plotting
):
    if geodesic_fn is None:
        raise ValueError("geodesic_fn must be provided (e.g. compute_geodesics_dijkstra).")
    if geodesic_kwargs is None:
        geodesic_kwargs = {}

    seg_vars = {}


    # --- Round 1 detection ---
    crypts, enc_vars = detect_crypts_by_encoding(
        vocab, 
        mesh,
        L_ref=L_ref, 
        crypt_vocab_idx=crypt_vocab_idx,
        threshold=threshold, 
        min_patch_verts=min_patch_verts, 
        min_patch_area=min_patch_area, 
        return_intermediates=True
    )

    seg_vars["encoding"] = enc_vars["encoding"]
    seg_vars["ts_mesh"] = enc_vars["ts_mesh"]
    seg_vars["ts_vocab"] = enc_vars["ts_vocab"]
    seg_vars["hks"] = enc_vars["hks"]
    seg_vars["normalised_hks"] = enc_vars["norm_hks"]


    # --- Filters after detection ---
    crypts, filt_info = apply_filters(
        crypts, filters=filter_fn_list, mesh=mesh, seg_vars=seg_vars,
    )
    seg_vars["filterer_info"] = filt_info


    # --- Optional refinement ---
    if refine_crypts:
        sub_crypts = subdivide_crypts_by_encoding(
            vocab, 
            mesh,
            L_ref=L_ref,
            crypt_vocab_idx=crypt_vocab_idx,
            patches=crypts,
            threshold=refine_threshold,
            min_patch_verts=min_patch_verts,     
            min_patch_area=min_patch_area,
            refine_only_if_verts_at_least=refine_only_if_verts_at_least,
            refine_only_if_area_at_least=refine_only_if_area_at_least,
        )
    else:
        sub_crypts = crypts


    # --- Axis + neckline normalize + extend ---
    dnorm_all, L_crypt_all, bottom_vertex_ids = compute_crypt_axis(
        mesh, sub_crypts, geodesic_fn, geodesic_kwargs=geodesic_kwargs
    )

    n_bins = 100
    bin_edges = np.linspace(0.0, float(extend_max), int(n_bins) + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    search_interval = (0.8, bin_centers[-1])

    CC_all, dnorm_all, L_crypt_all = normalize_crypt_axis_to_neckline(
        mesh, dnorm_all, bin_centers, search_interval=search_interval, L_crypt=L_crypt_all
    )

    crypts_extended, best_feature, best_dist = assign_features_by_distance(dnorm_all)

    if not return_vars:
        return crypts_extended

    seg_vars.update({
        "bottom_vertex_ids": bottom_vertex_ids,
        "d_crypts": dnorm_all,
        "L_crypts": L_crypt_all,
        "circumference_crypts": CC_all,
        "bin_centers": bin_centers,
        "best_feature": best_feature,
        "best_dist": best_dist,
    })
    return crypts_extended, seg_vars




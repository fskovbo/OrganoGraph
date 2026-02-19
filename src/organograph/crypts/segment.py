import numpy as np

from organograph.crypts.features import seed_features_by_vocab, grow_crypts_toward_necks, assign_features_by_distance
from organograph.crypts.axis import calc_crypt_axis, normalize_crypt_axis_to_neckline
from organograph.mesh.geodesics import compute_geodesics_dijkstra
from organograph.graph.access import graph_get

def segment_crypts_organoid(
    G,                     # networkx cell-graph
    mesh,                  # OrganoidMesh 
    bin_centers,           # (B,) axis used for circumference curves + rescaling
    crypt_vocab_idx,       # iterable[int] indices into vocab_encoding indicating "crypt" vocab bins
    neck_vocab_idx=None,   # iterable[int] or None; indices indicating "neck" vocab bins
    crypt_seed_thresh=0.2, # threshold on max(vocab_encoding[crypt_vocab_idx]) to seed crypt
    neck_seed_thresh=0.5,  # threshold on max(vocab_encoding[neck_vocab_idx]) to seed neck
    min_crypt_seed_size=10,# minimum connected-component size for crypt (and neck if enabled)
    grow_steps=2,          # iterations for grow_crypts_toward_necks; set 0 to disable
    geodesic_fn=compute_geodesics_dijkstra,  # geodesics(mesh, sources=[...], **geodesic_kwargs)
    geodesic_kwargs=None,  # dict of kwargs forwarded to geodesic_fn
    search_interval=(0.75, 1.25),  # allowed stretch range for finding min circumference
    window_length=9,       # smoothing window for adjust_cryptlength_by_circumference
    polyorder=3,           # polynomial order for adjust_cryptlength_by_circumference smoothing
    min_prominence=0.05,   # peak prominence threshold for adjust_cryptlength_by_circumference
    debug=False,           # if True, also return intermediate regions + internals
):
    """
    One-stop organoid segmentation + crypt-axis computation.

    Pipeline
    --------
    1) seed crypt/neck/villus regions from vocab_encoding
    2) optionally grow crypts toward necks
    3) compute crypt axis (raw + normalized) using geodesics from crypt bottoms
    4) rescale axis by circumference so minimum aligns with s=1 on bin_centers
    5) map vertex distances to cell-center distances
    6) finalize crypt membership by neckline (s_thresh=1.0)
    7) villus_final = all cells not assigned to any crypt

    Returns
    -------
    crypts_final : list[set[int]]
    villi_final  : list[set[int]]          # single patch: all non-crypt cells
    d_norm       : (K, V) float            # rescaled normalized distances at vertices
    L_crypt      : (K,) float              # mean boundary distance (rescaled consistently with dnorm_v)
    C            : (K, B) float            # circumference curves aligned to bin_centers
    dbg          : dict (only if debug=True)
    """
    if geodesic_kwargs is None:
        geodesic_kwargs = {}

    # --- 1) seed regions ---
    crypts_seed, necks_seed, villi_seed = seed_features_by_vocab(
        G,
        crypt_vocab_idx=crypt_vocab_idx,
        crypt_thresh=crypt_seed_thresh,
        min_crypt_region_size=min_crypt_seed_size,
        neck_vocab_idx=neck_vocab_idx,
        neck_thresh=neck_seed_thresh,
    )

    # --- 2) optionally grow/refine ---
    if grow_steps and grow_steps > 0:
        crypts_grow, necks_grow, villi_grow = grow_crypts_toward_necks(
            G,
            crypt_regions=crypts_seed,
            neck_regions=necks_seed,
            N=grow_steps,
            min_villus_region_size=1,  # fixed default
        )
    else:
        crypts_grow, necks_grow, villi_grow = crypts_seed, necks_seed, villi_seed

    boundary_patches = villi_grow + necks_grow

    # --- 3) compute crypt axis (raw + normalized) ---
    d_norm, L_crypt, bottom_vertex_ids = calc_crypt_axis(
        G=G,
        mesh=mesh,
        crypt_patches=crypts_grow,
        boundary_patches=boundary_patches,
        geodesic_fn=geodesic_fn,
        geodesic_kwargs=geodesic_kwargs,
    )

    # --- 4) rescale by circumference (always returns (K,B), (K,V), (K,)) ---
    Circ, d_norm, L_crypt = normalize_crypt_axis_to_neckline(
        mesh=mesh,
        dnorm_vertices=d_norm,
        bin_centers=bin_centers,
        search_interval=search_interval,
        L_crypt=L_crypt,
        window_length=window_length,
        polyorder=polyorder,
        min_prominence=min_prominence,
    )

    # --- 5) vertex -> cell center distances ---
    proj_vertex_ids = graph_get(G, "proj_vertex", dtype=np.int32)
    dnorm_c = d_norm[:, proj_vertex_ids]  # (K, N_cells)

    # --- 6) finalize crypt membership by neckline (default s_thresh=1.0) ---
    crypts_final, best_feature, best_dist = assign_features_by_distance(dnorm_c)

    # --- 7) villus = all cells not in any crypt (single patch) ---
    N_cells = G.number_of_nodes()
    crypt_union = set().union(*crypts_final) if crypts_final else set()
    villus_cells = set(range(N_cells)) - crypt_union
    villi_final = [villus_cells] if villus_cells else []

    if not debug:
        return crypts_final, villi_final, d_norm, L_crypt, Circ

    dbg = {
        "crypts_seed": crypts_seed,
        "necks_seed": necks_seed,
        "villi_seed": villi_seed,
        "crypts_grow": crypts_grow,
        "necks_grow": necks_grow,
        "villi_grow": villi_grow,
        "boundary_patches": boundary_patches,
        "bottom_vertex_ids": bottom_vertex_ids,
        "best_feature": best_feature,  # (N_cells,)
        "best_dist": best_dist,        # (N_cells,)
    }

    return crypts_final, villi_final, d_norm, L_crypt, Circ, dbg
import numpy as np
from collections import deque


def seed_features_by_vocab(
    G,
    crypt_vocab_idx,
    crypt_thresh=0.9,
    min_crypt_region_size=5,
    min_villus_region_size=1,
    neck_vocab_idx=None,
    neck_thresh=0.9,
):
    """
    Segment an organoid graph into crypt / (optional neck) / villus regions using only
    the 'vocab_encoding' node attribute.

    A node is:
      - crypt-positive if max(vocab_encoding[crypt_vocab_idx]) >= crypt_thresh
      - neck-positive  if (neck_vocab_idx is not None) and
                        max(vocab_encoding[neck_vocab_idx])  >= neck_thresh

    Precedence:
      crypt wins over neck (i.e. neck_mask excludes crypt nodes).

    Returns
    -------
    crypt_regions : list[set[int]]
    neck_regions  : list[set[int]]   # empty if neck_vocab_idx is None
    villus_regions: list[set[int]]   # everything not crypt nor neck
    """

    # Assume nodes are 0..N-1 as in your graph-building code
    nodes = sorted(G.nodes())
    N = len(nodes)

    # Safety check: we rely on node ids as indices
    if nodes != list(range(N)):
        raise ValueError("seed_regions_by_vocab assumes nodes are 0..N-1.")

    # Collect vocab encodings into an (N, P) array
    vocab_enc = np.stack(
        [np.asarray(G.nodes[n]["vocab_encoding"], dtype=float) for n in nodes],
        axis=0,
    )

    crypt_vocab_idx = np.asarray(list(crypt_vocab_idx), dtype=int)
    if crypt_vocab_idx.ndim != 1:
        raise ValueError("crypt_vocab_idx must be a 1D iterable of indices.")

    # --- Helper: connected components restricted to mask == True ---
    def _find_regions_from_mask(mask, min_region_size):
        regions = []
        visited = np.zeros(N, dtype=bool)

        for i in range(N):
            if not mask[i] or visited[i]:
                continue

            comp = set()
            stack = [i]
            while stack:
                u = stack.pop()
                if visited[u] or not mask[u]:
                    continue
                visited[u] = True
                comp.add(u)
                for v in G.neighbors(u):
                    if (not visited[v]) and mask[v]:
                        stack.append(v)

            if len(comp) >= min_region_size:
                regions.append(comp)

        return regions

    # --- 1) Crypt mask ---
    crypt_scores = vocab_enc[:, crypt_vocab_idx].max(axis=1)   # (N,)
    crypt_mask = crypt_scores >= crypt_thresh                  # boolean (N,)

    # --- 2) Optional neck mask ---
    neck_mask = np.zeros(N, dtype=bool)
    if neck_vocab_idx is not None:
        neck_vocab_idx = np.asarray(list(neck_vocab_idx), dtype=int)
        if neck_vocab_idx.ndim != 1:
            raise ValueError("neck_vocab_idx must be a 1D iterable of indices.")
        neck_scores = vocab_enc[:, neck_vocab_idx].max(axis=1)
        neck_mask = (neck_scores >= neck_thresh)

        # precedence: crypt overrides neck
        neck_mask = neck_mask & (~crypt_mask)

    # --- 3) Regions ---
    crypt_regions = _find_regions_from_mask(crypt_mask, min_crypt_region_size)
    neck_regions = _find_regions_from_mask(neck_mask, min_crypt_region_size) if neck_vocab_idx is not None else []

    # --- 4) Villus: everything not crypt nor neck (if necks enabled) ---
    occupied = crypt_mask | neck_mask
    villus_mask = ~occupied
    villus_regions = _find_regions_from_mask(villus_mask, min_villus_region_size)

    return crypt_regions, neck_regions, villus_regions



def assign_features_by_distance(dnorm_per_feature, s_thresh=1.0):
    """
    Assign each item (cell, vertex, etc.) to at most one feature using
    a normalized-distance threshold and nearest-feature rule.

    Rule
    ----
    An item i is eligible for feature k if:
        dnorm_per_feature[k, i] < s_thresh

    If multiple features qualify, the item is assigned to the feature
    with the smallest distance.

    This works identically whether “items” are:
      - cells  → distances at cell centers
      - vertices → distances at mesh vertices
      - any other indexed objects

    Parameters
    ----------
    dnorm_per_feature : array, shape (K, N_items)
        Normalized distances from each feature k to each item i.
        Example:
            K = number of crypts/features
            N_items = number of cells OR vertices
        Distances should already include any axis rescaling (e.g. / s_star).
    s_thresh : float
        Threshold for membership (default 1.0).

    Returns
    -------
    feature_patches : list[set[int]]
        Disjoint sets of assigned item indices, one set per feature (length K).
    best_feature : (N_items,) int
        Assigned feature index per item, or -1 if unassigned.
    best_dist : (N_items,) float
        Winning (smallest) distance per item, or +inf if unassigned.
    """
    D = np.asarray(dnorm_per_feature, dtype=float)
    if D.ndim != 2:
        raise ValueError("dnorm_per_feature must have shape (K, N_items)")

    K, N_items = D.shape

    best_dist = np.full(N_items, np.inf, dtype=float)
    best_feature = np.full(N_items, -1, dtype=int)

    for k in range(K):
        dk = D[k]
        mask = np.isfinite(dk) & (dk < float(s_thresh)) & (dk < best_dist)
        best_dist[mask] = dk[mask]
        best_feature[mask] = k

    feature_patches = [
        set(np.where(best_feature == k)[0].tolist())
        for k in range(K)
    ]

    return feature_patches, best_feature, best_dist



def grow_crypts_toward_necks(
    G,
    crypt_regions,
    neck_regions=None,
    N=3,
    min_villus_region_size=1,
):
    """
    Typically needed after segmenting using vocab.
    
    1) For each crypt region, check if there are neck cells within <= N hops,
       where the path can traverse only villus/unassigned cells (NOT neck, NOT any crypt).
       If yes, grow that crypt outward by up to N hops, only over villus/unassigned cells.
       (Never grow into neck or any crypt.)

    2) Reassign villus as all nodes that are not crypt nor neck, and return villus
       connected components.

    Parameters
    ----------
    G : networkx.Graph
    crypt_regions : list[set[int]]
    neck_regions : list[set[int]] or None
        If None or empty, neck set is treated as empty.
    N : int
        Hop radius for "neck nearby" and growth.
    min_villus_region_size : int
        Minimum size of villus connected components to return.

    Returns
    -------
    crypt_regions_grown : list[set[int]]
    neck_regions : list[set[int]]
        Returned unchanged (or empty list if None provided).
    villus_regions : list[set[int]]
        Recomputed after crypt growth.
    """

    # --- normalize necks ---
    if neck_regions is None:
        neck_regions = []
    neck_nodes = set().union(*neck_regions) if len(neck_regions) else set()

    # --- global crypt occupancy (to prevent overlap across crypts) ---
    crypt_owner = {}
    for ci, reg in enumerate(crypt_regions):
        for u in reg:
            crypt_owner[u] = ci
    all_crypt_nodes = set(crypt_owner.keys())

    crypt_regions_grown = [set(reg) for reg in crypt_regions]

    # --- helper: connected components from a boolean mask ---
    nodes = sorted(G.nodes())
    Nnodes = len(nodes)
    if nodes != list(range(Nnodes)):
        # You can relax this if your graph node ids aren't 0..N-1, but then
        # you'd want a node->index mapping. Keeping it strict like your pipeline.
        raise ValueError("This function assumes nodes are 0..N-1.")

    def _regions_from_mask(mask, min_size):
        visited = [False] * Nnodes
        out = []
        for i in range(Nnodes):
            if not mask[i] or visited[i]:
                continue
            comp = set()
            stack = [i]
            while stack:
                u = stack.pop()
                if visited[u] or not mask[u]:
                    continue
                visited[u] = True
                comp.add(u)
                for v in G.neighbors(u):
                    if (not visited[v]) and mask[v]:
                        stack.append(v)
            if len(comp) >= min_size:
                out.append(comp)
        return out

    # --- grow crypts toward nearby necks ---
    for ci, crypt in enumerate(crypt_regions_grown):
        other_crypt_nodes = all_crypt_nodes - crypt

        dist = {}
        q = deque()

        for s in crypt:
            dist[s] = 0
            q.append(s)

        candidates = set()
        neck_within_N = False

        while q:
            u = q.popleft()
            du = dist[u]
            if du == N:
                continue

            for v in G.neighbors(u):
                if v in dist:
                    continue

                # detect neck (but never traverse into it)
                if v in neck_nodes:
                    neck_within_N = True
                    continue

                # never traverse/grow into any crypt node
                if v in all_crypt_nodes:
                    continue

                # otherwise villus/unassigned => allowed
                dist[v] = du + 1
                q.append(v)
                candidates.add(v)

        if neck_within_N and candidates:
            # ensure we don't overlap forbidden (should already be true, but keep safe)
            candidates -= neck_nodes
            candidates -= other_crypt_nodes

            # apply growth
            crypt |= candidates

            # update global occupancy so later crypts can't claim these nodes
            for v in candidates:
                all_crypt_nodes.add(v)
                crypt_owner[v] = ci

    # --- recompute villus as everything not crypt nor neck ---
    crypt_nodes_final = set().union(*crypt_regions_grown) if len(crypt_regions_grown) else set()
    villus_mask = [((i not in crypt_nodes_final) and (i not in neck_nodes)) for i in range(Nnodes)]
    villus_regions = _regions_from_mask(villus_mask, min_villus_region_size)

    return crypt_regions_grown, neck_regions, villus_regions
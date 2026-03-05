import numpy as np
from collections import deque
from organograph.mesh.hks import compute_hks


# =============================================================================
# Helpers 
# =============================================================================

def _validate_v_subset(v_subset, V: int) -> np.ndarray:
    if v_subset is None:
        return np.arange(V, dtype=np.int64)

    arr = np.asarray(v_subset)
    if arr.ndim != 1:
        raise ValueError(f"v_subset must be 1D (got shape {arr.shape}).")

    if not np.issubdtype(arr.dtype, np.integer):
        try:
            arr_int = arr.astype(np.int64)
        except Exception as e:
            raise TypeError("v_subset must contain integer indices.") from e
        if not np.all(arr_int == arr):
            raise TypeError("v_subset must contain integer indices (no floats).")
        arr = arr_int
    else:
        arr = arr.astype(np.int64, copy=False)

    if arr.size == 0:
        raise ValueError("v_subset must be non-empty (or use None for all vertices).")

    mn = int(arr.min())
    mx = int(arr.max())
    if mn < 0 or mx >= V:
        raise IndexError(f"v_subset indices must be in [0, {V-1}] (got min={mn}, max={mx}).")

    uniq = np.unique(arr)
    if uniq.size != arr.size:
        vals, counts = np.unique(arr, return_counts=True)
        dups = vals[counts > 1][:10]
        raise ValueError(f"v_subset contains duplicate indices (e.g. {dups.tolist()}).")

    return arr


def _vertex_adjacency_from_faces(n_verts: int, faces: np.ndarray):
    faces = np.asarray(faces, dtype=np.int64)
    adj_sets = [set() for _ in range(n_verts)]
    for a, b, c in faces:
        adj_sets[a].add(b); adj_sets[a].add(c)
        adj_sets[b].add(a); adj_sets[b].add(c)
        adj_sets[c].add(a); adj_sets[c].add(b)
    return [list(s) for s in adj_sets]


def _connected_components_from_mask(adj, mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    V = int(mask.size)
    visited = np.zeros(V, dtype=bool)
    comps = []

    for v in range(V):
        if not mask[v] or visited[v]:
            continue
        q = deque([v])
        visited[v] = True
        comp = {int(v)}
        while q:
            x = q.popleft()
            for y in adj[x]:
                if mask[y] and not visited[y]:
                    visited[y] = True
                    q.append(y)
                    comp.add(int(y))
        comps.append(comp)
    return comps


def _patch_area(vertex_areas: np.ndarray, patch_set: set) -> float:
    idx = np.fromiter(patch_set, dtype=np.int64)
    return float(np.sum(vertex_areas[idx]))


def _keep_patch(patch_set: set, *, min_verts: int, vertex_areas=None, min_area=None) -> bool:
    if len(patch_set) < int(min_verts):
        return False
    if min_area is not None:
        if vertex_areas is None:
            raise ValueError("min_area was provided but vertex_areas is None.")
        return _patch_area(vertex_areas, patch_set) >= float(min_area)
    return True


# =============================================================================
# Encoding and classification
# =============================================================================


def compute_vocabulary_encoding(
    bag_of_features: dict,
    mesh,
    *,
    L_ref=None,
    v_subset=None,
    return_hks: bool = False,
):
    """
    Compute vocab encoding for either all vertices or a subset.

    - Uses timescales stored in bag_of_features['ts'] (vocab reference times)
    - Rescales times by (L_mesh/L_ref)^2 where L_mesh = sqrt(area(subset))
    - Computes HKS on the *full* mesh at those rescaled times (so it matches your compute_hks signature)
    - Normalizes using the subset mean (your "meanratio") then applies the stored scaler
    - Returns:
        encoding_sub : (|subset|, M)  if v_subset is not None else (V, M)
        ts_mesh      : rescaled time vector (T,)
      Optionally returns normalized_hks_sub and/or full hks if you want.

    Requirements inside bag_of_features:
      - 'vocab': (M,T)
      - 'scaler': sklearn-like transformer; may be stored as 0-d object array (use .item())
      - 'sigma': float
      - 'ts': (T,)
    """
    vocab = np.asarray(bag_of_features["vocab"], float)
    ts_vocab = np.asarray(bag_of_features["ts"], float)
    sigma = float(bag_of_features["sigma"])

    scaler = bag_of_features["scaler"]
    # common pattern: saved as 0-d object array
    if hasattr(scaler, "item") and not hasattr(scaler, "transform"):
        scaler = scaler.item()

    V = len(mesh.v)
    v_subset = _validate_v_subset(v_subset, V)

    # time rescaling using subset area
    vertex_areas = np.asarray(mesh.vertex_areas())
    L_mesh = float(np.sqrt(np.sum(vertex_areas[v_subset])))

    # --- obtain L_ref ---
    if "L_ref" in bag_of_features:
        L_ref_val = float(bag_of_features["L_ref"])
    elif L_ref is not None:
        L_ref_val = float(L_ref)
    else:
        # no re-scaling of time axis
        L_ref_val = L_mesh

    scale = (L_mesh / float(L_ref_val)) ** 2
    ts_mesh = ts_vocab * scale

    # HKS on full mesh, then subset out
    hks_full = compute_hks(mesh, ts_mesh, coeffs=False)  # (V,T)
    hks_sub = hks_full[v_subset, :]                        # (|S|,T)

    # subset meanratio normalization
    normalised_hks_sub = hks_sub / np.mean(hks_sub, axis=0, keepdims=True) - 1.0
    normalised_hks_sub = scaler.transform(normalised_hks_sub)

    # encoding (|S|, M)
    if vocab.shape[1] != normalised_hks_sub.shape[1]:
        raise ValueError(
            f"vocab has T={vocab.shape[1]} but HKS features have T={normalised_hks_sub.shape[1]}"
        )
    dist = np.linalg.norm(
        normalised_hks_sub[:, None, :] - vocab[None, :, :],
        axis=2
    )
    encoding_sub = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))

    if return_hks:
        return encoding_sub, ts_mesh, ts_vocab, hks_sub, normalised_hks_sub

    return encoding_sub, ts_mesh, ts_vocab



def crypt_vertex_score(
    encoding: np.ndarray,          # (V,M) encoding per vertex against vocabulary features
    crypt_idx,                     # indices of vocabulary entries representing crypt-like features
    noncrypt_idx,                  # indices of vocabulary entries representing non-crypt features
    *,
    eps=1e-12                      # small constant to avoid division by zero
) -> np.ndarray:
    """
    Compute a scalar 'cryptness' score for each vertex.

    The score compares the strongest crypt feature response to the strongest
    non-crypt response and returns a normalized difference in [-1,1].

    Returns
    -------
    vertex_score : (V,) ndarray
        Crypt likelihood per vertex. Positive values indicate crypt-like regions.
    """
    E = np.asarray(encoding, float)
    cidx = np.asarray(crypt_idx, dtype=np.int64)
    nidx = np.asarray(noncrypt_idx, dtype=np.int64)
    cmax = np.max(E[:, cidx], axis=1)
    nmax = np.max(E[:, nidx], axis=1)
    return (cmax - nmax) / (cmax + nmax + float(eps))



def patches_from_score(
    vertex_score: np.ndarray,      # (V,) scalar crypt score per vertex
    faces: np.ndarray,             # (F,3) mesh triangle indices
    *,
    threshold: float,              # minimum vertex score to be considered part of a crypt
    min_patch_verts: int,          # minimum number of vertices required for a patch
    vertex_areas=None,             # (V,) optional vertex areas for area-based filtering
    min_patch_area=None,           # minimum surface area required for a patch
):
    """
    Extract connected crypt patches from a vertex score field.

    Vertices with score >= threshold are grouped into connected components
    on the mesh. Small components are filtered by vertex count and optionally
    surface area.

    Returns
    -------
    patches : list[set[int]]
        List of vertex index sets representing detected crypt patches.
    patch_scores : list[dict]
        Per-patch statistics (mean score, max score, vertex count, area).
    adj : list[list[int]]
        Vertex adjacency used for the connected component computation.
    """
    s = np.asarray(vertex_score, float)
    V = int(s.shape[0])

    adj = _vertex_adjacency_from_faces(V, faces)
    mask = s >= float(threshold)
    patches = _connected_components_from_mask(adj, mask)

    patches = [
        p for p in patches
        if _keep_patch(p, min_verts=int(min_patch_verts), vertex_areas=vertex_areas, min_area=min_patch_area)
    ]

    patch_scores = []
    for p in patches:
        idx = np.fromiter(p, dtype=np.int64)
        sp = s[idx]
        d = {"mean": float(np.mean(sp)), "max": float(np.max(sp)), "n_verts": int(idx.size)}
        if vertex_areas is not None:
            d["area"] = float(np.sum(vertex_areas[idx]))
        patch_scores.append(d)

    return patches, patch_scores, adj



def detect_crypts_by_encoding(
    bag_of_features: dict,         # dictionary containing vocabulary, scaler, sigma, ts
    mesh,                          # mesh object providing vertices, faces, and vertex areas
    *,
    L_ref: float = None,           # reference length scale used for HKS time normalization (optional override)
    crypt_vocab_idx = None,        # indices of vocabulary features representing crypt shapes (optional override)
    threshold: float,              # vertex crypt score threshold for patch detection
    min_patch_verts: int = 25,     # minimum number of vertices required for a crypt patch
    min_patch_area=None,           # optional minimum surface area for a patch
    return_intermediates=False,    # if True, also return intermediate data
):
    """
    Detect crypt patches on a mesh using HKS vocabulary encoding.

    The mesh surface is encoded using the HKS vocabulary, converted into a
    vertex-level crypt score, and thresholded to produce connected patches.

    Returns
    -------
    patches : list[set[int]]
        Vertex index sets corresponding to detected crypts.

    If return_intermediates=True:
    inter_res : dict
        Additional intermediate results including encoding, vertex scores,
        adjacency, vertex areas, and patch statistics.
    """

    # --- extract defaults from bag_of_features if needed ---
    if crypt_vocab_idx is None:
        if "crypt_vocab_idx" not in bag_of_features:
            raise ValueError("crypt_vocab_idx not provided and not found in bag_of_features.")
        crypt_vocab_idx = bag_of_features["crypt_vocab_idx"]

    crypt_vocab_idx = np.asarray(crypt_vocab_idx, dtype=np.int64)

    # --- determine non-crypt vocab indices automatically ---
    M = int(np.asarray(bag_of_features["vocab"]).shape[0])
    all_idx = np.arange(M, dtype=np.int64)
    noncrypt_vocab_idx = np.setdiff1d(all_idx, crypt_vocab_idx, assume_unique=False)

    # --- compute encoding ---
    vertex_areas = np.asarray(mesh.vertex_areas())

    encoding_all, ts_mesh, ts_vocab, hks, norm_hks = compute_vocabulary_encoding(
        bag_of_features, mesh, L_ref=L_ref, v_subset=None, return_hks=True,
    )

    vscore = crypt_vertex_score(encoding_all, crypt_vocab_idx, noncrypt_vocab_idx)

    patches, patch_scores, adj = patches_from_score(
        vscore, mesh.f,
        threshold=float(threshold),
        min_patch_verts=int(min_patch_verts),
        vertex_areas=vertex_areas,
        min_patch_area=min_patch_area,
    )

    if return_intermediates:
        inter_res = {
            "ts_mesh": ts_mesh,
            "ts_vocab": ts_vocab,
            "encoding": encoding_all,
            "vertex_score": vscore,
            "patch_scores": patch_scores,
            "adj": adj,
            "hks": hks,
            "norm_hks": norm_hks,
        }

    return (patches, inter_res) if return_intermediates else patches



def subdivide_crypts_by_encoding(
    bag_of_features: dict,             # vocabulary and HKS normalization data
    mesh,                              # mesh object
    *,
    patches,                           # list of initial crypt patches (vertex sets)
    L_ref: float = None,               # reference length scale for HKS normalization
    crypt_vocab_idx = None,            # vocabulary indices corresponding to crypt features
    threshold: float = 0.0,            # vertex score threshold used during subdivision
    min_patch_verts: int = 20,         # minimum vertex count for refined patches
    min_patch_area=None,               # optional minimum surface area for refined patches
    refine_only_if_area_at_least=None, # skip refinement if patch area below this
    return_intermediates=False,        # if True return encoding and scores
):
    """
    Refine previously detected crypt patches by recomputing HKS encoding
    within each patch and subdividing large patches into multiple crypts.

    Only patches that exceed the refinement size thresholds are subdivided.

    Returns
    -------
    final_patches : list[set[int]]
        Refined list of crypt patches.

    If return_intermediates=True:
    inter_res : dict
        Intermediate results including encoding, vertex scores, adjacency,
        vertex areas, and refined patches.
    """
    V = int(len(mesh.v))
    M = np.asarray(bag_of_features["vocab"], float).shape[0]

    # --- extract defaults from bag_of_features if needed ---
    if crypt_vocab_idx is None:
        if "crypt_vocab_idx" not in bag_of_features:
            raise ValueError("crypt_vocab_idx not provided and not found in bag_of_features.")
        crypt_vocab_idx = bag_of_features["crypt_vocab_idx"]

    crypt_vocab_idx = np.asarray(crypt_vocab_idx, dtype=np.int64)

    # --- determine non-crypt vocab indices automatically ---
    M = int(np.asarray(bag_of_features["vocab"]).shape[0])
    all_idx = np.arange(M, dtype=np.int64)
    noncrypt_vocab_idx = np.setdiff1d(all_idx, crypt_vocab_idx, assume_unique=False)

    # --- compute encoding ---
    vertex_areas = np.asarray(mesh.vertex_areas())
    adj = _vertex_adjacency_from_faces(V, mesh.f)

    final_patches = []
    encoding = np.zeros((V, M))
    encoding[:, noncrypt_vocab_idx] = 1

    for p in patches:
        if not _keep_patch(
            p,
            min_verts=1,
            vertex_areas=vertex_areas,
            min_area=refine_only_if_area_at_least,
        ):
            final_patches.append(p)
            continue

        idx = np.fromiter(p, dtype=np.int64)

        enc_sub, _, _ = compute_vocabulary_encoding(
            bag_of_features, mesh, L_ref=L_ref, v_subset=idx,
        )
        encoding[idx, :] = enc_sub

        s = crypt_vertex_score(encoding, crypt_vocab_idx, noncrypt_vocab_idx)

        mask = np.zeros(V, dtype=bool)
        mask[idx] = True
        mask &= (s >= float(threshold))

        refined = _connected_components_from_mask(adj, mask)
        refined = [
            rp for rp in refined
            if _keep_patch(rp, min_verts=int(min_patch_verts), vertex_areas=vertex_areas, min_area=min_patch_area)
        ]

        if len(refined) <= 1:
            final_patches.append(p)
        else:
            final_patches.extend(refined)

    s_final = crypt_vertex_score(encoding, crypt_vocab_idx, noncrypt_vocab_idx)

    if return_intermediates:
        inter_res = {
            "encoding": encoding,
            "vertex_score": s_final,
            "patches": final_patches,
            "adj": adj,
        }

    return (final_patches, inter_res) if return_intermediates else final_patches
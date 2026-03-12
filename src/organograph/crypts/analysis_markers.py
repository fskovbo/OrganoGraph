import numpy as np
from organograph.graph.access import graph_get

# ============================================================
# Marker binning
# ============================================================

def bin_marker_positivity(
    markers,   # (N_cells, M) array, marker positivity per cell
    distance,  # (N_cells,) array, distance coordinate per cell
    bin_edges, # (B+1,) array, bin edges
):
    """
    Bin marker positivity vs distance.

    Output
    ------
    counts_pos : (M, B) array
    counts_total : (B,) array
    """
    markers = np.asarray(markers)
    distance = np.asarray(distance)
    bin_edges = np.asarray(bin_edges)

    N, M = markers.shape
    B = len(bin_edges) - 1
    bin_ids = np.digitize(distance, bin_edges) - 1

    counts_pos = np.zeros((M, B), dtype=int)
    counts_total = np.zeros(B, dtype=int)

    for b in range(B):
        mask = (bin_ids == b)
        if np.any(mask):
            counts_total[b] = mask.sum()
            counts_pos[:, b] = markers[mask].astype(bool).sum(axis=0)

    return counts_pos, counts_total



def get_marker_counts_per_patch(G, patch):
    """
    Count marker-positive cells within a graph patch.

    Given a set of node indices defining a patch (e.g. a crypt), this function
    returns the number of cells positive for each marker as well as the total
    number of cells in the patch. Marker positivity is read from the binary
    node attribute ``markers_bin`` stored in the graph.
    """
    markers_bin = graph_get(G, "markers_bin")
    idx = np.fromiter(patch, dtype=np.int64)
    num_pos_cells = np.sum(markers_bin[idx, :], axis=0).astype(np.int64)
    num_cells = len(idx)
    return num_pos_cells, num_cells


def assign_coexpression_category(G, patch, i_LGR5, i_Sero, i_Lyso, pos_threshold=1):
    """
    Assign a co-expression category to a graph patch based on marker presence.

    The function determines whether LGR5, Serotonin, and Lysozyme are present
    in the patch (i.e. at least ``pos_threshold`` positive cells per marker),
    and maps the resulting presence/absence pattern to one of eight mutually
    exclusive co-expression categories (0–7). Categories 0–3 correspond to
    LGR5-positive patches, while 4–7 correspond to LGR5-negative patches.
    """
    markers_bin = graph_get(G, "markers_bin")
    idx = np.fromiter(patch, dtype=np.int64)
    bin_markers_patch = markers_bin[idx, :]

    has_marker = np.sum(bin_markers_patch, axis=0) >= pos_threshold

    # Write out full co-expression conditions (same style as marker_coexpression)
    c0 = has_marker[i_LGR5] and has_marker[i_Sero] and (not has_marker[i_Lyso])
    c1 = has_marker[i_LGR5] and has_marker[i_Lyso] and (not has_marker[i_Sero])
    c2 = has_marker[i_LGR5] and (not has_marker[i_Lyso]) and (not has_marker[i_Sero])
    c3 = has_marker[i_LGR5] and (has_marker[i_Lyso] and has_marker[i_Sero])

    c4 = (not has_marker[i_LGR5]) and has_marker[i_Sero] and (not has_marker[i_Lyso])
    c5 = (not has_marker[i_LGR5]) and has_marker[i_Lyso] and (not has_marker[i_Sero])
    c6 = (not has_marker[i_LGR5]) and (not has_marker[i_Lyso]) and (not has_marker[i_Sero])
    c7 = (not has_marker[i_LGR5]) and (has_marker[i_Lyso] and has_marker[i_Sero])

    # Convert the one-hot-ish conditions to a single index.
    # These should be mutually exclusive and cover all cases.
    conds = np.array([c0, c1, c2, c3, c4, c5, c6, c7], dtype=np.int64)

    hits = np.flatnonzero(conds)
    if hits.size == 1:
        return int(hits[0])
    else:
        return np.nan
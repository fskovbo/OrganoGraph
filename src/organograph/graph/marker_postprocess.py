import networkx as nx
import numpy as np

from organograph.graph.access import graph_get, graph_marker_index
from organograph.io_utils.cells_table import suppress_marker_if_coexpressed


def set_graph_markers_bin(G, markers_bin, *, marker_field="markers_bin"):
    """Write an (N, M) marker matrix back to graph node attributes."""
    markers_bin = np.asarray(markers_bin, dtype=np.int8)
    node_ids = list(range(G.number_of_nodes()))
    if markers_bin.ndim != 2 or markers_bin.shape[0] != len(node_ids):
        raise ValueError(
            f"markers_bin must have shape (N, M) with N={len(node_ids)}; got {markers_bin.shape}"
        )

    for i, node in enumerate(node_ids):
        G.nodes[node][marker_field] = markers_bin[i].tolist()
    return G


def copy_graph_markers_bin(G, *, source_field="markers_bin", target_field="markers_bin_raw"):
    """Keep a per-node copy of the marker matrix before graph-level postprocessing."""
    for node in range(G.number_of_nodes()):
        G.nodes[node][target_field] = list(G.nodes[node][source_field])
    return G


def ablate_lysozyme_not_agr2_in_clusters(
    G,
    *,
    lysozyme_marker="Lysozyme",
    agr2_marker="Agr2",
    min_cluster_size=1,
    marker_field="markers_bin",
):
    """
    Remove Lysozyme signal from Lysozyme+ graph components unless cells are also Agr2+.

    A Lysozyme cluster is a connected component of the subgraph induced by Lysozyme+
    cells. Within each cluster with size >= min_cluster_size, Lysozyme is retained
    only on cells that are also Agr2+.
    """
    markers_bin = np.asarray(graph_get(G, marker_field), dtype=np.int8)
    lyso_idx = graph_marker_index(G, lysozyme_marker)
    agr2_idx = graph_marker_index(G, agr2_marker)

    lyso = markers_bin[:, lyso_idx].astype(bool)
    agr2 = markers_bin[:, agr2_idx].astype(bool)
    lyso_nodes = np.flatnonzero(lyso).astype(int).tolist()

    removed = 0
    processed_clusters = 0
    for component in nx.connected_components(G.subgraph(lyso_nodes)):
        component = np.asarray(sorted(component), dtype=int)
        if component.size < int(min_cluster_size):
            continue
        processed_clusters += 1
        remove = component[~agr2[component]]
        removed += int(remove.size)
        markers_bin[remove, lyso_idx] = 0

    set_graph_markers_bin(G, markers_bin, marker_field=marker_field)
    G.graph["lysozyme_agr2_ablation"] = {
        "lysozyme_marker": str(lysozyme_marker),
        "agr2_marker": str(agr2_marker),
        "min_cluster_size": int(min_cluster_size),
        "processed_clusters": int(processed_clusters),
        "removed_lysozyme_cells": int(removed),
    }
    return G


def suppress_graph_marker_if_coexpressed(
    G,
    *,
    exclusive_marker,
    forbidden_markers,
    marker_field="markers_bin",
    copy=True,
    ignore_missing=False,
):
    """Apply table-style coexpression suppression to the current graph markers."""
    markers_bin = np.asarray(graph_get(G, marker_field), dtype=np.int8)
    marker_names = list(G.graph.get("marker_names", []))
    out = suppress_marker_if_coexpressed(
        markers_bin,
        marker_names,
        exclusive_marker=exclusive_marker,
        forbidden_markers=forbidden_markers,
        copy=copy,
        ignore_missing=ignore_missing,
    )
    return set_graph_markers_bin(G, out, marker_field=marker_field)

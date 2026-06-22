import networkx as nx
import numpy as np

from organograph.graph.access import graph_get, graph_marker_index
from organograph.io_utils.cells_table import suppress_marker_if_coexpressed


def set_graph_markers_bin(G, markers_bin, *, marker_field="markers_bin"):
    """Write an (N, M) binary marker matrix back to graph node attributes."""
    markers_bin = (np.asarray(markers_bin) > 0).astype(np.int8)
    node_ids = list(range(G.number_of_nodes()))
    if markers_bin.ndim != 2 or markers_bin.shape[0] != len(node_ids):
        raise ValueError(
            f"markers_bin must have shape (N, M) with N={len(node_ids)}; got {markers_bin.shape}"
        )

    for i, node in enumerate(node_ids):
        G.nodes[node][marker_field] = markers_bin[i].tolist()
    return G


def set_graph_markers_int(G, markers_int, *, marker_field="markers_int"):
    """Write an (N, M) marker intensity matrix back to graph node attributes."""
    markers_int = np.asarray(markers_int, dtype=float)
    node_ids = list(range(G.number_of_nodes()))
    if markers_int.ndim != 2 or markers_int.shape[0] != len(node_ids):
        raise ValueError(
            f"markers_int must have shape (N, M) with N={len(node_ids)}; got {markers_int.shape}"
        )

    for i, node in enumerate(node_ids):
        G.nodes[node][marker_field] = markers_int[i].tolist()
    return G


def copy_graph_markers_bin(G, *, source_field="markers_bin", target_field="markers_bin_raw"):
    """Keep a per-node copy of the marker matrix before graph-level postprocessing."""
    for node in range(G.number_of_nodes()):
        G.nodes[node][target_field] = list(G.nodes[node][source_field])
    return G


def copy_graph_marker_fields(
    G,
    *,
    source_fields=("markers_int", "markers_bin"),
    target_suffix="_raw",
    ignore_missing=True,
):
    """Keep per-node copies of marker fields before graph-level postprocessing."""
    for source_field in source_fields:
        if G.number_of_nodes() == 0:
            continue
        if source_field not in G.nodes[0]:
            if ignore_missing:
                continue
            raise KeyError(f"Node attribute '{source_field}' not found on graph nodes.")
        target_field = f"{source_field}{target_suffix}"
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
    intensity_field="markers_int",
):
    """
    Remove marker signal from marker+ graph components unless cells are also Agr2+.

    A marker cluster is a connected component of the subgraph induced by marker+
    cells. Within each cluster with size >= min_cluster_size, the marker is retained
    only on cells that are also Agr2+.
    """
    markers_bin = (np.asarray(graph_get(G, marker_field)) > 0).astype(np.int8)
    markers_int = None
    if G.number_of_nodes() > 0 and intensity_field in G.nodes[0]:
        markers_int = np.asarray(graph_get(G, intensity_field), dtype=float)
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
        if markers_int is not None:
            markers_int[remove, lyso_idx] = 0.0

    set_graph_markers_bin(G, markers_bin, marker_field=marker_field)
    if markers_int is not None:
        set_graph_markers_int(G, markers_int, marker_field=intensity_field)
    safe_marker = str(lysozyme_marker).strip().lower().replace(" ", "_")
    metadata_key = f"{safe_marker}_agr2_ablation"
    G.graph[metadata_key] = {
        "target_marker": str(lysozyme_marker),
        "agr2_marker": str(agr2_marker),
        "min_cluster_size": int(min_cluster_size),
        "processed_clusters": int(processed_clusters),
        "removed_marker_positive_cells": int(removed),
    }
    if str(lysozyme_marker).strip().lower() == "lysozyme":
        G.graph["lysozyme_agr2_ablation"] = G.graph[metadata_key]
    return G


def suppress_graph_marker_if_coexpressed(
    G,
    *,
    exclusive_marker,
    forbidden_markers,
    marker_field="markers_bin",
    intensity_field="markers_int",
    copy=True,
    ignore_missing=False,
):
    """Apply table-style coexpression suppression to the current graph markers."""
    markers_bin = (np.asarray(graph_get(G, marker_field)) > 0).astype(np.int8)
    markers_int = None
    if G.number_of_nodes() > 0 and intensity_field in G.nodes[0]:
        markers_int = np.asarray(graph_get(G, intensity_field), dtype=float)
    marker_names = list(G.graph.get("marker_names", []))
    out = suppress_marker_if_coexpressed(
        markers_bin,
        marker_names,
        exclusive_marker=exclusive_marker,
        forbidden_markers=forbidden_markers,
        copy=copy,
        ignore_missing=ignore_missing,
    )
    if markers_int is not None:
        changed_to_zero = (markers_bin > 0) & (np.asarray(out) <= 0)
        markers_int[changed_to_zero] = 0.0
        set_graph_markers_int(G, markers_int, marker_field=intensity_field)
    return set_graph_markers_bin(G, out, marker_field=marker_field)

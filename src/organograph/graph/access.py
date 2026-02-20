import numpy as np


def graph_get(G, field, nodes=None, dtype=None):
    """
    Return a node attribute as a NumPy array.

    Parameters
    ----------
    G : networkx.Graph
    field : str
        Node attribute key.
    nodes : iterable of node ids or None
        If None → use nodes 0..N-1
        Else → return values only for these node ids (in given order).
    dtype : numpy dtype or None
        Optional dtype cast.

    Raises
    ------
    KeyError if field not present (lists available fields)
    KeyError if a requested node id is missing
    """
    n = G.number_of_nodes()
    if n == 0:
        raise ValueError("Graph has no nodes")

    available_fields = list(G.nodes[0].keys())
    if field not in available_fields:
        raise KeyError(
            f"Field '{field}' not found in graph nodes. "
            f"Available fields: {sorted(available_fields)}"
        )

    if nodes is None:
        node_ids = range(n)
    else:
        # allow scalar node id
        if isinstance(nodes, (int, np.integer)):
            node_ids = [int(nodes)]
        else:
            node_ids = list(nodes)

        for u in node_ids:
            if u not in G.nodes:
                raise KeyError(f"Node id {u} not in graph")

    arr = np.asarray([G.nodes[i][field] for i in node_ids])
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def graph_get_meta(G, key):
    """
    Return a graph-level attribute stored in G.graph.

    Raises
    ------
    KeyError if key is not present, listing available graph metadata keys.
    """
    if key not in G.graph:
        available = sorted(G.graph.keys())
        raise KeyError(
            f"Graph attribute '{key}' not found. "
            f"Available graph attributes: {available}"
        )

    return G.graph[key]


def graph_marker_index(G, marker):
    """
    Resolve a marker name to its column index using G.graph["marker_names"].

    Parameters
    ----------
    marker : str

    Returns
    -------
    int
    """
    names = G.graph.get("marker_names", None)
    if names is None:
        raise KeyError('Graph has no "marker_names" in G.graph metadata.')

    # Normalize: case-insensitive exact match after stripping
    target = str(marker).strip().lower()
    norm = [str(x).strip().lower() for x in list(names)]

    if target not in norm:
        raise KeyError(
            f"Marker '{marker}' not found in G.graph['marker_names']. "
            f"Available: {list(names)}"
        )
    return int(norm.index(target))


def graph_get_marker_bin(G, marker, nodes=None, dtype=np.int64):
    """
    Return the binary marker vector for a marker specified by name or index.

    If marker is str: uses G.graph["marker_names"] to find the index.
    If marker is int: uses it directly.

    Returns
    -------
    (N,) numpy array
    """
    # get (N,M) markers_bin
    markers_bin = graph_get(G, "markers_bin", nodes=nodes, dtype=dtype)

    if isinstance(marker, str):
        idx = graph_marker_index(G, marker)
    else:
        idx = int(marker)

    if markers_bin.ndim != 2:
        raise ValueError('"markers_bin" should be (N, M).')
    if idx < 0 or idx >= markers_bin.shape[1]:
        raise IndexError(f"marker index {idx} out of range (M={markers_bin.shape[1]}).")

    return markers_bin[:, idx]


def graph_inspect(G):
    """
    Inspect node fields and graph metadata.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    node_fields = sorted(G.nodes[0].keys()) if n > 0 else []
    meta_keys = sorted(G.graph.keys())

    preview = {}
    for k in meta_keys:
        v = G.graph[k]
        if isinstance(v, (str, int, float, bool)):
            preview[k] = v

    return {
        "n_nodes": n,
        "n_edges": m,
        "node_fields": node_fields,
        "graph_metadata_keys": meta_keys,
        "graph_metadata_preview": preview,
    }
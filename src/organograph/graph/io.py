
import numpy as np
import networkx as nx
import pickle

def save_cell_graph(path, G):
    """
    Save the organoid cell graph using Python pickle.
    Preserves all edges, attributes, metadata.
    """
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[INFO] Saved graph to {path}")


def load_cell_graph(path):
    """
    Load a saved organoid cell graph.

    Returns
    -------
    G : networkx.Graph
    """

    with open(path, "rb") as f:
        G = pickle.load(f)

    return G


def load_cell_graph_from_npz(data: np.lib.npyio.NpzFile) -> nx.Graph:
    """
    Reconstruct a NetworkX graph from edges and node count stored in an NPZ file.
    This matches the new preprocessing format where `edges` and `n_nodes`
    are explicitly saved.
    """
    if "graph_edges" not in data.files or "graph_n_nodes" not in data.files:
        raise KeyError("NPZ file must contain 'edges' and 'n_nodes' to load the cell graph.")

    edges = data["graph_edges"]
    n_nodes = int(data["graph_n_nodes"])

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    if edges.size > 0:
        edges = edges.reshape(-1, 2)
        G.add_edges_from(edges.tolist())

    return G
import numpy as np
import networkx as nx

from organograph.projection.project import project_nuclei_to_mesh
from organograph.projection.voronoi import voronoi_on_mesh_dijkstra
from organograph.mesh.transform import transform_is_applied, apply_transform_to_points
from organograph.graph.access import graph_get


def build_organoid_graph(
    mesh,                    # OrganoidMesh with mesh.v, mesh.f; mesh.label_uid set
    extract_fn,              # callable(label_uid)->(nuclei_xyz_raw (N,3), markers_bin (N,M), marker_names)
    *,
    precomputed=None,        # dict or None: {"proj_vertex_ids":..., "vertex_owner":..., "proj_points":...}
    resolve_duplicates=True,
    project_fn=project_nuclei_to_mesh,
    voronoi_fn=voronoi_on_mesh_dijkstra,
):
    """
    Build the organoid cell graph.

    Returns
    -------
    G : networkx.Graph
    aux : dict with useful arrays (proj_vertex_ids, proj_points, vertex_owner, dist_mat if computed)
    """
    if getattr(mesh, "label_uid", None) is None:
        raise ValueError("mesh.label_uid must be set")

    # ---- 1) extract RAW nuclei + markers for this label_uid ----
    nuclei_xyz_raw, markers_bin, marker_names = extract_fn(mesh.label_uid)

    nuclei_xyz_raw = np.asarray(nuclei_xyz_raw, float)
    markers_bin = np.asarray(markers_bin)

    if nuclei_xyz_raw.ndim != 2 or nuclei_xyz_raw.shape[1] != 3:
        raise ValueError(f"extract_fn must return nuclei_xyz shape (N,3), got {nuclei_xyz_raw.shape}")
    if markers_bin.ndim != 2 or markers_bin.shape[0] != nuclei_xyz_raw.shape[0]:
        raise ValueError("extract_fn must return markers_bin shape (N,M) with same N as nuclei_xyz")

    # ---- 2) apply mesh transform to nuclei (if mesh is in a different frame) ----
    nuclei_xyz = _apply_mesh_transform_to_nuclei(mesh, nuclei_xyz_raw)

    # ---- 3) projection + vertex_owner OR use precomputed ----
    aux = {}
    if precomputed is None:
        if project_fn is None or voronoi_fn is None:
            raise ValueError("project_fn and voronoi_fn are required when precomputed is None")

        proj_vertex_ids, proj_points = project_fn(
            nuclei_xyz, mesh, resolve_duplicates=resolve_duplicates
        )
        vertex_owner, dist_mat = voronoi_fn(mesh, proj_vertex_ids)

        aux.update({
            "proj_vertex_ids": proj_vertex_ids,
            "proj_points": proj_points,
            "vertex_owner": vertex_owner,
            "dist_mat": dist_mat,
        })
    else:
        if "vertex_owner" not in precomputed or "proj_vertex_ids" not in precomputed:
            raise ValueError("precomputed must contain at least 'vertex_owner' and 'proj_vertex_ids'")

        vertex_owner = np.asarray(precomputed["vertex_owner"])
        proj_vertex_ids = np.asarray(precomputed["proj_vertex_ids"])
        proj_points = precomputed.get("proj_points", None)

        aux.update({
            "proj_vertex_ids": proj_vertex_ids,
            "proj_points": proj_points,
            "vertex_owner": vertex_owner,
        })

    # ---- 4) build graph ----
    G = _build_graph_from_voronoi(
        nuclei_xyz=nuclei_xyz,
        markers_bin=markers_bin,
        mesh=mesh,
        vertex_owner=vertex_owner,
        proj_vertex_ids=proj_vertex_ids,
        proj_points=proj_points,
    )

    # ---- 5) metadata ----
    G.graph["label_uid"] = mesh.label_uid
    G.graph["marker_names"] = list(marker_names) if marker_names is not None else []
    if hasattr(mesh, "coord_transform"):
        G.graph["coord_transform"] = mesh.coord_transform

    return G, aux


def add_vertex_field_to_graph(G, vertex_field, attr_name):
    """
    Attach a mesh-defined field to each node in the cell graph.

    Graph nodes must have a "proj_vertex" attribute giving the mesh vertex ID.
    """
    vertex_field = np.asarray(vertex_field)

    for n in G.nodes:
        v_id = G.nodes[n]["proj_vertex"]
        value = vertex_field[v_id]

        if np.ndim(value) == 0:
            G.nodes[n][attr_name] = value.item()
        else:
            G.nodes[n][attr_name] = np.array(value, copy=True)

    return G


def _apply_mesh_transform_to_nuclei(mesh, nuclei_xyz_raw):
    tr = getattr(mesh, "coord_transform", None)
    if tr is None or (not transform_is_applied(tr)):
        return nuclei_xyz_raw
    return apply_transform_to_points(nuclei_xyz_raw, tr)


def _build_graph_from_voronoi(
    nuclei_xyz,
    markers_bin,
    mesh,
    vertex_owner,
    proj_vertex_ids,
    proj_points,
):
    G = nx.Graph()
    N_cells = len(nuclei_xyz)

    # add nodes
    for i in range(N_cells):
        G.add_node(
            i,
            centroid=nuclei_xyz[i].tolist(),
            markers_bin=markers_bin[i].tolist(),
            proj_vertex=int(proj_vertex_ids[i]),
            proj_point=(proj_points[i].tolist() if proj_points is not None else None),
        )

    owners = vertex_owner
    f = mesh.f

    # add edges from faces
    for tri in f:
        a, b, c = tri
        ca, cb, cc = owners[a], owners[b], owners[c]
        if ca < 0 or cb < 0 or cc < 0:
            continue

        for (ci, cj) in ((ca, cb), (cb, cc), (cc, ca)):
            if ci == cj:
                continue
            u = int(ci)
            v = int(cj)
            if u != v:
                G.add_edge(u, v)

    return G


def assign_mesh_patches_to_graph(
    G,
    crypt_patches_mesh,
    *,
    proj_field: str = "proj_vertex",
    nodes=None,
    drop_empty: bool = True,
    return_node_ids: bool = False,
):
    """
    Assign mesh-vertex crypt patches to a cell graph using each node's projected mesh vertex.

    A graph node belongs to crypt patch k iff its node attribute `proj_field` is contained
    in the corresponding mesh patch (a set / list / array of mesh vertex indices).

    Parameters
    ----------
    G
        Graph (e.g. networkx.Graph) whose nodes contain an integer node attribute `proj_field`.
    crypt_patches_mesh
        Iterable of patches on the mesh. Each patch is an iterable (commonly set[int])
        of mesh vertex indices.
    proj_field
        Name of the node attribute containing the projected mesh vertex index (default: "proj_vertex").
    nodes
        Optional iterable of node ids to consider. If None, uses nodes 0..N-1
        (same convention as graph_get).
    drop_empty
        If True, patches that receive no graph nodes are omitted from the returned list.
        If False, returns an entry for every mesh patch (possibly empty).
    return_node_ids
        If False (default), returned patches contain indices 0..(len(nodes)-1) in the
        order used for assignment.
        If True, returned patches contain the actual node ids from G.

    Returns
    -------
    graph_patches
        List[set[int]] of graph patches. Each set contains node indices (or node ids if
        return_node_ids=True).
    info
        Dict with debug metadata:
          - "nodes_used": array of node ids in the order used
          - "proj_vertex": array of projected mesh vertices for those nodes
          - "mesh_patch_sizes": list of sizes of input mesh patches
          - "graph_patch_sizes": list of sizes of output graph patches (after drop_empty policy)
          - "mesh_to_graph_index": list mapping mesh patch index -> output index (or -1 if dropped)
    """
    # get node ids (order) consistent with graph_get convention
    n_total = G.number_of_nodes()
    if n_total == 0:
        raise ValueError("Graph has no nodes")

    if nodes is None:
        node_ids = np.arange(n_total, dtype=np.int64)
    else:
        if isinstance(nodes, (int, np.integer)):
            node_ids = np.array([int(nodes)], dtype=np.int64)
        else:
            node_ids = np.array(list(nodes), dtype=np.int64)

    # fetch projected vertices in the same order
    proj_v = graph_get(G, proj_field, nodes=node_ids, dtype=np.int64)  # (Nnodes,)

    # build mapping mesh_vertex -> list of positions in node_ids (or node ids)
    # This makes patch assignment O(sum |patch|) rather than O(#patches * #nodes).
    vert_to_nodes = {}
    if return_node_ids:
        for pos, u in enumerate(node_ids):
            v = int(proj_v[pos])
            vert_to_nodes.setdefault(v, []).append(int(u))
    else:
        # store indices into node_ids array (0..len(node_ids)-1)
        for pos in range(node_ids.size):
            v = int(proj_v[pos])
            vert_to_nodes.setdefault(v, []).append(int(pos))

    graph_patches = []
    mesh_to_graph_index = []
    mesh_patch_sizes = []
    graph_patch_sizes = []

    for k, patch in enumerate(crypt_patches_mesh):
        # iterate patch vertices and collect nodes whose proj_vertex hits this patch
        patch_set = patch if isinstance(patch, set) else set(patch)
        mesh_patch_sizes.append(len(patch_set))

        nodes_in_patch = set()
        for v in patch_set:
            lst = vert_to_nodes.get(int(v))
            if lst:
                # add all graph nodes that project to this mesh vertex
                nodes_in_patch.update(lst)

        if drop_empty and len(nodes_in_patch) == 0:
            mesh_to_graph_index.append(-1)
            continue

        mesh_to_graph_index.append(len(graph_patches))
        graph_patches.append(nodes_in_patch)
        graph_patch_sizes.append(len(nodes_in_patch))

    info = {
        "nodes_used": node_ids,
        "proj_vertex": proj_v,
        "mesh_patch_sizes": mesh_patch_sizes,
        "graph_patch_sizes": graph_patch_sizes,
        "mesh_to_graph_index": mesh_to_graph_index,
    }
    return graph_patches, info
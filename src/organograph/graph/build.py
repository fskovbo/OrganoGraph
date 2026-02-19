import numpy as np
import networkx as nx

from organograph.projection.project import project_nuclei_to_mesh
from organograph.projection.voronoi import voronoi_on_mesh_dijkstra
from organograph.mesh.transform import transform_is_applied, apply_transform_to_points


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
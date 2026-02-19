import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm


def compute_geodesics_heat(mesh, t=None, sources=None):
    """
    Compute approximate geodesic distances on a mesh using the Heat Method
    (Crane et al., 2013). Solves a pair of linear systems per source.

    Parameters
    ----------
    mesh : OrganoidMesh
        Must provide:
            - v : (V,3) vertex coordinates
            - f : (F,3) triangle faces
            - laplacian : (V,V) sparse cotangent Laplacian
            - mass_matrix : (V,V) sparse lumped mass matrix
    t : float, optional
        Diffusion time. If None, set to mean_edge_length².
    sources : array of int, optional
        Indices of source vertices. If None, compute for all vertices (O(V²)).

    Returns
    -------
    D_out : (S,V) ndarray
        Geodesic distances from each source to all vertices.
    """
    
    V = mesh.v.shape[0]

    # build per-face barycentric gradients and calculate per-face areas
    G_face = build_G_face(mesh.v, mesh.f)
    face_areas = mesh.calc_face_areas()

    # regularize Laplacian for stability
    reg = 1e-12
    L_reg = mesh.laplacian + reg * sparse.eye(V)
    L_factor = splu(L_reg.tocsc())

    # compute time scale if not given
    if t is None:
        edges = np.vstack([mesh.f[:, [0, 1]], mesh.f[:, [1, 2]], mesh.f[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        mean_edge = np.mean(np.linalg.norm(mesh.v[edges[:, 0]] - mesh.v[edges[:, 1]], axis=1))
        k = 5.0 # timescale spans around 5 mean edge lengths
        t = (k * mean_edge) ** 2

    A = (mesh.mass_matrix + t * mesh.laplacian).tocsc()
    A_factor = splu(A)

    # define sources
    if sources is None:
        sources = np.arange(V, dtype=np.int32)
    else:
        sources = np.asarray(sources, dtype=np.int32)

    S = len(sources)
    D_out = np.zeros((S, V), dtype=np.float64)

    for si, s in enumerate(tqdm(sources, desc='heat sources')):
        # delta at source
        delta = np.zeros(V)
        delta[s] = 1.0
        rhs = mesh.mass_matrix @ delta

        # heat diffusion
        u = A_factor.solve(rhs)

        # grad u per face
        grad_u = np.einsum('ka,kai->ki', u[mesh.f], G_face)  # (F,3)

        # normalized field
        norms = np.linalg.norm(grad_u, axis=1)
        norms[norms == 0] = 1e-12
        X_face = -(grad_u.T / norms).T

        # divergence at vertices
        contrib = np.einsum('ki,kai->ka', X_face, G_face)
        contrib *= face_areas[:, None]
        div = np.zeros(V)
        for a in range(3):
            np.add.at(div, mesh.f[:, a], contrib[:, a])

        # solve Poisson: L φ = div
        phi = L_factor.solve(div)
        phi -= phi[s]
        D_out[si, :] = phi

    return D_out


def build_G_face(v, f):
    """
    Compute per-face gradients of barycentric basis functions.

    Parameters
    ----------
    v : (V,3) array
        Vertex coordinates
    f : (F,3) int array
        Triangle indices

    Returns
    -------
    G_face : (F,3,3) array
        G_face[k,a,:] = ∇φ_a on face k
    """
    tri = v[f]  # (F,3,3)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    n = np.cross(e1, e2)
    dblA = np.linalg.norm(n, axis=1)
    n_unit = n / (dblA[:, None] + 1e-20)

    F = f.shape[0]
    G_face = np.zeros((F, 3, 3), dtype=np.float64)

    for a in range(3):
        i1, i2 = (a + 1) % 3, (a + 2) % 3
        edge = tri[:, i2] - tri[:, i1]
        grad = np.cross(n_unit, edge) / (dblA[:, None] + 1e-20)
        G_face[:, a, :] = grad

    return G_face


def compute_geodesics_dijkstra(mesh, sources=None):
    """
    Approx geodesic distances via Dijkstra on the mesh edge graph.
    Distances are shortest paths constrained to edges (fast, less accurate).

    Parameters
    ----------
    mesh : must provide mesh.v (V,3) and mesh.f (F,3)
    sources : (S,) int array or None
        Source vertices. If None -> all vertices (expensive).

    Returns
    -------
    D : (S,V) float64
        Distances from each source.
    pred : (S,V) int32 (optional)
        Predecessors for path reconstruction.
    """
    v = mesh.v
    f = mesh.f
    V = v.shape[0]

    # Unique undirected edges from faces
    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)  # (E,2)
    i = edges[:, 0]
    j = edges[:, 1]

    # Edge weights = Euclidean edge lengths
    w = np.linalg.norm(v[i] - v[j], axis=1)

    # Build symmetric adjacency (CSR)
    row = np.concatenate([i, j])
    col = np.concatenate([j, i])
    data = np.concatenate([w, w])
    A = sparse.csr_matrix((data, (row, col)), shape=(V, V))

    if sources is None:
        sources = np.arange(V, dtype=np.int32)
    else:
        sources = np.asarray(sources, dtype=np.int32)

    dist = dijkstra(csgraph=A, directed=False, indices=sources,
                          return_predecessors=False)

    dist = np.asarray(dist, dtype=np.float64)  # (S,V)
    return dist
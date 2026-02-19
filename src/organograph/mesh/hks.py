import numpy as np

def compute_hks(mesh, t=[1, 5, 10], coeffs=True):
    """
    Compute Heat Kernel Signatures (HKS) for an OrganoidMesh.

    The HKS characterizes local geometric structure over multiple diffusion
    timescales and is invariant under isometric transformations of the mesh.
    It is widely used in shape analysis and diffusion-based descriptors.

    Parameters
    ----------
    mesh : OrganoidMesh
        Mesh object with precomputed Laplacian eigendecomposition.
        Must have attributes:
            - eigvals : (K,) ndarray
            - eigvecs : (V,K) ndarray
            - mass_matrix : (V,V) sparse matrix
    t : list of float, optional
        Diffusion times for which to compute HKS.
    coeffs : bool, optional
        If True, return HKS projected into Laplacian eigenbasis.

    Returns
    -------
    hks : (V, len(t)) ndarray
        Heat kernel signatures at each vertex and time.
    coeffs_hks : (K, len(t)) ndarray, optional
        HKS coefficients in the Laplacian basis (if coeffs=True).
    """
    # Ensure the mesh has eigen-decomposition
    mesh._ensure_eigendecomposition()

    # Compute HKS
    hks = np.array([
        np.einsum("i,ji->j", np.exp(-mesh.eigvals * ti), mesh.eigvecs ** 2)
        for ti in t
    ]).T  # shape (V, len(t))

    if coeffs:
        coeffs_hks = mesh.eigvecs.T @ (mesh.mass_matrix @ hks)
        return hks, coeffs_hks

    return hks

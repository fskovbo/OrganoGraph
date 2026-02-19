import os
import igl
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from organograph.mesh.transform import make_identity_transform, warn_if_already_transformed
import vtk
import pickle


# =============================================================================
# 1. Core geometry + spectral class 
# =============================================================================

class OrganoidMesh:
    """
    A class representing an organoid surface mesh.

    This core class supports:
    - Loading meshes (STL, OBJ, or VTP) geometry only.
    - Optional PCA alignment (but no rescaling on load).
    - Construction of the cotangent Laplace–Beltrami operator and mass matrix.
    - Spectral decomposition and reconstruction from eigen-coefficients.
    - Saving and loading geometry + spectral data.
    """

    def __init__(self, path: str | None = None):
        """Initialize empty mesh and data containers.
        If `path` is given, load mesh geometry from that file."""
        self.v = None              # (V, 3) vertices
        self.f = None              # (F, 3) faces

        # Spectral / FEM data
        self.laplacian = None
        self.mass_matrix = None
        self.eigvals = None
        self.eigvecs = None
        self.lmax = None
        self.coeffs_v = None       # spectral coefficients of vertex coords
        self.coord_transform = make_identity_transform()

        label_uid = None

        if path is not None:
            self.load_mesh_from_file(path)

    # -------------------------------------------------------------------------
    # --- Mesh loading 
    # -------------------------------------------------------------------------

    def _load_stl(self, path):
        """Load vertices and faces from an STL file."""
        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()
        polydata = reader.GetOutput()

        self.v = np.array(polydata.GetPoints().GetData())
        n_faces = polydata.GetNumberOfPolys()
        cells = polydata.GetPolys()
        cells.InitTraversal()

        faces = np.zeros((n_faces, 3), dtype=np.int64)
        for i in range(n_faces):
            cell = vtk.vtkIdList()
            cells.GetNextCell(cell)
            for j in range(3):
                faces[i, j] = cell.GetId(j)
        self.f = faces


    def _load_vtp_geometry(self, path):
        """Load vertices and faces from a VTP file. Ignore point data."""
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        polydata = reader.GetOutput()

        self.v = np.array(polydata.GetPoints().GetData())
        n_faces = polydata.GetNumberOfPolys()
        cells = polydata.GetPolys()
        cells.InitTraversal()

        faces = np.zeros((n_faces, 3), dtype=np.int64)
        for i in range(n_faces):
            cell = vtk.vtkIdList()
            cells.GetNextCell(cell)
            for j in range(3):
                faces[i, j] = cell.GetId(j)
        self.f = faces


    def load_mesh_from_file(self, path):
        """
        Load a mesh file (STL, OBJ, or VTP), geometry only.

        IMPORTANT: No rescaling or centering is applied automatically.
        """
        if path.endswith(".stl"):
            self._load_stl(path)
        elif path.endswith(".obj"):
            self.v, self.f = igl.read_triangle_mesh(path)
        elif path.endswith(".vtp"):
            self._load_vtp_geometry(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        return self


    def load_from_arrays(self, vertices, faces):
        """Load mesh geometry directly from arrays."""
        self.v = np.asarray(vertices)
        self.f = np.asarray(faces, dtype=np.int64)
        return self


    # -------------------------------------------------------------------------
    # --- Optional scaling and alignment 
    # -------------------------------------------------------------------------

    def normalize_inplace(self, scale=10.0, center="mean"):
        v = np.asarray(self.v, dtype=float)

        # warn if already transformed
        warn_if_already_transformed(getattr(self, "coord_transform", None), obj_name="OrganoidMesh")

        if center != "mean":
            raise ValueError("center must be 'mean'")

        c = v.mean(axis=0)
        s = float(scale)

        self.v = (v - c) / s

        # record transform (rotation unchanged)
        tr = getattr(self, "coord_transform", None) or make_identity_transform()
        tr["center"] = np.asarray(c, float)
        tr["scale"] = s
        self.coord_transform = tr
        return c, s


    def align_with_pca(self):
        """
        Rotate mesh into PCA principal axes.
        Assumes centering/scaling (if desired) is handled separately.
        """
        from sklearn.decomposition import PCA

        v = np.asarray(self.v, dtype=float)

        # warn if already transformed (because you're composing transforms)
        warn_if_already_transformed(getattr(self, "coord_transform", None), obj_name="OrganoidMesh")

        pca = PCA(n_components=3)
        pca.fit(v)
        R = pca.components_.copy()  # (3,3)

        self.v = v @ R.T

        tr = getattr(self, "coord_transform", None) or make_identity_transform()
        tr["rotation"] = np.asarray(R, float)
        self.coord_transform = tr

        # set center to zeros in order to make transformation "visible"
        if not np.isfinite(tr["center"]).all():
            tr["center"] = np.zeros(3, float)
            tr["scale"] = 1.0

        return self


    # -------------------------------------------------------------------------
    # --- Laplace–Beltrami operator and eigen decomposition
    # -------------------------------------------------------------------------

    @staticmethod
    def build_cotangent_laplacian_and_mass(v, f):
        """
        Construct the cotangent Laplacian L and mass matrix M (both sparse).
        L is the positive-semidefinite Laplace–Beltrami operator such that
        uᵀLu = ∫ |∇u|².
        """
        L = -igl.cotmatrix(v, f)
        M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
        return sparse.csr_matrix(L), sparse.csr_matrix(M)


    def _eig_decomp(self, k=225, sigma=0):
        """
        Compute the first k Laplace–Beltrami eigenmodes using the cotangent operator.
        """
        L, M = self.build_cotangent_laplacian_and_mass(self.v, self.f)
        self.laplacian = L
        self.mass_matrix = M
        self.eigvals, self.eigvecs = eigsh(L, k=k, M=M, sigma=sigma, which="LM")
        return self.eigvals, self.eigvecs, self.mass_matrix


    def _ensure_eigendecomposition(self, k=225, sigma=0):
        """
        Ensure that the Laplace–Beltrami eigendecomposition has been computed.
        If not, perform it now.
        """
        if self.eigvals is None or self.eigvecs is None:
            print("[Info] Eigen-decomposition not found. Computing now...")
            self._eig_decomp(k=k, sigma=sigma)

    # -------------------------------------------------------------------------
    # --- Spectral computations
    # -------------------------------------------------------------------------

    def compute_spectral_coefficients(self, lmax=15):
        """
        Project vertex coordinates onto the Laplacian eigenbasis.
        Stores coefficients in self.coeffs_v.

        Parameters
        ----------
        lmax : int
            Maximum eigenlevel (number of eigenfunctions ~ lmax²).
        """
        self.lmax = lmax
        k = int(lmax ** 2)
        self._ensure_eigendecomposition(k=k)

        # (k x V) @ (V x 3) = (k x 3)
        self.coeffs_v = self.eigvecs.T @ (self.mass_matrix @ self.v)
        return self.coeffs_v


    def reconstruct_from_coeffs(self, coeffs, lmax=15):
        """
        Reconstruct spatial fields from their Laplacian coefficients up to lmax.

        Parameters
        ----------
        coeffs : ndarray
            Laplacian coefficients (shape = [k, d]).
        lmax : int
            Reconstruction level cutoff.
        """
        self._ensure_eigendecomposition()
        if self.lmax is None or self.lmax < lmax:
            raise ValueError(f"Stored lmax={self.lmax} is smaller than requested lmax={lmax}.")
        return self.eigvecs[:, : int(lmax ** 2)] @ coeffs[: int(lmax ** 2), :]


    def compute_power_spectrum(self, coeffs, lmax=15):
        """
        Compute the power spectrum of Laplacian coefficients, grouped by eigenlevel l.

        Parameters
        ----------
        coeffs : ndarray
            Laplacian coefficients.
        lmax : int
            Maximum eigenlevel to include in the power spectrum.
        """
        if self.lmax is None or lmax > self.lmax:
            raise ValueError("Compute coefficients first with sufficient lmax.")
        return np.array([
            np.sum(coeffs[i ** 2 : (i + 1) ** 2] ** 2, axis=0)
            for i in range(lmax)
        ])


    def compute_reconstruction_quality(self, lmax=None):
        """
        Quantify mesh reconstruction quality via L2 error between
        the original vertex coordinates and those reconstructed from coefficients.
        """
        if lmax is None:
            lmax = self.lmax
        v_recon = self.reconstruct_from_coeffs(self.coeffs_v, lmax=lmax)
        return np.sqrt(np.sum((self.v - v_recon) ** 2))


    def remove_lowest_modes(self, field=None, coeffs=None, l_remove=1, lmax=None):
        """
        Remove the lowest Laplace–Beltrami modes from a scalar/vector field
        defined on the mesh (or from its spectral coefficients).
        """
        self._ensure_eigendecomposition()

        if lmax is None:
            lmax = self.lmax
        k = int(lmax ** 2)

        if coeffs is None:
            if field is None:
                raise ValueError("Either 'field' or 'coeffs' must be provided.")
            coeffs = self.eigvecs.T @ (self.mass_matrix @ field)

        filtered_coeffs = coeffs.copy()
        cutoff = l_remove ** 2
        filtered_coeffs[:cutoff, ...] = 0
        filtered_field = self.eigvecs[:, :k] @ filtered_coeffs[:k, ...]
        return filtered_field, filtered_coeffs

    # -------------------------------------------------------------------------
    # --- Volume and area utilities
    # -------------------------------------------------------------------------

    def volume(self) -> float:
        """
        Compute the total volume enclosed by the mesh using signed tetrahedra.
        """
        tri = self.v[self.f]      # (F, 3, 3)
        a = tri[:, 0, :]
        b = tri[:, 1, :]
        c = tri[:, 2, :]

        cross_ab = np.cross(a, b)
        tet_signed = np.einsum('ij,ij->i', cross_ab, c) / 6.0
        volume = np.abs(np.sum(tet_signed))
        return volume


    def vertex_areas(self, from_mass_matrix: bool = True) -> np.ndarray:
        """
        Compute per-vertex surface areas on the mesh.
        """
        if from_mass_matrix:
            if self.mass_matrix is None:
                raise ValueError("Mass matrix not found. Build it first with build_cotangent_laplacian_and_mass().")
            vertex_areas = np.array(self.mass_matrix.diagonal())
        else:
            V = self.v.shape[0]
            tri = self.v[self.f]
            face_areas = 0.5 * np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                axis=1
            )
            vertex_areas = np.zeros(V, dtype=np.float64)
            for i in range(3):
                vertex_areas[self.f[:, i]] += face_areas / 3.0
        return vertex_areas


    def face_areas(self) -> np.ndarray:
        """
        Compute per-face surface areas on the mesh.
        """
        tri = self.v[self.f]
        face_areas = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
            axis=1
        )
        return face_areas

    # -------------------------------------------------------------------------
    # --- Saving / loading (geometry + spectral only)
    # -------------------------------------------------------------------------

    def save_results(self, path):
        """
        Save mesh geometry and spectral analysis results to a pickle file.
        """
        data = dict(
            v=self.v,
            f=self.f,
            eigvals=self.eigvals,
            eigvecs=self.eigvecs,
            mass_matrix=self.mass_matrix,
            coeffs_v=self.coeffs_v,
            coord_transform=self.coord_transform,
            lmax=self.lmax,
        )
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_results(self, path):
        """
        Load previously saved mesh geometry and spectral analysis results.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            setattr(self, k, v)
        return self


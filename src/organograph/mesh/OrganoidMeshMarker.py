from organograph.mesh.OrganoidMesh import *

class OrganoidMeshMarkers(OrganoidMesh):
    """
    Extension of OrganoidMesh that carries marker-related functionality.

    - Extraction of standardized fate markers from raw VTP point data.
    - Optional LGR5 coexpression filtering.
    - Mapping between cell graph markers and per-vertex marker fields.
    - Per-cell statistics using a label field, if available.
    """

    # Default annotation mapping 
    annotation_names = {
        'LGR5': '0.C02.percentile99_class',
        'Chromogranin A': '0.C03.percentile99_class',
        'Cyclin D': '0.C04.percentile99_class',
        'Mucin 2': '1.C03.percentile99_class',
        'AldoB': '1.C04.percentile99_class',
        'Glucagon': '2.C02.percentile99_class',
        'Cyclin A': '2.C03.percentile99_class',
        'Agr2': '2.C04.percentile99_class',
        'Serotonin': '3.C02.percentile99_class',
        'Lysozyme': '3.C03.percentile99_class',
    }

    def __init__(self):
        super().__init__()
        # Marker-related containers
        self.marker_fields = None      # (V, P) per-vertex marker data
        self.marker_names = []         # list of marker names (standardized)
        self.cell_label_field = None   # (V,) optional reference positional field
        self.coeffs_fm = None          # spectral coefficients of marker fields

    # -------------------------------------------------------------------------
    # --- Loading mesh + markers from VTP
    # -------------------------------------------------------------------------

    def load_mesh_with_markers_from_vtp(self, path, filter_lgr5=True):
        """
        Load vertices, faces, and vertex-associated marker fields from a VTP file.
        Extracts standardized fate markers and optionally filters LGR5 coexpression.
        """
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

        # Extract all point fields
        point_data = polydata.GetPointData()
        field_names = []
        all_fields = []

        for i in range(point_data.GetNumberOfArrays()):
            array = point_data.GetArray(i)
            field_names.append(array.GetName())
            all_fields.append(np.array(array))

        all_fields = np.array(all_fields).T  # (V, F)

        self.cell_label_field, self.marker_fields, self.marker_names = \
            self.extract_markers_from_raw(all_fields, field_names, filter_lgr5=filter_lgr5)

        return self

    def load_from_arrays_with_markers(self, vertices, faces,
                                      marker_fields, marker_names,
                                      filter_lgr5=True):
        """
        Load mesh geometry plus raw marker fields from arrays, then extract the
        standardized marker set.
        """
        self.v = np.asarray(vertices)
        self.f = np.asarray(faces, dtype=np.int64)
        self.cell_label_field, self.marker_fields, self.marker_names = \
            self.extract_markers_from_raw(marker_fields, marker_names,
                                          filter_lgr5=filter_lgr5)
        return self

    # -------------------------------------------------------------------------
    # --- Marker extraction and filtering 
    # -------------------------------------------------------------------------

    @staticmethod
    def _remap_labels_to_contiguous(labels):
        """
        Remap possibly non-contiguous labels (and -1 for unlabeled) into contiguous [0..C-1].
        Returns labels_contig, mapping dict old->new.
        """
        labs = np.asarray(labels)
        uniq = np.unique(labs[labs >= 0])
        mapping = {int(old): new for new, old in enumerate(uniq)}
        labels_contig = np.full_like(labs, -1)
        for old, new in mapping.items():
            labels_contig[labs == old] = new
        return labels_contig, mapping

    def extract_markers_from_raw(self, raw_fields, raw_names, filter_lgr5=True):
        """
        Extract standardized fate markers from raw field arrays and optionally
        filter LGR5 coexpression.
        """
        field_names = list(raw_names)
        if "4.label" not in field_names:
            print(field_names)
            raise ValueError("Reference field '4.label' not found in raw field names.")
        ref_idx = field_names.index("4.label")
        ref_field = np.asarray(raw_fields)[:, ref_idx]

        markers_list = []
        marker_names = []
        for human_name, raw_key in self.annotation_names.items():
            if raw_key not in field_names:
                raise ValueError(f"Marker field '{raw_key}' for '{human_name}' not found in raw names.")
            idx = field_names.index(raw_key)
            markers_list.append(np.asarray(raw_fields)[:, idx])
            marker_names.append(human_name)

        markers_vertex = np.column_stack(markers_list)  # (V, P)

        # remap label ids to contiguous 0..C-1
        ref_field, _ = self._remap_labels_to_contiguous(ref_field)
        ref_field = np.asarray(ref_field, dtype=np.int32)

        if filter_lgr5:
            markers_vertex = self.filter_lgr5_coexpression(markers_vertex, marker_names)

        return ref_field, markers_vertex, marker_names

    @staticmethod
    def filter_lgr5_coexpression(
        markers_vertex,
        marker_names,
        coexpress_markers=("Lysozyme", "Mucin 2", "Agr2", "Serotonin", "Glucagon", "Chromogranin A"),
        lgr5_marker="LGR5",
    ):
        """
        Set LGR5+ to 0 in cells coexpressing any of a set of differentiated markers.
        """
        filtered = markers_vertex.copy()
        try:
            lgr5_idx = marker_names.index(lgr5_marker)
        except ValueError:
            raise ValueError(f"LGR5 marker '{lgr5_marker}' not found in marker_names.")

        coexpr_idx = [marker_names.index(m) for m in coexpress_markers if m in marker_names]
        if not coexpr_idx:
            return filtered

        coexpr_mask = filtered[:, coexpr_idx].any(axis=1)
        filtered[coexpr_mask, lgr5_idx] = 0
        return filtered

    # -------------------------------------------------------------------------
    # --- Spectral coefficients including marker fields
    # -------------------------------------------------------------------------

    def compute_spectral_coefficients(self, lmax=15):
        """
        Project vertex coordinates and (if present) marker fields onto the Laplacian
        eigenbasis. Extends the base implementation by also computing coeffs_fm.
        """
        self.lmax = lmax
        k = int(lmax ** 2)
        self._ensure_eigendecomposition(k=k)

        self.coeffs_v = self.eigvecs.T @ (self.mass_matrix @ self.v)
        self.coeffs_fm = None
        if self.marker_fields is not None:
            self.coeffs_fm = self.eigvecs.T @ (self.mass_matrix @ self.marker_fields)
        return self.coeffs_v, self.coeffs_fm

    # -------------------------------------------------------------------------
    # --- Assign per-vertex markers from a cell graph + vertex_owner
    # -------------------------------------------------------------------------

    def assign_markers_from_graph(
        self,
        G,
        vertex_owner,
        marker_attr="markers_bin",
        marker_names=None,
        fill_value=0,
    ):
        """
        Given a cell graph and a vertex_owner array, construct per-vertex
        marker fields on the mesh by copying each cell's markers onto all
        vertices owned by that cell.

        Parameters
        ----------
        G : networkx.Graph
            Graph whose nodes represent cells. Node IDs should be integers
            0..N_cells-1 (matching indices used in vertex_owner).
            Each node must have an attribute `marker_attr` giving a 1D array
            (or list) of markers for that cell.
        vertex_owner : (V,) ndarray of int
            For each mesh vertex v, vertex_owner[v] is the cell index (node id)
            that owns this vertex (e.g., from Voronoi partition).
            A negative value can indicate "unassigned".
        marker_attr : str, default "markers_bin"
            Name of the node attribute containing the cell marker vector.
        marker_names : list of str, optional
            Names of the marker channels. If given, they will be stored in
            self.marker_names.
        fill_value : scalar, default 0
            Value used for vertices with owner < 0 (if any).

        Sets
        ----
        self.marker_fields : (V, P) ndarray
            Per-vertex marker fields constructed from the graph.
        self.marker_names : list of str (if marker_names is not None)
        """
        vertex_owner = np.asarray(vertex_owner)
        V = self.v.shape[0]

        if vertex_owner.shape[0] != V:
            raise ValueError(
                f"vertex_owner has length {vertex_owner.shape[0]} but mesh has {V} vertices."
            )

        # Infer marker dimension P from the first node
        nodes = list(G.nodes)
        if len(nodes) == 0:
            raise ValueError("Graph has no nodes.")
        first_val = G.nodes[nodes[0]].get(marker_attr, None)
        if first_val is None:
            raise ValueError(f"Node attribute '{marker_attr}' not found on graph nodes.")

        first_arr = np.asarray(first_val)
        if first_arr.ndim == 0:
            P = 1
        else:
            P = first_arr.shape[0]

        markers_vertex = np.full((V, P), fill_value, dtype=float)

        for v_idx in range(V):
            cell_idx = int(vertex_owner[v_idx])
            if cell_idx < 0:
                continue
            if cell_idx not in G.nodes:
                # In case vertex_owner refers to a cell not present in G
                continue
            cell_markers = np.asarray(G.nodes[cell_idx][marker_attr], dtype=float)
            if cell_markers.ndim == 0:
                markers_vertex[v_idx, 0] = cell_markers
            else:
                markers_vertex[v_idx, :] = cell_markers

        self.marker_fields = markers_vertex
        if marker_names is not None:
            self.marker_names = list(marker_names)

        return markers_vertex

    # -------------------------------------------------------------------------
    # --- Per-cell utilities 
    # -------------------------------------------------------------------------

    def get_centroid_vertices(self, labels=None):
        """
        For each unique label in `labels` (vertex-wise), compute the patch centroid
        and return the vertex index closest to that centroid.
        """
        if labels is None:
            labels = self.cell_label_field
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        centers_idx = np.empty(len(unique_labels), dtype=int)

        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            patch_vertices = self.v[mask]
            centroid = patch_vertices.mean(axis=0)
            patch_indices = np.nonzero(mask)[0]
            diffs = self.v[patch_indices] - centroid
            dists = np.einsum("ij,ij->i", diffs, diffs)
            best_idx = patch_indices[np.argmin(dists)]
            centers_idx[i] = int(best_idx)
        return centers_idx

    def compute_cell_statistics(self, vertex_fields=None):
        """
        Aggregate vertex-level data into per-cell statistics using the mesh's
        stored cell labels and vertex areas.

        Vertex areas are obtained from self.calc_vertex_areas().
        Cell labels are taken from self.cell_label_field.
        Marker fields can optionally be provided; if None, self.marker_fields is used.
        """
        if vertex_fields is None:
            if self.marker_fields is None:
                raise ValueError("No vertex_fields provided and self.marker_fields is None.")
            vertex_fields = self.marker_fields

        vf = np.asarray(vertex_fields)
        if vf.ndim == 1:
            vf = vf[:, None]

        V = self.v.shape[0]
        if self.cell_label_field is None:
            raise ValueError("cell_label_field is None; cannot compute cell statistics.")
        C = int(self.cell_label_field.max() + 1) if (self.cell_label_field >= 0).any() else 0

        vertex_areas = self.vertex_areas()

        weighted_pos = np.zeros((C, 3), dtype=float)
        weighted_field = np.zeros((C, vf.shape[1]), dtype=float)
        cell_area_v = np.zeros(C, dtype=float)

        for idx in range(V):
            lab = int(self.cell_label_field[idx])
            if lab < 0:
                continue
            a = float(vertex_areas[idx])
            weighted_pos[lab] += a * self.v[idx]
            weighted_field[lab] += a * vf[idx]
            cell_area_v[lab] += a

        valid = cell_area_v > 0
        centroids = np.zeros((C, 3), dtype=float)
        fields = np.zeros((C, vf.shape[1]), dtype=float)
        centroids[valid] = weighted_pos[valid] / cell_area_v[valid, None]
        fields[valid] = weighted_field[valid] / cell_area_v[valid, None]

        return centroids, cell_area_v, fields, valid

    # -------------------------------------------------------------------------
    # --- Saving / loading including markers
    # -------------------------------------------------------------------------

    def save_results(self, path):
        """
        Save mesh, markers, eigenmodes, and coefficients to a pickle file.
        """
        data = dict(
            v=self.v,
            f=self.f,
            marker_fields=self.marker_fields,
            marker_names=self.marker_names,
            cell_label_field=self.cell_label_field,
            eigvals=self.eigvals,
            eigvecs=self.eigvecs,
            mass_matrix=self.mass_matrix,
            coeffs_v=self.coeffs_v,
            coeffs_fm=self.coeffs_fm,
            transform_matrix=self.transform_matrix,
            lmax=self.lmax,
        )
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_results(self, path):
        """
        Load previously saved mesh, markers, and spectral analysis results.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            setattr(self, k, v)
        return self

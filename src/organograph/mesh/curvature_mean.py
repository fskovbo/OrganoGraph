"""
Refactored curvature pipeline for closed triangular organoid meshes.

Main design choice
------------------
The default pipeline first defines an effective/smoothed surface by spectral
Laplace--Beltrami reconstruction. Both Gaussian curvature K and mean curvature H
are then computed on that same surface:

    K: angle defect divided by vertex area
    H: cotangent Laplace--Beltrami mean-curvature normal, 0.5 * M^{-1} L X

HKS is used by default for short-scale defect/debris detection only. The same
mask is then applied to both H and K before neighbor-based inpainting.

Expected mesh interface
-----------------------
The input mesh should provide at least:
    mesh.v, mesh.f
    mesh.build_cotangent_laplacian_and_mass(v, f)
    mesh.compute_spectral_coefficients(lmax)
    mesh.reconstruct_from_coeffs(coeffs, lmax)
    mesh.vertex_areas(...)

This is compatible with the OrganoidMesh class you attached.
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

try:
    from organograph.mesh.hks import compute_hks
except Exception:  # pragma: no cover - keeps non-HKS parts importable
    compute_hks = None


# =============================================================================
# Basic mesh utilities
# =============================================================================


def ensure_laplacian_and_mass(mesh) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Ensure mesh.laplacian and mesh.mass_matrix exist and return them."""
    if getattr(mesh, "laplacian", None) is None or getattr(mesh, "mass_matrix", None) is None:
        L, M = mesh.build_cotangent_laplacian_and_mass(mesh.v, mesh.f)
        mesh.laplacian = sparse.csr_matrix(L)
        mesh.mass_matrix = sparse.csr_matrix(M)
    return mesh.laplacian, mesh.mass_matrix


def mesh_neighbors_from_faces(mesh) -> list[np.ndarray]:
    """Build 1-ring vertex-neighbor lists from triangular faces."""
    V = len(mesh.v)
    faces = np.asarray(mesh.f, dtype=np.int64)
    neigh = [set() for _ in range(V)]
    for tri in faces:
        i, j, k = map(int, tri)
        neigh[i].update((j, k))
        neigh[j].update((i, k))
        neigh[k].update((i, j))
    return [np.fromiter(s, dtype=np.int64) for s in neigh]


def face_areas(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Triangle areas."""
    tri = v[f]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def vertex_areas_barycentric(mesh) -> np.ndarray:
    """Barycentric per-vertex areas, independent of the mass matrix."""
    v = np.asarray(mesh.v, dtype=float)
    f = np.asarray(mesh.f, dtype=np.int64)
    A_face = face_areas(v, f)
    A = np.zeros(len(v), dtype=float)
    for a in range(3):
        np.add.at(A, f[:, a], A_face / 3.0)
    return A


def vertex_areas(mesh, area_mode: str = "mass") -> np.ndarray:
    """Per-vertex surface areas."""
    if area_mode == "mass":
        ensure_laplacian_and_mass(mesh)
        return np.asarray(mesh.mass_matrix.diagonal(), dtype=float)
    if area_mode == "barycentric":
        return vertex_areas_barycentric(mesh)
    raise ValueError("area_mode must be 'mass' or 'barycentric'")


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """Median/MAD robust z-score."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    denom = 1.4826 * mad
    if not np.isfinite(denom) or denom <= 0:
        denom = np.nanstd(x)
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    return (x - med) / denom


def dilate_vertex_mask(mask: np.ndarray, neighbors: list[np.ndarray], n_steps: int = 1) -> np.ndarray:
    """Expand a vertex mask by repeated 1-ring dilation."""
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(n_steps)):
        new = out.copy()
        for i in np.flatnonzero(out):
            nb = neighbors[i]
            if nb.size:
                new[nb] = True
        out = new
    return out


def inpaint_by_neighbor_averaging(
    values: np.ndarray,
    mask: np.ndarray,
    neighbors: list[np.ndarray],
    n_iter: int = 30,
) -> np.ndarray:
    """
    Inpaint masked entries by iterative neighbor averaging.

    Existing finite, unmasked entries are held fixed. Masked entries are filled
    from their finite neighbors and may then propagate inward.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same shape")

    out = values.copy()
    out[mask] = np.nan
    fixed = np.isfinite(out)

    for _ in range(int(n_iter)):
        changed = False
        new = out.copy()
        for i in np.flatnonzero(~fixed):
            vals = out[neighbors[i]]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                new[i] = float(np.mean(vals))
                changed = True
        out = new
        fixed = np.isfinite(out)
        if not changed:
            break
    return out


# =============================================================================
# Smoothing
# =============================================================================


def spectral_smooth_mesh(
    mesh,
    lmax: int = 15,
    recompute_operators: bool = True,
    copy_eigensystem: bool = False,
):
    """
    Reconstruct a smoothed mesh using the lowest Laplace--Beltrami modes.

    The returned mesh has the same class and face connectivity as `mesh`, but
    with smoothed vertex coordinates. Operators are recomputed on the smoothed
    geometry by default; this is recommended before computing curvature.
    """
    if lmax <= 0:
        raise ValueError("lmax must be positive")

    mesh.compute_spectral_coefficients(lmax=lmax)
    v_smooth = np.asarray(mesh.reconstruct_from_coeffs(mesh.coeffs_v, lmax=lmax), dtype=float)

    smoothed = mesh.__class__()
    smoothed.load_from_arrays(v_smooth, np.asarray(mesh.f, dtype=np.int64).copy())

    if recompute_operators:
        smoothed.laplacian, smoothed.mass_matrix = smoothed.build_cotangent_laplacian_and_mass(
            smoothed.v, smoothed.f
        )

    if copy_eigensystem:
        # Usually not needed for curvature; useful only if downstream code expects these fields.
        smoothed.eigvals = getattr(mesh, "eigvals", None)
        smoothed.eigvecs = getattr(mesh, "eigvecs", None)
        smoothed.lmax = lmax

    return smoothed


def smooth_scalar_field_lbo(mesh, field, diffusion_time=0.05, n_steps=1):
    """
    Smooth scalar field by implicit LB diffusion:

        (M + t L) u_new = M u_old

    Uses the same cotan Laplacian/mass matrix as the curvature pipeline.
    """
    import numpy as np
    import scipy.sparse.linalg as spla

    if mesh.laplacian is None or mesh.mass_matrix is None:
        mesh.laplacian, mesh.mass_matrix = mesh.build_cotangent_laplacian_and_mass(
            mesh.v, mesh.f
        )

    L = mesh.laplacian
    M = mesh.mass_matrix

    u = np.asarray(field, float).copy()

    A = M + diffusion_time * L

    for _ in range(int(n_steps)):
        rhs = M @ u
        u = spla.spsolve(A, rhs)

    return u

# =============================================================================
# Curvature estimators on a fixed geometry
# =============================================================================


def vertex_normals(mesh, outward: bool = True) -> np.ndarray:
    """Area-weighted vertex normals, optionally oriented away from centroid."""
    v = np.asarray(mesh.v, dtype=float)
    f = np.asarray(mesh.f, dtype=np.int64)
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])

    n = np.zeros_like(v, dtype=float)
    for a in range(3):
        np.add.at(n, f[:, a], fn)

    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-15)

    if outward:
        radial = v - np.mean(v, axis=0, keepdims=True)
        orientation_score = np.nanmedian(np.einsum("ij,ij->i", n, radial))
        if orientation_score < 0:
            n = -n
    return n


def compute_mean_curvature_lbo(
    mesh,
    signed: bool = True,
    outward_normals: bool = True,
    solver: str = "diag",
) -> np.ndarray:
    """
    Mean curvature from the cotangent Laplace--Beltrami mean-curvature normal.

    For the positive semidefinite operator L = -cotmatrix, M^{-1} L X approximates
    -Delta_S X = 2 H n. Therefore Hn = 0.5 * M^{-1} L X.

    Parameters
    ----------
    signed:
        If True, return H = <Hn, n>. If False, return ||Hn||.
    solver:
        'diag' uses the lumped diagonal mass matrix and is usually appropriate
        for libigl Voronoi mass matrices. 'spsolve' solves the sparse mass system.
    """
    L, M = ensure_laplacian_and_mass(mesh)
    X = np.asarray(mesh.v, dtype=float)
    LX = L @ X

    if solver == "diag":
        m = np.asarray(M.diagonal(), dtype=float)
        if np.any(m <= 0) or np.any(~np.isfinite(m)):
            raise ValueError("Mass matrix has non-positive or non-finite diagonal entries")
        Minv_LX = LX / m[:, None]
    elif solver == "spsolve":
        Minv_LX = np.column_stack([spla.spsolve(M, LX[:, d]) for d in range(3)])
    else:
        raise ValueError("solver must be 'diag' or 'spsolve'")

    Hn = 0.5 * Minv_LX
    if not signed:
        return np.linalg.norm(Hn, axis=1)

    n = vertex_normals(mesh, outward=outward_normals)
    return np.einsum("ij,ij->i", Hn, n)


def compute_gaussian_curvature_angle_defect(
    mesh,
    area_mode: str = "mass",
    boundary_value: float = np.nan,
) -> np.ndarray:
    """
    Gaussian curvature from angle defect divided by vertex area.

    For closed genus-0 surfaces, sum_i K_i A_i should be close to 4*pi.
    Boundary vertices, if any, are assigned `boundary_value` because the closed
    angle-defect formula does not apply there.
    """
    v = np.asarray(mesh.v, dtype=float)
    f = np.asarray(mesh.f, dtype=np.int64)
    V = len(v)

    angle_sum = np.zeros(V, dtype=float)

    # Angles at each corner.
    for local in range(3):
        i = f[:, local]
        j = f[:, (local + 1) % 3]
        k = f[:, (local + 2) % 3]
        u = v[j] - v[i]
        w = v[k] - v[i]
        nu = np.linalg.norm(u, axis=1)
        nw = np.linalg.norm(w, axis=1)
        cosang = np.einsum("ij,ij->i", u, w) / np.maximum(nu * nw, 1e-15)
        angles = np.arccos(np.clip(cosang, -1.0, 1.0))
        np.add.at(angle_sum, i, angles)

    A = vertex_areas(mesh, area_mode=area_mode)
    K = (2.0 * np.pi - angle_sum) / A

    boundary = boundary_vertices(mesh)
    if np.any(boundary):
        K[boundary] = boundary_value
    return K


def boundary_vertices(mesh) -> np.ndarray:
    """Return boolean mask of vertices incident to boundary edges."""
    f = np.asarray(mesh.f, dtype=np.int64)
    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges_unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = edges_unique[counts == 1]
    mask = np.zeros(len(mesh.v), dtype=bool)
    if boundary_edges.size:
        mask[np.unique(boundary_edges)] = True
    return mask


# =============================================================================
# HKS-based defect detection
# =============================================================================


def compute_length_scale(mesh, area_mode: str = "mass") -> float:
    """Characteristic surface length sqrt(total area)."""
    return float(np.sqrt(np.sum(vertex_areas(mesh, area_mode=area_mode))))


def rescale_times_from_tau(
    mesh,
    tau_ref: list[float] | tuple[float, ...] | np.ndarray,
    area_mode: str = "mass",
) -> tuple[np.ndarray, float]:
    """Convert dimensionless tau values to mesh times t = tau * L_mesh^2."""
    L_mesh = compute_length_scale(mesh, area_mode=area_mode)
    return np.asarray(tau_ref, dtype=float) * L_mesh**2, L_mesh


def fit_line_prefixes(t: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized linear fits for prefixes of the time axis."""
    t = np.asarray(t, dtype=float)
    Y = np.asarray(Y, dtype=float)
    V, T = Y.shape
    n = np.arange(1, T + 1, dtype=float)

    c_t = np.cumsum(t)
    c_t2 = np.cumsum(t**2)
    c_y = np.cumsum(Y, axis=1)
    c_y2 = np.cumsum(Y**2, axis=1)
    c_ty = np.cumsum(Y * t[None, :], axis=1)

    den = n * c_t2 - c_t**2
    den = np.where(np.abs(den) < 1e-15, np.nan, den)
    slopes = (n[None, :] * c_ty - c_t[None, :] * c_y) / den[None, :]
    intercepts = (c_y - slopes * c_t[None, :]) / n[None, :]

    sse = (
        c_y2
        + slopes**2 * c_t2[None, :]
        + n[None, :] * intercepts**2
        + 2.0 * slopes * intercepts * c_t[None, :]
        - 2.0 * slopes * c_ty
        - 2.0 * intercepts * c_y
    )
    mean_y = c_y / n[None, :]
    sst = c_y2 - n[None, :] * mean_y**2
    r2 = 1.0 - sse / np.where(sst > 1e-15, sst, np.nan)
    r2 = np.where(sst <= 1e-15, 1.0, r2)
    return slopes, intercepts, r2


def choose_early_window(
    t: np.ndarray,
    Y: np.ndarray,
    min_points: int = 3,
    r2_drop: float = 0.02,
    min_r2: float = 0.985,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose per-vertex early-time prefix fit window."""
    slopes, intercepts, r2 = fit_line_prefixes(t, Y)
    V, T = Y.shape
    if T < min_points:
        raise ValueError("Not enough HKS time points for early-window fit")

    end_idx = np.full(V, min_points - 1, dtype=np.int64)
    for j in range(min_points, T):
        prev = r2[:, j - 1]
        cur = r2[:, j]
        good = np.isfinite(cur) & (cur >= min_r2) & ((prev - cur) <= r2_drop)
        end_idx[good] = j
    return end_idx, slopes, intercepts, r2


def gather_by_index(arr2d: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Gather arr2d[i, idx[i]] for every row i."""
    rows = np.arange(arr2d.shape[0], dtype=np.int64)
    return arr2d[rows, np.asarray(idx, dtype=np.int64)]


def detect_defects_hks(
    mesh,
    tau_ref: list[float] | tuple[float, ...] | np.ndarray | None = None,
    zmax: float = 5.0,
    min_points: int = 3,
    r2_drop: float = 0.02,
    min_r2: float = 0.985,
    dilation_steps: int = 2,
    positive_gate: np.ndarray | None = None,
    positive_gate_only: bool = True,
    area_mode: str = "mass",
    return_diag: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Detect short-scale debris/defects using early-time HKS slope outliers.

    The score is a robust z-score of the early slope of
        4*pi*t*HKS(x,t) - 1.
    If `positive_gate` is supplied and `positive_gate_only` is True, only
    vertices with positive gate values are allowed to be marked as defects.
    """
    if compute_hks is None:
        raise ImportError("organograph.mesh.hks.compute_hks could not be imported")

    if tau_ref is None:
        tau_ref = np.geomspace(5e-5, 3e-2, 24)

    ensure_laplacian_and_mass(mesh)
    ts, L_mesh = rescale_times_from_tau(mesh, tau_ref, area_mode=area_mode)
    hks = np.asarray(compute_hks(mesh, ts, coeffs=False), dtype=float)
    Y = 4.0 * np.pi * hks * ts[None, :] - 1.0

    end_idx, slopes, intercepts, r2 = choose_early_window(
        ts, Y, min_points=min_points, r2_drop=r2_drop, min_r2=min_r2
    )
    early_slope = gather_by_index(slopes, end_idx)
    early_intercept = gather_by_index(intercepts, end_idx)
    early_r2 = gather_by_index(r2, end_idx)

    score = robust_zscore(early_slope)
    mask = score > float(zmax)
    if positive_gate_only and positive_gate is not None:
        gate = np.asarray(positive_gate, dtype=float)
        mask &= np.isfinite(gate) & (gate > 0)

    neighbors = mesh_neighbors_from_faces(mesh)
    mask_pre_dilation = mask.copy()
    if dilation_steps > 0:
        mask = dilate_vertex_mask(mask, neighbors, n_steps=dilation_steps)

    debug = {}
    if return_diag:
        debug = {
            "tau_ref": np.asarray(tau_ref, dtype=float),
            "ts": ts,
            "L_mesh": L_mesh,
            "hks": hks,
            "fit_signal": Y,
            "early_end_idx": end_idx,
            "early_slope": early_slope,
            "early_intercept": early_intercept,
            "early_r2": early_r2,
            "defect_score": score,
            "defect_mask_pre_dilation": mask_pre_dilation,
            "defect_mask": mask,
            "zmax": float(zmax),
        }
    return mask, score, debug


# =============================================================================
# Diagnostics and master pipeline
# =============================================================================


def integrate_curvature(mesh, curvature: np.ndarray, area_mode: str = "mass") -> float:
    """Compute sum_i curvature_i * area_i over finite vertices."""
    c = np.asarray(curvature, dtype=float)
    A = vertex_areas(mesh, area_mode=area_mode)
    finite = np.isfinite(c) & np.isfinite(A)
    return float(np.sum(c[finite] * A[finite]))



def renormalize_gaussian_curvature_to_4pi(mesh, curvature: np.ndarray, area_mode: str = "mass") -> tuple[np.ndarray, dict]:
    """
    Rescale an inpainted Gaussian-curvature field so that integral K dA = 4*pi.

    This should be applied only after masking and inpainting. Raw/debug
    curvature fields should be kept unnormalized.
    """
    K = np.asarray(curvature, dtype=float).copy()
    total_before = integrate_curvature(mesh, K, area_mode=area_mode)
    target = 4.0 * np.pi

    if np.isfinite(total_before) and abs(total_before) > 1e-15:
        scale = target / total_before
    else:
        scale = 1.0

    K_final = K * scale
    total_after = integrate_curvature(mesh, K_final, area_mode=area_mode)
    info = {
        "gaussian_renormalized": True,
        "gaussian_renormalization_target": float(target),
        "gaussian_integral_before_renormalization": float(total_before),
        "gaussian_integral_after_renormalization": float(total_after),
        "gaussian_renormalization_scale": float(scale),
    }
    return K_final, info


def curvature_diagnostics(
    mesh,
    H: np.ndarray,
    K: np.ndarray,
    defect_mask: np.ndarray | None = None,
    area_mode: str = "mass",
) -> dict:
    """Useful scalar checks for a closed genus-0 organoid."""
    A = vertex_areas(mesh, area_mode=area_mode)
    total_area = float(np.sum(A))
    int_K = integrate_curvature(mesh, K, area_mode=area_mode)
    out = {
        "total_area": total_area,
        "integrated_gaussian_curvature": int_K,
        "gauss_bonnet_target_genus0": float(4.0 * np.pi),
        "gauss_bonnet_error": float(int_K - 4.0 * np.pi),
        "gauss_bonnet_relative_error": float((int_K - 4.0 * np.pi) / (4.0 * np.pi)),
        "estimated_euler_characteristic": float(int_K / (2.0 * np.pi)),
        "mean_H": float(np.nanmean(H)),
        "median_H": float(np.nanmedian(H)),
        "std_H": float(np.nanstd(H)),
        "mean_K": float(np.nanmean(K)),
        "median_K": float(np.nanmedian(K)),
        "std_K": float(np.nanstd(K)),
        "n_vertices": int(len(mesh.v)),
        "has_boundary": bool(np.any(boundary_vertices(mesh))),
    }
    if defect_mask is not None:
        out["n_defect_vertices"] = int(np.sum(defect_mask))
        out["defect_fraction"] = float(np.mean(defect_mask))
    return out


def compute_organoid_curvatures(
    mesh,
    lmax: int = 12,
    defect_detection: str = "hks",
    defect_detection_mesh: str = "raw",
    defect_tau_ref: list[float] | tuple[float, ...] | np.ndarray | None = None,
    defect_zmax: float = 5.0,
    defect_dilation_steps: int = 3,
    inpaint_iters: int = 30,
    diffusion_smoothen_time: float = 0.1,
    area_mode: str = "mass",
    mean_signed: bool = True,
    mean_solver: str = "diag",
    positive_defects_only: bool = True,
    return_diag: bool = False,
):
    """
    Master pipeline for one organoid.

    Steps
    -----
    1. Spectrally smooth the mesh by retaining low LB modes up to `lmax`.
    2. Compute H and K on that same smoothed geometry.
    3. Detect short-scale defects using HKS, by default on the raw mesh.
       The topology is assumed unchanged, so the mask transfers vertexwise.
    4. Apply the same mask to H and K and inpaint both fields.
    5. Renormalize only the inpainted Gaussian curvature so integral K dA = 4*pi.
    6. Return final Gaussian and mean curvature fields; optionally return diagnostics.

    Notes
    -----
    - K is angle-defect / vertex-area on the smoothed mesh by default; only the final inpainted K is Gauss--Bonnet renormalized.
    - H is 0.5 * M^{-1} L X projected onto outward vertex normals.
    - HKS is not used as the final Gaussian-curvature estimator by default;
      it is used to identify debris/defect regions.
    """
    if lmax <= 0:
        raise ValueError("lmax must be positive")

    smoothed = spectral_smooth_mesh(mesh, lmax=lmax, recompute_operators=True)
    neighbors = mesh_neighbors_from_faces(smoothed)

    H_raw = compute_mean_curvature_lbo(
        smoothed, signed=mean_signed, outward_normals=True, solver=mean_solver
    )
    K_raw = compute_gaussian_curvature_angle_defect(smoothed, area_mode=area_mode)

    hks_debug = {}
    if defect_detection == "none":
        defect_mask = np.zeros(len(smoothed.v), dtype=bool)
        defect_score = np.zeros(len(smoothed.v), dtype=float)
    elif defect_detection == "hks":
        detector_mesh = mesh if defect_detection_mesh == "raw" else smoothed
        # If using raw detection, use raw K is not geometrically comparable after smoothing;
        # use smoothed K as positivity gate because the mask transfers vertexwise.
        positive_gate = K_raw if positive_defects_only else None
        defect_mask, defect_score, hks_debug = detect_defects_hks(
            detector_mesh,
            tau_ref=defect_tau_ref,
            zmax=defect_zmax,
            dilation_steps=defect_dilation_steps,
            positive_gate=positive_gate,
            positive_gate_only=positive_defects_only,
            area_mode=area_mode,
            return_diag=return_diag,
        )
    else:
        raise ValueError("defect_detection must be 'hks' or 'none'")

    if defect_mask.shape != H_raw.shape:
        raise ValueError("Defect mask shape does not match curvature field shape")

    H_inpainted = inpaint_by_neighbor_averaging(H_raw, defect_mask, neighbors, n_iter=inpaint_iters)
    H = smooth_scalar_field_lbo(smoothed, H_inpainted, diffusion_time=diffusion_smoothen_time, n_steps=1,)

    K_inpainted = inpaint_by_neighbor_averaging(K_raw, defect_mask, neighbors, n_iter=inpaint_iters)
    K_inpainted = smooth_scalar_field_lbo(smoothed, K_inpainted, diffusion_time=diffusion_smoothen_time, n_steps=1,)
    K, gaussian_renorm_info = renormalize_gaussian_curvature_to_4pi(
        smoothed, K_inpainted, area_mode=area_mode
    )

    diag = curvature_diagnostics(smoothed, H, K, defect_mask=defect_mask, area_mode=area_mode)
    diag.update(
        {
            "lmax": int(lmax),
            "defect_detection": defect_detection,
            "defect_detection_mesh": defect_detection_mesh,
            "area_mode": area_mode,
            "mean_signed": bool(mean_signed),
            "mean_solver": mean_solver,
            "inpaint_iters": int(inpaint_iters),
            **gaussian_renorm_info,
        }
    )
    if return_diag:
        diag["hks_debug"] = hks_debug
        diag["mean_curvature_raw"] = H_raw
        diag["gaussian_curvature_raw"] = K_raw
        diag["gaussian_curvature_inpainted_unrenormalized"] = K_inpainted
        diag["defect_mask"] = defect_mask
        diag["defect_score"] = defect_score
        diag["smoothed_mesh"] = smoothed
        return K, H, diag

    return K, H


# =============================================================================
# Optional scale sweep helper
# =============================================================================


def sweep_smoothing_scales(
    mesh,
    lmax_values: list[int] | tuple[int, ...] | np.ndarray,
    **pipeline_kwargs,
) -> dict[int, tuple]:
    """
    Run compute_organoid_curvatures for several lmax values.

    This is useful for finding a plateau where H, K, Gauss--Bonnet error, and
    eventually Helfrich parameters are stable under moderate smoothing changes.
    """
    results: dict[int, tuple] = {}
    for lmax in lmax_values:
        results[int(lmax)] = compute_organoid_curvatures(mesh, lmax=int(lmax), **pipeline_kwargs)
    return results
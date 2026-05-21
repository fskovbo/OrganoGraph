"""
Extrinsic rigid symmetry scores for low-pass organoid meshes.

The functions in this module deliberately score simple, interpretable rigid
symmetries only: reflection, C2 rotations, and C3 rotations.  Low-pass meshes
are reconstructed with the existing Laplace-Beltrami eigenbasis stored on
``OrganoidMesh``; centering and scale normalization happen explicitly in this
pipeline so distances remain biologically interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SymmetryScore:
    """Best score for one symmetry family at one smoothing level.

    Distances are normalized by ``characteristic_scale``.  Lower RMS/trimmed
    RMS/median means the transformed coarse surface lies closer to itself.
    Higher ``matched_fraction`` means a larger area fraction is explained by
    the tested symmetry within ``close_threshold`` organoid radii.
    """

    l: int | None
    symmetry: str
    axis_name: str
    axis: np.ndarray
    transformation_matrix: np.ndarray
    transformation_matrices: tuple[np.ndarray, ...]
    normalized_rms: float
    trimmed_rms: float
    median: float
    matched_fraction: float
    n_samples: int
    characteristic_scale: float
    centroid: np.ndarray
    trim_fraction: float
    close_threshold: float
    candidate_source: str = "PCA"
    per_transform_scores: tuple[dict[str, float], ...] = field(default_factory=tuple)

    def to_record(self, organoid_id: str | None = None) -> dict[str, object]:
        """Return a flat dictionary suitable for tables or CSV export."""
        record = {
            "l": self.l,
            "n_modes": None if self.l is None else int(self.l**2),
            "symmetry": self.symmetry,
            "best_axis": self.axis_name,
            "candidate_source": self.candidate_source,
            "axis_x": float(self.axis[0]),
            "axis_y": float(self.axis[1]),
            "axis_z": float(self.axis[2]),
            "trimmed_rms": float(self.trimmed_rms),
            "normalized_rms": float(self.normalized_rms),
            "median": float(self.median),
            "matched_fraction": float(self.matched_fraction),
            "n_samples": int(self.n_samples),
            "characteristic_scale": float(self.characteristic_scale),
            "close_threshold": float(self.close_threshold),
        }
        if organoid_id is not None:
            record = {"organoid_id": organoid_id, **record}
        return record


def laplace_beltrami_low_pass_vertices(mesh, l: int | None) -> np.ndarray:
    """Reconstruct mesh coordinates through LB level ``l``.

    Parameters
    ----------
    mesh
        ``OrganoidMesh``-like object with ``v``, ``f``, and the spectral
        attributes used by :class:`organograph.mesh.OrganoidMesh.OrganoidMesh`.
    l
        Laplace-Beltrami reconstruction level, matching the convention in
        ``OrganoidMesh``.  The number of retained modes is ``l**2``.  ``None``
        returns the original vertices and is useful for comparing raw and
        smoothed symmetry scores.

    Notes
    -----
    This function does not center, rotate, or rescale the mesh.  The constant
    LB mode is kept, so centroid/scale information is preserved as faithfully
    as the FEM projection allows.
    """
    vertices = np.asarray(mesh.v, dtype=float)
    if l is None:
        return vertices.copy()

    l = int(l)
    if l < 1:
        raise ValueError("l must be a positive integer or None.")
    n_modes = int(l**2)
    if n_modes >= vertices.shape[0]:
        raise ValueError("l**2 must be smaller than the number of vertices for eigsh.")

    eigvecs = getattr(mesh, "eigvecs", None)
    mass_matrix = getattr(mesh, "mass_matrix", None)
    have_enough_modes = eigvecs is not None and eigvecs.shape[1] >= n_modes

    if (
        hasattr(mesh, "compute_spectral_coefficients")
        and hasattr(mesh, "reconstruct_from_coeffs")
    ):
        if not have_enough_modes or mass_matrix is None:
            if not hasattr(mesh, "_eig_decomp"):
                raise TypeError("mesh must provide _eig_decomp(k=...) or precomputed eigvecs.")
            mesh._eig_decomp(k=n_modes)

        stored_l = getattr(mesh, "lmax", None)
        coeffs_v = getattr(mesh, "coeffs_v", None)
        if coeffs_v is None or stored_l is None or stored_l < l:
            mesh.compute_spectral_coefficients(lmax=l)
        return np.asarray(mesh.reconstruct_from_coeffs(mesh.coeffs_v, lmax=l), dtype=float)

    if mass_matrix is None:
        from organograph.mesh.OrganoidMesh import OrganoidMesh

        _laplacian, mass_matrix = OrganoidMesh.build_cotangent_laplacian_and_mass(
            vertices, np.asarray(mesh.f, dtype=np.int64)
        )
        mesh.mass_matrix = mass_matrix

    if not have_enough_modes:
        if not hasattr(mesh, "_eig_decomp"):
            raise TypeError(
                "mesh must provide OrganoidMesh reconstruction methods, "
                "_eig_decomp(k=...), or precomputed eigvecs."
            )
        mesh._eig_decomp(k=n_modes)
        eigvecs = mesh.eigvecs
        mass_matrix = mesh.mass_matrix

    eigvals = getattr(mesh, "eigvals", None)
    if eigvals is not None and len(eigvals) >= n_modes:
        mode_idx = np.argsort(np.asarray(eigvals, dtype=float))[:n_modes]
        basis = np.asarray(eigvecs[:, mode_idx], dtype=float)
    else:
        basis = np.asarray(eigvecs[:, :n_modes], dtype=float)
    coeffs = basis.T @ (mass_matrix @ vertices)
    return basis @ coeffs


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Triangle areas for a mesh."""
    tri = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=np.int64)]
    return 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )


def surface_area_centroid(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted surface centroid using triangle centroids."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    areas = face_areas(vertices, faces)
    total_area = float(np.sum(areas))
    if total_area <= 0:
        raise ValueError("Mesh has zero surface area.")
    triangle_centroids = vertices[faces].mean(axis=1)
    return (triangle_centroids * areas[:, None]).sum(axis=0) / total_area


def sample_surface_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_samples: int = 5000,
    *,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Sample points approximately uniformly by surface area.

    Faces are drawn with probability proportional to area and barycentric
    coordinates are uniform within each selected triangle.  The returned points
    therefore have equal statistical weight in downstream area-fraction scores.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    generator = np.random.default_rng(rng)
    areas = face_areas(vertices, faces)
    total_area = float(np.sum(areas))
    if total_area <= 0:
        raise ValueError("Mesh has zero surface area.")

    face_idx = generator.choice(faces.shape[0], size=int(n_samples), p=areas / total_area)
    tri = vertices[faces[face_idx]]

    u = generator.random(int(n_samples))
    v = generator.random(int(n_samples))
    sqrt_u = np.sqrt(u)
    w0 = 1.0 - sqrt_u
    w1 = sqrt_u * (1.0 - v)
    w2 = sqrt_u * v
    return tri[:, 0] * w0[:, None] + tri[:, 1] * w1[:, None] + tri[:, 2] * w2[:, None]


def characteristic_size(points: np.ndarray, method: str = "rms") -> float:
    """Characteristic radius used to make symmetry distances dimensionless."""
    points = np.asarray(points, dtype=float)
    if method == "rms":
        scale = float(np.sqrt(np.mean(np.sum(points**2, axis=1))))
    elif method == "bbox":
        scale = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)) / 2.0)
    else:
        raise ValueError("method must be 'rms' or 'bbox'.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Characteristic scale is not positive.")
    return scale


def pca_candidate_axes(points: np.ndarray) -> list[tuple[str, np.ndarray, str]]:
    """Return deterministic PCA axes as candidate symmetry directions.

    PCA is used as the first robust extrinsic proposal mechanism because the
    candidate directions are easy to explain biologically: long, middle, and
    short coarse axes of the smoothed organoid.
    """
    points = np.asarray(points, dtype=float)
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = []
    for i, idx in enumerate(order, start=1):
        axis = eigvecs[:, idx].astype(float)
        axis /= np.linalg.norm(axis)

        # Deterministic sign.  Axis and -axis are equivalent for these tests.
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0:
            axis = -axis
        axes.append((f"PCA{i}", axis, "PCA"))
    return axes


def lb_candidate_axes(*_args, **_kwargs) -> list[tuple[str, np.ndarray, str]]:
    """Placeholder for future LB-eigenfunction-derived candidate axes.

    Eigenvalues alone are scalar invariants and do not define directions.
    Deriving stable axes from eigenfunctions/eigenspaces is possible, but it
    needs project-specific validation around eigenvalue multiplicity, signs,
    nodal sets, and correlation with embedded coordinates.  For this first
    version we expose a clean hook and keep PCA as the robust default.
    """
    return []


def rotation_matrix(axis: np.ndarray, angle_radians: float) -> np.ndarray:
    """3D rotation matrix using Rodrigues' formula."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = float(np.cos(angle_radians))
    s = float(np.sin(angle_radians))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def reflection_matrix(normal: np.ndarray) -> np.ndarray:
    """Matrix for reflection through the plane through the origin."""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    return np.eye(3) - 2.0 * np.outer(normal, normal)


def score_transformation(
    source_points: np.ndarray,
    target_tree: cKDTree,
    matrix: np.ndarray,
    scale: float,
    *,
    trim_fraction: float = 0.95,
    close_threshold: float = 0.05,
) -> dict[str, float]:
    """Score one rigid transform by nearest-neighbor surface-sample distance."""
    if not 0 < trim_fraction <= 1:
        raise ValueError("trim_fraction must be in (0, 1].")
    transformed = np.asarray(source_points, dtype=float) @ np.asarray(matrix, dtype=float).T
    distances, _indices = target_tree.query(transformed, k=1)
    normalized = distances / float(scale)
    squared = normalized**2

    keep = max(1, int(ceil(trim_fraction * squared.shape[0])))
    trimmed_squared = np.partition(squared, keep - 1)[:keep]
    return {
        "normalized_rms": float(np.sqrt(np.mean(squared))),
        "trimmed_rms": float(np.sqrt(np.mean(trimmed_squared))),
        "median": float(np.median(normalized)),
        "matched_fraction": float(np.mean(normalized < close_threshold)),
    }


def _combine_transform_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    keys = ("normalized_rms", "trimmed_rms", "median", "matched_fraction")
    return {key: float(np.mean([score[key] for score in scores])) for key in keys}


def _score_candidate_family(
    symmetry: str,
    axis_name: str,
    axis: np.ndarray,
    source_points: np.ndarray,
    target_tree: cKDTree,
    scale: float,
    *,
    l: int | None,
    centroid: np.ndarray,
    trim_fraction: float,
    close_threshold: float,
    candidate_source: str,
) -> SymmetryScore:
    if symmetry == "reflection":
        matrices = [reflection_matrix(axis)]
    elif symmetry == "C2":
        matrices = [rotation_matrix(axis, np.pi)]
    elif symmetry == "C3":
        matrices = [
            rotation_matrix(axis, 2.0 * np.pi / 3.0),
            rotation_matrix(axis, 4.0 * np.pi / 3.0),
        ]
    else:
        raise ValueError("symmetry must be 'reflection', 'C2', or 'C3'.")

    per_transform_scores = [
        score_transformation(
            source_points,
            target_tree,
            matrix,
            scale,
            trim_fraction=trim_fraction,
            close_threshold=close_threshold,
        )
        for matrix in matrices
    ]
    combined = _combine_transform_scores(per_transform_scores)
    return SymmetryScore(
        l=l,
        symmetry=symmetry,
        axis_name=axis_name,
        axis=np.asarray(axis, dtype=float),
        transformation_matrix=matrices[0],
        transformation_matrices=tuple(matrices),
        normalized_rms=combined["normalized_rms"],
        trimmed_rms=combined["trimmed_rms"],
        median=combined["median"],
        matched_fraction=combined["matched_fraction"],
        n_samples=int(source_points.shape[0]),
        characteristic_scale=float(scale),
        centroid=np.asarray(centroid, dtype=float),
        trim_fraction=float(trim_fraction),
        close_threshold=float(close_threshold),
        candidate_source=candidate_source,
        per_transform_scores=tuple(per_transform_scores),
    )


def score_symmetry_at_level(
    mesh,
    l: int | None,
    *,
    n_samples: int = 5000,
    trim_fraction: float = 0.95,
    close_threshold: float = 0.05,
    scale_method: str = "rms",
    candidate_sources: Iterable[str] = ("pca",),
    rng: np.random.Generator | int | None = None,
) -> list[SymmetryScore]:
    """Score reflection, C2, and C3 symmetries for one smoothing level.

    The returned list contains the best candidate for each symmetry family.
    ``l=None`` scores the raw mesh; integer ``l`` scores the LB low-pass mesh
    reconstructed from the first ``l**2`` modes, following ``OrganoidMesh``.
    """
    all_scores = score_all_symmetry_candidates_at_level(
        mesh,
        l,
        n_samples=n_samples,
        trim_fraction=trim_fraction,
        close_threshold=close_threshold,
        scale_method=scale_method,
        candidate_sources=candidate_sources,
        rng=rng,
    )
    best_by_symmetry: list[SymmetryScore] = []
    for symmetry in ("reflection", "C2", "C3"):
        scored = [score for score in all_scores if score.symmetry == symmetry]
        best_by_symmetry.append(min(scored, key=lambda score: score.trimmed_rms))
    return best_by_symmetry


def score_all_symmetry_candidates_at_level(
    mesh,
    l: int | None,
    *,
    n_samples: int = 5000,
    trim_fraction: float = 0.95,
    close_threshold: float = 0.05,
    scale_method: str = "rms",
    candidate_sources: Iterable[str] = ("pca",),
    rng: np.random.Generator | int | None = None,
) -> list[SymmetryScore]:
    """Score every generated axis candidate for reflection, C2, and C3.

    The returned list contains one score per symmetry family and candidate
    axis, for example all nine PCA candidates: 3 axes x 3 symmetry groups.
    Use :func:`score_symmetry_at_level` when only the best axis per symmetry
    family is needed.
    """
    faces = np.asarray(mesh.f, dtype=np.int64)
    vertices = laplace_beltrami_low_pass_vertices(mesh, l)
    centroid = surface_area_centroid(vertices, faces)
    centered_vertices = vertices - centroid[None, :]

    points = sample_surface_points(centered_vertices, faces, n_samples=n_samples, rng=rng)
    scale = characteristic_size(points, method=scale_method)
    target_tree = cKDTree(points)

    candidates: list[tuple[str, np.ndarray, str]] = []
    requested_sources = {str(source).lower() for source in candidate_sources}
    if "pca" in requested_sources:
        candidates.extend(pca_candidate_axes(points))
    if "lb" in requested_sources:
        candidates.extend(lb_candidate_axes(mesh, l, centered_vertices, points))
    if not candidates:
        raise ValueError("No candidate axes were generated.")

    scores: list[SymmetryScore] = []
    for symmetry in ("reflection", "C2", "C3"):
        for axis_name, axis, source in candidates:
            scores.append(
                _score_candidate_family(
                    symmetry,
                    axis_name,
                    axis,
                    points,
                    target_tree,
                    scale,
                    l=l,
                    centroid=centroid,
                    trim_fraction=trim_fraction,
                    close_threshold=close_threshold,
                    candidate_source=source,
                )
            )
    return scores


def run_multiscale_symmetry_analysis(
    mesh,
    l_values: Iterable[int | None] = (3, 5, 8, 10),
    *,
    n_samples: int = 5000,
    trim_fraction: float = 0.95,
    close_threshold: float = 0.05,
    scale_method: str = "rms",
    candidate_sources: Iterable[str] = ("pca",),
    random_seed: int = 0,
    k_values: Iterable[int | None] | None = None,
) -> list[SymmetryScore]:
    """Run the symmetry scoring pipeline over several smoothing levels."""
    if k_values is not None:
        raise TypeError("Use l_values instead of k_values; OrganoidMesh uses lmax levels.")
    l_values = tuple(l_values)
    results: list[SymmetryScore] = []
    seed_sequence = np.random.SeedSequence(random_seed)
    child_seeds = seed_sequence.spawn(len(l_values))
    for l, child_seed in zip(l_values, child_seeds):
        rng = np.random.default_rng(child_seed)
        results.extend(
            score_symmetry_at_level(
                mesh,
                l,
                n_samples=n_samples,
                trim_fraction=trim_fraction,
                close_threshold=close_threshold,
                scale_method=scale_method,
                candidate_sources=candidate_sources,
                rng=rng,
            )
        )
    return results


def symmetry_results_to_records(
    results: Iterable[SymmetryScore],
    *,
    organoid_id: str | None = None,
) -> list[dict[str, object]]:
    """Convert result objects into table-like rows.

    Column meanings:
    - ``trimmed_rms``: robust normalized mismatch; lower means stronger coarse
      approximate symmetry.
    - ``matched_fraction``: sampled surface area fraction within
      ``close_threshold`` of the original shape; higher means more of the
      organoid surface is explained by the symmetry.
    - ``median``: typical normalized mismatch, less sensitive to outliers.
    """
    return [result.to_record(organoid_id=organoid_id) for result in results]


def best_symmetry_per_level(
    results: Iterable[SymmetryScore],
    *,
    metric: str = "trimmed_rms",
) -> dict[int | None, SymmetryScore]:
    """Select the lowest-scoring symmetry class at each smoothing level."""
    grouped: dict[int | None, list[SymmetryScore]] = {}
    for result in results:
        grouped.setdefault(result.l, []).append(result)
    return {
        l: min(level_results, key=lambda result: getattr(result, metric))
        for l, level_results in grouped.items()
    }


def best_overall_symmetry(
    results: Iterable[SymmetryScore],
    *,
    metric: str = "trimmed_rms",
) -> SymmetryScore:
    """Select the best coarse symmetry among all levels and classes."""
    results = list(results)
    if not results:
        raise ValueError("No symmetry results were provided.")
    return min(results, key=lambda result: getattr(result, metric))


def describe_symmetry_score(score: SymmetryScore) -> str:
    """Human-readable, appropriately cautious one-line interpretation."""
    level = "raw mesh" if score.l is None else f"l={score.l}"
    return (
        f"Best coarse symmetry: {score.symmetry} at {level}, "
        f"trimmed RMS={score.trimmed_rms:.3f}, "
        f"matched area fraction={score.matched_fraction:.2f}."
    )

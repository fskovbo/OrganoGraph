"""Geometric errors, separate from errors at coefficient-fitting samples."""
import numpy as np
from .patches import triangle_areas, surface_moments


def surface_distance_metrics(vertices, faces, reconstructed_vertices, reconstructed_faces):
    """Area-weighted, bidirectional point-to-triangle errors (physical units).

    Uses exact nearest triangles via libigl and deterministic triangle quadrature.
    The maximum is a sampled maximum, not a certified Hausdorff distance.
    This does not certify absence of self-intersections.
    """
    import igl

    def directed(v, f, other_v, other_f):
        bary = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
        points = np.einsum("qa,fad->fqd", bary, np.asarray(v)[f]).reshape(-1, 3)
        w = np.repeat(triangle_areas(v, f) / 3, 3)
        d2 = np.maximum(igl.point_mesh_squared_distance(points, other_v, other_f)[0], 0)
        return float(np.sqrt(np.average(d2, weights=w))), float(np.sqrt(d2.max()))

    a, max_a = directed(vertices, faces, reconstructed_vertices, reconstructed_faces)
    b, max_b = directed(reconstructed_vertices, reconstructed_faces, vertices, faces)
    area, _, rms = surface_moments(vertices, faces)
    ra = triangle_areas(reconstructed_vertices, reconstructed_faces)
    return {
        "source_to_reconstruction_rmse": a, "reconstruction_to_source_rmse": b,
        "symmetric_rmse": float(np.sqrt((a * a + b * b) / 2)),
        "relative_symmetric_rmse": float(np.sqrt((a * a + b * b) / 2) / rms),
        "sampled_max_distance": max(max_a, max_b),
        "relative_area_error": float(ra.sum() / area - 1),
        "degenerate_reconstruction_faces": int(np.sum(ra <= np.finfo(float).eps * rms ** 2)),
        "self_intersections_checked": False,
    }

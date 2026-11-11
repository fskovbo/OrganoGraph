"""Positive mean-value disk mapping followed by inverse stereographic mapping.

The boundary is distributed by physical arclength. Its start vertex is a gauge
choice, subsequently handled by angular phase alignment, not an anatomical tip.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from .datatypes import CapMapping
from .patches import triangle_areas


def disk_to_cap(uv, half_angle):
    uv = np.asarray(uv, float)
    r = np.linalg.norm(uv, axis=1)
    theta = 2 * np.arctan(r * np.tan(half_angle / 2))
    return np.column_stack([theta, np.arctan2(uv[:, 1], uv[:, 0])])


def cap_to_disk(theta_phi, half_angle):
    theta, phi = np.asarray(theta_phi, float).T
    r = np.tan(theta / 2) / np.tan(half_angle / 2)
    return r[:, None] * np.column_stack([np.cos(phi), np.sin(phi)])


def parameterize_patch(patch, *, half_angle=np.pi / 2):
    if not np.isfinite(half_angle) or not 0 < half_angle < np.pi:
        raise ValueError("half_angle must lie strictly between 0 and pi")
    v, f, b = patch.vertices, patch.faces, patch.boundary
    rows, cols, values = [], [], []
    for i in range(3):
        a, c, d = f[:, i], f[:, (i + 1) % 3], f[:, (i + 2) % 3]
        u, w = v[c] - v[a], v[d] - v[a]
        lu, lw = np.linalg.norm(u, axis=1), np.linalg.norm(w, axis=1)
        # tan(angle/2), using atan2 to avoid cancellation near small angles.
        angle = np.arctan2(np.linalg.norm(np.cross(u, w), axis=1), np.sum(u * w, axis=1))
        t = np.tan(angle / 2)
        rows.extend([a, a]); cols.extend([c, d]); values.extend([t / lu, t / lw])
    weights = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(v), len(v))).tocsr()
    lap = diags(np.asarray(weights.sum(axis=1)).ravel()) - weights
    edge_lengths = np.linalg.norm(v[np.roll(b, -1)] - v[b], axis=1)
    phi = 2 * np.pi * np.r_[0, np.cumsum(edge_lengths[:-1])] / edge_lengths.sum()
    uv = np.zeros((len(v), 2))
    uv[b] = np.column_stack([np.cos(phi), np.sin(phi)])
    interior = np.setdiff1d(np.arange(len(v)), b)
    if len(interior):
        uv[interior] = spsolve(lap[interior][:, interior], -lap[interior][:, b] @ uv[b])
    if not np.isfinite(uv).all():
        raise ValueError("Nonfinite disk mapping")
    p = uv[f]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    signed = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2
    if np.any(signed <= 0):
        raise ValueError("Disk mapping contains flipped or collapsed triangles")
    tp = disk_to_cap(uv, half_angle)
    # Distortion of the actual surface-to-cap chart (linearized per triangle).
    th, ph = tp.T
    cap_v = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    cap_a, physical_a = triangle_areas(cap_v, f), triangle_areas(v, f)
    area_log_ratio = np.log((cap_a / cap_a.sum()) / (physical_a / physical_a.sum()))
    angle_errors = []
    for i in range(3):
        def angles(points):
            a = points[f[:, (i + 1) % 3]] - points[f[:, i]]
            c = points[f[:, (i + 2) % 3]] - points[f[:, i]]
            return np.arctan2(np.linalg.norm(np.cross(a, c), axis=1), np.sum(a * c, axis=1))
        angle_errors.append(np.abs(angles(cap_v) - angles(v)))
    diagnostics = {
        "method": "mean_value_disk_inverse_stereographic_v1",
        "flipped_triangles": 0,
        "area_log_ratio_rms": float(np.sqrt(np.average(area_log_ratio ** 2, weights=physical_a))),
        "area_log_ratio_max_abs": float(np.max(np.abs(area_log_ratio))),
        "angle_error_p95_degrees": float(np.rad2deg(np.percentile(angle_errors, 95))),
        "min_disk_triangle_area": float(signed.min()),
    }
    return CapMapping(uv, tp, float(half_angle), diagnostics)

"""
Simple Laplace-Beltrami reconstruction complexity scores.

The core idea is intentionally modest: a shape is more complex if it needs
more LB coordinate modes to reconstruct its embedded surface geometry.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def area_weighted_relative_reconstruction_error(mesh, l: int) -> float:
    """Return scale-normalized reconstruction error at spectral level ``l``.

    The numerator is the area-weighted RMS distance between the current mesh
    vertices and their LB reconstruction through level ``l``.  The denominator
    is the area-weighted RMS radius of the current mesh.  The result is
    dimensionless, so an error of 0.05 means roughly 5% of organoid size.

    This assumes the mesh coordinates have already been normalized or otherwise
    put into the coordinate system in which shape complexity should be measured.
    """
    l = int(l)
    if l < 1:
        raise ValueError("l must be a positive integer.")

    if getattr(mesh, "coeffs_v", None) is None or getattr(mesh, "lmax", None) is None or mesh.lmax < l:
        mesh.compute_spectral_coefficients(lmax=l)

    vertices = np.asarray(mesh.v, dtype=float)
    reconstructed = np.asarray(mesh.reconstruct_from_coeffs(mesh.coeffs_v, lmax=l), dtype=float)

    try:
        vertex_areas = np.asarray(mesh.vertex_areas(from_mass_matrix=True), dtype=float)
    except Exception:
        vertex_areas = np.asarray(mesh.vertex_areas(from_mass_matrix=False), dtype=float)

    centroid = np.average(vertices, axis=0, weights=vertex_areas)
    squared_error = np.sum((vertices - reconstructed) ** 2, axis=1)
    squared_radius = np.sum((vertices - centroid[None, :]) ** 2, axis=1)

    numerator = float(np.sqrt(np.sum(vertex_areas * squared_error) / np.sum(vertex_areas)))
    denominator = float(np.sqrt(np.sum(vertex_areas * squared_radius) / np.sum(vertex_areas)))
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("Cannot normalize reconstruction error: mesh radius is not positive.")
    return numerator / denominator


def reconstruction_error_curve(mesh, l_values: Iterable[int]) -> list[dict[str, float]]:
    """Compute relative reconstruction error for several spectral levels."""
    l_values = sorted({int(l) for l in l_values})
    if not l_values:
        raise ValueError("l_values must contain at least one level.")
    if min(l_values) < 1:
        raise ValueError("l_values must be positive integers.")

    max_l = max(l_values)
    if getattr(mesh, "coeffs_v", None) is None or getattr(mesh, "lmax", None) is None or mesh.lmax < max_l:
        mesh.compute_spectral_coefficients(lmax=max_l)

    rows = []
    for l in l_values:
        rows.append(
            {
                "l": int(l),
                "n_modes": int(l**2),
                "relative_error": float(area_weighted_relative_reconstruction_error(mesh, l)),
            }
        )
    return rows


def summarize_reconstruction_complexity(
    rows: Iterable[dict[str, float]],
    *,
    threshold: float = 0.05,
) -> dict[str, float | int | None]:
    """Summarize an error curve with a threshold crossing and smooth AUC."""
    rows = sorted(list(rows), key=lambda row: int(row["l"]))
    if not rows:
        raise ValueError("rows must not be empty.")

    l_values = np.asarray([row["l"] for row in rows], dtype=float)
    errors = np.asarray([row["relative_error"] for row in rows], dtype=float)

    passing = [row for row in rows if float(row["relative_error"]) <= float(threshold)]
    l_at_threshold = int(passing[0]["l"]) if passing else None
    n_modes_at_threshold = None if l_at_threshold is None else int(l_at_threshold**2)

    if len(rows) == 1:
        auc_error = float(errors[0])
    else:
        auc_error = float(np.trapz(errors, l_values) / (l_values[-1] - l_values[0]))

    return {
        "complexity_threshold": float(threshold),
        "complexity_l_at_threshold": l_at_threshold,
        "complexity_n_modes_at_threshold": n_modes_at_threshold,
        "complexity_error_auc": auc_error,
    }


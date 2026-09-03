"""Curve samplers retained solely for reading historical shape exports."""

from __future__ import annotations

import numpy as np


def sample_quadratic_bezier(start, control, end, *, n_samples=64):
    u = np.linspace(0.0, 1.0, max(2, int(n_samples)))
    start, control, end = map(
        lambda value: np.asarray(value, dtype=float), (start, control, end)
    )
    return (
        (1.0 - u)[:, None] ** 2 * start
        + 2.0 * (1.0 - u)[:, None] * u[:, None] * control
        + u[:, None] ** 2 * end
    )


def sample_cubic_bezier(start, control_1, control_2, end, *, n_samples=64):
    u = np.linspace(0.0, 1.0, max(2, int(n_samples)))
    start, control_1, control_2, end = map(
        lambda value: np.asarray(value, dtype=float),
        (start, control_1, control_2, end),
    )
    return (
        (1.0 - u)[:, None] ** 3 * start
        + 3.0 * (1.0 - u)[:, None] ** 2 * u[:, None] * control_1
        + 3.0 * (1.0 - u)[:, None] * u[:, None] ** 2 * control_2
        + u[:, None] ** 3 * end
    )


def sample_sinusoidal_bend(start, end, bend_vector, *, n_samples=64):
    u = np.linspace(0.0, 1.0, max(2, int(n_samples)))
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    bend = np.asarray(bend_vector, dtype=float)
    points = start + u[:, None] * (end - start) + np.sin(np.pi * u)[:, None] * bend
    points[[0, -1]] = [start, end]
    return points


def sample_circular_arc(start, end, sagitta_vector, *, n_samples=64):
    """Sample the minor-circle representation used by historical v4 exports."""
    start = np.asarray(start, dtype=float).reshape(3)
    end = np.asarray(end, dtype=float).reshape(3)
    sagitta = np.asarray(sagitta_vector, dtype=float).reshape(3)
    chord = end - start
    length = float(np.linalg.norm(chord))
    count = max(2, int(n_samples))
    if length <= 1e-12:
        return np.repeat(start[None, :], count, axis=0)
    chord_direction = chord / length
    normal = sagitta - chord_direction * float(np.dot(sagitta, chord_direction))
    height = min(float(np.linalg.norm(normal)), 0.499 * length)
    if height <= 1e-8 * max(length, 1.0):
        return start + np.linspace(0.0, 1.0, count)[:, None] * chord
    normal /= height
    radius = length**2 / (8.0 * height) + 0.5 * height
    center_offset = 0.5 * height - length**2 / (8.0 * height)
    center = 0.5 * (start + end) + center_offset * normal
    half_angle = float(np.arctan2(0.5 * length, -center_offset))
    theta = np.linspace(-half_angle, half_angle, count)
    arc = center + radius * (
        np.cos(theta)[:, None] * normal
        + np.sin(theta)[:, None] * chord_direction
    )
    arc[[0, -1]] = [start, end]
    return arc

from __future__ import annotations

import unittest

import numpy as np

from organograph.skeleton.primitive.crypt_geometry import (
    boundary_tip_ratio_field,
    centerline_radius_observations,
    fit_tangent_constrained_hermite,
    hermite_curvature_diagnostics,
    minimum_contour_radius,
    monotonic_project_points_to_polyline,
    sample_tangent_hermite,
)
from organograph.skeleton.primitive_geometry import point_at_polyline_arclength


class CryptGeometryTests(unittest.TestCase):
    def test_centerline_radius_contours_use_centerline_arclength(self):
        n_theta = 24
        z_values = np.linspace(0.0, 0.9, 19)
        vertices = []
        for z in z_values:
            radius = 1.5 - 0.5 * z
            for theta in np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False):
                vertices.append([radius * np.cos(theta), radius * np.sin(theta), z])
        tip_index = len(vertices)
        vertices.append([0.0, 0.0, 1.0])
        vertices = np.asarray(vertices, dtype=float)
        faces = []
        for ring in range(len(z_values) - 1):
            for column in range(n_theta):
                next_column = (column + 1) % n_theta
                a = ring * n_theta + column
                b = ring * n_theta + next_column
                c = (ring + 1) * n_theta + column
                d = (ring + 1) * n_theta + next_column
                faces.extend(([a, b, c], [b, d, c]))
        last_ring = (len(z_values) - 1) * n_theta
        for column in range(n_theta):
            faces.append(
                [
                    last_ring + column,
                    last_ring + (column + 1) % n_theta,
                    tip_index,
                ]
            )
        faces = np.asarray(faces, dtype=np.int64)
        centerline = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        observations, coordinate = centerline_radius_observations(
            vertices,
            faces,
            np.arange(vertices.shape[0]),
            centerline,
            np.arange(n_theta),
            n_contours=10,
            max_s=0.95,
        )

        section_s = np.asarray([item["s"] for item in observations])
        mean_radii = np.asarray([item["mean_radius"] for item in observations])
        minimum_radii = np.asarray([item["min_radius"] for item in observations])
        np.testing.assert_allclose(coordinate, vertices[:, 2], atol=1e-12)
        self.assertAlmostEqual(section_s[0], 0.0)
        self.assertGreaterEqual(section_s[-1], 0.94)
        body = section_s <= 0.9 + 1e-12
        np.testing.assert_allclose(
            mean_radii[body], 1.5 - 0.5 * section_s[body], atol=0.04
        )
        self.assertLess(mean_radii[-1], mean_radii[-2])
        np.testing.assert_allclose(minimum_radii, mean_radii, atol=0.04)

    def test_minimum_contour_radius_uses_piecewise_linear_segments(self):
        contour = np.array(
            [
                [[-2.0, -1.0, 0.0], [2.0, -1.0, 0.0]],
                [[2.0, -1.0, 0.0], [2.0, 1.0, 0.0]],
                [[2.0, 1.0, 0.0], [-2.0, 1.0, 0.0]],
                [[-2.0, 1.0, 0.0], [-2.0, -1.0, 0.0]],
            ]
        )
        self.assertAlmostEqual(minimum_contour_radius(contour, [0, 0, 0]), 1.0)

    def test_ratio_field_runs_from_boundary_to_tip(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        ratio, _, _, _ = boundary_tip_ratio_field(
            vertices, faces, np.arange(4), 3, boundary=np.array([0])
        )
        np.testing.assert_allclose(ratio, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])

    def test_hermite_fit_preserves_endpoint_normals_without_folding(self):
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([4.0, 0.0, 0.0])
        start_normal = np.array([1.0, 0.6, 0.0])
        end_normal = np.array([1.0, -0.4, 0.0])
        tangent_length = 3.2
        expected = sample_tangent_hermite(
            start,
            end,
            tangent_length * start_normal / np.linalg.norm(start_normal),
            tangent_length * end_normal / np.linalg.norm(end_normal),
            n_samples=201,
        )
        parameters = np.linspace(0.1, 0.9, 9)
        centers = np.asarray(
            [point_at_polyline_arclength(expected, value) for value in parameters]
        )
        fit = fit_tangent_constrained_hermite(
            start,
            end,
            centers,
            parameters,
            start_normal,
            end_normal,
            n_samples=201,
        )
        np.testing.assert_allclose(fit.centerline_points[[0, -1]], [start, end])
        self.assertGreater(np.dot(fit.start_tangent, start_normal), 0.0)
        self.assertGreater(np.dot(fit.end_tangent, end_normal), 0.0)
        self.assertAlmostEqual(
            np.linalg.norm(fit.start_tangent), fit.start_tangent_length
        )
        self.assertAlmostEqual(
            np.linalg.norm(fit.end_tangent), fit.end_tangent_length
        )
        self.assertLess(fit.fit_rmse, 0.06)
        self.assertTrue(np.all(np.diff(fit.centerline_points[:, 0]) > 0.0))

    def test_monotonic_projection_preserves_contour_order(self):
        centerline = np.column_stack(
            [np.linspace(0.0, 4.0, 21), np.zeros(21), np.zeros(21)]
        )
        points = np.array(
            [
                [0.8, 0.1, 0.0],
                [2.6, 0.1, 0.0],
                [2.0, 0.1, 0.0],
                [3.6, 0.1, 0.0],
            ]
        )
        projection = monotonic_project_points_to_polyline(points, centerline)
        self.assertTrue(np.all(np.diff(projection["s"]) >= 0.0))

    def test_curvature_weight_reduces_physical_bending_energy(self):
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([4.0, 0.0, 0.0])
        start_normal = np.array([1.0, 1.0, 0.0])
        end_normal = np.array([1.0, -1.0, 0.0])
        target = sample_tangent_hermite(
            start,
            end,
            4.5 * start_normal / np.linalg.norm(start_normal),
            0.8 * end_normal / np.linalg.norm(end_normal),
            n_samples=201,
        )
        parameters = np.linspace(0.05, 0.95, 12)
        centers = np.asarray(
            [point_at_polyline_arclength(target, value) for value in parameters]
        )
        unregularized = fit_tangent_constrained_hermite(
            start,
            end,
            centers,
            parameters,
            start_normal,
            end_normal,
            n_samples=201,
            curvature_weight=0.0,
            reference_length=4.0,
        )
        regularized = fit_tangent_constrained_hermite(
            start,
            end,
            centers,
            parameters,
            start_normal,
            end_normal,
            n_samples=201,
            curvature_weight=1.0,
            reference_length=4.0,
        )
        self.assertLess(
            regularized.dimensionless_bending_energy,
            unregularized.dimensionless_bending_energy,
        )
        self.assertGreaterEqual(regularized.fit_rmse, unregularized.fit_rmse - 1e-8)
        for fit in (unregularized, regularized):
            diagnostics = hermite_curvature_diagnostics(
                start, end, fit.start_tangent, fit.end_tangent
            )
            self.assertTrue(np.isfinite(diagnostics["bending_energy"]))


if __name__ == "__main__":
    unittest.main()

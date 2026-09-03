from __future__ import annotations

import unittest

import numpy as np

from organograph.skeleton.primitive.crypt_geometry import (
    boundary_tip_ratio_field,
    fit_tangent_constrained_hermite,
    sample_tangent_hermite,
)
from organograph.skeleton.primitive_geometry import point_at_polyline_arclength


class CryptGeometryTests(unittest.TestCase):
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
        fitted, start_tangent, end_tangent, fitted_length, rmse = (
            fit_tangent_constrained_hermite(
                start,
                end,
                centers,
                parameters,
                start_normal,
                end_normal,
                n_samples=201,
            )
        )
        np.testing.assert_allclose(fitted[[0, -1]], [start, end])
        self.assertGreater(np.dot(start_tangent, start_normal), 0.0)
        self.assertGreater(np.dot(end_tangent, end_normal), 0.0)
        self.assertAlmostEqual(np.linalg.norm(start_tangent), fitted_length)
        self.assertAlmostEqual(np.linalg.norm(end_tangent), fitted_length)
        self.assertLess(rmse, 0.06)
        self.assertTrue(np.all(np.diff(fitted[:, 0]) > 0.0))


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace

import numpy as np

from organograph.mesh.symmetry import (
    best_symmetry_per_level,
    laplace_beltrami_low_pass_vertices,
    run_multiscale_symmetry_analysis,
    score_all_symmetry_candidates_at_level,
    score_symmetry_at_level,
)


def make_uv_sphere(n_lat=18, n_lon=36, radius_fn=None):
    vertices = [[0.0, 0.0, 1.0]]

    for i in range(1, n_lat):
        theta = np.pi * i / n_lat
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / n_lon
            r = 1.0 if radius_fn is None else float(radius_fn(theta, phi))
            vertices.append(
                [
                    r * sin_theta * np.cos(phi),
                    r * sin_theta * np.sin(phi),
                    r * cos_theta,
                ]
            )

    vertices.append([0.0, 0.0, -1.0])
    bottom = len(vertices) - 1
    faces = []

    for j in range(n_lon):
        faces.append([0, 1 + j, 1 + (j + 1) % n_lon])

    for i in range(n_lat - 2):
        ring0 = 1 + i * n_lon
        ring1 = ring0 + n_lon
        for j in range(n_lon):
            a = ring0 + j
            b = ring0 + (j + 1) % n_lon
            c = ring1 + j
            d = ring1 + (j + 1) % n_lon
            faces.append([a, c, b])
            faces.append([b, c, d])

    last_ring = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        faces.append([last_ring + j, bottom, last_ring + (j + 1) % n_lon])

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


class SymmetryTests(unittest.TestCase):
    def test_ellipsoid_has_strong_reflection_and_c2_scores(self):
        vertices, faces = make_uv_sphere()
        vertices = vertices * np.array([2.0, 1.2, 0.75])
        mesh = SimpleNamespace(v=vertices, f=faces)

        results = score_symmetry_at_level(mesh, None, n_samples=3500, rng=11)
        by_symmetry = {result.symmetry: result for result in results}

        self.assertLess(by_symmetry["reflection"].trimmed_rms, 0.08)
        self.assertLess(by_symmetry["C2"].trimmed_rms, 0.08)
        self.assertGreater(by_symmetry["reflection"].matched_fraction, 0.65)

    def test_tripod_prefers_c3_over_c2(self):
        def tripod_radius(theta, phi):
            return 1.0 + 0.35 * (np.sin(theta) ** 2) * np.cos(3.0 * phi)

        vertices, faces = make_uv_sphere(radius_fn=tripod_radius)
        mesh = SimpleNamespace(v=vertices, f=faces)

        results = score_symmetry_at_level(mesh, None, n_samples=4500, rng=7)
        by_symmetry = {result.symmetry: result for result in results}

        self.assertLess(by_symmetry["C3"].trimmed_rms, by_symmetry["C2"].trimmed_rms)
        self.assertLess(by_symmetry["C3"].trimmed_rms, 0.10)

    def test_multiscale_helpers_return_best_per_level(self):
        vertices, faces = make_uv_sphere()
        mesh = SimpleNamespace(v=vertices, f=faces)

        results = run_multiscale_symmetry_analysis(
            mesh,
            l_values=[None],
            n_samples=2000,
            random_seed=5,
        )
        best = best_symmetry_per_level(results)

        self.assertIn(None, best)
        self.assertIn(best[None].symmetry, {"reflection", "C2", "C3"})

    def test_all_symmetry_candidates_keeps_all_pca_axes(self):
        vertices, faces = make_uv_sphere()
        mesh = SimpleNamespace(v=vertices, f=faces)

        results = score_all_symmetry_candidates_at_level(mesh, None, n_samples=2000, rng=13)
        pairs = {(result.symmetry, result.axis_name) for result in results}

        self.assertEqual(len(results), 9)
        self.assertEqual(
            pairs,
            {
                ("reflection", "PCA1"),
                ("reflection", "PCA2"),
                ("reflection", "PCA3"),
                ("C2", "PCA1"),
                ("C2", "PCA2"),
                ("C2", "PCA3"),
                ("C3", "PCA1"),
                ("C3", "PCA2"),
                ("C3", "PCA3"),
            },
        )

    def test_low_pass_uses_organoidmesh_reconstruction_methods(self):
        class FakeSpectralMesh:
            def __init__(self):
                self.v = np.arange(15, dtype=float).reshape(5, 3)
                self.f = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype=np.int64)
                self.eigvecs = np.eye(5, 4)
                self.mass_matrix = np.eye(5)
                self.lmax = None
                self.coeffs_v = None
                self.compute_calls = []
                self.reconstruct_calls = []

            def compute_spectral_coefficients(self, lmax=15):
                self.compute_calls.append(lmax)
                self.lmax = lmax
                self.coeffs_v = self.v.copy()
                return self.coeffs_v

            def reconstruct_from_coeffs(self, coeffs, lmax=15):
                self.reconstruct_calls.append(lmax)
                return coeffs + 1.0

        mesh = FakeSpectralMesh()
        reconstructed = laplace_beltrami_low_pass_vertices(mesh, l=2)

        self.assertEqual(mesh.compute_calls, [2])
        self.assertEqual(mesh.reconstruct_calls, [2])
        np.testing.assert_allclose(reconstructed, mesh.v + 1.0)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre
from scipy.spatial.transform import Rotation

from organograph.crypt_morphology import (
    CapBasis, PatchTopologyError, extract_patch, fit_representation, load_basis,
    load_representation, parameterize_patch, reconstruct, save_basis, save_representation,
    surface_distance_metrics, surface_moments, synthetic_crypt, with_size_mode,
)


class RepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.basis = CapBasis.build(8)
        cls.v, cls.f = synthetic_crypt("asymmetric", n_rings=18, n_sectors=48)
        cls.patch = extract_patch(cls.v, cls.f)
        cls.mapping = parameterize_patch(cls.patch)

    def fit(self, size_mode="physical"):
        return fit_representation(self.patch, self.mapping, self.basis, size_mode=size_mode)

    def test_patch_preserves_source_ids_and_units(self):
        p = extract_patch(self.v, self.f, np.arange(len(self.v)))
        np.testing.assert_array_equal(p.vertices, self.v[p.source_vertex_ids])
        np.testing.assert_array_equal(p.source_vertex_ids[p.faces], self.f[p.source_face_ids])
        self.assertEqual(len(p.boundary), 48)

    def test_reject_hole_and_inconsistent_winding(self):
        with self.assertRaises(PatchTopologyError):
            extract_patch(self.v, self.f[48:])
        f = self.f.copy()
        f[50] = f[50, ::-1]
        with self.assertRaises(PatchTopologyError):
            extract_patch(self.v, f)

    def test_reject_disconnected_and_duplicate_faces(self):
        with self.assertRaises(PatchTopologyError):
            extract_patch(np.vstack([self.v, self.v + 5]), np.vstack([self.f, self.f + len(self.v)]))
        with self.assertRaises(PatchTopologyError):
            extract_patch(self.v, np.vstack([self.f, self.f[:1]]))

    def test_mapping_has_no_folds_and_is_rigid_invariant(self):
        rotation = Rotation.from_rotvec([0.5, -0.2, 0.7]).as_matrix()
        p = extract_patch(self.v @ rotation.T + [300, -80, 12], self.f)
        mapped = parameterize_patch(p)
        np.testing.assert_allclose(mapped.uv, self.mapping.uv, atol=1e-12)
        self.assertEqual(mapped.diagnostics["flipped_triangles"], 0)
        np.testing.assert_allclose(np.linalg.norm(mapped.uv[p.boundary], axis=1), 1)

    def test_phase_shift_is_exact_and_does_not_collapse_rim(self):
        rng = np.random.default_rng(8)
        c = rng.normal(size=(self.basis.n_modes, 3))
        q, _ = self.basis.reference_mesh(10, 40)
        shifted = q + [0, 0.731]
        np.testing.assert_allclose(self.basis.evaluate(shifted) @ c,
                                   self.basis.evaluate(q) @ self.basis.phase_shift(c, 0.731), atol=1e-12)
        rim = self.basis.evaluate(q[-40:])
        interior_ids = [i for i, mode in enumerate(self.basis.modes) if mode[0] == "interior"]
        np.testing.assert_allclose(rim[:, interior_ids], 0, atol=1e-12)
        self.assertGreater(np.linalg.matrix_rank(rim), 5)

    def test_interior_modes_orthonormal_on_both_domains(self):
        for angle in [np.pi / 2, np.pi / 3]:
            basis = CapBasis.build(5, angle)
            x, w = roots_legendre(100)
            x = (1 + np.cos(angle)) / 2 + (1 - np.cos(angle)) * x / 2
            w = w * (1 - np.cos(angle)) / 2
            phi = np.arange(32) * 2 * np.pi / 32
            q = np.column_stack([np.repeat(np.arccos(x), 32), np.tile(phi, 100)])
            b = basis.evaluate(q)[:, [i for i, m in enumerate(basis.modes) if m[0] == "interior"]]
            gram = b.T @ (np.repeat(w * 2 * np.pi / 32, 32)[:, None] * b)
            np.testing.assert_allclose(gram, np.eye(len(gram)), atol=2e-9)

    def test_size_modes_and_inverse_are_reversible(self):
        physical, shape = self.fit(), self.fit("unit_rms")
        p = extract_patch(self.v * 4 + [100, 50, -20], self.f)
        mapping = parameterize_patch(p)
        enlarged = fit_representation(p, mapping, self.basis)
        enlarged_shape = fit_representation(p, mapping, self.basis, size_mode="unit_rms")
        np.testing.assert_allclose(enlarged.coefficients, physical.coefficients * 4, atol=1e-10)
        np.testing.assert_allclose(enlarged_shape.coefficients, shape.coefficients, atol=1e-10)
        q, _ = self.basis.reference_mesh(8, 32)
        np.testing.assert_allclose(reconstruct(physical, self.basis, q, original_coordinates=True),
                                   reconstruct(shape, self.basis, q, original_coordinates=True), atol=1e-10)
        np.testing.assert_allclose(shape.pose.inverse(shape.pose.apply(self.v)), self.v, atol=1e-14)
        converted = with_size_mode(physical, "unit_rms")
        np.testing.assert_allclose(converted.coefficients, shape.coefficients, atol=1e-12)
        restored = with_size_mode(converted, "physical")
        np.testing.assert_allclose(restored.coefficients, physical.coefficients, atol=1e-12)
        np.testing.assert_allclose(converted.pose.inverse(converted.pose.apply(self.v)), self.v, atol=1e-14)

    def test_vertex_relabeling_changes_only_cap_phase(self):
        from organograph.crypt_morphology import align_representations
        permutation = np.random.default_rng(42).permutation(len(self.v))
        inverse = np.argsort(permutation)
        patch = extract_patch(self.v[permutation], inverse[self.f])
        rep = fit_representation(patch, parameterize_patch(patch), self.basis)
        result = align_representations([self.fit(), rep], self.basis)
        np.testing.assert_allclose(result.aligned[0], result.aligned[1], atol=2e-6)

    def test_exact_surface_moments_are_subdivision_invariant(self):
        centers = self.v[self.f].mean(axis=1)
        v = np.vstack([self.v, centers])
        c = np.arange(len(self.f)) + len(self.v)
        f = np.vstack([np.column_stack([self.f[:, j], self.f[:, (j + 1) % 3], c]) for j in range(3)])
        a, center, rms = surface_moments(self.v, self.f)
        a2, center2, rms2 = surface_moments(v, f)
        np.testing.assert_allclose([a, rms], [a2, rms2], atol=1e-12)
        np.testing.assert_allclose(center, center2, atol=1e-12)

    def test_higher_order_reduces_held_out_vertex_error(self):
        errors = []
        for degree in [3, 8]:
            basis = CapBasis.build(degree)
            rep = fit_representation(self.patch, self.mapping, basis)
            pred = reconstruct(rep, basis, self.mapping.theta_phi, original_coordinates=True)
            errors.append(np.sqrt(np.mean(np.sum((pred - self.v) ** 2, axis=1))))
        self.assertLess(errors[1], 0.5 * errors[0])

    def test_bidirectional_surface_error_and_remeshing(self):
        rep = self.fit()
        v, f = reconstruct(rep, self.basis, original_coordinates=True)
        metrics = surface_distance_metrics(self.v, self.f, v, f)
        self.assertLess(metrics["relative_symmetric_rmse"], 0.04)
        self.assertEqual(metrics["degenerate_reconstruction_faces"], 0)
        dense_v, dense_f = synthetic_crypt("asymmetric", n_rings=30, n_sectors=80)
        patch = extract_patch(dense_v, dense_f)
        dense = fit_representation(patch, parameterize_patch(patch), self.basis)
        q, _ = self.basis.reference_mesh(16, 64)
        drift = reconstruct(rep, self.basis, q, original_coordinates=True) - reconstruct(dense, self.basis, q, original_coordinates=True)
        self.assertLess(np.sqrt(np.mean(np.sum(drift ** 2, axis=1))) / rep.rms_radius, 0.03)

    def test_basis_identity_and_single_sample_interchange(self):
        rep = self.fit()
        with tempfile.TemporaryDirectory() as root:
            save_basis(Path(root) / "basis.json", self.basis)
            basis = load_basis(Path(root) / "basis.json")
            self.assertEqual(basis.identifier, self.basis.identifier)
            save_representation(Path(root) / "sample.npz", rep, patch=self.patch, mapping=self.mapping)
            loaded = load_representation(Path(root) / "sample.npz", basis)
            np.testing.assert_array_equal(rep.coefficients, loaded.coefficients)
            with np.load(Path(root) / "sample.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["source_vertex_ids"], self.patch.source_vertex_ids)
            with self.assertRaises(ValueError):
                load_representation(Path(root) / "sample.npz", CapBasis.build(7))


if __name__ == "__main__":
    unittest.main()

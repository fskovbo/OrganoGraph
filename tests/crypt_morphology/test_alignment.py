from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from organograph.crypt_morphology import (
    CapBasis, CapRepresentation, SimilarityTransform, align_representations, align_to_reference,
    generalized_procrustes, load_alignment_reference, make_alignment_reference,
    save_alignment_reference, weighted_procrustes,
)


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(123)
        self.x = self.rng.normal(size=(70, 3)) * [1, 2, 0.8]
        self.rotation = Rotation.from_rotvec([0.4, 1.1, -0.7]).as_matrix()
        self.y = self.x @ self.rotation.T + [3, -8, 4]
        self.w = self.rng.uniform(0.1, 2, len(self.x))

    def test_weighted_rotation_translation_and_inverse(self):
        pose = weighted_procrustes(self.x, self.y, self.w)
        np.testing.assert_allclose(pose.apply(self.x), self.y, atol=1e-13)
        np.testing.assert_allclose(pose.inverse(self.y), self.x, atol=1e-13)
        self.assertAlmostEqual(np.linalg.det(pose.rotation), 1)

    def test_zero_weight_outliers_do_not_change_alignment(self):
        target = self.y.copy()
        target[-1] += 1000
        w = self.w.copy(); w[-1] = 0
        pose = weighted_procrustes(self.x, target, w)
        np.testing.assert_allclose(pose.apply(self.x), self.y, atol=1e-13)

    def test_reflection_is_not_removed(self):
        reflected = self.x * [-1, 1, 1]
        pose = weighted_procrustes(self.x, reflected)
        self.assertGreater(np.linalg.norm(pose.apply(self.x) - reflected), 1)
        self.assertAlmostEqual(np.linalg.det(pose.rotation), 1)

    def test_gpa_recovers_rotated_copies_and_frozen_reference(self):
        result = generalized_procrustes([self.x, self.y], self.w)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.aligned[0], result.aligned[1], atol=1e-13)
        for raw, pose, aligned in zip([self.x, self.y], result.transforms, result.aligned):
            np.testing.assert_allclose(pose.inverse(aligned), raw, atol=1e-13)
        frozen = generalized_procrustes([self.y], self.w, template=result.template)
        np.testing.assert_array_equal(frozen.template, result.template)

    def test_size_modes_keep_or_remove_scale(self):
        physical = generalized_procrustes([self.x, 3 * self.y], self.w)
        shape = generalized_procrustes([self.x, 3 * self.y], self.w, size_mode="unit_rms")
        self.assertGreater(np.linalg.norm(physical.aligned[0] - physical.aligned[1]), 1)
        np.testing.assert_allclose(shape.aligned[0], shape.aligned[1], atol=1e-13)

    def test_gpa_objective_does_not_increase(self):
        shapes = [self.x, self.y + 0.2 * self.rng.normal(size=self.y.shape), self.x * [1.2, 0.8, 1.0]]
        result = generalized_procrustes(shapes, self.w)
        self.assertTrue(np.all(np.diff(result.objective_history) <= 1e-10))

    def example_rep(self, symmetric=False):
        basis = CapBasis.build(5)
        c = np.zeros((basis.n_modes, 3))
        c[1, 0], c[2, 1] = 1.0, 1.0
        interior0 = next(i for i, m in enumerate(basis.modes) if m[0] == "interior")
        c[interior0, 2] = 1.0
        if not symmetric:
            c += self.rng.normal(scale=0.025, size=c.shape)
        rep = CapRepresentation(c, basis.identifier, SimilarityTransform(), "physical", 1, 1, {})
        return basis, rep

    def test_joint_phase_and_rigid_alignment(self):
        basis, rep = self.example_rep()
        c = basis.phase_shift(rep.coefficients, 0.413) @ self.rotation.T
        c[0] += [3, -8, 4]
        other = replace(rep, coefficients=c)
        result = align_representations([rep, other], basis, n_rings=8, n_sectors=40, n_phase=40)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.aligned[0], result.aligned[1], atol=2e-6)
        self.assertTrue(np.all(np.diff(result.objective_history) <= 1e-9))
        for source, target, phase in zip([rep, other], result.representations, result.phases):
            q, _ = basis.reference_mesh(8, 40)
            expected = basis.evaluate(q + [0, phase]) @ source.coefficients
            actual = target.pose.inverse(basis.evaluate(q) @ target.coefficients)
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        reference = make_alignment_reference(result, basis, n_rings=8, n_sectors=40, n_phase=40)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "reference.npz"
            save_alignment_reference(path, reference)
            reference = load_alignment_reference(path, basis)
            frozen = align_to_reference([other], basis, reference)
            np.testing.assert_array_equal(frozen.template, result.template)
            np.testing.assert_allclose(frozen.aligned[0], result.template, atol=2e-6)

    def test_symmetry_is_reported_without_requiring_unique_phase(self):
        basis, rep = self.example_rep(symmetric=True)
        result = align_representations([rep], basis, n_rings=6, n_sectors=32)
        self.assertTrue(result.phase_ambiguous[0])

    def test_invalid_weights_and_mixed_conventions(self):
        with self.assertRaises(ValueError):
            weighted_procrustes(self.x, self.y, -self.w)
        basis, rep = self.example_rep()
        with self.assertRaises(ValueError):
            align_representations([rep, replace(rep, size_mode="unit_rms")], basis)


if __name__ == "__main__":
    unittest.main()

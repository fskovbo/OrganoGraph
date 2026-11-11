"""Correspondence-based phase and exact pose changes for known boundary families."""
from dataclasses import replace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from organograph.crypt_morphology import (
    CapBasis, SimilarityTransform, align_known_boundary_variant, extract_patch,
    fit_representation, parameterize_patch, reframe_representation,
    shared_vertex_phase, synthetic_crypt, vertex_areas, with_size_mode,
)
from organograph.crypt_morphology.parameterization import disk_to_cap


class FamilyAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.basis = CapBasis.build(5)
        v, f = synthetic_crypt(n_rings=12, n_sectors=32)
        cls.patch = extract_patch(v, f)
        cls.mapping = parameterize_patch(cls.patch)
        cls.rep = fit_representation(cls.patch, cls.mapping, cls.basis)
        cls.ids = cls.patch.source_vertex_ids[1:193]
        cls.weights = vertex_areas(v, f)[1:193]

    def rotated_chart(self, angle, radius=1.):
        z = (self.mapping.uv[:, 0]+1j*self.mapping.uv[:, 1])*np.exp(1j*angle)*radius
        uv = np.column_stack([z.real, z.imag])
        return replace(self.mapping, uv=uv, theta_phi=disk_to_cap(uv, self.mapping.half_angle))

    def test_phase_recovers_known_rotation_across_wrap(self):
        for angle in [.83, -3.13, 3.15, 2*np.pi+.7]:
            with self.subTest(angle=angle):
                fit = shared_vertex_phase(self.patch, self.rotated_chart(angle), self.patch,
                                          self.mapping, self.ids, self.weights)
                self.assertAlmostEqual(np.cos(fit.phase_offset-angle), 1.)
                self.assertAlmostEqual(fit.coherence, 1.)
                self.assertLess(fit.disk_rmse_after, 1e-14)
                self.assertGreater(fit.disk_rmse_before, .01)

    def test_radial_change_is_retained_as_residual(self):
        fit = shared_vertex_phase(self.patch, self.rotated_chart(.8, .8), self.patch,
                                  self.mapping, self.ids, self.weights)
        self.assertAlmostEqual(fit.phase_offset, .8)
        self.assertGreater(fit.disk_rmse_after, .01)
        self.assertLess(fit.disk_rmse_after, fit.disk_rmse_before)

    def test_source_ids_match_after_reordering_and_ignore_cut_vertices(self):
        order = np.random.default_rng(4).permutation(len(self.patch.vertices))
        inverse = np.argsort(order)
        patch = replace(self.patch, vertices=np.vstack([self.patch.vertices[order], [[3, 4, 5]]]),
                        source_vertex_ids=np.r_[self.patch.source_vertex_ids[order], -1],
                        faces=inverse[self.patch.faces], boundary=inverse[self.patch.boundary])
        mapping = self.rotated_chart(-.9)
        mapping = replace(mapping, uv=np.vstack([mapping.uv[order], [[0, 0]]]))
        fit = shared_vertex_phase(patch, mapping, self.patch, self.mapping, self.ids, self.weights)
        self.assertAlmostEqual(fit.phase_offset, -.9)
        self.assertLess(fit.disk_rmse_after, 1e-14)

    def test_invalid_correspondences_and_ambiguous_phase_rejected(self):
        for ids in [self.ids[:2], np.r_[self.ids, self.ids[0]], np.r_[self.ids, 100000]]:
            with self.subTest(ids=ids.shape), self.assertRaises(ValueError):
                shared_vertex_phase(self.patch, self.mapping, self.patch, self.mapping, ids)
        bad_patches = [replace(self.patch, vertices=self.patch.vertices+[1, 0, 0]),
                       replace(self.patch, source_vertex_ids=np.zeros(len(self.patch.vertices), int))]
        for patch in bad_patches:
            with self.assertRaises(ValueError):
                shared_vertex_phase(patch, self.mapping, self.patch, self.mapping, self.ids)
        for weights in [np.zeros(len(self.ids)), -self.weights]:
            with self.assertRaises(ValueError):
                shared_vertex_phase(self.patch, self.mapping, self.patch, self.mapping, self.ids, weights)
        uv = self.mapping.uv.copy()
        # Four equal levers cancel; no preferred phase remains.
        uv[1:5] = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        other = self.mapping.uv.copy(); other[1:5] = [1, 0]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            shared_vertex_phase(self.patch, replace(self.mapping, uv=uv), self.patch,
                                replace(self.mapping, uv=other), self.ids[:4])

    def test_reframe_preserves_original_surface_with_arbitrary_pose_phase_scale(self):
        first = SimilarityTransform(Rotation.from_rotvec([.2, .7, -.6]).as_matrix(), [3, -4, 8], 2.)
        second = SimilarityTransform(Rotation.from_rotvec([-.9, .1, .3]).as_matrix(), [-2, 1, 4], .7)
        original = self.rep.coefficients.copy()
        a = reframe_representation(self.rep, self.basis, first, phase=6.1)
        b = reframe_representation(a, self.basis, second, phase=-.4)
        q, _ = self.basis.reference_mesh(8, 32)
        expected = self.rep.pose.inverse(self.basis.evaluate(q) @ original)
        actual = b.pose.inverse(self.basis.evaluate(q-[0, b.phase]) @ b.coefficients)
        np.testing.assert_allclose(actual, expected, atol=2e-13)
        restored = reframe_representation(b, self.basis, self.rep.pose, phase=0.)
        np.testing.assert_allclose(restored.coefficients, original, atol=2e-13)
        np.testing.assert_array_equal(self.rep.coefficients, original)

    def test_family_frame_preserves_origin_and_variant_normalization(self):
        for mode in ["physical", "unit_rms"]:
            with self.subTest(mode=mode):
                baseline = with_size_mode(self.rep, mode)
                pose = baseline.pose.then(SimilarityTransform(
                    Rotation.from_rotvec([.5, -.8, .3]).as_matrix(), [1, 2, -3]))
                baseline = reframe_representation(baseline, self.basis, pose, phase=6.1)
                # Exact rotated chart of the same mesh, with an explicit scale control.
                delta = .8
                variant = replace(self.rep, coefficients=self.basis.phase_shift(self.rep.coefficients, -delta))
                variant = with_size_mode(variant, mode, radius=1.2*self.rep.rms_radius if mode == "unit_rms" else None)
                aligned, fit = align_known_boundary_variant(
                    variant, self.patch, self.rotated_chart(delta), baseline, self.patch,
                    self.mapping, self.basis, self.ids, self.weights)
                np.testing.assert_allclose(aligned.pose.inverse(np.zeros(3)), baseline.pose.inverse(np.zeros(3)), atol=1e-13)
                np.testing.assert_array_equal(aligned.pose.rotation, baseline.pose.rotation)
                self.assertEqual(aligned.pose.scale, variant.pose.scale)
                np.testing.assert_allclose(aligned.coefficients*variant.pose.scale,
                                           baseline.coefficients*baseline.pose.scale, atol=2e-13)
                self.assertAlmostEqual(fit.phase_offset, delta)
                # Prior alignment of the input must not change the family result.
                other_pose = SimilarityTransform(Rotation.from_rotvec([-.6, .3, .2]).as_matrix(),
                                                 [-5, 4, 2], variant.pose.scale)
                previously_aligned = reframe_representation(variant, self.basis, other_pose, phase=.47)
                again, _ = align_known_boundary_variant(
                    previously_aligned, self.patch, self.rotated_chart(delta), baseline, self.patch,
                    self.mapping, self.basis, self.ids, self.weights)
                np.testing.assert_allclose(again.coefficients, aligned.coefficients, atol=2e-13)
                fixed, _ = align_known_boundary_variant(
                    variant, self.patch, self.rotated_chart(delta), baseline, self.patch,
                    self.mapping, self.basis, self.ids, self.weights, preserve_variant_scale=False)
                np.testing.assert_allclose(fixed.coefficients, baseline.coefficients, atol=2e-13)


if __name__ == "__main__":
    unittest.main()

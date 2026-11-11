"""Geometry/provenance, weighting, and paired stability controls."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from organograph.crypt_morphology import (
    CapBasis, OpeningShift, clip_level_set, evaluate_opening_family, extract_patch,
    fit_representation, opening_shifts, parameterize_patch, prepare_boundary_family,
    reconstruct, restrict_mapping, surface_moments, synthetic_crypt, with_size_mode,
)


def parent(n=24, sectors=48):
    v, f = synthetic_crypt("asymmetric", n_rings=n, n_sectors=sectors)
    ids = np.arange(1 + int(n * .75) * sectors)
    r = surface_moments(extract_patch(v, f, ids).vertices, extract_patch(v, f, ids).faces)[2]
    family = prepare_boundary_family(v, f, ids, max_displacement=.1*r, collar_width=.3*r)
    return family, r


def test_clipping_has_exact_geometry_and_scalar_provenance():
    v = np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    cut = clip_level_set(v, f, v[:, 0] - .37)
    assert surface_moments(cut.patch.vertices, cut.patch.faces)[0] == pytest.approx(.37)
    np.testing.assert_allclose(cut.transfer(v), cut.patch.vertices)
    np.testing.assert_allclose(cut.transfer(2*v[:, 0] - v[:, 1] + 7),
                               2*cut.patch.vertices[:, 0]-cut.patch.vertices[:, 1]+7)
    np.testing.assert_allclose(np.asarray(cut.interpolation.sum(axis=1)).ravel(), 1)
    assert np.count_nonzero(cut.patch.source_vertex_ids < 0) == 3
    assert set(cut.patch.source_face_ids) == {0, 1}
    with pytest.raises(ValueError):
        clip_level_set(v, f, np.ones(4))


def test_baseline_uniform_and_sideways_shifts_and_rigid_equivariance():
    family, r = parent()
    original_area = surface_moments(family.original.vertices, family.original.faces)[0]
    base = family.cut(OpeningShift("baseline"))
    assert surface_moments(base.patch.vertices, base.patch.faces)[0] == pytest.approx(original_area)
    areas = [surface_moments(p.vertices, p.faces)[0] for p in
             [family.cut(OpeningShift("in", -.1*r)).patch,
              family.original, family.cut(OpeningShift("out", .1*r)).patch]]
    assert areas[0] < areas[1] < areas[2]
    rim = family.original.source_vertex_ids[family.original.boundary]
    level = family.level_set(OpeningShift("sideways", cosine=(.1*r,)))
    assert level[rim].min() < -.09*r and level[rim].max() > .09*r
    rot = Rotation.from_rotvec([.2, .4, -.1]).as_matrix()
    rotated = prepare_boundary_family(family.vertices @ rot.T + 100, family.faces,
                                      family.original.source_vertex_ids,
                                      max_displacement=.1*r, collar_width=.3*r)
    np.testing.assert_allclose(rotated.signed_distance, family.signed_distance, atol=1e-12)
    np.testing.assert_allclose(rotated.angular_fields, family.angular_fields, atol=1e-12)
    with pytest.raises(ValueError, match="band"):
        family.cut(OpeningShift("too_far", .11*r))


def test_neighbor_protection_and_fixed_chart_interpolation():
    family, r = parent()
    exterior = np.flatnonzero(family.signed_distance > 0)
    outer = exterior[np.argsort(family.signed_distance[exterior])[:1]]
    protected = prepare_boundary_family(family.vertices, family.faces, family.original.source_vertex_ids,
                                        max_displacement=.1*r, collar_width=.3*r,
                                        forbidden_vertex_ids=outer[:1])
    with pytest.raises(ValueError, match="neighbor"):
        protected.cut(OpeningShift("out", .1*r))
    support = family.padded_patch()
    mapping = parameterize_patch(support.patch)
    cut = family.cut(OpeningShift("side", sine=(.08*r,)))
    restricted = restrict_mapping(cut, support, mapping)
    # Every retained original vertex must keep precisely its old UV coordinate.
    lookup = {i: uv for i, uv in zip(support.patch.source_vertex_ids, mapping.uv) if i >= 0}
    for i, uv in zip(cut.patch.source_vertex_ids, restricted.uv):
        if i >= 0:
            np.testing.assert_allclose(uv, lookup[i])


def test_distance_refinement_on_a_known_planar_disk():
    areas = []
    for rings, sectors in [(20, 40), (40, 80)]:
        q, faces = CapBasis.build(1).reference_mesh(rings, sectors)
        rho, phi = q[:, 0]/(np.pi/2), q[:, 1]
        v = np.column_stack([rho*np.cos(phi), rho*np.sin(phi), np.zeros(len(q))])
        ids = np.arange(1 + (rings//2)*sectors)
        family = prepare_boundary_family(v, faces, ids, max_displacement=.1, collar_width=.2)
        cut = family.cut(OpeningShift("out", .1)).patch
        areas.append(surface_moments(cut.vertices, cut.faces)[0])
        # Signed graph distance is exact along these radial edges.
        np.testing.assert_allclose(family.signed_distance, rho-.5, atol=1e-12)
    expected = np.pi*.6**2
    assert abs(areas[1]-expected) < abs(areas[0]-expected)
    assert abs(areas[1]/expected-1) < .002


def test_support_width_control_keeps_the_actual_openings_fixed():
    family, r = parent()
    wider = family.with_collar_width(.4*r)
    for shift in opening_shifts(r):
        np.testing.assert_array_equal(wider.level_set(shift), family.level_set(shift))
        original_cut, wider_cut = family.cut(shift).patch, wider.cut(shift).patch
        np.testing.assert_array_equal(original_cut.vertices, wider_cut.vertices)
        np.testing.assert_array_equal(original_cut.faces, wider_cut.faces)
    small, large = family.padded_patch().patch, wider.padded_patch().patch
    assert surface_moments(large.vertices, large.faces)[0] > surface_moments(small.vertices, small.faces)[0]
    assert family.collar_width == pytest.approx(.3*r)
    with pytest.raises(ValueError):
        family.with_collar_width(.05*r)


def test_residual_weights_exclude_outliers_without_shrinking_coordinates():
    q, f = CapBasis.build(1).reference_mesh(16, 48)
    rho, phi = q[:, 0]/(np.pi/2), q[:, 1]
    v = np.column_stack([rho*np.cos(phi), rho*np.sin(phi), np.zeros(len(q))])
    patch = extract_patch(v, f)
    mapping = parameterize_patch(patch)
    basis = CapBasis.build(6)
    original = fit_representation(patch, mapping, basis)
    ones = fit_representation(patch, mapping, basis, quadrature_weights=np.ones((len(f), 3)))
    np.testing.assert_allclose(original.coefficients, ones.coefficients, atol=1e-12)
    bad_v = v.copy()
    bad_v[rho > .8, 2] = 3
    bad = replace(patch, vertices=bad_v)
    weights = np.repeat((rho[f].max(axis=1) <= .8)[:, None], 3, axis=1)
    weighted = fit_representation(bad, mapping, basis, quadrature_weights=weights)
    unweighted = fit_representation(bad, mapping, basis)
    core = mapping.theta_phi[rho < .5]
    z_weighted = reconstruct(weighted, basis, core, original_coordinates=True)[:, 2]
    z_unweighted = reconstruct(unweighted, basis, core, original_coordinates=True)[:, 2]
    assert np.max(abs(z_weighted)) < 1e-9
    assert np.sqrt(np.mean(z_unweighted**2)) > 1e-3
    for value in [-1., np.nan, 0.]:
        with pytest.raises(ValueError):
            fit_representation(patch, mapping, basis, quadrature_weights=np.full((len(f), 3), value))


def test_fixed_radius_conversion_is_reversible_and_does_not_change_provenance():
    family, r = parent()
    patch = family.original
    basis = CapBasis.build(4)
    rep = fit_representation(patch, parameterize_patch(patch), basis)
    scaled = with_size_mode(rep, "unit_rms", radius=2*r)
    rescaled = with_size_mode(scaled, "unit_rms", radius=3*r)
    restored = with_size_mode(rescaled, "physical")
    np.testing.assert_allclose(restored.coefficients, rep.coefficients, atol=1e-12)
    assert rescaled.rms_radius == rep.rms_radius
    assert rescaled.pose.scale == pytest.approx(3*r)
    q, _ = basis.reference_mesh(6, 24)
    np.testing.assert_allclose(reconstruct(rescaled, basis, q, original_coordinates=True),
                               reconstruct(rep, basis, q, original_coordinates=True), atol=1e-12)


def test_experiment_fixed_core_baselines_and_explicit_rejections():
    family, r = parent()
    shifts = opening_shifts(r, fractions=(.1,), n_directions=2, include_second=False)
    result = evaluate_opening_family(family, CapBasis.build(5), shifts, blur_width=.04*r,
                                     store_samples=True)
    assert len(result.rows) == len(shifts)*3
    assert all(row["status"] == "ok" for row in result.rows)
    assert 0 < result.diagnostics["core_area_fraction"] < 1
    assert len({row["core_area_fraction"] for row in result.rows}) == 1
    for row in result.rows:
        assert np.isfinite(row["core_fidelity_relative"])
        if row["variant"] == "baseline":
            assert row["core_drift_relative"] < 1e-12
            assert row["cap_drift_fixed_rms"] < 1e-6
    # Rejected variants are recorded, not silently removed from the denominator.
    boundary_face = family.faces[np.flatnonzero(
        (family.signed_distance[family.faces] > 0).any(axis=1)
        & (family.signed_distance[family.faces] <= 0).any(axis=1))[0]]
    exterior = boundary_face[family.signed_distance[boundary_face] > 0][0]
    protected = prepare_boundary_family(family.vertices, family.faces, family.original.source_vertex_ids,
                                        max_displacement=.1*r, collar_width=.3*r,
                                        forbidden_vertex_ids=[exterior])
    rejected = evaluate_opening_family(protected, CapBasis.build(4),
                                       [OpeningShift("baseline"), OpeningShift("out", .1*r)],
                                       blur_width=.04*r)
    assert sum(row["status"] == "rejected_cut" for row in rejected.rows) == 3

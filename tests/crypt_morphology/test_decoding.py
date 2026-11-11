from dataclasses import replace
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from organograph.crypt_morphology import (
    CapBasis, DecodingProbe, SimilarityTransform, coefficient_surface_distance, decode_coefficients,
    extract_patch, fit_representation, interpolate_coefficients, mesh_intersections,
    mesh_quality_metrics, parameterize_patch, rim_coefficient_error, synthetic_crypt,
    transport_coefficient_error,
)


def test_decoding_is_linear_and_interpolation_keeps_endpoints():
    basis = CapBasis.build(8)
    rng = np.random.default_rng(2026)
    a, b = rng.normal(size=(2, basis.n_modes, 3))
    q, _ = basis.reference_mesh(12, 48)
    for t in [0, .25, .5, .75, 1]:
        c = interpolate_coefficients(a, b, t, basis)
        np.testing.assert_allclose(decode_coefficients(c, basis, q),
                                   (1-t)*decode_coefficients(a, basis, q)+t*decode_coefficients(b, basis, q), atol=1e-13)
    delta = decode_coefficients(a-b, basis, q)
    assert coefficient_surface_distance(a, b, basis, q, np.ones(len(q))) == pytest.approx(
        np.sqrt(np.mean(np.sum(delta**2, axis=1))))
    for t in [-.1, 1.1, np.nan]:
        with pytest.raises(ValueError):
            interpolate_coefficients(a, b, t, basis)
    with pytest.raises(ValueError):
        decode_coefficients(a[:-1], basis)
    with pytest.raises(ValueError):
        coefficient_surface_distance(a, b, basis, q, np.zeros(len(q)))


def test_rim_error_has_requested_rms_and_harmonic_extension_is_bounded():
    basis = CapBasis.build(12)
    delta = rim_coefficient_error(basis, .07, max_mode=3)
    phi = np.arange(2048)*2*np.pi/2048
    rim = np.column_stack([np.full(len(phi), basis.half_angle), phi])
    dr = decode_coefficients(delta, basis, rim)
    np.testing.assert_allclose(dr.mean(axis=0), 0, atol=1e-14)
    assert np.sqrt(np.mean(np.sum(dr**2, axis=1))) == pytest.approx(.07)
    assert all(np.all(delta[i] == 0) for i, mode in enumerate(basis.modes)
               if mode[0] == "interior" or mode[1] == 0)
    q, _ = basis.reference_mesh(32, 128)
    d = decode_coefficients(delta, basis, q)
    assert np.linalg.norm(d, axis=1).max() <= np.linalg.norm(dr, axis=1).max()+1e-12
    np.testing.assert_allclose(d[0], 0, atol=1e-14)
    for amplitude in [-1, np.nan]:
        with pytest.raises(ValueError):
            rim_coefficient_error(basis, amplitude)


def test_error_transport_preserves_the_same_physical_perturbation():
    basis = CapBasis.build(5)
    v, f = synthetic_crypt("asymmetric", n_rings=12, n_sectors=32)
    patch = extract_patch(v, f)
    source = fit_representation(patch, parameterize_patch(patch), basis)
    rotation = Rotation.from_rotvec([.3, -.2, .8]).as_matrix()
    transform = SimilarityTransform(rotation, np.array([1, -2, 3]), scale=source.rms_radius)
    phase = .731
    c = basis.phase_shift(source.coefficients, phase) @ rotation.T / transform.scale
    c[0] += transform.translation
    target = replace(source, coefficients=c, pose=source.pose.then(transform), phase=phase, size_mode="unit_rms")
    delta = rim_coefficient_error(basis, .05)
    converted = transport_coefficient_error(delta, basis, source, target)
    q, _ = basis.reference_mesh(10, 40)
    qs = q.copy(); qs[:, 1] += phase
    before = decode_coefficients(delta, basis, qs) @ source.pose.rotation * source.pose.scale
    after = decode_coefficients(converted, basis, q) @ target.pose.rotation * target.pose.scale
    np.testing.assert_allclose(before, after, atol=1e-12)


def test_intersections_detect_crossing_coplanar_and_incident_overlap(monkeypatch):
    monkeypatch.setitem(sys.modules, "rtree", None)
    a = np.array([[0., 0, 0], [2, 0, 0], [0, 2, 0]])
    b = np.array([[.5, .5, -1], [.5, .5, 1], [1.5, .5, 0]])
    f = np.array([[0, 1, 2], [3, 4, 5]])
    for other in [b, a*.5+.2*np.array([1, 1, 0])]:
        v = np.vstack([a, other])
        np.testing.assert_array_equal(mesh_intersections(v, f), [[0, 1]])
        rot = Rotation.from_rotvec([.5, .2, -.6]).as_matrix()
        np.testing.assert_array_equal(mesh_intersections(7*v @ rot.T + 100, f), [[0, 1]])
    assert not len(mesh_intersections(np.vstack([a, b+[0, 0, 3]]), f))
    # Shared edge alone is valid; overlapping area beyond a shared edge is not.
    v = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    adjacent = np.array([[0, 1, 2], [1, 3, 2]])
    assert not len(mesh_intersections(v, adjacent))
    folded = np.array([[0, 1, 2], [1, 0, 3]])
    v[3] = [.2, .4, 0]
    np.testing.assert_array_equal(mesh_intersections(v, folded), [[0, 1]])
    shared_vertex = np.array([[0, 1, 2], [0, 3, 4]])
    v = np.vstack([a, [.2, .8, 0], [.8, .2, 0]])
    np.testing.assert_array_equal(mesh_intersections(v, shared_vertex), [[0, 1]])


def test_mesh_checks_have_positive_and_negative_controls():
    v, f = synthetic_crypt("hemisphere", n_rings=10, n_sectors=32)
    clean = mesh_quality_metrics(v, f, reference_vertices=v)
    assert clean["intersection_pairs"] == 0
    assert clean["collapsed_faces"] == 0 and not clean["geometry_warning"]
    assert clean["min_face_area_ratio"] == pytest.approx(1)
    assert clean["max_edge_stretch"] == pytest.approx(1)
    bad = v.copy(); bad[f[0, 1]] = bad[f[0, 0]]
    bad_metrics = mesh_quality_metrics(bad, f, reference_vertices=v)
    assert bad_metrics["collapsed_faces"] > 0 and bad_metrics["geometry_warning"]
    collinear = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]])
    result = mesh_quality_metrics(collinear, np.array([[0, 1, 2]]), radius=1)
    assert result["collapsed_faces"] == 1 and result["intersection_pairs"] == 0
    with pytest.raises(ValueError):
        mesh_quality_metrics(v*np.nan, f)


def test_intersection_predicate_matches_independent_convex_feasibility():
    from scipy.optimize import linprog
    rng = np.random.default_rng(76)
    f = np.array([[0, 1, 2], [3, 4, 5]])
    for coplanar in [False, True]:
        for _ in range(20):
            v = rng.normal(size=(6, 3))
            if coplanar:
                v[:, 2] = 0
            # Convex barycentric combinations coincide iff the triangles meet.
            equality = np.vstack([np.column_stack([v[:3].T, -v[3:].T]),
                                  [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
            oracle = linprog(np.zeros(6), A_eq=equality, b_eq=[0, 0, 0, 1, 1],
                             bounds=(0, None), method="highs")
            assert bool(len(mesh_intersections(v, f))) == oracle.success


def test_curved_height_field_does_not_report_shared_corner_contacts():
    # A single-valued height field triangulated over a plane cannot cross itself.
    # Small, slightly curved triangles used to trigger a predicate's fixed eps.
    x, y = np.meshgrid(np.linspace(0, 1, 18), np.linspace(0, 1, 18))
    v = np.column_stack([x.ravel(), y.ravel(), (.01*x*x+.02*y*y+.05*np.sin(x)*np.sin(y)).ravel()])
    faces = []
    for i in range(17):
        for j in range(17):
            a = i*18+j
            faces.extend([[a, a+1, a+18], [a+1, a+19, a+18]])
    for scale in [.001, 1, 1000]:
        assert not len(mesh_intersections(v*scale+np.array([10, -20, 30]), np.asarray(faces)))


def test_probe_distinguishes_core_geometry_error_from_correspondence_motion(monkeypatch):
    # Reproduce a notebook kernel without the optional external spatial index.
    monkeypatch.setitem(sys.modules, "rtree", None)
    basis = CapBasis.build(8)
    v, f = synthetic_crypt("hemisphere", n_rings=18, n_sectors=48)
    patch = extract_patch(v, f)
    mapping = parameterize_patch(patch)
    rep = fit_representation(patch, mapping, basis)
    core = np.linalg.norm(mapping.uv[f], axis=2).max(axis=1) < .7
    probe = DecodingProbe(patch, mapping, rep, basis, core, n_rings=12, n_sectors=48)
    baseline, _ = probe.evaluate(rep.coefficients)
    shifted = rep.coefficients.copy(); shifted[0] += [10, 0, 0]
    moved, _ = probe.evaluate(shifted)
    assert baseline["core_parameter_rms_relative"] == 0
    assert baseline["core_surface_rmse_relative"] < .03
    assert moved["core_parameter_rms_relative"] == pytest.approx(10/probe.radius)
    assert moved["core_surface_rmse_relative"] > 5/probe.radius
    assert moved["intersection_pairs"] == 0  # Translation itself preserves shape.
    # Known source transforms distinguish observed alignment changes from shape.
    transform = SimilarityTransform(Rotation.from_rotvec([.3, -.1, .7]).as_matrix(),
                                    np.array([3, 2, -1]), scale=2.5)
    changed = rep.coefficients @ transform.rotation.T / transform.scale
    changed[0] += transform.translation
    reframed = replace(rep, coefficients=changed, pose=rep.pose.then(transform), size_mode="unit_rms")
    control = probe.source_frame_control(reframed)
    assert control["core_source_frame_rmse_relative"] == pytest.approx(baseline["core_surface_rmse_relative"], abs=1e-12)
    assert control["normalization_scale_ratio"] == pytest.approx(2.5)


def test_bounding_box_sweep_matches_exhaustive_pairs_including_contacts():
    from organograph.crypt_morphology.mesh_quality import _overlapping_face_pairs
    rng = np.random.default_rng(2026)
    lower = rng.uniform(-2, 2, size=(60, 3))
    upper = lower + rng.uniform(0, 2, size=(60, 3))
    # Planar boxes and exact touching boxes must not be pruned.
    lower[:3] = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    upper[:3] = [[1, 1, 0], [2, 1, 0], [3, 1, 0]]
    bounds = np.column_stack([lower, upper])
    for ids in [np.arange(60), np.r_[0, 1, 2, np.arange(5, 60, 3)], np.array([1]), np.array([], int)]:
        expected = [(int(i), int(j)) for i in ids for j in ids if i < j
                    and np.all(lower[i] <= upper[j]) and np.all(lower[j] <= upper[i])]
        actual = _overlapping_face_pairs(bounds, ids)
        np.testing.assert_array_equal(actual, np.asarray(expected, int).reshape(-1, 2))
    np.testing.assert_array_equal(_overlapping_face_pairs(bounds, np.array([2, 0, 1])), [[0, 1], [1, 2]])

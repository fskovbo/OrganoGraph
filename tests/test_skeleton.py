import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from organograph.skeleton.build import (
    _grow_parent_patch_to_neck,
    _earlier_second_derivative_transition_level,
    _penalize_short_crypt_bending,
    _refine_body_transition_width_outliers,
    _refine_broad_transition_opening,
    _select_hks_tips_from_axis,
    _validate_split_branch_geometry,
)
from organograph.skeleton import (
    BlendConfig,
    PrimitiveAttachment,
    SoftBarrierEllipsoidFit,
    SkeletonGraph,
    analyze_neck_circumference_profile,
    assign_crypt_attachments_from_barrier_crossings,
    attach_body_primitive,
    attach_body_branch_neck_primitives,
    attach_branch_primitives,
    attach_crypt_tube_primitives,
    build_skeleton_from_crypt_detections,
    create_attachment_blends,
    crypt_bend_angle,
    crypt_path_length,
    crypt_straight_distance,
    crypt_terminal_paths,
    crypt_tortuosity,
    detect_crypts_for_skeleton,
    fit_soft_barrier_ellipsoid,
    fit_soft_barrier_ellipsoid_sampled,
    estimate_smooth_crypt_centerline,
    fit_asymmetric_superellipsoid_to_points,
    fit_crypt_tube_to_points,
    fit_ellipsoid_to_points,
    fit_soft_barrier_primitive,
    fit_straight_neck_cylinder,
    find_barrier_boundary_crossing,
    load_skeleton_json,
    number_of_crypts,
    number_of_split_crypts,
    protect_detection_regions_from_mask,
    protect_patches_from_mask,
    primitive_components_from_crypt_detections,
    project_crypt_attachments_to_barrier_surfaces,
    relative_height_field,
    sampled_vertex_indices,
    save_skeleton_json,
    villus_mask_from_ellipsoid,
)
from organograph.skeleton.primitive.blobs import blob_surface_radius
from organograph.plotting.skeletons import _primitive_mesh


VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ],
    dtype=float,
)
FACES = np.array(
    [
        [0, 1, 2],
        [1, 4, 2],
        [0, 1, 3],
        [1, 5, 3],
    ],
    dtype=np.int64,
)


def make_grid_mesh(n_rows=5, n_cols=7):
    vertices = []
    for r in range(n_rows):
        for c in range(n_cols):
            vertices.append([float(c), float(r), 0.0])
    faces = []
    for r in range(n_rows - 1):
        for c in range(n_cols - 1):
            a = r * n_cols + c
            b = a + 1
            d = (r + 1) * n_cols + c
            e = d + 1
            faces.append([a, b, d])
            faces.append([b, e, d])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def make_radial_ring_test_mesh(n=9):
    _, faces = make_grid_mesh(n, n)
    center = (n - 1) / 2.0
    radii = {0: 0.0, 1: 0.2, 2: 10.0, 3: 10.2, 4: 30.0}
    vertices = []
    for r in range(n):
        for c in range(n):
            dx = c - center
            dy = r - center
            layer = int(max(abs(dx), abs(dy)))
            radius = radii.get(layer, 30.0 + 10.0 * layer)
            norm = math.sqrt(dx * dx + dy * dy)
            if norm == 0.0:
                vertices.append([0.0, 0.0, 0.0])
            else:
                vertices.append([radius * dx / norm, radius * dy / norm, 0.0])
    return np.asarray(vertices, dtype=float), faces


def make_ellipsoid_points(center, axes, n_u=24, n_v=13):
    center = np.asarray(center, dtype=float)
    axes = np.asarray(axes, dtype=float)
    u = np.linspace(0.0, 2.0 * math.pi, n_u, endpoint=False)
    v = np.linspace(0.0, math.pi, n_v)
    pts = []
    for vv in v:
        for uu in u:
            pts.append(
                center
                + axes
                * np.array(
                    [math.cos(uu) * math.sin(vv), math.sin(uu) * math.sin(vv), math.cos(vv)]
                )
            )
    return np.asarray(pts, dtype=float)


def make_axis_ring_mesh(n_rings=5, n_theta=16):
    """Tube-like mesh whose ring centers move from x=3 to x=-1."""
    axis_levels = np.linspace(0.0, 2.0, int(n_rings))
    x_centers = 3.0 - 2.0 * axis_levels
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    vertices = []
    distance_field = []
    for level, x_center in zip(axis_levels, x_centers):
        for angle in theta:
            vertices.append([x_center, 0.2 * np.cos(angle), 0.2 * np.sin(angle)])
            distance_field.append(level)
    faces = []
    for ring in range(int(n_rings) - 1):
        for j in range(int(n_theta)):
            a = ring * int(n_theta) + j
            b = ring * int(n_theta) + (j + 1) % int(n_theta)
            c = (ring + 1) * int(n_theta) + j
            d = (ring + 1) * int(n_theta) + (j + 1) % int(n_theta)
            faces.extend(([a, b, c], [b, d, c]))
    return (
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=np.int64),
        np.asarray(distance_field, dtype=float),
    )


def make_tube_points(
    centerline,
    radii=(1.0, 1.0, 1.0),
    n_s=21,
    n_theta=24,
    body_s=0.5,
    distal_taper_start=0.85,
    constriction_s=None,
    r_constriction=None,
):
    from organograph.skeleton.primitive_geometry import capped_tube_radius

    centerline = np.asarray(centerline, dtype=float)
    pts = []
    for s in np.linspace(0.0, 1.0, n_s):
        if centerline.shape[0] == 2:
            center = centerline[0] + s * (centerline[1] - centerline[0])
            tangent = centerline[1] - centerline[0]
        else:
            if s <= 0.5:
                t = s / 0.5
                center = centerline[0] + t * (centerline[1] - centerline[0])
                tangent = centerline[1] - centerline[0]
            else:
                t = (s - 0.5) / 0.5
                center = centerline[1] + t * (centerline[2] - centerline[1])
                tangent = centerline[2] - centerline[1]
        tangent = tangent / np.linalg.norm(tangent)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(ref, tangent))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        normal = np.cross(tangent, ref)
        normal = normal / np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        radius = float(
            capped_tube_radius(
                np.array([s]),
                *radii,
                body_s=body_s,
                taper_start=distal_taper_start,
                constriction_s=constriction_s,
                r_constriction=r_constriction,
            )[0]
        )
        for theta in np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False):
            pts.append(center + radius * (math.cos(theta) * normal + math.sin(theta) * binormal))
    return np.asarray(pts, dtype=float)


class SkeletonTests(unittest.TestCase):
    def test_circumference_profile_distinguishes_constriction_and_transition(self):
        levels = np.linspace(0.01, 2.0, 200)
        constricted = 8.0 + 10.0 * (levels - 1.0) ** 2
        profile = analyze_neck_circumference_profile(levels, constricted)

        self.assertEqual(profile["kind"], "constriction")
        self.assertAlmostEqual(profile["constriction_level"], 1.0, delta=0.02)
        self.assertLess(profile["distal_boundary_level"], 1.0)
        self.assertGreater(profile["attachment_level"], 1.0)
        self.assertAlmostEqual(
            profile["c_half"],
            0.5 * (profile["c_min"] + profile["c_max"]),
        )

        transition = 3.0 + 4.0 * levels + 1.5 * np.tanh(8.0 * (levels - 1.0))
        transition_profile = analyze_neck_circumference_profile(
            levels,
            transition,
        )
        self.assertEqual(transition_profile["kind"], "transition")
        self.assertEqual(transition_profile["attachment_level"], 1.0)

    def test_strong_second_derivative_transition_is_scored_highly(self):
        levels = np.linspace(0.01, 2.0, 200)
        transition = 8.0 + 2.0 * levels + np.logaddexp(
            0.0,
            25.0 * (levels - 1.0),
        ) / 25.0
        profile = analyze_neck_circumference_profile(levels, transition)

        self.assertEqual(profile["kind"], "transition")
        self.assertGreater(profile["second_derivative_peak_score"], 0.7)
        self.assertAlmostEqual(
            profile["second_derivative_peak_level"],
            1.0,
            delta=0.06,
        )

    def test_circumference_profile_does_not_relocate_neck_to_later_minimum(self):
        levels = np.linspace(0.01, 2.0, 200)
        circumference = (
            8.0
            + 2.0 * levels
            - 5.0 * np.exp(-((levels - 1.5) / 0.06) ** 2)
        )
        profile = analyze_neck_circumference_profile(levels, circumference)

        self.assertEqual(profile["kind"], "transition")
        self.assertIsNone(profile["constriction_level"])
        self.assertEqual(profile["attachment_level"], 1.0)
        self.assertEqual(profile["distal_boundary_level"], 1.0)

    def test_body_branch_neck_uses_twice_minimum_circumference(self):
        levels = np.linspace(0.01, 2.0, 200)
        circumference = 8.0 + 10.0 * (levels - 1.0) ** 2
        profile = analyze_neck_circumference_profile(
            levels,
            circumference,
            relation="body_branch",
        )

        self.assertEqual(profile["kind"], "constriction")
        self.assertAlmostEqual(profile["c_max"], 2.0 * profile["c_min"])
        self.assertAlmostEqual(profile["c_half"], 1.5 * profile["c_min"])

    def test_short_constricted_neck_collapses_to_one_attachment(self):
        levels = np.linspace(0.01, 2.0, 200)
        circumference = 10.0 - 2.0 * np.exp(
            -((levels - 1.0) / 0.02) ** 2
        )
        profile = analyze_neck_circumference_profile(
            levels,
            circumference,
            min_neck_length=0.05,
        )

        self.assertEqual(profile["kind"], "transition")
        self.assertEqual(profile["reason"], "constricted_neck_below_min_length")
        self.assertEqual(profile["attachment_level"], 1.0)
        self.assertIsNone(profile["constriction_level"])
        self.assertLess(profile["candidate_neck_length"], 0.05)

    def test_nearby_resampling_minimum_classifies_fixed_neck_as_constriction(self):
        levels = np.linspace(0.01, 2.0, 200)
        circumference = 12.0 + 20.0 * (levels - 1.02) ** 2
        profile = analyze_neck_circumference_profile(levels, circumference)

        self.assertEqual(profile["kind"], "constriction")
        self.assertEqual(profile["constriction_level"], 1.0)
        self.assertAlmostEqual(
            profile["classification_minimum_level"],
            1.02,
            delta=0.011,
        )

    def test_explicit_neck_profile_builds_attachment_and_constriction_nodes(self):
        graph = build_skeleton_from_crypt_detections(
            VERTICES,
            FACES,
            [
                {
                    "crypt_id": "budded",
                    "crypt_vertices": [0, 1, 2, 4],
                    "attachment_position": [0.0, 0.0, -0.2],
                    "constriction_position": [0.0, 0.0, 0.0],
                    "tip_position": [0.0, 0.0, 2.0],
                    "attachment_level": 1.3,
                    "neck_profile": {
                        "kind": "constriction",
                        "attachment_level": 1.3,
                        "constriction_level": 1.0,
                        "distal_boundary_level": 0.7,
                        "c_min": 1.0,
                        "c_half": 1.5,
                    },
                }
            ],
            body_center=[0.0, 0.0, -1.0],
            bend_strategy="crypt_centroid",
        )

        self.assertIn("crypt_budded_attachment", graph.nodes)
        self.assertIn("crypt_budded_constriction", graph.nodes)
        self.assertEqual(
            graph.node("crypt_budded_attachment").node_type,
            "attachment",
        )
        self.assertEqual(
            graph.node("crypt_budded_constriction").node_type,
            "constriction",
        )
        self.assertEqual(
            graph.edge("crypt_budded_attachment_to_constriction").source,
            "crypt_budded_attachment",
        )
        path = crypt_terminal_paths(graph, "budded")[0]
        self.assertEqual(path[0], "crypt_budded_attachment")
        self.assertIn("crypt_budded_constriction", path)

    def test_transition_profile_builds_attachment_without_constriction(self):
        graph = build_skeleton_from_crypt_detections(
            VERTICES,
            FACES,
            [
                {
                    "crypt_id": "bulged",
                    "crypt_vertices": [0, 1, 2, 4],
                    "attachment_position": [0.0, 0.0, 0.0],
                    "tip_position": [0.0, 0.0, 2.0],
                    "neck_profile": {
                        "kind": "transition",
                        "attachment_level": 1.0,
                    },
                }
            ],
            body_center=[0.0, 0.0, -1.0],
        )

        self.assertIn("crypt_bulged_attachment", graph.nodes)
        self.assertNotIn("crypt_bulged_constriction", graph.nodes)
        self.assertEqual(
            graph.edge("crypt_bulged_attachment_to_tip").source,
            "crypt_bulged_attachment",
        )

    def test_explicit_body_and_branch_centers_override_region_centroids(self):
        body_center = np.array([-2.0, -1.0, -0.5])
        branch_center = np.array([0.4, 0.5, 0.6])
        graph = build_skeleton_from_crypt_detections(
            VERTICES,
            FACES,
            [
                {
                    "crypt_id": "split",
                    "crypt_vertices": [0, 1, 2, 3, 4, 5],
                    "neck_position": [0.0, 0.0, 0.0],
                    "daughters": [
                        {
                            "crypt_vertices": [0, 1, 3],
                            "neck_position": [0.2, 0.0, 0.2],
                            "tip_position": [0.0, 0.0, 1.5],
                        },
                        {
                            "crypt_vertices": [2, 4, 5],
                            "neck_position": [0.8, 0.8, 0.2],
                            "tip_position": [1.0, 1.0, 1.5],
                        },
                    ],
                }
            ],
            body_center=body_center,
            branch_center_overrides={"crypt_split_branch": branch_center},
            refine_body_center_from_necks=True,
            refine_branch_centers_from_necks=True,
        )

        np.testing.assert_allclose(graph.body_node().position, body_center)
        np.testing.assert_allclose(
            graph.node("crypt_split_branch").position,
            branch_center,
        )
        self.assertEqual(
            graph.body_node().metadata["center_source"],
            "explicit_override",
        )
        self.assertEqual(
            graph.node("crypt_split_branch").metadata["center_source"],
            "explicit_override",
        )

    def test_parent_patch_growth_accepts_boundary_minimum(self):
        vertices, faces = make_radial_ring_test_mesh(9)
        patch = [r * 9 + c for r in [3, 4, 5] for c in [3, 4, 5]]
        result = _grow_parent_patch_to_neck(
            vertices,
            faces,
            patch,
            max_size_factor=6.0,
            max_mesh_fraction=1.0,
            smooth_perimeter=False,
        )

        self.assertTrue(result["kept_as_split"])
        self.assertEqual(result["reason"], "boundary_minimum_found")
        self.assertIsNotNone(result["neck_position"])
        self.assertEqual(result["final_region_size"], len(result["final_region_vertices"]))
        self.assertGreater(result["final_region_size"], result["neck_region_size"])

    def test_parent_patch_growth_rejects_if_size_doubles_first(self):
        vertices, faces = make_grid_mesh(7, 7)
        patch = [r * 7 + c for r in [1, 2, 3] for c in [1, 2, 3]]
        result = _grow_parent_patch_to_neck(
            vertices,
            faces,
            patch,
            max_size_factor=2.0,
            max_mesh_fraction=1.0,
        )

        self.assertFalse(result["kept_as_split"])
        self.assertEqual(result["reason"], "minimum_at_initial_boundary")

    def test_parent_patch_growth_respects_mesh_fraction_cap(self):
        vertices, faces = make_grid_mesh(7, 7)
        patch = [r * 7 + c for r in [1, 2, 3] for c in [1, 2, 3]]
        result = _grow_parent_patch_to_neck(
            vertices,
            faces,
            patch,
            max_size_factor=4.0,
            max_mesh_fraction=0.35,
        )

        self.assertLessEqual(result["final_region_size"], result["mesh_fraction_size_limit"])
        self.assertLessEqual(result["max_allowed_size"], int(np.floor(0.35 * len(vertices))))

    def test_branch_geometry_rejects_broad_shallow_body_side_neck(self):
        vertices, faces = make_grid_mesh(7, 7)
        parent = [r * 7 + c for r in range(1, 6) for c in range(1, 6)]
        daughter = [r * 7 + c for r in range(1, 6) for c in [4, 5]]
        validation = {
            "kept_as_split": True,
            "reason": "boundary_minimum_found",
            "neck_position": [3.0, 3.0, 0.0],
            "neck_region_vertices": parent,
        }

        result = _validate_split_branch_geometry(
            vertices,
            faces,
            parent,
            [{"crypt_vertices": daughter}],
            validation,
            min_confidence=0.6,
        )

        self.assertFalse(result["kept_as_split"])
        self.assertEqual(result["reason"], "branch_confidence_below_threshold")
        self.assertTrue(result["branch_geometry_validation"]["applied"])
        self.assertLess(result["branch_geometry_validation"]["confidence"], 0.6)

    def test_branch_confidence_does_not_depend_on_residual_stem_width(self):
        vertices, faces = make_grid_mesh(9, 9)
        parent = [r * 9 + c for r in range(1, 8) for c in range(1, 8)]
        validation = {
            "kept_as_split": True,
            "reason": "boundary_minimum_found",
            "neck_position": [1.0, 1.0, 0.0],
            "neck_region_vertices": parent,
            "boundary_lengths": [30.0, 20.0, 12.0, 22.0],
            "minimum_index": 2,
            "max_mesh_fraction": 0.4,
        }
        small_daughters = [
            {"crypt_vertices": [5 * 9 + 5, 5 * 9 + 6, 6 * 9 + 5]},
            {"crypt_vertices": [3 * 9 + 5, 3 * 9 + 6, 4 * 9 + 6]},
        ]
        large_daughters = [
            {
                "crypt_vertices": [
                    r * 9 + c for r in range(3, 8) for c in range(4, 8)
                ]
            },
            {
                "crypt_vertices": [
                    r * 9 + c for r in range(1, 5) for c in range(4, 8)
                ]
            },
        ]

        small = _validate_split_branch_geometry(
            vertices,
            faces,
            parent,
            small_daughters,
            validation,
            min_confidence=0.0,
        )
        large = _validate_split_branch_geometry(
            vertices,
            faces,
            parent,
            large_daughters,
            validation,
            min_confidence=0.0,
        )

        self.assertAlmostEqual(
            small["branch_geometry_validation"]["confidence"],
            large["branch_geometry_validation"]["confidence"],
        )
        self.assertNotEqual(
            small["branch_geometry_validation"]["n_branch_vertices"],
            large["branch_geometry_validation"]["n_branch_vertices"],
        )

    def test_branch_validation_applies_final_body_radius_veto(self):
        vertices, faces = make_grid_mesh(9, 9)
        parent = [r * 9 + c for r in range(2, 8) for c in range(2, 8)]
        daughters = [
            {"crypt_vertices": [5 * 9 + 5, 5 * 9 + 6, 6 * 9 + 5]},
            {"crypt_vertices": [3 * 9 + 5, 3 * 9 + 6, 4 * 9 + 6]},
        ]
        validation = {
            "kept_as_split": True,
            "reason": "boundary_minimum_found",
            "neck_position": [2.0, 2.0, 0.0],
            "neck_region_vertices": parent,
            "boundary_lengths": [30.0, 20.0, 12.0, 22.0],
            "minimum_index": 2,
            "max_mesh_fraction": 0.4,
        }
        diagnostic = _validate_split_branch_geometry(
            vertices,
            faces,
            parent,
            daughters,
            validation,
            min_confidence=0.0,
            max_neck_to_body_radius_ratio=10.0,
        )
        ratio = diagnostic["branch_geometry_validation"][
            "neck_to_body_radius_ratio"
        ]
        self.assertIsNotNone(ratio)

        rejected = _validate_split_branch_geometry(
            vertices,
            faces,
            parent,
            daughters,
            validation,
            min_confidence=0.0,
            max_neck_to_body_radius_ratio=max(float(ratio) - 1e-6, 0.0),
        )

        self.assertFalse(rejected["kept_as_split"])
        self.assertEqual(
            rejected["reason"],
            "body_side_neck_too_broad_for_body",
        )
        self.assertFalse(
            rejected["branch_geometry_validation"]["body_radius_check_passed"]
        )

    def test_short_crypt_bend_penalty_reduces_lateral_waypoint_offset(self):
        source = np.array([0.0, 0.0, 0.0])
        tip = np.array([1.0, 0.0, 0.0])
        candidate = np.array([0.5, 1.0, 0.0])
        vertices = np.array(
            [
                [0.1, 0.2, 0.0],
                [0.3, -0.2, 0.0],
                [0.5, 0.2, 0.0],
                [0.7, -0.2, 0.0],
                [0.9, 0.2, 0.0],
            ]
        )

        refined, diagnostics = _penalize_short_crypt_bending(
            vertices,
            np.arange(len(vertices)),
            source,
            candidate,
            tip,
            max_dimensionless_curvature=0.05,
            penalty_strength=12.0,
        )

        self.assertTrue(diagnostics["applied"])
        self.assertLess(abs(refined[1]), abs(candidate[1]))
        self.assertLess(
            diagnostics["final_dimensionless_curvature"],
            diagnostics["original_dimensionless_curvature"],
        )

    def test_broad_transition_opening_moves_attachment_tipward(self):
        vertices, faces = make_grid_mesh(7, 7)
        mesh = SimpleNamespace(v=vertices, f=faces)
        patch_vertices = [r * 7 + c for r in range(7) for c in range(5)]
        old_field = vertices[:, 0] / 4.0
        detection = {
            "crypt_vertices": patch_vertices,
            "bottom_vertex_id": 3 * 7,
            "d_crypt": old_field,
            "L_crypt": 4.0,
            "attachment_level": 1.0,
            "neck_profile": {"kind": "transition", "attachment_level": 1.0},
        }
        levels = np.linspace(0.05, 1.5, 60)
        detection["circumference_levels"] = levels
        detection["circumference"] = 5.0 + 7.0 * levels

        def geodesics(_mesh, sources=None, **kwargs):
            return vertices[:, 0][None, :]

        with patch(
            "organograph.crypts.analysis.crypt_circumference",
            return_value=5.0 + 7.0 * levels,
        ):
            refined = _refine_broad_transition_opening(
                mesh,
                detection,
                levels,
                geodesic_fn=geodesics,
                geodesic_kwargs={},
                max_opening_to_crypt_body_ratio=0.85,
                min_attachment_level=0.35,
            )

        diagnostics = refined["broad_opening_validation"]
        self.assertTrue(diagnostics["refined"])
        self.assertLess(refined["attachment_level"], 1.0)
        self.assertEqual(refined["bottom_vertex_id"], 3 * 7)

    def test_broad_opening_preserves_structured_transition_profile(self):
        vertices, faces = make_grid_mesh(7, 7)
        mesh = SimpleNamespace(v=vertices, f=faces)
        levels = np.linspace(0.05, 1.5, 60)
        structured = (
            4.0
            + 5.0 * levels
            - 1.2 * np.exp(-((levels - 0.65) / 0.13) ** 2)
            + 1.5 * np.logaddexp(0.0, 18.0 * (levels - 0.9)) / 18.0
        )
        detection = {
            "crypt_vertices": np.arange(len(vertices)),
            "bottom_vertex_id": 0,
            "d_crypt": np.zeros(len(vertices)),
            "attachment_level": 1.0,
            "circumference_levels": levels,
            "circumference": structured,
            "neck_profile": {
                "kind": "transition",
                "relation": "body_crypt",
                "attachment_level": 1.0,
            },
        }

        refined = _refine_broad_transition_opening(
            mesh,
            detection,
            levels,
            geodesic_fn=lambda *args, **kwargs: self.fail(
                "structured transitions should not recompute geodesics"
            ),
            geodesic_kwargs={},
        )

        self.assertEqual(refined["attachment_level"], 1.0)
        self.assertFalse(refined["broad_opening_validation"]["refined"])
        self.assertEqual(
            refined["broad_opening_validation"]["reason"],
            "structured_transition_profile_preserved",
        )
        self.assertLess(
            refined["broad_opening_validation"]["linear_profile_r2"],
            0.985,
        )

    def test_branch_crypt_uses_looser_opening_ratio(self):
        vertices, faces = make_grid_mesh(7, 7)
        mesh = SimpleNamespace(v=vertices, f=faces)
        patch_vertices = [r * 7 + c for r in range(7) for c in range(5)]
        detection = {
            "crypt_vertices": patch_vertices,
            "bottom_vertex_id": 3 * 7,
            "d_crypt": vertices[:, 0] / 4.0,
            "attachment_level": 1.0,
            "neck_profile": {
                "kind": "transition",
                "relation": "branch_crypt",
                "attachment_level": 1.0,
                "second_derivative_peak_score": 0.0,
            },
        }
        levels = np.linspace(0.05, 1.5, 60)
        detection["circumference_levels"] = levels
        detection["circumference"] = 5.0 + 7.0 * levels
        circumference = 10.0 - np.exp(-((levels - 1.0) / 0.08) ** 2)

        with patch(
            "organograph.crypts.analysis.crypt_circumference",
            return_value=circumference,
        ):
            refined = _refine_broad_transition_opening(
                mesh,
                detection,
                levels,
                geodesic_fn=lambda _mesh, sources=None, **kwargs: vertices[:, 0][None, :],
                geodesic_kwargs={},
                max_opening_to_crypt_body_ratio=0.85,
                branch_max_opening_to_crypt_body_ratio=0.95,
            )

        self.assertFalse(refined["broad_opening_validation"]["refined"])
        self.assertAlmostEqual(
            refined["broad_opening_validation"]["max_opening_to_crypt_body_ratio"],
            0.95,
        )

    def test_body_transition_width_outlier_uses_earlier_second_derivative_peak(self):
        vertices, faces = make_grid_mesh(9, 9)
        mesh = SimpleNamespace(v=vertices, f=faces)
        levels = np.linspace(0.05, 1.0, 120)
        circumference = 5.0 + 16.0 * levels + 8.0 * np.logaddexp(
            0.0,
            35.0 * (levels - 0.55),
        ) / 35.0
        detection = {
            "crypt_id": "wide",
            "crypt_vertices": np.arange(len(vertices)),
            "bottom_vertex_id": 4 * 9,
            "d_crypt": vertices[:, 0] / 8.0,
            "attachment_level": 1.0,
            "attachment_position": [8.0, 4.0, 0.0],
            "circumference_levels": levels,
            "circumference": circumference,
            "neck_profile": {
                "kind": "transition",
                "relation": "body_crypt",
                "attachment_level": 1.0,
            },
        }

        refined = _refine_body_transition_width_outliers(
            mesh,
            [detection],
            max_crypt_to_host_width_ratio=0.8,
            min_second_derivative_score=0.5,
            min_attachment_level=0.3,
        )[0]

        diagnostics = refined["body_transition_width_validation"]
        self.assertTrue(diagnostics["refined"])
        self.assertEqual(diagnostics["reason"], "earlier_second_derivative_transition")
        self.assertLess(refined["attachment_level"], 1.0)
        self.assertAlmostEqual(refined["attachment_level"], 0.55, delta=0.08)

    def test_second_derivative_transition_selects_earliest_plausible_peak(self):
        levels = np.linspace(0.05, 1.0, 160)
        smooth = (
            4.0
            + 5.0 * levels
            + 4.0 * np.logaddexp(0.0, 45.0 * (levels - 0.32)) / 45.0
            + 9.0 * np.logaddexp(0.0, 45.0 * (levels - 0.72)) / 45.0
        )

        level, details = _earlier_second_derivative_transition_level(
            levels,
            smooth,
            current_level=1.0,
            min_level=0.25,
            min_score=0.5,
            window_length=9,
        )

        self.assertIsNotNone(level)
        self.assertAlmostEqual(level, 0.32, delta=0.06)
        self.assertGreaterEqual(len(details["accepted_candidate_levels"]), 2)
        self.assertAlmostEqual(level, details["accepted_candidate_levels"][0])

    def test_body_transition_width_outlier_shrinks_to_threshold_without_peak(self):
        vertices, faces = make_grid_mesh(9, 9)
        mesh = SimpleNamespace(v=vertices, f=faces)
        levels = np.linspace(0.05, 1.0, 120)
        circumference = 5.0 + 22.0 * levels
        detection = {
            "crypt_id": "wide",
            "crypt_vertices": np.arange(len(vertices)),
            "bottom_vertex_id": 4 * 9,
            "d_crypt": vertices[:, 0] / 8.0,
            "attachment_level": 1.0,
            "attachment_position": [8.0, 4.0, 0.0],
            "circumference_levels": levels,
            "circumference": circumference,
            "neck_profile": {
                "kind": "transition",
                "relation": "body_crypt",
                "attachment_level": 1.0,
            },
        }

        refined = _refine_body_transition_width_outliers(
            mesh,
            [detection],
            max_crypt_to_host_width_ratio=0.8,
            min_second_derivative_score=0.99,
            min_attachment_level=0.3,
        )[0]

        diagnostics = refined["body_transition_width_validation"]
        self.assertTrue(diagnostics["refined"])
        self.assertIn("width_threshold", diagnostics["reason"])
        self.assertLessEqual(
            diagnostics["refined_crypt_to_host_width_ratio"],
            0.8 + 1e-6,
        )

    def test_final_hks_tip_selection_uses_refined_axis_bottom_fraction(self):
        vertices, faces = make_grid_mesh(7, 7)
        patch = [r * 7 + c for r in [1, 2, 3, 4, 5] for c in [1, 2, 3, 4, 5]]
        center_vertex = 3 * 7 + 3
        high_hks_inner_vertex = 3 * 7 + 2
        high_hks_outer_vertex = 5 * 7 + 5
        dnorm = np.full((1, len(vertices)), np.nan, dtype=float)
        dnorm[0, patch] = np.linspace(1.0, 0.0, len(patch))
        dnorm[0, center_vertex] = 0.0
        dnorm[0, high_hks_inner_vertex] = 0.1
        dnorm[0, high_hks_outer_vertex] = 1.0
        hks = np.zeros((len(vertices), 1), dtype=float)
        hks[high_hks_inner_vertex, 0] = 10.0
        hks[high_hks_outer_vertex, 0] = 100.0

        tips, info = _select_hks_tips_from_axis(
            vertices,
            [patch],
            dnorm,
            hks,
            np.array([1.0]),
            np.array([center_vertex]),
            hks_time=1.0,
            bottom_fraction=0.5,
        )

        self.assertEqual(int(tips[0]), high_hks_inner_vertex)
        self.assertEqual(info[0]["hks_time_actual"], 1.0)
        self.assertEqual(info[0]["bottom_fraction"], 0.5)
        self.assertTrue(info[0]["update_accepted"])

    def test_final_hks_tip_selection_respects_minimum_percent_increase(self):
        vertices, faces = make_grid_mesh(7, 7)
        patch = [r * 7 + c for r in [1, 2, 3, 4, 5] for c in [1, 2, 3, 4, 5]]
        initial_vertex = 3 * 7 + 3
        candidate_vertex = 3 * 7 + 2
        dnorm = np.full((1, len(vertices)), np.nan, dtype=float)
        dnorm[0, patch] = 0.5
        dnorm[0, initial_vertex] = 0.0
        dnorm[0, candidate_vertex] = 0.1
        hks = np.zeros((len(vertices), 1), dtype=float)
        hks[initial_vertex, 0] = 10.0
        hks[candidate_vertex, 0] = 10.5

        tips, info = _select_hks_tips_from_axis(
            vertices,
            [patch],
            dnorm,
            hks,
            np.array([1.0]),
            np.array([initial_vertex]),
            hks_time=1.0,
            bottom_fraction=0.5,
            min_hks_percent_increase=10.0,
        )

        self.assertEqual(int(tips[0]), initial_vertex)
        self.assertEqual(info[0]["fallback"], "hks_increase_below_threshold")
        self.assertFalse(info[0]["update_accepted"])
        self.assertAlmostEqual(info[0]["hks_percent_increase"], 5.0)

    def test_one_straight_crypt(self):
        graph = build_skeleton_from_crypt_detections(
            VERTICES,
            FACES,
            [
                {
                    "crypt_id": "a",
                    "neck_position": [0.0, 0.0, 0.0],
                    "tip_position": [0.0, 0.0, 2.0],
                }
            ],
            body_center=[0.0, 0.0, -1.0],
        )

        self.assertEqual(len(graph.nodes), 3)
        self.assertEqual(len(graph.edges), 2)
        self.assertEqual(number_of_crypts(graph), 1)
        self.assertEqual(len(graph.nodes_for_crypt("a", node_type="crypt")), 0)
        self.assertAlmostEqual(crypt_path_length(graph, "a"), 2.0)
        self.assertAlmostEqual(crypt_straight_distance(graph, "a"), 2.0)
        self.assertAlmostEqual(crypt_tortuosity(graph, "a"), 1.0)
        self.assertAlmostEqual(crypt_bend_angle(graph, "a"), 0.0)

    def test_one_bent_crypt_uses_explicit_bend_node(self):
        graph = build_skeleton_from_crypt_detections(
            VERTICES,
            FACES,
            [
                {
                    "crypt_id": 0,
                    "neck_position": [0.0, 0.0, 0.0],
                    "bend_position": [1.0, 0.0, 0.0],
                    "tip_position": [1.0, 1.0, 0.0],
                }
            ],
            body_center=[-1.0, 0.0, 0.0],
            bend_strategy="midpoint",
        )

        self.assertEqual(len(graph.nodes), 4)
        self.assertEqual(len(graph.edges), 3)
        self.assertAlmostEqual(crypt_path_length(graph, 0), 2.0)
        self.assertAlmostEqual(crypt_straight_distance(graph, 0), math.sqrt(2.0))
        self.assertAlmostEqual(crypt_tortuosity(graph, 0), math.sqrt(2.0))
        self.assertAlmostEqual(crypt_bend_angle(graph, 0), math.pi / 2.0)

    def test_one_split_crypt(self):
        graph = build_skeleton_from_crypt_detections(
            VERTICES,
            FACES,
            [
                {
                    "crypt_id": "split",
                    "neck_position": [0.0, 0.0, 0.0],
                    "branch_position": [0.0, 0.0, 1.0],
                    "daughters": [
                        {
                            "neck_position": [0.5, 0.0, 1.5],
                            "tip_position": [1.0, 0.0, 2.0],
                        },
                        {
                            "neck_position": [-0.5, 0.0, 1.5],
                            "tip_position": [-1.0, 0.0, 2.0],
                        },
                    ],
                }
            ],
            body_center=[0.0, 0.0, -1.0],
        )

        self.assertEqual(len(graph.nodes), 7)
        self.assertEqual(len(graph.edges), 6)
        self.assertEqual(number_of_crypts(graph), 1)
        self.assertEqual(number_of_split_crypts(graph), 1)
        self.assertEqual(len(graph.nodes_for_crypt("split", node_type="neck")), 3)
        self.assertEqual(
            len(graph.nodes_for_crypt("split", node_type="attachment")),
            0,
        )
        self.assertEqual(
            len(graph.nodes_for_crypt("split", node_type="constriction")),
            0,
        )
        self.assertEqual(len(graph.nodes_for_crypt("split", node_type="crypt")), 0)
        self.assertEqual(len(graph.nodes_for_crypt("split", node_type="tip")), 2)

    def test_crypt_centroid_bend_strategy_adds_crypt_node(self):
        graph = build_skeleton_from_crypt_detections(
            VERTICES,
            FACES,
            [
                {
                    "crypt_id": "centroid",
                    "neck_position": [0.0, 0.0, 0.0],
                    "tip_position": [0.0, 0.0, 2.0],
                    "crypt_vertices": [1, 2, 4],
                }
            ],
            body_center=[0.0, 0.0, -1.0],
            bend_strategy="crypt_centroid",
            bend_max_dimensionless_curvature=None,
        )

        self.assertEqual(len(graph.nodes_for_crypt("centroid", node_type="crypt")), 1)
        np.testing.assert_allclose(
            graph.node("crypt_centroid_crypt").position,
            np.mean(VERTICES[[1, 2, 4]], axis=0),
        )

    def test_body_and_branch_centers_can_use_neck_bounded_regions(self):
        vertices = np.array(
            [
                [-2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 4.0, 0.0],
                [1.0, 6.0, 0.0],
                [-1.0, 6.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        graph = build_skeleton_from_crypt_detections(
            vertices,
            np.empty((0, 3), dtype=np.int64),
            [
                {
                    "crypt_id": "split",
                    "neck_position": [0.0, 0.0, 0.0],
                    "neck_region_vertices": [1, 2, 3, 4, 5],
                    "daughters": [
                        {
                            "neck_position": [0.5, 5.0, 0.0],
                            "tip_position": [1.0, 6.0, 0.0],
                            "crypt_vertices": [4],
                        },
                        {
                            "neck_position": [-0.5, 5.0, 0.0],
                            "tip_position": [-1.0, 6.0, 0.0],
                            "crypt_vertices": [5],
                        },
                    ],
                }
            ],
        )

        np.testing.assert_allclose(graph.node("body").position, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(graph.node("crypt_split_branch").position, [0.0, 2.0, 0.0])
        self.assertTrue(graph.node("body").metadata["center_refined_from_neck_regions"])
        self.assertTrue(graph.node("crypt_split_branch").metadata["center_refined_from_neck_regions"])

    def test_json_round_trip_preserves_positions_and_topology(self):
        graph = build_skeleton_from_crypt_detections(
            VERTICES,
            FACES,
            [
                {
                    "crypt_id": "a",
                    "neck_position": [0.0, 0.0, 0.0],
                    "tip_position": [0.0, 0.0, 2.0],
                }
            ],
            body_center=[0.0, 0.0, -1.0],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skeleton.json"
            save_skeleton_json(graph, path)
            loaded = load_skeleton_json(path)

        self.assertEqual(set(graph.nodes), set(loaded.nodes))
        self.assertEqual(set(graph.edges), set(loaded.edges))
        for node_id in graph.nodes:
            np.testing.assert_allclose(
                graph.node(node_id).position,
                loaded.node(node_id).position,
            )

    def test_ellipsoid_point_cloud_fit_recovers_center_and_axes(self):
        center = np.array([2.0, -1.0, 0.5])
        axes = np.array([3.0, 1.5, 0.75])
        points = make_ellipsoid_points(center, axes)
        fit = fit_ellipsoid_to_points(points, axis_quantile=1.0)

        np.testing.assert_allclose(fit.parameters["center"], center, atol=1e-12)
        np.testing.assert_allclose(
            np.sort(fit.parameters["axis_lengths"]),
            np.sort(axes),
            atol=0.08,
        )

    def test_soft_barrier_ellipsoid_point_cloud_fit_is_sane(self):
        center = np.array([0.5, -0.25, 0.1])
        axes = np.array([2.0, 1.0, 0.6])
        points = make_ellipsoid_points(center, axes, n_u=32, n_v=15)

        fit = fit_soft_barrier_ellipsoid(
            points,
            config={
                "barrier_weight": 20.0,
                "underfill_weight": 1.0,
                "center_regularization": 0.01,
                "maxiter": 300,
            },
            require_inside_center=False,
        )
        field = relative_height_field(points, fit)

        self.assertEqual(fit.radii.shape, (3,))
        self.assertTrue(np.all(fit.radii > 0.0))
        np.testing.assert_allclose(fit.center, center, atol=0.15)
        np.testing.assert_allclose(
            np.sort(fit.radii),
            np.sort(axes),
            rtol=0.2,
            atol=0.12,
        )
        self.assertAlmostEqual(float(np.median(field["level"])), 1.0, delta=0.08)
        self.assertGreater(np.count_nonzero(villus_mask_from_ellipsoid(points, fit)), 0)

    def test_soft_barrier_ellipsoid_anisotropy_penalty_reduces_axis_ratio(self):
        center = np.zeros(3)
        axes = np.array([3.0, 1.0, 0.7])
        points = make_ellipsoid_points(center, axes, n_u=36, n_v=16)

        free_fit = fit_soft_barrier_ellipsoid(
            points,
            config={
                "barrier_weight": 20.0,
                "underfill_weight": 1.0,
                "center_regularization": 0.01,
                "maxiter": 300,
            },
            require_inside_center=False,
        )
        penalized_fit = fit_soft_barrier_ellipsoid(
            points,
            config={
                "barrier_weight": 20.0,
                "underfill_weight": 1.0,
                "center_regularization": 0.01,
                "anisotropy_regularization": 5.0,
                "maxiter": 300,
            },
            require_inside_center=False,
        )

        free_ratio = float(np.max(free_fit.radii) / np.min(free_fit.radii))
        penalized_ratio = float(np.max(penalized_fit.radii) / np.min(penalized_fit.radii))
        self.assertLess(penalized_ratio, free_ratio)

    def test_soft_barrier_superellipsoid_recovers_flattened_body_shape(self):
        center = np.array([0.3, -0.2, 0.1])
        radii = np.array([2.2, 1.5, 0.75])
        expected_epsilon = 0.55
        u = np.linspace(-np.pi, np.pi, 48, endpoint=False)
        v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 25)
        uu, vv = np.meshgrid(u, v)

        def signed_power(values, exponent):
            return np.sign(values) * np.abs(values) ** exponent

        points = np.stack(
            [
                radii[0] * signed_power(np.cos(vv), expected_epsilon) * np.cos(uu),
                radii[1] * signed_power(np.cos(vv), expected_epsilon) * np.sin(uu),
                radii[2] * signed_power(np.sin(vv), expected_epsilon),
            ],
            axis=-1,
        ).reshape(-1, 3)
        points += center

        fit = fit_soft_barrier_primitive(
            points,
            config={
                "primitive_type": "superellipsoid",
                "barrier_weight": 1.0,
                "underfill_weight": 1.0,
                "center_regularization": 0.01,
                "initial_radius_quantile": 0.9,
                "initial_radius_scale": 1.0,
                "initial_epsilon_1": 0.9,
                "epsilon_1_bounds": (0.35, 1.0),
                "epsilon_1_regularization": 0.001,
                "maxiter": 600,
            },
            require_inside_center=False,
        )
        field = relative_height_field(points, fit)

        self.assertEqual(fit.primitive_type, "superellipsoid")
        self.assertAlmostEqual(fit.epsilon_1, expected_epsilon, delta=0.08)
        np.testing.assert_allclose(fit.center, center, atol=0.05)
        np.testing.assert_allclose(
            np.sort(fit.radii),
            np.sort(radii),
            rtol=0.08,
            atol=0.05,
        )
        self.assertAlmostEqual(float(np.median(field["level"])), 1.0, delta=0.02)
        self.assertEqual(fit.to_primitive_parameters()["fit_family"], "soft_barrier_superellipsoid")

        attachment = PrimitiveAttachment(
            primitive_type=fit.primitive_type,
            parameters=fit.to_primitive_parameters(),
        )
        surface_vertices, surface_faces = _primitive_mesh(
            attachment,
            n_s=16,
            n_theta=12,
        )
        self.assertGreater(surface_vertices.shape[0], 0)
        self.assertGreater(surface_faces.shape[0], 0)
        self.assertTrue(np.all(np.isfinite(surface_vertices)))

    def test_crypt_attachments_project_to_body_and_branch_barrier_surfaces(self):
        body_fit = SoftBarrierEllipsoidFit(
            center=np.zeros(3),
            axes=np.eye(3),
            radii=np.array([2.0, 2.0, 2.0]),
            primitive_type="superellipsoid",
            epsilon_1=0.6,
            epsilon_2=1.0,
        )
        branch_fit = SoftBarrierEllipsoidFit(
            center=np.array([5.0, 0.0, 0.0]),
            axes=np.eye(3),
            radii=np.ones(3),
        )
        detections = [
            {
                "crypt_id": "body_crypt",
                "attachment_position": [1.0, 0.0, 0.0],
                "neck_position": [1.0, 0.0, 0.0],
                "neck_profile": {"kind": "transition"},
            },
            {
                "crypt_id": "budded",
                "attachment_position": [0.0, 1.0, 0.0],
                "neck_position": [0.0, 1.5, 0.0],
                "constriction_position": [0.0, 1.5, 0.0],
                "neck_profile": {"kind": "constriction"},
            },
            {
                "crypt_id": "split",
                "daughters": [
                    {
                        "attachment_position": [5.5, 0.0, 0.0],
                        "neck_position": [5.5, 0.0, 0.0],
                        "neck_profile": {"kind": "transition"},
                    }
                ],
            },
        ]

        refined = project_crypt_attachments_to_barrier_surfaces(
            detections,
            body_fit,
            branch_fits={"crypt_split_branch": branch_fit},
        )

        np.testing.assert_allclose(
            refined[0]["attachment_position"],
            [2.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(refined[0]["neck_position"], [2.0, 0.0, 0.0])
        np.testing.assert_allclose(
            refined[1]["attachment_position"],
            [0.0, 2.0, 0.0],
        )
        np.testing.assert_allclose(refined[1]["neck_position"], [0.0, 1.5, 0.0])
        np.testing.assert_allclose(
            refined[2]["daughters"][0]["attachment_position"],
            [6.0, 0.0, 0.0],
        )
        self.assertTrue(
            refined[0]["metadata"]["barrier_attachment_projection"]["moved"]
        )
        self.assertEqual(detections[0]["attachment_position"], [1.0, 0.0, 0.0])

    def test_barrier_crossing_follows_geodesic_ring_centers(self):
        vertices, faces, distance_field = make_axis_ring_mesh()
        host_fit = SoftBarrierEllipsoidFit(
            center=np.zeros(3),
            axes=np.eye(3),
            radii=np.ones(3),
        )

        crossing = find_barrier_boundary_crossing(
            vertices,
            faces,
            distance_field,
            host_fit,
            prefer_vertices=np.arange(vertices.shape[0]),
            n_samples=32,
            persistence=2,
        )

        self.assertTrue(crossing["found"])
        self.assertAlmostEqual(crossing["axis_level"], 1.0, delta=0.02)
        np.testing.assert_allclose(crossing["position"], [1.0, 0.0, 0.0], atol=0.03)
        self.assertAlmostEqual(crossing["primitive_level"], 1.0, delta=0.03)

    def test_barrier_crossing_replaces_inside_and_outside_attachments(self):
        vertices, faces, distance_field = make_axis_ring_mesh()
        host_fit = SoftBarrierEllipsoidFit(
            center=np.zeros(3),
            axes=np.eye(3),
            radii=np.ones(3),
        )
        detections = [
            {
                "crypt_id": "inside",
                "crypt_vertices": np.arange(vertices.shape[0]),
                "bottom_vertex_id": 0,
                "d_crypt": distance_field,
                "attachment_level": 1.5,
                "attachment_position": [0.0, 0.0, 0.0],
                "neck_position": [0.0, 0.0, 0.0],
                "neck_profile": {"kind": "transition", "attachment_level": 1.5},
            },
            {
                "crypt_id": "outside",
                "crypt_vertices": np.arange(vertices.shape[0]),
                "bottom_vertex_id": 0,
                "d_crypt": distance_field,
                "attachment_level": 0.5,
                "attachment_position": [2.0, 0.0, 0.0],
                "neck_position": [2.0, 0.0, 0.0],
                "neck_profile": {"kind": "transition", "attachment_level": 0.5},
            },
        ]

        refined = assign_crypt_attachments_from_barrier_crossings(
            vertices,
            faces,
            detections,
            host_fit,
            crossing_kwargs={"n_samples": 32, "persistence": 2},
        )

        for detection in refined:
            np.testing.assert_allclose(
                detection["attachment_position"],
                [1.0, 0.0, 0.0],
                atol=0.03,
            )
            np.testing.assert_allclose(
                detection["neck_position"],
                detection["attachment_position"],
            )
            self.assertTrue(
                detection["metadata"]["barrier_boundary_crossing"]["found"]
            )
        self.assertEqual(detections[0]["attachment_position"], [0.0, 0.0, 0.0])

    def test_body_barrier_fit_precedes_hks_candidate_detection(self):
        events = []
        mesh = SimpleNamespace(
            v=np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            f=np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64),
            vertex_areas=lambda: np.ones(4),
        )
        fit = SoftBarrierEllipsoidFit(
            center=np.zeros(3),
            axes=np.eye(3),
            radii=np.ones(3),
        )

        def fit_body(*args, **kwargs):
            events.append("body_fit")
            return fit

        def detect_candidates(*args, **kwargs):
            events.append("hks_candidates")
            return [], {
                "encoding": None,
                "ts_mesh": None,
                "ts_vocab": None,
                "hks": None,
                "norm_hks": None,
            }

        with patch(
            "organograph.skeleton.detection.pipeline.fit_soft_barrier_primitive_sampled",
            side_effect=fit_body,
        ), patch(
            "organograph.skeleton.detection.pipeline.villus_mask_from_barrier_primitive",
            return_value=np.zeros(4, dtype=bool),
        ), patch(
            "organograph.crypts.vocab.detect_crypts_by_encoding",
            side_effect=detect_candidates,
        ):
            detections, intermediates = detect_crypts_for_skeleton(
                mesh,
                vocab=object(),
                geodesic_fn=lambda *args, **kwargs: None,
                body_barrier_ellipsoid=True,
                return_intermediates=True,
            )

        self.assertEqual(events, ["body_fit", "hks_candidates"])
        self.assertEqual(detections, [])
        self.assertEqual(
            intermediates["body_barrier_ellipsoid"]["fit_stage"],
            "before_hks_candidate_detection",
        )

    def test_sampled_soft_barrier_ellipsoid_records_sample_metadata(self):
        center = np.array([0.2, 0.1, -0.1])
        axes = np.array([2.0, 1.2, 0.8])
        points = make_ellipsoid_points(center, axes, n_u=40, n_v=20)

        fit = fit_soft_barrier_ellipsoid_sampled(
            points,
            sample_fraction=0.2,
            random_seed=12,
            config={
                "barrier_weight": 20.0,
                "underfill_weight": 1.0,
                "center_regularization": 0.01,
                "maxiter": 300,
            },
            require_inside_center=False,
        )

        self.assertEqual(
            fit.metadata["sample_n_vertices"],
            int(np.ceil(0.2 * points.shape[0])),
        )
        self.assertEqual(fit.metadata["full_n_vertices"], points.shape[0])
        np.testing.assert_allclose(fit.center, center, atol=0.25)
        self.assertTrue(np.all(fit.radii > 0.0))

    def test_sampled_vertex_indices_are_deterministic_and_fractional(self):
        idx_a = sampled_vertex_indices(100, sample_fraction=0.2, random_seed=5)
        idx_b = sampled_vertex_indices(100, sample_fraction=0.2, random_seed=5)
        idx_c = sampled_vertex_indices(100, sample_fraction=0.2, random_seed=6)

        self.assertEqual(idx_a.size, 20)
        np.testing.assert_array_equal(idx_a, idx_b)
        self.assertFalse(np.array_equal(idx_a, idx_c))

    def test_protected_mask_filters_detection_regions_recursively(self):
        detections = [
            {
                "crypt_id": "a",
                "crypt_vertices": [0, 1, 2, 3],
                "attachment_region_vertices": [1, 2, 3, 4],
                "neck_side_vertices": [{0, 1}, {2, 3, 4}],
                "daughters": [
                    {
                        "crypt_id": "a.0",
                        "crypt_vertices": [2, 3, 5],
                        "neck_region_vertices": [3, 5],
                        "neck_side_vertices": ({2, 3}, {5}),
                    }
                ],
            }
        ]
        protected = np.zeros(8, dtype=bool)
        protected[[1, 3]] = True

        filtered = protect_detection_regions_from_mask(detections, protected)

        self.assertEqual(filtered[0]["crypt_vertices"], [0, 2])
        self.assertEqual(filtered[0]["attachment_region_vertices"], [2, 4])
        self.assertEqual(filtered[0]["neck_side_vertices"], [0, 2, 4])
        self.assertEqual(filtered[0]["daughters"][0]["crypt_vertices"], [2, 5])
        self.assertEqual(filtered[0]["daughters"][0]["neck_region_vertices"], [5])
        self.assertEqual(filtered[0]["daughters"][0]["neck_side_vertices"], [2, 5])
        self.assertEqual(
            filtered[0]["metadata"]["protected_region_filter"]["n_protected_vertices"],
            2,
        )

    def test_protected_mask_filters_candidate_patches_before_refinement(self):
        protected = np.zeros(8, dtype=bool)
        protected[[1, 3, 4, 6]] = True
        patches, info = protect_patches_from_mask(
            [[0, 1, 2, 3], [3, 4, 6], [5, 6, 7]],
            protected,
            min_vertices=2,
        )

        self.assertEqual([patch.tolist() for patch in patches], [[0, 2], [5, 7]])
        self.assertEqual([record["kept"] for record in info], [True, False, True])
        self.assertEqual([record["removed_size"] for record in info], [2, 3, 1])

    def test_body_blob_fit_is_constrained_before_descendant_tip(self):
        graph = SkeletonGraph()
        graph.add_node("body", "body", [0.0, 0.0, 0.0])
        graph.add_node("crypt_a_attachment", "attachment", [1.0, 0.0, 0.0], crypt_id="a")
        graph.add_node("crypt_a_tip", "tip", [2.0, 0.0, 0.0], crypt_id="a")
        graph.add_edge(
            "body_to_attachment",
            "body",
            "crypt_a_attachment",
            edge_type="body_to_attachment",
            crypt_id="a",
        )
        graph.add_edge(
            "attachment_to_tip",
            "crypt_a_attachment",
            "crypt_a_tip",
            edge_type="attachment_to_tip",
            crypt_id="a",
        )
        points = make_ellipsoid_points(np.zeros(3), np.array([4.0, 1.0, 1.0]))

        attachment = attach_body_primitive(
            graph,
            points,
            primitive_type="ellipsoid",
            axis_quantile=1.0,
            tip_constraint_margin_fraction=0.0,
        )

        center = np.asarray(attachment.parameters["center"], dtype=float)
        direction = graph.node("crypt_a_tip").position - center
        tip_distance = float(np.linalg.norm(direction))
        radius = blob_surface_radius(
            attachment.parameters,
            attachment.primitive_type,
            direction,
        )
        self.assertLessEqual(radius, tip_distance + 1e-8)
        self.assertIn("surface_radius_constraints", attachment.metadata)

    def test_body_blob_fit_adds_attachment_cap_support_points(self):
        graph = SkeletonGraph()
        graph.add_node("body", "body", [0.0, 0.0, 0.0])
        graph.add_node("crypt_a_attachment", "attachment", [1.0, 0.0, 0.0], crypt_id="a")
        graph.add_node("crypt_a_crypt", "crypt", [1.6, 0.0, 0.0], crypt_id="a")
        graph.add_node("crypt_a_tip", "tip", [2.0, 0.0, 0.0], crypt_id="a")
        graph.add_edge(
            "body_to_attachment",
            "body",
            "crypt_a_attachment",
            edge_type="body_to_attachment",
            crypt_id="a",
        )
        graph.add_edge(
            "attachment_to_crypt",
            "crypt_a_attachment",
            "crypt_a_crypt",
            edge_type="attachment_to_crypt",
            crypt_id="a",
        )
        graph.add_edge(
            "crypt_to_tip",
            "crypt_a_crypt",
            "crypt_a_tip",
            edge_type="crypt_to_tip",
            crypt_id="a",
        )
        points = make_ellipsoid_points(np.zeros(3), np.array([2.0, 1.0, 1.0]))

        attachment = attach_body_primitive(
            graph,
            points,
            primitive_type="ellipsoid",
            axis_quantile=0.95,
            cap_support_points_per_attachment=12,
            cap_support_radius_fraction=0.5,
            constrain_to_descendant_tips=False,
        )

        support = attachment.metadata["attachment_cap_support"]
        self.assertTrue(support["enabled"])
        self.assertEqual(support["n_attachments"], 1)
        self.assertEqual(support["attachment_ids"], ["crypt_a_attachment"])
        self.assertGreaterEqual(support["n_points"], 12)
        self.assertEqual(attachment.metadata["n_real_points"], points.shape[0])
        self.assertEqual(
            attachment.metadata["n_points"],
            points.shape[0] + support["n_points"],
        )

    def test_branch_blob_fit_is_constrained_before_daughter_tip(self):
        graph = SkeletonGraph()
        graph.add_node("body", "body", [-3.0, 0.0, 0.0])
        graph.add_node("crypt_split_neck", "neck", [-1.0, 0.0, 0.0], crypt_id="split")
        graph.add_node("crypt_split_branch", "branch", [0.0, 0.0, 0.0], crypt_id="split")
        graph.add_node("crypt_split_tip_0", "tip", [2.0, 0.0, 0.0], crypt_id="split")
        graph.add_edge("body_to_neck", "body", "crypt_split_neck", crypt_id="split")
        graph.add_edge("neck_to_branch", "crypt_split_neck", "crypt_split_branch", crypt_id="split")
        graph.add_edge("branch_to_tip", "crypt_split_branch", "crypt_split_tip_0", crypt_id="split")
        points = make_ellipsoid_points(np.zeros(3), np.array([4.0, 1.0, 1.0]))

        attachment = attach_branch_primitives(
            graph,
            points,
            {"crypt_split_branch": np.arange(points.shape[0])},
            primitive_type="asymmetric_superellipsoid",
            axis_quantile=1.0,
            tip_constraint_margin_fraction=0.0,
        )["crypt_split_branch"]

        center = np.asarray(attachment.parameters["center"], dtype=float)
        direction = graph.node("crypt_split_tip_0").position - center
        tip_distance = float(np.linalg.norm(direction))
        radius = blob_surface_radius(
            attachment.parameters,
            attachment.primitive_type,
            direction,
        )
        self.assertLessEqual(radius, tip_distance + 1e-8)
        self.assertIn("surface_radius_constraints", attachment.metadata)

    def test_asymmetric_superellipsoid_recovers_directional_axes(self):
        eta = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 28)
        omega = np.linspace(-math.pi, math.pi, 48, endpoint=False)
        ee, ww = np.meshgrid(eta, omega, indexing="ij")

        def signed_power(values, exponent):
            return np.sign(values) * np.abs(values) ** exponent

        epsilon_1, epsilon_2 = 0.8, 1.2
        base = np.stack(
            [
                signed_power(np.cos(ee), epsilon_1)
                * signed_power(np.cos(ww), epsilon_2),
                signed_power(np.cos(ee), epsilon_1)
                * signed_power(np.sin(ww), epsilon_2),
                signed_power(np.sin(ee), epsilon_1),
            ],
            axis=-1,
        )
        negative = np.array([1.2, 1.5, 0.9])
        positive = np.array([3.0, 2.0, 1.1])
        points = (base * np.where(base >= 0.0, positive, negative)).reshape(-1, 3)
        fit = fit_asymmetric_superellipsoid_to_points(points)

        self.assertEqual(fit.primitive_type, "asymmetric_superellipsoid")
        fitted_negative = np.asarray(fit.parameters["axis_lengths_negative"])
        fitted_positive = np.asarray(fit.parameters["axis_lengths_positive"])
        directional_ratios = np.maximum(
            fitted_positive / fitted_negative,
            fitted_negative / fitted_positive,
        )
        self.assertGreater(float(np.max(directional_ratios)), 1.5)
        self.assertTrue(0.3 <= fit.parameters["epsilon_1"] <= 2.0)
        self.assertTrue(0.3 <= fit.parameters["epsilon_2"] <= 2.0)

    def test_straight_tapered_tube_fit_recovers_radii(self):
        centerline = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
        points = make_tube_points(centerline, radii=(1.0, 2.0, 0.5))
        fit = fit_crypt_tube_to_points(
            points,
            centerline,
            radius_quantile=0.5,
            neck_window=(0.0, 0.01),
            body_window=(0.48, 0.52),
            tip_window=(0.84, 0.86),
            distal_taper_start=0.85,
            optimize_radius_profile=False,
        )

        self.assertAlmostEqual(fit.parameters["r_neck"], 1.0, delta=0.15)
        self.assertAlmostEqual(fit.parameters["r_body"], 2.0, delta=0.15)
        self.assertAlmostEqual(fit.parameters["r_tip"], 0.5, delta=0.15)
        self.assertAlmostEqual(fit.derived_parameters["length"], 10.0)
        self.assertAlmostEqual(fit.derived_parameters["bend_angle"], 0.0)
        self.assertEqual(
            fit.parameters["distal_taper"],
            "smooth_squared_radius_to_zero",
        )

    def test_tube_fit_optimizes_ordered_profile_positions(self):
        centerline = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
        points = make_tube_points(
            centerline,
            radii=(0.8, 2.0, 0.6),
            n_s=41,
            body_s=0.35,
            distal_taper_start=0.76,
        )
        fit = fit_crypt_tube_to_points(
            points,
            centerline,
            initial_body_position=0.5,
            initial_taper_position=0.85,
        )

        self.assertGreaterEqual(fit.parameters["s_body"], 0.2)
        self.assertLessEqual(fit.parameters["s_body"], 0.7)
        self.assertGreaterEqual(
            fit.parameters["s_taper"],
            fit.parameters["s_body"] + 0.1 - 1e-12,
        )
        self.assertLessEqual(fit.parameters["s_taper"], 0.9)
        self.assertAlmostEqual(fit.parameters["s_body"], 0.35, delta=0.05)
        self.assertAlmostEqual(fit.parameters["s_taper"], 0.76, delta=0.05)
        self.assertTrue(fit.metadata["profile_optimization"]["success"])

    def test_tube_radius_profile_is_smooth_at_taper_control(self):
        from organograph.skeleton.primitive_geometry import capped_tube_radius

        taper = 0.78
        epsilon = 1e-5
        samples = np.linspace(0.0, 1.0, 1001)
        radii = capped_tube_radius(
            samples,
            0.7,
            2.0,
            0.6,
            body_s=0.38,
            taper_start=taper,
        )
        local = capped_tube_radius(
            np.array([taper - epsilon, taper, taper + epsilon]),
            0.7,
            2.0,
            0.6,
            body_s=0.38,
            taper_start=taper,
        )
        left_slope = (local[1] - local[0]) / epsilon
        right_slope = (local[2] - local[1]) / epsilon

        self.assertTrue(np.all(radii >= 0.0))
        self.assertAlmostEqual(radii[-1], 0.0)
        self.assertAlmostEqual(left_slope, right_slope, delta=1e-3)
        self.assertLess(left_slope, -0.1)

    def test_constricted_crypt_tube_recovers_internal_neck_radius(self):
        centerline = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
        points = make_tube_points(
            centerline,
            radii=(1.4, 2.1, 0.6),
            n_s=61,
            body_s=0.5,
            distal_taper_start=0.82,
            constriction_s=0.18,
            r_constriction=0.7,
        )
        fit = fit_crypt_tube_to_points(
            points,
            centerline,
            constriction_s=0.18,
            initial_body_position=0.5,
            initial_taper_position=0.82,
        )

        self.assertAlmostEqual(fit.parameters["s_constriction"], 0.18)
        self.assertAlmostEqual(
            fit.parameters["r_constriction"],
            0.7,
            delta=0.12,
        )
        self.assertLess(
            fit.parameters["r_constriction"],
            fit.parameters["r_neck"],
        )
        self.assertAlmostEqual(
            fit.derived_parameters["constriction_ratio"],
            fit.parameters["r_constriction"] / fit.parameters["r_body"],
        )

    def test_smooth_crypt_centerline_uses_geodesic_band_centers(self):
        vertices = []
        distances = []
        n_rings = 21
        n_theta = 20
        for s in np.linspace(0.0, 1.0, n_rings):
            angle = 0.5 * math.pi * s
            center = np.array([math.sin(angle), 0.0, 1.0 - math.cos(angle)])
            tangent = np.array([math.cos(angle), 0.0, math.sin(angle)])
            normal = np.array([-math.sin(angle), 0.0, math.cos(angle)])
            for theta in np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False):
                offset = 0.15 * (
                    math.cos(theta) * normal
                    + math.sin(theta) * np.array([0.0, 1.0, 0.0])
                )
                vertices.append(center + offset)
                distances.append(1.0 - s)
        vertices = np.asarray(vertices, dtype=float)
        result = estimate_smooth_crypt_centerline(
            vertices,
            np.arange(vertices.shape[0]),
            np.asarray(distances),
            neck_position=[0.0, 0.0, 0.0],
            tip_position=[1.0, 0.0, 1.0],
            n_bands=7,
            n_samples=65,
        )

        centerline = result["centerline_points"]
        np.testing.assert_allclose(centerline[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(centerline[-1], [1.0, 0.0, 1.0])
        expected_midpoint = np.array([math.sqrt(0.5), 0.0, 1.0 - math.sqrt(0.5)])
        np.testing.assert_allclose(centerline[32], expected_midpoint, atol=0.04)
        self.assertEqual(
            result["method"],
            "geodesic_band_centroids_quadratic_bezier",
        )

    def test_smooth_centerline_is_influenced_by_constriction_center(self):
        vertices = []
        distances = []
        neck_level = 1.25
        for s in np.linspace(0.0, 1.0, 21):
            center = np.array([0.0, 0.0, s])
            for theta in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
                vertices.append(
                    center
                    + 0.1
                    * np.array([math.cos(theta), math.sin(theta), 0.0])
                )
                distances.append(neck_level * (1.0 - s))
        vertices = np.asarray(vertices, dtype=float)
        distances = np.asarray(distances, dtype=float)
        constriction = np.array([0.4, 0.0, 0.2])

        unanchored = estimate_smooth_crypt_centerline(
            vertices,
            np.arange(vertices.shape[0]),
            distances,
            neck_position=[0.0, 0.0, 0.0],
            tip_position=[0.0, 0.0, 1.0],
            neck_level=neck_level,
            n_samples=101,
        )
        anchored = estimate_smooth_crypt_centerline(
            vertices,
            np.arange(vertices.shape[0]),
            distances,
            neck_position=[0.0, 0.0, 0.0],
            tip_position=[0.0, 0.0, 1.0],
            neck_level=neck_level,
            n_samples=101,
            constriction_position=constriction,
            constriction_level=1.0,
            constriction_weight=4.0,
        )

        unanchored_distance = np.linalg.norm(
            unanchored["centerline_points"][20] - constriction
        )
        anchored_distance = np.linalg.norm(
            anchored["centerline_points"][20] - constriction
        )
        self.assertLess(anchored_distance, unanchored_distance)
        self.assertTrue(anchored["constriction_used"])
        self.assertAlmostEqual(anchored["constriction_parameter"], 0.2)
        self.assertEqual(
            anchored["method"],
            "geodesic_bands_constriction_anchored_quadratic_bezier",
        )
        np.testing.assert_allclose(
            anchored["centerline_points"][[0, -1]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        )

    def test_bent_tube_fit_reports_length_and_bend_angle(self):
        centerline = np.array(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0]],
            dtype=float,
        )
        points = make_tube_points(centerline, radii=(1.0, 1.0, 1.0))
        fit = fit_crypt_tube_to_points(
            points,
            centerline,
            optimize_radius_profile=False,
        )

        self.assertAlmostEqual(fit.derived_parameters["length"], 10.0)
        self.assertAlmostEqual(fit.derived_parameters["bend_angle"], math.pi / 2.0)
        self.assertAlmostEqual(fit.derived_parameters["tortuosity"], math.sqrt(2.0))

    def test_primitive_attachments_survive_json_round_trip(self):
        graph = build_skeleton_from_crypt_detections(
            VERTICES,
            FACES,
            [
                {
                    "crypt_id": "a",
                    "neck_position": [0.0, 0.0, 0.0],
                    "tip_position": [0.0, 0.0, 2.0],
                    "crypt_vertices": [1, 2, 4],
                }
            ],
            body_center=[0.0, 0.0, -1.0],
            bend_strategy="crypt_centroid",
        )
        attach_body_primitive(graph, VERTICES)
        tube_points = make_tube_points(
            np.vstack(
                [
                    graph.node("crypt_a_neck").position,
                    graph.node("crypt_a_crypt").position,
                    graph.node("crypt_a_tip").position,
                ]
            )
        )
        attach_crypt_tube_primitives(graph, VERTICES, {"a": tube_points})

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skeleton_with_primitives.json"
            save_skeleton_json(graph, path)
            loaded = load_skeleton_json(path)

        self.assertEqual(graph.body_node().primitive_attachment.primitive_type, "ellipsoid")
        self.assertEqual(loaded.body_node().primitive_attachment.primitive_type, "ellipsoid")
        self.assertEqual(len(loaded.primitive_attachments), 1)
        attachment = next(iter(loaded.primitive_attachments.values()))
        self.assertEqual(attachment.primitive_type, "tapered_capped_tube")
        self.assertGreater(len(attachment.parameters["centerline_points"]), 3)
        self.assertTrue(
            loaded.node("crypt_a_crypt").metadata[
                "position_refined_from_smooth_centerline"
            ]
        )

    def test_bulged_crypt_centerline_smoothing_can_be_disabled(self):
        graph = build_skeleton_from_crypt_detections(
            VERTICES,
            FACES,
            [
                {
                    "crypt_id": "bulged",
                    "attachment_position": [0.0, 0.0, 0.0],
                    "tip_position": [0.0, 0.0, 2.0],
                    "crypt_vertices": [1, 2, 4],
                    "neck_profile": {
                        "kind": "transition",
                        "attachment_level": 1.0,
                    },
                }
            ],
            body_center=[0.0, 0.0, -1.0],
            bend_strategy="crypt_centroid",
        )
        original_crypt_position = graph.node("crypt_bulged_crypt").position.copy()
        graph_centerline = np.vstack(
            [
                graph.node("crypt_bulged_attachment").position,
                graph.node("crypt_bulged_crypt").position,
                graph.node("crypt_bulged_tip").position,
            ]
        )
        tube_points = make_tube_points(graph_centerline)

        attachments = attach_crypt_tube_primitives(
            graph,
            VERTICES,
            {"bulged": tube_points},
            centerline_data={
                "bulged": {
                    "vertex_indices": [1, 2, 4],
                    "distance_field": np.linspace(0.0, 1.0, VERTICES.shape[0]),
                    "neck_level": 1.0,
                    "neck_profile": {"kind": "transition", "attachment_level": 1.0},
                }
            },
            smooth_centerline=True,
            smooth_bulged_centerlines=False,
        )

        attachment = next(iter(attachments.values()))
        self.assertEqual(attachment.metadata["centerline_method"], "straight_attachment_to_tip")
        self.assertTrue(attachment.metadata["bulged_centerline_smoothing_disabled"])
        np.testing.assert_allclose(
            attachment.parameters["centerline_points"],
            graph_centerline[[0, -1]],
        )
        np.testing.assert_allclose(
            graph.node("crypt_bulged_crypt").position,
            original_crypt_position,
        )
        self.assertNotIn(
            "position_refined_from_smooth_centerline",
            graph.node("crypt_bulged_crypt").metadata,
        )

    def test_blend_attachments_are_visualization_only(self):
        graph = SkeletonGraph()
        graph.add_node("body", "body", [0.0, 0.0, 0.0])
        graph.add_node("crypt_0_attachment", "attachment", [1.8, 0.0, 0.0], crypt_id=0)
        graph.add_node("crypt_0_crypt", "crypt", [2.6, 0.0, 0.0], crypt_id=0)
        graph.add_node("crypt_0_tip", "tip", [3.2, 0.0, 0.0], crypt_id=0)
        graph.add_edge(
            "crypt_0_body_to_attachment",
            "body",
            "crypt_0_attachment",
            edge_type="body_to_attachment",
            crypt_id=0,
        )
        graph.add_edge(
            "crypt_0_attachment_to_crypt",
            "crypt_0_attachment",
            "crypt_0_crypt",
            edge_type="attachment_to_crypt",
            crypt_id=0,
        )
        graph.add_edge(
            "crypt_0_crypt_to_tip",
            "crypt_0_crypt",
            "crypt_0_tip",
            edge_type="crypt_to_tip",
            crypt_id=0,
        )
        graph.body_node().primitive_attachment = PrimitiveAttachment(
            primitive_type="ellipsoid",
            parameters={
                "center": np.array([0.0, 0.0, 0.0]),
                "orientation": np.eye(3),
                "axis_lengths": np.array([3.0, 3.0, 3.0]),
            },
            attachment_type="node",
            target_ids=["body"],
        )
        graph.add_primitive_attachment(
            "crypt_0_path_0",
            PrimitiveAttachment(
                primitive_type="tapered_capped_tube",
                parameters={
                    "centerline_points": np.array(
                        [[1.8, 0.0, 0.0], [1.8, 0.2, 0.0], [3.2, 0.0, 0.0]],
                        dtype=float,
                    ),
                    "r_neck": 0.2,
                    "r_body": 0.35,
                    "r_tip": 0.08,
                    "s_body": 0.5,
                    "s_taper": 0.85,
                },
                attachment_type="path",
                target_ids=["crypt_0_attachment", "crypt_0_crypt", "crypt_0_tip"],
            ),
        )
        blends = create_attachment_blends(
            graph,
            vertices=np.array(
                [[1.6, 0.2, 0.0], [1.7, -0.2, 0.0], [2.0, 0.15, 0.0]],
                dtype=float,
            ),
            config=BlendConfig(
                extension_length_fraction=0.5,
            ),
        )
        self.assertEqual(list(blends), ["blend_crypt_0_path_0"])
        blend = blends["blend_crypt_0_path_0"]
        self.assertEqual(blend.blend_type, "tapered_attachment_extension_tube")
        self.assertFalse(blend.metadata["vae_parameter"])
        self.assertNotIn("blend_crypt_0_path_0", graph.primitive_attachments)
        self.assertEqual(blend.target_ids[0], "body")
        self.assertAlmostEqual(blend.parameters["r_crypt"], 0.2)
        self.assertAlmostEqual(
            blend.parameters["r_host"],
            math.sqrt(9.0 - 0.9**2) - 1.8,
            places=6,
        )
        self.assertEqual(
            blend.parameters["radius_profile"],
            "linear_host_local_to_attachment",
        )
        self.assertEqual(
            blend.diagnostics["host_radius_source"],
            "endpoint_disk_expanded_to_host_primitive",
        )
        self.assertAlmostEqual(blend.diagnostics["length"], 0.9)
        self.assertAlmostEqual(
            blend.diagnostics["attachment_to_host_node_distance"],
            1.8,
        )
        np.testing.assert_allclose(
            blend.parameters["centerline_points"],
            [[1.8, -0.9, 0.0], [1.8, 0.0, 0.0]],
        )

    def test_primitive_components_cut_body_and_branch_at_necks(self):
        vertices = np.zeros((9, 3), dtype=float)
        graph = build_skeleton_from_crypt_detections(
            vertices,
            np.empty((0, 3), dtype=np.int64),
            [
                {
                    "crypt_id": "split",
                    "neck_position": [0.0, 0.0, 0.0],
                    "branch_position": [1.0, 0.0, 0.0],
                    "neck_region_vertices": [2, 3, 4, 5, 6, 7],
                    "daughters": [
                        {
                            "neck_position": [1.0, 1.0, 0.0],
                            "tip_position": [1.0, 2.0, 0.0],
                            "neck_region_vertices": [5, 6],
                            "crypt_vertices": [5, 6],
                        },
                        {
                            "neck_position": [1.0, -1.0, 0.0],
                            "tip_position": [1.0, -2.0, 0.0],
                            "neck_region_vertices": [7],
                            "crypt_vertices": [7],
                        },
                    ],
                }
            ],
            body_center=[0.0, 0.0, 0.0],
        )
        components = primitive_components_from_crypt_detections(
            vertices,
            [
                {
                    "crypt_id": "split",
                    "neck_region_vertices": [2, 3, 4, 5, 6, 7],
                    "daughters": [
                        {"neck_region_vertices": [5, 6], "crypt_vertices": [5, 6]},
                        {"neck_region_vertices": [7], "crypt_vertices": [7]},
                    ],
                }
            ],
            graph=graph,
        )

        self.assertEqual(components["body"], [0, 1, 8])
        self.assertEqual(components["branches"]["crypt_split_branch"], [2, 3, 4])
        self.assertEqual(components["crypts"]["crypt_split_tip_0"], [5, 6])
        self.assertEqual(components["crypts"]["crypt_split_tip_1"], [7])

    def test_body_branch_neck_cylinder_fits_before_blob_components(self):
        theta = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
        ring = np.column_stack(
            [
                np.zeros(theta.size),
                np.cos(theta),
                np.sin(theta),
            ]
        )
        body_points = np.array(
            [
                [-3.0, 0.0, 0.0],
                [-2.0, 1.5, 0.0],
                [-2.0, -1.5, 0.0],
                [-2.0, 0.0, 1.5],
            ]
        )
        branch_points = np.array(
            [
                [3.0, 0.0, 0.0],
                [2.0, 1.5, 0.0],
                [2.0, -1.5, 0.0],
                [2.0, 0.0, 1.5],
            ]
        )
        vertices = np.vstack([ring, body_points, branch_points])
        boundary = np.arange(ring.shape[0])
        fit = fit_straight_neck_cylinder(
            vertices,
            boundary,
            body_center=[-2.0, 0.0, 0.0],
            neck_center=[0.0, 0.0, 0.0],
            branch_center=[2.0, 0.0, 0.0],
            max_extent_fraction=0.25,
        )

        self.assertEqual(fit.primitive_type, "straight_cylinder")
        self.assertAlmostEqual(fit.parameters["radius"], 1.0, delta=0.05)
        np.testing.assert_allclose(fit.parameters["axis"], [1.0, 0.0, 0.0])
        self.assertLessEqual(fit.derived_parameters["length"], 1.0 + 1e-12)

        graph = build_skeleton_from_crypt_detections(
            vertices,
            np.empty((0, 3), dtype=np.int64),
            [
                {
                    "crypt_id": "split",
                    "neck_position": [0.0, 0.0, 0.0],
                    "branch_position": [2.0, 0.0, 0.0],
                    "neck_region_vertices": list(range(24, 28)),
                    "daughters": [
                        {
                            "neck_position": [2.0, 1.0, 0.0],
                            "tip_position": [2.0, 2.0, 0.0],
                            "crypt_vertices": [28],
                        },
                        {
                            "neck_position": [2.0, -1.0, 0.0],
                            "tip_position": [2.0, -2.0, 0.0],
                            "crypt_vertices": [29],
                        },
                    ],
                }
            ],
            body_center=[-2.0, 0.0, 0.0],
        )
        result = attach_body_branch_neck_primitives(
            graph,
            vertices,
            {
                "crypt_split_neck_cylinder": {
                    "body_node_id": "body",
                    "neck_node_id": "crypt_split_neck",
                    "branch_node_id": "crypt_split_branch",
                    "boundary_vertices": boundary,
                }
            },
            body_component=np.arange(vertices.shape[0]),
            branch_components={"crypt_split_branch": np.arange(vertices.shape[0])},
            max_extent_fraction=0.25,
        )
        self.assertIn("crypt_split_neck_cylinder", graph.primitive_attachments)
        self.assertLess(len(result["body"]), vertices.shape[0])
        self.assertLess(
            len(result["branches"]["crypt_split_branch"]),
            vertices.shape[0],
        )

        graph.body_node().primitive_attachment = PrimitiveAttachment(
            primitive_type="ellipsoid",
            parameters={
                "center": np.array([-2.0, 0.0, 0.0]),
                "orientation": np.eye(3),
                "axis_lengths": np.array([3.0, 2.0, 2.0]),
            },
            attachment_type="node",
            target_ids=["body"],
        )
        graph.node("crypt_split_branch").primitive_attachment = PrimitiveAttachment(
            primitive_type="ellipsoid",
            parameters={
                "center": np.array([2.0, 0.0, 0.0]),
                "orientation": np.eye(3),
                "axis_lengths": np.array([3.0, 2.0, 2.0]),
            },
            attachment_type="node",
            target_ids=["crypt_split_branch"],
        )
        blends = create_attachment_blends(graph, config=BlendConfig(n_samples=33))
        self.assertIn("blend_crypt_split_neck_cylinder", blends)
        blend = blends["blend_crypt_split_neck_cylinder"]
        self.assertEqual(blend.blend_type, "body_branch_neck_replacement_tube")
        self.assertEqual(
            blend.metadata["replaces_primitive_attachment_id"],
            "crypt_split_neck_cylinder",
        )
        self.assertEqual(
            blend.parameters["radius_profile"],
            "linear_body_neck_branch",
        )
        self.assertAlmostEqual(
            blend.parameters["r_neck"],
            graph.primitive_attachments["crypt_split_neck_cylinder"].parameters["radius"],
        )
        centerline = np.asarray(blend.parameters["centerline_points"], dtype=float)
        body = graph.node("body").position
        neck = graph.node("crypt_split_neck").position
        branch = graph.node("crypt_split_branch").position
        np.testing.assert_allclose(centerline[0], 0.5 * (body + neck))
        np.testing.assert_allclose(centerline[16], neck)
        np.testing.assert_allclose(centerline[-1], 0.5 * (branch + neck))

    def test_neck_from_distance_field_uses_ring_center(self):
        vertices = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
            ],
            dtype=np.int64,
        )
        graph = build_skeleton_from_crypt_detections(
            vertices,
            faces,
            [
                {
                    "crypt_id": "ring",
                    "crypt_vertices": [0, 1, 2, 3, 4],
                    "bottom_vertex_id": 0,
                    "d_crypt": np.array([0.0, 1.0, 1.0, 1.0, 1.0]),
                }
            ],
            body_center=[0.0, 0.0, -1.0],
        )

        np.testing.assert_allclose(
            graph.node("crypt_ring_neck").position,
            [0.0, 0.0, 0.0],
            atol=1e-12,
        )

    def test_neck_ring_center_uses_full_contour_not_partial_patch_arc(self):
        vertices = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 4],
                [0, 4, 1],
            ],
            dtype=np.int64,
        )
        graph = build_skeleton_from_crypt_detections(
            vertices,
            faces,
            [
                {
                    "crypt_id": "partial",
                    "crypt_vertices": [0, 1],
                    "bottom_vertex_id": 0,
                    "d_crypt": np.array([0.0, 1.0, 1.0, 1.0, 1.0]),
                }
            ],
            body_center=[0.0, 0.0, -1.0],
        )

        np.testing.assert_allclose(
            graph.node("crypt_partial_neck").position,
            [0.0, 0.0, 0.0],
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()

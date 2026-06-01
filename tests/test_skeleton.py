import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from organograph.skeleton.build import _grow_parent_patch_to_neck, _select_hks_tips_from_axis
from organograph.skeleton import (
    attach_body_primitive,
    attach_crypt_tube_primitives,
    build_skeleton_from_crypt_detections,
    crypt_bend_angle,
    crypt_path_length,
    crypt_straight_distance,
    crypt_tortuosity,
    fit_crypt_tube_to_points,
    fit_ellipsoid_to_points,
    load_skeleton_json,
    number_of_crypts,
    number_of_split_crypts,
    primitive_components_from_crypt_detections,
    save_skeleton_json,
)


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


def make_tube_points(centerline, radii=(1.0, 1.0, 1.0), n_s=21, n_theta=24):
    from organograph.skeleton.primitive_geometry import quadratic_radius

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
        radius = float(quadratic_radius(np.array([s]), *radii)[0])
        for theta in np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False):
            pts.append(center + radius * (math.cos(theta) * normal + math.sin(theta) * binormal))
    return np.asarray(pts, dtype=float)


class SkeletonTests(unittest.TestCase):
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

    def test_straight_tapered_tube_fit_recovers_radii(self):
        centerline = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
        points = make_tube_points(centerline, radii=(1.0, 2.0, 0.5))
        fit = fit_crypt_tube_to_points(
            points,
            centerline,
            radius_quantile=0.5,
            neck_window=(0.0, 0.01),
            body_window=(0.48, 0.52),
            tip_window=(0.99, 1.0),
        )

        self.assertAlmostEqual(fit.parameters["r_neck"], 1.0, delta=0.15)
        self.assertAlmostEqual(fit.parameters["r_body"], 2.0, delta=0.15)
        self.assertAlmostEqual(fit.parameters["r_tip"], 0.5, delta=0.15)
        self.assertAlmostEqual(fit.derived_parameters["length"], 10.0)
        self.assertAlmostEqual(fit.derived_parameters["bend_angle"], 0.0)

    def test_bent_tube_fit_reports_length_and_bend_angle(self):
        centerline = np.array(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0]],
            dtype=float,
        )
        points = make_tube_points(centerline, radii=(1.0, 1.0, 1.0))
        fit = fit_crypt_tube_to_points(points, centerline)

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

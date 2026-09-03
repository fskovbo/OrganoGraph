import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from organograph.skeleton.detection.branch_validation import (
    _grow_parent_patch_to_neck,
    _validate_split_branch_geometry,
)
from organograph.skeleton.detection.graph_builder import build_skeleton_graph
from organograph.skeleton.detection.neck_profiles import analyze_neck_circumference_profile
from organograph.skeleton.detection.tips import _select_hks_tips_from_axis
from organograph.skeleton.detection.attachments import (
    assign_crypt_attachments_from_projected_boundaries,
    barrier_surface_normal,
    find_projected_opening_attachment,
)
from organograph.skeleton.detection.pipeline import detect_crypts_for_skeleton
from organograph.skeleton import (
    BarrierStageResult,
    BlendConfig,
    CryptOverlapConfig,
    DetectionConfig,
    PrimitiveAttachment,
    PrimitiveFitConfig,
    SkeletonGraph,
    SkeletonizationResult,
    fit_primitives_for_skeletonization_result,
    load_skeleton_json,
    save_skeleton_json,
)
from organograph.skeleton.blending import create_attachment_blends
from organograph.skeleton.geometry import (
    crypt_bend_angle,
    crypt_path_length,
    crypt_straight_distance,
    crypt_tortuosity,
    number_of_crypts,
    number_of_split_crypts,
)
from organograph.skeleton.primitive import (
    BarrierPrimitiveFit,
    CryptOverlapAssessment,
    TerminalCryptReference,
    assess_crypt_primitive_overlaps,
    attach_body_branch_neck_primitives,
    attach_body_primitive,
    attach_branch_primitives,
    attach_crypt_tube_primitives,
    crypt_terminal_paths,
    fit_asymmetric_superellipsoid_to_points,
    fit_barrier_primitive,
    fit_barrier_primitive_sampled,
    fit_crypt_tube_to_points,
    fit_ellipsoid_to_points,
    fit_straight_neck_cylinder,
    merge_overlapping_crypt_detections,
    primitive_components_from_crypt_detections,
    relative_height_field,
    tube_overlap_fraction,
)
from organograph.skeleton.primitive.barriers import (
    exclude_host_vertices_from_detections,
    exclude_host_vertices_from_patches,
    host_mask_from_barrier,
    sampled_vertex_indices,
)
from organograph.skeleton.primitive.crypt_geometry import sample_tangent_hermite
from organograph.skeleton.primitive.radius_profiles import fitted_radius_volume_center
from organograph.plotting.skeletons import _tube_surface
from organograph.skeleton.primitive.blobs import blob_surface_radius
from organograph.plotting.skeletons import _centerline_curvature_profile, _primitive_mesh
from organograph.mesh.geodesics import compute_geodesics_dijkstra


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
    @staticmethod
    def _overlap_tube(offset=(0.0, 0.0, 0.0)):
        offset = np.asarray(offset, dtype=float)
        return PrimitiveAttachment(
            primitive_type="tapered_capped_tube",
            parameters={
                "centerline_points": np.array(
                    [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    dtype=float,
                )
                + offset,
                "r_neck": 0.5,
                "r_body": 0.5,
                "r_tip": 0.5,
                "r_taper": 0.5,
                "s_body": 0.5,
                "s_taper": 0.85,
            },
        )

    def test_tube_overlap_fraction_uses_smaller_volume(self):
        first = self._overlap_tube()
        identical = self._overlap_tube()
        separate = self._overlap_tube((0.0, 3.0, 0.0))

        overlap = tube_overlap_fraction(first, identical, samples=8192, random_seed=3)
        disjoint = tube_overlap_fraction(first, separate, samples=8192, random_seed=3)

        self.assertGreater(overlap["fraction_of_smaller"], 0.95)
        self.assertEqual(disjoint["fraction_of_smaller"], 0.0)

    def test_overlap_skips_crypts_on_opposite_sides_of_same_host(self):
        graph = SkeletonGraph()
        graph.add_node("body", "body", [0.0, 0.0, 0.0])
        graph.add_node("crypt_a_attachment", "attachment", [1.0, 0.0, 0.0])
        graph.add_node("crypt_a_tip", "tip", [2.0, 0.0, 0.0], crypt_id="a")
        graph.add_node("crypt_b_attachment", "attachment", [-1.0, 0.0, 0.0])
        graph.add_node("crypt_b_tip", "tip", [-2.0, 0.0, 0.0], crypt_id="b")
        first = self._overlap_tube()
        first.attachment_id = "tube_a"
        first.target_ids = ["crypt_a_attachment", "crypt_a_tip"]
        second = self._overlap_tube()
        second.attachment_id = "tube_b"
        second.target_ids = ["crypt_b_attachment", "crypt_b_tip"]
        primitive_result = SimpleNamespace(
            graph=graph,
            attachments={"crypts": {"tube_a": first, "tube_b": second}},
            components={"crypts": {"a": [0, 1], "b": [2, 3]}},
            skeleton=SimpleNamespace(
                detections=[
                    {"crypt_id": "a", "crypt_vertices": [0, 1]},
                    {"crypt_id": "b", "crypt_vertices": [2, 3]},
                ]
            ),
        )

        filtered = assess_crypt_primitive_overlaps(
            primitive_result,
            CryptOverlapConfig(samples=1024, max_host_attachment_angle=np.pi / 3),
        )
        unfiltered = assess_crypt_primitive_overlaps(
            primitive_result,
            CryptOverlapConfig(samples=1024, max_host_attachment_angle=None),
        )

        self.assertFalse(filtered.requires_merge)
        self.assertTrue(filtered.pairs[0]["skipped"])
        self.assertAlmostEqual(filtered.pairs[0]["attachment_angle"], np.pi)
        self.assertTrue(unfiltered.requires_merge)
        self.assertFalse(unfiltered.pairs[0]["skipped"])

    def test_merge_overlapping_body_crypt_detections_unions_components(self):
        references = [
            TerminalCryptReference(
                attachment_id=f"tube_{name}",
                tip_node_id=f"crypt_{name}_tip",
                host_id="body",
                top_index=index,
                daughter_index=None,
                component_key=name,
                support_size=size,
                path_length=length,
            )
            for index, (name, size, length) in enumerate(
                [("a", 3, 1.0), ("b", 5, 1.5)]
            )
        ]
        assessment = CryptOverlapAssessment(
            pairs=[
                {
                    "first_attachment_id": "tube_a",
                    "second_attachment_id": "tube_b",
                    "fraction_of_smaller": 0.6,
                }
            ],
            groups=[references],
            threshold=0.3,
        )
        detections = [
            {"crypt_id": "a", "crypt_vertices": [0, 1, 2]},
            {"crypt_id": "b", "crypt_vertices": [2, 3, 4, 5, 6]},
        ]

        merged = merge_overlapping_crypt_detections(
            detections,
            assessment,
            np.zeros((7, 3), dtype=float),
        )

        self.assertTrue(merged.changed)
        self.assertEqual(len(merged.detections), 1)
        self.assertEqual(merged.detections[0]["crypt_id"], "b")
        np.testing.assert_array_equal(
            merged.detections[0]["crypt_vertices"],
            np.arange(7),
        )
        diagnostics = merged.detections[0]["metadata"][
            "crypt_primitive_overlap_merge"
        ]
        self.assertEqual(diagnostics["merged_attachment_ids"], ["tube_a", "tube_b"])

    def test_merge_overlapping_split_daughters_collapses_branch(self):
        references = [
            TerminalCryptReference(
                attachment_id=f"daughter_{index}",
                tip_node_id=f"crypt_split_tip_{index}",
                host_id="crypt_split_branch",
                top_index=0,
                daughter_index=index,
                component_key=f"tip_{index}",
                support_size=4 - index,
                path_length=2.0 - 0.2 * index,
            )
            for index in range(2)
        ]
        assessment = CryptOverlapAssessment(
            pairs=[
                {
                    "first_attachment_id": "daughter_0",
                    "second_attachment_id": "daughter_1",
                    "fraction_of_smaller": 0.5,
                }
            ],
            groups=[references],
        )
        vertices = np.column_stack(
            [np.arange(8, dtype=float), np.zeros(8), np.zeros(8)]
        )
        distance_field = np.arange(8, dtype=float) / 3.0
        detections = [
            {
                "crypt_id": "split",
                "crypt_vertices": [3, 4, 5, 6, 7],
                "attachment_position": vertices[6],
                "daughters": [
                    {
                        "crypt_id": "split.0",
                        "crypt_vertices": [0, 1, 2, 3],
                        "attachment_position": vertices[3],
                        "tip_vertex_id": 0,
                        "d_crypt": distance_field,
                    },
                    {
                        "crypt_id": "split.1",
                        "crypt_vertices": [1, 2, 3],
                        "attachment_position": vertices[3],
                        "tip_vertex_id": 1,
                        "d_crypt": distance_field,
                    },
                ],
            }
        ]

        merged = merge_overlapping_crypt_detections(
            detections,
            assessment,
            vertices,
        )

        self.assertEqual(len(merged.detections), 1)
        detection = merged.detections[0]
        self.assertNotIn("daughters", detection)
        np.testing.assert_allclose(detection["attachment_position"], vertices[6])
        np.testing.assert_allclose(detection["constriction_position"], vertices[3])
        self.assertEqual(detection["neck_profile"]["kind"], "constriction")
        self.assertTrue(merged.records[0]["collapsed_branch"])

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

    def test_constriction_is_kept_in_profile_not_graph_topology(self):
        graph = build_skeleton_graph(
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
        )

        self.assertIn("crypt_budded_attachment", graph.nodes)
        self.assertNotIn("crypt_budded_constriction", graph.nodes)
        self.assertEqual(
            graph.node("crypt_budded_attachment").node_type,
            "attachment",
        )
        path = crypt_terminal_paths(graph, "budded")[0]
        self.assertEqual(path[0], "crypt_budded_attachment")
        self.assertEqual(
            graph.node("crypt_budded_attachment").metadata["neck_profile"]["kind"],
            "constriction",
        )

    def test_transition_profile_builds_attachment_without_constriction(self):
        graph = build_skeleton_graph(
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
            graph.edge("crypt_bulged_attachment_to_crypt").source,
            "crypt_bulged_attachment",
        )

    def test_explicit_body_and_branch_centers_override_region_centroids(self):
        body_center = np.array([-2.0, -1.0, -0.5])
        branch_center = np.array([0.4, 0.5, 0.6])
        graph = build_skeleton_graph(
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
            branch_centers={"crypt_split_branch": branch_center},
        )

        np.testing.assert_allclose(graph.body_node().position, body_center)
        np.testing.assert_allclose(
            graph.node("crypt_split_branch").position,
            branch_center,
        )
        self.assertEqual(
            graph.body_node().metadata["center_source"],
            "body_barrier_primitive",
        )
        self.assertEqual(
            graph.node("crypt_split_branch").metadata["center_source"],
            "branch_barrier_primitive",
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

    def test_tangent_hermite_has_fixed_endpoints_and_endpoint_directions(self):
        curve = sample_tangent_hermite(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.4, 0.0],
            [1.0, -0.3, 0.0],
            n_samples=33,
        )
        np.testing.assert_allclose(curve[[0, -1]], [[0, 0, 0], [2, 0, 0]])
        self.assertGreater(np.dot(curve[1] - curve[0], [1.0, 0.4, 0.0]), 0.0)
        self.assertGreater(np.dot(curve[-1] - curve[-2], [1.0, -0.3, 0.0]), 0.0)
        np.testing.assert_allclose(curve[:, 2], 0.0)

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
        graph = build_skeleton_graph(
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

        self.assertEqual(len(graph.nodes), 4)
        self.assertEqual(len(graph.edges), 3)
        self.assertEqual(number_of_crypts(graph), 1)
        self.assertEqual(len(graph.nodes_for_crypt("a", node_type="crypt")), 1)
        self.assertAlmostEqual(crypt_path_length(graph, "a"), 2.0)
        self.assertAlmostEqual(crypt_straight_distance(graph, "a"), 2.0)
        self.assertAlmostEqual(crypt_tortuosity(graph, "a"), 1.0)
        self.assertAlmostEqual(crypt_bend_angle(graph, "a"), 0.0)

    def test_one_split_crypt(self):
        graph = build_skeleton_graph(
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

        self.assertEqual(len(graph.nodes), 9)
        self.assertEqual(len(graph.edges), 8)
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
        self.assertEqual(len(graph.nodes_for_crypt("split", node_type="crypt")), 2)
        self.assertEqual(len(graph.nodes_for_crypt("split", node_type="tip")), 2)

    def test_graph_always_adds_crypt_centroid_node(self):
        graph = build_skeleton_graph(
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
        )

        self.assertEqual(len(graph.nodes_for_crypt("centroid", node_type="crypt")), 1)
        np.testing.assert_allclose(
            graph.node("crypt_centroid_crypt").position,
            np.mean(VERTICES[[1, 2, 4]], axis=0),
        )

    def test_json_round_trip_preserves_positions_and_topology(self):
        graph = build_skeleton_graph(
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

        fit = fit_barrier_primitive(
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
        self.assertGreater(np.count_nonzero(host_mask_from_barrier(points, fit)), 0)

    def test_soft_barrier_ellipsoid_anisotropy_penalty_reduces_axis_ratio(self):
        center = np.zeros(3)
        axes = np.array([3.0, 1.0, 0.7])
        points = make_ellipsoid_points(center, axes, n_u=36, n_v=16)

        free_fit = fit_barrier_primitive(
            points,
            config={
                "barrier_weight": 20.0,
                "underfill_weight": 1.0,
                "center_regularization": 0.01,
                "maxiter": 300,
            },
            require_inside_center=False,
        )
        penalized_fit = fit_barrier_primitive(
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

        fit = fit_barrier_primitive(
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

    def test_projected_opening_attachment_lies_on_host_surface(self):
        angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        vertices = np.vstack(
            [
                np.column_stack(
                    [np.ones(angles.size), 0.3 * np.cos(angles), 0.3 * np.sin(angles)]
                ),
                [2.0, 0.0, 0.0],
            ]
        )
        faces = np.asarray(
            [[i, (i + 1) % angles.size, angles.size] for i in range(angles.size)],
            dtype=np.int64,
        )
        host_fit = BarrierPrimitiveFit(
            center=np.zeros(3),
            axes=np.eye(3),
            radii=np.ones(3),
        )
        opening = find_projected_opening_attachment(
            vertices, faces, np.arange(vertices.shape[0]), host_fit
        )
        self.assertTrue(opening["found"])
        np.testing.assert_allclose(opening["position"], [1.0, 0.0, 0.0], atol=0.04)
        self.assertAlmostEqual(opening["primitive_level"], 1.0, delta=1e-5)
        normal = barrier_surface_normal(opening["position"], host_fit)
        np.testing.assert_allclose(normal, [1.0, 0.0, 0.0], atol=0.04)

        refined = assign_crypt_attachments_from_projected_boundaries(
            vertices,
            faces,
            [{"crypt_id": "a", "crypt_vertices": np.arange(vertices.shape[0])}],
            host_fit,
        )
        self.assertTrue(
            refined[0]["metadata"]["projected_opening_attachment"]["found"]
        )

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
        fit = BarrierPrimitiveFit(
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
            "organograph.skeleton.detection.pipeline.fit_barrier_primitive_sampled",
            side_effect=fit_body,
        ), patch(
            "organograph.skeleton.detection.pipeline.host_mask_from_barrier",
            return_value=np.zeros(4, dtype=bool),
        ), patch(
            "organograph.crypts.vocab.detect_crypts_by_encoding",
            side_effect=detect_candidates,
        ):
            result = detect_crypts_for_skeleton(
                mesh,
                vocab=object(),
                geodesic_fn=lambda *args, **kwargs: None,
                config=DetectionConfig(),
            )

        self.assertEqual(events, ["body_fit", "hks_candidates"])
        self.assertEqual(result.detections, [])
        self.assertIs(result.barriers.body_fit, fit)

    def test_sampled_soft_barrier_ellipsoid_records_sample_metadata(self):
        center = np.array([0.2, 0.1, -0.1])
        axes = np.array([2.0, 1.2, 0.8])
        points = make_ellipsoid_points(center, axes, n_u=40, n_v=20)

        fit = fit_barrier_primitive_sampled(
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

    def test_primitive_stage_reuses_barriers_by_default(self):
        graph = SkeletonGraph()
        graph.add_node("body", "body", [0.0, 0.0, 0.0])
        fit = BarrierPrimitiveFit(
            center=np.zeros(3),
            axes=np.eye(3),
            radii=np.ones(3),
            success=True,
        )
        skeleton = SkeletonizationResult(
            graph=graph,
            detections=[],
            barriers=BarrierStageResult(
                body_fit=fit,
                body_mask=np.ones(4, dtype=bool),
            ),
            mesh=SimpleNamespace(v=VERTICES[:4], f=FACES[:2]),
        )
        components = {
            "body": np.arange(4, dtype=np.int64),
            "branches": {},
            "body_branch_necks": {},
            "crypts": {},
            "crypt_centerlines": {},
        }
        with patch(
            "organograph.skeleton.workflow.primitive_components_from_crypt_detections",
            return_value=components,
        ), patch("organograph.skeleton.workflow.attach_body_primitive") as refine_body:
            result = fit_primitives_for_skeletonization_result(
                skeleton,
                config=PrimitiveFitConfig(refine_host_primitives=False),
            )

        refine_body.assert_not_called()
        self.assertEqual(result.attachments["body"].metadata["source"], "detection_barrier")
        self.assertIs(graph.node("body").primitive_attachment, result.attachments["body"])

    def test_primitive_stage_merges_overlapping_crypts_and_refits(self):
        detections = [
            {
                "crypt_id": crypt_id,
                "crypt_vertices": [0, 1, 2, 3, 4, 5],
                "attachment_position": [0.0, 0.0, 0.0],
                "attachment_level": 1.0,
                "tip_position": [1.0, 0.0, 0.0],
                "neck_profile": {"kind": "transition", "attachment_level": 1.0},
            }
            for crypt_id in ("a", "b")
        ]
        graph = build_skeleton_graph(
            VERTICES,
            FACES,
            detections,
            body_center=[-1.0, 0.0, 0.0],
        )
        fit = BarrierPrimitiveFit(
            center=np.array([-1.0, 0.0, 0.0]),
            axes=np.eye(3),
            radii=np.ones(3),
            success=True,
        )
        skeleton = SkeletonizationResult(
            graph=graph,
            detections=detections,
            barriers=BarrierStageResult(
                body_fit=fit,
                body_mask=np.ones(VERTICES.shape[0], dtype=bool),
            ),
            intermediates={
                "hks": np.array([[1.0], [100.0], [0.0], [0.0], [0.0], [0.0]]),
                "ts_mesh": np.array([1.0]),
            },
            mesh=SimpleNamespace(v=VERTICES, f=FACES),
            geodesic_fn=compute_geodesics_dijkstra,
        )

        primitives = fit_primitives_for_skeletonization_result(
            skeleton,
            config=PrimitiveFitConfig(
                refine_host_primitives=False,
                crypt_tube_kwargs={
                    "optimize_radius_profile": False,
                },
                crypt_overlap=CryptOverlapConfig(
                    threshold=0.3,
                    samples=4096,
                    max_passes=2,
                ),
            ),
        )

        self.assertEqual(len(skeleton.detections), 1)
        self.assertEqual(len(primitives.attachments["crypts"]), 1)
        self.assertEqual(len(skeleton.graph.nodes_for_crypt("a")), 3)
        self.assertNotIn("tip_position", skeleton.detections[0])
        self.assertIn("d_crypt", skeleton.detections[0])
        self.assertEqual(
            skeleton.detections[0]["boundary_distance_bottom_vertex_id"],
            0,
        )
        self.assertEqual(skeleton.detections[0]["bottom_vertex_id"], 1)
        self.assertAlmostEqual(skeleton.detections[0]["d_crypt"][1], 0.0)
        self.assertGreater(skeleton.detections[0]["d_crypt"][0], 0.0)
        recomputation = skeleton.detections[0]["metadata"][
            "merged_geometry_recomputation"
        ]
        self.assertTrue(recomputation["success"])
        self.assertEqual(recomputation["reason"], "recomputed_from_union_region")
        self.assertEqual(recomputation["distance_field_source"], "final_hks_tip")
        merge_info = primitives.metadata["crypt_overlap_merge"]
        self.assertEqual(merge_info["n_merge_groups"], 1)
        self.assertEqual(len(merge_info["geometry_recomputations"]), 1)
        self.assertTrue(merge_info["converged"])

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

        filtered = exclude_host_vertices_from_detections(detections, protected)

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
        patches, info = exclude_host_vertices_from_patches(
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
            tip_window=(0.84, 0.86),
            center_s=0.5,
            fixed_taper_position=0.85,
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

    def test_tube_fit_uses_derived_center_and_fixed_taper_positions(self):
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
            center_s=0.35,
            fixed_taper_position=0.85,
        )

        expected_center = fitted_radius_volume_center(
            r_attachment=fit.parameters["r_attachment"],
            r_center=fit.parameters["r_center"],
            r_distal=fit.parameters["r_distal"],
            s_center=fit.parameters["s_center"],
            s_taper=fit.parameters["s_taper"],
        )
        self.assertAlmostEqual(fit.parameters["crypt_node_s"], expected_center)
        self.assertNotAlmostEqual(fit.parameters["crypt_node_s"], 0.35, places=3)
        self.assertEqual(fit.parameters["s_taper"], 0.85)
        self.assertTrue(fit.metadata["profile_optimization"]["success"])
        self.assertEqual(
            fit.metadata["volume_center"]["source"],
            "fitted_radius_squared_volume_centroid",
        )

    def test_tube_fit_uses_equal_arclength_radius_observations(self):
        centerline = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
        points = make_tube_points(
            centerline,
            radii=(0.8, 2.0, 0.6),
            n_s=41,
            body_s=0.35,
            distal_taper_start=0.76,
        )
        body_slice = points[(points[:, 2] >= 3.0) & (points[:, 2] <= 4.0)]
        oversampled = np.vstack([points, *([body_slice] * 20)])

        reference = fit_crypt_tube_to_points(points, centerline)
        biased = fit_crypt_tube_to_points(oversampled, centerline)

        for parameter in ("r_neck", "r_body", "r_tip", "s_body", "s_taper"):
            self.assertAlmostEqual(
                reference.parameters[parameter],
                biased.parameters[parameter],
                delta=0.04,
            )
        observations = biased.metadata["profile_observations"]
        self.assertEqual(len(observations["s"]), 20)
        self.assertGreater(max(observations["counts"]), min(observations["counts"]))

    def test_tube_fit_enforces_distal_and_constriction_radius_semantics(self):
        centerline = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
        points = make_tube_points(
            centerline,
            radii=(1.0, 1.2, 1.5),
            n_s=61,
            body_s=0.5,
            distal_taper_start=0.82,
            constriction_s=0.18,
            r_constriction=1.1,
        )
        fit = fit_crypt_tube_to_points(
            points,
            centerline,
            constriction_s=0.18,
            fixed_taper_position=0.82,
        )

        self.assertLess(fit.parameters["r_tip"], fit.parameters["r_body"])
        self.assertLess(
            fit.parameters["r_constriction"],
            min(fit.parameters["r_neck"], fit.parameters["r_body"]),
        )
        diagnostics = fit.metadata["profile_optimization"]
        self.assertGreaterEqual(diagnostics["n_supported_bins"], 6)

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
            fixed_taper_position=0.82,
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

    def test_centerline_curvature_profile_detects_straight_and_circular_paths(self):
        straight = np.column_stack(
            [np.linspace(0.0, 2.0, 21), np.zeros((21, 2), dtype=float)]
        )
        s_straight, curvature_straight = _centerline_curvature_profile(straight)
        np.testing.assert_allclose(s_straight, np.linspace(0.0, 1.0, 21))
        np.testing.assert_allclose(curvature_straight[1:-1], 0.0, atol=1e-12)

        radius = 2.5
        angle = np.linspace(0.0, 0.5 * np.pi, 81)
        circular = np.column_stack(
            [radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)]
        )
        _, curvature_circular = _centerline_curvature_profile(circular)
        np.testing.assert_allclose(
            curvature_circular[1:-1],
            1.0 / radius,
            rtol=0.01,
        )

    def test_primitive_attachments_survive_json_round_trip(self):
        graph = build_skeleton_graph(
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
        fit = fit_crypt_tube_to_points(
            tube_points,
            np.vstack(
                [
                    graph.node("crypt_a_neck").position,
                    graph.node("crypt_a_crypt").position,
                    graph.node("crypt_a_tip").position,
                ]
            ),
        )
        graph.add_primitive_attachment(
            "crypt_a_path_0",
            fit.to_attachment(
                attachment_type="path",
                attachment_id="crypt_a_path_0",
                target_ids=["crypt_a_neck", "crypt_a_crypt", "crypt_a_tip"],
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skeleton_with_primitives.json"
            save_skeleton_json(graph, path)
            loaded = load_skeleton_json(path)

        self.assertEqual(graph.body_node().primitive_attachment.primitive_type, "ellipsoid")
        self.assertEqual(loaded.body_node().primitive_attachment.primitive_type, "ellipsoid")
        self.assertEqual(len(loaded.primitive_attachments), 1)
        attachment = next(iter(loaded.primitive_attachments.values()))
        self.assertEqual(attachment.primitive_type, "tapered_capped_tube")
        self.assertGreaterEqual(len(attachment.parameters["centerline_points"]), 2)

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
        graph = build_skeleton_graph(
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

        graph = build_skeleton_graph(
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
        graph = build_skeleton_graph(
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
        graph = build_skeleton_graph(
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

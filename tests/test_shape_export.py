from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from organograph.skeleton import (
    PrimitiveAttachment,
    PrimitiveFitResult,
    SkeletonGraph,
    SkeletonizationResult,
    definitive_filter_options,
    definitive_primitive_fit_config,
    definitive_skeletonization_config,
    graph_from_shape_export_payload,
    load_shape_export_graph,
    save_shape_export,
    shape_export_payload,
)
from organograph.skeleton.primitive_geometry import sample_cubic_bezier


def _shape_result(*, with_branch: bool = False) -> PrimitiveFitResult:
    graph = SkeletonGraph()
    graph.add_node("body", "body", [0.0, 0.0, 0.0])
    graph.add_node("crypt_9_attachment", "attachment", [1.0, 0.0, 0.0], crypt_id="9")
    graph.add_node("crypt_9_crypt", "crypt", [1.5, 0.25, 0.0], crypt_id="9")
    graph.add_node("crypt_9_tip", "tip", [2.0, 0.0, 0.0], crypt_id="9")
    graph.add_edge("e0", "body", "crypt_9_attachment", crypt_id="9")
    graph.add_edge("e1", "crypt_9_attachment", "crypt_9_crypt", crypt_id="9")
    graph.add_edge("e2", "crypt_9_crypt", "crypt_9_tip", crypt_id="9")
    if with_branch:
        graph.add_node("crypt_4_branch", "branch", [0.0, 1.0, 0.0], crypt_id="4")
        graph.add_edge("e3", "body", "crypt_4_branch", crypt_id="4")

    graph.node("body").primitive_attachment = PrimitiveAttachment(
        primitive_type="superellipsoid",
        parameters={
            "center": np.zeros(3),
            "orientation": np.eye(3),
            "axis_lengths": np.array([1.0, 0.8, 0.6]),
            "epsilon_1": 0.75,
            "epsilon_2": 1.0,
            "fit_family": "not_exported",
        },
        fit_error=123.0,
        residuals={"rmse": 12.0},
        attachment_type="node",
        attachment_id="body",
        target_ids=["body"],
    )
    controls = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.25, 0.35, 0.0],
            [1.75, 0.35, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    graph.add_primitive_attachment(
        "crypt_9_tube",
        PrimitiveAttachment(
            primitive_type="tapered_capped_tube",
            parameters={
                "centerline_points": sample_cubic_bezier(*controls, n_samples=64),
                "r_neck": 0.2,
                "r_attachment": 0.2,
                "r_body": 0.45,
                "r_tip": 0.15,
                "r_taper": 0.15,
                "s_body": 0.45,
                "s_taper": 0.82,
                "r_constriction": None,
                "s_constriction": None,
            },
            fit_error=0.1,
            residuals={"rmse": 0.1},
            metadata={
                "centerline_method": "skeleton_anchored_bulged_cubic_bezier",
                "centerline_control_points": controls,
                "centerline_band_sizes": [8, 9, 10],
            },
            attachment_type="path",
            attachment_id="crypt_9_tube",
            target_ids=[
                "crypt_9_attachment",
                "crypt_9_crypt",
                "crypt_9_tip",
            ],
        ),
    )

    theta = np.pi / 2.0
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mesh = SimpleNamespace(
        coord_transform={
            "center": np.array([10.0, -2.0, 4.0]),
            "scale": 2.0,
            "rotation": rotation,
        }
    )
    skeleton = SkeletonizationResult(
        graph=graph,
        detections=[],
        metadata={"record": {"dataset": "synthetic", "label_uid": "shape_1"}},
        mesh=mesh,
    )
    return PrimitiveFitResult(
        graph=graph,
        components={},
        skeleton=skeleton,
    )


class ShapeExportV2Test(unittest.TestCase):
    def test_definitive_profile_matches_exported_tutorial_settings(self):
        filter_options = definitive_filter_options()
        skeleton_config = definitive_skeletonization_config(
            filter_options=filter_options
        )
        primitive_config = definitive_primitive_fit_config()

        self.assertEqual(filter_options["min_percent_greater"], 1.0)
        self.assertEqual(
            skeleton_config.detection.barriers.branch_fit_options["barrier_weight"],
            5.0,
        )
        self.assertEqual(
            primitive_config.body_branch_neck_kwargs["radius_quantile"],
            0.25,
        )
        self.assertTrue(
            primitive_config.crypt_tube_kwargs["smooth_bulged_centerlines"]
        )
        self.assertEqual(primitive_config.crypt_overlap.samples, 8192)

    def test_payload_is_minimal_and_reconstructs_final_geometry(self):
        result = _shape_result()
        payload = shape_export_payload(result)

        self.assertEqual(payload["schema_version"], "organograph_shape_v2")
        self.assertEqual(set(payload), {
            "schema_version",
            "sample",
            "coordinate_transform",
            "summary",
            "skeleton",
            "primitives",
        })
        encoded = json.dumps(payload, allow_nan=False)
        self.assertNotIn("fit_error", encoded)
        self.assertNotIn("residuals", encoded)
        self.assertNotIn("detections", encoded)
        self.assertNotIn("centerline_points", encoded)

        reconstructed = graph_from_shape_export_payload(payload)
        np.testing.assert_allclose(
            reconstructed.node("crypt_9_crypt").position,
            result.graph.node("crypt_9_crypt").position,
        )
        original_tube = result.graph.primitive_attachments["crypt_9_tube"]
        reconstructed_tube = reconstructed.primitive_attachments["crypt_9_tube"]
        np.testing.assert_allclose(
            reconstructed_tube.parameters["centerline_points"],
            original_tube.parameters["centerline_points"],
            atol=1e-12,
        )
        self.assertEqual(reconstructed_tube.parameters["r_neck"], 0.2)

    def test_source_coordinates_restore_positions_scales_and_orientations(self):
        result = _shape_result()
        payload = shape_export_payload(result)
        source_graph = graph_from_shape_export_payload(payload, coordinate_system="source")
        matrix = np.asarray(payload["coordinate_transform"]["fitted_to_source"])

        fitted_tip = result.graph.node("crypt_9_tip").position
        expected_tip = fitted_tip @ matrix[:3, :3].T + matrix[:3, 3]
        np.testing.assert_allclose(source_graph.node("crypt_9_tip").position, expected_tip)

        body = source_graph.node("body").primitive_attachment
        np.testing.assert_allclose(body.parameters["axis_lengths"], [2.0, 1.6, 1.2])
        expected_rotation = matrix[:3, :3] / 2.0
        np.testing.assert_allclose(body.parameters["orientation"], expected_rotation)
        tube = source_graph.primitive_attachments["crypt_9_tube"]
        self.assertAlmostEqual(tube.parameters["r_body"], 0.9)

    def test_save_and_load_use_strict_json(self):
        result = _shape_result()
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_shape_export(result, tmp)
            raw = Path(paths["json"]).read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            loaded = load_shape_export_graph(paths["json"])
            self.assertEqual(len(loaded.nodes), len(result.graph.nodes))

    def test_nonfinite_required_geometry_is_rejected(self):
        result = _shape_result()
        result.graph.node("crypt_9_tip").position[0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            shape_export_payload(result)

    def test_branched_shapes_are_marked_ineligible_without_dropping_geometry(self):
        payload = shape_export_payload(_shape_result(with_branch=True))
        self.assertTrue(payload["sample"]["has_branches"])
        self.assertFalse(payload["sample"]["vae_eligible"])
        self.assertEqual(payload["summary"]["n_branches"], 1)

    def test_export_defines_no_crypt_slot_order(self):
        payload = shape_export_payload(_shape_result())
        self.assertFalse(any("slot" in key for key in payload))
        self.assertTrue(
            all("crypt_id" in node for node in payload["skeleton"]["nodes"])
        )


if __name__ == "__main__":
    unittest.main()

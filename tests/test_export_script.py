import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from organograph.skeleton import (
    PrimitiveAttachment, SkeletonGraph, shape_export_payload, shape_quality_payload,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_skeleton_primitives.py"
spec = importlib.util.spec_from_file_location("export_script", SCRIPT_PATH)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportScriptTests(unittest.TestCase):
    def test_perturbation_conditions_reach_both_export_payloads(self):
        graph = SkeletonGraph()
        graph.add_node("body", "body", [0, 0, 0])
        graph.node("body").primitive_attachment = PrimitiveAttachment(
            primitive_type="ellipsoid",
            parameters={
                "center": [0, 0, 0],
                "orientation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "axis_lengths": [1, 1, 1],
            },
            attachment_type="node",
            attachment_id="body",
            target_ids=["body"],
        )
        expected = {
            ("normal", "B02"): "ta-Yapa",
            ("normal", "B03"): "ta-Yapa",
            ("normal", "C06"): "stem-ChirVpaD1",
            ("small", "F02"): "sec-DaptHi",
            ("small", "C04"): "sec-DaptIwp2iMekD1",
            ("small", "G06"): "abs-Iwp2",
        }
        for (group, well), condition in expected.items():
            with self.subTest(group=group, well=well):
                metadata = {
                    "dataset": "perturbations2",
                    "timepoint": group,
                    "well": well,
                    "condition": exporter.condition_for_organoid("perturbations2", group, well),
                }
                self.assertEqual(shape_export_payload(graph, metadata=metadata)["sample"]["condition"], condition)
                self.assertEqual(shape_quality_payload(graph, metadata=metadata)["sample"]["condition"], condition)
        with self.assertRaises(ValueError):
            exporter.condition_for_organoid("perturbations2", "normal", "F02")
        self.assertIsNone(exporter.condition_for_organoid("20250929", "day4p5", "B02"))

    def test_manifest_preserves_prior_datasets_and_skipped_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            old = {"dataset": "20250929", "timepoint": "day4p5", "label_uid": "x", "cell_count": "120"}
            new = {"dataset": "perturbations2", "timepoint": "normal", "label_uid": "x", "condition": "ta-Yapa"}
            exporter.write_manifest(path, [old])
            exporter.write_manifest(path, [new])
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            for key, value in old.items():
                self.assertEqual(rows[0][key], value)
            self.assertEqual(rows[1]["condition"], "ta-Yapa")
            before = path.read_bytes()
            exporter.write_manifest(path, [])
            self.assertEqual(before, path.read_bytes())
            exporter.write_manifest(path, [new])
            self.assertEqual(before, path.read_bytes())

    def test_defaults_preserve_existing_shapes(self):
        parser = exporter.build_arg_parser()
        args = parser.parse_args([])
        self.assertFalse(args.overwrite)
        self.assertEqual(exporter.selected_datasets(args), ["perturbations2"])
        self.assertFalse(parser.parse_args(["--no-overwrite"]).overwrite)

    def test_dry_run_does_not_touch_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "export"
            with patch.object(exporter, "discover_mesh_paths", return_value=[]), patch.object(
                exporter, "load_mesh_dataset_config", return_value={
                    "zarr_name_by_tp": {}, "round_by_tp": {}, "meshname_by_tp": {},
                }
            ), patch.object(exporter, "load_optional_blacklist", return_value=set()), patch.object(
                exporter.np, "load"
            ), patch.object(Path, "exists", return_value=True):
                result = exporter.main(["--dry-run", "--quiet", "--output-root", str(root)])
            self.assertEqual(result, 0)
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_crypt_segmentation.py"


def load_script():
    spec = importlib.util.spec_from_file_location("crypt_segmentation_settings_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    config = {
        "zarr_name_by_tp": {"day3p5": "test.zarr"},
        "round_by_tp": {"day3p5": "0"},
        "meshname_by_tp": {"day3p5": "nnorg"},
    }
    with patch("organograph.io_utils.dataset_config.load_mesh_dataset_config", return_value=config):
        spec.loader.exec_module(module)
    return module


class CryptSegmentationSettingsTests(unittest.TestCase):
    def test_json_records_filter_parameters_and_preserves_names(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(script, "SEG_DIR", tmp), patch.object(script, "VERBOSE", False):
                script.write_crypt_run_settings(elapsed_s=1.5, stats={"segmentations_saved": 0})
            paths = list(Path(tmp).glob("run_settings_*.json"))
            self.assertEqual(len(paths), 1)
            settings = json.loads(paths[0].read_text())
        expected = {
            "filter_crypts_by_hks_percent": {
                "min_percent_greater": 2.0, "t_min": None, "t_max": 10,
            },
            "filter_crypts_by_size": {"min_patch_verts": 25, "min_patch_area": 5},
        }
        self.assertEqual(settings["parameters"]["filter_kwargs"], expected)
        self.assertEqual(settings["postprocessing_functions"]["filters"], script.FILTER_NAMES)
        self.assertEqual(settings["parameters"]["segment_kwargs"], script.SEGMENT_KWARGS)

    def test_filters_and_export_use_the_same_editable_parameters(self):
        script = load_script()
        script.FILTER_KWARGS["filter_crypts_by_hks_percent"].update(
            min_percent_greater=7.0, t_min=0.5, t_max=12,
        )
        script.FILTER_KWARGS["filter_crypts_by_size"].update(
            min_patch_verts=40, min_patch_area=None,
        )
        patches, mesh, seg_vars = [{0, 1, 2}], object(), {}
        for name, configured_filter in zip(script.FILTER_NAMES, script.FILTERS):
            with self.subTest(filter=name), patch.object(script, name, return_value=[True]) as target:
                result = configured_filter(patches, mesh=mesh, seg_vars=seg_vars, return_info=True)
                self.assertEqual(result, [True])
                target.assert_called_once_with(
                    patches, mesh=mesh, seg_vars=seg_vars, return_info=True,
                    **script.FILTER_KWARGS[name],
                )
        with patch.object(script, "write_run_settings") as writer:
            script.write_crypt_run_settings(elapsed_s=0, stats={})
        payload = writer.call_args.kwargs["payload"]
        self.assertEqual(payload["parameters"]["filter_kwargs"], script.FILTER_KWARGS)


if __name__ == "__main__":
    unittest.main()

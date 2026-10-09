from pathlib import Path
import tempfile
import unittest

from organograph.io_utils.path_parsing import discover_mesh_paths, parse_mesh_path


class MeshDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.zarrs = {tp: "sample.zarr" for tp in ["day3p5", "day4p5"]}
        self.rounds = {tp: "0" for tp in self.zarrs}
        self.meshes = {tp: "nnorg" for tp in self.zarrs}

    def mesh_file(self, name, tp="day3p5", well="C07", mesh_name="nnorg"):
        path = (self.root / tp / "sample.zarr" / well[0] / well[1:]
                / "0" / "meshes" / mesh_name / name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    def discover(self, timepoints=None, wells=None):
        return discover_mesh_paths(self.root, timepoints, self.zarrs,
                                   self.rounds, self.meshes, wells)

    def test_mixed_formats_prefer_vtp_per_organoid(self):
        self.mesh_file("1.stl")
        vtp = self.mesh_file("1.vtp")
        stl_only = self.mesh_file("2.stl")
        vtp_only = self.mesh_file("3.vtp")
        self.mesh_file("4.obj")
        expected = sorted([vtp, stl_only, vtp_only])
        self.assertEqual(self.discover(), expected)
        self.assertEqual(self.discover(["day3p5", "day3p5"]), expected)
        self.assertEqual(self.discover(["day3p5"], {"day3p5": ["C07"]}), expected)

    def test_timepoints_wells_and_mesh_folder_remain_restricted(self):
        keep = self.mesh_file("1.stl")
        other_well = self.mesh_file("1.vtp", well="D03")
        other_tp = self.mesh_file("1.stl", tp="day4p5")
        self.mesh_file("2.stl", mesh_name="other")
        self.assertEqual(self.discover(["day3p5"], {"day3p5": ["C07"]}), [keep])
        self.assertEqual(self.discover(["day3p5"]), sorted([keep, other_well]))
        self.assertEqual(self.discover(), sorted([keep, other_well, other_tp]))

    def test_stl_only_and_matching_metadata(self):
        path = self.mesh_file("101.stl")
        self.assertEqual(self.discover(), [path])
        self.assertEqual(parse_mesh_path(path), parse_mesh_path(path[:-4] + ".vtp"))
        self.assertEqual(parse_mesh_path(path)["label_uid"], "day3p5_C07_101")

    def test_empty_missing_config_and_directory_named_as_mesh(self):
        self.assertEqual(self.discover(), [])
        path = Path(self.mesh_file("1.stl"))
        path.unlink()
        path.mkdir()
        self.assertEqual(self.discover(), [])
        with self.assertRaisesRegex(ValueError, "Missing folder config"):
            self.discover(["unknown"])


if __name__ == "__main__":
    unittest.main()

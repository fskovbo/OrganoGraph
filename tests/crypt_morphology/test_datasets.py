import tempfile
from pathlib import Path
import unittest
import numpy as np
from organograph.crypt_morphology.datasets import biological_day, discover_hks_crypts, select_validation_crypts


class DatasetTests(unittest.TestCase):
    def test_age_filter_and_membership_only_reader(self):
        with tempfile.TemporaryDirectory() as root:
            for ds, tp in [("20250929", "day3"), ("20250929", "day3p5"),
                           ("20250929", "day4p5-more"), ("20251201", "day4p5")]:
                folder = Path(root) / ds / "crypt_segmentations_mesh" / tp
                folder.mkdir(parents=True)
                masks = np.empty(1, object); masks[0] = [0, 1, 2]
                # No detector tip, axis, circumference or HKS arrays are supplied.
                np.savez(folder / "example.npz", label_uid="example", mesh_path="/missing/mesh.vtp", crypts_ll=masks)
            records = discover_hks_crypts(root, respect_blacklist=False)
            self.assertEqual(len(records), 3)
            self.assertTrue(all(r["day"] >= 3.5 for r in records))
            self.assertEqual(select_validation_crypts(records), select_validation_crypts(records))

    def test_age_parsing(self):
        self.assertEqual(biological_day("day4p5-more"), 4.5)
        self.assertEqual(biological_day("day3p5"), 3.5)
        with self.assertRaises(ValueError):
            biological_day("normal")

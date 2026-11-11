"""Opt-in integration test against the user's existing, read-only HKS outputs."""
import os
import unittest
from organograph.crypt_morphology import (
    CapBasis, fit_representation, parameterize_patch, reconstruct, surface_distance_metrics,
)
from organograph.crypt_morphology.datasets import discover_hks_crypts, load_hks_crypt


@unittest.skipUnless(os.environ.get("ORGANOGRAPH_DATA_ROOT"), "Set ORGANOGRAPH_DATA_ROOT to run real-data checks")
class RealDataTests(unittest.TestCase):
    def test_five_dataset_timepoint_groups_including_pilot_worst_case(self):
        wanted = [
            ("20250929", "day3p5_A02_31", 0),
            ("20250929", "day4_A05_36", 1),
            ("20250929", "day4p5_A06_55", 1),
            ("20250929", "day4p5-more_C04_109", 0),
            ("20251201", "day4p5_B05_19", 1),
        ]
        records = {(r["dataset"], r["label_uid"], r["crypt_index"]): r
                   for r in discover_hks_crypts(os.environ["ORGANOGRAPH_DATA_ROOT"])}
        basis = CapBasis.build(12)
        for key in wanted:
            with self.subTest(crypt=key):
                patch, metadata = load_hks_crypt(records[key])
                rep = fit_representation(patch, parameterize_patch(patch), basis, metadata=metadata)
                self.assertEqual(rep.diagnostics["data_rank"], basis.n_modes)
                v, f = reconstruct(rep, basis, original_coordinates=True, n_rings=32, n_sectors=96)
                metrics = surface_distance_metrics(patch.vertices, patch.faces, v, f)
                self.assertLess(metrics["relative_symmetric_rmse"], 0.02)
                self.assertEqual(metrics["degenerate_reconstruction_faces"], 0)

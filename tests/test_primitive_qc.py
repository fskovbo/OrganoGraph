from __future__ import annotations

import unittest

from organograph.skeleton import (
    CryptPrimitiveQCConfig,
    crypt_primitive_qc_records,
)


def _payload(*, budded=False, constriction_radius=None):
    nodes = [
        {"node_id": "body", "node_type": "body", "crypt_id": None, "position": [0, 0, 0]},
        {"node_id": "attachment", "node_type": "attachment", "crypt_id": "0", "position": [1, 0, 0]},
        {"node_id": "crypt", "node_type": "crypt", "crypt_id": "0", "position": [2, 0, 0]},
        {"node_id": "tip", "node_type": "tip", "crypt_id": "0", "position": [3, 0, 0]},
    ]
    targets = ["attachment", "crypt", "tip"]
    if budded:
        nodes.insert(
            2,
            {
                "node_id": "constriction",
                "node_type": "constriction",
                "crypt_id": "0",
                "position": [1.4, 0, 0],
            },
        )
        targets.insert(1, "constriction")
    return {
        "sample": {
            "dataset": "test",
            "timepoint": "day4",
            "label_uid": "day4_A01_1",
            "has_branches": False,
        },
        "skeleton": {"nodes": nodes, "edges": []},
        "primitives": [
            {
                "primitive_id": "body",
                "role": "body",
                "parameters": {"axis_lengths": [2.0, 2.0, 2.0]},
            },
            {
                "primitive_id": "crypt_0_path_0",
                "role": "crypt",
                "target_node_ids": targets,
                "parameters": {
                    "centerline_type": "line",
                    "centerline_control_points": [[1, 0, 0], [3, 0, 0]],
                    "centerline_samples": 64,
                    "r_neck": 0.5,
                    "r_body": 1.0,
                    "r_tip": 0.4,
                    "s_body": 0.5,
                    "s_taper": 0.8,
                    "r_constriction": constriction_radius,
                    "s_constriction": 0.2 if constriction_radius is not None else None,
                },
            },
        ],
    }


class CryptPrimitiveQCTest(unittest.TestCase):
    def test_extracts_interpretable_ratios_and_identity(self):
        record = crypt_primitive_qc_records(_payload())[0]
        self.assertEqual(record["sample_key"], "test::day4::day4_A01_1")
        self.assertEqual(record["subtype"], "bulged")
        self.assertAlmostEqual(record["neck_to_body_ratio"], 0.5)
        self.assertAlmostEqual(record["distal_to_body_ratio"], 0.4)
        self.assertAlmostEqual(record["crypt_body_to_host_scale"], 0.5)

    def test_flags_missing_budded_constriction(self):
        record = crypt_primitive_qc_records(_payload(budded=True))[0]
        self.assertTrue(record["flag_missing_budded_constriction"])
        self.assertGreaterEqual(record["qc_severity"], 3)

    def test_flags_nonminimal_constriction(self):
        record = crypt_primitive_qc_records(
            _payload(budded=True, constriction_radius=0.75),
            config=CryptPrimitiveQCConfig(constriction_margin_fraction=0.0),
        )[0]
        self.assertTrue(record["flag_invalid_constriction_minimum"])

    def test_accepts_genuine_constriction(self):
        record = crypt_primitive_qc_records(
            _payload(budded=True, constriction_radius=0.25)
        )[0]
        self.assertFalse(record["flag_missing_budded_constriction"])
        self.assertFalse(record["flag_invalid_constriction_minimum"])

    def test_reads_optional_quality_diagnostics(self):
        quality = {
            "crypt_primitives": [
                {
                    "primitive_id": "crypt_0_path_0",
                    "fit_error": 0.12,
                    "residuals": {"rmse": 0.12, "mae": 0.08},
                    "n_points": 240,
                    "tip_source": "hks",
                    "tip_vertex_id": 42,
                    "centerline_kind": "smooth",
                    "profile_optimization": {
                        "success": True,
                        "nfev": 9,
                        "n_supported_bins": 18,
                        "observation_rmse": 0.05,
                        "outside_volume_proxy": 0.03,
                        "missing_volume_proxy": 0.04,
                        "candidate_score": 0.10,
                    },
                }
            ]
        }
        record = crypt_primitive_qc_records(
            _payload(),
            quality_payload=quality,
        )[0]

        self.assertEqual(record["fit_rmse"], 0.12)
        self.assertEqual(record["fit_mae"], 0.08)
        self.assertTrue(record["optimizer_success"])
        self.assertEqual(record["n_supported_bins"], 18)
        self.assertEqual(record["selected_tip_source"], "hks")
        self.assertEqual(record["selected_centerline_kind"], "smooth")
        self.assertAlmostEqual(record["candidate_score"], 0.10)


if __name__ == "__main__":
    unittest.main()

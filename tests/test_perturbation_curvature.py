import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gnn_exporter", ROOT / "scripts/setup_GNN_dataset.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def notebook_namespace():
    notebook = json.loads((ROOT / "notebooks/curvature_marker_heatmaps.ipynb").read_text())
    ns = {}
    exec("".join(notebook["cells"][1]["source"]), ns)
    config = ast.parse("".join(notebook["cells"][3]["source"]))
    wanted = {
        "DATASET_SELECTION", "GROUP_BY", "CONDITIONS", "GROUP_COLUMN", "TIMEPOINT_GROUPS",
        "EXCLUSIVITY_RULES", "UNMEASURED_MARKERS_BY_DATASET", "NONE_MARKER_NAME",
        "PROLIFERATION_MARKER_ALIASES", "Y_FIELD", "Y_COLUMN",
    }
    nodes = [n for n in config.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "config", "exec"), ns)
    for index in [5, 7, 8]:
        tree = ast.parse("".join(notebook["cells"][index]["source"]))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) or (
            isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {"METRIC_COLUMNS", "METRIC_LABELS"}
                for t in n.targets))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "helpers", "exec"), ns)
    return ns


class PerturbationCurvatureTests(unittest.TestCase):
    def test_conditions_validate_wells_and_folder(self):
        for folder, wells in exporter.CONDITIONS_BY_DATASET["perturbations2"].items():
            for well, condition in wells.items():
                uid = f"{condition}_day4p5_{well}_1"
                self.assertEqual(exporter.condition_for_organoid("perturbations2", folder, uid), condition)
        for uid in ["normal_B02_1", "wrong_day4p5_B02_1", "ta-Yapa_day4p5_C06_1"]:
            with self.assertRaises(ValueError):
                exporter.condition_for_organoid("perturbations2", "normal", uid)
        self.assertIsNone(exporter.condition_for_organoid("20250929", "day4", "day4_A01_1"))

    def test_export_preserves_condition_and_two_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = nx.path_graph(3)
            out = exporter.export_organoid_npz(
                directory, "sample", np.ones((3, 2)), np.eye(3), graph,
                marker_names=["A", "B", "C"], aux_meta={"condition": "ta-Yapa"},
            )
            with np.load(out) as arrays:
                self.assertEqual(arrays["y"].shape, (3, 2))
            aux = json.loads(Path(directory, "organoid_sample_aux.json").read_text())
            self.assertEqual(aux["condition"], "ta-Yapa")

    def test_condition_grouping_and_timepoint_fallback(self):
        ns = notebook_namespace()
        self.assertEqual(ns["group_for_metadata"]({"condition": "ta-Yapa"}), "ta-Yapa")
        for condition in [None, "normal", "unknown"]:
            with self.assertRaises(ValueError):
                ns["group_for_metadata"]({"condition": condition, "npz_path": "sample"})
        ns["GROUP_BY"] = "timepoint"
        self.assertEqual(ns["group_for_metadata"]({"timepoint": "day4p5-more"}), "day4_day4p5")
        self.assertEqual(ns["dataset_from_metadata_paths"]("/data/NicoleData/perturbations2/graphs/x"), "perturbations2")

    def test_missing_unmeasured_markers_and_none(self):
        ns = notebook_namespace()
        names = ["LGR5", "Chroma", "AldoB", "Serotonin", "Lysozyme", "Agr2", "KI67"]
        markers = np.zeros((4, 7), dtype=bool)
        markers[0, [0, 4]] = True
        markers[1, 6] = True
        markers[3, [1, 4]] = True
        out, out_names = ns["apply_marker_exclusivity"](markers, names, "perturbations2")
        control, control_names = ns["apply_marker_exclusivity"](markers, names, "20251201")
        np.testing.assert_array_equal(out, control)
        self.assertEqual(out_names, control_names)
        np.testing.assert_array_equal(out[0], [False, False, False, False, True, False, False, False])
        self.assertTrue(out[1, 6])
        self.assertTrue(out[2, -1])
        self.assertTrue(out[3, -1])  # Conflicts are evaluated on the original matrix.
        self.assertEqual(out_names, names + ["none"])
        with self.assertRaises(KeyError):
            ns["apply_marker_exclusivity"](markers[:, :-1], names[:-1], "perturbations2")
        with self.assertRaises(KeyError):
            ns["apply_marker_exclusivity"](markers, names, "20250929")

    def test_control_selection_and_condition_override(self):
        ns = notebook_namespace()
        control = {"dataset": "20251201", "timepoint": "day4p5", "condition": "legacy"}
        self.assertTrue(ns["metadata_is_selected"](control))
        self.assertEqual(ns["group_for_metadata"](control), "control")
        self.assertFalse(ns["metadata_is_selected"]({**control, "timepoint": "day3p5"}))
        self.assertFalse(ns["metadata_is_selected"]({**control, "dataset": "20250929"}))
        self.assertFalse(ns["metadata_is_selected"]({**control, "timepoint": "day4p5-more"}))

    def test_control_metadata_and_source_graph_fallback(self):
        ns = notebook_namespace()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "organoid_day4p5_B02_1.npz"
            path.with_name(path.stem + "_aux.json").write_text(json.dumps({
                "dataset": "20251201", "timepoint": "day4p5", "label_uid": "day4p5_B02_1",
            }))
            metadata = ns["metadata_for_npz"](path, {})
            self.assertEqual(metadata["dataset"], "20251201")
            self.assertEqual(metadata["condition"], "control")
            graph = root / "20251201/graphs_preprocessed/day4p5/day4p5_B02_1.gpickle"
            graph.parent.mkdir(parents=True)
            graph.touch()
            ns.update(SOURCE_DATA_ROOT=root, SOURCE_GRAPH_SUBDIR="graphs_preprocessed")
            self.assertEqual(ns["resolve_source_graph_path"](path, metadata), graph)

    def test_joint_loader_filters_before_condition_validation(self):
        ns = notebook_namespace()
        metadata = {
            Path("perturb.npz"): dict(dataset="perturbations2", timepoint="normal", condition="ta-Yapa", label_uid="p"),
            Path("control.npz"): dict(dataset="20251201", timepoint="day4p5", label_uid="c"),
            Path("other_day.npz"): dict(dataset="20251201", timepoint="day3p5", label_uid="o"),
            Path("other_batch.npz"): dict(dataset="20250929", timepoint="day4p5", label_uid="o"),
        }
        replacements = {
            "GNN_EXPORT_DIRS": [],
            "load_manifest_tables": lambda _: {},
            "discover_organoid_npz_files": lambda _: list(metadata),
            "metadata_for_npz": lambda p, _: metadata[p],
            "organoid_marker_curvature_rows": lambda p, m, g: [dict(condition=g, dataset=m["dataset"])],
        }
        with patch.dict(ns, replacements):
            result = ns["load_all_marker_curvature_rows"]()
        self.assertEqual(set(result.condition), {"control", "ta-Yapa"})
        self.assertEqual(len(result), 2)
        with patch.dict(ns, {**replacements, "discover_organoid_npz_files": lambda _: [Path("perturb.npz")]}):
            with self.assertRaises(FileNotFoundError):
                ns["load_all_marker_curvature_rows"]()
        with patch.dict(ns, {**replacements, "discover_organoid_npz_files": lambda _: [Path("perturb.npz")] * 2}):
            with self.assertRaises(ValueError):
                ns["load_all_marker_curvature_rows"]()

    def test_pruned_marker_alignment(self):
        ns = notebook_namespace()
        markers = np.eye(3, dtype=bool)
        out = ns["align_graph_markers_to_export"](
            markers, nx.path_graph(3), {"kept_mask": [True, False, True]}, 2,
        )
        np.testing.assert_array_equal(out, markers[[0, 2]])

    def test_equal_organoid_means_within_conditions(self):
        ns = notebook_namespace()
        rows = []
        for uid, condition, value, n in [("a", "ta-Yapa", 1., 100),
                                          ("b", "ta-Yapa", 3., 1),
                                          ("c", "abs-Iwp2", 9., 5)]:
            rows.append(dict(label_uid=uid, condition=condition, marker="LGR5",
                             n_marker_positive_cells=n, frac_marker_positive_cells=0.5,
                             **{metric: value for metric in ns["METRIC_COLUMNS"]}))
        summary = ns["summarize_marker_curvature"](pd.DataFrame(rows)).set_index("condition")
        self.assertEqual(summary.loc["ta-Yapa", "marker_mean_gaussian_curvature"], 2.)
        self.assertEqual(summary.loc["ta-Yapa", "n_organoids"], 2)
        self.assertEqual(summary.loc["abs-Iwp2", "marker_mean_gaussian_curvature"], 9.)


if __name__ == "__main__":
    unittest.main()

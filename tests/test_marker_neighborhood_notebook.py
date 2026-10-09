import ast
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import networkx as nx
import numpy as np
import pandas as pd


class HopEnrichmentNotebookTests(unittest.TestCase):
    def setUp(self):
        notebook = Path(__file__).resolve().parents[1] / "notebooks" / "plot_marker_neighborhood.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        self.ns = {}
        exec("".join(cells[2]["source"]), self.ns)
        # Load configuration without executing mkdir or notebook analysis cells.
        config = ast.parse("".join(cells[4]["source"]))
        config.body = [node for node in config.body if not isinstance(node, ast.Expr)]
        exec(compile(config, str(notebook), "exec"), self.ns)
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            exec(compile(tree, str(notebook), "exec"), self.ns)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ns.update(CACHE_DIR=Path(self.temp.name), TIMEPOINT_ORDER=["day 4.5"],
                       SELECTED_TIMEPOINTS=["day4p5"], USE_CACHE=False, POOL_SELECTED_TIMEPOINTS=False)
        show_patch = patch.object(self.ns["plt"], "show")
        show_patch.start()
        self.addCleanup(show_patch.stop)
        self.addCleanup(self.ns["plt"].close, "all")
        self.graph = nx.cycle_graph(10)
        names = self.ns["MARKER_PANEL_TARGETS"]
        self.graph.graph["marker_names"] = names
        for node in self.graph:
            signal = np.zeros(len(names))
            if node < len(names):
                signal[node] = 1
            self.graph.nodes[node]["markers_bin"] = signal
            self.graph.nodes[node]["markers_bin_raw"] = np.ones(len(names))
        self.ns["load_graph_cached"] = lambda path: self.graph
        self.reference = pd.DataFrame([
            {"sample_label": "day 4.5", "target": target,
             "crypt_reference_probability": 0.0 if target == "other" else 0.1}
            for target in self.ns["HOP_ENRICHMENT_TARGET_MARKERS"] + ["other"]
        ])
        self.table = pd.DataFrame([dict(
            dataset="test", timepoint="day4p5", sample_label="day 4.5",
            time_numeric=4.5, label_uid="test_graph", graph_path="test",
        )])

    def test_all_centers_reach_summary_and_long_export(self):
        ns = self.ns
        sources = ns["HOP_ENRICHMENT_SOURCE_MARKERS"]
        targets = ns["HOP_ENRICHMENT_TARGET_MARKERS"]
        self.assertEqual(sources, ns["KHOP_TARGET_MARKERS"])
        self.assertEqual(ns["KHOP_FOCAL_MARKERS"], ["LGR5"])
        self.assertIn("KI67", sources)
        self.assertIn("unassigned", sources)
        result = ns["compute_hop_pair_enrichment_results"](self.table, max_hops=2, n_perms=11)
        self.assertEqual(set(zip(result.source, result.target)), {(a, b) for a in sources for b in targets})
        summary = ns["summarize_hop_pair_enrichment"](result, max_hops=2)
        self.assertEqual(len(summary), len(sources) * len(targets) * 3)
        self.assertTrue((summary.n_graphs == 1).all())
        exported = ns["hop_pair_enrichment_to_long"](result)
        self.assertEqual(len(exported), len(summary))
        self.assertEqual(set(exported.source), set(sources))
        path = Path(self.temp.name) / "long.csv"
        exported.to_csv(path, index=False)
        self.assertEqual(set(pd.read_csv(path).source), set(sources))
        direct = ns["graph_hop_pair_enrichment_with_null"](
            self.graph, sources[0], targets[0], max_hops=2, n_perms=11,
            rng=np.random.default_rng(ns["RANDOM_SEED"]),
        )
        for key, expected in zip(["observed_enrichment", "null_low", "null_high", "null_mean", "total_counts"], direct):
            np.testing.assert_allclose(result.iloc[0][key], expected)
        for source in sources:
            pairs = [(source, target) for target in targets]
            fig, _, plotted = ns["plot_hop_pair_enrichment_ribbons"](result, pairs, max_hops=2)
            self.assertEqual(set(plotted.source), {source})
            ns["plt"].close(fig)

    def test_missing_sources_and_targets_produce_empty_panels(self):
        for node in self.graph:
            self.graph.nodes[node]["markers_bin"][:] = 0
        pairs = [("LGR5", "KI67"), ("unassigned", "KI67"), ("Missing", "unassigned")]
        result = self.ns["compute_hop_pair_enrichment_results"](self.table, pairs, max_hops=2, n_perms=3)
        self.assertTrue(result.empty)
        fig, axes, summary = self.ns["plot_hop_pair_enrichment_ribbons"](result, pairs, max_hops=2)
        self.assertTrue((summary.n_graphs == 0).all())
        self.assertTrue(summary.observed_mean.isna().all())
        for ax in axes.ravel():
            self.assertIn("No eligible organoids", [text.get_text() for text in ax.texts])

    def test_batched_null_matches_explicit_permutations_exactly(self):
        ns = self.ns
        # Use unequal degrees, repeated neighbors, and unreachable hop shells.
        graph = self.graph.copy()
        graph.remove_edges_from([(0, 1), (4, 5)])
        X, names = ns["effective_marker_matrix"](
            ns["graph_get"](graph, "markers_bin"), graph.graph["marker_names"],
        )
        for source, target in [("LGR5", "KI67"), ("KI67", "LGR5"), ("unassigned", "unassigned")]:
            focal = ns["marker_vector"](X, names, source)
            values = ns["marker_vector"](X, names, target)
            shells = [[] for _ in range(7)]
            for node in np.flatnonzero(focal):
                for neighbor, distance in nx.single_source_shortest_path_length(graph, int(node), cutoff=6).items():
                    shells[distance].append(neighbor)
            fractions = float(values.mean())
            observed = np.array([values[nodes].mean() / fractions if nodes else np.nan for nodes in shells])
            counts = np.array([len(nodes) for nodes in shells], dtype=float)
            rng = np.random.default_rng(12)
            curves = []
            for _ in range(65):  # Cross the 64-permutation batch boundary.
                shuffled = rng.permutation(values)
                curves.append([shuffled[nodes].mean() / fractions if nodes else np.nan for nodes in shells])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                low, high = np.nanpercentile(curves, ns["HOP_NULL_Q"], axis=0)
                mean = np.nanmean(curves, axis=0)
                actual_rng = np.random.default_rng(12)
                actual = ns["graph_hop_pair_enrichment_with_null"](
                    graph, source, target, max_hops=6, n_perms=65, rng=actual_rng,
                )
            for expected, value in zip([observed, low, high, mean, counts], actual):
                np.testing.assert_array_equal(expected, value)
            self.assertEqual(rng.bit_generator.state, actual_rng.bit_generator.state)

    def test_probabilities_include_absent_targets_and_average_organoids_equally(self):
        ns = self.ns
        second = nx.cycle_graph(20)
        second.graph["marker_names"] = list(self.graph.graph["marker_names"])
        for node in second:
            values = np.zeros(len(second.graph["marker_names"]))
            if node < 4:
                values[0] = 1
            second.nodes[node]["markers_bin"] = values
        graphs = {"test": self.graph, "second": second}
        ns["load_graph_cached"] = graphs.__getitem__
        table = pd.concat([self.table, self.table.assign(label_uid="second", graph_path="second")], ignore_index=True)
        data = ns["compute_hop_probability_results"](table, max_hops=12)
        summary = ns["summarize_hop_probabilities"](data)
        row = summary[(summary.source == "LGR5") & (summary.target == "Chroma") & (summary.distance_hops == 1)].iloc[0]
        self.assertEqual(row.n_graphs, 2)
        self.assertAlmostEqual(row.observed_probability_mean, 0.25)
        self.assertAlmostEqual(row.null_probability_mean, 0.05)
        self.assertAlmostEqual(row.excess_probability_mean, 0.20)
        absent = data[(data.label_uid == "second") & (data.source == "LGR5") & (data.target == "Chroma") & (data.distance_hops == 1)].iloc[0]
        self.assertEqual(absent.observed_probability, 0)
        self.assertEqual(absent.null_probability, 0)
        self.assertEqual(absent.total_counts, 8)
        groups = data.groupby(["label_uid", "source", "distance_hops"])
        for _, group in groups:
            if group.total_counts.iloc[0] > 0:
                self.assertAlmostEqual(group.observed_probability.sum(), 1)
                self.assertAlmostEqual(group.null_probability.sum(), 1)
                self.assertAlmostEqual(group.excess_probability.sum(), 0)
            else:
                self.assertTrue(group.observed_probability.isna().all())
        fig, axes = ns["plot_hop_probability_stacks"](ns["compare_hop_probabilities_to_crypts"](summary, self.reference), "LGR5", max_hops=6)
        self.assertEqual(axes.shape, (1, 4))
        for bar in axes[0, 0].patches:
            self.assertAlmostEqual(bar.get_width(), 0.98)
        for bar in axes[0, 2].patches:
            if bar.get_height() > 0:
                self.assertGreaterEqual(bar.get_y(), -1e-12)
            elif bar.get_height() < 0:
                self.assertLessEqual(bar.get_y(), 1e-12)
        path = Path(self.temp.name) / "probability.csv"
        summary.to_csv(path, index=False)
        self.assertEqual(len(pd.read_csv(path)), len(summary))

    def test_probability_stacks_distinguish_missing_and_overlapping_markers(self):
        ns = self.ns
        idx = self.graph.graph["marker_names"].index("KI67")
        self.graph.graph["marker_names"] = [name for name in self.graph.graph["marker_names"] if name != "KI67"]
        for node in self.graph:
            self.graph.nodes[node]["markers_bin"] = np.delete(self.graph.nodes[node]["markers_bin"], idx)
        with self.assertWarnsRegex(UserWarning, "incomplete target panels"):
            data = ns["compute_hop_probability_results"](self.table, max_hops=2)
        self.assertTrue(data.empty)
        summary = ns["summarize_hop_probabilities"](data)
        fig, axes = ns["plot_hop_probability_stacks"](ns["compare_hop_probabilities_to_crypts"](summary, self.reference), "LGR5", max_hops=2)
        self.assertIn("No eligible\norganoids", [text.get_text() for text in axes[0, 0].texts])
        ns["ENABLE_MARKER_EXCLUSIVITY"] = False
        self.graph.nodes[0]["markers_bin"][:2] = 1
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            ns["compute_hop_probability_results"](self.table, target_markers=["LGR5", "Chroma"], max_hops=2)

    def test_smoothed_compositions_preserve_hops_totals_and_missing_gaps(self):
        smooth = self.ns["smooth_hop_probability_segments"]
        hops = np.arange(1, 7)
        probabilities = np.array([
            [0.8, 0.2, 0], [0.1, 0.7, 0.2], [0.4, 0, 0.6],
            [0, 1, 0], [0.2, 0.3, 0.5], [0.4, 0.1, 0.5],
        ])
        segments = smooth(hops, probabilities)
        self.assertEqual(len(segments), 1)
        x, values = segments[0]
        self.assertEqual(x[0], 0.5)
        self.assertEqual(x[-1], 6.5)
        self.assertTrue((values >= 0).all())
        np.testing.assert_allclose(values.sum(axis=1), 1)
        np.testing.assert_allclose(values[np.searchsorted(x, hops)], probabilities, atol=1e-12)
        probabilities[2] = np.nan
        segments = smooth(hops, probabilities)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0][0][-1], 2.5)
        self.assertEqual(segments[1][0][0], 3.5)
        self.assertEqual(smooth(hops, np.full((6, 3), np.nan)), [])
        x, values = smooth([1], [[0.2, 0.8]])[0]
        np.testing.assert_allclose(values, np.tile([0.2, 0.8], (len(x), 1)))

    def test_smoothed_export_matches_plot_coordinates_and_preserves_gaps(self):
        summary = pd.DataFrame({
            "sample_label": ["day 4.5"] * 6, "source": ["LGR5"] * 6,
            "target": ["LGR5", "AldoB"] * 3, "distance_hops": [1, 1, 2, 2, 3, 3],
            "observed_probability_mean": [0.25, 0.75, np.nan, np.nan, 0.6, 0.4],
            "crypt_reference_probability": [0.5] * 6,
        })
        result = self.ns["build_smoothed_hop_probability_plot_table"](
            summary, target_markers=["LGR5", "AldoB"], max_hops=3,
        )
        expected = self.ns["smooth_hop_probability_segments"](
            [1, 2, 3], [[0.25, 0.75], [np.nan, np.nan], [0.6, 0.4]],
        )
        self.assertEqual(set(result.segment_id), {0, 1})
        for segment_id, (x, values) in enumerate(expected):
            for j, target in enumerate(["LGR5", "AldoB"]):
                rows = result[(result.segment_id == segment_id) & (result.target == target)]
                np.testing.assert_array_equal(rows.distance_hops, x)
                np.testing.assert_array_equal(rows.observed_probability_smoothed, values[:, j])
                self.assertTrue((rows.target_order == j).all())
        empty = self.ns["build_smoothed_hop_probability_plot_table"](summary.iloc[:0])
        self.assertTrue(empty.empty)
        self.assertEqual(empty.columns.tolist(), result.columns.tolist())

    def test_crypt_reference_loader_uses_combined_timepoint_roi(self):
        path = Path(self.temp.name) / "crypt_pooled_marker_composition_bi_regions.csv"
        base = dict(figure_group="day 4.5", axis_region="s<0.75", budding_index_category="combined",
                    total_cells=100, n_crypts=5, LGR5_pct_cells=70, Chroma_pct_cells=30)
        table = pd.DataFrame([base, dict(base, budding_index_category="bulged", LGR5_pct_cells=95, Chroma_pct_cells=5),
                              dict(base, axis_region="s<=1", LGR5_pct_cells=10, Chroma_pct_cells=90)])
        table.to_csv(path, index=False)
        result = self.ns["load_crypt_reference_composition"](path, target_markers=["LGR5", "Chroma"])
        self.assertEqual(set(result.crypt_reference_category), {"combined"})
        np.testing.assert_allclose(result.crypt_reference_probability, [0.7, 0.3, 0])
        self.assertTrue((result.crypt_reference_n_cells == 100).all())
        with self.assertRaisesRegex(ValueError, "Expected one combined"):
            self.ns["load_crypt_reference_composition"](path, target_markers=["LGR5", "Chroma"], timepoint_order=["day 3.5"])
        pd.DataFrame([base, base]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.ns["load_crypt_reference_composition"](path, target_markers=["LGR5", "Chroma"])
        table.to_csv(path, index=False)
        pd.DataFrame([dict(dataset="another-dataset", timepoint="day4p5", sample_label="day 4.5")]).to_csv(
            path.with_name("crypt_composition_by_crypt.csv"), index=False,
        )
        with self.assertRaisesRegex(ValueError, "datasets not selected"):
            self.ns["load_crypt_reference_composition"](path, target_markers=["LGR5", "Chroma"])

    def test_crypt_reference_difference_and_ratio_handle_zero_denominators(self):
        summary = pd.DataFrame({
            "sample_label": ["day 4.5"] * 3, "source": ["LGR5"] * 3,
            "target": ["LGR5", "Chroma", "other"], "distance_hops": [1] * 3,
            "observed_probability_mean": [0.25, 0.75, 0], "n_graphs": [3] * 3,
        })
        reference = pd.DataFrame({"sample_label": ["day 4.5"] * 3,
                                  "target": ["LGR5", "Chroma", "other"],
                                  "crypt_reference_probability": [0, 1, 0]})
        result = self.ns["compare_hop_probabilities_to_crypts"](summary, reference)
        np.testing.assert_allclose(result.observed_minus_crypt, [0.25, -0.25, 0])
        np.testing.assert_allclose(result.observed_over_crypt, [np.nan, 0.75, np.nan], equal_nan=True)
        self.assertEqual(result.crypt_ratio_defined.tolist(), [False, True, False])
        fig, axes = self.ns["plot_hop_probability_stacks"](
            result, "LGR5", target_markers=["LGR5", "Chroma"], max_hops=1,
        )
        self.assertEqual(axes.shape, (1, 4))
        np.testing.assert_allclose(result.crypt_normalized_difference, [1, -1 / 7, np.nan], equal_nan=True)
        self.assertEqual(result.crypt_normalized_difference_defined.tolist(), [True, True, False])
        heatmap = axes[0, 3].images[0]
        np.testing.assert_allclose(heatmap.get_array(), [[1], [-1 / 7]])
        self.assertEqual(heatmap.get_clim(), (-1, 1))
        with self.assertRaisesRegex(ValueError, "matching crypt-composition"):
            self.ns["compare_hop_probabilities_to_crypts"](summary, reference.iloc[:1])

    def test_aldob_is_included_in_all_probability_panels(self):
        summary = pd.DataFrame({
            "sample_label": ["day 4.5"] * 3, "source": ["LGR5"] * 3,
            "target": ["LGR5", "AldoB", "Chroma"], "distance_hops": [1] * 3,
            "observed_probability_mean": [0.25, 0.5, 0.25], "n_graphs": [3] * 3,
        })
        reference = pd.DataFrame({
            "sample_label": ["day 4.5"] * 3,
            "target": ["LGR5", "AldoB", "Chroma"],
            "crypt_reference_probability": [0.25, 0.001, 0.749],
        })
        result = self.ns["compare_hop_probabilities_to_crypts"](summary, reference)
        original = result.copy(deep=True)
        fig, axes = self.ns["plot_hop_probability_stacks"](
            result, "LGR5", target_markers=["LGR5", "AldoB", "Chroma"], max_hops=1,
        )
        self.assertEqual(len(axes[0, 0].patches), 3)
        self.assertEqual(len(axes[0, 1].collections), 3)
        self.assertEqual(len(axes[0, 2].patches), 3)
        heatmap = axes[0, 3].images[0]
        np.testing.assert_allclose(heatmap.get_array(), [[0], [(0.5 - 0.001) / (0.5 + 0.001)], [(0.25 - 0.749) / (0.25 + 0.749)]])
        self.assertEqual([tick.get_text() for tick in axes[0, 3].get_yticklabels()], ["LGR5", "AldoB", "Chroma"])
        np.testing.assert_allclose(heatmap.get_extent(), [0.5, 1.5, 2.5, -0.5])
        self.assertEqual(heatmap.get_clim(), (-1, 1))
        self.assertIn("AldoB", axes[0, 0].get_legend_handles_labels()[1])
        pd.testing.assert_frame_equal(result, original)

    def test_normalized_difference_bounds_scale_invariance_and_missing_values(self):
        observed = [0.002, 0.2, 0.0005, 0, 0.1, 0, np.nan]
        baseline = [0.001, 0.1, 0.001, 0.1, 0, 0, 0.1]
        summary = pd.DataFrame({
            "sample_label": ["day 4.5"] * 7, "source": ["LGR5"] * 7,
            "target": ["Chroma"] * 7, "distance_hops": range(1, 8),
            "observed_probability_mean": observed, "n_graphs": [3] * 6 + [0],
        })
        # Separate reference labels exercise different baselines in the calculation.
        calculation = summary.assign(target=[str(i) for i in range(7)])
        reference = calculation[["sample_label", "target"]].assign(crypt_reference_probability=baseline)
        result = self.ns["compare_hop_probabilities_to_crypts"](calculation, reference)
        np.testing.assert_allclose(result.crypt_normalized_difference,
                                   [1 / 3, 1 / 3, -1 / 3, -1, 1, np.nan, np.nan], equal_nan=True)
        self.assertEqual(result.crypt_normalized_difference_defined.tolist(), [True] * 5 + [False] * 2)
        result["target"] = "Chroma"
        fig, axes = self.ns["plot_hop_probability_stacks"](
            result, "LGR5", target_markers=["Chroma"], max_hops=7,
        )
        heatmap = axes[0, 3].images[0]
        values = heatmap.get_array()
        np.testing.assert_allclose(values.filled(np.nan), [[1 / 3, 1 / 3, -1 / 3, -1, 1, np.nan, np.nan]], equal_nan=True)
        np.testing.assert_array_equal(np.ma.getmaskarray(values), [[False] * 5 + [True] * 2])
        self.assertEqual(heatmap.get_clim(), (-1, 1))
        self.assertFalse(np.allclose(heatmap.cmap.get_bad(), heatmap.cmap(heatmap.norm(0))))
        self.assertEqual(heatmap.get_interpolation(), "nearest")

    def test_heatmap_uses_consistent_rows_and_scale_across_timepoints(self):
        summary = pd.DataFrame({
            "sample_label": ["day 4.5"] * 4, "source": ["LGR5"] * 4,
            "target": ["Chroma", "LGR5", "Chroma", "LGR5"], "distance_hops": [2, 1, 1, 2],
            "observed_probability_mean": [0.2, 0.2, 0.4, 0], "n_graphs": [3] * 4,
        })
        result = self.ns["compare_hop_probabilities_to_crypts"](summary, self.reference)
        result = pd.concat([result, result.assign(sample_label="day 3.5", crypt_normalized_difference=np.nan, n_graphs=0)])
        self.ns["TIMEPOINT_ORDER"] = ["day 3.5", "day 4.5"]
        fig, axes = self.ns["plot_hop_probability_stacks"](
            result, "LGR5", target_markers=["LGR5", "Chroma"], max_hops=2,
        )
        self.assertEqual(axes.shape, (2, 4))
        for row in range(2):
            self.assertEqual(axes[row, 3].images[0].get_clim(), (-1, 1))
            self.assertEqual([t.get_text() for t in axes[row, 3].get_yticklabels()], ["LGR5", "Chroma"])
        self.assertTrue(np.ma.getmaskarray(axes[0, 3].images[0].get_array()).all())
        np.testing.assert_allclose(axes[1, 3].images[0].get_array(), [[1 / 3, -1], [0.6, 1 / 3]])
        self.assertEqual(len(fig.axes), 9)  # Eight panels and one shared colorbar.
        self.assertIn("No eligible\norganoids", [text.get_text() for text in axes[0, 3].texts])

    def test_day4p5_and_more_pool_with_equal_organoid_weight(self):
        labels = [self.ns["timepoint_label"](tp) for tp in ["day4p5", "day4p5-more"]]
        self.assertEqual(labels, ["day 4.5", "day 4.5"])
        data = pd.DataFrame({
            "sample_label": ["day 4.5"] * 3, "source": ["LGR5"] * 3,
            "target": ["AldoB"] * 3, "distance_hops": [1] * 3,
            "timepoint": ["day4p5", "day4p5-more", "day4p5-more"],
            "observed_probability": [0.9, 0, 0.3],
            "null_probability": [0.1] * 3, "excess_probability": [0.8, -0.1, 0.2],
        })
        summary = self.ns["summarize_hop_probabilities"](data)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0].n_graphs, 3)
        self.assertAlmostEqual(summary.iloc[0].observed_probability_mean, 0.4)
        self.ns["SELECTED_TIMEPOINTS"] = ["day4p5", "day4p5-more"]
        pd.testing.assert_frame_equal(self.ns["filter_to_plot_timepoints"](data), data)

    def test_selected_timepoints_pool_before_organoid_averaging(self):
        self.ns.update(POOL_SELECTED_TIMEPOINTS=True, POOLED_SAMPLE_LABEL="pooled",
                       SELECTED_TIMEPOINTS=["day4", "day4p5", "day4p5-more"], TIMEPOINT_ORDER=["pooled"])
        data = pd.DataFrame({
            "timepoint": ["day4", "day4p5", "day4p5-more"],
            "sample_label": ["day 4", "day 4.5", "day 4.5"],
            "source": ["LGR5"] * 3, "target": ["Chroma"] * 3, "distance_hops": [1] * 3,
            "observed_probability": [0.9, 0, 0.3], "null_probability": [0.1] * 3,
            "excess_probability": [0.8, -0.1, 0.2],
        })
        grouped = self.ns["group_selected_organoids"](data)
        self.assertEqual(set(grouped.sample_label), {"pooled"})
        pd.testing.assert_series_equal(grouped.timepoint, data.timepoint)
        summary = self.ns["summarize_hop_probabilities"](grouped)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary.iloc[0].observed_probability_mean, 0.4)
        cached = pd.concat([data, data.iloc[:1].assign(timepoint="day3p5")], ignore_index=True)
        pd.testing.assert_frame_equal(self.ns["filter_to_plot_timepoints"](cached), grouped)
        self.ns["POOL_SELECTED_TIMEPOINTS"] = False
        pd.testing.assert_frame_equal(self.ns["group_selected_organoids"](grouped), data)

    def test_pooled_crypt_reference_uses_cell_weights_and_excludes_other_dates(self):
        self.ns.update(POOL_SELECTED_TIMEPOINTS=True, POOLED_SAMPLE_LABEL="pooled",
                       SOURCE_TIMEPOINT_ORDER=["day 4", "day 4.5"],
                       SELECTED_TIMEPOINTS=["day4", "day4p5", "day4p5-more"])
        path = Path(self.temp.name) / "reference.csv"
        base = dict(axis_region="s<0.75", budding_index_category="combined")
        pd.DataFrame([
            dict(base, figure_group="day 4", n_crypts=5, total_cells=100, LGR5_pct_cells=70, Chroma_pct_cells=30),
            dict(base, figure_group="day 4.5", n_crypts=15, total_cells=300, LGR5_pct_cells=10, Chroma_pct_cells=90),
            dict(base, figure_group="day 3.5", n_crypts=50, total_cells=1000, LGR5_pct_cells=100, Chroma_pct_cells=0),
            dict(base, figure_group="all timepoints", n_crypts=70, total_cells=1400, LGR5_pct_cells=90, Chroma_pct_cells=10),
        ]).to_csv(path, index=False)
        reference = self.ns["load_crypt_reference_composition"](path, target_markers=["LGR5", "Chroma"])
        self.assertEqual(set(reference.sample_label), {"pooled"})
        np.testing.assert_allclose(reference.crypt_reference_probability, [0.25, 0.75, 0])
        self.assertTrue((reference.crypt_reference_n_cells == 400).all())
        self.assertTrue((reference.crypt_reference_n_crypts == 20).all())
        self.assertEqual(set(reference.crypt_reference_component_labels), {"day 4;day 4.5"})

    def test_panel_exports_have_exact_values_stacks_and_undefined_flags(self):
        targets = ["LGR5", "AldoB", "Chroma", "Glucagon"]
        self.ns["HOP_ENRICHMENT_TARGET_MARKERS"] = targets
        summary = pd.DataFrame({
            "sample_label": ["day 4.5"] * 8, "source": ["LGR5"] * 8,
            "target": targets * 2, "distance_hops": [0] * 4 + [1] * 4,
            "observed_probability_mean": [0.2, 0.6, 0.2, 0] * 2,
            "observed_probability_sem": [0.01] * 8, "n_graphs": [3] * 8,
        })
        reference = pd.DataFrame({
            "sample_label": ["day 4.5"] * 4, "target": targets,
            "crypt_reference_probability": [0.4, 0.1, 0.5, 0],
            "crypt_reference_n_cells": [100] * 4, "crypt_reference_n_crypts": [5] * 4,
        })
        compared = self.ns["compare_hop_probabilities_to_crypts"](summary, reference)
        smoothed = self.ns["build_smoothed_hop_probability_plot_table"](compared, target_markers=targets, max_hops=1)
        raw = pd.DataFrame({"dataset": ["test"], "timepoint": ["day4p5"], "label_uid": ["test_graph"]})
        output = Path(self.temp.name) / "neighborhood_probabilities"
        tables = self.ns["export_neighborhood_probability_tables"](raw, summary, compared, smoothed, reference, directory=output)
        self.assertEqual(len(list(output.glob("*.csv"))), 9)
        observed = tables["panel_1_observed_probability.csv"]
        self.assertEqual(set(observed.distance_hops), {1})
        self.assertEqual(observed.target.tolist(), targets)
        np.testing.assert_allclose(observed.stack_bottom_pct, [0, 20, 80, 100])
        np.testing.assert_allclose(observed.stack_top_pct, [20, 80, 100, 100])
        self.assertTrue(observed.target_color.str.match(r"^#[0-9a-fA-F]{6}$").all())
        difference = tables["panel_3_observed_minus_crypt.csv"]
        np.testing.assert_allclose(difference.observed_minus_crypt_pp, [-20, 50, -30, 0])
        np.testing.assert_allclose(difference.stack_bottom_pp, [0, 0, -20, 50])
        np.testing.assert_allclose(difference.stack_top_pp, [-20, 50, -50, 50])
        heatmap = pd.read_csv(output / "panel_4_normalized_difference.csv")
        np.testing.assert_allclose(heatmap.crypt_normalized_difference, [-1 / 3, 5 / 7, -3 / 7, np.nan], equal_nan=True)
        self.assertEqual(heatmap.crypt_normalized_difference_defined.tolist(), [True, True, True, False])
        smooth = tables["panel_2_smoothed_probability.csv"]
        np.testing.assert_allclose(smooth.groupby(["source", "segment_id", "distance_hops"]).stack_top_pct.max(), 100)
        self.assertTrue((output / "plot_settings.json").exists())


    def test_cache_filename_is_short_and_configuration_sensitive(self):
        cache_path = self.ns["hop_pair_cache_path"]
        path = cache_path()
        self.assertLess(len(path.name.encode()), 255)
        self.assertNotEqual(path, cache_path(pairs=[("LGR5", "LGR5")]))
        self.assertNotEqual(path, cache_path(n_perms=7))
        self.assertNotEqual(path, cache_path(max_hops=2))
        self.ns["SELECTED_TIMEPOINTS"] = ["day3p5"]
        self.assertNotEqual(path, cache_path())


if __name__ == "__main__":
    unittest.main()

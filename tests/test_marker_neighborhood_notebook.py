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
                       SELECTED_TIMEPOINTS=["day4p5"], USE_CACHE=False)
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

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/test_curvature_geometric.ipynb"


def fit_helpers():
    namespace = {"np": np, "pd": pd, "plt": plt}
    spec = importlib.util.spec_from_file_location(
        "mesh_plotting", NOTEBOOK.parents[1] / "src/organograph/plotting/meshes.py")
    plotting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotting)
    namespace["plot_organoid_mesh"] = plotting.plot_organoid_mesh
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"select_hks_fit_examples", "plot_hks_fit_examples",
                                   "plot_hks_slope_distribution", "plot_curvature_comparison",
                                   "compute_hks_time_derivative", "plot_hks_derivative_summary"}]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "fit_helpers", "exec"), namespace)
    return namespace


def debug_data():
    times = np.array([0.1, 0.2, 0.4, 0.8, 1.6])
    slope = np.arange(8, dtype=float) + 0.25
    intercept = np.arange(8, dtype=float) * -0.1
    return dict(
        ts=times, early_slope=slope, early_intercept=intercept,
        early_end_idx=np.full(8, 2), early_r2=np.ones(8),
        fit_signal=slope[:, None] * times + intercept[:, None],
        defect_score=np.array([10., 11., 2., 1., 0., -1., 3., 7.]), zmax=5.,
        defect_mask_pre_dilation=np.array([True, True, False, False, False, False, False, False]),
        defect_mask=np.array([True, True, True, False, False, False, False, False]),
    )


class GeometricCurvatureNotebookTests(unittest.TestCase):
    def test_hks_derivative_matches_spectral_formula_and_finite_difference(self):
        helper = fit_helpers()["compute_hks_time_derivative"]
        mesh = SimpleNamespace(
            eigvals=np.array([0., 0.5, 2.]),
            eigvecs=np.array([[1., 2., 3.], [1., 0.5, 0.2]]),
            _ensure_eigendecomposition=lambda: None,
        )
        times = np.array([0.1, 0.4, 2.])
        result = helper(mesh, times)
        expected = (mesh.eigvecs ** 2) @ (
            -mesh.eigvals[:, None] * np.exp(-mesh.eigvals[:, None] * times))
        np.testing.assert_allclose(result, expected)
        def hks(t):
            return (mesh.eigvecs ** 2) @ np.exp(-mesh.eigvals[:, None] * t)
        step = 1e-6
        np.testing.assert_allclose(result, (hks(times + step) - hks(times - step)) / (2 * step),
                                   rtol=1e-8, atol=1e-9)
        self.assertTrue(np.all(result < 0))

    def test_hks_derivative_rejects_invalid_times(self):
        helper = fit_helpers()["compute_hks_time_derivative"]
        for times in [[0., 1.], [], [1., np.nan], [-1., 1.], [[1., 2.]]]:
            with self.assertRaises(ValueError):
                helper(None, times)

    def test_derivative_statistics_use_only_the_displayed_interval(self):
        helper = fit_helpers()["plot_hks_derivative_summary"]
        times = np.array([0.1, 0.5, 2., 4.])
        curves = np.array([[-8., -4., -1., 100.],
                           [-6., -3., -2., np.nan],
                           [0., 0., 0., 1000.],
                           [np.nan, 1., 2., 3.],
                           [-9., -1., -5., 1.],
                           [5., 1., 0., 300.]])
        mesh = SimpleNamespace(v=np.arange(18).reshape(6, 3),
                               f=np.array([[0, 1, 2], [3, 4, 5]]))
        fig, table, sample = helper(times, curves, mesh, n_sample=1, n_highlight=1)
        means = np.array([-13/3, -11/3, 0., np.nan, -5., 2.])
        minima = np.array([-8., -6., 0., np.nan, -9., 0.])
        np.testing.assert_allclose(table.mean_derivative, means, equal_nan=True)
        np.testing.assert_allclose(table.minimum_derivative, minima, equal_nan=True)
        self.assertEqual(table.included.tolist(), [True, True, True, False, True, True])
        self.assertEqual(table.loc[table.highlighted, "vertex_id"].tolist(), [4])
        self.assertEqual(table.minimum_time.iloc[5], times[2])
        self.assertEqual(len(sample), 1)
        histograms = [t for t in fig.data if t.type == "bar"]
        self.assertEqual(len(histograms), 2)
        for histogram in histograms:
            self.assertEqual(sum(histogram.y), 5)
            self.assertEqual(len(histogram.y), 60)
        np.testing.assert_array_equal(histograms[0].x, histograms[1].x)
        edges = np.histogram_bin_edges(np.r_[means[np.isfinite(means)], minima[np.isfinite(minima)]],
                                       bins=60)
        for histogram, values in zip(histograms, [means, minima]):
            expected, _ = np.histogram(values[np.isfinite(values)], bins=edges)
            np.testing.assert_array_equal(histogram.y, expected)
        surface = next(t for t in fig.data if t.type == "mesh3d")
        np.testing.assert_allclose(surface.customdata, minima, equal_nan=True)
        self.assertEqual(surface.vertexcolor[3], "rgb(190,190,190)")
        color_scale = next(t for t in fig.data if t.name == "Minimum-derivative color scale")
        self.assertEqual(color_scale.marker.cmin, -9.)
        self.assertEqual(color_scale.marker.cmax, 0.)
        np.testing.assert_allclose(fig.layout.xaxis.range, np.log10([0.1, 2.]))
        self.assertEqual(fig.layout.yaxis.type, "linear")
        self.assertEqual(fig.layout.xaxis2.type, "linear")
        self.assertEqual(fig.layout.barmode, "overlay")
        for trace in fig.data:
            if trace.type == "scatter":
                self.assertTrue(np.all(np.asarray(trace.x) <= 2.))

    def test_derivative_sampling_and_empty_curves(self):
        helper = fit_helpers()["plot_hks_derivative_summary"]
        times = np.geomspace(0.1, 2., 50)
        curves = np.full((500, 50), -1.)
        curves[:12] = -2.
        curves[-1] = np.nan
        mesh = SimpleNamespace(v=np.arange(1500).reshape(500, 3), f=np.array([[0, 1, 2]]))
        fig, table, sample = helper(times, curves, mesh)
        _, again, sample2 = helper(times, curves, mesh)
        ids = table.loc[table.highlighted, "vertex_id"].to_numpy()
        self.assertEqual(len(ids), 5)
        self.assertTrue(np.all(ids < 12))
        np.testing.assert_array_equal(ids, again.loc[again.highlighted, "vertex_id"])
        self.assertEqual(len(sample), 400)
        self.assertNotIn(499, sample)
        np.testing.assert_array_equal(sample, sample2)
        for trace in fig.data:
            if trace.type == "bar":
                self.assertEqual(sum(trace.y), 499)
        fig, table, _ = helper(times, np.full_like(curves, np.nan), mesh)
        self.assertFalse(table.included.any())
        self.assertFalse(table.highlighted.any())
        for trace in fig.data:
            if trace.type == "bar":
                self.assertEqual(sum(trace.y), 0)

    def test_derivative_mesh_constant_negative_field_is_not_zero_centered(self):
        helper = fit_helpers()["plot_hks_derivative_summary"]
        mesh = SimpleNamespace(v=np.array([[0., 0, 0], [1, 0, 0], [0, 1, 1]]),
                               f=np.array([[0, 1, 2]]))
        fig, _, _ = helper([0.1, 2.], np.full((3, 2), -4.), mesh)
        scale = next(t for t in fig.data if t.name == "Minimum-derivative color scale").marker
        self.assertLess(scale.cmin, -4.)
        self.assertGreater(scale.cmax, -4.)
        self.assertLess(scale.cmax, 0.)

    def test_comparison_shares_coloraxis_and_scene_bounds(self):
        helper = fit_helpers()["plot_curvature_comparison"]
        mesh = SimpleNamespace(v=np.array([[0., 0, 0], [1, 0, 0], [0, 1, 1]]),
                               f=np.array([[0, 1, 2]]))
        source = SimpleNamespace(v=mesh.v * 2, f=mesh.f)
        stages = [("Initial", mesh, np.array([-2., 0, 2])),
                  ("Final", mesh, np.array([-1., 0, 1])),
                  ("Cell-patch averaged", source, np.array([0.5, 0.5, -0.5]))]
        fig = helper(stages, (-2., 2.), "curvature units",
                     dict(azim=-135, elev=25, projection="orthographic"))
        self.assertEqual(len(fig.data), 3)
        self.assertEqual(fig.layout.coloraxis.cmin, -2.)
        self.assertEqual(fig.layout.coloraxis.cmax, 2.)
        self.assertFalse(fig.layout.coloraxis.cauto)
        for trace, scene_name, (_, mesh, values) in zip(
                fig.data, ["scene", "scene2", "scene3"], stages):
            self.assertEqual(trace.coloraxis, "coloraxis")
            self.assertEqual(trace.scene, scene_name)
            np.testing.assert_array_equal(trace.intensity, values)
            np.testing.assert_array_equal(trace.x, mesh.v[:, 0])
            scene = fig.layout[scene_name]
            self.assertEqual(scene.camera, fig.layout.scene.camera)
            self.assertEqual(scene.camera.projection.type, "orthographic")
            for axis in "xyz":
                self.assertEqual(scene[f"{axis}axis"].range,
                                 fig.layout.scene[f"{axis}axis"].range)
        self.assertGreater(fig.layout.scene.xaxis.range[1], 2.)

    def test_slope_histogram_threshold_and_counts(self):
        helper = fit_helpers()["plot_hks_slope_distribution"]
        debug = debug_data()
        slopes = debug["early_slope"]
        expected = np.median(slopes) + debug["zmax"] * 1.4826 * np.median(
            np.abs(slopes - np.median(slopes)))
        fig, cutoff = helper(debug, bins=4)
        try:
            self.assertAlmostEqual(cutoff, expected)
            ax = fig.axes[0]
            np.testing.assert_allclose(ax.lines[0].get_xdata(), [expected, expected])
            counts = [sum(p.get_height() for p in container) for container in ax.containers]
            self.assertEqual(counts, [6, 2])
            fig.canvas.draw()
        finally:
            plt.close(fig)

    def test_slope_histogram_scale_fallbacks_and_missing_data(self):
        helper = fit_helpers()["plot_hks_slope_distribution"]
        debug = debug_data()
        for slopes in [np.array([0.] * 7 + [8.]), np.ones(8)]:
            debug["early_slope"] = slopes
            fig, cutoff = helper(debug)
            try:
                scale = np.std(slopes) or 1.0
                self.assertAlmostEqual(cutoff, np.median(slopes) + debug["zmax"] * scale)
            finally:
                plt.close(fig)
        for missing in [{}, dict(debug, early_slope=np.full(8, np.nan))]:
            fig, cutoff = helper(missing)
            self.assertIsNone(fig)
            self.assertTrue(np.isnan(cutoff))

    def test_sampling_excludes_dilation_only_and_high_score_normal_vertices(self):
        helper = fit_helpers()["select_hks_fit_examples"]
        debug = debug_data()
        examples = helper(debug)
        self.assertEqual(examples, helper(debug))
        self.assertEqual({v for kind, v in examples if kind == "defect"}, {0, 1})
        self.assertEqual({v for kind, v in examples if kind == "normal"}, {3, 4, 5, 6})
        self.assertEqual(len(set(examples)), 6)

    def test_fewer_examples_and_disabled_detection(self):
        ns = fit_helpers()
        self.assertEqual(ns["select_hks_fit_examples"]({}), [])
        fig, table = ns["plot_hks_fit_examples"]({}, [])
        self.assertIsNone(fig)
        self.assertTrue(table.empty)
        debug = debug_data()
        debug["defect_mask"][:] = True
        debug["defect_mask_pre_dilation"][1] = False
        self.assertEqual(ns["select_hks_fit_examples"](debug), [("defect", 0)])
        debug["early_slope"][0] = np.nan
        self.assertEqual(ns["select_hks_fit_examples"](debug), [])

    def test_plot_uses_saved_coefficients_and_exact_fit_window(self):
        ns = fit_helpers()
        debug = debug_data()
        examples = ns["select_hks_fit_examples"](debug)
        fig, table = ns["plot_hks_fit_examples"](debug, examples)
        try:
            self.assertEqual(len(fig.axes), 6)
            self.assertEqual(table.n_fit_points.tolist(), [3] * 6)
            for ax, (_, vertex) in zip(fig.axes, examples):
                self.assertEqual(ax.get_xscale(), "linear")
                fit = ax.lines[1]
                t = np.asarray(fit.get_xdata())
                self.assertEqual(t[0], debug["ts"][0])
                self.assertEqual(t[-1], debug["ts"][debug["early_end_idx"][vertex]])
                np.testing.assert_allclose(fit.get_ydata(), debug["early_slope"][vertex] * t
                                           + debug["early_intercept"][vertex])
            fig.canvas.draw()
        finally:
            plt.close(fig)

    def test_plot_limits_display_to_seven_times_without_changing_fit(self):
        ns = fit_helpers()
        debug = debug_data()
        debug["ts"] = np.geomspace(0.1, 10, 10)
        debug["early_end_idx"][:] = 9
        debug["fit_signal"] = (debug["early_slope"][:, None] * debug["ts"]
                               + debug["early_intercept"][:, None])
        fig, table = ns["plot_hks_fit_examples"](debug, [("defect", 0)])
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.get_xscale(), "linear")
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), debug["ts"][:7])
            np.testing.assert_array_equal(ax.collections[0].get_offsets()[:, 0], debug["ts"][:7])
            fit = ax.lines[1]
            t = np.asarray(fit.get_xdata())
            self.assertEqual(t[-1], debug["ts"][6])
            np.testing.assert_allclose(fit.get_ydata(), debug["early_slope"][0] * t
                                       + debug["early_intercept"][0])
            self.assertEqual(table.iloc[0].n_fit_points, 10)
            self.assertEqual(table.iloc[0].fit_t_end, debug["ts"][-1])
            fig.canvas.draw()
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()

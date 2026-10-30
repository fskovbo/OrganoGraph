import ast
import json
from pathlib import Path
import re
import tempfile
import unittest

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from organograph.graph.access import graph_marker_index
from organograph.mesh.OrganoidMesh import OrganoidMesh


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/plot_proliferation_symmetry_breaking.ipynb'


def helpers():
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = dict(np=np, json=json, re=re, Path=Path, graph_marker_index=graph_marker_index)
    cell = next(c for c in notebook['cells'] if c['id'] == 'marker-helpers')
    exec(''.join(cell['source']), namespace)
    return namespace


def graph_with_processed_markers():
    graph = nx.Graph()
    graph.graph['marker_names'] = ['Cyclin A', 'Cyclin D', 'Lysozyme', 'Serotonin']
    for node, values in zip([4, 8, 12, 16], [[1, 0, 0, 0], [0, 1, 0, 0],
                                          [1, 1, 0, 0], [0, 0, 0, 1]]):
        graph.add_node(node, markers_bin=values, markers_bin_raw=[1, 1, 1, 1])
    return graph


def tetrahedron(scale=1.0, offset=0.0):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return OrganoidMesh().load_from_arrays(vertices * scale + offset, faces)


class ProliferationNotebookTests(unittest.TestCase):
    def test_ta_excludes_every_non_cyclin_marker_but_counts_double_cyclin_once(self):
        graph = nx.Graph()
        graph.graph['marker_names'] = ['Cyclin A', 'Cyclin D', 'Lysozyme', 'Serotonin', 'Chroma']
        for i, values in enumerate([[1, 1, 0, 0, 0], [1, 0, 0, 0, 1],
                                    [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]]):
            graph.add_node(i, markers_bin=values, markers_bin_raw=[1] * 5)
        result = helpers()['summarize_processed_markers'](graph)
        self.assertEqual(result['n_proliferating'], 3)
        self.assertEqual(result['n_ta'], 1)
        self.assertEqual(result['fraction_ta'], 0.25)

    def test_organoid_level_marker_status_truth_table(self):
        table = pd.DataFrame({'n_lysozyme_positive': [0, 1, 0, 2],
                              'n_serotonin_positive': [0, 0, 1, 3]})
        result = helpers()['add_marker_status_columns'](table)
        self.assertEqual(result.has_lysozyme_only.tolist(), [False, True, False, False])
        self.assertEqual(result.has_serotonin_only.tolist(), [False, False, True, False])
        self.assertEqual(result.has_lysozyme_or_serotonin.tolist(), [False, True, True, True])
        self.assertNotIn('has_lysozyme_only', table)

    def test_grouped_boxplots_keep_zeros_and_report_empty_groups(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = ''.join(next(c for c in notebook['cells'] if c['id'] == 'shape-categories')['source'])
        functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        shapes = ['Non-spherical', 'Near-spherical (lower)', 'Near-spherical (upper)', 'Spherical']
        namespace = dict(np=np, SHAPE_CATEGORY_ORDER=shapes)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<plot helpers>', 'exec'), namespace)
        table = pd.DataFrame({'shape_category': ['Spherical'] * 7,
                              'status': [False] * 6 + [True],
                              'fraction_proliferating': [0., .2, .4, .6, 1., np.nan, .8]})
        fig, ax = plt.subplots()
        rows, handles = namespace['grouped_shape_boxplots'](
            ax, table, 'fraction_proliferating', 'status', [False, True],
            {False: '#999999', True: '#CC3377'}, {False: 'Other', True: 'Positive'}, 'test')
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(row['n_organoids'] for row in rows), 6)
        row = next(row for row in rows if row['shape_category'] == 'Spherical' and not row['color_group'])
        self.assertEqual(row['n_organoids'], 5)
        self.assertAlmostEqual(row['median'], .4)
        self.assertAlmostEqual(row['q1'], .2)
        self.assertAlmostEqual(row['q3'], .6)
        self.assertEqual(row['whisker_low'], 0.)
        self.assertEqual(len(ax.patches), 2)
        self.assertEqual(len(handles), 2)
        plt.close(fig)

    def test_minimum_cells_and_distribution_zero_handling(self):
        notebook = json.loads(NOTEBOOK.read_text())
        cells = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            graph_dir = root / 'day3'
            graph_dir.mkdir()
            graphs = {}
            for size in [19, 20, 21]:
                path = graph_dir / f'organoid_{size}.gpickle'
                path.touch()
                graph = graph_with_processed_markers()
                for node in range(100, 100 + size - 4):
                    graph.add_node(node, markers_bin=[0, 0, 0, 0])
                if size == 21:
                    for _, attrs in graph.nodes(data=True):
                        attrs['markers_bin'] = [0, 0, 0, 0]
                graphs[path] = graph
            namespace = helpers()
            namespace.update(pd=pd, plt=plt, GRAPH_DIR=root, SELECTED_TIMEPOINTS=['day3'],
                             BLACKLIST_PATH=None, MIN_CELLS=20, DATASET='test', DATASET_ROOT=root,
                             TIMEPOINT_ALIASES={}, MESH_COORD_TO_UM=1.,
                             load_cell_graph=lambda path: graphs[path],
                             resolve_mesh_path=lambda graph, root: root / 'mesh.vtp',
                             OrganoidMesh=lambda path: None,
                             measure_mesh_geometry=lambda *args: dict(
                                 sphericity=0.98, surface_area_um2=10., volume_um3=1.),
                             display=lambda *args: None,
                             NEAR_SPHERICAL_MIN=0.94, NEAR_SPHERICAL_SPLIT=0.965,
                             SPHERICAL_MIN=0.99,
                             SHAPE_CATEGORY_ORDER=['Non-spherical', 'Near-spherical (lower)',
                                                   'Near-spherical (upper)', 'Spherical'])
            namespace['SHAPE_COLORS'] = dict(zip(namespace['SHAPE_CATEGORY_ORDER'],
                                                 ['#CC6677', '#DDCC77', '#44AA99', '#4477AA']))
            exec(cells['load-data'], namespace)
            self.assertEqual(namespace['organoid_table'].n_cells.tolist(), [20, 21])
            self.assertEqual(namespace['audit_table'].reason.tolist(), ['fewer than 20 cells'])
            exec(cells['shape-categories'], namespace)
            exec(cells['timepoint-plots'], namespace)
            self.assertEqual(len(namespace['geometry_table']), 2)
            axes = namespace['fig_timepoint_boxplots'].axes
            self.assertEqual(len(axes), 4)
            self.assertTrue(all(len(ax.get_xticks()) == 4 for ax in axes))
            exec(cells['marker-status-and-boxplots'], namespace)
            stats = namespace['boxplot_table']
            self.assertNotIn('n_cells', stats.value_column.unique())
            self.assertTrue((stats.groupby('panel').n_organoids.sum() == 2).all())
            violin_stats = namespace['secretory_violin_table']
            self.assertTrue((violin_stats.groupby('metric').n_organoids.sum() == 2).all())
            self.assertTrue((violin_stats.loc[(~violin_stats.has_secretory_cell) &
                                            (violin_stats.n_organoids > 0), 'median'] == 0).all())
            for name in ['fig_timepoint_boxplots', 'fig_secretory_violins']:
                self.assertIn('Global (whole-organoid)', namespace[name]._suptitle.get_text())
                self.assertTrue(all('Global (whole organoid)' in ax.get_title()
                                    for ax in namespace[name].axes))
            self.assertNotIn('total-cell-count-plots', cells)
            plt.close('all')

    def test_secretory_timepoint_violins_do_not_pool_timepoints(self):
        notebook = json.loads(NOTEBOOK.read_text())
        cells = {c['id']: ''.join(c['source']) for c in notebook['cells']}
        namespace = helpers()
        namespace.update(pd=pd, plt=plt,
                         SHAPE_CATEGORY_ORDER=['Non-spherical', 'Near-spherical (lower)',
                                               'Near-spherical (upper)', 'Spherical'],
                         CELL_POPULATIONS=[('proliferating', 'Proliferating cells', 'Fraction of proliferating cells'),
                                           ('ta', 'TA cells', 'Fraction of TA cells')],
                         SECRETORY_COLORS={False: '#999999', True: '#CC3377'},
                         SECRETORY_LABELS={False: 'All other organoids', True: 'With secretory cell'})
        for name in ['shape-categories', 'secretory-fractions-by-timepoint']:
            functions = [n for n in ast.parse(cells[name]).body if isinstance(n, ast.FunctionDef)]
            exec(compile(ast.Module(body=functions, type_ignores=[]), name, 'exec'), namespace)
        table = pd.DataFrame(dict(
            timepoint=['day2', 'day3', 'day3', 'day3p5', 'day4'],
            shape_category=['Spherical'] * 5,
            has_lysozyme_or_serotonin=[False, False, True, False, True],
            fraction_proliferating=[1., .2, .4, .6, 0.], fraction_ta=[.9, .1, .3, .5, 0.]))
        fig, stats = namespace['plot_secretory_fractions_by_timepoint'](
            table, ['day3', 'day3p5', 'day4'])
        self.assertEqual(len(fig.axes), 6)
        self.assertEqual(set(stats.timepoint), {'day3', 'day3p5', 'day4'})
        self.assertEqual(stats.groupby('timepoint').n_organoids.sum().to_dict(),
                         {'day3': 4, 'day3p5': 2, 'day4': 2})
        selected = stats[(stats.shape_category == 'Spherical') &
                         (stats.metric == 'fraction_proliferating') & (stats.n_organoids > 0)]
        self.assertEqual(selected['median'].tolist(), [.2, .4, .6, 0.])
        self.assertTrue(stats.loc[stats.n_organoids == 0, 'median'].isna().all())
        self.assertEqual(set(stats.loc[stats.has_secretory_cell, 'side']), {'right'})
        self.assertEqual(set(stats.loc[~stats.has_secretory_cell, 'side']), {'left'})
        self.assertTrue(fig._suptitle.get_text())
        self.assertTrue(fig.axes[2].get_title().startswith('Day 3.5:'))
        self.assertIn('Global (whole-organoid)', fig._suptitle.get_text())
        self.assertTrue(all('global (whole organoid)' in ax.get_title() for ax in fig.axes))
        self.assertTrue(all(ax.get_ylim() == (-.02, 1.02) for ax in fig.axes))
        plt.close(fig)

    def test_split_violin_halves_quantiles_and_degenerate_groups(self):
        draw = helpers()['draw_half_violin']
        fig, ax = plt.subplots()
        values = [0., .1, .2, .3, .9, 1., np.nan]
        for side in ['left', 'right']:
            stats = draw(ax, values, 2., side, '#4477AA')
            self.assertEqual(stats['n_finite_values'], 6)
            self.assertAlmostEqual(stats['median'], .25)
            self.assertAlmostEqual(stats['q1'], .125)
            self.assertAlmostEqual(stats['q3'], .75)
            vertices = ax.collections[-1].get_paths()[0].vertices
            self.assertTrue((vertices[:, 0] <= 2.).all() if side == 'left'
                            else (vertices[:, 0] >= 2.).all())
            self.assertGreaterEqual(vertices[:, 1].min(), 0.)
            self.assertLessEqual(vertices[:, 1].max(), 1.)
        before = len(ax.collections)
        self.assertEqual(draw(ax, [0.], 0, 'left', '#4477AA')['median'], 0.)
        self.assertEqual(draw(ax, [1., 1.], 0, 'right', '#4477AA')['median'], 1.)
        self.assertEqual(draw(ax, [np.nan], 1, 'left', '#4477AA')['n_finite_values'], 0)
        self.assertEqual(len(ax.collections), before)
        plt.close(fig)

    def test_deviation_scatter_coordinates_selection_and_marker_shapes(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = ''.join(next(c for c in notebook['cells'] if c['id'] == 'sphericity-deviation-scatter')['source'])
        functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        namespace = dict(np=np, plt=plt, SPHERICAL_MIN=.99,
                         NEAR_SPHERICAL_SPLIT=.965, NEAR_SPHERICAL_MIN=.94)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<scatter>', 'exec'), namespace)
        table = pd.DataFrame(dict(
            timepoint=['day2', 'day3', 'day3p5', 'day4', 'day4', 'day3', 'day4'],
            geometry_valid=[True, True, True, True, False, True, True],
            sphericity=[.8, 1., .95, .9, .7, np.nan, .8],
            fraction_proliferating=[.2, 0., .4, .8, .9, .5, np.inf],
            has_lysozyme_or_serotonin=[False, False, True, False, True, False, False]))
        fig, selected = namespace['plot_sphericity_deviation_scatter'](
            table, ['day3', 'day3p5', 'day4'],
            {'day3': '#009900', 'day3p5': '#cc0000', 'day4': '#770099'},
            {False: '#999999', True: '#CC3377'})
        self.assertEqual(selected.index.tolist(), [1, 2, 3])
        np.testing.assert_allclose(selected.sphericity_deviation, [0., .05, .1])
        self.assertEqual(selected.scatter_marker.tolist(), ['o', '^', 'o'])
        self.assertEqual(selected.scatter_color.tolist(), ['#009900', '#cc0000', '#770099'])
        self.assertEqual(selected.single_timepoint_color.tolist(), ['#999999', '#cc3377', '#999999'])
        ax = fig.axes[0]
        offsets = np.concatenate([collection.get_offsets() for collection in ax.collections])
        np.testing.assert_allclose(offsets, [[.4, .05], [.8, .1]])
        self.assertEqual(selected.included_in_log_scatter.tolist(), [False, True, True])
        self.assertEqual(ax.get_yscale(), 'log')
        np.testing.assert_allclose([line.get_ydata()[0] for line in ax.lines], [.01, .035, .06])
        self.assertTrue(all(line.get_linestyle() == '--' for line in ax.lines))
        self.assertEqual([text.get_text() for text in ax.texts],
                         ['Sphericity = 0.99', 'Sphericity = 0.965', 'Sphericity = 0.94'])
        self.assertNotEqual(ax.collections[0].get_paths()[0].vertices.shape,
                            ax.collections[3].get_paths()[0].vertices.shape)
        self.assertIn('Global (whole organoid)', ax.get_title())
        self.assertIn('Fraction of proliferating cells', fig._supxlabel.get_text())
        self.assertIn('1 - sphericity', fig._supylabel.get_text())
        self.assertEqual(len(fig.axes), 6)
        expected_titles = ['All selected organoids', 'No secretory cell', 'With secretory cell',
                           'Day 3', 'Day 3.5', 'Day 4']
        from matplotlib.colors import to_rgba
        for panel_index, panel in enumerate(fig.axes):
            for collection_index, collection in enumerate(panel.collections):
                color = (['#999999', '#CC3377'][collection_index % 2] if panel_index >= 3
                         else ['#009900', '#cc0000', '#770099'][collection_index // 2])
                np.testing.assert_allclose(collection.get_facecolors()[0], to_rgba(color, .65))
        expected_offsets = [offsets, [[.8, .1]], [[.4, .05]], [], [[.4, .05]], [[.8, .1]]]
        for panel, title, expected in zip(fig.axes, expected_titles, expected_offsets):
            self.assertTrue(panel.get_title().startswith(title))
            self.assertIn('Global (whole organoid)', panel.get_title())
            actual = np.concatenate([collection.get_offsets() for collection in panel.collections])
            np.testing.assert_allclose(actual, np.asarray(expected).reshape(-1, 2))
            self.assertEqual(panel.get_yscale(), 'log')
            self.assertEqual(panel.get_xlim(), ax.get_xlim())
            self.assertEqual(panel.get_ylim(), ax.get_ylim())
            self.assertTrue(panel.spines['top'].get_visible())
            self.assertTrue(panel.spines['right'].get_visible())
            self.assertTrue(panel.xaxis.get_major_ticks()[0].tick2line.get_visible())
            self.assertTrue(panel.yaxis.get_major_ticks()[0].tick2line.get_visible())
            np.testing.assert_allclose([line.get_ydata()[0] for line in panel.lines], [.01, .035, .06])
        self.assertNotIn('sphericity_deviation', table)
        plt.close(fig)

    def test_covariance_ellipses_in_log_display_coordinates(self):
        notebook = json.loads(NOTEBOOK.read_text())
        source = ''.join(next(c for c in notebook['cells'] if c['id'] == 'deviation-distribution-ellipses')['source'])
        functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        namespace = dict(np=np, pd=pd, plt=plt, SPHERICAL_MIN=.98,
                         NEAR_SPHERICAL_SPLIT=.95, NEAR_SPHERICAL_MIN=.92)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<ellipses>', 'exec'), namespace)
        ellipse = namespace['covariance_ellipse_coordinates']
        x, y = np.array([.1, .3, .7, .9]), np.array([.01, .1, .03, .4])
        contour, stats = ellipse(x, y, 1.)
        points = np.column_stack([x, np.log10(y)])
        centered = np.column_stack([contour[:, 0], np.log10(contour[:, 1])]) - points.mean(axis=0)
        covariance = np.cov(points, rowvar=False, ddof=1)
        distances = np.einsum('ij,jk,ik->i', centered, np.linalg.inv(covariance), centered)
        np.testing.assert_allclose(distances, 1.)
        twice, _ = ellipse(x, y, 2.)
        twice_centered = np.column_stack([twice[:, 0], np.log10(twice[:, 1])]) - points.mean(axis=0)
        np.testing.assert_allclose(twice_centered, 2 * centered, atol=1e-14)
        self.assertEqual(stats['n_organoids'], 4)
        self.assertEqual(stats['ellipse_status'], 'drawn')
        self.assertAlmostEqual(stats['cov_xy'], covariance[0, 1])
        self.assertAlmostEqual(stats['mean_log10_deviation'], points[:, 1].mean())
        self.assertIsNone(ellipse([.2], [.1], 2.)[0])
        self.assertEqual(ellipse([.2]*3, [.1]*3, 2.)[1]['ellipse_status'], 'degenerate covariance')
        for radius in [0, -1, np.nan]:
            with self.assertRaises(ValueError):
                ellipse(x, y, radius)
        table = pd.DataFrame(dict(timepoint=['day3']*4 + ['day4']*4 + ['day2'],
                                  fraction_proliferating=[.1, .3, .7, .9, 0., .3, .7, .9, .2],
                                  sphericity_deviation=[.01, .1, .03, .4]*2 + [.05],
                                  has_lysozyme_or_serotonin=[False, True]*4 + [False]))
        fig, positive, summary = namespace['plot_deviation_distribution_ellipses'](
            table, ['day3', 'day3p5', 'day4'],
            {'day3': '#009900', 'day3p5': '#cc0000', 'day4': '#770099'},
            {False: '#999999', True: '#CC3377'}, 1.)
        self.assertEqual(len(positive), 7)
        self.assertEqual(len(fig.axes), 4)
        self.assertTrue((positive.fraction_proliferating > 0).all())
        self.assertEqual(summary.groupby('panel').n_organoids.sum().to_dict(), {
            'timepoint': 7, 'secretory_status': 7,
            'timepoint_no_secretory': 3, 'timepoint_with_secretory': 4})
        self.assertTrue((summary.n_std == 1).all())
        for ax, count in zip(fig.axes, [7, 7, 3, 4]):
            self.assertEqual(sum(len(c.get_offsets()) for c in ax.collections), count)
            self.assertEqual(ax.get_yscale(), 'log')
            self.assertTrue(ax.spines['top'].get_visible())
            self.assertTrue(ax.spines['right'].get_visible())
            self.assertTrue(ax.xaxis.get_major_ticks()[0].tick2line.get_visible())
            self.assertTrue(ax.yaxis.get_major_ticks()[0].tick2line.get_visible())
            self.assertIn('Global (whole organoid)', ax.get_title())
            self.assertEqual(fig.axes[0].get_ylim(), ax.get_ylim())
            self.assertEqual(fig.axes[0].get_xlim(), ax.get_xlim())
        for ax, present in zip(fig.axes[2:], [False, True]):
            offsets = np.concatenate([c.get_offsets() for c in ax.collections])
            subset = positive.loc[positive.has_lysozyme_or_serotonin == present]
            np.testing.assert_allclose(offsets, subset[['fraction_proliferating', 'sphericity_deviation']])
            panel = 'timepoint_with_secretory' if present else 'timepoint_no_secretory'
            for _, row in summary.loc[summary.panel == panel].iterrows():
                group = subset.loc[subset.timepoint == row['group']]
                self.assertEqual(row.n_organoids, len(group))
                if len(group):
                    self.assertAlmostEqual(row.mean_fraction, group.fraction_proliferating.mean())
                self.assertEqual(row.secretory_filter, present)
        plt.close(fig)

    def test_configuration_and_only_requested_standalone_scatter(self):
        notebook = json.loads(NOTEBOOK.read_text())
        cells = {c['id']: ''.join(c['source']) for c in notebook['cells']}
        tree = ast.parse(cells['configuration'])
        config = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                try:
                    config[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
        self.assertIn('day4', config['SELECTED_TIMEPOINTS'])
        near, split, spherical = [config[k] for k in ['NEAR_SPHERICAL_MIN',
                                                     'NEAR_SPHERICAL_SPLIT', 'SPHERICAL_MIN']]
        self.assertTrue(0 < near < split < spherical < 1)
        self.assertEqual(config['DEVIATION_SCATTER_TIMEPOINTS'], ['day3', 'day3p5', 'day4'])
        self.assertEqual([name for name, source in cells.items() if '.scatter(' in source],
                         ['sphericity-deviation-scatter', 'deviation-distribution-ellipses',
                          'crypt-villus-proliferation-scatter'])

    def test_sphericity_categories_and_inclusive_boundaries(self):
        classify = helpers()['classify_sphericity']
        values = [0.8, 0.93999, 0.94, 0.96499, 0.965, 0.98999, 0.99, 1., np.nan, np.inf]
        self.assertEqual(classify(values, 0.94, 0.965, 0.99).tolist(), [
            'Non-spherical', 'Non-spherical', 'Near-spherical (lower)', 'Near-spherical (lower)',
            'Near-spherical (upper)', 'Near-spherical (upper)',
            'Spherical', 'Spherical', None, None])
        self.assertEqual(classify([0.97], 0.95, 0.96, 0.965).tolist(), ['Spherical'])
        for near, split, spherical in [(0.99, 0.975, 0.96), (0.96, 0.96, 0.99),
                                        (0.96, 0.99, 0.99), (0, 0.975, 0.99), (0.96, 0.975, 1.1)]:
            with self.assertRaises(ValueError):
                classify(values, near, split, spherical)

    def test_union_uses_processed_markers_and_nonconsecutive_nodes(self):
        graph = graph_with_processed_markers()
        result = helpers()['summarize_processed_markers'](graph)
        self.assertEqual(result['n_cells'], 4)
        self.assertEqual(result['n_proliferating'], 3)
        self.assertEqual(result['fraction_proliferating'], 0.75)
        self.assertEqual(result['n_cyclin_a_and_d_positive'], 1)
        self.assertEqual(result['n_lysozyme_positive'], 0)
        self.assertTrue(result['has_lysozyme_or_serotonin'])
        graph.nodes[16]['markers_bin'] = [0, 0, 0, 0]
        self.assertFalse(helpers()['summarize_processed_markers'](graph)['has_lysozyme_or_serotonin'])

    def test_missing_processed_marker_does_not_fall_back_to_raw(self):
        graph = graph_with_processed_markers()
        del graph.nodes[4]['markers_bin']
        with self.assertRaises(KeyError):
            helpers()['summarize_processed_markers'](graph)

    def test_invalid_or_missing_marker_panel(self):
        graph = graph_with_processed_markers()
        graph.nodes[4]['markers_bin'][0] = np.nan
        with self.assertRaises(ValueError):
            helpers()['summarize_processed_markers'](graph)
        graph.graph['marker_names'][0] = 'not Cyclin A'
        with self.assertRaises(KeyError):
            helpers()['summarize_processed_markers'](graph)

    def test_geometry_units_scale_and_translation(self):
        measure = helpers()['measure_mesh_geometry']
        result = measure(tetrahedron(offset=1000.0))
        area = (3 + np.sqrt(3)) / 2
        self.assertAlmostEqual(result['volume_um3'], 1 / 6)
        self.assertAlmostEqual(result['surface_area_um2'], area)
        self.assertAlmostEqual(result['sphericity'], np.cbrt(np.pi) / area)
        scaled = measure(tetrahedron(), coord_to_um=10.)
        self.assertAlmostEqual(scaled['volume_um3'], 1000 / 6)
        self.assertAlmostEqual(scaled['surface_area_um2'], 100 * area)
        self.assertAlmostEqual(scaled['sphericity'], result['sphericity'])

    def test_open_and_misoriented_meshes_are_rejected(self):
        measure = helpers()['measure_mesh_geometry']
        mesh = tetrahedron()
        mesh.f = mesh.f[:-1]
        with self.assertRaisesRegex(ValueError, 'closed/manifold'):
            measure(mesh)
        mesh = tetrahedron()
        mesh.f[0] = mesh.f[0, ::-1]
        with self.assertRaisesRegex(ValueError, 'orientation'):
            measure(mesh)

    def test_relocated_mesh_path(self):
        graph = nx.Graph()
        graph.graph['mesh_path'] = '/old/dataset/fractal_output/example.vtp'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fractal_output' / 'example.vtp'
            path.parent.mkdir()
            path.touch()
            self.assertEqual(helpers()['resolve_mesh_path'](graph, folder), path)

    def test_notebook_syntax_and_timepoint_order(self):
        notebook = json.loads(NOTEBOOK.read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
        self.assertEqual(sorted(['day10', 'day4p5', 'day3'], key=helpers()['timepoint_sort_key']),
                         ['day3', 'day4p5', 'day10'])


if __name__ == '__main__':
    unittest.main()

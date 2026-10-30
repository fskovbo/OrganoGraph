import ast
import json
from pathlib import Path
import tempfile
import unittest

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from organograph.graph.io import load_cell_graph


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/plot_proliferation_symmetry_breaking.ipynb'
OPTIONS = dict(s_star=0.75, interval=(-0.05, 0.05), crypt_max=0.75, villus_min=1.15)


def helpers():
    nb = json.loads(NOTEBOOK.read_text())
    source = ''.join(next(c for c in nb['cells'] if c['id'] == 'crypt-region-helpers')['source'])
    namespace = dict(np=np, pd=pd, Path=Path, DATASET='20250929', load_cell_graph=load_cell_graph)
    exec(source, namespace)
    return namespace


def sample_graph():
    graph = nx.Graph()
    graph.graph.update(label_uid='day3p5_A01_1', marker_names=[
        'Cyclin A', 'Cyclin D', 'LGR5', 'Lysozyme', 'Serotonin'])
    values = [[1, 0, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0],
              [1, 0, 0, 0, 1], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
    # Insertion order is deliberately different from the stored distance-column order.
    for i in reversed(range(len(values))):
        graph.add_node(i, markers_bin=values[i], markers_bin_raw=[1] * 5)
    return graph


def sample_segmentation():
    return dict(crypts_ll=[[0, 1, 2], [4, 5]], n_crypts=2,
                d_crypts_graph=np.array([[0., .75, .8, 1.15, 2., 3., 4., 6.],
                                        [4., 3., 2., 2., .5, .76, 1.15, 2.]]),
                d_discretized=np.array([0., .75, 1., 1.2]),
                circumference_crypts=np.array([[.1, .8, 1., 1.1], [.1, 1., .8, .9]]))


class CryptRegionTests(unittest.TestCase):
    def test_linear_one_sd_crypt_villus_ellipses_include_zeros_and_day4p5(self):
        nb = json.loads(NOTEBOOK.read_text())
        source = ''.join(next(c for c in nb['cells'] if c['id'] == 'crypt-villus-proliferation-scatter')['source'])
        functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        namespace = dict(np=np, pd=pd, plt=plt, REGION_CRYPT_MAX=.75, REGION_VILLUS_MIN=1.15)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<crypt-villus>', 'exec'), namespace)
        timepoints = ['day3p5', 'day4', 'day4p5']
        x, y = np.array([0., .2, .5, 1.]), np.array([.1, .6, .3, .9])
        pairs = pd.DataFrame([dict(timepoint=tp, label_ID=f'{tp}_{i}', crypt_category='bulged',
                                   crypt_has_lgr5=True, crypt_has_secretory=True, included_in_plot=True,
                                   fraction_proliferating_villus=v, fraction_proliferating_crypt=c)
                              for tp in timepoints for i, (v, c) in enumerate(zip(x, y))])
        fig, summary = namespace['plot_crypt_villus_proliferation'](
            pairs, timepoints, ['bulged', 'budded'],
            dict(zip(timepoints, ['#d62728', '#9467bd', '#8c564b'])))
        covariance = np.cov(np.column_stack([x, y]), rowvar=False, ddof=1)
        self.assertEqual(len(fig.axes[0].patches), 3)
        self.assertEqual(summary.n_plotted_crypts.sum(), 12)
        self.assertEqual(set(summary.timepoint), set(timepoints))
        for patch in fig.axes[0].patches:
            angle = np.deg2rad(patch.angle)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            reconstructed = rotation @ np.diag([(patch.width / 2)**2, (patch.height / 2)**2]) @ rotation.T
            np.testing.assert_allclose(reconstructed, covariance)
            np.testing.assert_allclose(patch.center, [x.mean(), y.mean()])
        drawn = summary.loc[summary.ellipse_status == 'drawn']
        self.assertEqual(drawn.n_plotted_crypts.tolist(), [4, 4, 4])
        self.assertTrue((drawn.ellipse_n_std == 1).all())
        np.testing.assert_allclose(drawn.cov_villus_crypt, covariance[0, 1])
        plt.close(fig)

    def test_regional_loading_pools_day4p5_sources_and_retains_provenance(self):
        nb = json.loads(NOTEBOOK.read_text())
        source = ''.join(next(c for c in nb['cells'] if c['id'] == 'crypt-region-loading')['source'])
        table = pd.DataFrame(dict(timepoint=['day4', 'day4p5', 'day4p5-more'],
                                  label_ID=['A', 'B', 'C'], crypt_id=[1, 1, 1],
                                  crypt_category=['bulged', 'weakly budded', 'fully budded']))
        namespace = helpers()
        namespace.update(BLACKLIST_PATH=None, GRAPH_DIR=Path('.'), REGION_SEGMENTATION_DIR=Path('.'),
                         REGION_SOURCE_TIMEPOINTS=['day4', 'day4p5', 'day4p5-more'], MIN_CELLS=20,
                         REGION_S_STAR=.75, REGION_BI_INTERVAL=(-.05, .05), REGION_CRYPT_MAX=.75,
                         REGION_VILLUS_MIN=1.15, TIMEPOINT_ALIASES={'day4p5-more': 'day4p5'},
                         display=lambda *args: None,
                         load_regional_proliferation=lambda *a, **k: (
                             table.copy(), pd.DataFrame(columns=['timepoint', 'reason'])))
        exec(source, namespace)
        result = namespace['regional_crypt_table']
        self.assertEqual(result.timepoint.tolist(), ['day4', 'day4p5', 'day4p5'])
        self.assertEqual(result.source_timepoint.tolist(), ['day4', 'day4p5', 'day4p5-more'])
        self.assertEqual(result.crypt_category.tolist(), ['bulged', 'budded', 'budded'])
        self.assertEqual(len(result), 3)

    def test_crypt_villus_pairing_and_eight_scatter_panels(self):
        nb = json.loads(NOTEBOOK.read_text())
        source = ''.join(next(c for c in nb['cells'] if c['id'] == 'crypt-villus-proliferation-scatter')['source'])
        functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        namespace = dict(np=np, pd=pd, plt=plt, REGION_CRYPT_MAX=.75, REGION_VILLUS_MIN=1.15)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<crypt-villus>', 'exec'), namespace)
        table = pd.DataFrame(helpers()['summarize_regional_proliferation'](
            sample_graph(), sample_segmentation(), **OPTIONS))
        table = helpers()['merge_regional_categories'](table)
        table['dataset'], table['timepoint'], table['label_ID'] = 'test', 'day4', 'same_label'
        earlier = table.assign(timepoint='day3p5')
        no_crypt = table.iloc[[0]].assign(unit_type='organoid', crypt_id=0, crypt_category='No crypts')
        other_timepoint = table.assign(timepoint='day3')
        full = pd.concat([table, earlier, no_crypt, other_timepoint], ignore_index=True)
        pair = namespace['build_crypt_villus_pairs']
        pairs = pair(full.sample(frac=1, random_state=0), ['day3p5', 'day4'], ['bulged', 'budded'])
        self.assertEqual(len(pairs), 4)
        self.assertTrue(pairs.included_in_plot.all())
        np.testing.assert_allclose(pairs.fraction_proliferating_crypt, 1.)
        np.testing.assert_allclose(pairs.loc[pairs.crypt_id == 1, 'fraction_proliferating_villus'], 0.)
        np.testing.assert_allclose(pairs.loc[pairs.crypt_id == 2, 'fraction_proliferating_villus'], .5)
        fig, summary = namespace['plot_crypt_villus_proliferation'](
            pairs, ['day3p5', 'day4'], ['bulged', 'budded'],
            {'day3p5': '#cc0000', 'day4': '#770099'})
        self.assertEqual(len(fig.axes), 8)
        self.assertEqual(summary.n_plotted_crypts.sum(), 4)
        for index, ax in enumerate(fig.axes):
            self.assertIn('Regional fractions', ax.get_title())
            self.assertTrue(ax.spines['top'].get_visible())
            self.assertTrue(ax.spines['right'].get_visible())
            offsets = np.concatenate([c.get_offsets() for c in ax.collections])
            if index == 1:  # Bulged, LGR5+ / secretory-.
                np.testing.assert_allclose(offsets, [[0., 1.], [0., 1.]])
            elif index == 6:  # Budded, LGR5- / secretory+.
                np.testing.assert_allclose(offsets, [[.5, 1.], [.5, 1.]])
            else:
                self.assertEqual(len(offsets), 0)
        plt.close(fig)
        invalid = table.loc[~((table.crypt_id == 2) & (table.region == 'Villus'))].copy()
        invalid.loc[(invalid.crypt_id == 1) & (invalid.region == 'Villus'),
                    ['n_cells', 'n_proliferating', 'fraction_proliferating']] = [0, 0, np.nan]
        invalid_pairs = pair(invalid, ['day4'], ['bulged', 'budded']).set_index('crypt_id')
        self.assertEqual(invalid_pairs.loc[1, 'exclusion_reason'], 'empty region')
        self.assertEqual(invalid_pairs.loc[2, 'exclusion_reason'], 'missing region')
        self.assertFalse(invalid_pairs.included_in_plot.any())
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            pair(pd.concat([table, table.iloc[[0]]]), ['day4'], ['bulged', 'budded'])
        inconsistent = table.copy()
        inconsistent.loc[inconsistent.index[0], 'crypt_has_lgr5'] = False
        with self.assertRaisesRegex(ValueError, 'Inconsistent crypt metadata'):
            pair(inconsistent, ['day4'], ['bulged', 'budded'])

    def test_crypt_status_stack_counts_unique_crypts_and_excludes_crypt_free_organoids(self):
        nb = json.loads(NOTEBOOK.read_text())
        source = ''.join(next(c for c in nb['cells'] if c['id'] == 'crypt-status-stacked-bars')['source'])
        functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        statuses = ['LGR5+ / secretory+', 'LGR5+ / secretory-',
                    'LGR5- / secretory+', 'LGR5- / secretory-']
        namespace = dict(np=np, pd=pd, CRYPT_STATUS_ORDER=statuses)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<stacked bars>', 'exec'), namespace)
        rows = []
        for i, (lgr5, secretory) in enumerate([(True, True), (True, False), (False, True), (False, False)]):
            for region in ['Crypt', 'Neck', 'Villus']:
                rows.append(dict(timepoint='day4', label_ID='example', crypt_id=i + 1,
                                 unit_type='crypt',
                                 crypt_category=['bulged', 'weakly budded', 'fully budded', 'bulged'][i],
                                 region=region,
                                 crypt_has_lgr5=lgr5, crypt_has_secretory=secretory))
        rows.append(dict(timepoint='day4', label_ID='no_crypt', crypt_id=0,
                         unit_type='organoid', crypt_category='No crypts',
                         crypt_has_lgr5=None, crypt_has_secretory=None))
        table, categories = namespace['build_crypt_status_fractions'](
            helpers()['merge_regional_categories'](pd.DataFrame(rows)),
            ['day3p5', 'day4'], ['bulged', 'budded'])
        all_crypts = table[(table.timepoint == 'day4') & (table.crypt_category == 'All crypts')]
        self.assertEqual(all_crypts.n_crypts.tolist(), [1, 1, 1, 1])
        self.assertEqual(all_crypts.total_crypts.tolist(), [4] * 4)
        self.assertEqual(all_crypts.fraction.tolist(), [.25] * 4)
        self.assertTrue(table.loc[table.total_crypts == 0, 'fraction'].isna().all())
        budded = table[(table.timepoint == 'day4') & (table.crypt_category == 'budded')]
        self.assertEqual(budded.total_crypts.tolist(), [2] * 4)
        self.assertEqual(budded.n_crypts.tolist(), [0, 1, 1, 0])
        self.assertEqual(categories, ['All crypts', 'bulged', 'budded'])
        ids = [c['id'] for c in nb['cells']]
        for plot in ['timepoint-plots', 'marker-status-and-boxplots',
                     'crypt-region-plotting']:
            self.assertLess(ids.index('crypt-status-stacked-bars'), ids.index(plot))

    def test_shared_count_axis_includes_whole_organoid_extremes(self):
        nb = json.loads(NOTEBOOK.read_text())
        source = ''.join(next(c for c in nb['cells'] if c['id'] == 'crypt-region-plotting')['source'])
        functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        namespace = dict(np=np, pd=pd, plt=plt, REGION_CATEGORIES=['bulged', 'budded'])
        marker_source = ''.join(next(c for c in nb['cells'] if c['id'] == 'marker-helpers')['source'])
        exec(marker_source, namespace)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<plots>', 'exec'), namespace)
        rows = helpers()['summarize_regional_proliferation'](sample_graph(), sample_segmentation(), **OPTIONS)
        for row in rows:
            row.update(timepoint='day4', label_ID='with_crypts')
        rows.append(dict(timepoint='day4', label_ID='without_crypts', crypt_id=0,
                         crypt_category='No crypts', region='Whole organoid', n_cells=2000,
                         n_proliferating=1000, fraction_proliferating=.5, n_ta=700, fraction_ta=.35))
        table = helpers()['merge_regional_categories'](pd.DataFrame(rows))
        for split in [None, 'crypt_has_lgr5', 'crypt_has_secretory']:
            fig, stats = namespace['plot_regional_proliferation'](table, 'day4', split)
            self.assertEqual(len(fig.axes), 6)
            for ax in fig.axes[3:]:
                self.assertGreater(ax.get_ylim()[1], 1000)
            self.assertEqual(set(stats.loc[stats.population == 'proliferating', 'side']), {'left'})
            self.assertEqual(set(stats.loc[stats.population == 'ta', 'side']), {'right'})
            self.assertTrue((stats.groupby('metric').n_units.sum() == len(table)).all())
            self.assertTrue(fig._suptitle.get_text())
            self.assertIn('No-crypt reference: global (whole organoid)', fig._suptitle.get_text())
            for i, ax in enumerate(fig.axes):
                scope = 'Global (whole organoid)' if i % 3 == 2 else 'Regional'
                self.assertIn(scope, ax.get_title())
            self.assertEqual(len(fig.axes[0].get_xticks()), 6 if split else 3)
            plt.close(fig)

    def test_category_merge_preserves_originals_and_unknowns(self):
        categories = ['bulged', 'weakly budded', 'fully budded', 'No crypts', 'unknown']
        table = pd.DataFrame({'crypt_category': categories})
        merge = helpers()['merge_regional_categories']
        merged = merge(table)
        self.assertEqual(table.crypt_category.tolist(), categories)
        self.assertEqual(merged.crypt_category_original.tolist(), categories)
        self.assertEqual(merged.crypt_category.tolist(), ['bulged', 'budded', 'budded', 'No crypts', 'unknown'])
        pd.testing.assert_frame_equal(merged, merge(merged))

    def test_regions_partition_and_exact_boundaries(self):
        seg = sample_segmentation()
        owners, s, regions = helpers()['assign_cells_to_crypt_regions'](seg['d_crypts_graph'], .75, 1.15)
        self.assertEqual(owners.tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(regions.tolist(), ['Crypt', 'Crypt', 'Neck', 'Villus',
                                          'Crypt', 'Neck', 'Villus', 'Villus'])

    def test_processed_markers_core_status_and_counts(self):
        table = pd.DataFrame(helpers()['summarize_regional_proliferation'](
            sample_graph(), sample_segmentation(), **OPTIONS))
        self.assertEqual(table.n_cells.sum(), 8)
        self.assertEqual(table.n_proliferating.sum(), 5)
        self.assertEqual(table.n_ta.sum(), 3)
        self.assertTrue((table.n_ta <= table.n_proliferating).all())
        self.assertEqual(table.n_ta.tolist(), [1, 1, 0, 0, 0, 1])
        self.assertEqual(table.n_cells.tolist(), [2, 1, 1, 1, 1, 2])
        self.assertEqual(table.n_proliferating.tolist(), [2, 1, 0, 1, 0, 1])
        self.assertTrue(table.loc[table.crypt_id == 1, 'crypt_has_lgr5'].all())
        self.assertFalse(table.loc[table.crypt_id == 1, 'crypt_has_secretory'].any())
        self.assertFalse(table.loc[table.crypt_id == 2, 'crypt_has_lgr5'].any())
        self.assertTrue(table.loc[table.crypt_id == 2, 'crypt_has_secretory'].all())
        self.assertEqual(table.crypt_category.tolist(), ['bulged'] * 3 + ['fully budded'] * 3)
        ids = [int(i) for cell_ids in table.cell_node_ids for i in cell_ids.split(';') if i]
        self.assertEqual(sorted(ids), list(range(8)))

    def test_empty_regions_are_nan_fraction_and_zero_count(self):
        seg = sample_segmentation()
        seg['d_crypts_graph'] = np.full((2, 8), .5)
        table = pd.DataFrame(helpers()['summarize_regional_proliferation'](sample_graph(), seg, **OPTIONS))
        empty = table[table.n_cells == 0]
        self.assertTrue(empty.fraction_proliferating.isna().all())
        self.assertTrue(empty.fraction_ta.isna().all())
        self.assertTrue((empty.n_proliferating == 0).all())
        self.assertEqual(table.n_cells.sum(), 8)

    def test_no_crypt_is_one_unsplit_whole_organoid(self):
        rows = helpers()['summarize_regional_proliferation'](
            sample_graph(), dict(crypts_ll=[], n_crypts=0), **OPTIONS)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['region'], 'Whole organoid')
        self.assertEqual(rows[0]['crypt_category'], 'No crypts')
        self.assertIsNone(rows[0]['crypt_has_lgr5'])
        self.assertEqual(rows[0]['fraction_proliferating'], 5 / 8)

    def test_interpolated_bi_uses_only_maximum_inside_boundary(self):
        fn = helpers()['regional_budding_index']
        value, category = fn([0., .5, 1., 1.2], [.1, 1., .8, 100.], .75, (-.05, .05))
        self.assertAlmostEqual(value, .1)
        self.assertEqual(category, 'fully budded')
        _, category = fn([0., .75, 1.], [.1, .97, 1.], .75, (-.05, .05))
        self.assertEqual(category, 'weakly budded')
        # Both interval endpoints are included in weakly budded.
        for interval in [(value, value + 1.), (value - 1., value)]:
            self.assertEqual(fn([0., .5, 1., 1.2], [.1, 1., .8, 100.], .75, interval)[1],
                             'weakly budded')
        self.assertEqual(fn([0., .5], [.1, 1.], .75, (-.05, .05))[1], 'unknown')

    def test_invalid_distances_alignment_and_missing_processed_markers(self):
        fn = helpers()['summarize_regional_proliferation']
        seg = sample_segmentation()
        seg['d_crypts_graph'][:, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            fn(sample_graph(), seg, **OPTIONS)
        seg['d_crypts_graph'] = np.zeros((2, 7))
        with self.assertRaisesRegex(ValueError, 'shape'):
            fn(sample_graph(), seg, **OPTIONS)
        graph = sample_graph()
        del graph.nodes[0]['markers_bin']
        with self.assertRaises(KeyError):
            fn(graph, sample_segmentation(), **OPTIONS)

    def test_missing_segmentation_not_counted_as_no_crypts(self):
        namespace = helpers()
        graph = sample_graph()
        namespace['load_cell_graph'] = lambda path: graph
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'day3p5').mkdir()
            (root / 'day3p5' / 'day3p5_A01_1.gpickle').touch()
            with self.assertRaisesRegex(ValueError, 'Missing graph segmentation'):
                namespace['load_regional_proliferation'](
                    root, root / 'segmentations', ['day3p5'], set(), 1, **OPTIONS)


if __name__ == '__main__':
    unittest.main()

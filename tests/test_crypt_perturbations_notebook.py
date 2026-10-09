import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/plot_crypt_perturbations.ipynb'


def helpers():
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = dict(Path=Path, re=re, np=np, pd=pd)
    cell = next(c for c in notebook['cells'] if c['id'] == 'loading-functions')
    exec(''.join(cell['source']), namespace)
    return namespace


def save_segmentation(root, label, lengths, n_crypts=None, **kwargs):
    n = len(lengths) if n_crypts is None else n_crypts
    patches = np.empty(n, dtype=object)
    for i in range(n):
        patches[i] = [1, 2, 3]
    np.savez(root / f'{label}.npz', label_uid=label, timepoint='day3p5',
             crypts_ll=patches, L_crypts=lengths, **kwargs)


class CryptPerturbationNotebookTests(unittest.TestCase):
    def test_zero_counts_pooling_units_and_invalid_lengths(self):
        functions = helpers()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_segmentation(root, 'day3p5_D02_1', [])
            save_segmentation(root, 'day3p5_D03_2', [2., np.nan, -1.])
            save_segmentation(root, 'day3p5_C07_3', [4.])
            organoids, crypts, audit = functions['load_segmentations'](
                root, {'D02': 'control', 'D03': 'control', 'C07': 'treated'})
            summary = functions['summarize_conditions'](
                organoids, crypts, ['control', 'treated', 'absent']).set_index('condition')
        self.assertEqual(len(audit), 0)
        self.assertEqual(summary.loc['control', 'mean_n_crypts'], 1.5)
        self.assertAlmostEqual(summary.loc['control', 'sd_n_crypts'], np.std([0, 3], ddof=1))
        self.assertEqual(summary.loc['control', 'n_zero_crypt_organoids'], 1)
        self.assertEqual(summary.loc['control', 'n_wells'], 2)
        self.assertEqual(summary.loc['control', 'mean_length_um'], 20.)
        self.assertEqual(summary.loc['control', 'n_valid_lengths'], 1)
        self.assertEqual(summary.loc['absent', 'n_organoids'], 0)

    def test_all_zero_crypts_and_unmapped_wells(self):
        functions = helpers()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_segmentation(root, 'day3p5_D02_1', [])
            save_segmentation(root, 'day3p5_A01_2', [1.])
            organoids, crypts, audit = functions['load_segmentations'](root, {'D02': 'control'})
            summary = functions['summarize_conditions'](organoids, crypts, ['control'])
        self.assertEqual(len(crypts), 0)
        self.assertEqual(len(audit), 1)
        self.assertEqual(summary.mean_n_crypts.iloc[0], 0)

    def test_length_alignment_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_segmentation(root, 'day3p5_D02_1', [1.], n_crypts=2)
            with self.assertRaisesRegex(ValueError, 'count mismatch'):
                helpers()['load_segmentations'](root, {'D02': 'control'})

    def test_timepoint_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_segmentation(root, 'day3p5_D02_1', [])
            with self.assertRaisesRegex(ValueError, 'No organoids match'):
                helpers()['load_segmentations'](root, {'D02': 'control'}, ['day4p5'])

    def test_well_parsing_and_notebook_syntax(self):
        parse = helpers()['well_from_label']
        self.assertEqual(parse('day3p5_C07_104'), 'C07')
        with self.assertRaises(ValueError):
            parse('day3p5_no_well')
        notebook = json.loads(NOTEBOOK.read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')


if __name__ == '__main__':
    unittest.main()

import csv
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import networkx as nx

from organograph.skeleton.cell_counts import CellCountLookup, update_export_cell_counts


class CellCountTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def graph(self, dataset, label, count):
        path = self.root / dataset / "graphs_preprocessed" / "day4" / f"{label}.gpickle"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            # All nodes isolated: counting edges or populated markers is wrong.
            pickle.dump(nx.empty_graph(count), handle)
        return path

    def test_count_is_dataset_specific_and_supports_remapped_index_labels(self):
        first = self.graph("a", "renamed", 17)
        self.graph("b", "day4_B03_4", 23)
        with (first.parent / "index.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["label_uid", "parsed_label_uid", "graph_path", "N_cells"])
            writer.writerow(["renamed", "day4_B03_4", "/old/location/renamed.gpickle", 999])
        lookup = CellCountLookup(self.root)
        self.assertEqual(lookup.count("a", "day4", "day4_B03_4"), 17)
        self.assertEqual(lookup.count("b", "day4", "day4_B03_4"), 23)
        with self.assertRaises(FileNotFoundError):
            lookup.count("c", "day4", "day4_B03_4")

    def test_backfill_preserves_geometry_quality_and_manifest_rows(self):
        self.graph("a", "day4_B03_4", 17)
        export = self.root / "export"
        directory = export / "a" / "day4" / "day4_B03_4"
        directory.mkdir(parents=True)
        sample = {"dataset": "a", "timepoint": "day4", "label_uid": "day4_B03_4"}
        shape = {"sample": sample, "skeleton": {"nodes": [[1, 2, 3]]}, "primitives": [7]}
        quality = {"sample": sample, "crypt_primitives": [{"score": None}]}
        for filename, data in [("shape.json", shape), ("quality.json", quality)]:
            (directory / filename).write_text(json.dumps(data))
        manifest = export / "manifest.csv"
        manifest.write_text(
            "dataset,timepoint,label_uid,json_path\n"
            "a,day4,day4_B03_4,/old/export/shape.json\n"
            "a,day4,missing,/old/missing/shape.json\n"
        )
        lookup = CellCountLookup(self.root)
        before = {p: p.read_bytes() for p in export.rglob("*") if p.is_file()}
        report = update_export_cell_counts(export, lookup, dry_run=True)
        self.assertEqual(report["updated"], 1)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        report = update_export_cell_counts(export, lookup)
        self.assertEqual(report["updated"], 1)
        for filename, expected in [("shape.json", shape), ("quality.json", quality)]:
            actual = json.loads((directory / filename).read_text())
            self.assertEqual(actual["sample"].pop("cell_count"), 17)
            self.assertEqual(actual, expected)
        with manifest.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["cell_count"], "17")
        self.assertEqual(rows[0]["json_path"], "/old/export/shape.json")
        self.assertEqual(rows[1]["cell_count"], "")
        self.assertEqual(update_export_cell_counts(export, lookup)["unchanged"], 1)
        report = update_export_cell_counts(export, CellCountLookup(self.root / "missing"))
        self.assertEqual(report["missing_graph"], 1)
        self.assertEqual(json.loads((directory / "shape.json").read_text())["sample"]["cell_count"], 17)


if __name__ == "__main__":
    unittest.main()

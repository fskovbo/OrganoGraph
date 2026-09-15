"""Cell-graph counts for new and existing reconstructive shape exports."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import tempfile

from organograph.graph.io import load_cell_graph


class CellCountLookup:
    """Resolve preprocessed graphs by dataset, timepoint and either label alias.

    As in run_map_segmentation_to_graph.py, direct graph filenames and the
    label_uid/parsed_label_uid columns of each timepoint's index.csv are used.
    Counts come from the graph itself, including isolated cell nodes.
    """

    def __init__(self, data_root, graphs_subdir="graphs_preprocessed"):
        self.data_root = Path(data_root)
        self.graphs_subdir = graphs_subdir
        self._indices = {}
        self._counts = {}

    def graph_path(self, dataset, timepoint, label_uid) -> Path:
        directory = self.data_root / str(dataset) / self.graphs_subdir / str(timepoint)
        direct = directory / f"{label_uid}.gpickle"
        if direct.is_file():
            return direct
        if directory not in self._indices:
            index = {}
            index_path = directory / "index.csv"
            if index_path.is_file():
                with index_path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        stored = row.get("graph_path")
                        if not stored:
                            continue
                        path = Path(stored)
                        # Export/graph datasets may have moved since indexing.
                        candidates = [directory / path.name]
                        if path.is_absolute():
                            candidates.append(path)
                        else:
                            candidates.extend([directory / path, directory.parent / path])
                        for key in ("label_uid", "parsed_label_uid"):
                            if row.get(key):
                                index.setdefault(row[key], []).extend(candidates)
            self._indices[directory] = index
        matches = {
            path.resolve()
            for path in self._indices[directory].get(str(label_uid), [])
            if path.is_file()
        }
        if len(matches) > 1:
            raise ValueError(f"Ambiguous cell graphs for {dataset}/{timepoint}/{label_uid}")
        if matches:
            return matches.pop()
        raise FileNotFoundError(f"No preprocessed cell graph for {dataset}/{timepoint}/{label_uid}")

    def count(self, dataset, timepoint, label_uid) -> int:
        path = self.graph_path(dataset, timepoint, label_uid).resolve()
        if path not in self._counts:
            self._counts[path] = int(load_cell_graph(path).number_of_nodes())
        return self._counts[path]


def _replace_text(path: Path, text: str) -> None:
    """Atomically replace a metadata file after serialization has succeeded."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        if path.exists():
            os.chmod(temporary, path.stat().st_mode)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def update_export_cell_counts(
    output_root,
    lookup: CellCountLookup,
    *,
    datasets=None,
    timepoints=None,
    max_meshes=None,
    dry_run=False,
    strict=False,
):
    """Append counts to existing shapes and diagnostics without reconstructing fits.

    The sample identities inside shape.json drive graph matching; old absolute
    paths in manifest.csv need not still exist. Missing graphs leave a sample
    untouched and are listed in cell_count_update_report.json. A dry run only
    reads files. Dataset/timepoint filters restrict the update, and all other
    manifest rows and all non-cell-count JSON fields are preserved.
    """
    root = Path(output_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Export directory does not exist: {root}")
    records = []
    counts = {}
    for path in sorted(root.rglob("shape.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample = payload["sample"]
        dataset, timepoint, label = (str(sample[key]) for key in ("dataset", "timepoint", "label_uid"))
        if datasets is not None and dataset not in datasets:
            continue
        if timepoints is not None and timepoint not in timepoints:
            continue
        if max_meshes is not None and len(records) >= max_meshes:
            break
        record = {"dataset": dataset, "timepoint": timepoint, "label_uid": label}
        try:
            count = lookup.count(dataset, timepoint, label)
            payloads = [(path, payload)]
            quality_path = path.with_name("quality.json")
            if quality_path.is_file():
                quality = json.loads(quality_path.read_text(encoding="utf-8"))
                if any(str(quality["sample"].get(key)) != record[key] for key in record):
                    raise ValueError(f"Shape/quality sample identities disagree: {path.parent}")
                payloads.append((quality_path, quality))
            pending = []
            for target, data in payloads:
                if data["sample"].get("cell_count") != count:
                    data["sample"]["cell_count"] = count
                    pending.append((target, json.dumps(data, indent=2, allow_nan=False) + "\n"))
            if not dry_run:
                for target, text in pending:
                    _replace_text(target, text)
            counts[(dataset, timepoint, label)] = count
            record.update(cell_count=count, status="updated" if pending else "unchanged")
        except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
            if strict:
                raise
            record.update(status="missing_graph" if isinstance(exc, FileNotFoundError) else "failed", error=str(exc))
        records.append(record)

    manifest_path = root / "manifest.csv"
    if manifest_path.is_file() and counts and not dry_run:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        if "cell_count" not in fields:
            fields.append("cell_count")
        for row in rows:
            key = tuple(row.get(field) for field in ("dataset", "timepoint", "label_uid"))
            if key in counts:
                row["cell_count"] = counts[key]
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        _replace_text(manifest_path, buffer.getvalue())

    report = {
        "dry_run": bool(dry_run),
        "samples": len(records),
        **{status: sum(r["status"] == status for r in records)
           for status in ("updated", "unchanged", "missing_graph", "failed")},
        "records": records,
    }
    if not dry_run:
        _replace_text(root / "cell_count_update_report.json", json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report

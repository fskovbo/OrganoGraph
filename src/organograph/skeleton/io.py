"""Serialization helpers for organoid skeleton graphs."""

from __future__ import annotations

import json
from pathlib import Path

from organograph.skeleton.datatypes import SkeletonGraph


def save_skeleton_json(graph: SkeletonGraph, path) -> None:
    """Save a skeleton graph as JSON."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, indent=2, sort_keys=True)


def load_skeleton_json(path) -> SkeletonGraph:
    """Load a skeleton graph saved with `save_skeleton_json`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return SkeletonGraph.from_dict(data)

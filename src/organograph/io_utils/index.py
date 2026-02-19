# io/index.py
import os, glob

def iter_graph_records(graphs_dir, timepoints=None):
    """
    Yields dicts with keys:
      - timepoint (or None)
      - graph_path
    Supports:
      - graphs_dir/<tp>/*.gpickle
      - graphs_dir/*.gpickle
    """
    if timepoints is not None:
        for tp in timepoints:
            tp_dir = os.path.join(graphs_dir, tp)
            for p in sorted(glob.glob(os.path.join(tp_dir, "*.gpickle"))):
                yield {"timepoint": tp, "graph_path": p}
        return

    subdirs = [d for d in os.listdir(graphs_dir) if os.path.isdir(os.path.join(graphs_dir, d))]
    if subdirs:
        for tp in sorted(subdirs):
            for p in sorted(glob.glob(os.path.join(graphs_dir, tp, "*.gpickle"))):
                yield {"timepoint": tp, "graph_path": p}
    else:
        for p in sorted(glob.glob(os.path.join(graphs_dir, "*.gpickle"))):
            yield {"timepoint": None, "graph_path": p}

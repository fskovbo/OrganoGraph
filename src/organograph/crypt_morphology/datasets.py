"""Read-only adapters for saved HKS membership masks and original mesh geometry."""
from pathlib import Path
import hashlib
import re
import numpy as np


def biological_day(timepoint):
    """Parse e.g. day3p5 and day4p5-more; reject non-age folder names."""
    match = re.fullmatch(r"day(\d+)(?:p(\d+))?(?:-.*)?", str(timepoint))
    if match is None:
        raise ValueError(f"Not a biological day: {timepoint}")
    return float(match[1]) + (float("0." + match[2]) if match[2] else 0)


def discover_hks_crypts(data_root, *, datasets=("20250929", "20251201"), min_day=3.5,
                        respect_blacklist=True):
    """Enumerate crypts without reading detector axes/tips or normalizing geometry."""
    from organograph.io_utils.blacklist import default_discard_labels_path, load_optional_blacklist
    root = Path(data_root)
    records = []
    for dataset in datasets:
        folder = root / dataset / "crypt_segmentations_mesh"
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        blacklist = (load_optional_blacklist(default_discard_labels_path(str(root / dataset)),
                                              verbose=False) if respect_blacklist else set())
        for path in sorted(folder.glob("*/*.npz")):
            try:
                day = biological_day(path.parent.name)
            except ValueError:
                continue
            if day < min_day:
                continue
            with np.load(path, allow_pickle=True) as data:
                label = str(data["label_uid"].item())
                if label in blacklist:
                    continue
                mesh_path = Path(str(data["mesh_path"].item()))
                if not mesh_path.exists():
                    # Relocate an archived absolute path using its dataset suffix.
                    parts = mesh_path.parts
                    if dataset in parts:
                        mesh_path = root / Path(*parts[parts.index(dataset):])
                for i, ids in enumerate(data["crypts_ll"]):
                    if len(ids):
                        records.append({"dataset": dataset, "timepoint": path.parent.name,
                                        "day": day, "label_uid": label, "crypt_index": i,
                                        "n_selected_vertices": len(ids), "segmentation_path": str(path.resolve()),
                                        "mesh_path": str(mesh_path.resolve())})
    return records


def select_validation_crypts(records, *, per_group=4, seed=2026):
    """Deterministic size-stratified pilot, favoring distinct organoids per group."""
    if per_group < 1:
        raise ValueError("per_group must be positive")
    groups = {}
    for record in records:
        groups.setdefault((record["dataset"], record["timepoint"]), []).append(record)
    rng, selected = np.random.default_rng(seed), []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: (r["n_selected_vertices"], r["label_uid"], r["crypt_index"]))
        used = set()
        for indices in np.array_split(np.arange(len(group)), min(per_group, len(group))):
            options = [i for i in indices if group[i]["label_uid"] not in used]
            index = int(rng.choice(options if options else indices))
            selected.append(dict(group[index]))
            used.add(group[index]["label_uid"])
    return selected


def load_hks_crypt(record):
    """Load original coordinate units and preserve source mesh identities."""
    from .patches import extract_patch
    vertices, faces, ids, _, metadata = load_hks_parent(record)
    return extract_patch(vertices, faces, ids), metadata


def load_hks_parent(record):
    """Return parent geometry, selected/neighbor membership and provenance.

    Only membership masks are read from segmentation; no bottom, axis or
    detector-derived longitudinal coordinate participates in the representation.
    """
    from organograph.mesh.OrganoidMesh import OrganoidMesh
    mesh = OrganoidMesh(record["mesh_path"])
    with np.load(record["segmentation_path"], allow_pickle=True) as data:
        masks = data["crypts_ll"]
        ids = np.asarray(masks[record["crypt_index"]], int)
        others = [np.asarray(m, int) for i, m in enumerate(masks) if i != record["crypt_index"] and len(m)]
        neighbors = np.unique(np.concatenate(others)) if others else np.empty(0, int)
    fingerprint = hashlib.sha256()
    fingerprint.update(np.asarray(mesh.v, dtype="<f8").tobytes())
    fingerprint.update(np.asarray(mesh.f, dtype="<i8").tobytes())
    metadata = dict(record, mesh_n_vertices=len(mesh.v), mesh_n_faces=len(mesh.f),
                    mesh_geometry_sha256=fingerprint.hexdigest(), coordinate_units="source_mesh_units")
    return mesh.v, mesh.f, ids, neighbors, metadata

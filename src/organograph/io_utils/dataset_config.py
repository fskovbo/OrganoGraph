import json


def load_mesh_dataset_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)

    required = ["timepoints", "zarr_name_by_tp", "round_by_tp", "meshname_by_tp"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Mesh config missing required key: {k}")

    cfg.setdefault("wells_by_tp", {})
    return cfg


def load_cell_table_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)

    required = ["coord_cols", "marker_cols", "lgr5_marker", "coexp_markers"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Cell-table config missing required key: {k}")

    return cfg
# io/config.py

def validate_cfg(cfg):
    """
    Required keys:
      - graphs_dir: root directory containing .gpickle files (optionally per timepoint)
      - seg_dir: directory to write segmentation .npz
      - mesh_path_from_label_uid: callable(label_uid)->mesh_path

    Optional keys:
      - features_dir: directory to write compact feature .npz
      - blacklist: iterable of label_uid
    """
    for k in ["graphs_dir", "seg_dir", "mesh_path_from_label_uid"]:
        if k not in cfg:
            raise ValueError(f"Missing cfg['{k}']")
    if not callable(cfg["mesh_path_from_label_uid"]):
        raise ValueError("cfg['mesh_path_from_label_uid'] must be callable(label_uid)->mesh_path")
    if cfg.get("blacklist", None) is None:
        cfg["blacklist"] = []
    return cfg

# io/segmentation_io.py
import os
import numpy as np

def patches_to_ll(patches):
    # list[set[int]] or list[list[int]] -> list[list[int]]
    if patches is None:
        return []
    out = []
    for p in patches:
        if p is None:
            continue
        if isinstance(p, set):
            p = list(p)
        else:
            p = list(p)
        if len(p) > 0:
            out.append(p)
    return out


def as_patch_list(x):
    """Normalize region definitions to list[set[int]]."""
    if x is None:
        return []
    if isinstance(x, set):
        return [x]
    if isinstance(x, (list, tuple)):
        return [set(p) for p in x if p is not None and len(p) > 0]
    return []


def load_mesh_crypt_segmentation(seg_path):
    """
    Load one mesh-based crypt segmentation .npz.

    Expects at least:
      - label_uid
      - timepoint
      - mesh_path
      - crypts_ll
    """
    z = np.load(seg_path, allow_pickle=True)

    if "crypts_ll" not in z:
        raise KeyError(f"{seg_path} does not contain 'crypts_ll'")

    label_uid = str(z["label_uid"]) if "label_uid" in z else None
    timepoint = str(z["timepoint"]) if "timepoint" in z else None
    mesh_path = str(z["mesh_path"]) if "mesh_path" in z else None

    crypts_mesh = [set(map(int, p)) for p in z["crypts_ll"]]

    out = {
        "label_uid": label_uid,
        "timepoint": timepoint,
        "mesh_path": mesh_path,
        "crypts_mesh": crypts_mesh,
    }

    # carry through optional fields if they exist
    for k in ("bottom_vertex_ids", "L_crypts", "circumference_crypts", "d_discretized"):
        if k in z:
            out[k] = z[k]

    return out
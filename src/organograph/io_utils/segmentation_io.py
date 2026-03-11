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


def _npz_to_python_scalar(x):
    """Convert 0-d numpy arrays to plain Python scalars where possible."""
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    return x


def _load_segmentation_npz_generic(seg_path, required_keys, patch_key, out_patch_key):
    """
    Generic loader for segmentation npz files that store patches as `patch_key`.

    Parameters
    ----------
    seg_path : str
    required_keys : tuple[str]
    patch_key : str
        Name of object-array/list-like patch storage in the npz, e.g. 'crypts_ll'
    out_patch_key : str
        Name of normalized patch-list field in output, e.g. 'crypts_graph'

    Returns
    -------
    dict
        All keys are loaded; patch_key is normalized to list[set[int]] under out_patch_key.
    """
    z = np.load(seg_path, allow_pickle=True)

    missing = [k for k in required_keys if k not in z]
    if missing:
        raise KeyError(f"{seg_path} is missing required keys: {missing}")

    out = {}
    for k in z.files:
        out[k] = _npz_to_python_scalar(z[k])

    for k in ("label_uid", "timepoint", "mesh_path", "graph_path", "mesh_seg_path"):
        if k in out:
            out[k] = str(out[k])

    out[out_patch_key] = [set(map(int, p)) for p in out.pop(patch_key)]
    return out


def load_mesh_crypt_segmentation(seg_path):
    """
    Load a mesh-based crypt segmentation (.npz).

    Required keys
    -------------
    label_uid
    timepoint
    mesh_path
    crypts_ll

    Returns
    -------
    dict
        Dictionary containing the segmentation and all stored variables.
        crypts_ll is converted to list[set[int]] under the key 'crypts_mesh'.
    """
    return _load_segmentation_npz_generic(
        seg_path,
        required_keys=("label_uid", "timepoint", "mesh_path", "crypts_ll"),
        patch_key="crypts_ll",
        out_patch_key="crypts_mesh",
    )


def load_graph_crypt_segmentation(seg_path):
    """
    Load a graph-based crypt segmentation (.npz).

    Required keys
    -------------
    label_uid
    timepoint
    graph_path
    crypts_ll

    Returns
    -------
    dict
        Dictionary containing the segmentation and all stored variables.
        crypts_ll is converted to list[set[int]] under the key 'crypts_graph'.
    """
    return _load_segmentation_npz_generic(
        seg_path,
        required_keys=("label_uid", "timepoint", "graph_path", "crypts_ll"),
        patch_key="crypts_ll",
        out_patch_key="crypts_graph",
    )


def load_graph_crypt_segmentation_if_present(seg_path):
    """Return loaded graph segmentation dict or None if seg_path is missing."""
    if not isinstance(seg_path, str) or not os.path.exists(seg_path):
        return None
    return load_graph_crypt_segmentation(seg_path)
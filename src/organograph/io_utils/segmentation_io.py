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


def save_segmentation_npz(
    path,
    *,
    label_uid,
    crypts_ll,
    villi_ll,
    bin_centers=None,
    d_norm=None,
    L_crypt=None,
    Circ=None,
    extra=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "label_uid": np.array([label_uid], dtype=object),
        "crypts": np.array(crypts_ll, dtype=object),
        "villi":  np.array(villi_ll,  dtype=object),
    }
    if bin_centers is not None: payload["bin_centers"] = np.asarray(bin_centers, dtype=np.float32)
    if d_norm is not None:      payload["dnorm_v"]     = np.asarray(d_norm, dtype=np.float32)
    if L_crypt is not None:     payload["L_crypt"]     = np.asarray(L_crypt, dtype=np.float32)
    if Circ is not None:        payload["Circ"]        = np.asarray(Circ,    dtype=np.float32)
    if extra:
        for k, v in extra.items():
            payload[k] = v
    np.savez_compressed(path, **payload)

def load_segmentation_npz(path):
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["label_uid"] = str(out["label_uid"][0])
    out["crypts_ll"] = out["crypts"].tolist() if "crypts" in out else []
    out["villi_ll"]  = out["villi"].tolist()  if "villi" in out else []
    return out

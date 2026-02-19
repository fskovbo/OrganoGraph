# io/features_io.py
import os
import numpy as np

def save_features_npz(path, *, label_uid, markers_bin=None, proj_vertex_ids=None, centroids=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"label_uid": np.array([label_uid], dtype=object)}
    if markers_bin is not None:
        payload["markers_bin"] = np.asarray(markers_bin)
    if proj_vertex_ids is not None:
        payload["proj_vertex_ids"] = np.asarray(proj_vertex_ids)
    if centroids is not None:
        payload["centroids"] = np.asarray(centroids)
    if extra:
        for k, v in extra.items():
            payload[k] = v
    np.savez_compressed(path, **payload)

def load_features_npz(path):
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["label_uid"] = str(out["label_uid"][0])
    return out

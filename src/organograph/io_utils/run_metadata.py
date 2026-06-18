import json
import os
from datetime import datetime

import numpy as np


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if callable(value):
        return getattr(value, "__name__", str(value))
    return value


def write_run_settings(output_dir, *, script_name, payload, prefix="run_settings", verbose=True):
    """
    Write a timestamped JSON sidecar describing a script run.

    The payload is intentionally lightweight provenance: script, dataset/subset,
    important paths, parameters, marker postprocessing functions, and run counts.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"{prefix}_{timestamp}.json")
    data = {
        "script_name": str(script_name),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **_json_safe(dict(payload)),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    if verbose:
        print(f"[settings] wrote {out_path}")
    return out_path

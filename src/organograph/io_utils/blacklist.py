import os
import numpy as np


def load_blacklist(path):
    """
    Load a blacklist of label_uid from various file formats.
    Returns a set of strings.
    """
    if path is None:
        return set()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Blacklist file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path) as f:
            return {line.strip() for line in f if line.strip()}

    if ext == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        return set(df.iloc[:, 0].astype(str))

    if ext == ".json":
        import json
        with open(path) as f:
            data = json.load(f)
        return {str(x) for x in data}

    if ext == ".npy":
        arr = np.load(path, allow_pickle=True)
        return {str(x) for x in arr}

    if ext == ".npz":
        arr = np.load(path, allow_pickle=True)
        key = list(arr.keys())[0]
        return {str(x) for x in arr[key]}

    raise ValueError(f"Unsupported blacklist format: {ext}")


def append_to_blacklist(path, label_uids):
    """
    Append label_uids to an existing blacklist file while avoiding duplicates.
    The blacklist is saved in the same format as the original file.

    Parameters
    ----------
    path : str
        Path to blacklist file.
    label_uids : iterable
        Iterable of label_uid to add.
    """

    # Load existing blacklist
    existing = load_blacklist(path)

    # Merge
    existing.update(str(x) for x in label_uids)

    ext = os.path.splitext(path)[1].lower()
    data = sorted(existing)

    if ext == ".txt":
        with open(path, "w") as f:
            for uid in data:
                f.write(f"{uid}\n")

    elif ext == ".csv":
        import pandas as pd
        pd.DataFrame({"label_uid": data}).to_csv(path, index=False)

    elif ext == ".json":
        import json
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    elif ext == ".npy":
        np.save(path, np.array(data))

    elif ext == ".npz":
        np.savez(path, label_uid=np.array(data))

    else:
        raise ValueError(f"Unsupported blacklist format: {ext}")
    

def create_blacklist(path, label_uids):
    """
    Create a new blacklist file from label_uids.

    Parameters
    ----------
    path : str
        Output file path.
    label_uids : iterable
        Iterable of label_uid.
    """

    data = sorted({str(x) for x in label_uids})
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path, "w") as f:
            for uid in data:
                f.write(f"{uid}\n")

    elif ext == ".csv":
        import pandas as pd
        pd.DataFrame({"label_uid": data}).to_csv(path, index=False)

    elif ext == ".json":
        import json
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    elif ext == ".npy":
        np.save(path, np.array(data))

    elif ext == ".npz":
        np.savez(path, label_uid=np.array(data))

    else:
        raise ValueError(f"Unsupported blacklist format: {ext}")
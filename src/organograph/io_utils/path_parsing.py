import os
import glob


def parse_mesh_path(mesh_path):
    """Parse identifiers from a mesh path.

    You must adapt this for your folder tree.

    Default folder structure assumed:
        {data_dir}/{timepoint}/{zarr_name}/{well_letter}/{well_field}/{round_name}/meshes/{mesh_name}/{organoid_id}.vtp

    Returns a dict with:
        - timepoint (str)  REQUIRED
        - label_uid (str)  REQUIRED
        - well (str)       OPTIONAL (e.g. "B02")
        - organoid_id (str) OPTIONAL
    """
    mesh_path = os.path.normpath(mesh_path)
    organoid_id = os.path.splitext(os.path.basename(mesh_path))[0]
    if organoid_id == "":
        raise ValueError("empty organoid_id")

    parts = mesh_path.split(os.sep)

    # Find the "meshes" folder; then we can read folders before it
    try:
        i_meshes = parts.index("meshes")
    except ValueError:
        raise ValueError(f"'meshes' folder not found in path: {mesh_path}")

    # Expected around meshes:
    # .../<timepoint>/<zarr>/<well_letter>/<well_field>/<round>/meshes/<mesh_name>/<organoid_id>.vtp
    if i_meshes < 5:
        raise ValueError(f"path too short to parse identifiers: {mesh_path}")

    well_field = parts[i_meshes - 2]
    well_letter = parts[i_meshes - 3]
    zarr_name = parts[i_meshes - 4]
    timepoint = parts[i_meshes - 5]

    if not timepoint:
        raise ValueError(f"could not parse timepoint from path: {mesh_path}")

    # Well parsing is optional; be strict if it looks like your intended layout
    well = None
    if len(well_letter) == 1 and well_letter.isalpha() and well_field.isdigit():
        well = f"{well_letter}{well_field}"

    if not zarr_name.endswith(".zarr"):
        raise ValueError(f"expected zarr folder, got '{zarr_name}' in path: {mesh_path}")

    label_uid = f"{timepoint}_{well}_{organoid_id}" if well is not None else f"{timepoint}_{organoid_id}"
    return {
        "timepoint": timepoint,
        "well": well,
        "organoid_id": organoid_id,
        "label_uid": label_uid,
    }


def discover_mesh_paths(data_dir, timepoints, zarr_names, rounds, meshes, wells=None):
    """Discover mesh files using a restrictive glob matching the expected layout.

    Expected layout:
        {data_dir}/{tp}/{zarr}/{well_letter}/{well_field}/{round}/meshes/{mesh_name}/*.vtp

    Parameters
    ----------
    data_dir : str
        Root folder.
    timepoints : list[str]
        Timepoints to search under.
    zarr_names, rounds, meshes : dict[str, str]
        Per-timepoint folder names.
    wells : dict[str, list[str]] or None
        Optional per-timepoint well allow-list like {"day4p5": ["B02", ...]}.

    Returns
    -------
    list[str] : sorted unique mesh paths
    """
    if wells is None:
        wells = {}

    found = []
    for tp in (timepoints or []):
        zarr = zarr_names.get(tp, None)
        rnd = rounds.get(tp, None)
        mesh_name = meshes.get(tp, None)

        if zarr is None or rnd is None or mesh_name is None:
            raise ValueError(
                f"Missing folder config for timepoint '{tp}'. Need zarr_names[{tp}], rounds[{tp}], meshes[{tp}]."
            )

        tp_wells = wells.get(tp, None)

        # If wells are provided, search only those; otherwise glob across any well_letter/well_field
        if tp_wells:
            for well in tp_wells:
                if not well or len(well) < 2:
                    continue
                wl = well[0]
                wf = well[1:]
                pattern = os.path.join(
                    data_dir, tp, zarr, wl, wf, rnd, "meshes", mesh_name, "*.vtp"
                )
                found.extend(glob.glob(pattern))
        else:
            pattern = os.path.join(
                data_dir, tp, zarr, "*", "*", rnd, "meshes", mesh_name, "*.vtp"
            )
            found.extend(glob.glob(pattern))

    # Unique + stable order
    return sorted(set(os.path.normpath(p) for p in found))
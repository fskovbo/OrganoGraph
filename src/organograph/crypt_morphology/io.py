"""Single-sample interchange for the external learning project; no batch export."""
from pathlib import Path
import json
import numpy as np

from .basis import CapBasis
from .datatypes import AlignmentReference, CapRepresentation, SimilarityTransform


def save_basis(path, basis):
    Path(path).write_text(json.dumps(basis.to_dict(), indent=2) + "\n")


def load_basis(path):
    return CapBasis.from_dict(json.loads(Path(path).read_text()))


def save_representation(path, representation, *, patch=None, mapping=None):
    """Numeric NPZ + JSON metadata, readable with allow_pickle=False.

    Optionally retain the original patch and mapping for later field transfer.
    NPZ stores representations, not learned latent vectors.
    """
    r = representation
    meta = {"schema": "organograph_cap_representation_v1", "basis_id": r.basis_id,
            "size_mode": r.size_mode, "surface_area": r.surface_area, "rms_radius": r.rms_radius,
            "phase": r.phase, "diagnostics": r.diagnostics, "metadata": r.metadata}
    arrays = {"coefficients": r.coefficients, "rotation": r.pose.rotation,
              "translation": r.pose.translation, "scale": r.pose.scale,
              "metadata_json": json.dumps(meta)}
    if (patch is None) != (mapping is None):
        raise ValueError("Supply patch and mapping together")
    if patch is not None:
        arrays.update(source_vertex_ids=patch.source_vertex_ids, source_face_ids=patch.source_face_ids,
                      source_vertices=patch.vertices, source_faces=patch.faces, boundary=patch.boundary,
                      disk_uv=mapping.uv, theta_phi=mapping.theta_phi,
                      cap_half_angle=mapping.half_angle)
    # File handles preserve the exact caller-supplied filename.
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_representation(path, basis):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        if meta.get("schema") != "organograph_cap_representation_v1" or meta["basis_id"] != basis.identifier:
            raise ValueError("Representation schema/basis mismatch")
        c = np.asarray(z["coefficients"], float)
        if c.shape != (basis.n_modes, 3) or not np.isfinite(c).all():
            raise ValueError("Invalid coefficients")
        pose = SimilarityTransform(z["rotation"].copy(), z["translation"].copy(), float(z["scale"]))
    return CapRepresentation(c, meta["basis_id"], pose, meta["size_mode"], meta["surface_area"],
                             meta["rms_radius"], meta["diagnostics"], meta["metadata"], meta["phase"])


def save_alignment_reference(path, reference):
    r = reference
    meta = {"schema": "organograph_cap_alignment_v1", "basis_id": r.basis_id,
            "size_mode": r.size_mode, "n_rings": r.n_rings, "n_sectors": r.n_sectors,
            "n_phase": r.n_phase}
    with open(path, "wb") as handle:
        np.savez_compressed(handle, vertices=r.vertices, weights=r.weights, metadata_json=json.dumps(meta))


def load_alignment_reference(path, basis):
    with np.load(path, allow_pickle=False) as z:
        m = json.loads(str(z["metadata_json"].item()))
        if m.get("schema") != "organograph_cap_alignment_v1" or m["basis_id"] != basis.identifier:
            raise ValueError("Alignment reference schema/basis mismatch")
        return AlignmentReference(z["vertices"].copy(), z["weights"].copy(), m["basis_id"],
                                  m["size_mode"], m["n_rings"], m["n_sectors"], m["n_phase"])

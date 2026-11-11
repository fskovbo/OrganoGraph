"""Deterministic analytical surfaces for inspecting representation limitations."""
import numpy as np
from .basis import CapBasis


def synthetic_crypt(kind="asymmetric", *, n_rings=24, n_sectors=64):
    """Original analytical mesh, independent of any fitted coefficient vector."""
    q, f = CapBasis.build(1).reference_mesh(n_rings, n_sectors)
    u, phi = q[:, 0] / (np.pi / 2), q[:, 1]
    settings = {
        "hemisphere": (np.pi / 2, 1.0, 0.0, 0.0),
        "elongated": (np.pi / 2, 2.8, 0.0, 0.0),
        "constricted": (2.45, 1.3, 0.0, 0.0),
        "bent": (2.0, 2.0, 0.65, 0.0),
        "asymmetric": (2.05, 1.5, 0.3, 0.18),
        "broad_bottom": (1.8, 0.6, 0.0, 0.08),
    }
    if kind not in settings:
        raise ValueError(f"Unknown kind; choose from {tuple(settings)}")
    angle, height, bend, asymmetry = settings[kind]
    t = u * angle
    radius = np.sin(t) * (1 + asymmetry * np.sin(3 * phi) * np.sin(t) ** 2)
    z = height * np.cos(t)
    x = radius * np.cos(phi) + bend * (1 - np.cos(t)) ** 2
    y = radius * np.sin(phi) * (1 - asymmetry / 2)
    return np.column_stack([x, y, z]), f

"""Array-based results; importing this package does not require VTK or plotting."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimilarityTransform:
    """Reversible convention: ``y = x @ rotation.T / scale + translation``."""

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: float = 1.0

    def __post_init__(self):
        self.rotation = np.asarray(self.rotation, float).copy()
        self.translation = np.asarray(self.translation, float).copy()
        if self.rotation.shape != (3, 3) or self.translation.shape != (3,):
            raise ValueError("Transform requires a 3x3 rotation and length-three translation")
        if not np.isfinite(self.rotation).all() or not np.isfinite(self.translation).all():
            raise ValueError("Transform must be finite")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Transform scale must be positive and finite")
        if not np.allclose(self.rotation @ self.rotation.T, np.eye(3), atol=1e-8, rtol=0) or np.linalg.det(self.rotation) <= 0:
            raise ValueError("Transform rotation must be orthogonal and preserve handedness")

    def apply(self, points):
        return np.asarray(points) @ self.rotation.T / self.scale + self.translation

    def inverse(self, points):
        return (np.asarray(points) - self.translation) @ self.rotation * self.scale

    def then(self, other):
        """Compose this transform followed by ``other``."""
        return SimilarityTransform(
            other.rotation @ self.rotation,
            other.apply(self.translation),
            self.scale * other.scale,
        )


@dataclass
class SurfacePatch:
    vertices: np.ndarray
    faces: np.ndarray
    source_vertex_ids: np.ndarray
    source_face_ids: np.ndarray
    boundary: np.ndarray
    unused_selected_vertex_ids: np.ndarray


@dataclass
class CapMapping:
    """One coordinate per patch vertex; no anatomical tip is required."""

    uv: np.ndarray
    theta_phi: np.ndarray
    half_angle: float
    diagnostics: dict


@dataclass
class CapRepresentation:
    coefficients: np.ndarray
    basis_id: str
    pose: SimilarityTransform
    size_mode: str
    surface_area: float
    rms_radius: float
    diagnostics: dict
    metadata: dict = field(default_factory=dict)
    phase: float = 0.0


@dataclass
class ProcrustesResult:
    template: np.ndarray
    aligned: np.ndarray
    transforms: list[SimilarityTransform]
    objective_history: list[float]
    converged: bool
    phases: np.ndarray | None = None
    representations: list[CapRepresentation] | None = None
    phase_ambiguous: np.ndarray | None = None
    weights: np.ndarray | None = None
    sampling: tuple[int, int, int] | None = None


@dataclass
class AlignmentReference:
    """Frozen template and the metric/sampling convention used to fit it."""

    vertices: np.ndarray
    weights: np.ndarray
    basis_id: str
    size_mode: str
    n_rings: int
    n_sectors: int
    n_phase: int = 32

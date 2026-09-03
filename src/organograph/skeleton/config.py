"""Typed configuration for the definitive organoid shape pipeline.

The public configuration mirrors biological stages rather than individual
implementation functions. Numerical details which are not useful tuning knobs
remain internal to those stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from math import pi
from typing import Any


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if callable(value):
        return getattr(value, "__name__", value.__class__.__name__)
    return value


def _coerce(cls, value):
    if isinstance(value, cls):
        return value
    return cls(**dict(value or {}))


@dataclass
class CandidateDetectionConfig:
    """HKS candidate, split-refinement, and final-tip settings."""

    threshold: float = 0.5
    filters: list[Any] | None = None
    vocab_reference: Any | None = None
    crypt_vocab_indices: Any | None = None
    refine_threshold: float = 0.0
    refine_min_area: float = 5.0
    min_child_fraction: float = 0.05
    final_tip_hks_time: float = 1.0
    final_tip_bottom_fraction: float = 0.6
    final_tip_min_hks_percent_increase: float = 5.0
    geodesic_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class NeckProfileConfig:
    """Circumference-profile settings used to classify crypt necks."""

    max_axis_level: float = 2.0
    resolution: int = 200
    search_interval: tuple[float, float] = (0.8, 2.0)
    min_prominence: float = 0.05
    min_length: float = 0.05


@dataclass
class BranchValidationConfig:
    """Settings for deciding whether a refined parent is a true branch."""

    min_confidence: float = 0.85
    max_neck_to_body_radius_ratio: float = 0.70
    max_growth_size_factor: float = 3.0
    max_mesh_fraction: float = 0.40
    min_perimeter_prominence_fraction: float = 0.01


def _default_body_barrier_options() -> dict[str, Any]:
    return {
        "primitive_type": "superellipsoid",
        "barrier_weight": 2.5,
        "underfill_weight": 0.02,
        "center_regularization": 0.01,
        "anisotropy_regularization": 0.1,
        "center_shift_limit_frac": 0.55,
        "initial_radius_quantile": 0.6,
        "initial_epsilon_1": 0.9,
        "epsilon_1_bounds": (0.35, 1.0),
        "epsilon_1_regularization": 0.01,
        "epsilon_2": 1.0,
        "maxiter": 2000,
    }


def _default_branch_barrier_options() -> dict[str, Any]:
    options = _default_body_barrier_options()
    options.update(
        primitive_type="ellipsoid",
        anisotropy_regularization=1.0,
        underfill_weight=0.5,
        barrier_weight=100.0,
    )
    return options


@dataclass
class BarrierConfig:
    """Body/branch barrier fitting, ownership, and opening projection settings."""

    body_fit_options: dict[str, Any] = field(default_factory=_default_body_barrier_options)
    branch_fit_options: dict[str, Any] = field(default_factory=_default_branch_barrier_options)
    body_ownership_level: float = 1.2
    branch_ownership_level: float = 1.1
    min_candidate_vertices: int = 4
    min_branch_vertices: int = 20
    sample_fraction: float = 1.0
    sample_seed: int | None = 0
    opening_grid_resolution: int = 31


@dataclass
class MeshPreparationConfig:
    """Optional low-pass mesh used consistently by all skeleton stages."""

    smooth: bool = False
    spectral_lmax: int = 5
    recompute_eigen: bool = True
    eigen_k: int | None = None


@dataclass
class DetectionConfig:
    """Configuration of the complete barrier-aware detection stage."""

    candidates: CandidateDetectionConfig = field(default_factory=CandidateDetectionConfig)
    necks: NeckProfileConfig = field(default_factory=NeckProfileConfig)
    branches: BranchValidationConfig = field(default_factory=BranchValidationConfig)
    barriers: BarrierConfig = field(default_factory=BarrierConfig)
    mesh: MeshPreparationConfig = field(default_factory=MeshPreparationConfig)

    def __post_init__(self):
        self.candidates = _coerce(CandidateDetectionConfig, self.candidates)
        self.necks = _coerce(NeckProfileConfig, self.necks)
        self.branches = _coerce(BranchValidationConfig, self.branches)
        self.barriers = _coerce(BarrierConfig, self.barriers)
        self.mesh = _coerce(MeshPreparationConfig, self.mesh)


@dataclass
class SkeletonizationConfig:
    """Settings for the single supported barrier-aware skeleton workflow."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)

    def __post_init__(self):
        self.detection = _coerce(DetectionConfig, self.detection)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SkeletonizationConfig":
        return cls(**dict(data or {}))


@dataclass
class CryptOverlapConfig:
    """Post-fit merging of strongly overlapping crypt tube primitives."""

    enabled: bool = True
    threshold: float = 0.30
    samples: int = 32768
    random_seed: int | None = 0
    max_passes: int = 3
    max_host_attachment_angle: float | None = pi / 3.0

    def __post_init__(self):
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("crypt overlap threshold must be between 0 and 1")
        if int(self.samples) < 256:
            raise ValueError("crypt overlap estimation requires at least 256 samples")
        if int(self.max_passes) < 1:
            raise ValueError("crypt overlap max_passes must be at least 1")
        if self.max_host_attachment_angle is not None and not (
            0.0 <= float(self.max_host_attachment_angle) <= pi
        ):
            raise ValueError(
                "max_host_attachment_angle must be between 0 and pi, or None"
            )


@dataclass
class PrimitiveFitConfig:
    """Settings for crypt/neck fitting and optional host-primitive refinement."""

    refine_host_primitives: bool = False
    component_kwargs: dict[str, Any] = field(default_factory=dict)
    body_kwargs: dict[str, Any] = field(default_factory=dict)
    branch_kwargs: dict[str, Any] = field(default_factory=dict)
    body_branch_neck_kwargs: dict[str, Any] = field(default_factory=dict)
    crypt_tube_kwargs: dict[str, Any] = field(default_factory=dict)
    crypt_overlap: CryptOverlapConfig = field(default_factory=CryptOverlapConfig)

    def __post_init__(self):
        self.crypt_overlap = _coerce(CryptOverlapConfig, self.crypt_overlap)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PrimitiveFitConfig":
        return cls(**dict(data or {}))


@dataclass
class BlendConfig:
    """Settings for downstream visualization-only primitive blending."""

    enabled: bool = True
    n_samples: int = 32
    extension_length_fraction: float = 0.5
    host_overlap_radius_fraction: float = 0.8
    max_host_overlap_distance_fraction: float = 0.2
    crypt_overlap_radius_fraction: float = 0.15
    host_radius_scale: float = 2.5
    min_host_to_crypt_radius_ratio: float = 1.4
    max_host_radius_fraction: float = 0.35
    use_mesh_radius_fit: bool = True
    mesh_radius_quantile: float = 0.65
    mesh_search_radius_scale: float = 2.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BlendConfig":
        return cls(**dict(data or {}))

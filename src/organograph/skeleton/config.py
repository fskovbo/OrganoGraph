"""Configuration containers for skeleton and primitive pipelines.

The objects here are intentionally light wrappers around keyword dictionaries.
They make batch runs easier to store and replay without freezing the rapidly
evolving algorithm into a large rigid parameter schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkeletonizationConfig:
    """Settings for crypt detection followed by skeleton graph construction."""

    detection_kwargs: dict[str, Any] = field(default_factory=dict)
    build_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_kwargs": dict(self.detection_kwargs),
            "build_kwargs": dict(self.build_kwargs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SkeletonizationConfig":
        data = dict(data or {})
        return cls(
            detection_kwargs=dict(data.get("detection_kwargs", {})),
            build_kwargs=dict(data.get("build_kwargs", {})),
        )


@dataclass
class PrimitiveFitConfig:
    """Settings for fitting interpretable primitives to skeleton components."""

    component_kwargs: dict[str, Any] = field(default_factory=dict)
    body_kwargs: dict[str, Any] = field(default_factory=dict)
    branch_kwargs: dict[str, Any] = field(default_factory=dict)
    body_branch_neck_kwargs: dict[str, Any] = field(default_factory=dict)
    crypt_tube_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_kwargs": dict(self.component_kwargs),
            "body_kwargs": dict(self.body_kwargs),
            "branch_kwargs": dict(self.branch_kwargs),
            "body_branch_neck_kwargs": dict(self.body_branch_neck_kwargs),
            "crypt_tube_kwargs": dict(self.crypt_tube_kwargs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PrimitiveFitConfig":
        data = dict(data or {})
        return cls(
            component_kwargs=dict(data.get("component_kwargs", {})),
            body_kwargs=dict(data.get("body_kwargs", {})),
            branch_kwargs=dict(data.get("branch_kwargs", {})),
            body_branch_neck_kwargs=dict(data.get("body_branch_neck_kwargs", {})),
            crypt_tube_kwargs=dict(data.get("crypt_tube_kwargs", {})),
        )


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
        return {
            "enabled": bool(self.enabled),
            "n_samples": int(self.n_samples),
            "extension_length_fraction": float(self.extension_length_fraction),
            "host_overlap_radius_fraction": float(self.host_overlap_radius_fraction),
            "max_host_overlap_distance_fraction": float(
                self.max_host_overlap_distance_fraction
            ),
            "crypt_overlap_radius_fraction": float(self.crypt_overlap_radius_fraction),
            "host_radius_scale": float(self.host_radius_scale),
            "min_host_to_crypt_radius_ratio": float(
                self.min_host_to_crypt_radius_ratio
            ),
            "max_host_radius_fraction": float(self.max_host_radius_fraction),
            "use_mesh_radius_fit": bool(self.use_mesh_radius_fit),
            "mesh_radius_quantile": float(self.mesh_radius_quantile),
            "mesh_search_radius_scale": float(self.mesh_search_radius_scale),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BlendConfig":
        data = dict(data or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            n_samples=int(data.get("n_samples", 32)),
            extension_length_fraction=float(data.get("extension_length_fraction", 0.5)),
            host_overlap_radius_fraction=float(
                data.get("host_overlap_radius_fraction", 0.8)
            ),
            max_host_overlap_distance_fraction=float(
                data.get("max_host_overlap_distance_fraction", 0.2)
            ),
            crypt_overlap_radius_fraction=float(
                data.get("crypt_overlap_radius_fraction", 0.15)
            ),
            host_radius_scale=float(data.get("host_radius_scale", 2.5)),
            min_host_to_crypt_radius_ratio=float(
                data.get("min_host_to_crypt_radius_ratio", 1.4)
            ),
            max_host_radius_fraction=float(data.get("max_host_radius_fraction", 0.35)),
            use_mesh_radius_fit=bool(data.get("use_mesh_radius_fit", True)),
            mesh_radius_quantile=float(data.get("mesh_radius_quantile", 0.65)),
            mesh_search_radius_scale=float(data.get("mesh_search_radius_scale", 2.5)),
            metadata=dict(data.get("metadata", {})),
        )

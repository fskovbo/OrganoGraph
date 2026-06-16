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

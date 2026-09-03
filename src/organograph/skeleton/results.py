"""Result objects passed between skeleton, primitive, and future blending stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from organograph.skeleton.config import (
    BlendConfig,
    DetectionConfig,
    PrimitiveFitConfig,
    SkeletonizationConfig,
)
from organograph.skeleton.datatypes import SkeletonGraph


@dataclass
class BarrierStageResult:
    """Body and branch barriers shared by detection, graph, and primitive stages."""

    body_fit: Any
    body_mask: Any
    branch_fits: dict[str, Any] = field(default_factory=dict)
    branch_masks: dict[str, Any] = field(default_factory=dict)
    protected_mask: Any | None = None
    branch_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.protected_mask is None:
            self.protected_mask = self.body_mask

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_fit": self.body_fit.to_dict(),
            "body_mask": self.body_mask,
            "branch_fits": {
                key: fit.to_dict() for key, fit in self.branch_fits.items()
            },
            "branch_masks": dict(self.branch_masks),
            "protected_mask": self.protected_mask,
            "branch_diagnostics": list(self.branch_diagnostics),
        }


@dataclass
class DetectionResult:
    """Typed boundary between crypt detection and graph construction."""

    detections: list[dict[str, Any]]
    barriers: BarrierStageResult
    config: DetectionConfig
    diagnostics: dict[str, Any] = field(default_factory=dict)
    detection_mesh: Any | None = None

    @property
    def attachment_projections(self) -> list[dict[str, Any]]:
        return list(self.diagnostics.get("attachment_projections", []))

    @property
    def failed_attachments(self) -> list[dict[str, Any]]:
        return list(self.diagnostics.get("attachment_projection_failures", []))


@dataclass
class SkeletonizationResult:
    """Output of crypt-detection-based skeletonization for one organoid."""

    graph: SkeletonGraph
    detections: list[dict[str, Any]]
    barriers: BarrierStageResult | None = None
    intermediates: dict[str, Any] = field(default_factory=dict)
    config: SkeletonizationConfig = field(default_factory=SkeletonizationConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    mesh: Any | None = None
    geodesic_fn: Any | None = field(default=None, repr=False)

    def to_node_dataframe(self):
        return self.graph.to_node_dataframe()

    def to_edge_dataframe(self):
        return self.graph.to_edge_dataframe()

    def to_primitive_dataframe(self):
        return self.graph.to_primitive_dataframe()

    @property
    def failed_attachments(self) -> list[dict[str, Any]]:
        return list(self.intermediates.get("attachment_projection_failures", []))

    def to_dict(self, *, include_intermediates: bool = False) -> dict[str, Any]:
        data = {
            "graph": self.graph.to_dict(),
            "detections": self.detections,
            "barriers": None if self.barriers is None else self.barriers.to_dict(),
            "config": self.config.to_dict(),
            "metadata": dict(self.metadata),
        }
        if include_intermediates:
            data["intermediates"] = self.intermediates
        return data


@dataclass
class PrimitiveFitResult:
    """Primitive fitting output for one skeletonized organoid."""

    graph: SkeletonGraph
    components: dict[str, Any]
    attachments: dict[str, Any] = field(default_factory=dict)
    config: PrimitiveFitConfig = field(default_factory=PrimitiveFitConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    skeleton: SkeletonizationResult | None = None

    def to_dataframe(self):
        return self.graph.to_primitive_dataframe()

    @property
    def mesh(self):
        return None if self.skeleton is None else self.skeleton.mesh


@dataclass
class BlendResult:
    """Visualization-only blends generated from fitted skeleton primitives."""

    graph: SkeletonGraph
    blend_attachments: dict[str, Any] = field(default_factory=dict)
    config: BlendConfig = field(default_factory=BlendConfig)
    primitive_result: PrimitiveFitResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mesh(self):
        return None if self.primitive_result is None else self.primitive_result.mesh

    def to_dict(self) -> dict[str, Any]:
        return {
            "blend_attachments": {
                key: attachment.to_dict()
                for key, attachment in self.blend_attachments.items()
            },
            "config": self.config.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass
class OrganoidShapeResult:
    """Container for one organoid through skeletonization and primitive fitting."""

    skeleton: SkeletonizationResult
    primitives: PrimitiveFitResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def graph(self) -> SkeletonGraph:
        return self.skeleton.graph

    @property
    def detections(self) -> list[dict[str, Any]]:
        return self.skeleton.detections

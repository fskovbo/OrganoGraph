"""Result objects passed between skeleton, primitive, and future blending stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from organograph.skeleton.config import PrimitiveFitConfig, SkeletonizationConfig
from organograph.skeleton.datatypes import SkeletonGraph


@dataclass
class SkeletonizationResult:
    """Output of crypt-detection-based skeletonization for one organoid."""

    graph: SkeletonGraph
    detections: list[dict[str, Any]]
    intermediates: dict[str, Any] = field(default_factory=dict)
    config: SkeletonizationConfig = field(default_factory=SkeletonizationConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    mesh: Any | None = None

    def to_node_dataframe(self):
        return self.graph.to_node_dataframe()

    def to_edge_dataframe(self):
        return self.graph.to_edge_dataframe()

    def to_primitive_dataframe(self):
        return self.graph.to_primitive_dataframe()

    def to_dict(self, *, include_intermediates: bool = False) -> dict[str, Any]:
        data = {
            "graph": self.graph.to_dict(),
            "detections": self.detections,
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

"""Primitive abstractions for biology-aware organoid skeletons.

The skeleton graph stores straight-line topology.  Fitted primitives are an
optional layer attached to nodes, edges, or graph-level biological components
such as a whole crypt path.  The containers here deliberately keep parameters
generic so ellipsoids, superellipsoids, tapered tubes, and later implicit
representations can share the same serialization path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _jsonify(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


@dataclass
class Primitive:
    """Generic geometric primitive parameter container."""

    primitive_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive_type": self.primitive_type,
            "parameters": _jsonify(self.parameters),
            "metadata": _jsonify(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Primitive | None":
        if data is None:
            return None
        return cls(
            primitive_type=str(data["primitive_type"]),
            parameters=dict(data.get("parameters", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class PrimitiveFit:
    """Result of fitting a primitive to a mesh component.

    ``parameters`` are fitted degrees of freedom.  ``derived_parameters`` are
    descriptors computed from the primitive/skeleton, such as bend angle or
    constriction ratio, and should not be treated as independently fitted.
    """

    primitive_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    fit_error: float | None = None
    residuals: dict[str, Any] = field(default_factory=dict)
    derived_parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive_type": self.primitive_type,
            "parameters": _jsonify(self.parameters),
            "fit_error": _jsonify(self.fit_error),
            "residuals": _jsonify(self.residuals),
            "derived_parameters": _jsonify(self.derived_parameters),
            "metadata": _jsonify(self.metadata),
        }

    def to_attachment(
        self,
        *,
        attachment_type: str | None = None,
        attachment_id: str | None = None,
        target_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "PrimitiveAttachment":
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return PrimitiveAttachment(
            primitive_type=self.primitive_type,
            parameters=dict(self.parameters),
            fit_error=self.fit_error,
            residuals=dict(self.residuals),
            derived_parameters=dict(self.derived_parameters),
            metadata=merged_metadata,
            attachment_type=attachment_type,
            attachment_id=attachment_id,
            target_ids=list(target_ids or []),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PrimitiveFit | None":
        if data is None:
            return None
        return cls(
            primitive_type=str(data["primitive_type"]),
            parameters=dict(data.get("parameters", {})),
            fit_error=data.get("fit_error"),
            residuals=dict(data.get("residuals", {})),
            derived_parameters=dict(data.get("derived_parameters", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class PrimitiveAttachment:
    """Fitted primitive attached to a skeleton target."""

    primitive_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    fit_error: float | None = None
    residuals: dict[str, Any] = field(default_factory=dict)
    derived_parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    attachment_type: str | None = None
    attachment_id: str | None = None
    target_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive_type": self.primitive_type,
            "parameters": _jsonify(self.parameters),
            "fit_error": _jsonify(self.fit_error),
            "residuals": _jsonify(self.residuals),
            "derived_parameters": _jsonify(self.derived_parameters),
            "metadata": _jsonify(self.metadata),
            "attachment_type": self.attachment_type,
            "attachment_id": self.attachment_id,
            "target_ids": _jsonify(self.target_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PrimitiveAttachment | None":
        if data is None:
            return None
        return cls(
            primitive_type=str(data["primitive_type"]),
            parameters=dict(data.get("parameters", {})),
            fit_error=data.get("fit_error"),
            residuals=dict(data.get("residuals", {})),
            derived_parameters=dict(data.get("derived_parameters", {})),
            metadata=dict(data.get("metadata", {})),
            attachment_type=data.get("attachment_type"),
            attachment_id=data.get("attachment_id"),
            target_ids=list(data.get("target_ids", [])),
        )

"""Data containers for visualization-only primitive blending."""

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
class BlendAttachment:
    """A non-biological visual blend between already fitted primitives."""

    blend_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    attachment_id: str | None = None
    target_ids: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "blend_type": self.blend_type,
            "attachment_id": self.attachment_id,
            "target_ids": _jsonify(self.target_ids),
            "parameters": _jsonify(self.parameters),
            "diagnostics": _jsonify(self.diagnostics),
            "metadata": _jsonify(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BlendAttachment | None":
        if data is None:
            return None
        return cls(
            blend_type=str(data["blend_type"]),
            attachment_id=data.get("attachment_id"),
            target_ids=list(data.get("target_ids", [])),
            parameters=dict(data.get("parameters", {})),
            diagnostics=dict(data.get("diagnostics", {})),
            metadata=dict(data.get("metadata", {})),
        )

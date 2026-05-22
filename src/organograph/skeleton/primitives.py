"""Placeholder abstractions for future geometric primitive attachments.

The skeleton graph itself only stores straight-line topology.  Primitive
fitting, such as ellipsoids for the organoid body or tapered tubes for crypt
paths, is intentionally left for a later layer.
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
class PrimitiveAttachment:
    """Lightweight container for a future fitted primitive.

    Parameters are kept generic on purpose so the skeleton package does not
    hardcode any one primitive family.
    """

    primitive_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    fit_error: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive_type": self.primitive_type,
            "parameters": _jsonify(self.parameters),
            "fit_error": _jsonify(self.fit_error),
            "metadata": _jsonify(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PrimitiveAttachment | None":
        if data is None:
            return None
        return cls(
            primitive_type=str(data["primitive_type"]),
            parameters=dict(data.get("parameters", {})),
            fit_error=data.get("fit_error"),
            metadata=dict(data.get("metadata", {})),
        )

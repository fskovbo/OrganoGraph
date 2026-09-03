"""Data structures for biology-aware organoid skeleton graphs.

These classes represent compact skeletons derived from crypt detections, not
generic medial axes. Graph edges are straight segments. A crypt-center node
marks the volume center of its fitted tube primitive, while that primitive uses
a constrained smooth centerline between attachment and tip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from organograph.skeleton.primitives import PrimitiveAttachment


NODE_TYPES = {
    "body",
    "neck",
    "attachment",
    "constriction",  # Historical exports only; current fits keep this in the radius profile.
    "crypt",
    "bend",
    "branch",
    "tip",
}


def _jsonify(value: Any) -> Any:
    """Convert common numpy objects into JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, set):
        return [_jsonify(v) for v in sorted(value, key=lambda x: str(x))]
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _as_position(position: Iterable[float]) -> np.ndarray:
    arr = np.asarray(position, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"position must be a 3-vector, got shape {arr.shape}")
    return arr


@dataclass
class SkeletonNode:
    """Node in an organoid skeleton graph."""

    node_id: str
    node_type: str
    position: np.ndarray
    crypt_id: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    primitive_attachment: PrimitiveAttachment | None = None

    def __post_init__(self) -> None:
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"Unknown skeleton node_type {self.node_type!r}")
        self.node_id = str(self.node_id)
        self.position = _as_position(self.position)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "position": self.position.tolist(),
            "crypt_id": self.crypt_id,
            "metadata": _jsonify(self.metadata),
            "primitive_attachment": (
                self.primitive_attachment.to_dict()
                if self.primitive_attachment is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkeletonNode":
        return cls(
            node_id=str(data["node_id"]),
            node_type=str(data["node_type"]),
            position=np.asarray(data["position"], dtype=float),
            crypt_id=data.get("crypt_id"),
            metadata=dict(data.get("metadata", {})),
            primitive_attachment=PrimitiveAttachment.from_dict(
                data.get("primitive_attachment")
            ),
        )


@dataclass
class SkeletonEdge:
    """Straight segment between two skeleton nodes."""

    edge_id: str
    source: str
    target: str
    edge_type: str = "skeleton"
    crypt_id: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    primitive_attachment: PrimitiveAttachment | None = None

    def __post_init__(self) -> None:
        self.edge_id = str(self.edge_id)
        self.source = str(self.source)
        self.target = str(self.target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "crypt_id": self.crypt_id,
            "metadata": _jsonify(self.metadata),
            "primitive_attachment": (
                self.primitive_attachment.to_dict()
                if self.primitive_attachment is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkeletonEdge":
        return cls(
            edge_id=str(data["edge_id"]),
            source=str(data["source"]),
            target=str(data["target"]),
            edge_type=str(data.get("edge_type", "skeleton")),
            crypt_id=data.get("crypt_id"),
            metadata=dict(data.get("metadata", {})),
            primitive_attachment=PrimitiveAttachment.from_dict(
                data.get("primitive_attachment")
            ),
        )


@dataclass
class SkeletonGraph:
    """Small directed graph for a biology-aware organoid skeleton."""

    nodes: dict[str, SkeletonNode] = field(default_factory=dict)
    edges: dict[str, SkeletonEdge] = field(default_factory=dict)
    primitive_attachments: dict[str, PrimitiveAttachment] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    coordinate_frame: dict[str, Any] = field(default_factory=dict)

    def add_node(
        self,
        node_id: str,
        node_type: str,
        position: Iterable[float],
        *,
        crypt_id: str | int | None = None,
        metadata: dict[str, Any] | None = None,
        primitive_attachment: PrimitiveAttachment | None = None,
    ) -> SkeletonNode:
        node_id = str(node_id)
        if node_id in self.nodes:
            raise ValueError(f"Duplicate skeleton node id {node_id!r}")
        node = SkeletonNode(
            node_id=node_id,
            node_type=node_type,
            position=np.asarray(position, dtype=float),
            crypt_id=crypt_id,
            metadata=dict(metadata or {}),
            primitive_attachment=primitive_attachment,
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        edge_id: str,
        source: str,
        target: str,
        *,
        edge_type: str = "skeleton",
        crypt_id: str | int | None = None,
        metadata: dict[str, Any] | None = None,
        primitive_attachment: PrimitiveAttachment | None = None,
    ) -> SkeletonEdge:
        edge_id = str(edge_id)
        source = str(source)
        target = str(target)
        if edge_id in self.edges:
            raise ValueError(f"Duplicate skeleton edge id {edge_id!r}")
        if source not in self.nodes:
            raise KeyError(f"Unknown source node {source!r}")
        if target not in self.nodes:
            raise KeyError(f"Unknown target node {target!r}")
        edge = SkeletonEdge(
            edge_id=edge_id,
            source=source,
            target=target,
            edge_type=edge_type,
            crypt_id=crypt_id,
            metadata=dict(metadata or {}),
            primitive_attachment=primitive_attachment,
        )
        self.edges[edge_id] = edge
        return edge

    def node(self, node_id: str) -> SkeletonNode:
        return self.nodes[str(node_id)]

    def edge(self, edge_id: str) -> SkeletonEdge:
        return self.edges[str(edge_id)]

    def add_primitive_attachment(
        self,
        attachment_id: str,
        primitive_attachment: PrimitiveAttachment,
    ) -> PrimitiveAttachment:
        """Attach a fitted primitive to a graph-level target.

        Use this for primitives that belong to a biological component or path,
        for example a tapered tube fitted to a whole crypt path.
        """
        attachment_id = str(attachment_id)
        if attachment_id in self.primitive_attachments:
            raise ValueError(f"Duplicate primitive attachment id {attachment_id!r}")
        primitive_attachment.attachment_id = (
            primitive_attachment.attachment_id or attachment_id
        )
        self.primitive_attachments[attachment_id] = primitive_attachment
        return primitive_attachment

    def nodes_for_crypt(
        self,
        crypt_id: str | int,
        *,
        node_type: str | None = None,
    ) -> list[SkeletonNode]:
        nodes = [node for node in self.nodes.values() if node.crypt_id == crypt_id]
        if node_type is not None:
            nodes = [node for node in nodes if node.node_type == node_type]
        return nodes

    def edges_for_crypt(
        self,
        crypt_id: str | int,
        *,
        include_body_edge: bool = True,
    ) -> list[SkeletonEdge]:
        out = [edge for edge in self.edges.values() if edge.crypt_id == crypt_id]
        if not include_body_edge:
            out = [
                edge
                for edge in out
                if self.nodes[edge.source].node_type != "body"
                and self.nodes[edge.target].node_type != "body"
            ]
        return out

    def body_node(self) -> SkeletonNode:
        body_nodes = [node for node in self.nodes.values() if node.node_type == "body"]
        if len(body_nodes) != 1:
            raise ValueError(f"Expected exactly one body node, found {len(body_nodes)}")
        return body_nodes[0]

    def crypt_ids(self) -> list[str | int]:
        ids = {node.crypt_id for node in self.nodes.values() if node.crypt_id is not None}
        ids.update(edge.crypt_id for edge in self.edges.values() if edge.crypt_id is not None)
        return sorted(ids, key=lambda x: str(x))

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "primitive_attachments": [
                attachment.to_dict()
                for attachment in self.primitive_attachments.values()
            ],
            "metadata": _jsonify(self.metadata),
            "coordinate_frame": _jsonify(self.coordinate_frame),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkeletonGraph":
        graph = cls(
            metadata=dict(data.get("metadata", {})),
            coordinate_frame=dict(data.get("coordinate_frame", {})),
        )
        for node_data in data.get("nodes", []):
            node = SkeletonNode.from_dict(node_data)
            graph.nodes[node.node_id] = node
        for edge_data in data.get("edges", []):
            edge = SkeletonEdge.from_dict(edge_data)
            if edge.source not in graph.nodes or edge.target not in graph.nodes:
                raise ValueError(f"Edge {edge.edge_id!r} references missing node")
            graph.edges[edge.edge_id] = edge
        for attachment_data in data.get("primitive_attachments", []):
            attachment = PrimitiveAttachment.from_dict(attachment_data)
            if attachment is None:
                continue
            attachment_id = attachment.attachment_id or f"primitive_{len(graph.primitive_attachments)}"
            graph.primitive_attachments[str(attachment_id)] = attachment
        return graph

    def to_node_dataframe(self):
        """Return nodes as a pandas DataFrame."""
        import pandas as pd

        rows = []
        for node in self.nodes.values():
            rows.append(
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "crypt_id": node.crypt_id,
                    "x": float(node.position[0]),
                    "y": float(node.position[1]),
                    "z": float(node.position[2]),
                    "metadata": _jsonify(node.metadata),
                    "primitive_type": (
                        node.primitive_attachment.primitive_type
                        if node.primitive_attachment is not None
                        else None
                    ),
                }
            )
        return pd.DataFrame(rows)

    def to_primitive_dataframe(self):
        """Return graph-level primitive attachments as a pandas DataFrame."""
        import pandas as pd

        rows = []
        for attachment_id, attachment in self.primitive_attachments.items():
            row = {
                "attachment_id": attachment_id,
                "attachment_type": attachment.attachment_type,
                "primitive_type": attachment.primitive_type,
                "target_ids": _jsonify(attachment.target_ids),
                "fit_error": attachment.fit_error,
                "metadata": _jsonify(attachment.metadata),
                "derived_parameters": _jsonify(attachment.derived_parameters),
            }
            for key, value in attachment.parameters.items():
                if np.isscalar(value):
                    row[key] = value
                else:
                    row[key] = _jsonify(value)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_edge_dataframe(self):
        """Return straight skeleton edges as a pandas DataFrame."""
        import pandas as pd

        rows = []
        for edge in self.edges.values():
            rows.append(
                {
                    "edge_id": edge.edge_id,
                    "source": edge.source,
                    "target": edge.target,
                    "edge_type": edge.edge_type,
                    "crypt_id": edge.crypt_id,
                    "metadata": _jsonify(edge.metadata),
                    "primitive_type": (
                        edge.primitive_attachment.primitive_type
                        if edge.primitive_attachment is not None
                        else None
                    ),
                }
            )
        return pd.DataFrame(rows)

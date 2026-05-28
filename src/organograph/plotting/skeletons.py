"""Visualization helpers for organoid skeleton graphs."""

from __future__ import annotations

import numpy as np


NODE_COLORS = {
    "body": "#4c78a8",
    "neck": "#f58518",
    "crypt": "#72b7b2",
    "bend": "#54a24b",
    "branch": "#b279a2",
    "tip": "#e45756",
}

EDGE_COLORS = {
    "body_to_neck": "#7f7f7f",
    "neck_to_crypt": "#222222",
    "crypt_to_tip": "#222222",
    "neck_to_tip": "#222222",
    "neck_to_bend": "#222222",
    "bend_to_crypt": "#222222",
    "bend_to_tip": "#222222",
    "neck_to_branch": "#222222",
    "branch_to_neck": "#222222",
    "branch_to_tip": "#222222",
    "skeleton": "#222222",
}


def _node_arrays(graph):
    nodes = list(graph.nodes.values())
    xyz = np.vstack([node.position for node in nodes]) if nodes else np.empty((0, 3))
    return nodes, xyz


def _plotly_skeleton_traces(
    graph,
    *,
    show_node_labels=False,
    node_size=6,
    edge_width=5,
):
    import plotly.graph_objects as go

    traces = []
    for edge in graph.edges.values():
        p0 = graph.node(edge.source).position
        p1 = graph.node(edge.target).position
        traces.append(
            go.Scatter3d(
                x=[p0[0], p1[0]],
                y=[p0[1], p1[1]],
                z=[p0[2], p1[2]],
                mode="lines",
                line=dict(
                    color=EDGE_COLORS.get(edge.edge_type, EDGE_COLORS["skeleton"]),
                    width=float(edge_width),
                ),
                name=edge.edge_type,
                hovertext=edge.edge_id,
                showlegend=False,
            )
        )

    for node_type, color in NODE_COLORS.items():
        nodes = [node for node in graph.nodes.values() if node.node_type == node_type]
        if not nodes:
            continue
        xyz = np.vstack([node.position for node in nodes])
        labels = [node.node_id for node in nodes]
        traces.append(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers+text" if show_node_labels else "markers",
                marker=dict(size=float(node_size), color=color),
                text=labels if show_node_labels else None,
                textposition="top center",
                name=node_type,
                hovertext=labels,
            )
        )
    return traces


def plot_skeleton_3d(
    graph,
    ax=None,
    *,
    backend="mpl3d",
    show_node_labels=False,
    node_size=40,
    edge_width=2.0,
    fig_size=(7, 6),
    camera_eye=None,
):
    """Plot straight skeleton edges and typed nodes in 3D."""
    backend = str(backend).lower()
    if backend not in ("mpl3d", "plotly"):
        raise ValueError("backend must be 'mpl3d' or 'plotly'")

    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure(
            data=_plotly_skeleton_traces(
                graph,
                show_node_labels=show_node_labels,
                node_size=node_size,
                edge_width=edge_width,
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            )
        )
        if camera_eye is None:
            camera_eye = dict(x=1.05, y=1.05, z=0.75)
        fig.update_layout(scene_camera=dict(eye=camera_eye))
        return fig

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    plot_skeleton_edges(graph, ax=ax, backend="mpl3d", linewidth=edge_width)
    plot_skeleton_nodes(
        graph,
        ax=ax,
        backend="mpl3d",
        show_node_labels=show_node_labels,
        node_size=node_size,
    )
    ax.set_axis_off()
    return fig


def plot_skeleton_nodes(
    graph,
    ax=None,
    *,
    backend="mpl3d",
    show_node_labels=False,
    node_size=40,
):
    """Plot typed skeleton nodes."""
    backend = str(backend).lower()
    if backend != "mpl3d":
        raise ValueError("plot_skeleton_nodes currently supports backend='mpl3d'")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    for node_type, color in NODE_COLORS.items():
        nodes = [node for node in graph.nodes.values() if node.node_type == node_type]
        if not nodes:
            continue
        xyz = np.vstack([node.position for node in nodes])
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=float(node_size),
            color=color,
            label=node_type,
            depthshade=True,
        )
        if show_node_labels:
            for node in nodes:
                ax.text(
                    node.position[0],
                    node.position[1],
                    node.position[2],
                    node.node_id,
                    fontsize=8,
                )
    return ax


def plot_skeleton_edges(
    graph,
    ax=None,
    *,
    backend="mpl3d",
    linewidth=2.0,
):
    """Plot straight skeleton edges."""
    backend = str(backend).lower()
    if backend != "mpl3d":
        raise ValueError("plot_skeleton_edges currently supports backend='mpl3d'")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    for edge in graph.edges.values():
        p0 = graph.node(edge.source).position
        p1 = graph.node(edge.target).position
        pts = np.vstack([p0, p1])
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=EDGE_COLORS.get(edge.edge_type, EDGE_COLORS["skeleton"]),
            linewidth=float(linewidth),
        )
    return ax


def plot_mesh_with_skeleton(
    vertices,
    faces,
    graph,
    *,
    backend="mpl3d",
    mesh_alpha=0.18,
    mesh_color="lightgray",
    show_node_labels=False,
    node_size=10,
    edge_width=2.5,
    fig_size=(7, 6),
    camera_eye=None,
):
    """Overlay a skeleton on a mesh."""
    backend = str(backend).lower()
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)

    if backend == "plotly":
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=mesh_color,
                opacity=float(mesh_alpha),
                name="mesh",
                hoverinfo="skip",
            )
        )
        for trace in _plotly_skeleton_traces(
            graph,
            show_node_labels=show_node_labels,
            node_size=node_size,
            edge_width=edge_width,
        ):
            fig.add_trace(trace)
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            )
        )
        if camera_eye is None:
            camera_eye = dict(x=1.05, y=1.05, z=0.75)
        fig.update_layout(scene_camera=dict(eye=camera_eye))
        return fig

    if backend != "mpl3d":
        raise ValueError("backend must be 'mpl3d' or 'plotly'")

    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection="3d")
    tri = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles=faces)
    ax.plot_trisurf(
        tri,
        vertices[:, 2],
        color=mesh_color,
        alpha=float(mesh_alpha),
        linewidth=0.0,
        edgecolor="none",
        shade=True,
    )
    plot_skeleton_edges(graph, ax=ax, backend="mpl3d", linewidth=edge_width)
    plot_skeleton_nodes(
        graph,
        ax=ax,
        backend="mpl3d",
        show_node_labels=show_node_labels,
        node_size=node_size,
    )
    ax.set_axis_off()
    return fig

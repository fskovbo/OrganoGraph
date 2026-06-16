"""Visualization helpers for organoid skeleton graphs."""

from __future__ import annotations

import numpy as np

from organograph.skeleton.primitive_geometry import (
    capped_tube_radius,
    polyline_lengths,
)


NODE_COLORS = {
    "body": "#4c78a8",
    "neck": "#f58518",
    "attachment": "#ffbf79",
    "constriction": "#d95f02",
    "crypt": "#72b7b2",
    "bend": "#54a24b",
    "branch": "#b279a2",
    "tip": "#e45756",
}

EDGE_COLORS = {
    "body_to_neck": "#7f7f7f",
    "body_to_attachment": "#7f7f7f",
    "attachment_to_constriction": "#222222",
    "attachment_to_crypt": "#222222",
    "attachment_to_bend": "#222222",
    "attachment_to_tip": "#222222",
    "constriction_to_crypt": "#222222",
    "constriction_to_bend": "#222222",
    "constriction_to_tip": "#222222",
    "attachment_to_branch": "#222222",
    "constriction_to_branch": "#222222",
    "branch_to_attachment": "#222222",
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

PRIMITIVE_COLORS = {
    "ellipsoid": "#4c78a8",
    "superellipsoid_placeholder": "#4c78a8",
    "asymmetric_superellipsoid": "#4c78a8",
    "tapered_capped_tube": "#72b7b2",
    "straight_cylinder": "#e6ab02",
}

SMOOTH_CENTERLINE_COLOR = "#008b8b"


def _smooth_centerlines(graph):
    seen = set()
    centerlines = []
    for attachment_id, attachment in graph.primitive_attachments.items():
        if attachment.primitive_type != "tapered_capped_tube":
            continue
        points = attachment.parameters.get("centerline_points")
        if points is None:
            continue
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
            continue
        key = str(attachment_id)
        if key in seen:
            continue
        seen.add(key)
        centerlines.append((key, points))
    return centerlines


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
    show_smooth_centerlines=True,
    smooth_centerline_width=7,
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

    if show_smooth_centerlines:
        for attachment_id, points in _smooth_centerlines(graph):
            traces.append(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="lines",
                    line=dict(
                        color=SMOOTH_CENTERLINE_COLOR,
                        width=float(smooth_centerline_width),
                    ),
                    name="smooth crypt centerline",
                    hovertext=attachment_id,
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
    show_smooth_centerlines=True,
    smooth_centerline_width=3.0,
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
                show_smooth_centerlines=show_smooth_centerlines,
                smooth_centerline_width=smooth_centerline_width,
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

    plot_skeleton_edges(
        graph,
        ax=ax,
        backend="mpl3d",
        linewidth=edge_width,
        show_smooth_centerlines=show_smooth_centerlines,
        smooth_centerline_width=smooth_centerline_width,
    )
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
    show_smooth_centerlines=True,
    smooth_centerline_width=3.0,
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
    if show_smooth_centerlines:
        for _, points in _smooth_centerlines(graph):
            ax.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color=SMOOTH_CENTERLINE_COLOR,
                linewidth=float(smooth_centerline_width),
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
    show_smooth_centerlines=True,
    smooth_centerline_width=7,
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
            show_smooth_centerlines=show_smooth_centerlines,
            smooth_centerline_width=smooth_centerline_width,
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
    plot_skeleton_edges(
        graph,
        ax=ax,
        backend="mpl3d",
        linewidth=edge_width,
        show_smooth_centerlines=show_smooth_centerlines,
        smooth_centerline_width=smooth_centerline_width,
    )
    plot_skeleton_nodes(
        graph,
        ax=ax,
        backend="mpl3d",
        show_node_labels=show_node_labels,
        node_size=node_size,
    )
    ax.set_axis_off()
    return fig


def _ellipsoid_surface(parameters, *, n_u=32, n_v=16):
    center = np.asarray(parameters["center"], dtype=float)
    orientation = np.asarray(parameters["orientation"], dtype=float)
    axes = np.asarray(parameters["axis_lengths"], dtype=float)
    u = np.linspace(0.0, 2.0 * np.pi, int(n_u), endpoint=False)
    v = np.linspace(0.0, np.pi, int(n_v))
    uu, vv = np.meshgrid(u, v)
    local = np.stack(
        [
            axes[0] * np.cos(uu) * np.sin(vv),
            axes[1] * np.sin(uu) * np.sin(vv),
            axes[2] * np.cos(vv),
        ],
        axis=-1,
    )
    xyz = local.reshape(-1, 3) @ orientation.T + center[None, :]
    faces = []
    for i in range(int(n_v) - 1):
        for j in range(int(n_u)):
            a = i * int(n_u) + j
            b = i * int(n_u) + (j + 1) % int(n_u)
            c = (i + 1) * int(n_u) + j
            d = (i + 1) * int(n_u) + (j + 1) % int(n_u)
            faces.append([a, b, c])
            faces.append([b, d, c])
    return xyz, np.asarray(faces, dtype=np.int64)


def _signed_power(values, exponent):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.abs(values) ** float(exponent)


def _asymmetric_superellipsoid_surface(parameters, *, n_u=32, n_v=16):
    center = np.asarray(parameters["center"], dtype=float)
    orientation = np.asarray(parameters["orientation"], dtype=float)
    negative = np.asarray(parameters["axis_lengths_negative"], dtype=float)
    positive = np.asarray(parameters["axis_lengths_positive"], dtype=float)
    epsilon_1 = float(parameters["epsilon_1"])
    epsilon_2 = float(parameters["epsilon_2"])
    u = np.linspace(-np.pi, np.pi, int(n_u), endpoint=False)
    v = np.linspace(-0.5 * np.pi, 0.5 * np.pi, int(n_v))
    uu, vv = np.meshgrid(u, v)
    base = np.stack(
        [
            _signed_power(np.cos(vv), epsilon_1)
            * _signed_power(np.cos(uu), epsilon_2),
            _signed_power(np.cos(vv), epsilon_1)
            * _signed_power(np.sin(uu), epsilon_2),
            _signed_power(np.sin(vv), epsilon_1),
        ],
        axis=-1,
    )
    scales = np.where(base >= 0.0, positive[None, None, :], negative[None, None, :])
    local = base * scales
    xyz = local.reshape(-1, 3) @ orientation.T + center[None, :]
    faces = []
    for i in range(int(n_v) - 1):
        for j in range(int(n_u)):
            a = i * int(n_u) + j
            b = i * int(n_u) + (j + 1) % int(n_u)
            c = (i + 1) * int(n_u) + j
            d = (i + 1) * int(n_u) + (j + 1) % int(n_u)
            faces.extend(([a, b, c], [b, d, c]))
    return xyz, np.asarray(faces, dtype=np.int64)


def _polyline_point_at_s(centerline, s):
    centerline = np.asarray(centerline, dtype=float)
    lengths, cumulative, total = polyline_lengths(centerline)
    if total <= 1e-12 or centerline.shape[0] == 1:
        return centerline[0], np.array([1.0, 0.0, 0.0])
    target = float(np.clip(s, 0.0, 1.0)) * total
    i = int(np.searchsorted(cumulative, target, side="right") - 1)
    i = max(0, min(i, len(lengths) - 1))
    t = (target - cumulative[i]) / max(lengths[i], 1e-12)
    point = centerline[i] + t * (centerline[i + 1] - centerline[i])
    tangent = centerline[i + 1] - centerline[i]
    tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
    return point, tangent


def _tube_surface(parameters, *, n_s=32, n_theta=16):
    centerline = np.asarray(parameters["centerline_points"], dtype=float)
    radii = (
        float(parameters["r_neck"]),
        float(parameters["r_body"]),
        float(parameters.get("r_taper", parameters["r_tip"])),
    )
    body_s = float(parameters.get("s_body", 0.5))
    taper_start = float(
        parameters.get("s_taper", parameters.get("distal_taper_start", 0.85))
    )
    n_s = max(6, int(n_s))
    s_values = np.linspace(0.0, 1.0, n_s, endpoint=False)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)
    vertices = []
    previous_normal = None
    for s in s_values:
        center, tangent = _polyline_point_at_s(centerline, s)
        if previous_normal is None:
            ref = np.array([0.0, 0.0, 1.0])
            if abs(float(np.dot(ref, tangent))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            normal = np.cross(tangent, ref)
            normal = normal / max(np.linalg.norm(normal), 1e-12)
        else:
            normal = previous_normal - tangent * float(np.dot(previous_normal, tangent))
            normal = normal / max(np.linalg.norm(normal), 1e-12)
        binormal = np.cross(tangent, normal)
        binormal = binormal / max(np.linalg.norm(binormal), 1e-12)
        previous_normal = normal
        radius = float(
            capped_tube_radius(
                np.array([s]),
                *radii,
                body_s=body_s,
                taper_start=taper_start,
                constriction_s=parameters.get("s_constriction"),
                r_constriction=parameters.get("r_constriction"),
            )[0]
        )
        ring = [
            center + radius * (np.cos(a) * normal + np.sin(a) * binormal)
            for a in theta
        ]
        vertices.extend(ring)

    faces = []
    n_theta = int(n_theta)
    n_rings = len(s_values)
    for i in range(n_rings - 1):
        for j in range(n_theta):
            a = i * n_theta + j
            b = i * n_theta + (j + 1) % n_theta
            c = (i + 1) * n_theta + j
            d = (i + 1) * n_theta + (j + 1) % n_theta
            faces.append([a, b, c])
            faces.append([b, d, c])

    # Collapse the integrated taper to one vertex at the crypt-tip node.
    tip_index = len(vertices)
    vertices.append(centerline[-1])
    last_ring = (n_rings - 1) * n_theta
    for j in range(n_theta):
        faces.append(
            [
                last_ring + j,
                last_ring + (j + 1) % n_theta,
                tip_index,
            ]
        )
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def _cylinder_surface(parameters, *, n_s=16, n_theta=16):
    centerline = np.asarray(parameters["centerline_points"], dtype=float)
    start, end = centerline[0], centerline[-1]
    tangent = end - start
    length = float(np.linalg.norm(tangent))
    if length <= 1e-12:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
    tangent /= length
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, tangent))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    normal = np.cross(tangent, reference)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    binormal = np.cross(tangent, normal)
    radius = float(parameters["radius"])
    s_values = np.linspace(0.0, 1.0, max(2, int(n_s)))
    theta = np.linspace(0.0, 2.0 * np.pi, max(3, int(n_theta)), endpoint=False)
    vertices = []
    for s in s_values:
        center = start + s * (end - start)
        vertices.extend(
            center
            + radius * (np.cos(angle) * normal + np.sin(angle) * binormal)
            for angle in theta
        )
    faces = []
    n_theta = len(theta)
    for i in range(len(s_values) - 1):
        for j in range(n_theta):
            a = i * n_theta + j
            b = i * n_theta + (j + 1) % n_theta
            c = (i + 1) * n_theta + j
            d = (i + 1) * n_theta + (j + 1) % n_theta
            faces.extend(([a, b, c], [b, d, c]))
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def _primitive_mesh(attachment, *, n_s=32, n_theta=16):
    primitive_type = attachment.primitive_type
    if primitive_type in {"ellipsoid", "superellipsoid_placeholder"}:
        return _ellipsoid_surface(attachment.parameters, n_u=n_theta * 2, n_v=n_s)
    if primitive_type == "asymmetric_superellipsoid":
        return _asymmetric_superellipsoid_surface(
            attachment.parameters,
            n_u=n_theta * 2,
            n_v=n_s,
        )
    if primitive_type == "tapered_capped_tube":
        return _tube_surface(attachment.parameters, n_s=n_s, n_theta=n_theta)
    if primitive_type == "straight_cylinder":
        return _cylinder_surface(attachment.parameters, n_s=n_s, n_theta=n_theta)
    return None, None


def _plotly_primitive_traces(
    graph,
    *,
    primitive_alpha=0.35,
    n_s=32,
    n_theta=16,
):
    import plotly.graph_objects as go

    attachments = []
    for node in graph.nodes.values():
        if node.primitive_attachment is not None:
            attachments.append((node.node_id, node.primitive_attachment))
    for edge in graph.edges.values():
        if edge.primitive_attachment is not None:
            attachments.append((edge.edge_id, edge.primitive_attachment))
    attachments.extend(graph.primitive_attachments.items())

    traces = []
    for attachment_id, attachment in attachments:
        xyz, faces = _primitive_mesh(attachment, n_s=n_s, n_theta=n_theta)
        if xyz is None or faces is None or xyz.size == 0 or faces.size == 0:
            continue
        color = PRIMITIVE_COLORS.get(attachment.primitive_type, "#72b7b2")
        traces.append(
            go.Mesh3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=float(primitive_alpha),
                name=str(attachment_id),
                hovertext=str(attachment_id),
            )
        )
    return traces


def plot_primitives_3d(
    graph,
    *,
    backend="plotly",
    primitive_alpha=0.35,
    n_s=32,
    n_theta=16,
):
    """Plot primitive attachments without mesh or skeleton."""
    backend = str(backend).lower()
    if backend != "plotly":
        raise ValueError("plot_primitives_3d currently supports backend='plotly'")
    import plotly.graph_objects as go

    fig = go.Figure(
        data=_plotly_primitive_traces(
            graph,
            primitive_alpha=primitive_alpha,
            n_s=n_s,
            n_theta=n_theta,
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
    return fig


def plot_mesh_with_skeleton_and_primitives(
    vertices,
    faces,
    graph,
    *,
    backend="plotly",
    mesh_alpha=0.14,
    primitive_alpha=0.35,
    mesh_color="lightgray",
    show_node_labels=False,
    node_size=7,
    edge_width=3.0,
    camera_eye=None,
):
    """Overlay mesh, skeleton, and fitted primitive attachments."""
    backend = str(backend).lower()
    if backend != "plotly":
        raise ValueError(
            "plot_mesh_with_skeleton_and_primitives currently supports backend='plotly'"
        )
    fig = plot_mesh_with_skeleton(
        vertices,
        faces,
        graph,
        backend="plotly",
        mesh_alpha=mesh_alpha,
        mesh_color=mesh_color,
        show_node_labels=show_node_labels,
        node_size=node_size,
        edge_width=edge_width,
        camera_eye=camera_eye,
    )
    for trace in _plotly_primitive_traces(graph, primitive_alpha=primitive_alpha):
        fig.add_trace(trace)
    return fig

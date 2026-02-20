import numpy as np

# Plotly is optional if users only want static plots; import lazily where possible.
import plotly.graph_objects as go
from plotly import colors as pc

from organograph.graph.access import graph_get, graph_get_marker_bin


# =============================================================================
# Camera / view helpers
# =============================================================================

DEFAULT_VIEW = dict(
    azim=-135,          # degrees
    elev=25,            # degrees
    roll=0,             # degrees (matplotlib only; plotly ignores)
    projection="orthographic",  # "orthographic" or "perspective"
)

def _normalize_view(view):
    v = dict(DEFAULT_VIEW)
    if view:
        v.update(view)
    proj = str(v.get("projection", "orthographic")).lower()
    if proj not in ("orthographic", "perspective"):
        raise ValueError("view['projection'] must be 'orthographic' or 'perspective'")
    v["projection"] = proj
    return v


def _apply_plotly_view(fig, view=None, camera_eye=None, hide_axes=True, transparent_bg=True):
    if camera_eye is not None:
        fig.update_layout(scene_camera=dict(eye=camera_eye))
    elif view is not None:
        # Plotly doesn't have a true orthographic option in the same way Matplotlib does.
        # We still map azim/elev to a reasonable eye vector for a reproducible view.
        v = _normalize_view(view)
        az = np.deg2rad(float(v["azim"]))
        el = np.deg2rad(float(v["elev"]))
        r = 1.8  # distance; tweak for framing
        eye = dict(
            x=float(r * np.cos(el) * np.cos(az)),
            y=float(r * np.cos(el) * np.sin(az)),
            z=float(r * np.sin(el)),
        )
        fig.update_layout(scene_camera=dict(eye=eye))

    if hide_axes or transparent_bg:
        scene_updates = {}
        if hide_axes:
            scene_updates.update(
                dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                )
            )
        if transparent_bg:
            scene_updates["bgcolor"] = "rgba(0,0,0,0)"
        fig.update_layout(scene=scene_updates)
    return fig


# =============================================================================
# Core plotting
# =============================================================================

def plot_organoid_graph(
    graph,
    node_values=None,
    *,
    backend="plotly",                 # "plotly" | "mpl3d" | "orthoprojs"
    node_size=5,
    edge_width=1.0,
    colorscale="RdBu_r",
    center_at_zero=True,
    loops=None,
    loop_width=6.0,
    # view / styling
    view=None,                        # dict(azim=..., elev=..., projection=...)
    camera_eye=None,                  # plotly-only override
    hide_axes=True,
    transparent_bg=True,
    fig_size=(750, 550),              # plotly: px, matplotlib: inches via dpi scaling
    dpi=120,
    edges="thin",                     # mpl backends: "none" | "thin"
):
    """
    Plot a cell adjacency graph in 3D.

    Coordinates are read from node field "centroid" using graph_get(graph, "centroid").

    Backends
    --------
    backend="plotly"
        Interactive Plotly 3D plot (current behavior).
    backend="mpl3d"
        Static Matplotlib 3D plot (much lighter than Plotly; fixed view).
    backend="orthoprojs"
        Static Matplotlib 3-panel orthographic projections (XY, XZ, YZ). This is
        not truly 3D, but it's useful for QC and avoids hidden-side issues when
        combined with multiple views.

    Returns
    -------
    Plotly: plotly.graph_objects.Figure
    Matplotlib: matplotlib.figure.Figure
    """
    backend = str(backend).lower()
    if backend not in ("plotly", "mpl3d", "orthoprojs"):
        raise ValueError("backend must be 'plotly', 'mpl3d', or 'orthoprojs'")

    centroids = graph_get(graph, "centroid", dtype=float)
    if centroids.ndim != 2 or centroids.shape[1] < 3:
        raise ValueError('"centroid" must be an (N,3) array-like node field.')
    XYZ = centroids[:, :3]

    # --- prepare node colors/values ---
    marker_kwargs_plotly = dict(size=node_size, line=dict(width=0))
    node_colors_mpl = None
    node_c_mpl = None
    node_cmap_mpl = None
    node_norm_mpl = None

    if node_values is None:
        marker_kwargs_plotly["color"] = "gray"
        node_colors_mpl = "0.5"
    else:
        arr = np.asarray(list(node_values))
        if np.issubdtype(arr.dtype, np.number):
            # numeric scalar -> colormap
            finite = np.isfinite(arr)
            if not np.any(finite):
                marker_kwargs_plotly["color"] = "blue"
                node_colors_mpl = "C0"
            else:
                if center_at_zero:
                    vmax = float(np.nanmax(np.abs(arr[finite])))
                    vmax = 1.0 if vmax == 0.0 else vmax
                    marker_kwargs_plotly.update(
                        dict(color=arr, colorscale=colorscale, cmin=-vmax, cmax=+vmax, cmid=0.0)
                    )
                    node_c_mpl = arr
                    node_norm_mpl = (-vmax, vmax)
                else:
                    vmin = float(np.nanmin(arr[finite]))
                    vmax = float(np.nanmax(arr[finite]))
                    if vmin == vmax:
                        vmin, vmax = vmin - 1.0, vmin + 1.0
                    marker_kwargs_plotly.update(
                        dict(color=arr, colorscale=colorscale, cmin=vmin, cmax=vmax)
                    )
                    node_c_mpl = arr
                    node_norm_mpl = (vmin, vmax)
        else:
            # categorical CSS/hex colors
            marker_kwargs_plotly["color"] = arr
            node_colors_mpl = arr

    if backend == "plotly":
        x, y, z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]

        node_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=marker_kwargs_plotly,
            hoverinfo="skip",
            showlegend=False,
        )

        # Edges
        edge_x, edge_y, edge_z = [], [], []
        if edge_width and edge_width > 0:
            for i, j in graph.edges():
                edge_x += [XYZ[i, 0], XYZ[j, 0], None]
                edge_y += [XYZ[i, 1], XYZ[j, 1], None]
                edge_z += [XYZ[i, 2], XYZ[j, 2], None]

        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(width=edge_width, color="rgba(120,120,120,0.75)"),
            hoverinfo="skip",
            showlegend=False,
        )

        traces = [edge_trace, node_trace]

        # Optional loops
        if loops:
            palette = [
                "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
                "#ff7f00", "#a65628", "#f781bf", "#999999",
            ]
            
            for i, poly in enumerate(loops):
                if poly is None:
                    continue
                poly = np.asarray(poly, dtype=int)
                if poly.size < 2:
                    continue
                if poly[0] != poly[-1]:
                    poly = np.concatenate([poly, poly[:1]])
                traces.append(
                    go.Scatter3d(
                        x=XYZ[poly, 0], y=XYZ[poly, 1], z=XYZ[poly, 2],
                        mode="lines",
                        line=dict(width=loop_width, color=palette[i % len(palette)]),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        fig = go.Figure(traces)
        fig.update_layout(scene=dict(aspectmode="data"))
        width, height = fig_size
        fig.update_layout(width=int(width), height=int(height))
        fig = _apply_plotly_view(fig, view=view, camera_eye=camera_eye, hide_axes=hide_axes, transparent_bg=transparent_bg)
        return fig

    # --- Matplotlib backends ---
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    v = _normalize_view(view)

    # Convert fig_size from px-ish to inches for matplotlib
    width_px, height_px = fig_size
    fig_w = float(width_px) / float(dpi)
    fig_h = float(height_px) / float(dpi)

    if backend == "mpl3d":
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        # projection type
        try:
            ax.set_proj_type("ortho" if v["projection"] == "orthographic" else "persp")
        except Exception:
            # older matplotlib: ignore
            pass

        # edges
        if edges != "none" and edge_width and edge_width > 0:
            segs = []
            for i, j in graph.edges():
                segs.append([XYZ[i], XYZ[j]])
            if segs:
                lc = Line3DCollection(segs, linewidths=float(edge_width) * 0.25, alpha=0.35)
                ax.add_collection3d(lc)

        # nodes
        if node_c_mpl is not None:
            vmin, vmax = node_norm_mpl
            ax.scatter(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], c=node_c_mpl, s=float(node_size), vmin=vmin, vmax=vmax)
        else:
            ax.scatter(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], c=node_colors_mpl, s=float(node_size))

        # loops
        if loops:
            for poly in loops:
                if poly is None:
                    continue
                poly = np.asarray(poly, dtype=int)
                if poly.size < 2:
                    continue
                if poly[0] != poly[-1]:
                    poly = np.concatenate([poly, poly[:1]])
                ax.plot(XYZ[poly, 0], XYZ[poly, 1], XYZ[poly, 2], linewidth=float(loop_width) * 0.25)

        ax.view_init(elev=float(v["elev"]), azim=float(v["azim"]))
        # roll support exists in newer mpl via ax.roll (not universal); ignore if missing
        if hide_axes:
            ax.set_axis_off()
        if transparent_bg:
            fig.patch.set_alpha(0.0)
            ax.set_facecolor((0, 0, 0, 0))
        fig.tight_layout()
        return fig

    # orthoprojs: 3 panels (XY, XZ, YZ)
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), dpi=dpi)
    planes = [("x", "y", (0, 1)), ("x", "z", (0, 2)), ("y", "z", (1, 2))]
    for ax, (a_name, b_name, (ia, ib)) in zip(axes, planes):
        # edges
        if edges != "none" and edge_width and edge_width > 0:
            segs2 = []
            for i, j in graph.edges():
                segs2.append([[XYZ[i, ia], XYZ[i, ib]], [XYZ[j, ia], XYZ[j, ib]]])
            if segs2:
                lc2 = LineCollection(segs2, linewidths=float(edge_width) * 0.25, alpha=0.35)
                ax.add_collection(lc2)

        # nodes
        if node_c_mpl is not None:
            vmin, vmax = node_norm_mpl
            ax.scatter(XYZ[:, ia], XYZ[:, ib], c=node_c_mpl, s=float(node_size), vmin=vmin, vmax=vmax)
        else:
            ax.scatter(XYZ[:, ia], XYZ[:, ib], c=node_colors_mpl, s=float(node_size))

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{a_name.upper()}{b_name.upper()}")
        if hide_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

    if transparent_bg:
        fig.patch.set_alpha(0.0)
        for ax in axes:
            ax.set_facecolor((0, 0, 0, 0))
    fig.tight_layout()
    return fig


def add_region_overlays(
    fig_or_ax,
    graph,
    regions,
    *,
    backend="plotly",          # "plotly" | "mpl3d" | "orthoprojs"
    colors=None,
    colorscale=None,
    size=7.0,
    name_prefix="region",
    alpha=0.35,
):
    """
    Overlay region highlights (e.g., crypts) on an existing plot.

    backend="plotly"
        fig_or_ax is a plotly Figure; adds Scatter3d traces.
    backend="mpl3d"
        fig_or_ax is a matplotlib Axes3D; draws additional scatter points.
    backend="orthoprojs"
        fig_or_ax is a matplotlib Figure with 3 axes; draws scatters on each panel.

    Parameters
    ----------
    graph : networkx.Graph
        Graph containing node field "centroid".
    regions : list[set[int]]
        Each set is a group of node indices to display.
    """
    backend = str(backend).lower()
    if backend not in ("plotly", "mpl3d", "orthoprojs"):
        raise ValueError("backend must be 'plotly', 'mpl3d', or 'orthoprojs'")

    XYZ = graph_get(graph, "centroid", dtype=float)[:, :3]

    nonempty_regions = [reg for reg in regions if reg and len(reg) > 0]
    k = len(nonempty_regions)
    if k == 0:
        return

    # palette
    if colors and len(colors) > 0:
        palette = list(colors)
    elif colorscale is not None:
        cs = pc.get_colorscale(colorscale) if isinstance(colorscale, str) else colorscale
        vals = [0.5] if k == 1 else [i / (k - 1) for i in range(k)]
        palette = [pc.sample_colorscale(cs, v)[0] for v in vals]
    else:
        palette = [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
            "#ff7f00", "#a65628", "#f781bf", "#999999",
        ]

    if backend in ("mpl3d", "orthoprojs"):
        palette = [_plotly_rgb_to_mpl(c) for c in palette]


    if backend == "plotly":
        fig = fig_or_ax
        for i, reg in enumerate(nonempty_regions):
            idx = np.fromiter(reg, dtype=int)
            fig.add_trace(
                go.Scatter3d(
                    x=XYZ[idx, 0], y=XYZ[idx, 1], z=XYZ[idx, 2],
                    mode="markers",
                    marker=dict(size=size, color=palette[i % len(palette)]),
                    name=f"{name_prefix} {i+1}",
                    showlegend=True,
                    hoverinfo="skip",
                    opacity=alpha,
                )
            )
        return

    # Matplotlib backends
    if backend == "mpl3d":
        ax = fig_or_ax
        for i, reg in enumerate(nonempty_regions):
            idx = np.fromiter(reg, dtype=int)
            ax.scatter(
                XYZ[idx, 0], XYZ[idx, 1], XYZ[idx, 2],
                s=float(size),
                c=palette[i % len(palette)],
                alpha=float(alpha),
                label=f"{name_prefix} {i+1}",
            )
        # Let user call legend() if they want; legends on 3D can be messy.
        return

    # orthoprojs: fig with 3 axes
    fig = fig_or_ax
    axes = fig.get_axes()
    if len(axes) < 3:
        raise ValueError("orthoprojs expects a matplotlib Figure with 3 subplots.")
    planes = [(0, 1), (0, 2), (1, 2)]
    for ax, (ia, ib) in zip(axes[:3], planes):
        for i, reg in enumerate(nonempty_regions):
            idx = np.fromiter(reg, dtype=int)
            ax.scatter(
                XYZ[idx, ia], XYZ[idx, ib],
                s=float(size),
                c=palette[i % len(palette)],
                alpha=float(alpha),
            )


def plot_graph_by_markers(
    graph,
    marker_map,
    *,
    backend="plotly",
    baseline_color="darkgray",
    node_size=5,
    edge_width=1.0,
    priority="last",
    add_legend=True,
    legend_baseline_name="other",
    # view / styling
    view=None,
    camera_eye=None,
    hide_axes=True,
    transparent_bg=True,
    fig_size=(750, 550),
    dpi=120,
    edges="thin",
):
    """
    Plot an organoid graph where node colors are assigned by marker positivity.

    marker_map entries can use marker index or marker name (string). If marker is
    a string, graph_get_marker_bin resolves it using G.graph['marker_names'].
    """
    nodes_n = graph.number_of_nodes()
    if nodes_n == 0:
        raise ValueError("Graph has no nodes.")

    # normalize marker_map
    mm = []
    for entry in marker_map:
        if isinstance(entry, dict):
            marker = entry.get("marker", entry.get("idx", None))
            color = entry.get("color", None)
            name = entry.get("name", None)
        else:
            marker, color, name = entry

        if marker is None or color is None:
            raise ValueError("Each marker_map entry must provide marker (idx or name) and color.")
        if name is None:
            name = f"marker[{marker}]"
        mm.append({"marker": marker, "color": str(color), "name": str(name)})

    colors = np.full(nodes_n, baseline_color, dtype=object)

    if priority not in ("last", "first"):
        raise ValueError('priority must be "last" or "first".')

    if priority == "last":
        for m in mm:
            pos = graph_get_marker_bin(graph, m["marker"])
            colors[pos == 1] = m["color"]
    else:
        for m in mm:
            pos = graph_get_marker_bin(graph, m["marker"])
            mask = (pos == 1) & (colors == baseline_color)
            colors[mask] = m["color"]

    fig = plot_organoid_graph(
        graph,
        node_values=colors,
        backend=backend,
        node_size=node_size,
        edge_width=edge_width,
        view=view,
        camera_eye=camera_eye,
        hide_axes=hide_axes,
        transparent_bg=transparent_bg,
        fig_size=fig_size,
        dpi=dpi,
        edges=edges,
    )

    backend_l = str(backend).lower()
    if add_legend and backend_l == "plotly":
        # only plotly legend; matplotlib legends can be turned on by user if desired
        for m in mm:
            fig.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode="markers",
                    marker=dict(size=8, color=m["color"]),
                    name=m["name"],
                    showlegend=True,
                )
            )
        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(size=8, color=baseline_color),
                name=legend_baseline_name,
                showlegend=True,
            )
        )
        width, height = fig_size
        fig.update_layout(width=int(width), height=int(height))

    return fig


import re

_RGB_RE = re.compile(r"rgb\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)")

def _plotly_rgb_to_mpl(c):
    """
    Convert Plotly 'rgb(r,g,b)' strings to Matplotlib '#RRGGBB'.
    Leave other formats unchanged.
    """
    if isinstance(c, str):
        m = _RGB_RE.fullmatch(c.strip())
        if m:
            r, g, b = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            return f"#{r:02x}{g:02x}{b:02x}"
    return c
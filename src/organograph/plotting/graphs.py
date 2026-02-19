import numpy as np
import plotly.graph_objects as go
import networkx as nx
from plotly import colors as pc


def plot_organoid_graph(
    centroids: np.ndarray,
    graph: nx.Graph,
    node_values: np.ndarray | None = None,
    *,
    node_size: int = 5,
    edge_width: float = 1.0,
    colorscale: str | list = "RdBu_r",
    center_at_zero: bool = True,
    loops: list[np.ndarray] | None = None,
    loop_width: float = 6.0,
) -> go.Figure:
    """
    Minimal 3D plot of a cell adjacency graph with optional loop overlays.

    - Nodes are colored by `node_values`. By default a diverging colorscale (RdBu_r)
      is used and centered at zero (set `center_at_zero=False` to disable).
    - Edges are drawn as a light wireframe; `edge_width` controls their thickness.
    - If `loops` (list of index arrays) is provided, each loop is drawn as a polyline.

    Parameters
    ----------
    centroids : (N,3) ndarray
        Coordinates of each cell (x,y,z).
    graph : networkx.Graph
        Adjacency graph over N nodes (0..N-1).
    node_values : array-like or None
        Per-node values for coloring. If None, nodes are a uniform blue.
        If numeric, a colorscale is applied. If strings, they are used as colors.
    node_size : int
        Marker size for nodes.
    edge_width : float
        Line width for edges.
    colorscale : str or list
        Plotly colorscale for numeric `node_values` (default "RdBu_r").
    center_at_zero : bool
        If True, color limits are symmetric about 0 for numeric `node_values`.
    loops : list[np.ndarray] or None
        Optional list of index arrays defining loops to draw.
    loop_width : float
        Line width for loop polylines.

    Returns
    -------
    go.Figure
        Plotly figure (not shown).
    """
    # Coordinates
    x, y, z = centroids[:, 0], centroids[:, 1], centroids[:, 2]

    # --- Node coloring ---
    marker_kwargs = dict(size=node_size, line=dict(width=0))
    if node_values is None:
        marker_kwargs["color"] = "gray"
        showscale = False
    else:
        arr = np.asarray(list(node_values))
        if np.issubdtype(arr.dtype, np.number):
            finite = np.isfinite(arr)
            if not np.any(finite):
                # fallback if all NaN/inf
                marker_kwargs["color"] = "blue"
                showscale = False
            else:
                if center_at_zero:
                    vmax = float(np.nanmax(np.abs(arr[finite])))
                    vmax = 1.0 if vmax == 0.0 else vmax
                    marker_kwargs.update(
                        dict(
                            color=arr,
                            colorscale=colorscale,
                            cmin=-vmax,
                            cmax=+vmax,
                            cmid=0.0,
                        )
                    )
                else:
                    vmin = float(np.nanmin(arr[finite]))
                    vmax = float(np.nanmax(arr[finite]))
                    if vmin == vmax:
                        vmin, vmax = vmin - 1.0, vmin + 1.0
                    marker_kwargs.update(
                        dict(
                            color=arr,
                            colorscale=colorscale,
                            cmin=vmin,
                            cmax=vmax,
                        )
                    )
                showscale = False  # keep minimal; user can add later
        else:
            # Categorical colors provided as strings
            marker_kwargs["color"] = arr
            showscale = False

    node_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=marker_kwargs,
        hoverinfo="skip",
        showlegend=False,
    )

    # --- Edge wireframe ---
    edge_x, edge_y, edge_z = [], [], []
    for i, j in graph.edges():
        edge_x += [centroids[i, 0], centroids[j, 0], None]
        edge_y += [centroids[i, 1], centroids[j, 1], None]
        edge_z += [centroids[i, 2], centroids[j, 2], None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(width=edge_width, color="rgba(120,120,120,0.75)"),
        hoverinfo="skip",
        showlegend=False,
    )

    traces = [edge_trace, node_trace]

    # --- Optional loops ---
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
                    x=centroids[poly, 0], y=centroids[poly, 1], z=centroids[poly, 2],
                    mode="lines",
                    line=dict(width=loop_width, color=palette[i % len(palette)]),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig = go.Figure(traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        )
    )
    # Do not call fig.show(); return for user control
    return fig


def add_region_overlays(
    fig: go.Figure,
    coords: np.ndarray,
    regions: list[set[int]],
    *,
    colors: list[str] | None = None,
    colorscale: str | list | None = None,
    size: float = 7.0,
    name_prefix: str = "region",
    alpha: float = 0.35,
) -> None:
    """
    Overlay translucent 3D marker clusters for arbitrary region sets.

    Parameters
    ----------
    fig : go.Figure
        Figure to which traces are added (modified in place).
    coords : (N,3) array
        Node coordinates (uses first 3 columns).
    regions : list[set[int]]
        Each set is a group of node indices to display as one trace.
    colors : list[str], optional
        Explicit CSS/hex/rgba colors to cycle through per region (qualitative).
        If provided, this takes precedence over `colorscale`.
    colorscale : str or list, optional
        Plotly colorscale name (e.g. "YlOrRd", "Blues") or a colorscale list.
        Used to *sample distinct colors* when `colors` is not given.
    size : float, default 7.0
        Marker size.
    name_prefix : str, default "region"
        Legend/trace name prefix; regions are numbered 1..K.
    alpha : float, default 0.35
        Per-trace marker opacity.
    """
    XYZ = coords[:, :3]

    # Non-empty regions only
    nonempty_regions = [reg for reg in regions if len(reg) > 0]
    k = len(nonempty_regions)
    if k == 0:
        return

    # Build a color palette
    if colors and len(colors) > 0:
        palette = list(colors)
    elif colorscale is not None:
        # Get a valid colorscale list
        cs = pc.get_colorscale(colorscale) if isinstance(colorscale, str) else colorscale
        # Evenly sample k distinct colors in [0, 1]
        vals: Iterable[float] = [0.5] if k == 1 else [i / (k - 1) for i in range(k)]
        palette = [pc.sample_colorscale(cs, v)[0] for v in vals]
    else:
        # Fallback qualitative palette (highly distinguishable)
        palette = [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
            "#ff7f00", "#a65628", "#f781bf", "#999999",
        ]

    # Add a trace per region
    for i, reg in enumerate(nonempty_regions):
        idx = np.fromiter(reg, dtype=int)
        fig.add_trace(
            go.Scatter3d(
                x=XYZ[idx, 0], y=XYZ[idx, 1], z=XYZ[idx, 2],
                mode="markers",
                marker=dict(size=size, color=palette[i % len(palette)]),
                name=f"{name_prefix} {i+1}",
                showlegend=True, hoverinfo="skip", opacity=alpha,
            )
        )


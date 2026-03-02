import numpy as np

def plot_organoid_mesh(
    mesh,
    vertex_values=None,
    *,
    backend="plotly",            # "plotly" | "mpl3d" | "orthoprojs"
    colorscale="RdBu_r",
    center_at_zero=True,
    vmin=None,
    vmax=None,
    show_colorbar=True,
    alpha=1.0,
    view=None,                   # dict(azim=-135, elev=25, projection="orthographic")
    fig_size=(750, 550),
    edgecolor=None,              # e.g. "k" for mpl backends, None for no edges
    linewidth=0.0,
):
    """
    Plot a triangle mesh with optional per-vertex scalar coloring.

    Expects:
        mesh.v : (V,3) vertices
        mesh.f : (F,3) integer triangle indices

    vertex_values:
        None -> constant color (light gray)
        (V,) -> per-vertex scalar field (e.g. HKS[:,0])

    Returns:
        plotly.Figure if backend="plotly"
        matplotlib.figure.Figure if backend in ("mpl3d","orthoprojs")
    """
    backend = str(backend).lower()
    if backend not in ("plotly", "mpl3d", "orthoprojs"):
        raise ValueError("backend must be 'plotly', 'mpl3d', or 'orthoprojs'")

    V = np.asarray(mesh.v, float)
    F = np.asarray(mesh.f, int)
    if V.ndim != 2 or V.shape[1] < 3:
        raise ValueError("mesh.v must be (V,3)")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("mesh.f must be (F,3)")

    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    i, j, k = F[:, 0], F[:, 1], F[:, 2]

    if view is None:
        view = dict(azim=-135, elev=25, projection="orthographic")

    vals = None
    if vertex_values is not None:
        vals = np.asarray(vertex_values, float).reshape(-1)
        if vals.shape[0] != V.shape[0]:
            raise ValueError(f"vertex_values must have length V={V.shape[0]} (got {vals.shape[0]})")

    # Choose color limits
    if vals is not None:
        finite = np.isfinite(vals)
        if not np.any(finite):
            vals = None
        else:
            if vmin is None or vmax is None:
                if center_at_zero:
                    vv = vals[finite]
                    m = float(np.nanmax(np.abs(vv)))
                    m = 1.0 if m == 0.0 else m
                    vmin0, vmax0 = -m, +m
                else:
                    vv = vals[finite]
                    vmin0, vmax0 = float(np.nanmin(vv)), float(np.nanmax(vv))
                    if vmin0 == vmax0:
                        vmin0, vmax0 = vmin0 - 1.0, vmax0 + 1.0
                vmin = vmin0 if vmin is None else vmin
                vmax = vmax0 if vmax is None else vmax

    if backend == "plotly":
        import plotly.graph_objects as go

        trace_kwargs = dict(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=float(alpha),
        )

        if vals is None:
            # constant mesh color
            trace_kwargs["color"] = "lightgray"
        else:
            trace_kwargs.update(
                intensity=vals,
                colorscale=colorscale,
                cmin=float(vmin),
                cmax=float(vmax),
                showscale=bool(show_colorbar),
            )
            if center_at_zero:
                trace_kwargs["cmid"] = 0.0

        fig = go.Figure(data=[go.Mesh3d(**trace_kwargs)])
        w, h = fig_size
        fig.update_layout(width=int(w), height=int(h))
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="rgba(0,0,0,0)",
                aspectmode="data",
            )
        )

        # Optional: map view azim/elev to a camera_eye.
        # This is a lightweight mapping that behaves consistently enough for fixed views.
        az = np.deg2rad(float(view.get("azim", -135)))
        el = np.deg2rad(float(view.get("elev", 25)))
        r = 2.0
        eye = dict(
            x=float(r * np.cos(el) * np.cos(az)),
            y=float(r * np.cos(el) * np.sin(az)),
            z=float(r * np.sin(el)),
        )
        fig.update_layout(scene_camera=dict(eye=eye))
        return fig

    if backend == "mpl3d":
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import matplotlib.tri as mtri
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        fig = plt.figure(figsize=(fig_size[0] / 100, fig_size[1] / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        tri = mtri.Triangulation(x, y, triangles=F)

        if vals is None:
            # single color surface
            surf = ax.plot_trisurf(
                tri, z,
                color="lightgray",
                linewidth=float(linewidth),
                edgecolor=edgecolor if edgecolor is not None else "none",
                alpha=float(alpha),
                shade=True,
            )
        else:
            # Matplotlib trisurf doesn't natively do per-vertex scalar the same way as Plotly.
            # A simple workaround: compute per-face value as mean of vertex values.
            face_vals = (vals[F[:, 0]] + vals[F[:, 1]] + vals[F[:, 2]]) / 3.0
            norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
            cmap = cm.get_cmap(colorscale) if isinstance(colorscale, str) else colorscale
            face_colors = cmap(norm(face_vals))

            surf = ax.plot_trisurf(
                tri, z,
                linewidth=float(linewidth),
                edgecolor=edgecolor if edgecolor is not None else "none",
                alpha=float(alpha),
                shade=False,
            )
            surf.set_facecolors(face_colors)

            if show_colorbar:
                mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
                mappable.set_array([])
                fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)

        ax.view_init(elev=float(view.get("elev", 25)), azim=float(view.get("azim", -135)))
        # "projection" is not identical in mpl; orthographic feel is mostly from fixed view + no perspective tricks.
        ax.set_axis_off()
        fig.patch.set_facecolor("white")
        return fig

    # orthoprojs
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 3, figsize=(fig_size[0] / 100, fig_size[1] / 100), dpi=100)
    ax_xy, ax_xz, ax_yz = axes

    tri_xy = mtri.Triangulation(x, y, triangles=F)
    tri_xz = mtri.Triangulation(x, z, triangles=F)
    tri_yz = mtri.Triangulation(y, z, triangles=F)

    if vals is None:
        # mesh outline / flat shade
        ax_xy.triplot(tri_xy, linewidth=0.2, color="gray")
        ax_xz.triplot(tri_xz, linewidth=0.2, color="gray")
        ax_yz.triplot(tri_yz, linewidth=0.2, color="gray")
    else:
        norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))
        cmap = cm.get_cmap(colorscale) if isinstance(colorscale, str) else colorscale

        # tripcolor expects per-vertex or per-face; we provide per-vertex for 2D projections
        im0 = ax_xy.tripcolor(tri_xy, vals, shading="gouraud", cmap=cmap, norm=norm, alpha=float(alpha))
        ax_xz.tripcolor(tri_xz, vals, shading="gouraud", cmap=cmap, norm=norm, alpha=float(alpha))
        ax_yz.tripcolor(tri_yz, vals, shading="gouraud", cmap=cmap, norm=norm, alpha=float(alpha))

        if show_colorbar:
            fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)

    ax_xy.set_title("XY"); ax_xz.set_title("XZ"); ax_yz.set_title("YZ")
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig


def plot_mesh_by_field(mesh, field, *, backend="mpl3d", **kwargs):
    """
    field can be:
      - a (V,) numpy array
      - a callable(mesh) -> (V,) array
      - or a string if your mesh stores fields (e.g., mesh.fields[field])
    """
    if callable(field):
        vals = field(mesh)
    elif isinstance(field, str):
        # adjust to your mesh class storage convention
        vals = getattr(mesh, field) if hasattr(mesh, field) else mesh.fields[field]
    else:
        vals = field
    return plot_organoid_mesh(mesh, vertex_values=vals, backend=backend, **kwargs)


def add_mesh_overlay(
    fig_or_ax,
    mesh,
    vertex_idx,
    *,
    backend="plotly",          # "plotly" | "mpl3d" | "orthoprojs"
    mode="area",              # "points" | "area"
    color="#FFD700",
    alpha=0.6,
    size=4.0,                 # point size (plotly: px-ish, mpl: points^2-ish)
    name="overlay",
    face_dilate=0,            # optional: expand area by k "rings" of vertex adjacency (simple, see note)
):
    """
    Overlay highlight of selected mesh vertices.

    Parameters
    ----------
    fig_or_ax :
        plotly: go.Figure
        mpl3d:  matplotlib Axes3D
        orthoprojs: matplotlib Figure with 3 axes (XY/XZ/YZ)
    mesh :
        Must have mesh.v (V,3) and mesh.f (F,3).
    vertex_idx :
        list[int] or np.ndarray of vertex indices to highlight.
    mode :
        "points": draw points at selected vertices
        "area":   draw shaded surface region (faces incident to selected vertices)
    face_dilate :
        Currently only meaningful for mode="area" and mpl/orthoprojs.
        If >0, we expand the selected vertex set by adjacency rings (requires vertex adjacency; see note below).

    Returns
    -------
    None (modifies fig_or_ax in-place)
    """
    backend = str(backend).lower()
    if backend not in ("plotly", "mpl3d", "orthoprojs"):
        raise ValueError("backend must be 'plotly', 'mpl3d', or 'orthoprojs'")

    mode = str(mode).lower()
    if mode not in ("points", "area"):
        raise ValueError("mode must be 'points' or 'area'")

    V = np.asarray(mesh.v, float)
    F = np.asarray(mesh.f, int)
    if V.ndim != 2 or V.shape[1] < 3:
        raise ValueError("mesh.v must be (V,3)")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("mesh.f must be (F,3)")

    # Accept list/ndarray/set/any iterable of ints
    if vertex_idx is None:
        return

    if isinstance(vertex_idx, (int, np.integer)):
        vidx = np.asarray([int(vertex_idx)], dtype=int)
    else:
        # Works for list, set, tuple, np.ndarray, generators, etc.
        vidx = np.fromiter((int(u) for u in vertex_idx), dtype=int, count=-1)

    if vidx.size == 0:
        return

    # clip to valid range and unique
    vidx = vidx[(vidx >= 0) & (vidx < V.shape[0])]
    if vidx.size == 0:
        return
    vidx = np.unique(vidx)

    # --- helper: face mask for "area" mode ---
    def _incident_face_mask(F, selected_vertices):
        sel = np.zeros(V.shape[0], dtype=bool)
        sel[selected_vertices] = True
        return sel[F[:, 0]] | sel[F[:, 1]] | sel[F[:, 2]]

    # Optional dilation (simple adjacency expansion)
    # Note: requires vertex adjacency; we can approximate from faces quickly.
    if mode == "area" and face_dilate and int(face_dilate) > 0:
        k = int(face_dilate)
        # build adjacency from faces
        adj = [set() for _ in range(V.shape[0])]
        for a, b, c in F:
            adj[a].update([b, c])
            adj[b].update([a, c])
            adj[c].update([a, b])
        sel = set(map(int, vidx.tolist()))
        frontier = set(sel)
        for _ in range(k):
            new = set()
            for u in frontier:
                new.update(adj[u])
            new -= sel
            sel |= new
            frontier = new
            if not frontier:
                break
        vidx = np.asarray(sorted(sel), dtype=int)

    XYZ = V[:, :3]

    # =========================
    # plotly backend
    # =========================
    if backend == "plotly":
        import plotly.graph_objects as go

        if mode == "points":
            pts = XYZ[vidx]
            fig_or_ax.add_trace(
                go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=float(size), color=color),
                    name=name,
                    opacity=float(alpha),
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
            return

        # mode == "area": overlay a second semi-transparent mesh made of incident faces only
        fmask = _incident_face_mask(F, vidx)
        F_sel = F[fmask]
        if F_sel.size == 0:
            return

        # Reindex vertices used by selected faces to keep overlay small
        used = np.unique(F_sel.reshape(-1))
        remap = {int(u): i for i, u in enumerate(used.tolist())}
        V_sel = XYZ[used]
        F_remap = np.vectorize(remap.get)(F_sel)

        # A constant-color mesh overlay
        fig_or_ax.add_trace(
            go.Mesh3d(
                x=V_sel[:, 0], y=V_sel[:, 1], z=V_sel[:, 2],
                i=F_remap[:, 0], j=F_remap[:, 1], k=F_remap[:, 2],
                color=color,
                opacity=float(alpha),
                name=name,
                showlegend=True,
                hoverinfo="skip",
            )
        )
        return

    # =========================
    # matplotlib backends
    # =========================
    if backend == "mpl3d":
        if hasattr(fig_or_ax, "scatter"):
            # Already an Axes3D
            ax = fig_or_ax
        elif hasattr(fig_or_ax, "axes"):
            axes = fig_or_ax.axes
            if not axes:
                raise ValueError("Figure has no axes.")
            ax = axes[0]
        else:
            raise ValueError("mpl3d backend expects a Figure or Axes3D.")

        # Optional safety check
        if not hasattr(ax, "get_zlim"):
            raise ValueError("mpl3d backend requires a 3D axis.")

        if mode == "points":
            pts = XYZ[vidx]
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=float(size),  # Matplotlib: s ~ points^2-ish; caller can pass squared if desired
                c=color,
                alpha=float(alpha),
                label=name,
                depthshade=False,
            )
            return

        # area: draw a Poly3DCollection over incident faces
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fmask = _incident_face_mask(F, vidx)
        F_sel = F[fmask]
        if F_sel.size == 0:
            return

        polys = XYZ[F_sel]  # (K,3,3)
        coll = Poly3DCollection(polys, facecolors=color, edgecolors="none", alpha=float(alpha))
        ax.add_collection3d(coll)
        return

    # orthoprojs: matplotlib Figure with 3 axes
    import matplotlib.tri as mtri

    fig = fig_or_ax
    axes = fig.get_axes()
    if len(axes) < 3:
        raise ValueError("orthoprojs expects a matplotlib Figure with 3 subplots (XY, XZ, YZ).")

    ax_xy, ax_xz, ax_yz = axes[0], axes[1], axes[2]

    if mode == "points":
        pts = XYZ[vidx]
        ax_xy.scatter(pts[:, 0], pts[:, 1], s=float(size), c=color, alpha=float(alpha), label=name)
        ax_xz.scatter(pts[:, 0], pts[:, 2], s=float(size), c=color, alpha=float(alpha), label=name)
        ax_yz.scatter(pts[:, 1], pts[:, 2], s=float(size), c=color, alpha=float(alpha), label=name)
        return

    # area overlay in orthoprojs: tripcolor only the selected faces
    fmask = _incident_face_mask(F, vidx)
    F_sel = F[fmask]
    if F_sel.size == 0:
        return

    x, y, z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    tri_xy = mtri.Triangulation(x, y, triangles=F_sel)
    tri_xz = mtri.Triangulation(x, z, triangles=F_sel)
    tri_yz = mtri.Triangulation(y, z, triangles=F_sel)

    # Draw filled overlay triangles
    ax_xy.tripcolor(tri_xy, np.ones(x.shape[0]), shading="flat", color=color, alpha=float(alpha))
    ax_xz.tripcolor(tri_xz, np.ones(x.shape[0]), shading="flat", color=color, alpha=float(alpha))
    ax_yz.tripcolor(tri_yz, np.ones(x.shape[0]), shading="flat", color=color, alpha=float(alpha))


import numpy as np
import re

_RGB_RE = re.compile(r"rgb\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)")

def _plotly_rgb_to_mpl(c):
    if isinstance(c, str):
        m = _RGB_RE.fullmatch(c.strip())
        if m:
            r, g, b = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
            return f"#{r:02x}{g:02x}{b:02x}"
    return c

def _hex_to_rgba_str(color, alpha=1.0):
    """
    Convert '#RRGGBB' / '#RGB' / 'rgb(r,g,b)' into 'rgba(r,g,b,a)' for Plotly vertexcolor.
    If it's a named CSS color, we pass it through (Plotly usually accepts it),
    but it won't embed alpha unless it was rgb/hex.
    """
    a = float(alpha)
    if not isinstance(color, str):
        return color

    s = color.strip()
    m = _RGB_RE.fullmatch(s)
    if m:
        r, g, b = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return f"rgba({r},{g},{b},{a})"

    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:
            r = int(h[0] * 2, 16)
            g = int(h[1] * 2, 16)
            b = int(h[2] * 2, 16)
            return f"rgba({r},{g},{b},{a})"
        if len(h) == 6:
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            return f"rgba({r},{g},{b},{a})"

    # fallback: let plotly interpret (alpha may not apply)
    return s


def plot_mesh_by_regions(
    mesh,
    regions,
    *,
    backend="plotly",                 # "plotly" | "mpl3d" | "orthoprojs"
    region_kind="auto",               # "auto" | "vertex" | "face"
    region_names=None,                # list[str] or None -> "region 1", ...
    colors=None,                      # list of CSS/hex colors for regions (cycled)
    colorscale=None,                  # optional plotly colorscale name/list (used only if colors is None)
    baseline_color="lightgray",
    baseline_name="other",
    priority="last",                  # "last" | "first"
    alpha=1.0,
    view=None,
    fig_size=(750, 550),
    edgecolor=None,
    linewidth=0.0,
    add_legend=True,
):
    """
    Wrapper that colors regions via a per-vertex label field, calls plot_organoid_mesh(),
    hides any colorbar, and adds a region legend.

    Plotly backend uses Mesh3d(vertexcolor=...) to avoid colormap interpolation mismatches.
    Matplotlib backends use vertex_values numeric labels (note: plot_organoid_mesh uses Normalize(),
    so boundaries may still look slightly blended).
    """
    backend_l = str(backend).lower()
    if backend_l not in ("plotly", "mpl3d", "orthoprojs"):
        raise ValueError("backend must be 'plotly', 'mpl3d', or 'orthoprojs'")
    kind = str(region_kind).lower()
    if kind not in ("auto", "vertex", "face"):
        raise ValueError("region_kind must be 'auto', 'vertex', or 'face'")
    if priority not in ("last", "first"):
        raise ValueError('priority must be "last" or "first".')

    V = np.asarray(mesh.v, float)
    F = np.asarray(mesh.f, int)
    nV = V.shape[0]
    nF = F.shape[0]

    # normalize + drop empties
    regs = []
    for reg in (regions or []):
        if reg is None:
            continue
        try:
            if len(reg) == 0:
                continue
        except TypeError:
            reg = list(reg)
            if len(reg) == 0:
                continue
        regs.append(reg)

    K = len(regs)
    if K == 0:
        return plot_organoid_mesh(
            mesh,
            vertex_values=None,
            backend=backend_l,
            alpha=alpha,
            view=view,
            fig_size=fig_size,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )

    # names
    if region_names is None:
        region_names = [f"region {i+1}" for i in range(K)]
    else:
        region_names = list(region_names)
        if len(region_names) != K:
            raise ValueError(f"region_names must have length {K}")

    # palette
    if colors and len(colors) > 0:
        palette = list(colors)
    elif colorscale is not None:
        from plotly import colors as pc
        cs = pc.get_colorscale(colorscale) if isinstance(colorscale, str) else colorscale
        vals = [0.5] if K == 1 else [i / (K - 1) for i in range(K)]
        palette = [pc.sample_colorscale(cs, v)[0] for v in vals]
    else:
        palette = [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
            "#ff7f00", "#a65628", "#f781bf", "#999999",
        ]

    def _infer_kind(reg):
        if kind != "auto":
            return kind
        try:
            mx = max(reg)
        except Exception:
            reg = list(reg)
            mx = max(reg) if reg else -1
        if mx < nV:
            return "vertex"
        if mx < nF:
            return "face"
        return "vertex"

    def _region_to_vertices(reg):
        rk = _infer_kind(reg)
        if rk == "vertex":
            idx = np.fromiter((int(u) for u in reg), dtype=int, count=-1)
            idx = idx[(idx >= 0) & (idx < nV)]
            return np.unique(idx)
        # face -> collect incident vertices
        fidx = np.fromiter((int(u) for u in reg), dtype=int, count=-1)
        fidx = fidx[(fidx >= 0) & (fidx < nF)]
        if fidx.size == 0:
            return np.asarray([], dtype=int)
        return np.unique(F[fidx].reshape(-1))

    # build integer labels 0..K
    labels = np.zeros(nV, dtype=int)
    if priority == "last":
        for i, reg in enumerate(regs, start=1):
            vidx = _region_to_vertices(reg)
            labels[vidx] = i
    else:
        for i, reg in enumerate(regs, start=1):
            vidx = _region_to_vertices(reg)
            mask = labels[vidx] == 0
            labels[vidx[mask]] = i

    # =========================
    # Plotly: use vertexcolor (exact match)
    # =========================
    if backend_l == "plotly":
        # get base fig from your existing function (camera/layout etc.) :contentReference[oaicite:4]{index=4}
        fig = plot_organoid_mesh(
            mesh,
            vertex_values=None,
            backend="plotly",
            alpha=alpha,
            view=view,
            fig_size=fig_size,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )

        # per-vertex explicit colors
        vcols = np.empty(nV, dtype=object)
        base_rgba = _hex_to_rgba_str(baseline_color, alpha=1.0)
        vcols[:] = base_rgba
        for i in range(1, K + 1):
            c = palette[(i - 1) % len(palette)]
            vcols[labels == i] = _hex_to_rgba_str(c, alpha=1.0)

        # overwrite trace coloring: disable intensity/colorscale/colorbar
        tr = fig.data[0]
        tr.update(
            vertexcolor=vcols.tolist(),
            intensity=None,
            colorscale=None,
            cmin=None,
            cmax=None,
            cmid=None,
            showscale=False,
            color=None,  # remove constant color (plot_organoid_mesh uses "lightgray" if no vals) :contentReference[oaicite:5]{index=5}
        )

        if add_legend:
            import plotly.graph_objects as go
            # region entries
            for i in range(1, K + 1):
                c = palette[(i - 1) % len(palette)]
                fig.add_trace(
                    go.Scatter3d(
                        x=[None], y=[None], z=[None],
                        mode="markers",
                        marker=dict(size=8, color=c),
                        name=region_names[i - 1],
                        showlegend=True,
                    )
                )
            # baseline entry
            fig.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode="markers",
                    marker=dict(size=8, color=baseline_color),
                    name=baseline_name,
                    showlegend=True,
                )
            )
        return fig

    # =========================
    # Matplotlib: numeric labels via plot_organoid_mesh (single fig)
    # =========================
    # Note: plot_organoid_mesh uses Normalize() internally :contentReference[oaicite:6]{index=6}
    # so categorical boundaries can be slightly blended; fixing that would require
    # changing plot_organoid_mesh itself (or using overlays).
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap
    import matplotlib.lines as mlines

    palette_mpl = [_plotly_rgb_to_mpl(c) for c in palette]
    baseline_mpl = _plotly_rgb_to_mpl(baseline_color)
    cmap = ListedColormap([baseline_mpl] + palette_mpl[:K])

    fig = plot_organoid_mesh(
        mesh,
        vertex_values=labels.astype(float),  # plot_organoid_mesh casts to float :contentReference[oaicite:7]{index=7}
        backend=backend_l,
        colorscale=cmap,
        center_at_zero=False,
        vmin=0.0,
        vmax=float(K),
        show_colorbar=False,
        alpha=alpha,
        view=view,
        fig_size=fig_size,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    if add_legend:
        handles = [
            mlines.Line2D([], [], marker="o", linestyle="None", markersize=6,
                          markerfacecolor=palette_mpl[i], markeredgecolor="none",
                          label=region_names[i])
            for i in range(K)
        ]
        handles.append(
            mlines.Line2D([], [], marker="o", linestyle="None", markersize=6,
                          markerfacecolor=baseline_mpl, markeredgecolor="none",
                          label=baseline_name)
        )

        if backend_l == "mpl3d":
            ax = fig.axes[0] if getattr(fig, "axes", None) else None
            if ax is not None:
                ax.legend(handles=handles, loc="upper left")
        else:
            axes = fig.get_axes()
            if axes:
                axes[0].legend(handles=handles, loc="upper left")

    return fig
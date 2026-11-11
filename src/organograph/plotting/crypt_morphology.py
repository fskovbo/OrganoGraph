"""Interactive inspection of cap geometry and alignment; no fitting in plots."""
import numpy as np


def plot_cap_representation(patch, mapping, representation, basis, *, n_rings=32, n_sectors=96):
    """Original, parameter domain, reconstruction and correspondence residuals."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from organograph.crypt_morphology import reconstruct

    reconstructed, rf = reconstruct(representation, basis, original_coordinates=True,
                                    n_rings=n_rings, n_sectors=n_sectors)
    q = mapping.theta_phi.copy()
    th, phi = q.T
    cap = np.column_stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
    q[:, 1] -= representation.phase
    original_samples = reconstruct(representation, basis, q, original_coordinates=True)
    error = np.linalg.norm(original_samples - patch.vertices, axis=1)
    fig = make_subplots(rows=1, cols=4, specs=[[{"type": "scene"}] * 4],
                        subplot_titles=["Original surface", "Fixed cap coordinates", "Reconstructed surface",
                                        "Correspondence residual (source units)"])
    for col, (v, f, values) in enumerate([(patch.vertices, patch.faces, None), (cap, patch.faces, phi),
                                         (reconstructed, rf, None), (patch.vertices, patch.faces, error)], 1):
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                intensity=values, colorscale="Viridis", color="lightsteelblue",
                                showscale=col == 4, colorbar={"title": "Error", "thickness": 12}),
                      row=1, col=col)
    # Keep physical plots at identical axis ranges, so size changes remain visible.
    both = np.vstack([patch.vertices, reconstructed])
    low, high = both.min(axis=0), both.max(axis=0)
    fig.update_scenes(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    for col in (1, 3, 4):
        fig.update_scenes(xaxis_range=[low[0], high[0]], yaxis_range=[low[1], high[1]],
                          zaxis_range=[low[2], high[2]], row=1, col=col)
    fig.update_layout(height=420, width=1400, margin={"l": 0, "r": 10, "t": 45, "b": 0})
    return fig


def plot_cap_alignment(result, basis, *, n_rings=12, n_sectors=48, labels=None):
    import plotly.graph_objects as go
    _, faces = basis.reference_mesh(n_rings, n_sectors)
    if result.template.shape[0] != 1 + n_rings * n_sectors:
        raise ValueError("Plot sampling settings must match alignment settings")
    fig = go.Figure()
    for i, v in enumerate(result.aligned):
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                opacity=0.35, name=str(labels[i]) if labels is not None else f"Crypt {i}",
                                showlegend=True))
    v = result.template
    fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            color="black", opacity=0.4, name="Template", showlegend=True))
    fig.update_layout(scene={"aspectmode": "data"}, width=800, height=650)
    return fig


def plot_cap_error_summary(table, *, group_column="kind"):
    """Standalone Matplotlib chart of geometric error versus basis order."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for group, rows in table.groupby(group_column):
        summary = rows.groupby("degree")["relative_symmetric_rmse"].median()
        ax.plot(summary.index, 100 * summary.values, marker="o", label=str(group))
    ax.set(xlabel="Maximum interior degree at the hemisphere", ylabel="Surface RMSE / RMS radius (%)")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(title=group_column)
    return fig


def plot_cap_comparison_static(patch, representation, basis):
    """Exportable surface comparison in original units, using the same view/limits."""
    import matplotlib.pyplot as plt
    from organograph.crypt_morphology import reconstruct
    v, f = reconstruct(representation, basis, original_coordinates=True, n_rings=40, n_sectors=128)
    all_v = np.vstack([v, patch.vertices])
    center = (all_v.max(axis=0) + all_v.min(axis=0)) / 2
    radius = np.ptp(all_v, axis=0).max() / 2 * 1.05
    fig = plt.figure(figsize=(10, 4.8), layout="constrained")
    for i, (points, faces, title) in enumerate([(patch.vertices, patch.faces, "Original membrane patch"),
                                               (v, f, "Cap reconstruction")], 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.plot_trisurf(*points.T, triangles=faces, color="#80b1d3", linewidth=0,
                        antialiased=False, shade=True)
        ax.set(xlim=(center[0] - radius, center[0] + radius),
               ylim=(center[1] - radius, center[1] + radius), zlim=(center[2] - radius, center[2] + radius),
               title=title)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=25, azim=-60)
        ax.set_axis_off()
    return fig


def plot_opening_family(family, shifts, *, support=None):
    """Interactive prescribed rims on unchanged membrane; rejected rims omitted."""
    import plotly.graph_objects as go
    patch = support.patch if support is not None else family.padded_patch().patch
    v, f = patch.vertices, patch.faces
    fig = go.Figure(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                            i=f[:, 0], j=f[:, 1], k=f[:, 2], color="lightgray", opacity=.5,
                            name="Original membrane", showlegend=True))
    rejected = []
    for shift in shifts:
        try:
            cut = family.cut(shift).patch
        except ValueError:
            rejected.append(shift.name)
            continue
        rim = cut.vertices[np.r_[cut.boundary, cut.boundary[0]]]
        baseline = shift.bound == 0
        fig.add_trace(go.Scatter3d(x=rim[:, 0], y=rim[:, 1], z=rim[:, 2], mode="lines",
                                  line=dict(width=7 if baseline else 4,
                                            **({"color": "black"} if baseline else {})), name=shift.name))
    title = "Prescribed opening family (original coordinates)"
    if rejected:
        title += f" — {len(rejected)} rejected; see failure table"
    fig.update_layout(title=title, scene=dict(aspectmode="data"), height=650, width=1000)
    return fig


def plot_boundary_stability(table):
    """Paired core stability/fidelity and whole-cap response, all relative to R0.

    Each point is one accepted nonbaseline crypt/variant pair. Baselines and
    failed variants are excluded from this plot, not from the report's counts.
    """
    import matplotlib.pyplot as plt
    from organograph.crypt_morphology.stability import METHODS
    metrics = [("core_drift_relative", "Same physical core: prediction drift"),
               ("core_fidelity_relative", "Same physical core: reconstruction error"),
               ("cap_drift_fixed_rms", "Whole cap: aligned drift (fixed R₀)"),
               ("cap_drift_unit_rms", "Whole cap: aligned drift (recomputed RMS)")]
    colors = ["#4477aa", "#ee9944", "#228833"]
    names = ["Recut / hard", "Fixed chart / hard", "Fixed chart / soft"]
    valid = table[(table.status == "ok") & (table.variant != "baseline")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, (metric, title) in zip(axes.flat, metrics):
        for index, method in enumerate(METHODS):
            values = valid.loc[valid.method == method, metric].dropna().to_numpy()*100
            if not len(values):
                continue
            # Deterministic jitter avoids hiding repeated amplitudes/examples.
            jitter = np.random.default_rng(2026).uniform(-.18, .18, len(values))
            ax.scatter(index+jitter, values, color=colors[index], alpha=.3, s=13)
            ax.plot([index-.24, index+.24], [np.median(values)]*2, color="black", lw=2)
        ax.set_xticks(range(3), names)
        ax.set(title=title, ylabel="RMSE / original RMS radius (%)")
        if metric == "cap_drift_unit_rms":
            ax.set_ylabel("RMSE between unit-RMS surfaces (%)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
    return fig


def plot_opening_membership(family, support, shift, *, blur_width):
    """Show the mean of the three actual fit weights on each support triangle."""
    import plotly.graph_objects as go
    from scipy.special import expit
    from organograph.crypt_morphology.stability import BARYCENTRIC
    if not np.isfinite(blur_width) or blur_width <= 0:
        raise ValueError("blur_width must be finite and positive")
    patch = support.patch
    v, f = patch.vertices, patch.faces
    level = support.transfer(family.level_set(shift))
    weights = expit(-np.einsum("qa,fa->fq", BARYCENTRIC, level[f])/blur_width)
    weights[family.forbidden_faces[patch.source_face_ids]] = 0
    fig = go.Figure(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                            intensity=weights.mean(axis=1), intensitymode="cell", cmin=0, cmax=1,
                            colorscale="Viridis", colorbar=dict(title="Mean fit weight")))
    cut = family.cut(shift).patch
    rim = cut.vertices[np.r_[cut.boundary, cut.boundary[0]]]
    fig.add_trace(go.Scatter3d(x=rim[:, 0], y=rim[:, 1], z=rim[:, 2], mode="lines",
                              line=dict(color="black", width=6), name="Hard opening"))
    fig.update_layout(title=f"{shift.name}: soft membership on unchanged membrane",
                      scene=dict(aspectmode="data"), width=900, height=600)
    return fig


def plot_opening_reconstructions(result, basis, variant):
    """Inspect all three full-domain reconstructions, including extrapolated rims."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from organograph.crypt_morphology import reconstruct
    from organograph.crypt_morphology.stability import METHODS
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "scene"}]*3],
                        subplot_titles=["Recut / hard", "Fixed chart / hard", "Fixed chart / soft"])
    all_vertices = []
    for col, method in enumerate(METHODS, 1):
        sample = result.samples.get((variant, method))
        if sample is None:
            continue
        rep, patch, _ = sample
        v, f = reconstruct(rep, basis, original_coordinates=True)
        all_vertices.extend([v, patch.vertices])
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color="#80b1d3", opacity=.85, name="Reconstruction", showlegend=col == 1),
                      row=1, col=col)
        p, pf = patch.vertices, patch.faces
        fig.add_trace(go.Mesh3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], i=pf[:, 0], j=pf[:, 1], k=pf[:, 2],
                                color="gray", opacity=.3, name="Fitted membrane/support", showlegend=col == 1),
                      row=1, col=col)
    if all_vertices:
        points = np.vstack(all_vertices)
        lo, hi = points.min(axis=0), points.max(axis=0)
        fig.update_scenes(aspectmode="data", xaxis_range=[lo[0], hi[0]],
                          yaxis_range=[lo[1], hi[1]], zaxis_range=[lo[2], hi[2]])
    fig.update_layout(title=f"{variant}: full cap (fixed-chart outer rims include extrapolation)",
                      height=480, width=1350, margin=dict(l=0, r=0, t=70, b=0))
    return fig


def plot_decoding_sequence(vertices, faces, labels, *, reference=None, title="Decoded surfaces",
                           warnings=None):
    """Interactive comparison in one fixed frame; shared limits preserve size.

    Blue surfaces are supplied decoded meshes, gray is the frozen baseline, and
    black curves are their openings. No pose adjustment or smoothing is applied.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    vertices = [np.asarray(v) for v in vertices]
    if len(vertices) != len(labels) or not len(vertices):
        raise ValueError("One label per decoded mesh is required")
    warnings = [False]*len(vertices) if warnings is None else list(warnings)
    if len(warnings) != len(vertices):
        raise ValueError("One warning flag per mesh is required")
    labels = [f"{label}<br>Geometry warning" if warning else label for label, warning in zip(labels, warnings)]
    fig = make_subplots(rows=1, cols=len(vertices), specs=[[{"type": "scene"}]*len(vertices)],
                        subplot_titles=labels)
    faces = np.asarray(faces)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique[counts == 1]
    for col, points in enumerate(vertices, 1):
        if reference is not None:
            ref = np.asarray(reference)
            fig.add_trace(go.Mesh3d(x=ref[:, 0], y=ref[:, 1], z=ref[:, 2],
                                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color="gray", opacity=.18,
                                    name="Baseline", showlegend=col == 1), row=1, col=col)
        fig.add_trace(go.Mesh3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color="#80b1d3", opacity=.9,
                                name="Decoded surface", showlegend=col == 1), row=1, col=col)
        rim = np.full((len(boundary_edges), 3, 3), np.nan)
        rim[:, :2] = points[boundary_edges]
        rim = rim.reshape(-1, 3)
        fig.add_trace(go.Scatter3d(x=rim[:, 0], y=rim[:, 1], z=rim[:, 2], mode="lines",
                                  line=dict(color="black", width=4), name="Opening", showlegend=col == 1),
                      row=1, col=col)
    all_v = np.vstack(vertices + ([] if reference is None else [reference]))
    low, high = all_v.min(axis=0), all_v.max(axis=0)
    fig.update_scenes(aspectmode="data", xaxis_range=[low[0], high[0]], yaxis_range=[low[1], high[1]],
                      zaxis_range=[low[2], high[2]], xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout(title=title, width=max(900, 300*len(vertices)), height=470,
                      margin=dict(l=0, r=0, t=95, b=0))
    return fig


def plot_decoding_sequence_static(vertices, faces, labels, *, title="Decoded surfaces"):
    """Standalone PNG/PDF-ready snapshots with a shared scale and view."""
    import matplotlib.pyplot as plt
    points = np.vstack(vertices)
    center = (points.max(axis=0)+points.min(axis=0))/2
    extent = np.ptp(points, axis=0).max()/2*1.07
    fig = plt.figure(figsize=(3.2*len(vertices), 3.7), layout="constrained")
    for i, (v, label) in enumerate(zip(vertices, labels), 1):
        ax = fig.add_subplot(1, len(vertices), i, projection="3d")
        ax.plot_trisurf(*v.T, triangles=faces, color="#80b1d3", linewidth=0, antialiased=False)
        ax.set(xlim=(center[0]-extent, center[0]+extent), ylim=(center[1]-extent, center[1]+extent),
               zlim=(center[2]-extent, center[2]+extent), title=label)
        ax.set_box_aspect([1, 1, 1]); ax.set_axis_off(); ax.view_init(elev=25, azim=-60)
    fig.suptitle(title)
    return fig


def plot_decoding_comparison(sequences, faces, labels, *, reference=None, title="Alignment comparison",
                             static=False):
    """One row per alignment method, with a common view and scale for all rows.

    ``sequences`` maps row labels to decoded vertex arrays in interpolation order.
    Surfaces are never realigned for display. Static output is a Matplotlib figure;
    otherwise the interactive rows reuse the baseline/opening overlays above.
    """
    rows = list(sequences.items())
    if not rows or any(len(meshes) != len(labels) for _, meshes in rows) or not labels:
        raise ValueError("Each sequence requires one mesh per column label")
    all_v = np.vstack([v for _, meshes in rows for v in meshes] + ([] if reference is None else [reference]))
    low, high = all_v.min(axis=0), all_v.max(axis=0)
    if static:
        import matplotlib.pyplot as plt
        center, extent = (low+high)/2, (high-low).max()/2*1.07
        fig = plt.figure(figsize=(3.1*len(labels), 3.2*len(rows)), layout="constrained")
        for row, (method, meshes) in enumerate(rows):
            for col, (v, label) in enumerate(zip(meshes, labels)):
                ax = fig.add_subplot(len(rows), len(labels), row*len(labels)+col+1, projection="3d")
                ax.plot_trisurf(*v.T, triangles=faces, color="#80b1d3", linewidth=0, antialiased=False)
                ax.set(xlim=(center[0]-extent, center[0]+extent), ylim=(center[1]-extent, center[1]+extent),
                       zlim=(center[2]-extent, center[2]+extent), title=f"{method}\n{label}")
                ax.set_box_aspect([1, 1, 1]); ax.set_axis_off(); ax.view_init(elev=25, azim=-60)
        fig.suptitle(title)
        return fig
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=len(rows), cols=len(labels),
                        specs=[[{"type": "scene"}]*len(labels) for _ in rows],
                        subplot_titles=[f"{method}<br>{label}" for method, _ in rows for label in labels])
    for row, (_, meshes) in enumerate(rows, 1):
        sequence = plot_decoding_sequence(meshes, faces, labels, reference=reference)
        for trace in sequence.data:
            col = 1 if trace.scene == "scene" else int(trace.scene[5:])
            trace.showlegend = bool(trace.showlegend and row == 1)
            fig.add_trace(trace, row=row, col=col)
    fig.update_scenes(aspectmode="data", xaxis_range=[low[0], high[0]], yaxis_range=[low[1], high[1]],
                      zaxis_range=[low[2], high[2]], xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout(title=title, width=max(900, 290*len(labels)), height=390*len(rows),
                      margin=dict(l=0, r=0, t=100, b=0))
    return fig


def plot_alignment_interpolation_response(table, paths, *, methods=None):
    """Paired core errors for original versus shared-frame coefficient mixtures."""
    import matplotlib.pyplot as plt
    cases, modes = list(table.case.unique()), list(table.size_mode.unique())
    methods = methods or {m: m for m in table.method.unique()}
    fig, axes = plt.subplots(len(cases), len(modes), figsize=(6*len(modes), 3.8*len(cases)),
                             squeeze=False, layout="constrained")
    for row, case in enumerate(cases):
        for col, mode in enumerate(modes):
            ax = axes[row, col]
            group = table[(table.case == case) & (table.size_mode == mode)]
            baseline = group[group.scenario == "baseline"].iloc[0]
            ax.axhline(100*baseline.core_surface_rmse_relative, color="gray", ls=":", label="Baseline fit")
            for (method, label), color in zip(methods.items(), ["#cc6677", "#228833"]):
                by_name = group[group.method == method].set_index("scenario")
                for (path, (left, right)), style in zip(paths.items(), ["-", "--"]):
                    part = group[(group.method == method) & (group.kind == "interpolation") & (group.path == path)]
                    part = part.sort_values("fraction")
                    if left not in by_name.index or right not in by_name.index or part.empty:
                        continue
                    errors = [by_name.loc[left, "core_surface_rmse_relative"], *part.core_surface_rmse_relative,
                              by_name.loc[right, "core_surface_rmse_relative"]]
                    ax.plot([0, *part.fraction, 1], 100*np.array(errors), marker="o", ls=style,
                            color=color, label=f"{label}, {path}")
            ax.set(title=f"{case} — {mode}", xlabel="Fraction toward endpoint B",
                   ylabel="Core point-to-surface error / R₀ (%)", ylim=(0, None))
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    return fig


def plot_decoding_response(table, *, size_mode="physical"):
    """Opening-only perturbations: geometry error versus imposed rim RMS error."""
    import matplotlib.pyplot as plt
    cases = list(table.case.unique())
    fig, axes = plt.subplots(1, len(cases), figsize=(6*len(cases), 4.3), squeeze=False, layout="constrained")
    for ax, case in zip(axes.flat, cases):
        rows = table[(table.case == case) & (table.size_mode == size_mode)]
        baseline = rows[rows.scenario == "baseline"].iloc[0]
        ax.axhline(100*baseline.core_surface_rmse_relative, color="gray", ls="--", label="Baseline reconstruction")
        for (max_mode, sign), part in rows[rows.kind == "rim_error"].groupby(["max_mode", "sign"]):
            part = part.sort_values("amplitude_fraction")
            ax.plot(100*part.rim_rms_relative, 100*part.core_surface_rmse_relative, marker="o",
                    ls="-" if sign > 0 else ":", color="#4477aa" if max_mode <= 3 else "#dd8844",
                    label=f"Rim modes 1–{int(max_mode)}, {'+' if sign > 0 else '−'} direction")
        ax.set(title=case, xlabel="Imposed opening RMS displacement / R₀ (%)",
               ylabel="Core point-to-surface error / R₀ (%)")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    return fig

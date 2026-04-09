import matplotlib.pyplot as plt


DEFAULT_MARKER_COLORS = {
    "Agr2": "#359BD5",
    "Lysozyme": "#2C75D2",
    "Mucin 2": "#3EC1D9",
    "Chroma": "#D852CB",
    "Glucagon": "#983EE2",
    "Serotonin": "#F392E3",
    "LGR5": "#FFB431",
    "AldoB": "#F16C6A",
    "Cyclin D": "#808080",
    "Cyclin A": "#808080",
    "KI67": "#808080",
    "other": "#EBEBEB",
}


def resolve_category_colors(
    categories,
    color_scheme=None,
    other_label="other",
    fallback_cmap="tab20",
):
    """
    Resolve colors for plotting categories.

    Parameters
    ----------
    categories : list[str]
        Categories that will be plotted.
    color_scheme : dict or None
        Mapping category -> color.
        If None, DEFAULT_MARKER_COLORS is used.
    other_label : str
        Residual / negative category label.
    fallback_cmap : str
        Matplotlib cmap used for categories missing from color_scheme.

    Returns
    -------
    dict
        category -> color
    """
    if color_scheme is None:
        color_scheme = DEFAULT_MARKER_COLORS

    cmap = plt.get_cmap(fallback_cmap)
    fallback_colors = [cmap(i) for i in range(len(categories))]

    color_map = {}
    for i, cat in enumerate(categories):
        if cat in color_scheme:
            color_map[cat] = color_scheme[cat]
        else:
            color_map[cat] = fallback_colors[i]

    if other_label in categories and other_label not in color_scheme:
        color_map[other_label] = "#EBEBEB"

    return color_map
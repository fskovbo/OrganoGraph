import numpy as np
import warnings

def make_identity_transform():
    return {
        "center": np.array([np.nan, np.nan, np.nan], dtype=float),
        "scale":  1.0,
        "rotation": np.full((3, 3), np.nan, dtype=float),
    }

def transform_is_applied(tr):
    """True iff center is finite (meaning we consider the transform active)."""
    if tr is None:
        return False
    c = np.asarray(tr.get("center", [np.nan, np.nan, np.nan]), dtype=float)
    return c.shape == (3,) and np.isfinite(c).all()

def rotation_is_applied(tr):
    if tr is None:
        return False
    R = np.asarray(tr.get("rotation", np.full((3,3), np.nan)), dtype=float)
    return R.shape == (3,3) and np.isfinite(R).all()

def apply_transform_to_points(X, tr):
    """
    Apply center/scale and optional rotation to points.

    Convention:
      Y = (X - center) / scale
      if rotation present: Y = Y @ rotation.T   (matches sklearn PCA transform style)
    """
    X = np.asarray(X, dtype=float)
    if not transform_is_applied(tr):
        return X

    c = np.asarray(tr["center"], dtype=float)
    s = float(tr.get("scale", 1.0))
    if not np.isfinite(s) or s == 0:
        raise ValueError("Transform scale must be finite and non-zero")

    Y = (X - c) / s

    if rotation_is_applied(tr):
        R = np.asarray(tr["rotation"], dtype=float)
        Y = Y @ R.T

    return Y

def warn_if_already_transformed(tr, obj_name="object"):
    if transform_is_applied(tr):
        warnings.warn(
            f"{obj_name} already appears transformed (center is finite). "
            "Applying another transform may double-transform coordinates.",
            RuntimeWarning,
        )

def transforms_equal(a, b, tol=1e-9):
    """Equality check for transforms (center/scale/rotation), ignoring NaN parts consistently."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    a_center = np.asarray(a.get("center", [np.nan]*3), float)
    b_center = np.asarray(b.get("center", [np.nan]*3), float)
    if a_center.shape != (3,) or b_center.shape != (3,):
        return False

    a_on = np.isfinite(a_center).all()
    b_on = np.isfinite(b_center).all()
    if a_on != b_on:
        return False
    if not a_on and not b_on:
        return True  # both "identity" by your convention

    if np.max(np.abs(a_center - b_center)) > tol:
        return False
    if abs(float(a.get("scale", 1.0)) - float(b.get("scale", 1.0))) > tol:
        return False

    a_R_on = rotation_is_applied(a)
    b_R_on = rotation_is_applied(b)
    if a_R_on != b_R_on:
        return False
    if not a_R_on and not b_R_on:
        return True

    aR = np.asarray(a["rotation"], float)
    bR = np.asarray(b["rotation"], float)
    return aR.shape == (3,3) and np.max(np.abs(aR - bR)) <= tol


def ensure_mesh_graph_aligned(mesh, G, *, tol=1e-9):
    """
    Ensure mesh and graph share the same coordinate transform state.

    Your rule:
      - "transformed" iff transform['center'] is finite
      - if both transformed: do NOT try to align (assume they are already consistent)
      - if only one transformed: apply that transform to the other
      - if neither transformed: do nothing

    Returns
    -------
    changed : bool
        True if we applied a transform to either mesh or graph-stored coordinates.
    """
    mtr = getattr(mesh, "coord_transform", None)
    gtr = G.graph.get("coord_transform", None)

    m_has = transform_is_applied(mtr)
    g_has = transform_is_applied(gtr)

    # Case 1: both or neither transformed => don't touch anything
    if (m_has and g_has) or ((not m_has) and (not g_has)):
        # Optional: if both transformed you MAY still want to sanity-check equality:
        # if m_has and g_has and not transforms_equal(mtr, gtr, tol=tol): warn or raise.
        return False

    # Case 2: mesh transformed, graph not transformed => apply to graph coords
    if m_has and (not g_has):
        # Apply to any graph-stored coordinates that live in the same frame
        # (centroid and proj_point in your graph builder)
        for key in ("centroid", "proj_point"):
            # only if present
            if key in G.nodes[0]:
                X = np.asarray([G.nodes[i][key] for i in range(G.number_of_nodes())], float)
                X2 = apply_transform_to_points(X, mtr)
                for i in range(G.number_of_nodes()):
                    G.nodes[i][key] = X2[i].tolist() if isinstance(G.nodes[i][key], list) else X2[i]
        # Copy transform record into graph
        G.graph["coord_transform"] = mtr
        return True

    # Case 3: graph transformed, mesh not transformed => apply to mesh vertices
    if g_has and (not m_has):
        mesh.v = apply_transform_to_points(mesh.v, gtr)
        mesh.coord_transform = gtr
        return True

    return False

"""Build biology-aware organoid skeletons from crypt detections.

This module converts crypt detection outputs into a compact straight-edge graph:
body center -> neck center -> optional bend/branch nodes -> crypt tips.  It is
not a medial-axis extractor.  It deliberately keeps segmentation adapters thin
so parameters can be tuned upstream and the resulting skeleton can be rebuilt.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from organograph.skeleton.datatypes import SkeletonGraph
from organograph.skeleton.geometry import (
    as_points,
    as_vertex_indices,
    centroid,
    estimate_bend_position,
    surface_area_centroid,
)


def _first_present(mapping: dict[str, Any], names: tuple[str, ...], default=None):
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _json_safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    out = {}
    for key, value in metadata.items():
        if isinstance(value, set):
            out[key] = sorted(map(int, value))
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _mesh_like(vertices, faces):
    """Small mesh object for segmentation helpers that only need v, f, areas."""
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)

    def vertex_areas(from_mass_matrix: bool = False):
        tri = vertices[faces]
        face_areas = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
            axis=1,
        )
        areas = np.zeros(vertices.shape[0], dtype=float)
        for i in range(3):
            np.add.at(areas, faces[:, i], face_areas / 3.0)
        return areas

    return SimpleNamespace(v=vertices, f=faces, vertex_areas=vertex_areas)


def _coerce_patch(patch) -> np.ndarray:
    if patch is None:
        return np.empty(0, dtype=np.int64)
    return as_vertex_indices(patch)


def _point_from_vertex(vertices, vertex_id) -> np.ndarray | None:
    if vertex_id is None:
        return None
    vertex_id = int(vertex_id)
    if vertex_id < 0:
        return None
    return as_points(vertices)[vertex_id]


def _point_from_keys(vertices, detection: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    value = _first_present(detection, keys)
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape == (3,):
        return arr
    if arr.ndim == 0:
        return _point_from_vertex(vertices, int(arr))
    return None


def _centroid_from_vertex_keys(
    vertices,
    detection: dict[str, Any],
    keys: tuple[str, ...],
) -> np.ndarray | None:
    value = _first_present(detection, keys)
    if value is None:
        return None
    idx = _coerce_patch(value)
    if idx.size == 0:
        return None
    return centroid(as_points(vertices)[idx])


def _boundary_vertices_from_patch(faces, patch_vertices) -> np.ndarray:
    patch_vertices = _coerce_patch(patch_vertices)
    if patch_vertices.size == 0:
        return patch_vertices
    faces = np.asarray(faces, dtype=np.int64)
    keep = np.isin(faces, patch_vertices).all(axis=1)
    patch_faces = faces[keep]
    if patch_faces.size == 0:
        return patch_vertices
    edges = np.vstack(
        [
            patch_faces[:, [0, 1]],
            patch_faces[:, [1, 2]],
            patch_faces[:, [2, 0]],
        ]
    )
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if boundary_edges.size == 0:
        return patch_vertices
    return np.unique(boundary_edges.reshape(-1))


def _contour_center_from_distance_field(
    vertices,
    faces,
    dfield,
    *,
    level: float = 1.0,
    prefer_vertices=None,
    min_points: int = 3,
) -> np.ndarray | None:
    """Centroid of an isocontour component from triangle-edge intersections.

    The returned point is not forced to lie on the mesh.  For a closed neckline
    ring this produces the geometric center of the ring, which is the desired
    skeleton neck-node position.  The contour is extracted from the full mesh;
    `prefer_vertices` only helps choose a component if several are present.
    """
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    dfield = np.asarray(dfield, dtype=float).reshape(-1)
    if dfield.size != vertices.shape[0] or faces.size == 0:
        return None

    face_mask = np.isfinite(dfield[faces]).all(axis=1)
    candidate_faces = faces[face_mask]
    if candidate_faces.size == 0:
        return None

    level = float(level)
    eps = 1e-12
    segments = []

    def edge_intersection(a: int, b: int) -> np.ndarray | None:
        da = float(dfield[a])
        db = float(dfield[b])
        if abs(da - level) <= eps and abs(db - level) <= eps:
            return None
        if abs(da - level) <= eps:
            return vertices[a]
        if abs(db - level) <= eps:
            return vertices[b]
        if (da - level) * (db - level) > 0.0:
            return None
        t = (level - da) / (db - da)
        if -eps <= t <= 1.0 + eps:
            t = float(np.clip(t, 0.0, 1.0))
            return vertices[a] + t * (vertices[b] - vertices[a])
        return None

    for tri in candidate_faces:
        edge_points = []
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a = int(a)
            b = int(b)
            if abs(float(dfield[a]) - level) <= eps and abs(float(dfield[b]) - level) <= eps:
                edge_points.extend([vertices[a], vertices[b]])
                continue
            p = edge_intersection(a, b)
            if p is not None:
                edge_points.append(p)

        unique = []
        for p in edge_points:
            if not any(np.linalg.norm(p - q) <= 1e-10 for q in unique):
                unique.append(p)
        if len(unique) < 2:
            continue
        if len(unique) == 2:
            segments.append((unique[0], unique[1]))
        else:
            # Degenerate level-through-vertex cases can yield >2 points.  Connect
            # them around their centroid to keep the component closed enough for
            # center estimation.
            c = centroid(np.asarray(unique, dtype=float))
            normal = np.cross(unique[1] - unique[0], unique[2] - unique[0])
            if np.linalg.norm(normal) <= 1e-12:
                order = range(len(unique))
            else:
                axis0 = unique[0] - c
                n0 = np.linalg.norm(axis0)
                if n0 <= 1e-12:
                    order = range(len(unique))
                else:
                    axis0 = axis0 / n0
                    axis1 = np.cross(normal, axis0)
                    axis1 = axis1 / max(np.linalg.norm(axis1), 1e-12)
                    angles = [np.arctan2(np.dot(p - c, axis1), np.dot(p - c, axis0)) for p in unique]
                    order = np.argsort(angles)
            ordered = [unique[i] for i in order]
            for p0, p1 in zip(ordered, ordered[1:] + ordered[:1]):
                segments.append((p0, p1))

    if not segments:
        return None

    points = []
    adjacency = []
    key_to_index = {}

    def point_key(p: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.round(p / 1e-8).astype(np.int64).tolist())

    def add_point(p: np.ndarray) -> int:
        key = point_key(p)
        if key in key_to_index:
            return key_to_index[key]
        idx = len(points)
        key_to_index[key] = idx
        points.append(p)
        adjacency.append(set())
        return idx

    for p0, p1 in segments:
        i0 = add_point(np.asarray(p0, dtype=float))
        i1 = add_point(np.asarray(p1, dtype=float))
        if i0 == i1:
            continue
        adjacency[i0].add(i1)
        adjacency[i1].add(i0)

    if len(points) < int(min_points):
        return None

    visited = np.zeros(len(points), dtype=bool)
    components = []
    for start in range(len(points)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in adjacency[i]:
                if not visited[j]:
                    visited[j] = True
                    stack.append(j)
        if len(comp) >= int(min_points):
            components.append(comp)

    if not components:
        return None

    prefer = _coerce_patch(prefer_vertices)
    if prefer.size:
        prefer_points = vertices[prefer]

        def component_score(comp):
            pts = np.asarray([points[i] for i in comp], dtype=float)
            center = centroid(pts)
            min_dist = float(np.min(np.linalg.norm(prefer_points - center, axis=1)))
            return (-min_dist, len(comp))

        best = max(components, key=component_score)
    else:
        best = max(components, key=len)

    pts = np.asarray([points[i] for i in best], dtype=float)
    if pts.shape[0] < int(min_points):
        return None
    return centroid(pts)


def _neck_from_distance_field(
    vertices,
    faces,
    detection: dict[str, Any],
    *,
    tolerance: float = 0.05,
) -> np.ndarray | None:
    dfield = _first_present(detection, ("d_crypt", "distance_field", "dnorm", "dnorm_vertices"))
    if dfield is None:
        return None
    dfield = np.asarray(dfield, dtype=float).reshape(-1)
    if dfield.size != as_points(vertices).shape[0]:
        return None

    patch = _coerce_patch(
        _first_present(
            detection,
            ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
        )
    )

    ring_center = _contour_center_from_distance_field(
        vertices,
        faces,
        dfield,
        level=float(detection.get("neck_level", 1.0)),
        prefer_vertices=patch,
    )
    if ring_center is not None:
        return ring_center

    if patch.size:
        valid = patch[np.isfinite(dfield[patch])]
    else:
        valid = np.where(np.isfinite(dfield))[0]
    if valid.size == 0:
        return None

    delta = np.abs(dfield[valid] - 1.0)
    near = valid[delta <= float(tolerance)]
    if near.size == 0:
        best = np.nanmin(delta)
        near = valid[delta <= best + 1e-12]
    return centroid(as_points(vertices)[near])


def _neck_position(vertices, faces, detection: dict[str, Any]) -> np.ndarray:
    explicit = _point_from_keys(
        vertices,
        detection,
        ("neck_center", "neck_position", "neck", "neckline_center", "p_neck"),
    )
    if explicit is not None:
        return explicit

    from_vertices = _centroid_from_vertex_keys(
        vertices,
        detection,
        ("neck_vertices", "neckline_vertices", "boundary_vertices"),
    )
    if from_vertices is not None:
        return from_vertices

    from_distance = _neck_from_distance_field(vertices, faces, detection)
    if from_distance is not None:
        return from_distance

    patch = _coerce_patch(
        _first_present(
            detection,
            ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
        )
    )
    if patch.size:
        boundary = _boundary_vertices_from_patch(faces, patch)
        return centroid(as_points(vertices)[boundary])

    raise ValueError(
        "Crypt detection is missing neck coordinates, neck vertices, "
        "a normalized distance field, or patch vertices."
    )


def _tip_position(vertices, detection: dict[str, Any]) -> np.ndarray:
    explicit = _point_from_keys(
        vertices,
        detection,
        ("tip_position", "tip_center", "tip", "bottom_position", "crypt_tip", "p_tip"),
    )
    if explicit is not None:
        return explicit

    vertex_id = _first_present(
        detection,
        ("tip_vertex_id", "bottom_vertex_id", "bottom", "bottom_vertex"),
    )
    by_vertex = _point_from_vertex(vertices, vertex_id)
    if by_vertex is not None:
        return by_vertex

    patch = _coerce_patch(
        _first_present(
            detection,
            ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
        )
    )
    if patch.size:
        return centroid(as_points(vertices)[patch])
    raise ValueError("Crypt detection is missing a tip/bottom position or vertex id.")


def _branch_position(vertices, detection: dict[str, Any], neck, daughter_tips) -> np.ndarray:
    explicit = _point_from_keys(
        vertices,
        detection,
        ("branch_position", "branch_center", "branch", "split_position", "split_center"),
    )
    if explicit is not None:
        return explicit

    vertex_id = _first_present(detection, ("branch_vertex_id", "split_vertex_id"))
    by_vertex = _point_from_vertex(vertices, vertex_id)
    if by_vertex is not None:
        return by_vertex

    stem_vertices = _coerce_patch(_first_present(detection, ("stem_vertices", "trunk_vertices")))
    if stem_vertices.size:
        return centroid(as_points(vertices)[stem_vertices])

    daughter_mean = centroid(np.vstack(daughter_tips))
    return 0.5 * (np.asarray(neck, dtype=float) + daughter_mean)


def _daughter_detections(detection: dict[str, Any]) -> list[dict[str, Any]]:
    daughters = _first_present(detection, ("daughters", "daughter_tips", "branches", "children"))
    if daughters is None:
        return []
    out = []
    for daughter in daughters:
        if isinstance(daughter, dict):
            out.append(dict(daughter))
        else:
            arr = np.asarray(daughter)
            if arr.shape == (3,):
                out.append({"tip_position": arr})
            else:
                out.append({"tip_vertex_id": int(arr)})
    return out


def normalize_crypt_detections(crypt_detections) -> list[dict[str, Any]]:
    """Normalize common segmentation outputs to a list of detection dicts.

    Accepted inputs include:
    - list of dicts with explicit neck/tip fields;
    - list of vertex-index patches;
    - segmentation dictionaries containing `crypts_mesh`, `crypts_ll`, or
      `crypts`, optionally with per-crypt arrays such as `bottom_vertex_ids`
      and `d_crypts`.
    """
    if crypt_detections is None:
        return []

    if isinstance(crypt_detections, dict):
        patches = _first_present(
            crypt_detections,
            ("crypt_detections", "crypts_mesh", "crypts_ll", "crypts", "patches"),
        )
        if patches is not None and not isinstance(patches, dict):
            if all(isinstance(patch, dict) for patch in patches):
                return [dict(patch, crypt_id=patch.get("crypt_id", i)) for i, patch in enumerate(patches)]
            detections = []
            for i, patch in enumerate(patches):
                det = {"crypt_id": i, "crypt_vertices": patch}
                for src_key, dst_key in (
                    ("bottom_vertex_ids", "bottom_vertex_id"),
                    ("tip_vertex_ids", "tip_vertex_id"),
                    ("d_crypts", "d_crypt"),
                    ("L_crypts", "L_crypt"),
                    ("circumference_crypts", "circumference"),
                    ("crypt_constrictions", "constriction"),
                    ("crypt_elongations", "elongation"),
                ):
                    if src_key in crypt_detections:
                        values = crypt_detections[src_key]
                        if len(values) > i:
                            det[dst_key] = values[i]
                detections.append(det)
            return detections
        return [dict(crypt_detections)]

    detections = []
    for i, item in enumerate(crypt_detections):
        if isinstance(item, dict):
            det = dict(item)
            det.setdefault("crypt_id", i)
        else:
            det = {"crypt_id": i, "crypt_vertices": item}
        detections.append(det)
    return detections


def _body_center(vertices, faces, body_vertices, body_faces, body_center) -> np.ndarray:
    if body_center is not None:
        center = np.asarray(body_center, dtype=float)
        if center.shape != (3,):
            raise ValueError("body_center must be a 3-vector")
        return center
    vertices = as_points(vertices)
    if body_vertices is not None:
        idx = _coerce_patch(body_vertices)
        if idx.size:
            return centroid(vertices[idx])
    if body_faces is not None:
        return surface_area_centroid(vertices, np.asarray(body_faces, dtype=np.int64))
    if faces is not None:
        return surface_area_centroid(vertices, faces)
    return centroid(vertices)


def build_skeleton_from_crypt_detections(
    vertices,
    faces,
    crypt_detections,
    body_vertices=None,
    body_faces=None,
    body_center=None,
    add_bend_nodes: bool = False,
    bend_strategy: str = "none",
    metadata: dict[str, Any] | None = None,
) -> SkeletonGraph:
    """Build a straight-edge organoid skeleton from crypt detections.

    Each non-split crypt is represented as `body -> neck -> tip`.  When bend
    nodes are requested, the crypt path becomes `neck -> bend -> tip`.  Split
    detections with daughters become `neck -> branch -> daughter tips`.
    """
    vertices = as_points(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    detections = normalize_crypt_detections(crypt_detections)

    graph = SkeletonGraph(
        metadata=_json_safe_metadata(metadata),
        coordinate_frame={
            "kind": "raw",
            "body_center_node": "body",
            "description": "Raw mesh/world coordinates; edges are straight segments.",
        },
    )
    graph.add_node(
        "body",
        "body",
        _body_center(vertices, faces, body_vertices, body_faces, body_center),
        metadata={"role": "villus_body_center"},
    )

    for i, detection in enumerate(detections):
        crypt_id = detection.get("crypt_id", i)
        crypt_prefix = f"crypt_{crypt_id}"
        crypt_vertices = _coerce_patch(
            _first_present(
                detection,
                ("crypt_vertices", "patch_vertices", "vertex_ids", "vertices", "patch"),
            )
        )
        common_meta = _json_safe_metadata(
            {
                "source_detection_index": i,
                "n_crypt_vertices": int(crypt_vertices.size),
                **dict(detection.get("metadata", {})),
            }
        )

        neck = _neck_position(vertices, faces, detection)
        neck_id = f"{crypt_prefix}_neck"
        graph.add_node(
            neck_id,
            "neck",
            neck,
            crypt_id=crypt_id,
            metadata=common_meta,
        )
        graph.add_edge(
            f"{crypt_prefix}_body_to_neck",
            "body",
            neck_id,
            edge_type="body_to_neck",
            crypt_id=crypt_id,
        )

        daughters = _daughter_detections(detection)
        if daughters:
            daughter_tips = [_tip_position(vertices, daughter) for daughter in daughters]
            branch = _branch_position(vertices, detection, neck, daughter_tips)
            branch_id = f"{crypt_prefix}_branch"
            graph.add_node(
                branch_id,
                "branch",
                branch,
                crypt_id=crypt_id,
                metadata={**common_meta, "n_daughters": len(daughters)},
            )
            graph.add_edge(
                f"{crypt_prefix}_neck_to_branch",
                neck_id,
                branch_id,
                edge_type="neck_to_branch",
                crypt_id=crypt_id,
            )
            for j, daughter in enumerate(daughters):
                tip_id = f"{crypt_prefix}_tip_{j}"
                graph.add_node(
                    tip_id,
                    "tip",
                    daughter_tips[j],
                    crypt_id=crypt_id,
                    metadata=_json_safe_metadata(
                        {
                            **common_meta,
                            "daughter_index": j,
                            **dict(daughter.get("metadata", {})),
                        }
                    ),
                )
                graph.add_edge(
                    f"{crypt_prefix}_branch_to_tip_{j}",
                    branch_id,
                    tip_id,
                    edge_type="branch_to_tip",
                    crypt_id=crypt_id,
                )
            continue

        tip = _tip_position(vertices, detection)
        tip_id = f"{crypt_prefix}_tip"
        graph.add_node(
            tip_id,
            "tip",
            tip,
            crypt_id=crypt_id,
            metadata=common_meta,
        )

        bend = None
        if add_bend_nodes or bend_strategy != "none":
            bend = _point_from_keys(
                vertices,
                detection,
                ("bend_position", "bend_center", "bend", "p_bend"),
            )
            if bend is None:
                bend = estimate_bend_position(
                    vertices,
                    crypt_vertices,
                    neck,
                    tip,
                    strategy=bend_strategy,
                )

        if bend is None:
            graph.add_edge(
                f"{crypt_prefix}_neck_to_tip",
                neck_id,
                tip_id,
                edge_type="neck_to_tip",
                crypt_id=crypt_id,
            )
        else:
            bend_id = f"{crypt_prefix}_bend"
            graph.add_node(
                bend_id,
                "bend",
                bend,
                crypt_id=crypt_id,
                metadata={**common_meta, "bend_strategy": bend_strategy},
            )
            graph.add_edge(
                f"{crypt_prefix}_neck_to_bend",
                neck_id,
                bend_id,
                edge_type="neck_to_bend",
                crypt_id=crypt_id,
            )
            graph.add_edge(
                f"{crypt_prefix}_bend_to_tip",
                bend_id,
                tip_id,
                edge_type="bend_to_tip",
                crypt_id=crypt_id,
            )

    return graph


def detect_crypts_for_skeleton(
    mesh,
    vocab,
    *,
    geodesic_fn,
    L_ref=None,
    crypt_vocab_idx=None,
    threshold=0.5,
    filter_fn_list=None,
    refine_crypts=True,
    refine_threshold=0.0,
    refine_only_if_area_at_least=5.0,
    min_refined_frac_of_parent=0.1,
    geodesic_kwargs=None,
    extend_max=2.0,
    disc_resolution=200,
    neck_search_interval=(0.8, 2.0),
    return_intermediates=False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run crypt detection pieces needed for skeleton construction.

    This adapter intentionally starts from a fresh HKS candidate screen.  Parent
    candidate patches are kept as skeleton crypt trunks; if local refinement
    splits a parent into multiple child patches, the output marks that parent as
    a split crypt with daughter tips.  This preserves stem/neck regions that may
    later be grouped with the villus in final saved segmentations.
    """
    from organograph.crypts.axis import compute_crypt_axis, normalize_crypt_axis_to_neckline
    from organograph.crypts.filters import apply_filters
    from organograph.crypts.vocab import detect_crypts_by_encoding, subdivide_crypts_by_encoding

    if geodesic_kwargs is None:
        geodesic_kwargs = {}

    parents, enc_vars = detect_crypts_by_encoding(
        vocab,
        mesh,
        L_ref=L_ref,
        crypt_vocab_idx=crypt_vocab_idx,
        threshold=threshold,
        return_intermediates=True,
    )

    seg_vars = {
        "encoding": enc_vars.get("encoding"),
        "ts_mesh": enc_vars.get("ts_mesh"),
        "ts_vocab": enc_vars.get("ts_vocab"),
        "hks": enc_vars.get("hks"),
        "norm_hks": enc_vars.get("norm_hks"),
        "hks_segment": enc_vars.get("hks"),
        "normalised_hks_segment": enc_vars.get("norm_hks"),
        "vertex_areas": np.asarray(mesh.vertex_areas(), float),
    }
    if filter_fn_list is not None:
        parents, filter_info, keep_idx = apply_filters(
            parents,
            filters=filter_fn_list,
            mesh=mesh,
            seg_vars=seg_vars,
        )
        seg_vars["filter_info_initial"] = filter_info
        seg_vars["keep_idx_initial"] = keep_idx

    dnorm_parent, L_parent, bottom_parent = compute_crypt_axis(
        mesh,
        parents,
        geodesic_fn,
        geodesic_kwargs=geodesic_kwargs,
    )
    d_levels = np.linspace(0.01, float(extend_max), int(disc_resolution))
    _, dnorm_parent, L_parent = normalize_crypt_axis_to_neckline(
        mesh,
        dnorm_parent,
        d_levels,
        search_interval=neck_search_interval,
        L_crypt=L_parent,
    )

    detections = []
    refined_by_parent = []
    for i, parent in enumerate(parents):
        daughters = []
        if refine_crypts:
            refined = subdivide_crypts_by_encoding(
                vocab,
                mesh,
                L_ref=L_ref,
                crypt_vocab_idx=crypt_vocab_idx,
                patches=[parent],
                threshold=refine_threshold,
                refine_only_if_area_at_least=refine_only_if_area_at_least,
                min_refined_frac_of_parent=min_refined_frac_of_parent,
            )
        else:
            refined = [parent]
        refined_by_parent.append(refined)

        if len(refined) > 1:
            dnorm_child, L_child, bottom_child = compute_crypt_axis(
                mesh,
                refined,
                geodesic_fn,
                geodesic_kwargs=geodesic_kwargs,
            )
            _, dnorm_child, L_child = normalize_crypt_axis_to_neckline(
                mesh,
                dnorm_child,
                d_levels,
                search_interval=neck_search_interval,
                L_crypt=L_child,
            )
            daughter_union = set().union(*[set(child) for child in refined])
            stem_vertices = sorted(set(parent).difference(daughter_union))
            for j, child in enumerate(refined):
                daughters.append(
                    {
                        "crypt_id": f"{i}.{j}",
                        "crypt_vertices": child,
                        "bottom_vertex_id": int(bottom_child[j]),
                        "d_crypt": dnorm_child[j],
                        "L_crypt": float(L_child[j]),
                    }
                )
        else:
            stem_vertices = []

        det = {
            "crypt_id": i,
            "crypt_vertices": parent,
            "bottom_vertex_id": int(bottom_parent[i]),
            "d_crypt": dnorm_parent[i],
            "L_crypt": float(L_parent[i]),
            "metadata": {"detection_stage": "fresh_initial_candidate"},
        }
        if daughters:
            det["daughters"] = daughters
            det["stem_vertices"] = stem_vertices
        detections.append(det)

    intermediates = {
        "initial_patches": parents,
        "refined_by_parent": refined_by_parent,
        "encoding": enc_vars,
        "d_levels": d_levels,
        "dnorm_parent": dnorm_parent,
        "L_parent": L_parent,
        "bottom_parent": bottom_parent,
        **seg_vars,
    }
    if return_intermediates:
        return detections, intermediates
    return detections


def build_skeleton_from_segmentation_parameters(
    mesh,
    vocab,
    *,
    geodesic_fn,
    build_kwargs: dict[str, Any] | None = None,
    detection_kwargs: dict[str, Any] | None = None,
) -> SkeletonGraph:
    """Convenience wrapper that reruns detection and builds a skeleton."""
    detection_kwargs = dict(detection_kwargs or {})
    build_kwargs = dict(build_kwargs or {})
    detection_kwargs.pop("return_intermediates", None)
    detections = detect_crypts_for_skeleton(
        mesh,
        vocab,
        geodesic_fn=geodesic_fn,
        **detection_kwargs,
    )
    return build_skeleton_from_crypt_detections(
        vertices=mesh.v,
        faces=mesh.f,
        crypt_detections=detections,
        **build_kwargs,
    )

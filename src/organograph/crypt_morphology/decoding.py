"""Direct coefficient decoding and controlled tests of imperfect predictions.

Generated coefficients are plain arrays in an explicitly chosen common frame.
They do not inherit a source mesh's pose, area, size or provenance. No latent
model, coefficient refitting, or inverse parameterization is involved here.
"""
from __future__ import annotations

import numpy as np


def _coefficients(coefficients, basis):
    c = np.asarray(coefficients, float)
    if c.shape != (basis.n_modes, 3) or not np.isfinite(c).all():
        raise ValueError("Expected finite (basis.n_modes, 3) coordinate coefficients")
    return c


def decode_coefficients(coefficients, basis, theta_phi=None, *, n_rings=24, n_sectors=96):
    """Evaluate XYZ directly, in the coefficient frame and size convention.

    With supplied coordinates, return vertices only. Otherwise return vertices
    and the common cap triangulation. A generated surface has no original pose
    unless the caller explicitly assigns one.
    """
    c = _coefficients(coefficients, basis)
    if theta_phi is not None:
        return basis.evaluate(theta_phi) @ c
    q, faces = basis.reference_mesh(n_rings, n_sectors)
    return basis.evaluate(q) @ c, faces


def interpolate_coefficients(first, second, fraction, basis):
    """Linear interpolation; callers must supply the same aligned frame/units.

    This interpolates matched cap coordinates, not anatomical correspondences
    or a surface geodesic. Valid endpoints do not guarantee a valid midpoint.
    """
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must lie in [0, 1]")
    a, b = _coefficients(first, basis), _coefficients(second, basis)
    return (1-fraction)*a + fraction*b


def rim_coefficient_error(basis, rms_displacement, *, max_mode=3, seed=2026):
    """A smooth, mean-zero error in opening XYZ, with an exact angular RMS.

    Only nonconstant Fourier rim modes are changed; all interior coefficients
    remain zero. RMS uses uniform rim angle, not physical rim arclength. The
    three-vector Fourier coefficients give RMS² = sum(||a_m||²+||b_m||²)/2.
    The returned DELTA is in the caller's coordinate units. Its disk harmonic
    extension is smooth and attenuates each mode by r**m toward the cap pole.
    It is a controlled error direction, not a learned decoder-error model.
    """
    if not np.isfinite(rms_displacement) or rms_displacement < 0:
        raise ValueError("rms_displacement must be finite and nonnegative")
    if isinstance(max_mode, bool) or int(max_mode) != max_mode or not 1 <= max_mode <= basis.max_degree:
        raise ValueError("max_mode must be a positive integer supported by the basis")
    ids = [i for i, (kind, m, *_rest) in enumerate(basis.modes) if kind == "rim" and 1 <= m <= max_mode]
    delta = np.zeros((basis.n_modes, 3))
    delta[ids] = np.random.default_rng(seed).normal(size=(len(ids), 3))
    delta *= rms_displacement / np.sqrt(np.sum(delta**2)/2)
    return delta


def transport_coefficient_error(delta, basis, source, target):
    """Express the SAME physical perturbation in another pose/phase/size frame.

    ``source`` and ``target`` must be representations of the same source shape,
    differing only by size convention/alignment. Translation does not act on a
    displacement. This allows physical and unit-RMS tests to use identical
    physical errors, rather than different random directions in rotated frames.
    """
    if source.basis_id != basis.identifier or target.basis_id != basis.identifier:
        raise ValueError("Basis mismatch")
    c = _coefficients(delta, basis)
    rotation = target.pose.rotation @ source.pose.rotation.T
    return (basis.phase_shift(c, target.phase-source.phase) @ rotation.T
            * (source.pose.scale/target.pose.scale))


def coefficient_surface_distance(first, second, basis, theta_phi, weights):
    """Weighted XYZ RMS on a fixed grid; no realignment or coefficient norm."""
    delta = decode_coefficients(_coefficients(first, basis)-_coefficients(second, basis), basis, theta_phi)
    w = np.asarray(weights, float)
    if w.shape != (len(delta),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Expected nonnegative finite sampling weights with positive sum")
    return float(np.sqrt(np.sum(w[:, None]*delta**2)/w.sum()))


class DecodingProbe:
    """Frozen sampling and geometry checks around one aligned baseline crypt.

    Core samples are centroids of the caller's fixed source triangles, rather
    than the three interior points used for fitting. Correspondence displacement
    evaluates their *baseline cap coordinates* on each generated surface; it
    includes changes in parameterization. The separate point-to-mesh core error
    compares actual baseline membrane points to the generated surface and is
    insensitive to tangential reparameterization. Neither is a Hausdorff bound.
    """

    def __init__(self, patch, mapping, baseline, basis, core_face_mask, *, n_rings=24, n_sectors=96):
        from .parameterization import disk_to_cap
        from .patches import triangle_areas, vertex_areas
        if baseline.basis_id != basis.identifier:
            raise ValueError("Baseline basis mismatch")
        if mapping.uv.shape != (len(patch.vertices), 2) or abs(mapping.half_angle-basis.half_angle) > 1e-12:
            raise ValueError("Patch/mapping/basis mismatch")
        mask = np.asarray(core_face_mask)
        if mask.dtype != bool or mask.shape != (len(patch.faces),) or not mask.any():
            raise ValueError("core_face_mask must select at least one source triangle")
        self.basis = basis
        self.baseline_pose = baseline.pose
        self.baseline_coefficients = _coefficients(baseline.coefficients, basis).copy()
        self.theta_phi, self.faces = basis.reference_mesh(n_rings, n_sectors)
        self.matrix = basis.evaluate(self.theta_phi)
        self.baseline_vertices = self.matrix @ self.baseline_coefficients
        self.weights = vertex_areas(self.baseline_vertices, self.faces)
        self.weights /= self.weights.sum()
        self.radius = baseline.rms_radius / baseline.pose.scale
        core_faces = patch.faces[mask]
        self.core_theta_phi = disk_to_cap(mapping.uv[core_faces].mean(axis=1), basis.half_angle)
        self.core_theta_phi[:, 1] -= baseline.phase
        self.core_matrix = basis.evaluate(self.core_theta_phi)
        self.baseline_core = self.core_matrix @ self.baseline_coefficients
        self.core_targets = baseline.pose.apply(patch.vertices[core_faces].mean(axis=1))
        areas = triangle_areas(patch.vertices, patch.faces)
        self.core_weights = areas[mask]/areas[mask].sum()
        self.core_area_fraction = float(areas[mask].sum()/areas.sum())
        phi = np.arange(1024)*2*np.pi/1024
        self.rim_matrix = basis.evaluate(np.column_stack([np.full(len(phi), basis.half_angle), phi]))
        self.n_rings, self.n_sectors = n_rings, n_sectors

    def source_frame_control(self, fitted_representation):
        """Check an OBSERVED variant using its known original membrane pose.

        Restore a fitted variant to source coordinates, then apply the baseline
        pose/scale. This removes both differing alignment poses and differing
        size normalizers, exposing geometric fitting error on the common source
        core. It requires the same parent mesh and is unavailable for generated
        coefficient arrays without a known source transform. It never changes
        experiment coefficients or the rendered/interpolated predictions.
        """
        import igl
        rep = fitted_representation
        if rep.basis_id != self.basis.identifier:
            raise ValueError("Fitted variant basis mismatch")
        vertices = self.matrix @ _coefficients(rep.coefficients, self.basis)
        restored = self.baseline_pose.apply(rep.pose.inverse(vertices))
        distances2 = np.maximum(igl.point_mesh_squared_distance(self.core_targets, restored, self.faces)[0], 0)
        rotation = rep.pose.rotation @ self.baseline_pose.rotation.T
        angle = np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1)))
        return dict(core_source_frame_rmse_relative=float(np.sqrt(self.core_weights @ distances2)/self.radius),
                    source_pose_rotation_degrees=float(angle),
                    normalization_scale_ratio=float(rep.pose.scale/self.baseline_pose.scale))

    def evaluate(self, coefficients, *, check_intersections=True):
        """Return measurements and decoded vertices without realigning the output."""
        import igl
        from .mesh_quality import mesh_quality_metrics
        c = _coefficients(coefficients, self.basis)
        delta = c-self.baseline_coefficients
        vertices = self.matrix @ c
        core_delta, rim_delta = self.core_matrix @ delta, self.rim_matrix @ delta
        distances2 = np.maximum(igl.point_mesh_squared_distance(self.core_targets, vertices, self.faces)[0], 0)
        metrics = mesh_quality_metrics(vertices, self.faces, reference_vertices=self.baseline_vertices,
                                       radius=self.radius, check_intersections=check_intersections)
        metrics.update(
            core_parameter_rms_relative=float(np.sqrt(np.sum(self.core_weights[:, None]*core_delta**2))/self.radius),
            core_parameter_max_relative=float(np.linalg.norm(core_delta, axis=1).max()/self.radius),
            core_surface_rmse_relative=float(np.sqrt(self.core_weights @ distances2)/self.radius),
            core_surface_sampled_max_relative=float(np.sqrt(distances2.max())/self.radius),
            rim_rms_relative=float(np.sqrt(np.mean(np.sum(rim_delta**2, axis=1)))/self.radius),
            rim_max_relative=float(np.linalg.norm(rim_delta, axis=1).max()/self.radius),
            surface_rms_relative=float(np.sqrt(np.sum(self.weights[:, None]*(vertices-self.baseline_vertices)**2))/self.radius),
            radius_in_coefficient_units=float(self.radius), core_area_fraction=self.core_area_fraction,
            core_sampling="source_triangle_centroids", n_rings=self.n_rings, n_sectors=self.n_sectors,
        )
        return metrics, vertices

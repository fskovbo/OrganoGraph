"""A shared cap dictionary with an independently representable open rim.

Interior modes are real Dirichlet spherical-cap harmonics. Smooth harmonic
extensions of Fourier rim modes supply the boundary values. This avoids both
collapsing the opening (Dirichlet-only coordinates) and forcing zero transverse
derivatives at the opening (a finite Neumann-only expansion).

The combined dictionary is NOT orthogonal; coefficients are obtained by weighted
least squares. Its definition and numerical normalization are serialized together.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np
from scipy.optimize import brentq
from scipy.special import lpmv, roots_legendre


SCHEMA = "organograph_cap_basis_v1"


@dataclass(frozen=True)
class CapBasis:
    half_angle: float
    max_degree: int
    modes: tuple

    @classmethod
    def build(cls, max_degree=10, half_angle=np.pi / 2):
        if isinstance(max_degree, bool) or int(max_degree) != max_degree or not 1 <= max_degree <= 32:
            raise ValueError("max_degree must be an integer between 1 and 32")
        # The pilot implementation deliberately bounds special-function arguments.
        if not np.isfinite(half_angle) or not np.pi / 6 <= half_angle <= np.pi / 2:
            raise ValueError("Supported fixed cap angles are pi/6 through pi/2")
        degree, angle = int(max_degree), float(half_angle)
        nodes, qw = roots_legendre(max(128, 8 * degree))
        x = (1 + np.cos(angle)) / 2 + nodes * (1 - np.cos(angle)) / 2
        qw = qw * (1 - np.cos(angle)) / 2
        modes = [("rim", 0, 0.0, "cos", 1.0)]
        for m in range(1, degree + 1):
            modes.extend([("rim", m, float(m), trig, 1.0) for trig in ("cos", "sin")])
        for m in range(degree):
            count = (degree - m + 1) // 2
            if np.isclose(angle, np.pi / 2, atol=1e-14, rtol=0):
                nus = [float(m + 1 + 2 * j) for j in range(count)]
            else:
                nus = []
                left = float(m) + 1e-7
                fl = lpmv(m, left, np.cos(angle))
                while len(nus) < count and left < 512:
                    right = left + 0.25
                    fr = lpmv(m, right, np.cos(angle))
                    if not np.isfinite([fl, fr]).all():
                        raise ValueError("Cap basis evaluation overflow; lower expansion order")
                    if fl * fr < 0:
                        nus.append(float(brentq(lambda nu: lpmv(m, nu, np.cos(angle)), left, right)))
                    left, fl = right, fr
                if len(nus) != count:
                    raise ValueError("Could not bracket cap eigenvalues")
            for nu in nus:
                radial_norm = np.sum(qw * lpmv(m, nu, x) ** 2)
                normalization = float(1 / np.sqrt(radial_norm * (2 * np.pi if m == 0 else np.pi)))
                for trig in (("cos",) if m == 0 else ("cos", "sin")):
                    modes.append(("interior", m, nu, trig, normalization))
        return cls(angle, degree, tuple(modes))

    @property
    def n_modes(self):
        return len(self.modes)

    @property
    def identifier(self):
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self):
        return {"schema": SCHEMA, "half_angle": self.half_angle,
                "max_degree": self.max_degree, "modes": [list(m) for m in self.modes]}

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != SCHEMA:
            raise ValueError("Unknown cap basis schema")
        basis = cls(float(data["half_angle"]), int(data["max_degree"]),
                    tuple(tuple(m) for m in data["modes"]))
        # Check the schema structure, without recomputing floating point eigenvalues.
        if not basis.modes or basis.modes[0] != ("rim", 0, 0.0, "cos", 1.0):
            raise ValueError("Invalid constant mode")
        for kind, m, nu, trig, norm in basis.modes:
            if kind not in ("rim", "interior") or trig not in ("cos", "sin"):
                raise ValueError("Invalid mode")
            if not np.isfinite([m, nu, norm]).all() or m < 0 or norm <= 0:
                raise ValueError("Invalid mode parameters")
        return basis

    def evaluate(self, theta_phi):
        q = np.asarray(theta_phi, float)
        if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all():
            raise ValueError("theta_phi must be finite (N, 2) coordinates")
        th, ph = q.T
        if np.any(th < -1e-10) or np.any(th > self.half_angle + 1e-10):
            raise ValueError("Coordinates lie outside the fixed cap")
        th = np.clip(th, 0, self.half_angle)
        r = np.tan(th / 2) / np.tan(self.half_angle / 2)
        columns = []
        for kind, m, nu, trig, norm in self.modes:
            if kind == "rim":
                radial = r ** m
            else:
                radial = norm * lpmv(m, nu, np.cos(th))
                radial = np.where(np.isclose(th, self.half_angle, atol=1e-12, rtol=0), 0, radial)
            columns.append(radial * (np.cos(m * ph) if trig == "cos" else np.sin(m * ph)))
        result = np.column_stack(columns)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite cap basis evaluation")
        return result

    def phase_shift(self, coefficients, phase):
        """Coefficients of X(theta, phi + phase), without refitting."""
        c = np.asarray(coefficients, float)
        if c.shape[0] != self.n_modes:
            raise ValueError("Wrong number of coefficients")
        result = c.copy()
        i = 0
        while i < self.n_modes:
            m = self.modes[i][1]
            if m:
                if self.modes[i][3] != "cos" or self.modes[i + 1][3] != "sin":
                    raise ValueError("Invalid sine/cosine mode ordering")
                a, b = np.cos(m * phase), np.sin(m * phase)
                result[i] = a * c[i] + b * c[i + 1]
                result[i + 1] = -b * c[i] + a * c[i + 1]
                i += 2
            else:
                i += 1
        return result

    def regularization_weights(self):
        return np.array([0.0 if kind == "rim" and m == 0 else
                         (m ** 2 if kind == "rim" else nu * (nu + 1))
                         for kind, m, nu, _, _ in self.modes])

    def reference_mesh(self, n_rings=24, n_sectors=96):
        """Common cap samples and connectivity; no duplicated pole/seam vertices."""
        if n_rings < 2 or n_sectors < 3:
            raise ValueError("Reference mesh requires >=2 rings and >=3 sectors")
        theta = np.repeat(np.linspace(0, self.half_angle, n_rings + 1)[1:], n_sectors)
        phi = np.tile(np.arange(n_sectors) * 2 * np.pi / n_sectors, n_rings)
        q = np.vstack([[0, 0], np.column_stack([theta, phi])])
        faces = [(0, 1 + j, 1 + (j + 1) % n_sectors) for j in range(n_sectors)]
        for ring in range(n_rings - 1):
            start = 1 + ring * n_sectors
            for j in range(n_sectors):
                a, b = start + j, start + (j + 1) % n_sectors
                c, d = a + n_sectors, b + n_sectors
                faces.extend([(a, c, b), (b, c, d)])
        return q, np.asarray(faces, dtype=np.int64)

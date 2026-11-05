"""Plane mathematics (no Streamlit code here). Units are never converted."""
from dataclasses import dataclass
import numpy as np

COLLINEAR_TOL = 1e-9   # min sin(angle) between u and v
C_TOL = 1e-6           # min |C|/|n| for Z to be computable from X,Y
VERIFY_REL_TOL = 1e-9  # residual tolerance, relative to |n| * coordinate scale


class PlaneError(ValueError):
    """Raised when reference points are invalid or degenerate."""


@dataclass
class Plane:
    points: np.ndarray       # 3x3 (P1,P2,P3)
    u: np.ndarray
    v: np.ndarray
    n: np.ndarray
    A: float
    B: float
    C: float
    D: float
    normal_mag: float
    sin_angle: float         # |u x v| / (|u||v|); 0 means collinear
    c_ratio: float           # |C| / |n|; 0 means vertical plane
    z_usable: bool

    def residuals(self):
        p = self.points
        return self.A * p[:, 0] + self.B * p[:, 1] + self.C * p[:, 2] + self.D

    def tolerance(self):
        return VERIFY_REL_TOL * max(1.0, self.normal_mag * float(np.max(np.abs(self.points))))

    def z_at(self, x, y):
        """Z = -(Ax + By + D)/C, vectorised. Refuses when C is ~0."""
        if not self.z_usable:
            raise PlaneError("The calculated plane cannot reliably provide a unique Z coordinate "
                             "for every X-Y position because C is zero or nearly zero.")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return -(self.A * x + self.B * y + self.D) / self.C


def validate_points(points):
    """Return a 3x3 float array or raise PlaneError with a clear message."""
    if points is None or len(points) != 3:
        raise PlaneError("Exactly three reference points are required.")
    missing = [f"P{i+1} {a}" for i, p in enumerate(points) for a, v in zip("XYZ", p) if v is None]
    if missing:
        raise PlaneError("All nine coordinates must be entered. Missing: " + ", ".join(missing))
    try:
        arr = np.array(points, dtype=float)
    except (TypeError, ValueError):
        raise PlaneError("All coordinates must be numeric.")
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        raise PlaneError("All coordinates must be finite numbers (no NaN or infinity).")
    return arr


def compute_plane(points):
    arr = validate_points(points)
    p1, p2, p3 = arr
    u, v = p2 - p1, p3 - p1
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    if lu == 0 or lv == 0 or np.linalg.norm(p3 - p2) == 0:
        raise PlaneError("Two or more reference points are identical. Use three distinct points.")
    n = np.cross(u, v)
    nm = float(np.linalg.norm(n))
    sin_angle = nm / (lu * lv)
    if sin_angle < COLLINEAR_TOL:
        raise PlaneError("The three points are collinear (or nearly collinear) in 3D, so they do not "
                         f"define a stable plane (sin of angle between P1→P2 and P1→P3 = {sin_angle:.3e}).")
    A, B, C = (float(c) for c in n)
    D = float(-(A * p1[0] + B * p1[1] + C * p1[2]))
    c_ratio = abs(C) / nm
    return Plane(arr, u, v, n, A, B, C, D, nm, float(sin_angle), float(c_ratio), c_ratio > C_TOL)

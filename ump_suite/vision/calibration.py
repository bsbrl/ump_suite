"""
Camera calibration: how far the needle tip moves on screen per micrometre of each axis.

The calibration moves UMP 1 by known amounts on X, Y and D, finds the tip in
each frame, and fits

    vX, vY, vD   image displacement (pixels) per micrometre of X, Y and D

From those:

    um_per_px    1 / |vX| and 1 / |vY| agree when X and Y are horizontal and
                 square to each other; their mean is used
    theta_deg    the D axis's tilt below horizontal. D moves along the pipette,
                 so the camera sees only its horizontal part:
                 |vD| * um_per_px = cos(theta)
    J            the 2x2 matrix [vX vY]; solving J m = disp gives the X, Y move
                 (micrometres) that moves the tip by `disp` pixels on screen

The result is saved as JSON (default ~/.config/ump_suite/vision_calibration.json)
and read by the vision node, the orient skill and MicroVLA's vision step.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(os.path.expanduser("~/.config/ump_suite/vision_calibration.json"))
# X and Y scales that disagree by more than this fraction mean the axes are not
# both horizontal, or a tip was found in the wrong place.
SCALE_MISMATCH_LIMIT = 0.10


@dataclass
class Calibration:
    um_per_px: float
    vx: list
    vy: list
    vd: list
    theta_deg: float
    image_size: list
    created: float = field(default_factory=time.time)
    residual_px: float = 0.0
    notes: str = ""

    @property
    def jacobian(self) -> np.ndarray:
        """2x2 matrix whose columns are the pixel displacement per um of X and of Y."""
        return np.array([[self.vx[0], self.vy[0]], [self.vx[1], self.vy[1]]], dtype=np.float64)

    def xy_move_for_pixels(self, dx_px, dy_px) -> tuple[float, float]:
        """X and Y moves (um) that shift the tip by (dx_px, dy_px) on screen."""
        m = np.linalg.solve(self.jacobian, np.array([dx_px, dy_px], dtype=np.float64))
        return float(m[0]), float(m[1])

    def to_dict(self):
        return asdict(self)

    def save(self, path=None):
        path = Path(path or DEFAULT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)
        return path


def load(path=None) -> Calibration | None:
    """Calibration from JSON (default: DEFAULT_PATH), or None when the file does not exist."""
    path = Path(path or DEFAULT_PATH)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return Calibration(**{k: raw[k] for k in Calibration.__dataclass_fields__ if k in raw})


def fit(samples, image_size, notes="") -> Calibration:
    """
    Fit a calibration from moves.

    `samples` is a list of (axis, delta_um, (x0, y0), (x1, y1)): the axis moved
    ("X", "Y" or "D"), by how much, and the tip before and after. Each axis needs
    at least one move; several moves per axis are averaged by least squares.
    """
    per_axis = {"X": [], "Y": [], "D": []}
    for axis, delta, before, after in samples:
        axis = str(axis).upper()
        if axis not in per_axis or not delta:
            continue
        per_axis[axis].append(
            (float(delta), float(after[0]) - float(before[0]), float(after[1]) - float(before[1]))
        )
    vel, residuals = {}, []
    for axis, rows in per_axis.items():
        if not rows:
            raise ValueError(f"no usable {axis} move")
        d = np.array([r[0] for r in rows])
        disp = np.array([[r[1], r[2]] for r in rows])
        v = (d @ disp) / (d @ d)  # least squares through the origin, per component
        vel[axis] = v
        residuals.extend(np.linalg.norm(disp - np.outer(d, v), axis=1).tolist())
    sx, sy = np.linalg.norm(vel["X"]), np.linalg.norm(vel["Y"])
    if sx < 1e-6 or sy < 1e-6:
        raise ValueError("the tip did not move for X or Y")
    if abs(sx - sy) / max(sx, sy) > SCALE_MISMATCH_LIMIT:
        raise ValueError(
            f"X and Y scales disagree ({1 / sx:.3f} vs {1 / sy:.3f} um/px): "
            "check the tip was found correctly and both axes are horizontal"
        )
    um_per_px = 2.0 / (sx + sy)
    cos_t = float(np.linalg.norm(vel["D"])) * um_per_px
    if cos_t > 1.05:
        raise ValueError(f"D moves the tip further than X does (cos theta = {cos_t:.2f})")
    theta = math.degrees(math.acos(min(cos_t, 1.0)))
    jac = np.column_stack([vel["X"], vel["Y"]])
    if abs(np.linalg.det(jac)) < 1e-9:
        raise ValueError("X and Y move the tip along the same line on screen")
    return Calibration(
        um_per_px=float(um_per_px),
        vx=[float(v) for v in vel["X"]],
        vy=[float(v) for v in vel["Y"]],
        vd=[float(v) for v in vel["D"]],
        theta_deg=float(theta),
        image_size=[int(image_size[0]), int(image_size[1])],
        residual_px=float(max(residuals) if residuals else 0.0),
        notes=notes,
    )

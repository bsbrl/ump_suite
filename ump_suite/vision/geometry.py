"""
Turn segmentation instances into a scene: the numbers the skills act on.

Image coordinates are pixels with x to the right and y down. Angles are
reported the way they look on screen: 0 degrees points right (3 o'clock),
positive angles turn counterclockwise, so 90 is 12 o'clock and -90 is 6 o'clock.

A scene (a JSON-serialisable dict) holds

    embryos   one entry per chorion, sorted top to bottom (then left to right):
              chorion circle, yolk circle, cell centroid, the cell's clock
              angle around the yolk, whether the cell sits over the yolk's
              middle (on top or underneath), and whether the chorion is fully
              in view
    needle    tip, unit direction (base -> tip) and whether one was found
    target    index of the embryo the needle points at (the first chorion its
              line meets ahead of the tip), or None
    ray       for that embryo: distances along the needle line from the tip to
              the chorion, the cell's near edge and the cell's far edge, and
              whether the line meets the cell at all

The cell angle relative to the needle (`cell_angle_deg`) is measured from the
direction the needle comes from: 0 means the cell faces the needle, and a
positive value means the embryo must turn clockwise on screen to bring it there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# A cell centroid closer to the yolk centre than this fraction of the yolk
# radius is over the yolk's middle: on top of or underneath it, not at its edge.
CELL_ON_FACE_RATIO = 0.5
# Chorions smaller than this (pixels of area) are noise.
MIN_CHORION_AREA_PX = 2000
# A chorion whose fitted circle comes within this many pixels of the frame edge
# is not fully in view.
COMPLETE_MARGIN_PX = 2
# Instances below this score are ignored.
MIN_SCORE = 0.25
SCENE_VERSION = 1


@dataclass
class Instance:
    """One segmented object: class name, full-resolution boolean mask, score."""

    cls: str
    mask: np.ndarray
    score: float = 1.0
    extra: dict = field(default_factory=dict)

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def polygon_mask(polygon, shape) -> np.ndarray:
    """Rasterise an (N, 2) pixel polygon into a boolean mask of `shape` (H, W)."""
    import cv2

    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.round(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def fit_circle(points) -> tuple[float, float, float]:
    """Least-squares circle (cx, cy, r) through (N, 2) points (Kasa fit)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        raise ValueError("a circle needs at least three points")
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    return float(cx), float(cy), float(r)


def mask_boundary(mask) -> np.ndarray:
    """(N, 2) x, y pixels on the outer boundary of the largest blob in `mask`."""
    import cv2

    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2))
    c = max(contours, key=cv2.contourArea)
    return c.reshape(-1, 2).astype(np.float64)


def mask_circle(mask, *, inside_frame_only=True) -> tuple[float, float, float]:
    """
    Circle for a roughly round mask, fitted to its boundary.

    Boundary pixels lying on the frame edge are the frame, not the object, so
    they are left out; that keeps a chorion cut off by the frame edge round.
    """
    pts = mask_boundary(mask)
    if inside_frame_only and len(pts):
        h, w = mask.shape[:2]
        keep = (pts[:, 0] > 0) & (pts[:, 0] < w - 1) & (pts[:, 1] > 0) & (pts[:, 1] < h - 1)
        if keep.sum() >= 8:
            pts = pts[keep]
    if len(pts) < 3:
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise ValueError("empty mask")
        return float(xs.mean()), float(ys.mean()), float(math.sqrt(len(xs) / math.pi))
    return fit_circle(pts)


def mask_centroid(mask) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty mask")
    return float(xs.mean()), float(ys.mean())


def screen_angle_deg(dx, dy) -> float:
    """Angle of an image vector as seen on screen (0 = right, 90 = up)."""
    return math.degrees(math.atan2(-dy, dx))


def wrap_deg(a) -> float:
    """Wrap an angle into (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def needle_line(mask) -> dict | None:
    """
    Tip, base and unit direction of a needle mask (principal axis of its pixels).

    The base is the end nearer the frame edge (the needle enters from outside the
    view); the tip is the other end.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    axis = vt[0]
    proj = (pts - mean) @ axis
    a, b = mean + proj.min() * axis, mean + proj.max() * axis
    h, w = mask.shape[:2]

    def edge_distance(p):
        return min(p[0], p[1], w - 1 - p[0], h - 1 - p[1])

    base, tip = (a, b) if edge_distance(a) <= edge_distance(b) else (b, a)
    direction = tip - base
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return None
    return {"tip": tip, "base": base, "direction": direction / length, "length_px": length}


def ray_circle(origin, direction, circle) -> tuple[float, float] | None:
    """Distances t_in <= t_out along a unit ray to a circle, or None if it misses."""
    cx, cy, r = circle
    ox, oy = origin[0] - cx, origin[1] - cy
    dx, dy = direction
    b = ox * dx + oy * dy
    c = ox * ox + oy * oy - r * r
    disc = b * b - c
    if disc < 0:
        return None
    s = math.sqrt(disc)
    return -b - s, -b + s


def march(mask, origin, direction, t_start, t_stop, step=1.0) -> tuple[float, float] | None:
    """
    First run of `mask` pixels along the ray between t_start and t_stop.

    Returns (t_near, t_far) of the first contiguous run, or None if the ray never
    enters the mask in that interval.
    """
    h, w = mask.shape[:2]
    t = max(0.0, t_start)
    near = far = None
    while t <= t_stop:
        x = int(round(origin[0] + t * direction[0]))
        y = int(round(origin[1] + t * direction[1]))
        if 0 <= x < w and 0 <= y < h and mask[y, x]:
            if near is None:
                near = t
            far = t
        elif near is not None:
            break
        t += step
    return None if near is None else (near, far)


def _group(instances, chorions):
    """Assign each yolk/cell instance to the chorion containing its centroid."""
    members = [{"yolk": [], "cell": []} for _ in chorions]
    for inst in instances:
        if inst.cls not in ("yolk", "cell"):
            continue
        try:
            x, y = mask_centroid(inst.mask)
        except ValueError:
            continue
        best, best_d = None, None
        for i, ch in enumerate(chorions):
            cx, cy, r = ch["circle"]
            d = math.hypot(x - cx, y - cy)
            if d <= r * 1.05 and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is not None:
            members[best][inst.cls].append(inst)
    return members


def build_scene(instances, image_shape, *, um_per_px=None, stamp=None) -> dict:
    """Scene dict (see the module docstring) from segmentation instances."""
    h, w = image_shape[:2]
    instances = [i for i in instances if i.score >= MIN_SCORE and i.mask.any()]

    chorions = []
    for inst in instances:
        if inst.cls != "chorion" or inst.area < MIN_CHORION_AREA_PX:
            continue
        cx, cy, r = mask_circle(inst.mask)
        complete = (
            cx - r >= COMPLETE_MARGIN_PX
            and cy - r >= COMPLETE_MARGIN_PX
            and cx + r <= w - 1 - COMPLETE_MARGIN_PX
            and cy + r <= h - 1 - COMPLETE_MARGIN_PX
        )
        chorions.append({"inst": inst, "circle": (cx, cy, r), "complete": complete})
    chorions.sort(key=lambda c: (round(c["circle"][1]), c["circle"][0]))
    members = _group(instances, chorions)

    needle = None
    needles = [i for i in instances if i.cls == "needle"]
    if needles:
        nmask = np.zeros((h, w), dtype=bool)
        for n in needles:
            nmask |= n.mask
        needle = needle_line(nmask)

    embryos, cell_masks = [], []
    for idx, (ch, mem) in enumerate(zip(chorions, members)):
        cx, cy, r = ch["circle"]
        entry = {
            "id": idx,
            "chorion": {"cx": cx, "cy": cy, "r": r},
            "complete": bool(ch["complete"]),
            "yolk": None,
            "cell": None,
            "cell_angle_image_deg": None,
            "cell_angle_deg": None,
            "cell_on_face": None,
            "score": float(ch["inst"].score),
        }
        yolk_c = None
        if mem["yolk"]:
            yolk = max(mem["yolk"], key=lambda i: i.area)
            yx, yy, yr = mask_circle(yolk.mask)
            yolk_c = (yx, yy, yr)
            entry["yolk"] = {"cx": yx, "cy": yy, "r": yr}
        cell_mask = None
        if mem["cell"]:
            cell_mask = np.zeros((h, w), dtype=bool)
            for c in mem["cell"]:
                cell_mask |= c.mask
            ccx, ccy = mask_centroid(cell_mask)
            entry["cell"] = {"cx": ccx, "cy": ccy, "area": int(cell_mask.sum())}
            ox, oy, orad = yolk_c if yolk_c else (cx, cy, r * 0.5)
            ang = screen_angle_deg(ccx - ox, ccy - oy)
            entry["cell_angle_image_deg"] = ang
            entry["cell_on_face"] = bool(
                math.hypot(ccx - ox, ccy - oy) < CELL_ON_FACE_RATIO * orad
            )
            if needle is not None:
                toward = screen_angle_deg(-needle["direction"][0], -needle["direction"][1])
                entry["cell_angle_deg"] = wrap_deg(ang - toward)
        embryos.append(entry)
        cell_masks.append(cell_mask)

    scene = {
        "version": SCENE_VERSION,
        "stamp": stamp,
        "image_size": [int(w), int(h)],
        "embryos": embryos,
        "needle": None,
        "target": None,
        "ray": None,
        "um_per_px": um_per_px,
    }
    if needle is None:
        return _round(scene)

    tip, direction = needle["tip"], needle["direction"]
    scene["needle"] = {
        "tip": [float(tip[0]), float(tip[1])],
        "base": [float(needle["base"][0]), float(needle["base"][1])],
        "direction": [float(direction[0]), float(direction[1])],
        "length_px": needle["length_px"],
    }
    best, best_t = None, None
    for i, e in enumerate(embryos):
        hit = ray_circle(
            tip, direction, (e["chorion"]["cx"], e["chorion"]["cy"], e["chorion"]["r"])
        )
        if hit is None or hit[1] < 0:
            continue
        t_in = max(hit[0], 0.0)
        if best_t is None or t_in < best_t:
            best, best_t = i, t_in
    if best is None:
        return _round(scene)

    e = embryos[best]
    circle = (e["chorion"]["cx"], e["chorion"]["cy"], e["chorion"]["r"])
    t_in, t_out = ray_circle(tip, direction, circle)
    ray = {
        "chorion_px": max(t_in, 0.0),
        "chorion_exit_px": t_out,
        "tip_inside_chorion": t_in < 0,
        "hits_cell": False,
        "cell_near_px": None,
        "cell_far_px": None,
        "offset_px": abs(
            (circle[0] - tip[0]) * direction[1] - (circle[1] - tip[1]) * direction[0]
        ),
    }
    cell_mask = cell_masks[best]
    if cell_mask is not None:
        run = march(cell_mask, tip, direction, 0.0, t_out)
        if run is not None:
            ray.update(hits_cell=True, cell_near_px=run[0], cell_far_px=run[1])
    if um_per_px:
        for key in ("chorion_px", "cell_near_px", "cell_far_px", "offset_px"):
            value = ray[key]
            ray[key.replace("_px", "_um")] = None if value is None else value * um_per_px
    scene["target"] = best
    scene["ray"] = ray
    return _round(scene)


def _round(obj, nd=2):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, np.generic):
        return _round(obj.item(), nd)
    return obj


def injection_step_um(cell_near_um, *, depth_um, margin_um, theta_deg, min_um, max_um):
    """
    Step along the injection axis that puts the tip `depth_um` past the cell's edge.

        step = (cell_near_um + depth_um) / cos(theta) + margin_um

    `cell_near_um` is the horizontal distance the camera sees from the tip to the
    cell's near edge; dividing by cos(theta) turns it into travel along the
    tilted D axis. `margin_um` is the travel lost denting the chorion and the
    membrane before they give. The result is rounded to whole micrometres and
    clamped to [min_um, max_um]. Returns (step, unclamped).
    """
    if cell_near_um is None or not math.isfinite(cell_near_um):
        raise ValueError("no distance to the cell")
    c = math.cos(math.radians(theta_deg))
    if c <= 0.05:
        raise ValueError(f"D-axis tilt {theta_deg} deg is too steep")
    raw = (float(cell_near_um) + float(depth_um)) / c + float(margin_um)
    step = int(round(min(max(raw, min_um), max_um)))
    return step, raw


def embryo_features(scene, n_max=8) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed-size per-embryo features for a policy, plus a validity mask.

    Per embryo (in scene order, top to bottom): centre x and y and radius as
    fractions of the image width/height, cos and sin of the cell's screen angle,
    whether the cell is over the yolk's middle, whether the chorion is fully in
    view, and whether a cell was found.
    """
    feats = np.zeros((n_max, EMBRYO_FEATURE_DIM), dtype=np.float32)
    valid = np.zeros(n_max, dtype=bool)
    if not scene:
        return feats, valid
    w, h = scene["image_size"]
    for i, e in enumerate(scene["embryos"][:n_max]):
        ch = e["chorion"]
        ang = e.get("cell_angle_image_deg")
        has_cell = ang is not None
        rad = math.radians(ang) if has_cell else 0.0
        feats[i] = [
            ch["cx"] / w,
            ch["cy"] / h,
            ch["r"] / w,
            math.cos(rad) if has_cell else 0.0,
            math.sin(rad) if has_cell else 0.0,
            1.0 if e.get("cell_on_face") else 0.0,
            1.0 if e.get("complete") else 0.0,
            1.0 if has_cell else 0.0,
        ]
        valid[i] = True
    return feats, valid


EMBRYO_FEATURE_DIM = 8
EMBRYO_FEATURE_NAMES = (
    "cx",
    "cy",
    "r",
    "cell_cos",
    "cell_sin",
    "cell_on_face",
    "complete",
    "has_cell",
)


def nearest_embryo(scene, x, y, max_ratio=0.6):
    """Index of the embryo whose chorion centre is nearest (x, y), within max_ratio * r."""
    best, best_d = None, None
    for i, e in enumerate(scene.get("embryos", [])):
        ch = e["chorion"]
        d = math.hypot(ch["cx"] - x, ch["cy"] - y)
        if d <= max_ratio * ch["r"] and (best_d is None or d < best_d):
            best, best_d = i, d
    return best

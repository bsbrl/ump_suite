"""
Needle tip helpers that work on the image itself.

The tip is only a few micrometres wide and much fainter than the shank, so a
segmentation mask tends to stop short of it. `refine_tip` walks on along the
needle's line from the mask's tip while the line is still darker than the
background beside it.

`tip_from_difference` finds the needle with no model at all: it compares a frame
against a background frame taken with the needle out of view. The camera
calibration uses it, because it has to work before any model is trained.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import needle_line


def _sample(gray, x, y):
    h, w = gray.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if 0 <= xi < w and 0 <= yi < h:
        return float(gray[yi, xi])
    return None


def refine_tip(gray, tip, direction, *, max_extend_px=60, side_px=6, contrast=6.0, gap_px=3):
    """
    Extend `tip` along `direction` while the line stays darker than its sides.

    At each step the pixel on the line is compared with the mean of the two
    pixels `side_px` to either side; the line counts as needle while it is at
    least `contrast` grey levels darker. Up to `gap_px` lighter steps in a row
    are bridged (glints, noise). Returns the refined tip as (x, y).
    """
    gray = np.asarray(gray, dtype=np.float32)
    dx, dy = float(direction[0]), float(direction[1])
    nx, ny = -dy, dx
    best = (float(tip[0]), float(tip[1]))
    misses = 0
    for step in range(1, int(max_extend_px) + 1):
        x, y = tip[0] + dx * step, tip[1] + dy * step
        c = _sample(gray, x, y)
        a = _sample(gray, x + nx * side_px, y + ny * side_px)
        b = _sample(gray, x - nx * side_px, y - ny * side_px)
        if c is None or a is None or b is None:
            break
        if (a + b) / 2.0 - c >= contrast:
            best, misses = (x, y), 0
        else:
            misses += 1
            if misses > gap_px:
                break
    return best


def needle_mask_from_difference(frame_gray, background_gray, *, threshold=14, min_area=150):
    """
    Boolean mask of what changed between a background frame and this one.

    Only the largest changed blob that touches the frame edge is kept: the needle
    enters from outside the view, and anything else that changed (a drifting
    particle) is not the needle.
    """
    import cv2

    diff = cv2.absdiff(np.asarray(frame_gray, np.uint8), np.asarray(background_gray, np.uint8))
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    mask = (diff >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    best, best_area = None, 0
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        if touches and area >= min_area and area > best_area:
            best, best_area = k, area
    if best is None:
        return None
    return labels == best


def changed_area(a_gray, b_gray, *, threshold=14):
    """Pixels that differ clearly between two grey frames (after a light blur)."""
    import cv2

    diff = cv2.absdiff(np.asarray(a_gray, np.uint8), np.asarray(b_gray, np.uint8))
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    mask = cv2.morphologyEx((diff >= threshold).astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    return int(mask.sum())


def tip_from_difference(frame_gray, background_gray, **kwargs):
    """
    (tip_xy, direction) of the needle found by differencing against a background.

    The tip is refined along the needle line in the frame itself. Returns None
    when no needle-shaped change touches the frame edge.
    """
    mask = needle_mask_from_difference(frame_gray, background_gray, **kwargs)
    if mask is None:
        return None
    line = needle_line(mask)
    if line is None or line["length_px"] < 20:
        return None
    tip = refine_tip(frame_gray, line["tip"], line["direction"])
    return (float(tip[0]), float(tip[1])), (
        float(line["direction"][0]),
        float(line["direction"][1]),
    )


def extend_needle(instances, gray, **kwargs):
    """
    Lengthen the needle instances' mask to the tip found in the image.

    The union of the needle masks gets a 3 px line from its own tip to the
    refined tip; the other instances are returned unchanged.
    """
    import cv2

    needles = [i for i in instances if i.cls == "needle"]
    if not needles:
        return instances
    mask = np.zeros(gray.shape[:2], dtype=bool)
    for n in needles:
        mask |= n.mask
    line = needle_line(mask)
    if line is None:
        return instances
    # Compare the line with pixels just beyond the needle's own width, whatever the
    # camera scale: the width is the mask's area over its length.
    width = mask.sum() / max(line["length_px"], 1.0)
    side = int(min(max(round(width), 3), 15))
    opts = {"max_extend_px": 400, "contrast": 5.0, "gap_px": 8, "side_px": side, **kwargs}
    tip = refine_tip(gray, line["tip"], line["direction"], **opts)
    m = mask.astype(np.uint8)
    cv2.line(m, tuple(int(round(v)) for v in line["tip"]), tuple(int(round(v)) for v in tip), 1, 3)
    merged = type(needles[0])("needle", m.astype(bool), max(n.score for n in needles))
    return [i for i in instances if i.cls != "needle"] + [merged]


def tip_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

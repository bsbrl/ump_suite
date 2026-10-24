"""
First-pass labels from classical image processing, for a person to correct.

These are pre-annotations, not ground truth. They exist so that labelling starts
from outlines instead of empty frames, and so the training pipeline can be run
end to end before the first hand-checked labels exist. A model trained only on
them inherits their mistakes; check and correct them (for example in µSAM /
napari, or any tool that edits YOLO polygons) before trusting a model.

The cues, measured on the rig's brightfield frames:

    background          grey ~235, almost no colour (saturation ~2)
    inside the chorion  light brown, saturation ~50-60
    yolk                very dark (grey ~35), roughly round
    cell                greenish brown, saturation ~110+, next to the yolk
    needle              darker than the background, thin, enters from the edge

    python -m ump_suite.vision.tools.bootstrap_labels --frames-dir ~/embryo_labels/images \
        --out-dir ~/embryo_labels/labels_bootstrap --preview-dir ~/embryo_labels/preview \
        --sam ~/.local/share/ump_suite/models/sam2.1_b.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from .. import CLASS_NAMES
from ..geometry import Instance, fit_circle, mask_boundary, needle_line
from ..needle import refine_tip

CHORION_SAT = 24
CELL_SAT = 85
YOLK_GREY = 75
NEEDLE_CONTRAST = 14


def _largest(mask, min_area=1):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_area:
        return None
    return labels == k


def _fill(mask):
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, contours, -1, 1, thickness=-1)
    return out.astype(bool)


def _hull(mask):
    pts = cv2.findNonZero(mask.astype(np.uint8))
    out = np.zeros(mask.shape, np.uint8)
    if pts is not None:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out.astype(bool)


def ransac_circle(pts, *, iters=300, tol_ratio=0.02, seed=0):
    """
    Circle through the largest round part of a boundary.

    A chorion's blob often has the needle or the trough edge attached; those
    boundary points are outliers. Returns (cx, cy, r, inlier_fraction) or None.
    """
    rng = np.random.default_rng(seed)
    n = len(pts)
    if n < 20:
        return None
    best, best_count = None, 0
    for _ in range(iters):
        a, b, c = pts[rng.choice(n, 3, replace=False)]
        try:
            cx, cy, r = fit_circle(np.array([a, b, c]))
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not math.isfinite(r) or r < 20:
            continue
        inl = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < max(2.0, tol_ratio * r)
        count = int(inl.sum())
        if count > best_count:
            best, best_count = inl, count
    if best is None:
        return None
    cx, cy, r = fit_circle(pts[best])
    return cx, cy, r, best_count / n


def _disc(shape, cx, cy, r):
    out = np.zeros(shape[:2], np.uint8)
    cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)), 1, -1)
    return out.astype(bool)


def classical_instances(frame_bgr) -> list[Instance]:
    """Pre-annotation instances for one frame (see the module docstring)."""
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sat = cv2.medianBlur(hsv[:, :, 1], 9)
    out: list[Instance] = []

    inside = (sat > CHORION_SAT).astype(np.uint8)
    inside = cv2.morphologyEx(inside, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    inside = _fill(inside)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inside.astype(np.uint8), connectivity=8)
    chorions = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 40000:
            continue
        blob = labels == k
        pts = mask_boundary(blob)
        keep = (pts[:, 0] > 1) & (pts[:, 0] < w - 2) & (pts[:, 1] > 1) & (pts[:, 1] < h - 2)
        if keep.sum() < 50:
            continue
        found = ransac_circle(pts[keep])
        if found is None:
            continue
        cx, cy, r, frac = found
        if frac < 0.45 or not 0.08 * w <= r <= 0.45 * w:
            continue  # not round enough: debris, or two embryos fused into one blob
        chorions.append((cx, cy, r))
        out.append(Instance("chorion", _disc((h, w), cx, cy, r)))

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    for cx, cy, r in chorions:
        inner = _disc((h, w), cx, cy, r * 0.97)
        yolk = _largest((blur < YOLK_GREY) & inner, min_area=int(0.03 * math.pi * r * r))
        if yolk is None:
            continue
        yolk = _hull(
            cv2.morphologyEx(
                yolk.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)
            ).astype(bool)
        )
        out.append(Instance("yolk", yolk))
        hue = hsv[:, :, 0]
        cell = (sat > CELL_SAT) & (hue >= 15) & (hue <= 60) & (hsv[:, :, 2] > 60) & inner & ~yolk
        cell = cv2.morphologyEx(
            cell.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)
        )
        cell = cv2.morphologyEx(cell, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(bool)
        near_yolk = cv2.dilate(yolk.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(cell.astype(np.uint8), connectivity=8)
        keep = np.zeros((h, w), bool)
        for k in range(1, n2):
            comp = lab2 == k
            if st2[k, cv2.CC_STAT_AREA] >= 0.004 * math.pi * r * r and (comp & near_yolk).any():
                keep |= comp
        if keep.any():
            out.append(Instance("cell", _fill(keep) & ~yolk))

    background = cv2.medianBlur(gray, 51).astype(np.int16)
    dark = ((background - gray.astype(np.int16)) > NEEDLE_CONTRAST).astype(np.uint8)
    for cx, cy, r in chorions:  # the chorion rim is dark too; keep it out
        cv2.circle(dark, (int(cx), int(cy)), int(r + 6), 0, -1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n3, lab3, st3, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    best, best_len = None, 0.0
    for k in range(1, n3):
        x, y, bw, bh, area = st3[k]
        touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        if not touches or area < 300:
            continue
        line = needle_line(lab3 == k)
        if line is None:
            continue
        thickness = area / max(line["length_px"], 1.0)
        if line["length_px"] > 120 and thickness < 25 and line["length_px"] > best_len:
            best, best_len = (k, line), line["length_px"]
    if best is not None:
        k, line = best
        mask = (lab3 == k).astype(np.uint8)
        tip = refine_tip(gray, line["tip"], line["direction"])
        cv2.line(
            mask,
            tuple(int(round(v)) for v in line["tip"]),
            tuple(int(round(v)) for v in tip),
            1,
            thickness=3,
        )
        out.append(Instance("needle", mask.astype(bool)))
    return out


def _sam_masks(sam, frame, points, shape):
    """One SAM mask per point prompt, resized to the frame."""
    res = sam(
        frame,
        points=[[float(x), float(y)] for x, y in points],
        labels=[1] * len(points),
        verbose=False,
    )[0]
    out = []
    if res.masks is None:
        return out
    for m in res.masks.data.cpu().numpy().astype(bool):
        if m.shape != shape:
            m = cv2.resize(
                m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        out.append(m)
    return out


def _horizontal_needle(gray, chorions):
    """Dark, horizontal structure touching the right edge (the needle's shank)."""
    h, w = gray.shape
    bg = cv2.medianBlur(gray, 51).astype(np.int16)
    dark = ((bg - gray.astype(np.int16)) > 12).astype(np.uint8)
    horiz = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1))
    )
    for cx, cy, r in chorions:
        cv2.circle(horiz, (int(cx), int(cy)), int(r + 8), 0, -1)
    n, lab, st, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    best = None
    for k in range(1, n):
        x, y, bw, bh, _a = st[k]
        if x + bw >= w - 2 and bw > 120 and bh < 60 and (best is None or bw > st[best][2]):
            best = k
    return None if best is None else lab == best


def chorion_circle(gray, centre, body_r, *, iters=600, seed=0):
    """
    Find the chorion: the strongest circular edge around the embryo body.

    Canny edges in an annulus 1.2-2.6 body radii from the body's centre feed a
    RANSAC circle whose centre must stay near the body's. Returns (cx, cy, r)
    or None when no circle has edge support on at least a quarter of its rim.
    """
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 25, 75)
    ys, xs = np.nonzero(edges)
    d = np.hypot(xs - centre[0], ys - centre[1])
    sel = (d > 1.2 * body_r) & (d < 2.6 * body_r)
    pts = np.column_stack([xs[sel], ys[sel]]).astype(np.float64)
    if len(pts) < 50:
        return None
    rng = np.random.default_rng(seed)
    best, best_count = None, 0
    for _ in range(iters):
        try:
            cx, cy, r = fit_circle(pts[rng.choice(len(pts), 3, replace=False)])
        except (ValueError, np.linalg.LinAlgError):
            continue
        if (
            not (1.3 * body_r < r < 2.5 * body_r)
            or math.hypot(cx - centre[0], cy - centre[1]) > 0.6 * r
        ):
            continue
        inl = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < 3.0
        count = int(inl.sum())
        if count > best_count:
            best, best_count = inl, count
    if best is None:
        return None
    cx, cy, r = fit_circle(pts[best])
    # rim pixels inside the frame that have an edge nearby
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    rx, ry = cx + r * np.cos(t), cy + r * np.sin(t)
    inside = (rx >= 0) & (rx < w) & (ry >= 0) & (ry < h)
    near = cv2.dilate(edges, np.ones((7, 7), np.uint8))
    support = (
        near[ry[inside].astype(int), rx[inside].astype(int)].astype(bool).mean()
        if inside.any()
        else 0
    )
    if support < 0.25:
        return None
    return float(cx), float(cy), float(r)


def _saturation_chorion(frame_bgr, centre, body_r):
    """Chorion circle from the coloured blob inside it (RANSAC on its round edge)."""
    h, w = frame_bgr.shape[:2]
    sat = cv2.medianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 1], 9)
    inside = cv2.morphologyEx(
        (sat > CHORION_SAT).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)
    )
    inside = _fill(inside)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inside.astype(np.uint8), connectivity=8)
    best = None
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 40000:
            continue
        blob = labels == k
        if not blob[int(np.clip(centre[1], 0, h - 1)), int(np.clip(centre[0], 0, w - 1))]:
            continue
        pts = mask_boundary(blob)
        keep = (pts[:, 0] > 1) & (pts[:, 0] < w - 2) & (pts[:, 1] > 1) & (pts[:, 1] < h - 2)
        found = ransac_circle(pts[keep]) if keep.sum() > 50 else None
        if found is None:
            continue
        cx, cy, r, frac = found
        if (
            frac >= 0.45
            and 1.3 * body_r < r < 2.6 * body_r
            and math.hypot(cx - centre[0], cy - centre[1]) < 0.6 * r
        ):
            best = (cx, cy, r)
    return best


def sam_instances(frame_bgr, sam) -> list[Instance]:
    """
    Pre-annotation with SAM 2 point prompts, seeded by the dark yolk.

        body    SAM on the deepest point of the dark yolk ring: yolk + cell
        yolk    circle through the round part of the body's dark core
        cell    the body minus the yolk circle
        chorion SAM on points just outside the body (the perivitelline space),
                joined with the body, then a circle through its round edge
        needle  SAM on a point of the dark horizontal shank at the right edge

    Falls back to `classical_instances` when the yolk core is not found.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    core = _largest((blur < 55).astype(np.uint8), min_area=20000)
    if core is None:
        return classical_instances(frame_bgr)
    dist = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    sy, sx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    masks = _sam_masks(sam, frame_bgr, [(sx, sy)], (h, w))
    if not masks:
        return classical_instances(frame_bgr)
    body = _fill(masks[0])
    out: list[Instance] = []

    # The yolk is the round part of the body's outline; the cell is a bulge on it.
    # When the cell lines up with the yolk's edge the body's outline is round as
    # a whole, so the dark core's circle is tried too and the smaller one kept.
    body_r = math.sqrt(body.sum() / math.pi)
    cands = []
    for src, tol in (
        (body, 0.015),
        (_largest((body & (blur < 50)).astype(np.uint8), min_area=5000), 0.02),
    ):
        if src is None:
            continue
        fy = ransac_circle(mask_boundary(_fill(src)), tol_ratio=tol)
        if fy is not None and fy[3] > 0.4 and fy[2] >= 0.6 * body_r:
            cands.append(fy)
    yolk_disc = None
    if cands:
        fy = min(cands, key=lambda c: c[2])
        yolk_disc = _disc((h, w), fy[0], fy[1], fy[2]) & body
    if yolk_disc is None or yolk_disc.sum() < 5000:
        return classical_instances(frame_bgr)
    out.append(Instance("yolk", yolk_disc))
    rest = body & ~cv2.dilate(yolk_disc.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    rest = cv2.morphologyEx(
        rest.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)
    ).astype(bool)
    if rest.sum() > 0.02 * yolk_disc.sum():
        out.append(Instance("cell", rest))

    ys, xs = np.nonzero(body)
    bx, by = xs.mean(), ys.mean()
    br = math.sqrt(body.sum() / math.pi)
    chorions = []
    fc = None
    sat_circle = _saturation_chorion(frame_bgr, (bx, by), br)
    edge_circle = chorion_circle(gray, (bx, by), br)
    for c in (sat_circle, edge_circle):
        if c is not None and (fc is None or c[2] > fc[2]):
            fc = c
    if fc is not None:
        chorions.append(fc)
        out.insert(0, Instance("chorion", _disc((h, w), *fc)))

    shank = _horizontal_needle(gray, chorions)
    if shank is not None:
        ys, xs = np.nonzero(shank)
        seed = (float(np.median(xs[xs > np.percentile(xs, 70)])), float(np.median(ys)))
        found = _sam_masks(sam, frame_bgr, [seed], (h, w))
        needle = shank
        if found:
            cand = found[0]
            line = needle_line(cand)
            ys_c = np.nonzero(cand)[0]
            if (
                line is not None
                and cand.sum() / max(line["length_px"], 1) < 30
                and abs(line["direction"][1]) < 0.5
                and (cand & shank).sum() > 0.5 * shank.sum()
                and len(ys_c)
                and ys_c.max() - ys_c.min() < 80
            ):
                needle = cand | shank
        line = needle_line(needle)
        if line is not None:
            tip = refine_tip(
                gray, line["tip"], line["direction"], max_extend_px=400, contrast=5, gap_px=8
            )
            m = needle.astype(np.uint8)
            cv2.line(
                m,
                tuple(int(round(v)) for v in line["tip"]),
                tuple(int(round(v)) for v in tip),
                1,
                3,
            )
            needle = m.astype(bool)
        out.append(Instance("needle", needle))
    return out


def load_sam(weights):
    import os

    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    from ultralytics import SAM

    return SAM(str(weights))


def mask_polygons(mask, epsilon=1.5):
    """Outer polygons (N, 2) of every blob in a mask, simplified by `epsilon` px."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 30:
            continue
        approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
        if len(approx) >= 3:
            polys.append(approx)
    return polys


def yolo_lines(instances, shape) -> list[str]:
    """YOLO-seg label lines: class index, then the polygon normalised to 0..1."""
    h, w = shape[:2]
    lines = []
    for inst in instances:
        cls = CLASS_NAMES.index(inst.cls)
        for poly in mask_polygons(inst.mask):
            coords = " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly)
            lines.append(f"{cls} {coords}")
    return lines


COLOURS = {
    "chorion": (200, 120, 0),
    "yolk": (0, 0, 220),
    "cell": (0, 200, 0),
    "needle": (200, 0, 200),
}


def preview(frame_bgr, instances):
    img = frame_bgr.copy()
    for inst in instances:
        contours, _ = cv2.findContours(
            inst.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, contours, -1, COLOURS[inst.cls], 3)
    return img


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True, help="YOLO .txt labels are written here")
    p.add_argument("--preview-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true", help="replace existing label files")
    p.add_argument(
        "--sam",
        type=Path,
        default=None,
        help="SAM 2 weights (e.g. sam2.1_b.pt): outline with SAM point prompts "
        "instead of thresholds alone",
    )
    args = p.parse_args(argv)
    sam = load_sam(args.sam) if args.sam else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(
        q for q in args.frames_dir.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    counts = {c: 0 for c in CLASS_NAMES}
    written = skipped = 0
    for q in frames:
        target = args.out_dir / (q.stem + ".txt")
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        frame = cv2.imread(str(q))
        if frame is None:
            continue
        inst = sam_instances(frame, sam) if sam is not None else classical_instances(frame)
        for i in inst:
            counts[i.cls] += 1
        target.write_text("\n".join(yolo_lines(inst, frame.shape)) + "\n")
        written += 1
        if args.preview_dir:
            cv2.imwrite(str(args.preview_dir / (q.stem + ".jpg")), preview(frame, inst))
    print(
        f"[bootstrap] {written} frame(s) labelled, {skipped} kept as they were; "
        f"instances: {counts}"
    )
    print(
        "[bootstrap] these are pre-annotations: check and correct them "
        "before training a model you trust"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

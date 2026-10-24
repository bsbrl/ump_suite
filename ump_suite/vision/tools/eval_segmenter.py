"""
Score a segmentation model on labelled frames in the units the skills use.

    python -m ump_suite.vision.tools.eval_segmenter --data ~/embryo_labels \
        --labels labels --groups-from ~/embryo_seg_runs/<run>/model_note.json

For each labelled frame the model's scene is compared with the scene built from
the labels: mask overlap (IoU) per class, chorion centre error, the cell's
clock-angle error (degrees), the needle tip error, and the error in the
needle-to-cell distance the injection step is computed from. By default every
labelled frame is used; --groups-from limits it to the validation trials a
training run held out, which is the honest number.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .. import CLASS_NAMES
from ..geometry import Instance, build_scene, polygon_mask, wrap_deg
from ..segmenter import DEFAULT_MODEL_PATH, Segmenter


def read_label_instances(path: Path, shape) -> list[Instance]:
    h, w = shape[:2]
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = CLASS_NAMES[int(parts[0])]
        pts = np.array([float(v) for v in parts[1:]]).reshape(-1, 2) * [w, h]
        out.append(Instance(cls, polygon_mask(pts, shape)))
    return out


def class_masks(instances, shape):
    masks = {c: np.zeros(shape[:2], dtype=bool) for c in CLASS_NAMES}
    for inst in instances:
        masks[inst.cls] |= inst.mask
    return masks


def compare(truth_inst, pred_inst, shape):
    """Per-frame errors between a labelled and a predicted frame."""
    row = {}
    t_masks, p_masks = class_masks(truth_inst, shape), class_masks(pred_inst, shape)
    for c in CLASS_NAMES:
        union = (t_masks[c] | p_masks[c]).sum()
        row[f"iou_{c}"] = float((t_masks[c] & p_masks[c]).sum() / union) if union else None
    t_scene, p_scene = build_scene(truth_inst, shape), build_scene(pred_inst, shape)
    if t_scene["embryos"] and p_scene["embryos"]:
        te, pe = t_scene["embryos"][0], p_scene["embryos"][0]
        row["chorion_centre_px"] = math.hypot(
            te["chorion"]["cx"] - pe["chorion"]["cx"], te["chorion"]["cy"] - pe["chorion"]["cy"]
        )
        ta, pa = te.get("cell_angle_image_deg"), pe.get("cell_angle_image_deg")
        row["cell_angle_deg"] = (
            abs(wrap_deg(ta - pa)) if ta is not None and pa is not None else None
        )
    if t_scene["needle"] and p_scene["needle"]:
        tt, pt = t_scene["needle"]["tip"], p_scene["needle"]["tip"]
        row["tip_px"] = math.hypot(tt[0] - pt[0], tt[1] - pt[1])
    tr, pr = t_scene.get("ray") or {}, p_scene.get("ray") or {}
    if tr.get("cell_near_px") is not None and pr.get("cell_near_px") is not None:
        row["cell_distance_px"] = abs(tr["cell_near_px"] - pr["cell_near_px"])
    return row


def summarise(rows):
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = {
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p90": float(np.percentile(vals, 90)),
            }
    return out


def main(argv=None):
    import cv2

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--labels", default="labels")
    p.add_argument("--weights", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument(
        "--groups-from",
        type=Path,
        default=None,
        help="model_note.json of a training run: evaluate only its val_groups",
    )
    p.add_argument("--um-per-px", type=float, default=None, help="report distances in um too")
    p.add_argument("--device", default="0")
    args = p.parse_args(argv)
    data = args.data.expanduser()
    groups = None
    if args.groups_from:
        groups = set(json.loads(Path(args.groups_from).expanduser().read_text())["val_groups"])
    with (data / "manifest.csv").open() as f:
        manifest = {r["name"]: r["group"] for r in csv.DictReader(f)}
    seg = Segmenter(args.weights, device=args.device)
    rows = []
    for name, group in sorted(manifest.items()):
        lab = data / args.labels / f"{name}.txt"
        img_path = data / "images" / f"{name}.png"
        if not lab.exists() or not img_path.exists() or (groups and group not in groups):
            continue
        frame = cv2.imread(str(img_path))
        rows.append(compare(read_label_instances(lab, frame.shape), seg(frame), frame.shape))
    summary = summarise(rows)
    print(
        f"[eval] {len(rows)} frame(s)"
        + (f" from {len(groups)} held-out trial(s)" if groups else "")
    )
    for k, v in summary.items():
        extra = ""
        if args.um_per_px and k.endswith("_px"):
            extra = f"   (mean {v['mean'] * args.um_per_px:.1f} um)"
        print(
            f"  {k:22s} mean {v['mean']:8.3f}  median {v['median']:8.3f}  "
            f"p90 {v['p90']:8.3f}  n={v['n']}{extra}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

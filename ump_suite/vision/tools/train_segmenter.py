"""
Train the YOLO-seg embryo model, keeping whole trials out of training for validation.

    python -m ump_suite.vision.tools.train_segmenter --data ~/embryo_labels \
        --val-fraction 0.2 --epochs 80 --install

`--data` holds images/, labels/ (YOLO .txt, see labels_to_yolo.py) and
manifest.csv (from export_frames.py). Frames are split by their manifest group
(the trial), so validation scores are on embryos the model never trained on.
Only images that have a label file are used.

`--install` copies the best weights to the path the vision node loads
(~/.local/share/ump_suite/models/embryo_seg.pt) with a JSON note beside it
saying what it was trained on. `--test-only` marks a model trained on
unchecked labels (e.g. bootstrap_labels.py output) so nobody mistakes it for
a validated one.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

from .. import CLASS_NAMES
from ..segmenter import DEFAULT_MODEL_PATH

MODELS_DIR = DEFAULT_MODEL_PATH.parent


def split_groups(groups, val_fraction, seed=0, val_groups=None):
    groups = sorted(set(groups))
    if val_groups:
        val = set(val_groups) & set(groups)
    else:
        rng = random.Random(seed)
        shuffled = groups[:]
        rng.shuffle(shuffled)
        k = max(1, round(len(groups) * val_fraction)) if len(groups) > 1 else 0
        val = set(shuffled[:k])
    return [g for g in groups if g not in val], sorted(val)


def build_split(data: Path, work: Path, val_fraction, seed=0, val_groups=None, labels="labels"):
    """Symlinked train/val folders and data.yaml under `work`; returns (yaml, counts)."""
    with (data / "manifest.csv").open() as f:
        manifest = {r["name"]: r["group"] for r in csv.DictReader(f)}
    labelled = [
        n
        for n in manifest
        if (data / labels / f"{n}.txt").exists() and (data / "images" / f"{n}.png").exists()
    ]
    if not labelled:
        raise SystemExit("no labelled images found")
    train_g, val_g = split_groups([manifest[n] for n in labelled], val_fraction, seed, val_groups)
    if work.exists():
        shutil.rmtree(work)
    counts = {"train": 0, "val": 0}
    for n in labelled:
        part = "val" if manifest[n] in val_g else "train"
        for kind, src, ext in (("images", "images", ".png"), ("labels", labels, ".txt")):
            d = work / kind / part
            d.mkdir(parents=True, exist_ok=True)
            os.symlink((data / src / f"{n}{ext}").resolve(), d / f"{n}{ext}")
        counts[part] += 1
    yaml = work / "data.yaml"
    yaml.write_text(
        f"path: {work}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASS_NAMES))
    )
    return yaml, counts, train_g, val_g


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--labels", default="labels", help="labels folder inside --data")
    p.add_argument(
        "--base", default=str(MODELS_DIR / "yolo11s-seg.pt"), help="pretrained starting weights"
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument(
        "--val-group", action="append", default=None, help="manifest group kept for validation"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="0")
    p.add_argument("--project", type=Path, default=Path("~/embryo_seg_runs"))
    p.add_argument("--name", default=None)
    p.add_argument(
        "--install", action="store_true", help="copy best weights to the vision node's model path"
    )
    p.add_argument("--install-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument(
        "--test-only",
        action="store_true",
        help="labels were not checked by a person; mark the model as test-only",
    )
    args = p.parse_args(argv)

    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    from ultralytics import YOLO

    data = args.data.expanduser().resolve()
    project = args.project.expanduser().resolve()
    name = args.name or time.strftime("seg_%Y%m%d_%H%M%S")
    yaml, counts, train_g, val_g = build_split(
        data, project / f"{name}_split", args.val_fraction, args.seed, args.val_group, args.labels
    )
    print(
        f"[train] {counts['train']} train / {counts['val']} val frame(s); "
        f"{len(train_g)} train trial(s), validation trials: {', '.join(val_g)}"
    )
    model = YOLO(args.base)
    model.train(
        data=str(yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=name,
        seed=args.seed,
        plots=False,
        verbose=False,
        exist_ok=True,
        workers=4,
        # The needle always enters from the right: a top-bottom mirror and a
        # small tilt keep it there (and show cells at the mirrored clock
        # positions); a left-right flip or a large rotation would not.
        fliplr=0.0,
        flipud=0.5,
        degrees=10.0,
        mosaic=0.5,
    )
    best = project / name / "weights" / "best.pt"
    print(f"[train] best weights: {best}")
    note = {
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": str(data),
        "labels": args.labels,
        "frames": counts,
        "train_groups": train_g,
        "val_groups": val_g,
        "base": args.base,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "test_only": bool(args.test_only),
        "label_status": "unchecked (test-only)" if args.test_only else "as provided",
    }
    (project / name / "model_note.json").write_text(json.dumps(note, indent=2))
    if args.install:
        dest = args.install_path.expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        dest.with_suffix(".json").write_text(json.dumps(note, indent=2))
        print(
            f"[train] installed {dest}"
            + ("  (TEST-ONLY: trained on unchecked labels)" if args.test_only else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

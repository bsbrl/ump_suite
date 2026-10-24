"""
Convert polygons from an annotation tool into YOLO segmentation labels.

Draw one polygon per object with the class names chorion, yolk, cell and needle
(one chorion, one yolk and one cell polygon per embryo; the cell mass may be
several polygons). Most tools export YOLO segmentation directly (CVAT
"Ultralytics YOLO Segmentation", Label Studio "YOLO", Roboflow "YOLOv8/11
segmentation"); copy those .txt files into the labels folder and only run
`--check`. For other tools this converts

    --coco FILE      a COCO instances JSON (CVAT, Label Studio, Roboflow, µSAM)
    --labelme DIR    a folder of labelme JSON files, one per image

Class names are matched case-insensitively; anything else is skipped and
reported. The class order in the written files is chorion=0, yolk=1, cell=2,
needle=3 whatever order the tool used.

    python -m ump_suite.vision.tools.labels_to_yolo --coco export.json \
        --images ~/embryo_labels/images --out ~/embryo_labels/labels
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .. import CLASS_NAMES


def _norm_class(name):
    key = str(name).strip().lower()
    return CLASS_NAMES.index(key) if key in CLASS_NAMES else None


def _line(cls, points, w, h):
    coords = []
    for x, y in points:
        coords.append(f"{min(max(x / w, 0.0), 1.0):.5f} {min(max(y / h, 0.0), 1.0):.5f}")
    return f"{cls} " + " ".join(coords)


def from_coco(coco_path: Path, out_dir: Path):
    data = json.loads(Path(coco_path).read_text())
    cats = {c["id"]: _norm_class(c["name"]) for c in data.get("categories", [])}
    skipped = Counter(
        c["name"] for c in data.get("categories", []) if _norm_class(c["name"]) is None
    )
    images = {im["id"]: im for im in data["images"]}
    lines = {im_id: [] for im_id in images}
    for ann in data.get("annotations", []):
        cls = cats.get(ann["category_id"])
        seg = ann.get("segmentation")
        if cls is None or not isinstance(seg, list):
            continue  # RLE masks need pycocotools; export polygons instead
        im = images[ann["image_id"]]
        for poly in seg:
            pts = list(zip(poly[0::2], poly[1::2]))
            if len(pts) >= 3:
                lines[ann["image_id"]].append(_line(cls, pts, im["width"], im["height"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    for im_id, im in images.items():
        stem = Path(im["file_name"]).stem
        (out_dir / f"{stem}.txt").write_text(
            "\n".join(lines[im_id]) + ("\n" if lines[im_id] else "")
        )
    return len(images), skipped


def from_labelme(folder: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped, n = Counter(), 0
    for js in sorted(Path(folder).glob("*.json")):
        data = json.loads(js.read_text())
        w, h = data["imageWidth"], data["imageHeight"]
        lines = []
        for shape in data.get("shapes", []):
            cls = _norm_class(shape.get("label", ""))
            if cls is None:
                skipped[shape.get("label", "")] += 1
                continue
            if shape.get("shape_type", "polygon") != "polygon" or len(shape["points"]) < 3:
                continue
            lines.append(_line(cls, shape["points"], w, h))
        stem = Path(data.get("imagePath", js.name)).stem
        (out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    return n, skipped


def check(images_dir: Path, labels_dir: Path):
    """Count labels per class and list images without a label file."""
    counts, missing, empty = Counter(), [], []
    for im in sorted(
        q for q in images_dir.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg")
    ):
        lab = labels_dir / f"{im.stem}.txt"
        if not lab.exists():
            missing.append(im.name)
            continue
        rows = [r for r in lab.read_text().splitlines() if r.strip()]
        if not rows:
            empty.append(im.name)
        for r in rows:
            parts = r.split()
            cls = int(parts[0])
            if cls >= len(CLASS_NAMES) or (len(parts) - 1) % 2 or len(parts) < 7:
                raise ValueError(f"{lab.name}: bad line {r[:60]!r}")
            counts[CLASS_NAMES[cls]] += 1
    return counts, missing, empty


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group()
    src.add_argument("--coco", type=Path)
    src.add_argument("--labelme", type=Path)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="labels folder (YOLO .txt per image)")
    args = p.parse_args(argv)
    if args.coco:
        n, skipped = from_coco(args.coco, args.out)
        print(f"[labels] {n} image(s) converted from COCO")
    elif args.labelme:
        n, skipped = from_labelme(args.labelme, args.out)
        print(f"[labels] {n} image(s) converted from labelme")
    else:
        skipped = Counter()
    if skipped:
        print(f"[labels] skipped classes (not one of {', '.join(CLASS_NAMES)}): {dict(skipped)}")
    counts, missing, empty = check(args.images, args.out)
    print(f"[labels] objects per class: {dict(counts)}")
    print(f"[labels] {len(missing)} image(s) without a label file, {len(empty)} with an empty one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

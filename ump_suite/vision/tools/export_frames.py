"""
Copy frames from logged trials into one folder for labelling.

Every Nth frame of each trial is copied (plus every frame flagged as an
injection or an orient start, which are the moments that matter most), named
`<session>__t<trial>__f<frame>.png`, and listed in `manifest.csv` with its
source and its group. The group is the trial: frames of one trial show one
embryo, so the training split keeps whole trials together and the model is
always tested on embryos it never saw.

    python -m ump_suite.vision.tools.export_frames \
        --trials "~/Backup/New 3" --trials ~ --every 4 --out ~/embryo_labels
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

_TRIAL_RE = re.compile(r"trial_(\d+)\.csv$")


def _session_name(root: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", root.name).strip("_")
    return name or "home"


def export(roots, out_dir: Path, every=4, overwrite=False):
    import pandas as pd

    images = out_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"
    rows, seen = [], set()
    if manifest.exists():
        with manifest.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
                seen.add(r["name"])
    added = 0
    for root in roots:
        root = Path(root).expanduser().resolve()
        logs = root / "logs" if (root / "logs").is_dir() else root
        session = _session_name(root)
        for csv_path in sorted(
            logs.glob("trial_*.csv"), key=lambda q: int(_TRIAL_RE.search(q.name).group(1))
        ):
            trial = int(_TRIAL_RE.search(csv_path.name).group(1))
            df = pd.read_csv(csv_path)
            flagged = set()
            for col in ("Injection", "Orient"):
                if col in df.columns:
                    hits = df.index[pd.to_numeric(df[col], errors="coerce") == 1].tolist()
                    flagged.update(
                        max(i - 1, 0) for i in hits
                    )  # the frame the decision was made on
            for i, row in df.iterrows():
                if i % every and i not in flagged:
                    continue
                src = root / str(row["image_path"])
                if not src.exists():
                    continue
                name = f"{session}__t{trial}__f{i:06d}"
                if name in seen and not overwrite:
                    continue
                shutil.copy2(src, images / f"{name}.png")
                if name not in seen:
                    rows.append({"name": name, "source": str(src), "group": f"{session}:t{trial}"})
                    seen.add(name)
                added += 1
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "source", "group"])
        w.writeheader()
        w.writerows(rows)
    return added, len(rows), len({r["group"] for r in rows})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--trials",
        action="append",
        required=True,
        help="folder holding logs/trial_N.csv and saved_frames/ (repeatable)",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    added, total, groups = export(args.trials, args.out.expanduser(), args.every, args.overwrite)
    print(
        f"[export] {added} frame(s) copied; manifest lists {total} frame(s) from {groups} trial(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

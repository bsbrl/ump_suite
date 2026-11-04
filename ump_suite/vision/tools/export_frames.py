"""
Copy frames from logged trials into one folder for labelling.

Every Nth frame of each trial is copied (plus every frame flagged as an
injection or an orient start, which are the moments that matter most), named
`<session>__t<trial>__f<frame>.png`, and listed in `manifest.csv` with its
source and its group. The group is the trial: frames of one trial show the
same embryos, so the training split keeps whole trials together and the model
is always tested on embryos it never saw.

    python3 -m ump_suite.vision.tools.export_frames \
        --trials ~ --session inject1 --every 30 --out ~/embryo_labels

`<session>` is the folder's name unless --session gives one. Trial numbers
restart when a folder is emptied for a new dataset, so give each dataset
recorded in the same folder its own session: frames are kept apart by name, and
a name that already holds a different image stops the export.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import re
import shutil
from pathlib import Path

_TRIAL_RE = re.compile(r"trial_(\d+)\.csv$")


def _session_name(root: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", root.name).strip("_")
    return name or "home"


def export(roots, out_dir: Path, every=4, overwrite=False, session_name=None):
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
        session = session_name or _session_name(root)
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
                    copied = images / f"{name}.png"
                    if copied.exists() and not filecmp.cmp(copied, src, shallow=False):
                        raise ValueError(
                            f"{name} is already exported from a different recording, with the "
                            "same trial number; give this dataset its own --session"
                        )
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
    p.add_argument("--session", default=None,
                   help="name for these trials (default: the folder's); one --trials only")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if args.session and len(args.trials) > 1:
        p.error("--session names one folder's trials; run once per folder")
    added, total, groups = export(args.trials, args.out.expanduser(), args.every, args.overwrite,
                                  session_name=args.session)
    print(
        f"[export] {added} frame(s) copied; manifest lists {total} frame(s) from {groups} trial(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

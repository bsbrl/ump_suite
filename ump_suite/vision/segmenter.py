"""
YOLO-seg wrapper: one camera frame in, segmentation instances out.

The model must know the classes chorion, yolk, cell and needle (any order; they
are matched by name). ultralytics is imported only when a model is loaded, so
everything else in the package works without it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from . import CLASS_NAMES
from .geometry import Instance, polygon_mask

DEFAULT_MODEL_PATH = Path(os.path.expanduser("~/.local/share/ump_suite/models/embryo_seg.pt"))


def _quiet_ultralytics():
    # Never let ultralytics pip-install packages into the rig's Python at run
    # time, and keep it from phoning home.
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_VERBOSE", "false")


class Segmenter:
    """Run a YOLO-seg model and return `Instance`s at the frame's resolution."""

    def __init__(self, weights=DEFAULT_MODEL_PATH, *, conf=0.35, imgsz=1024, device=None):
        weights = Path(weights).expanduser()
        if not weights.exists():
            raise FileNotFoundError(f"no segmentation model at {weights}")
        _quiet_ultralytics()
        from ultralytics import YOLO

        self.weights = weights
        self.model = YOLO(str(weights))
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device
        names = {int(k): str(v) for k, v in self.model.names.items()}
        missing = [c for c in CLASS_NAMES if c not in names.values()]
        if missing:
            raise ValueError(
                f"model {weights.name} lacks classes {missing}; it has {sorted(names.values())}"
            )
        self.names = names
        self.last_inference_s = None

    def __call__(self, frame_bgr) -> list[Instance]:
        return self.predict(frame_bgr)

    def predict(self, frame_bgr) -> list[Instance]:
        t0 = time.monotonic()
        result = self.model.predict(
            frame_bgr, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        self.last_inference_s = time.monotonic() - t0
        if result.masks is None or result.boxes is None:
            return []
        shape = frame_bgr.shape[:2]
        out = []
        for poly, cls, conf in zip(
            result.masks.xy, result.boxes.cls.tolist(), result.boxes.conf.tolist()
        ):
            name = self.names.get(int(cls))
            if name not in CLASS_NAMES or len(poly) < 3:
                continue
            out.append(Instance(name, polygon_mask(np.asarray(poly), shape), float(conf)))
        return out

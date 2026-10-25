"""
Camera calibration on the rig: move UMP 1 by known amounts and watch the tip.

Put the needle tip in view, in clear liquid with nothing it can touch within a
few hundred micrometres, then run (with the rig launched):

    ros2 run ump_suite calibrate_camera --confirm

1. X is pulled back in --background-step-um steps (500) until a step changes
   nothing on screen: the needle has left the view, whatever the camera's field
   (at most --max-background-um, 6000). That frame is the background; the tip
   is then found by differencing, with no model. Use --tip-source vision to
   take the tip from /vision/scene instead.
2. The needle returns to where it started.
3. X and Y move +/-step and D moves back by step and 2 x step (D only ever
   retracts), each time returning to the start; the tip is found in each frame.
4. µm per pixel, the image <-> X/Y mapping and the D axis's tilt are fitted and
   saved to ~/.config/ump_suite/vision_calibration.json (--dry-run prints only).

The needle always ends where it started, also after an error or Ctrl-C.
--save-frames DIR keeps every frame used, with the tip marked, for checking.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray, String

from .ros_interfaces import (
    TOPIC_CAM_IMAGE_COMPRESSED,
    TOPIC_UMP_LIVE,
    TOPIC_UMP_TARGET,
    TOPIC_VISION_SCENE,
)
from .vision import calibration
from .vision.needle import changed_area, needle_mask_from_difference, tip_from_difference

AXIS = {"X": 0, "Y": 1, "Z": 2, "D": 3}
STEP_LIMIT_UM = 300


class CalibrationNode(Node):
    def __init__(self):
        super().__init__("camera_calibration")
        self._cond = threading.Condition()
        self.live = None
        self.live_stamp = 0.0
        self.frame = None
        self.frame_stamp = 0.0
        self.scene = None
        self.pub_target = self.create_publisher(Int32MultiArray, TOPIC_UMP_TARGET, 10)
        self.create_subscription(Int32MultiArray, TOPIC_UMP_LIVE, self._on_live, 10)
        self.create_subscription(CompressedImage, TOPIC_CAM_IMAGE_COMPRESSED, self._on_image, 2)
        self.create_subscription(String, TOPIC_VISION_SCENE, self._on_scene, 5)

    def _on_live(self, msg):
        with self._cond:
            self.live = [float(v) for v in msg.data[:4]]
            self.live_stamp = time.monotonic()
            self._cond.notify_all()

    def _on_image(self, msg):
        with self._cond:
            self.frame = msg
            self.frame_stamp = time.time()
            self._cond.notify_all()

    def _on_scene(self, msg):
        try:
            scene = json.loads(msg.data)
        except ValueError:
            return
        with self._cond:
            self.scene = scene
            self._cond.notify_all()

    def wait_live(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        with self._cond:
            while self.live is None or time.monotonic() - self.live_stamp > 0.5:
                if time.monotonic() > deadline:
                    raise RuntimeError("no fresh /ump/live (is the rig running?)")
                self._cond.wait(0.1)
            return list(self.live)

    def goto(self, pos, speed, tol=1.5, timeout=20.0):
        self.pub_target.publish(
            Int32MultiArray(data=[int(round(v)) for v in pos[:4]] + [int(speed)])
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            live = self.wait_live()
            if max(abs(a - b) for a, b in zip(live, pos)) <= tol:
                return live
            time.sleep(0.05)
        raise RuntimeError(f"UMP 1 did not reach {pos}")

    def gray_after(self, after, frames=3, timeout=5.0):
        """Mean of `frames` grey frames taken after wall time `after`."""
        import cv2

        imgs, last = [], None
        deadline = time.monotonic() + timeout
        while len(imgs) < frames:
            with self._cond:
                self._cond.wait(0.2)
                msg, stamp = self.frame, self.frame_stamp
            if time.monotonic() > deadline:
                raise RuntimeError("no fresh camera frames")
            if msg is None or stamp <= after or msg is last:
                continue
            last = msg
            img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs.append(img.astype(np.float32))
        return np.mean(imgs, axis=0).astype(np.uint8)

    def scene_after(self, after, timeout=5.0):
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                if self.scene and float(self.scene.get("stamp") or 0) > after:
                    return self.scene
                if time.monotonic() > deadline:
                    raise RuntimeError("no fresh /vision/scene")
                self._cond.wait(0.2)


def find_background(goto, grab, start, *, step_um=500.0, max_um=6000.0, min_change_px=150):
    """
    Pull X back step by step until the needle has left the view.

    `goto(position)` moves and waits; `grab()` returns a settled grey frame. A
    frame is accepted as the background when the step to it changed fewer than
    `min_change_px` pixels and, compared with the starting frame, the whole
    needle (a changed region reaching the frame edge, where the needle enters)
    has gone. The second test stops a needle that only crept a few pixels from
    passing for an empty view. Returns (background, distance pulled back);
    raises RuntimeError if the needle is still in view after `max_um`, or X
    would leave the stage range.
    """
    first = previous = grab()
    pulled = 0.0
    while pulled < max_um:
        pulled = min(pulled + step_um, max_um)
        position = list(start)
        position[0] -= pulled
        if position[0] < 0:
            raise RuntimeError("pulling X back would leave the stage range; start further in")
        goto(position)
        frame = grab()
        if (changed_area(frame, previous) < min_change_px
                and needle_mask_from_difference(first, frame) is not None):
            return frame, pulled
        previous = frame
    raise RuntimeError(f"the needle was still in view after pulling X back {max_um:.0f} um")


def run(node, args):
    import cv2

    start = node.wait_live()
    print(
        f"[calibrate] start position X {start[0]:.0f}  Y {start[1]:.0f}  "
        f"Z {start[2]:.0f}  D {start[3]:.0f} um"
    )
    save_dir = Path(args.save_frames).expanduser() if args.save_frames else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    background = None
    if args.tip_source == "difference":
        background, pulled = find_background(
            lambda pos: node.goto(pos, args.speed),
            lambda: (time.sleep(args.settle), node.gray_after(time.time(), frames=3))[1],
            start, step_um=args.background_step_um, max_um=args.max_background_um)
        print(f"[calibrate] background taken {pulled:.0f} um back along X")
        if save_dir:
            cv2.imwrite(str(save_dir / "background.png"), background)
        node.goto(start, args.speed)

    def tip_now(label):
        time.sleep(args.settle)
        t = time.time()
        if args.tip_source == "vision":
            scene = node.scene_after(t)
            if not scene.get("needle"):
                raise RuntimeError(f"{label}: the vision scene has no needle")
            tip = tuple(scene["needle"]["tip"])
            gray = node.gray_after(t, frames=1)
        else:
            gray = node.gray_after(t, frames=3)
            found = tip_from_difference(gray, background)
            if found is None:
                raise RuntimeError(
                    f"{label}: no needle found against the background; is the tip in view?"
                )
            tip = found[0]
        if save_dir:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.circle(vis, (int(tip[0]), int(tip[1])), 8, (0, 0, 255), 2)
            cv2.imwrite(str(save_dir / f"{label}.png"), vis)
        print(f"[calibrate] {label:8s} tip at ({tip[0]:7.1f}, {tip[1]:7.1f}) px")
        return tip, gray.shape

    base_tip, shape = tip_now("start")
    plan = [
        ("X", args.step),
        ("X", -args.step),
        ("Y", args.step),
        ("Y", -args.step),
        ("D", -args.step),
        ("D", -2 * args.step),
    ]
    samples = []
    for axis, delta in plan:
        pos = list(start)
        pos[AXIS[axis]] += delta
        node.goto(pos, args.speed)
        tip, _ = tip_now(f"{axis}{delta:+.0f}")
        samples.append((axis, delta, base_tip, tip))
        node.goto(start, args.speed)
    end_tip, _ = tip_now("end")
    drift = float(np.hypot(end_tip[0] - base_tip[0], end_tip[1] - base_tip[1]))
    cal = calibration.fit(
        samples,
        (shape[1], shape[0]),
        notes=f"step {args.step:.0f} um, tip source {args.tip_source}, "
        f"return drift {drift:.1f} px",
    )
    print(json.dumps(cal.to_dict(), indent=2))
    print(
        f"[calibrate] {cal.um_per_px:.3f} um/px, D tilt {cal.theta_deg:.1f} deg, "
        f"worst residual {cal.residual_px:.1f} px, return drift {drift:.1f} px"
    )
    if drift > 5.0:
        print(
            "[calibrate] WARNING: the tip did not come back to where it started; "
            "check for backlash or a bad tip"
        )
    if args.dry_run:
        print("[calibrate] dry run: not saved")
    else:
        path = cal.save(args.output)
        print(f"[calibrate] saved {path}")
    return cal


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--confirm",
        action="store_true",
        help="required: the needle is in clear liquid with nothing within reach",
    )
    p.add_argument("--step", type=float, default=100.0, help="move size per axis (um)")
    p.add_argument("--background-step-um", type=float, default=500.0,
                   help="X pull-back per step while looking for a needle-free frame (um)")
    p.add_argument("--max-background-um", type=float, default=6000.0,
                   help="give up if the needle is still in view after pulling X back this far")
    p.add_argument("--tip-source", choices=("difference", "vision"), default="difference")
    p.add_argument("--speed", type=int, default=500)
    p.add_argument("--settle", type=float, default=0.6, help="seconds to wait after each move")
    p.add_argument("--output", default=None, help=f"default {calibration.DEFAULT_PATH}")
    p.add_argument("--save-frames", default=None)
    p.add_argument("--dry-run", action="store_true", help="fit and print, do not save")
    args = p.parse_args(argv)
    if not args.confirm:
        p.error("refusing to move without --confirm (read the description first)")
    if not 10 <= args.step <= STEP_LIMIT_UM:
        p.error(f"--step must be within 10..{STEP_LIMIT_UM} um")
    if not 100 <= args.background_step_um <= 1000:
        p.error("--background-step-um must be within 100..1000 um")
    if not args.background_step_um <= args.max_background_um <= 10000:
        p.error("--max-background-um must be within --background-step-um..10000 um")

    rclpy.init()
    node = CalibrationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    start = None
    code = 0
    try:
        start = node.wait_live()
        run(node, args)
    except (RuntimeError, ValueError, KeyboardInterrupt) as exc:
        print(f"[calibrate] FAILED: {exc}", file=sys.stderr)
        code = 1
    finally:
        if start is not None:
            try:
                node.goto(start, args.speed)
                print("[calibrate] back at the start position")
            except Exception as exc:
                print(
                    f"[calibrate] could not return to the start position: {exc}", file=sys.stderr
                )
                code = 1
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

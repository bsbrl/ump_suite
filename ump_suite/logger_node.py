"""Dataset logger.

When acquisition is running, this node periodically writes one CSV row per
"timestep" containing:
  * the latest live state of UMP1 and UMP2
  * the most recent commanded *target* for each of them (latest, not consumed)
  * the path of the camera frame that was saved on this tick
  * the latest HEKA resistance estimate, when one is available
  * the pressure applied to the device and the pressure measured back, in mbar
  * wall-clock, camera and state timestamps, so a late tick or a stalled camera
    is detectable after the fact rather than silently producing a well-formed
    row whose image does not match its position

It also forwards a record path to the camera node so that the matching mp4
video file is captured for the same trial.
"""

import csv
import math
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, Int32MultiArray, String
from std_srvs.srv import Trigger

from .ros_interfaces import (
    SRV_ACQ_START,
    SRV_ACQ_STOP,
    TOPIC_CAM_IMAGE_COMPRESSED,
    TOPIC_CAM_REC_CMD,
    TOPIC_HEKA_RESISTANCE,
    TOPIC_PRESSURE_MEASURED,
    TOPIC_PRESSURE_TARGET,
    TOPIC_UMP_LIVE,
    TOPIC_UMP_TARGET,
    TOPIC_UMP2_LIVE,
    TOPIC_UMP2_TARGET,
    latched_qos,
)


CSV_HEADER = [
    "timestep",
    "current_x",  "current_y",  "current_z",  "current_d",
    "target_x",   "target_y",   "target_z",   "target_d",
    "current_x2", "current_y2", "current_z2", "current_d2",
    "target_x2",  "target_y2",  "target_z2",  "target_d2",
    "image_path",
    "resistance_mohm",
    "target_pressure",
    "measured_pressure",
    # Timing. Without these there is no way, after the fact, to tell a late tick
    # or a stalled camera from a healthy row: a frozen camera silently writes the
    # same frame into many rows and the dataset still looks well formed.
    #   wall_time    - POSIX time this row was written
    #   image_stamp  - POSIX time the camera stamped this frame, blank if unknown
    #   state_stamp  - POSIX time the newest manipulator state arrived
    #   image_age_s  - wall_time - image_stamp, the staleness of the saved frame
    "wall_time",
    "image_stamp",
    "state_stamp",
    "image_age_s",
]


def _xyzd(values):
    """Coerce a 4+ element list into a length-4 int tuple, defaulting to zeros."""
    if values is None or len(values) < 4:
        return (0, 0, 0, 0)
    return tuple(int(v) for v in values[:4])


class LoggerNode(Node):
    def __init__(self):
        super().__init__("logger_node")
        self.declare_parameter("log_interval_ms", 500)
        # Warn when the saved frame is older than this relative to the row.
        self.declare_parameter("stale_image_warn_s", 0.5)

        # Latest live state, each with the POSIX time it arrived.
        self.latest_live_ump = None
        self.latest_live_ump2 = None
        self.latest_state_stamp = None
        self.latest_image_msg = None
        self.latest_resistance_mohm = None
        # Pressure in mbar; None until the pressure node publishes. The target
        # is the value actually written to the device, not the raw request.
        self.latest_target_pressure = None
        self.latest_measured_pressure = None

        # Latest commanded target. These are *not* cleared after each tick:
        # if the user stops issuing commands, the most recent target keeps
        # appearing in subsequent rows so target/current can always be diffed.
        self.latest_target_ump = None
        self.latest_target_ump2 = None

        self.acquiring = False
        self.trial_name = None
        self.log_path = None
        self.frames_dir = None
        self.video_path = None
        self.frame_index = 0
        self.timestep = 0

        self.log_file = None
        self.writer = None
        self._stale_image_warn_s = float(
            self.get_parameter("stale_image_warn_s").value
        )
        self._stale_rows = 0
        self._first_row_time = None
        self._last_row_time = None

        # Live state subscribers.
        self.create_subscription(Int32MultiArray, TOPIC_UMP_LIVE,   self.on_ump_live,   10)
        self.create_subscription(Int32MultiArray, TOPIC_UMP2_LIVE,  self.on_ump2_live,  10)

        # Target subscribers (snoop on whatever the GUI / VLA publishes).
        self.create_subscription(Int32MultiArray, TOPIC_UMP_TARGET,  self.on_ump_target,  10)
        self.create_subscription(Int32MultiArray, TOPIC_UMP2_TARGET, self.on_ump2_target, 10)

        self.create_subscription(CompressedImage, TOPIC_CAM_IMAGE_COMPRESSED, self.on_img, 10)
        self.create_subscription(Float32, TOPIC_HEKA_RESISTANCE, self.on_resistance, 10)
        self.create_subscription(
            Float32, TOPIC_PRESSURE_TARGET, self.on_target_pressure, latched_qos()
        )
        self.create_subscription(
            Float32, TOPIC_PRESSURE_MEASURED, self.on_measured_pressure, 10
        )

        self.pub_rec_cmd = self.create_publisher(String, TOPIC_CAM_REC_CMD, 10)

        self.create_service(Trigger, SRV_ACQ_START, self.on_start)
        self.create_service(Trigger, SRV_ACQ_STOP,  self.on_stop)

        interval = int(self.get_parameter("log_interval_ms").value) / 1000.0
        self.create_timer(interval, self.tick)

    # ── Subscriber callbacks ───────────────────────────────────────────────
    def on_ump_live(self, msg: Int32MultiArray):
        self.latest_live_ump = list(msg.data)
        self.latest_state_stamp = time.time()

    def on_ump2_live(self, msg: Int32MultiArray):
        self.latest_live_ump2 = list(msg.data)
        self.latest_state_stamp = time.time()

    def on_ump_target(self, msg: Int32MultiArray):
        # /ump/target carries [x,y,z,d,speed]; we only log [x,y,z,d].
        self.latest_target_ump = list(msg.data)

    def on_ump2_target(self, msg: Int32MultiArray):
        self.latest_target_ump2 = list(msg.data)

    def on_img(self, msg: CompressedImage):
        self.latest_image_msg = msg

    def on_resistance(self, msg: Float32):
        self.latest_resistance_mohm = float(msg.data)

    def on_target_pressure(self, msg: Float32):
        value = float(msg.data)
        if math.isfinite(value):
            self.latest_target_pressure = value

    def on_measured_pressure(self, msg: Float32):
        value = float(msg.data)
        if math.isfinite(value):
            self.latest_measured_pressure = value

    # ── Trial setup ────────────────────────────────────────────────────────
    @staticmethod
    def _next_trial_id():
        """Lowest unused trial number across every output directory.

        Scanning only `logs/` is not enough: deleting a CSV while its frame
        directory survives would hand the number back out, and the new run would
        write `frame_000000.png` straight over the old trial's frames.
        """
        used = set()

        def note(stem):
            if stem.startswith("trial_") and stem[len("trial_"):].isdigit():
                used.add(int(stem[len("trial_"):]))

        for folder, suffix in (("logs", ".csv"), ("saved_frames", ""), ("saved_videos", ".mp4")):
            if not os.path.isdir(folder):
                continue
            for name in os.listdir(folder):
                note(name[: -len(suffix)] if suffix and name.endswith(suffix) else name)
        return max(used, default=0) + 1

    def _setup_trial(self):
        os.makedirs("logs", exist_ok=True)
        os.makedirs("saved_frames", exist_ok=True)
        os.makedirs("saved_videos", exist_ok=True)

        next_trial = self._next_trial_id()

        self.trial_name = f"trial_{next_trial}"
        self.log_path   = os.path.join("logs",         f"{self.trial_name}.csv")
        self.frames_dir = os.path.join("saved_frames", self.trial_name)
        self.video_path = os.path.join("saved_videos", f"{self.trial_name}.mp4")
        os.makedirs(self.frames_dir, exist_ok=True)

        self.frame_index = 0
        self.timestep = 0
        self._stale_rows = 0
        self._first_row_time = None
        self._last_row_time = None

    def _open_csv(self):
        self.log_file = open(self.log_path, "w", newline="")
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(CSV_HEADER)

    # ── Service handlers ───────────────────────────────────────────────────
    def on_start(self, _req, res):
        if self.acquiring:
            res.success = True
            res.message = "Already acquiring."
            return res

        self._setup_trial()
        self._open_csv()
        self.pub_rec_cmd.publish(String(data=self.video_path))

        self.acquiring = True
        res.success = True
        res.message = f"Acquisition started: {self.trial_name}"
        self.get_logger().info(res.message)
        return res

    def on_stop(self, _req, res):
        if not self.acquiring:
            res.success = True
            res.message = "Already stopped."
            return res

        self.acquiring = False
        self.pub_rec_cmd.publish(String(data=""))
        self._report_achieved_rate()

        try:
            if self.log_file:
                self.log_file.flush()
                self.log_file.close()
        except Exception:
            pass

        self.log_file = None
        self.writer = None

        res.success = True
        res.message = "Acquisition stopped."
        self.get_logger().info(res.message)
        return res

    # ── Per-tick logging ───────────────────────────────────────────────────
    def _save_current_frame(self):
        """Decode the latest JPEG and write it to disk; return its path or ''."""
        if self.latest_image_msg is None:
            return ""
        try:
            data = np.frombuffer(self.latest_image_msg.data, dtype=np.uint8)
            frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return ""
            fname = os.path.join(self.frames_dir, f"frame_{self.frame_index:06d}.png")
            cv2.imwrite(fname, frame_bgr)
            self.frame_index += 1
            return fname
        except Exception as e:
            self.get_logger().warn(f"Frame save error: {e}")
            return ""

    def _report_achieved_rate(self):
        """Compare the rate actually achieved with the configured one.

        The timer is best effort and `_save_current_frame` encodes a PNG inside
        the callback, so the configured period is a request rather than a fact.
        The dataset fps and the policy control rate are both derived from what
        actually happened here, so a drift is worth stating out loud.
        """
        if (self._first_row_time is None or self._last_row_time is None
                or self.timestep < 2):
            return
        span = self._last_row_time - self._first_row_time
        if span <= 0:
            return
        achieved = (self.timestep - 1) / span
        configured = 1000.0 / max(
            1, int(self.get_parameter("log_interval_ms").value)
        )
        self.get_logger().info(
            f"{self.trial_name}: {self.timestep} rows in {span:.1f}s = "
            f"{achieved:.2f} Hz (configured {configured:.2f} Hz)"
        )
        if achieved < 0.9 * configured:
            self.get_logger().warn(
                f"logging ran at {achieved:.2f} Hz, below 90% of the configured "
                f"{configured:.2f} Hz. Convert this data with --fps "
                f"{achieved:.0f} and match the policy control rate to it, or "
                "reduce the load on this node."
            )

    def _image_stamp(self):
        """POSIX time from the camera frame header, or "" when unavailable."""
        if self.latest_image_msg is None:
            return ""
        try:
            stamp = self.latest_image_msg.header.stamp
            value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        except AttributeError:
            return ""
        return value if math.isfinite(value) and value > 0 else ""

    def _warn_if_stale(self, image_age):
        """Say so when the saved frame is much older than the row it belongs to.

        A stalled camera is otherwise invisible: the logger keeps writing the
        last frame it received, and every row still looks complete.
        """
        if image_age == "" or image_age <= self._stale_image_warn_s:
            self._stale_rows = 0
            return
        self._stale_rows += 1
        if self._stale_rows % 10 == 1:
            self.get_logger().warn(
                f"camera frame is {image_age:.2f}s older than this row "
                f"({self._stale_rows} consecutive); the recorded images may not "
                "match the recorded positions"
            )

    def tick(self):
        if not self.acquiring or self.writer is None:
            return

        cx,  cy,  cz,  cd  = _xyzd(self.latest_live_ump)
        cx2, cy2, cz2, cd2 = _xyzd(self.latest_live_ump2)

        tx,  ty,  tz,  td  = _xyzd(self.latest_target_ump)
        tx2, ty2, tz2, td2 = _xyzd(self.latest_target_ump2)
        resistance = (
            float(self.latest_resistance_mohm)
            if self.latest_resistance_mohm is not None
            else ""
        )
        # Pressure in mbar, negative = pull. The target is what the pressure
        # node actually wrote to the device (post-clamp), so it can never claim
        # a pressure the controller never received. Blank until first published.
        target_pressure = (
            float(self.latest_target_pressure)
            if self.latest_target_pressure is not None
            else ""
        )
        measured_pressure = (
            float(self.latest_measured_pressure)
            if self.latest_measured_pressure is not None
            else ""
        )

        # Stamp the frame that is about to be written, not the one that may
        # arrive while it is being encoded.
        image_stamp = self._image_stamp()
        image_path = self._save_current_frame()
        wall_time = time.time()
        image_age = "" if image_stamp == "" else round(wall_time - image_stamp, 6)

        self.writer.writerow([
            self.timestep,
            cx, cy, cz, cd,
            tx, ty, tz, td,
            cx2, cy2, cz2, cd2,
            tx2, ty2, tz2, td2,
            image_path,
            resistance,
            target_pressure,
            measured_pressure,
            round(wall_time, 6),
            "" if image_stamp == "" else round(image_stamp, 6),
            "" if self.latest_state_stamp is None else round(self.latest_state_stamp, 6),
            image_age,
        ])
        if self._first_row_time is None:
            self._first_row_time = wall_time
        self._last_row_time = wall_time
        self.timestep += 1
        self._warn_if_stale(image_age)

        try:
            self.log_file.flush()
        except Exception:
            pass


def main():
    rclpy.init()
    node = LoggerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

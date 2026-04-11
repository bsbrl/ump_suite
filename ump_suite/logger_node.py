"""Dataset logger.

When acquisition is running, this node periodically writes one CSV row per
"timestep" containing:
  * the latest live state of UMP1, UMP2 and the motor
  * the most recent commanded *target* for each of them (latest, not consumed)
  * the path of the camera frame that was saved on this tick

It also forwards a record path to the camera node so that the matching mp4
video file is captured for the same trial.
"""

import csv
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32, Int32MultiArray, String
from std_srvs.srv import Trigger

from .ros_interfaces import (
    SRV_ACQ_START,
    SRV_ACQ_STOP,
    TOPIC_CAM_IMAGE_COMPRESSED,
    TOPIC_CAM_REC_CMD,
    TOPIC_MOTOR_LIVE,
    TOPIC_MOTOR_TGT,
    TOPIC_UMP_LIVE,
    TOPIC_UMP_TARGET,
    TOPIC_UMP2_LIVE,
    TOPIC_UMP2_TARGET,
)


CSV_HEADER = [
    "timestep",
    "current_x",  "current_y",  "current_z",  "current_d",  "current_motor",
    "target_x",   "target_y",   "target_z",   "target_d",   "target_motor",
    "current_x2", "current_y2", "current_z2", "current_d2",
    "target_x2",  "target_y2",  "target_z2",  "target_d2",
    "image_path",
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

        # Latest live state.
        self.latest_live_ump = None
        self.latest_live_ump2 = None
        self.latest_live_motor = None
        self.latest_image_msg = None

        # Latest commanded target. These are *not* cleared after each tick:
        # if the user stops issuing commands, the most recent target keeps
        # appearing in subsequent rows so target/current can always be diffed.
        self.latest_target_ump = None
        self.latest_target_ump2 = None
        self.latest_target_motor = None

        self.acquiring = False
        self.trial_name = None
        self.log_path = None
        self.frames_dir = None
        self.video_path = None
        self.frame_index = 0
        self.timestep = 0

        self.log_file = None
        self.writer = None

        # Live state subscribers.
        self.create_subscription(Int32MultiArray, TOPIC_UMP_LIVE,   self.on_ump_live,   10)
        self.create_subscription(Int32MultiArray, TOPIC_UMP2_LIVE,  self.on_ump2_live,  10)
        self.create_subscription(Int32,           TOPIC_MOTOR_LIVE, self.on_motor_live, 10)

        # Target subscribers (snoop on whatever the GUI / VLA publishes).
        self.create_subscription(Int32MultiArray, TOPIC_UMP_TARGET,  self.on_ump_target,  10)
        self.create_subscription(Int32MultiArray, TOPIC_UMP2_TARGET, self.on_ump2_target, 10)
        self.create_subscription(Int32,           TOPIC_MOTOR_TGT,   self.on_motor_target, 10)

        self.create_subscription(CompressedImage, TOPIC_CAM_IMAGE_COMPRESSED, self.on_img, 10)

        self.pub_rec_cmd = self.create_publisher(String, TOPIC_CAM_REC_CMD, 10)

        self.create_service(Trigger, SRV_ACQ_START, self.on_start)
        self.create_service(Trigger, SRV_ACQ_STOP,  self.on_stop)

        interval = int(self.get_parameter("log_interval_ms").value) / 1000.0
        self.create_timer(interval, self.tick)

    # ── Subscriber callbacks ───────────────────────────────────────────────
    def on_ump_live(self, msg: Int32MultiArray):
        self.latest_live_ump = list(msg.data)

    def on_ump2_live(self, msg: Int32MultiArray):
        self.latest_live_ump2 = list(msg.data)

    def on_motor_live(self, msg: Int32):
        self.latest_live_motor = int(msg.data)

    def on_ump_target(self, msg: Int32MultiArray):
        # /ump/target carries [x,y,z,d,speed]; we only log [x,y,z,d].
        self.latest_target_ump = list(msg.data)

    def on_ump2_target(self, msg: Int32MultiArray):
        self.latest_target_ump2 = list(msg.data)

    def on_motor_target(self, msg: Int32):
        self.latest_target_motor = int(msg.data)

    def on_img(self, msg: CompressedImage):
        self.latest_image_msg = msg

    # ── Trial setup ────────────────────────────────────────────────────────
    def _setup_trial(self):
        os.makedirs("logs", exist_ok=True)
        os.makedirs("saved_frames", exist_ok=True)
        os.makedirs("saved_videos", exist_ok=True)

        # Pick the next free trial number by inspecting `logs/`.
        existing = []
        for fn in os.listdir("logs"):
            if fn.startswith("trial_") and fn.endswith(".csv"):
                mid = fn[len("trial_"):-4]
                if mid.isdigit():
                    existing.append(int(mid))
        next_trial = max(existing, default=0) + 1

        self.trial_name = f"trial_{next_trial}"
        self.log_path   = os.path.join("logs",         f"{self.trial_name}.csv")
        self.frames_dir = os.path.join("saved_frames", self.trial_name)
        self.video_path = os.path.join("saved_videos", f"{self.trial_name}.mp4")
        os.makedirs(self.frames_dir, exist_ok=True)

        self.frame_index = 0
        self.timestep = 0

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

    def tick(self):
        if not self.acquiring or self.writer is None:
            return

        cx,  cy,  cz,  cd  = _xyzd(self.latest_live_ump)
        cx2, cy2, cz2, cd2 = _xyzd(self.latest_live_ump2)
        cm = int(self.latest_live_motor) if self.latest_live_motor is not None else 0

        tx,  ty,  tz,  td  = _xyzd(self.latest_target_ump)
        tx2, ty2, tz2, td2 = _xyzd(self.latest_target_ump2)
        tm = int(self.latest_target_motor) if self.latest_target_motor is not None else 0

        image_path = self._save_current_frame()

        self.writer.writerow([
            self.timestep,
            cx, cy, cz, cd, cm,
            tx, ty, tz, td, tm,
            cx2, cy2, cz2, cd2,
            tx2, ty2, tz2, td2,
            image_path,
        ])
        self.timestep += 1

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

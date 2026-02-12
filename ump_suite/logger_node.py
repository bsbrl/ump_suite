import os
import csv

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Int32, String
from std_srvs.srv import Trigger
from sensor_msgs.msg import CompressedImage

import numpy as np
import cv2

from .ros_interfaces import (
    TOPIC_UMP_TARGET, TOPIC_UMP_LIVE,
    TOPIC_MOTOR_TGT, TOPIC_MOTOR_LIVE,
    TOPIC_CAM_IMAGE_COMPRESSED, TOPIC_CAM_REC_CMD,
    SRV_ACQ_START, SRV_ACQ_STOP,
)

class LoggerNode(Node):
    def __init__(self):
        super().__init__("logger_node")
        self.declare_parameter("log_interval_ms", 500)

        self.latest_target_ump = None
        self.latest_live_ump = None
        self.latest_target_motor = None
        self.latest_live_motor = None
        self.latest_image_msg = None  # CompressedImage

        self.acquiring = False
        self.trial_name = None
        self.log_path = None
        self.frames_dir = None
        self.video_path = None
        self.frame_index = 0

        self.log_file = None
        self.writer = None

        self.sub_ump_tgt = self.create_subscription(
            Int32MultiArray, TOPIC_UMP_TARGET, self.on_ump_tgt, 10
        )
        self.sub_ump_live = self.create_subscription(
            Int32MultiArray, TOPIC_UMP_LIVE, self.on_ump_live, 10
        )
        self.sub_motor_tgt = self.create_subscription(
            Int32, TOPIC_MOTOR_TGT, self.on_motor_tgt, 10
        )
        self.sub_motor_live = self.create_subscription(
            Int32, TOPIC_MOTOR_LIVE, self.on_motor_live, 10
        )
        self.sub_img = self.create_subscription(
            CompressedImage, TOPIC_CAM_IMAGE_COMPRESSED, self.on_img, 10
        )

        self.pub_rec_cmd = self.create_publisher(String, TOPIC_CAM_REC_CMD, 10)

        self.srv_start = self.create_service(Trigger, SRV_ACQ_START, self.on_start)
        self.srv_stop = self.create_service(Trigger, SRV_ACQ_STOP, self.on_stop)

        interval = int(self.get_parameter("log_interval_ms").value) / 1000.0
        self.timer = self.create_timer(interval, self.tick)

    def on_ump_tgt(self, msg): self.latest_target_ump = list(msg.data)
    def on_ump_live(self, msg): self.latest_live_ump = list(msg.data)
    def on_motor_tgt(self, msg): self.latest_target_motor = int(msg.data)
    def on_motor_live(self, msg): self.latest_live_motor = int(msg.data)
    def on_img(self, msg): self.latest_image_msg = msg

    def _setup_trial(self):
        os.makedirs("logs", exist_ok=True)
        os.makedirs("saved_frames", exist_ok=True)
        os.makedirs("saved_videos", exist_ok=True)

        existing = []
        for fn in os.listdir("logs"):
            if fn.startswith("trial_") and fn.endswith(".csv"):
                mid = fn[len("trial_"):-4]
                if mid.isdigit():
                    existing.append(int(mid))
        next_trial = max(existing, default=0) + 1

        self.trial_name = f"trial_{next_trial}"
        self.log_path = os.path.join("logs", f"{self.trial_name}.csv")
        self.frames_dir = os.path.join("saved_frames", self.trial_name)
        self.video_path = os.path.join("saved_videos", f"{self.trial_name}.mp4")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.frame_index = 0

    def _open_csv(self):
        self.log_file = open(self.log_path, "w", newline="")
        self.writer = csv.writer(self.log_file)
        self.writer.writerow([
            "target_x", "target_y", "target_z", "target_d",
            "target_speed",
            "target_motor",
            "current_x", "current_y", "current_z", "current_d",
            "current_motor",
        ])

    def on_start(self, _req, res):
        if self.acquiring:
            res.success = True
            res.message = "Already acquiring."
            return res

        self._setup_trial()
        self._open_csv()

        # Tell camera to record to per-trial path
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

        # Tell camera to stop recording
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

    def tick(self):
        if not self.acquiring or self.writer is None:
            return

        tx, ty, tz, td, ts = (0, 0, 0, 0, 0)
        if self.latest_target_ump and len(self.latest_target_ump) >= 5:
            tx, ty, tz, td, ts = [int(v) for v in self.latest_target_ump[:5]]

        tm = int(self.latest_target_motor) if self.latest_target_motor is not None else 0

        cx, cy, cz, cd = (0, 0, 0, 0)
        if self.latest_live_ump and len(self.latest_live_ump) >= 4:
            cx, cy, cz, cd = [int(v) for v in self.latest_live_ump[:4]]

        cm = int(self.latest_live_motor) if self.latest_live_motor is not None else 0

        self.writer.writerow([tx, ty, tz, td, ts, tm, cx, cy, cz, cd, cm])
        try:
            self.log_file.flush()
        except Exception:
            pass

        # Frame snapshot (decode JPEG)
        if self.latest_image_msg is not None:
            try:
                data = np.frombuffer(self.latest_image_msg.data, dtype=np.uint8)
                frame_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    return
                fname = os.path.join(self.frames_dir, f"frame_{self.frame_index:06d}.png")
                cv2.imwrite(fname, frame_bgr)
                self.frame_index += 1
            except Exception as e:
                self.get_logger().warn(f"Frame save error: {e}")

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
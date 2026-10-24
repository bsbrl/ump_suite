"""
ROS2 node: run the embryo segmentation model on the camera and publish the scene.

Each processed frame becomes one JSON scene on /vision/scene (see
vision/geometry.py): every embryo's chorion, yolk and cell, the cell's angle
relative to the needle, the embryo the needle points at, and the distances
along the needle line to its chorion and cell. The scene's `stamp` is the
camera frame's own timestamp, so consumers can ask for a frame taken after a
move finished.

Only the newest frame is processed; frames that arrive while the model runs are
skipped. Without a model file the node stays up, publishes nothing on
/vision/scene and says why on /vision/status, so the rest of the rig is
unaffected.

ROS parameters:
    model_path        YOLO-seg weights (default ~/.local/share/ump_suite/models/embryo_seg.pt)
    conf              detection confidence threshold
    imgsz             inference size (pixels, long side)
    max_hz            upper bound on processed frames per second
    device            "0" for the first GPU, "cpu" to force the CPU
    calibration_path  camera calibration JSON ("" = the default path); adds
                      micrometre distances to the scene
"""

import json
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from .ros_interfaces import (
    TOPIC_CAM_IMAGE_COMPRESSED,
    TOPIC_VISION_SCENE,
    TOPIC_VISION_STATUS,
    latched_qos,
)
from .vision import calibration
from .vision.geometry import build_scene
from .vision.needle import extend_needle
from .vision.segmenter import DEFAULT_MODEL_PATH


class VisionNode(Node):
    def __init__(self, segmenter=None):
        super().__init__("vision_node")
        self.declare_parameter("model_path", str(DEFAULT_MODEL_PATH))
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("imgsz", 1024)
        self.declare_parameter("max_hz", 10.0)
        self.declare_parameter("device", "0")
        self.declare_parameter("calibration_path", "")

        self.pub_scene = self.create_publisher(String, TOPIC_VISION_SCENE, 5)
        self.pub_status = self.create_publisher(String, TOPIC_VISION_STATUS, latched_qos())
        self.create_subscription(CompressedImage, TOPIC_CAM_IMAGE_COMPRESSED, self._on_image, 2)

        self._lock = threading.Condition()
        self._latest = None
        self._stop = False
        self._fps = 0.0
        self._last_pub = None
        self._inference_ms = None
        self._message = "starting"
        self._ready = False
        self._note = {}
        self._calibration = None
        self._cal_mtime = None

        self.segmenter = segmenter
        if self.segmenter is None:
            self._load_model()
        else:
            self._ready, self._message = True, "ready (injected segmenter)"
        self._worker = threading.Thread(target=self._loop, name="vision", daemon=True)
        self._worker.start()
        self.create_timer(1.0, self._publish_status)
        self._publish_status()

    def _load_model(self):
        path = Path(str(self.get_parameter("model_path").value)).expanduser()
        if not path.exists():
            self._message = (
                f"no model at {path}; train one with "
                "python -m ump_suite.vision.tools.train_segmenter"
            )
            self.get_logger().warn(self._message)
            return
        try:
            from .vision.segmenter import Segmenter

            device = str(self.get_parameter("device").value)
            self.segmenter = Segmenter(
                path,
                conf=float(self.get_parameter("conf").value),
                imgsz=int(self.get_parameter("imgsz").value),
                device=None if device == "" else device,
            )
        except Exception as exc:
            self._message = f"model failed to load: {exc}"
            self.get_logger().error(self._message)
            return
        note_path = path.with_suffix(".json")
        if note_path.exists():
            try:
                self._note = json.loads(note_path.read_text())
            except ValueError:
                self._note = {}
        self._ready = True
        self._message = "ready" + (
            " (TEST-ONLY model: unchecked labels)" if self._note.get("test_only") else ""
        )
        self.get_logger().info(f"Segmentation model loaded: {path} - {self._message}")

    def _um_per_px(self):
        raw = str(self.get_parameter("calibration_path").value) or None
        path = Path(raw).expanduser() if raw else calibration.DEFAULT_PATH
        try:
            mtime = path.stat().st_mtime
        except OSError:
            self._calibration, self._cal_mtime = None, None
            return None
        if mtime != self._cal_mtime:
            try:
                self._calibration = calibration.load(path)
            except Exception as exc:
                self.get_logger().warn(f"camera calibration unreadable: {exc}")
                self._calibration = None
            self._cal_mtime = mtime
        return None if self._calibration is None else self._calibration.um_per_px

    def _on_image(self, msg: CompressedImage):
        with self._lock:
            self._latest = msg
            self._lock.notify()

    def _loop(self):
        import cv2

        period = 1.0 / max(float(self.get_parameter("max_hz").value), 0.1)
        last = 0.0
        while True:
            with self._lock:
                while self._latest is None and not self._stop:
                    self._lock.wait(0.5)
                if self._stop:
                    return
                msg, self._latest = self._latest, None
            if not self._ready:
                continue
            wait = period - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
            try:
                frame = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
                if stamp <= 0:
                    stamp = time.time()
                t0 = time.monotonic()
                instances = self.segmenter(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                instances = extend_needle(instances, gray)
                scene = build_scene(
                    instances, frame.shape, um_per_px=self._um_per_px(), stamp=stamp
                )
                elapsed = time.monotonic() - t0
                self._inference_ms = round(elapsed * 1000.0, 1)
                scene["test_only_model"] = bool(self._note.get("test_only"))
                self.pub_scene.publish(String(data=json.dumps(scene)))
                now = time.monotonic()
                if self._last_pub is not None:
                    self._fps = 0.8 * self._fps + 0.2 / max(now - self._last_pub, 1e-3)
                self._last_pub = now
            except Exception as exc:  # keep the node alive; say what went wrong
                self._message = f"frame failed: {exc}"
                self.get_logger().error(self._message)

    def _publish_status(self):
        self.pub_status.publish(
            String(
                data=json.dumps(
                    {
                        "ready": self._ready,
                        "message": self._message,
                        "model": str(self.get_parameter("model_path").value),
                        "test_only": bool(self._note.get("test_only")),
                        "fps": round(self._fps, 2),
                        "inference_ms": self._inference_ms,
                        "calibrated": self._um_per_px() is not None,
                        "stamp": time.time(),
                    }
                )
            )
        )

    def destroy_node(self):
        with self._lock:
            self._stop = True
            self._lock.notify_all()
        self._worker.join(timeout=2.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

# ruff: noqa
import io
import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray
from PIL import Image as PILImage


@dataclass
class SensapexObs:
    image_rgb: np.ndarray  # HxWx3 uint8 RGB
    state: np.ndarray  # shape (8,) float32: [x1,y1,z1,d1, x2,y2,z2,d2]


def _decode_compressed_jpeg_to_rgb(msg: CompressedImage) -> np.ndarray:
    """
    Decode sensor_msgs/CompressedImage (jpeg) into RGB numpy array (H, W, 3), uint8.
    Uses PIL only (no cv2), which avoids numpy/opencv binary version conflicts.
    """
    jpeg_bytes = bytes(msg.data)
    pil = PILImage.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    return np.array(pil, dtype=np.uint8)


class _SensapexROSNode(Node):
    def __init__(
        self,
        *,
        save_preview: bool = True,
        preview_path: str = "sensapex_live.png",
        preview_every_n_frames: int = 5,
    ):
        super().__init__("openpi_sensapex_bridge")

        # --- Subscribers ---
        self.sub_img = self.create_subscription(CompressedImage, "/camera/image/compressed", self._on_img, 10)
        self.sub_ump1_live = self.create_subscription(Int32MultiArray, "/ump/live", self._on_ump1_live, 10)
        self.sub_ump2_live = self.create_subscription(Int32MultiArray, "/ump2/live", self._on_ump2_live, 10)

        # --- Publishers ---
        self.pub_ump1_target = self.create_publisher(Int32MultiArray, "/ump/target", 10)
        self.pub_ump2_target = self.create_publisher(Int32MultiArray, "/ump2/target", 10)

        # --- Latest state ---
        self._lock = threading.Lock()
        self._latest_image_rgb = None  # np.ndarray HxWx3
        self._latest_ump1 = None  # [x1, y1, z1, d1]
        self._latest_ump2 = None  # [x2, y2, z2, d2]

        # --- Preview saving (works even when SSH'ed in) ---
        self._save_preview = bool(save_preview)
        self._preview_path = str(preview_path)
        self._preview_every_n_frames = int(max(1, preview_every_n_frames))
        self._frame_counter = 0

    def _on_img(self, msg: CompressedImage):
        try:
            rgb = _decode_compressed_jpeg_to_rgb(msg)
            with self._lock:
                self._latest_image_rgb = rgb

            if self._save_preview:
                self._frame_counter += 1
                if self._frame_counter % self._preview_every_n_frames == 0:
                    try:
                        PILImage.fromarray(rgb).save(self._preview_path)
                    except Exception as e:
                        self.get_logger().warn(f"Preview save failed: {e}")

        except Exception as e:
            self.get_logger().warn(f"Image decode failed: {e}")

    def _on_ump1_live(self, msg: Int32MultiArray):
        if len(msg.data) < 4:
            return
        with self._lock:
            self._latest_ump1 = [int(v) for v in msg.data[:4]]

    def _on_ump2_live(self, msg: Int32MultiArray):
        if len(msg.data) < 4:
            return
        with self._lock:
            self._latest_ump2 = [int(v) for v in msg.data[:4]]

    def get_latest(self):
        with self._lock:
            img = None if self._latest_image_rgb is None else self._latest_image_rgb.copy()
            ump1 = None if self._latest_ump1 is None else list(self._latest_ump1)
            ump2 = None if self._latest_ump2 is None else list(self._latest_ump2)
        return img, ump1, ump2

    def send_action_absolute(self, x1, y1, z1, d1, x2, y2, z2, d2, speed=100):
        """
        Publish absolute targets for both Sensapex uMps.
        Each /ump*/target expects [x, y, z, d, speed].
        """
        ump1_msg = Int32MultiArray()
        ump1_msg.data = [int(x1), int(y1), int(z1), int(d1), int(speed)]
        self.pub_ump1_target.publish(ump1_msg)

        ump2_msg = Int32MultiArray()
        ump2_msg.data = [int(x2), int(y2), int(z2), int(d2), int(speed)]
        self.pub_ump2_target.publish(ump2_msg)


class SensapexEnv:
    def __init__(
        self,
        *,
        save_preview: bool = True,
        preview_path: str = "sensapex_live.png",
        preview_every_n_frames: int = 5,
        default_speed: int = 100,
        wait_timeout_s: float = 10.0,
    ):
        self.default_speed = int(default_speed)

        # IMPORTANT: OpenPI scripts should be run with python3.10 (ROS Humble python)
        rclpy.init(args=None)

        self.node = _SensapexROSNode(
            save_preview=save_preview,
            preview_path=preview_path,
            preview_every_n_frames=preview_every_n_frames,
        )

        # Spin ROS callbacks in a background thread
        self._executor_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._executor_thread.start()

        self._wait_for_first_messages(timeout_s=wait_timeout_s)

    def _wait_for_first_messages(self, timeout_s=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            img, ump1, ump2 = self.node.get_latest()
            if img is not None and ump1 is not None and ump2 is not None:
                return
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for /camera/image/compressed, /ump/live, /ump2/live")

    def get_observation(self) -> SensapexObs:
        img, ump1, ump2 = self.node.get_latest()
        if img is None or ump1 is None or ump2 is None:
            raise RuntimeError("Missing observation components (image/ump1/ump2).")

        x1, y1, z1, d1 = ump1
        x2, y2, z2, d2 = ump2
        state = np.array([x1, y1, z1, d1, x2, y2, z2, d2], dtype=np.float32)
        return SensapexObs(image_rgb=img, state=state)

    def step_absolute(self, action_8d: np.ndarray):
        """
        action_8d: [x1', y1', z1', d1', x2', y2', z2', d2'] (absolute target)
        """
        action_8d = np.asarray(action_8d).reshape(-1)
        if action_8d.shape != (8,):
            raise ValueError(f"Expected action shape (8,), got {action_8d.shape}")

        x1, y1, z1, d1, x2, y2, z2, d2 = action_8d
        self.node.send_action_absolute(x1, y1, z1, d1, x2, y2, z2, d2, speed=self.default_speed)

    def close(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

# ruff: noqa
import io
import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32, Int32MultiArray
from PIL import Image as PILImage


@dataclass
class SensapexObs:
    image_rgb: np.ndarray  # HxWx3 uint8 RGB
    state: np.ndarray  # shape (5,) float32: [x,y,z,d,h_ticks]


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
        self.sub_ump_live = self.create_subscription(Int32MultiArray, "/ump/live", self._on_ump_live, 10)
        self.sub_motor_live = self.create_subscription(Int32, "/motor/live_counts", self._on_motor_live, 10)

        # --- Publishers ---
        self.pub_ump_target = self.create_publisher(Int32MultiArray, "/ump/target", 10)
        self.pub_motor_target = self.create_publisher(Int32, "/motor/target_counts", 10)

        # --- Latest state ---
        self._lock = threading.Lock()
        self._latest_image_rgb = None  # np.ndarray HxWx3
        self._latest_ump = None  # [x,y,z,d]
        self._latest_motor = None  # int ticks

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

    def _on_ump_live(self, msg: Int32MultiArray):
        if len(msg.data) < 4:
            return
        with self._lock:
            self._latest_ump = [int(v) for v in msg.data[:4]]

    def _on_motor_live(self, msg: Int32):
        with self._lock:
            self._latest_motor = int(msg.data)

    def get_latest(self):
        with self._lock:
            img = None if self._latest_image_rgb is None else self._latest_image_rgb.copy()
            ump = None if self._latest_ump is None else list(self._latest_ump)
            mot = self._latest_motor
        return img, ump, mot

    def send_action_absolute(self, x, y, z, d, h_ticks, speed=1000):
        """
        Publish absolute targets.
        - /ump/target expects [x,y,z,d,speed]
        - /motor/target_counts expects Int32 ticks
        """
        ump_msg = Int32MultiArray()
        ump_msg.data = [int(x), int(y), int(z), int(d), int(speed)]
        self.pub_ump_target.publish(ump_msg)

        self.pub_motor_target.publish(Int32(data=int(h_ticks)))


class SensapexEnv:
    def __init__(
        self,
        *,
        save_preview: bool = True,
        preview_path: str = "sensapex_live.png",
        preview_every_n_frames: int = 5,
        default_speed: int = 1000,
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
            img, ump, mot = self.node.get_latest()
            if img is not None and ump is not None and mot is not None:
                return
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for /camera/image/compressed, /ump/live, /motor/live_counts")

    def get_observation(self) -> SensapexObs:
        img, ump, mot = self.node.get_latest()
        if img is None or ump is None or mot is None:
            raise RuntimeError("Missing observation components (image/ump/motor).")

        x, y, z, d = ump
        h_ticks = mot
        state = np.array([x, y, z, d, h_ticks], dtype=np.float32)
        return SensapexObs(image_rgb=img, state=state)

    def step_absolute(self, action_5d: np.ndarray):
        """
        action_5d: [x', y', z', d', h_ticks'] (absolute target)
        """
        action_5d = np.asarray(action_5d).reshape(-1)
        if action_5d.shape != (5,):
            raise ValueError(f"Expected action shape (5,), got {action_5d.shape}")

        x, y, z, d, h = action_5d
        self.node.send_action_absolute(x, y, z, d, h, speed=self.default_speed)

    def close(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

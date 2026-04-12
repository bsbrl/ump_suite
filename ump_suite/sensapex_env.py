"""Thin ROS2 client used by the OpenPI rollout scripts (`main.py`).

`SensapexEnv` exposes a tiny synchronous interface (`get_observation`,
`step_absolute`, `close`) on top of an internal rclpy node that runs in a
background thread. The wrapper deliberately uses PIL — not OpenCV — for the
JPEG decode, because some openpi virtualenvs ship a numpy ABI that conflicts
with the system OpenCV.

The rig has **two** Sensapex micromanipulators (`/ump/*`, `/ump2/*`) plus a
single ODrive motor (`/motor/*`), so the state/action vector is 9-dim:
``[x1, y1, z1, d1, x2, y2, z2, d2, h_ticks]``.
"""

import io
import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32, Int32MultiArray


@dataclass
class SensapexObs:
    image_rgb: np.ndarray            # HxWx3 uint8 RGB
    state: np.ndarray                # shape (9,) float32:
                                     # [x1, y1, z1, d1, x2, y2, z2, d2, h_ticks]


def _decode_compressed_jpeg_to_rgb(msg: CompressedImage) -> np.ndarray:
    pil = PILImage.open(io.BytesIO(bytes(msg.data))).convert("RGB")
    return np.array(pil, dtype=np.uint8)


class _SensapexROSNode(Node):
    """Minimal subscriber/publisher node owned by `SensapexEnv`."""

    def __init__(
        self,
        *,
        save_preview: bool = True,
        preview_path: str = "sensapex_live.png",
        preview_every_n_frames: int = 5,
    ):
        super().__init__("openpi_sensapex_bridge")

        self.sub_img = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self._on_img, 10
        )
        self.sub_ump1_live = self.create_subscription(
            Int32MultiArray, "/ump/live", self._on_ump1_live, 10
        )
        self.sub_ump2_live = self.create_subscription(
            Int32MultiArray, "/ump2/live", self._on_ump2_live, 10
        )
        self.sub_motor_live = self.create_subscription(
            Int32, "/motor/live_counts", self._on_motor_live, 10
        )

        self.pub_ump1_target = self.create_publisher(Int32MultiArray, "/ump/target", 10)
        self.pub_ump2_target = self.create_publisher(Int32MultiArray, "/ump2/target", 10)
        self.pub_motor_target = self.create_publisher(Int32, "/motor/target_counts", 10)

        # Latest sensor state, guarded by `_lock` so the rollout thread can
        # snapshot it without races against the rclpy spin thread.
        self._lock = threading.Lock()
        self._latest_image_rgb = None
        self._latest_ump1 = None
        self._latest_ump2 = None
        self._latest_motor = None

        # Periodic disk-write of the live image so SSH users can preview it.
        self._save_preview = bool(save_preview)
        self._preview_path = str(preview_path)
        self._preview_every_n_frames = max(1, int(preview_every_n_frames))
        self._frame_counter = 0

    # ── Subscriber callbacks ───────────────────────────────────────────────
    def _on_img(self, msg: CompressedImage):
        try:
            rgb = _decode_compressed_jpeg_to_rgb(msg)
        except Exception as e:
            self.get_logger().warn(f"Image decode failed: {e}")
            return

        with self._lock:
            self._latest_image_rgb = rgb

        if self._save_preview:
            self._frame_counter += 1
            if self._frame_counter % self._preview_every_n_frames == 0:
                try:
                    PILImage.fromarray(rgb).save(self._preview_path)
                except Exception as e:
                    self.get_logger().warn(f"Preview save failed: {e}")

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

    def _on_motor_live(self, msg: Int32):
        with self._lock:
            self._latest_motor = int(msg.data)

    # ── Public helpers ─────────────────────────────────────────────────────
    def get_latest(self):
        with self._lock:
            img = None if self._latest_image_rgb is None else self._latest_image_rgb.copy()
            ump1 = None if self._latest_ump1 is None else list(self._latest_ump1)
            ump2 = None if self._latest_ump2 is None else list(self._latest_ump2)
            mot = self._latest_motor
        return img, ump1, ump2, mot

    def send_action_absolute(
        self, x1, y1, z1, d1, x2, y2, z2, d2, h_ticks, speed=1000
    ):
        ump1_msg = Int32MultiArray()
        ump1_msg.data = [int(x1), int(y1), int(z1), int(d1), int(speed)]
        self.pub_ump1_target.publish(ump1_msg)

        ump2_msg = Int32MultiArray()
        ump2_msg.data = [int(x2), int(y2), int(z2), int(d2), int(speed)]
        self.pub_ump2_target.publish(ump2_msg)

        self.pub_motor_target.publish(Int32(data=int(h_ticks)))


class SensapexEnv:
    """Synchronous wrapper around `_SensapexROSNode`."""

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

        # Note: openpi rollout scripts must be run with ROS Humble's python
        # (python3.10) so that rclpy is importable.
        rclpy.init(args=None)

        self.node = _SensapexROSNode(
            save_preview=save_preview,
            preview_path=preview_path,
            preview_every_n_frames=preview_every_n_frames,
        )

        self._executor_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True
        )
        self._executor_thread.start()

        self._wait_for_first_messages(timeout_s=wait_timeout_s)

    def _wait_for_first_messages(self, timeout_s=10.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            img, ump1, ump2, mot = self.node.get_latest()
            if (
                img is not None
                and ump1 is not None
                and ump2 is not None
                and mot is not None
            ):
                return
            time.sleep(0.05)
        raise RuntimeError(
            "Timed out waiting for /camera/image/compressed, /ump/live, "
            "/ump2/live, /motor/live_counts"
        )

    def get_observation(self) -> SensapexObs:
        img, ump1, ump2, mot = self.node.get_latest()
        if img is None or ump1 is None or ump2 is None or mot is None:
            raise RuntimeError(
                "Missing observation components (image/ump1/ump2/motor)."
            )

        x1, y1, z1, d1 = ump1
        x2, y2, z2, d2 = ump2
        state = np.array(
            [x1, y1, z1, d1, x2, y2, z2, d2, mot], dtype=np.float32
        )
        return SensapexObs(image_rgb=img, state=state)

    def step_absolute(self, action_9d: np.ndarray):
        """Send an absolute target [x1, y1, z1, d1, x2, y2, z2, d2, h_ticks]."""
        action_9d = np.asarray(action_9d).reshape(-1)
        if action_9d.shape != (9,):
            raise ValueError(f"Expected action shape (9,), got {action_9d.shape}")

        x1, y1, z1, d1, x2, y2, z2, d2, h = action_9d
        self.node.send_action_absolute(
            x1, y1, z1, d1, x2, y2, z2, d2, h, speed=self.default_speed
        )

    def close(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

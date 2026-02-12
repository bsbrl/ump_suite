import threading
import tkinter as tk
from tkinter import ttk
from tkinter import IntVar, StringVar, N, S, E, W

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Int32
from std_srvs.srv import Trigger
from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np

try:
    from PIL import Image as PILImage
    from PIL import ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

from .ros_interfaces import (
    TOPIC_UMP_TARGET, TOPIC_UMP_LIVE,
    TOPIC_MOTOR_TGT, TOPIC_MOTOR_JOG, TOPIC_MOTOR_LIVE,
    SRV_ACQ_START, SRV_ACQ_STOP, SRV_ZERO
)

TOPIC_CAM_IMAGE_COMPRESSED = "/camera/image/compressed"

AXIS_MIN, AXIS_MAX = -10000, 10000
SPEED_MIN, SPEED_MAX = 10, 2000
MOTOR_MIN, MOTOR_MAX = -1_000_000, 1_000_000

DEFAULT_AXIS_STEP = 100
DEFAULT_SPEED_STEP = 50

LIVE_POLL_MS = 50
SEND_THROTTLE_MS = 60
CAM_UPDATE_MS = 30

CAM_TEXT = "Blackfly S Live"


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


class GuiNode(Node):
    def __init__(self):
        super().__init__("gui_node")

        self.pub_ump_target = self.create_publisher(Int32MultiArray, TOPIC_UMP_TARGET, 10)
        self.pub_motor_tgt  = self.create_publisher(Int32, TOPIC_MOTOR_TGT, 10)
        self.pub_motor_jog  = self.create_publisher(Int32, TOPIC_MOTOR_JOG, 10)

        self.sub_ump_live = self.create_subscription(Int32MultiArray, TOPIC_UMP_LIVE, self._on_ump_live, 10)
        self.sub_motor_live = self.create_subscription(Int32, TOPIC_MOTOR_LIVE, self._on_motor_live, 10)

        # Compressed image subscription (no cv_bridge)
        self.sub_cam = self.create_subscription(CompressedImage, TOPIC_CAM_IMAGE_COMPRESSED, self._on_cam_image, 10)

        self.cli_acq_start = self.create_client(Trigger, SRV_ACQ_START)
        self.cli_acq_stop  = self.create_client(Trigger, SRV_ACQ_STOP)
        self.cli_zero      = self.create_client(Trigger, SRV_ZERO)

        self.latest_live_ump = [0, 0, 0, 0]
        self.latest_live_motor = 0

        self.latest_frame_bgr = None

    def _on_ump_live(self, msg: Int32MultiArray):
        if len(msg.data) >= 4:
            self.latest_live_ump = [int(v) for v in msg.data[:4]]

    def _on_motor_live(self, msg: Int32):
        self.latest_live_motor = int(msg.data)

    def _on_cam_image(self, msg: CompressedImage):
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR
            if frame is not None:
                self.latest_frame_bgr = frame
        except Exception:
            pass

    def call_trigger(self, client):
        if not client.wait_for_service(timeout_sec=1.0):
            return False, "service not available"
        req = Trigger.Request()
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        if fut.result() is None:
            return False, "no response"
        return bool(fut.result().success), str(fut.result().message)


class UMPGuiApp:
    def __init__(self, node: GuiNode):
        self.node = node

        self.root = tk.Tk()
        self.root.title("Sensapex UMP + Blackfly S + ODrive (ROS2)")
        self.root.geometry("1100x700")

        self.axis_step = IntVar(value=DEFAULT_AXIS_STEP)
        self.speed_step = IntVar(value=DEFAULT_SPEED_STEP)

        self.x = IntVar(value=0)
        self.y = IntVar(value=0)
        self.z = IntVar(value=0)
        self.d = IntVar(value=0)
        self.speed = IntVar(value=1000)

        self.motor_target = IntVar(value=0)

        self.live_x = StringVar(value="—")
        self.live_y = StringVar(value="—")
        self.live_z = StringVar(value="—")
        self.live_d = StringVar(value="—")
        self.live_motor = StringVar(value="—")

        self.status = StringVar(value="Ready")
        self.acq_status = StringVar(value="Data acquisition: STOPPED")

        self._send_job_id = None
        self._tkimg = None

        self._build_ui()
        self._bind_keys()

        self.root.after(LIVE_POLL_MS, self._poll_live_to_gui)
        self.root.after(CAM_UPDATE_MS, self._update_camera_view)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.left = ttk.Frame(self.root, padding=12)
        self.left.grid(row=0, column=0, sticky=(N, S, E, W))

        self.right = ttk.Frame(self.root, padding=6)
        self.right.grid(row=0, column=1, sticky=(N, S, E, W))
        self.right.rowconfigure(1, weight=1)
        self.right.columnconfigure(0, weight=1)

        ttk.Label(self.left, text="Sensapex UMP Controller + ODrive", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, columnspan=4, sticky=W, pady=(0, 8)
        )

        ttk.Label(self.left, text="Axis step:").grid(row=1, column=0, sticky=W)
        ttk.Entry(self.left, width=7, textvariable=self.axis_step, justify="right").grid(row=1, column=1, sticky=W, padx=(6, 12))

        ttk.Label(self.left, text="Speed step:").grid(row=1, column=2, sticky=W)
        ttk.Entry(self.left, width=7, textvariable=self.speed_step, justify="right").grid(row=1, column=3, sticky=W)

        ttk.Label(self.left, text="Target (centered / counts)", font=("Segoe UI", 10, "bold")).grid(row=2, column=3, sticky=W)
        ttk.Label(self.left, text="Live (centered / counts)", font=("Segoe UI", 10, "bold")).grid(row=2, column=4, sticky=W)

        self._make_axis_row(self.left, 3, "X", self.x, self.live_x, AXIS_MIN, AXIS_MAX)
        self._make_axis_row(self.left, 4, "Y", self.y, self.live_y, AXIS_MIN, AXIS_MAX)
        self._make_axis_row(self.left, 5, "Z", self.z, self.live_z, AXIS_MIN, AXIS_MAX)
        self._make_axis_row(self.left, 6, "D", self.d, self.live_d, AXIS_MIN, AXIS_MAX)

        self._make_speed_row(self.left, 7)
        self._make_motor_row(self.left, 8)

        ttk.Button(self.left, text="Start Data Acquisition", command=self._acq_start).grid(row=9, column=0, sticky=W, pady=(10, 4))
        ttk.Button(self.left, text="Stop Data Acquisition", command=self._acq_stop).grid(row=9, column=1, sticky=W, pady=(10, 4))
        ttk.Label(self.left, textvariable=self.acq_status, foreground="#006400", font=("Segoe UI", 10, "bold")).grid(
            row=9, column=2, columnspan=2, sticky=W, pady=(10, 4)
        )

        ttk.Button(self.left, text="Send Now", command=self._send_now).grid(row=10, column=0, sticky=W, pady=(10, 4))
        ttk.Button(self.left, text="Home (0,0,0,0)", command=self._home).grid(row=10, column=1, sticky=W, pady=(10, 4))
        ttk.Button(self.left, text="Sync Target to Live", command=self._sync_targets_to_live).grid(row=10, column=2, sticky=W, pady=(10, 4))
        ttk.Button(self.left, text="Calibrate Zero Here", command=self._zero).grid(row=10, column=3, sticky=W, pady=(10, 4))

        ttk.Label(self.left, textvariable=self.status, foreground="#333").grid(row=12, column=0, columnspan=4, sticky=W, pady=(8, 0))

        ttk.Label(self.right, text=CAM_TEXT, font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky=W, pady=(0, 6))
        self.cam_label = ttk.Label(self.right)
        self.cam_label.grid(row=1, column=0, sticky=(N, S, E, W))

    def _make_axis_row(self, parent, row, name, var, live_var, vmin, vmax):
        ttk.Label(parent, text=name, width=8).grid(row=row, column=0, padx=(0, 6), sticky=W)
        ttk.Button(parent, text="▲", command=lambda: self._bump_axis(var, +1, vmin, vmax)).grid(row=row, column=1, padx=2, sticky=W)
        ttk.Button(parent, text="▼", command=lambda: self._bump_axis(var, -1, vmin, vmax)).grid(row=row, column=2, padx=2, sticky=W)

        entry = ttk.Entry(parent, width=10, textvariable=var, justify="right")
        entry.grid(row=row, column=3, padx=(6, 6), sticky=W)
        ttk.Label(parent, textvariable=live_var, width=12, anchor="e").grid(row=row, column=4, padx=(12, 0), sticky=W)

        def on_commit(_e=None):
            try:
                v = int(var.get())
            except Exception:
                v = 0
            var.set(clamp(v, vmin, vmax))
            self._schedule_send()

        entry.bind("<FocusOut>", on_commit)
        entry.bind("<Return>", on_commit)

    def _make_speed_row(self, parent, row):
        ttk.Label(parent, text="Speed", width=8).grid(row=row, column=0, padx=(0, 6), sticky=W)
        ttk.Button(parent, text="▲", command=lambda: self._bump_speed(+1)).grid(row=row, column=1, padx=2, sticky=W)
        ttk.Button(parent, text="▼", command=lambda: self._bump_speed(-1)).grid(row=row, column=2, padx=2, sticky=W)

        entry = ttk.Entry(parent, width=10, textvariable=self.speed, justify="right")
        entry.grid(row=row, column=3, padx=(6, 6), sticky=W)

        def on_commit(_e=None):
            try:
                v = int(self.speed.get())
            except Exception:
                v = 1000
            self.speed.set(clamp(v, SPEED_MIN, SPEED_MAX))
            self._schedule_send()

        entry.bind("<FocusOut>", on_commit)
        entry.bind("<Return>", on_commit)

    def _make_motor_row(self, parent, row):
        ttk.Label(parent, text="Motor", width=8).grid(row=row, column=0, padx=(0, 6), sticky=W)

        btn_up = ttk.Button(parent, text="▲")
        btn_dn = ttk.Button(parent, text="▼")
        btn_up.grid(row=row, column=1, padx=2, sticky=W)
        btn_dn.grid(row=row, column=2, padx=2, sticky=W)

        btn_up.bind("<ButtonPress-1>", lambda _e: self._motor_jog(+1))
        btn_up.bind("<ButtonRelease-1>", lambda _e: self._motor_jog(0))
        btn_dn.bind("<ButtonPress-1>", lambda _e: self._motor_jog(-1))
        btn_dn.bind("<ButtonRelease-1>", lambda _e: self._motor_jog(0))

        entry = ttk.Entry(parent, width=10, textvariable=self.motor_target, justify="right")
        entry.grid(row=row, column=3, padx=(6, 6), sticky=W)
        ttk.Label(parent, textvariable=self.live_motor, width=12, anchor="e").grid(row=row, column=4, padx=(12, 0), sticky=W)

        def on_commit(_e=None):
            try:
                v = int(self.motor_target.get())
            except Exception:
                v = 0
            self.motor_target.set(clamp(v, MOTOR_MIN, MOTOR_MAX))
            self.node.pub_motor_tgt.publish(Int32(data=int(self.motor_target.get())))

        entry.bind("<FocusOut>", on_commit)
        entry.bind("<Return>", on_commit)

    def _bind_keys(self):
        self.root.bind_all("<KeyPress>", self._on_key_press)

    def _on_key_press(self, event):
        ks = (event.keysym or "").lower()
        mapping = {
            "up": ("Z", +1), "down": ("Z", -1),
            "a": ("X", -1), "d": ("X", +1),
            "s": ("Y", -1), "w": ("Y", +1),
            "less": ("D", -1), "comma": ("D", -1),
            "greater": ("D", +1), "period": ("D", +1),
            "left": ("Speed", -1), "right": ("Speed", +1),
        }
        if ks not in mapping:
            return
        name, delta = mapping[ks]
        if name == "Speed":
            self._bump_speed(delta)
        else:
            var = {"X": self.x, "Y": self.y, "Z": self.z, "D": self.d}[name]
            self._bump_axis(var, delta, AXIS_MIN, AXIS_MAX)

    def _bump_axis(self, var, delta, vmin, vmax):
        step = int(self.axis_step.get())
        var.set(clamp(int(var.get()) + delta * step, vmin, vmax))
        self._schedule_send()

    def _bump_speed(self, delta):
        step = int(self.speed_step.get())
        self.speed.set(clamp(int(self.speed.get()) + delta * step, SPEED_MIN, SPEED_MAX))
        self._schedule_send()

    def _schedule_send(self):
        if self._send_job_id is not None:
            return
        self._send_job_id = self.root.after(SEND_THROTTLE_MS, self._send_now)

    def _send_now(self):
        self._send_job_id = None
        x, y, z, d = int(self.x.get()), int(self.y.get()), int(self.z.get()), int(self.d.get())
        speed = clamp(int(self.speed.get()), SPEED_MIN, SPEED_MAX)
        self.speed.set(speed)

        msg = Int32MultiArray()
        msg.data = [x, y, z, d, speed]
        self.node.pub_ump_target.publish(msg)

        mt = clamp(int(self.motor_target.get()), MOTOR_MIN, MOTOR_MAX)
        self.motor_target.set(mt)
        self.node.pub_motor_tgt.publish(Int32(data=mt))

        self.status.set(f"Sent: X={x}, Y={y}, Z={z}, D={d} @ {speed}")

    def _home(self):
        self.x.set(0); self.y.set(0); self.z.set(0); self.d.set(0)
        self._send_now()

    def _sync_targets_to_live(self):
        lx, ly, lz, ld = self.node.latest_live_ump
        self.x.set(lx); self.y.set(ly); self.z.set(lz); self.d.set(ld)
        self.motor_target.set(self.node.latest_live_motor)
        self.status.set("Targets synced to live.")

    def _zero(self):
        ok, msg = self.node.call_trigger(self.node.cli_zero)
        self.status.set(f"Zero: {ok} ({msg})")

    def _acq_start(self):
        ok, msg = self.node.call_trigger(self.node.cli_acq_start)
        self.acq_status.set("Data acquisition: RUNNING" if ok else "Data acquisition: START FAILED")
        self.status.set(f"Acq start: {ok} ({msg})")

    def _acq_stop(self):
        ok, msg = self.node.call_trigger(self.node.cli_acq_stop)
        self.acq_status.set("Data acquisition: STOPPED")
        self.status.set(f"Acq stop: {ok} ({msg})")

    def _motor_jog(self, direction):
        self.node.pub_motor_jog.publish(Int32(data=int(direction)))
        if int(direction) == 0:
            self.motor_target.set(self.node.latest_live_motor)

    def _poll_live_to_gui(self):
        lx, ly, lz, ld = self.node.latest_live_ump
        self.live_x.set(f"{lx:d}")
        self.live_y.set(f"{ly:d}")
        self.live_z.set(f"{lz:d}")
        self.live_d.set(f"{ld:d}")
        self.live_motor.set(f"{int(self.node.latest_live_motor):d}")
        self.root.after(LIVE_POLL_MS, self._poll_live_to_gui)

    def _update_camera_view(self):
        frame = self.node.latest_frame_bgr
        if frame is not None and PIL_AVAILABLE:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PILImage.fromarray(frame_rgb)

            w = self.right.winfo_width() or 640
            h = self.right.winfo_height() or 480
            h = max(50, h - 30)

            img = img.copy()
            img.thumbnail((w, h))
            self._tkimg = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=self._tkimg)

        self.root.after(CAM_UPDATE_MS, self._update_camera_view)

    def _on_close(self):
        try:
            self.node.pub_motor_jog.publish(Int32(data=0))
        except Exception:
            pass
        self.root.destroy()


def main():
    rclpy.init()
    node = GuiNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    app = UMPGuiApp(node)
    app.root.mainloop()

    node.destroy_node()
    rclpy.shutdown()
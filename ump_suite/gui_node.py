"""Tk-based control panel for both Sensapex UMPs, the ODrive motor and the camera.

The window is split in two columns:
  * left  – two UMP control panels, the motor row, and the data-acquisition controls
  * right – the live camera preview

Two `_UmpPanel` instances handle UMP1 and UMP2; they encapsulate the per-stage
Tk variables, publishers, and the small "send / bump / home / zero" actions.
Keyboard shortcuts (WASD, arrows, comma/period) drive UMP1 only.

All commands are *absolute* targets. The arrow buttons and keys nudge the
locally-stored target value by `axis_step` and then publish the full target.
"""

import threading
import tkinter as tk
from tkinter import IntVar, StringVar, ttk
from tkinter.constants import E, N, S, W

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32, Int32MultiArray
from std_srvs.srv import Trigger

from .ros_interfaces import (
    SRV_ACQ_START,
    SRV_ACQ_STOP,
    SRV_ZERO,
    SRV_ZERO2,
    TOPIC_CAM_IMAGE_COMPRESSED,
    TOPIC_MOTOR_LIVE,
    TOPIC_MOTOR_TGT,
    TOPIC_SOL1_CMD,
    TOPIC_SOL1_STATE,
    TOPIC_SOL2_CMD,
    TOPIC_SOL2_STATE,
    TOPIC_UMP_LIVE,
    TOPIC_UMP_TARGET,
    TOPIC_UMP2_LIVE,
    TOPIC_UMP2_TARGET,
)


try:
    from PIL import Image as PILImage
    from PIL import ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# ── Limits & defaults ──────────────────────────────────────────────────────
AXIS_MIN, AXIS_MAX = -10000, 10000
SPEED_MIN, SPEED_MAX = 10, 2000
MOTOR_MIN, MOTOR_MAX = -1_000_000, 1_000_000

DEFAULT_AXIS_STEP = 50
DEFAULT_SPEED = 1000
DEFAULT_MOTOR_STEP = 500

LIVE_POLL_MS = 50
SEND_THROTTLE_MS = 60
CAM_UPDATE_MS = 30

CAM_TEXT = "Blackfly S Live"


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


# ─────────────────────────────────────────────────────────────────────────────
# ROS node
# ─────────────────────────────────────────────────────────────────────────────
class GuiNode(Node):
    """All ROS publishers/subscribers/clients used by the GUI."""

    def __init__(self):
        super().__init__("gui_node")

        # Absolute target publishers (one per movable axis group).
        self.pub_ump_target  = self.create_publisher(Int32MultiArray, TOPIC_UMP_TARGET,  10)
        self.pub_ump2_target = self.create_publisher(Int32MultiArray, TOPIC_UMP2_TARGET, 10)
        self.pub_motor_tgt   = self.create_publisher(Int32,           TOPIC_MOTOR_TGT,   10)
        self.pub_sol1_cmd    = self.create_publisher(Bool,            TOPIC_SOL1_CMD,    10)
        self.pub_sol2_cmd    = self.create_publisher(Bool,            TOPIC_SOL2_CMD,    10)

        # Live state subscribers.
        self.create_subscription(Int32MultiArray, TOPIC_UMP_LIVE,   self._on_ump_live,   10)
        self.create_subscription(Int32MultiArray, TOPIC_UMP2_LIVE,  self._on_ump2_live,  10)
        self.create_subscription(Int32,           TOPIC_MOTOR_LIVE, self._on_motor_live, 10)
        self.create_subscription(CompressedImage, TOPIC_CAM_IMAGE_COMPRESSED, self._on_cam_image, 10)
        self.create_subscription(Bool, TOPIC_SOL1_STATE, self._on_sol1_state, 10)
        self.create_subscription(Bool, TOPIC_SOL2_STATE, self._on_sol2_state, 10)

        # Service clients.
        self.cli_acq_start = self.create_client(Trigger, SRV_ACQ_START)
        self.cli_acq_stop  = self.create_client(Trigger, SRV_ACQ_STOP)
        self.cli_zero      = self.create_client(Trigger, SRV_ZERO)
        self.cli_zero2     = self.create_client(Trigger, SRV_ZERO2)

        # Latest values polled by the Tk main loop.
        self.latest_live_ump = [0, 0, 0, 0]
        self.latest_live_ump2 = [0, 0, 0, 0]
        self.latest_live_motor = 0
        self.latest_frame_bgr = None
        self.latest_sol1_state = False
        self.latest_sol2_state = False

    # ── Subscriber callbacks ───────────────────────────────────────────────
    def _on_ump_live(self, msg: Int32MultiArray):
        if len(msg.data) >= 4:
            self.latest_live_ump = [int(v) for v in msg.data[:4]]

    def _on_ump2_live(self, msg: Int32MultiArray):
        if len(msg.data) >= 4:
            self.latest_live_ump2 = [int(v) for v in msg.data[:4]]

    def _on_motor_live(self, msg: Int32):
        self.latest_live_motor = int(msg.data)

    def _on_sol1_state(self, msg: Bool):
        self.latest_sol1_state = bool(msg.data)

    def _on_sol2_state(self, msg: Bool):
        self.latest_sol2_state = bool(msg.data)

    def _on_cam_image(self, msg: CompressedImage):
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is not None:
                self.latest_frame_bgr = frame
        except Exception:
            pass

    # ── Sync service call helper ───────────────────────────────────────────
    def call_trigger(self, client):
        if not client.wait_for_service(timeout_sec=1.0):
            return False, "service not available"
        fut = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        if fut.result() is None:
            return False, "no response"
        return bool(fut.result().success), str(fut.result().message)


# ─────────────────────────────────────────────────────────────────────────────
# Per-UMP panel
# ─────────────────────────────────────────────────────────────────────────────
class _UmpPanel:
    """State + actions for one Sensapex UMP (UMP1 or UMP2).

    Owns its own Tk variables and the publisher it talks to. The `app`
    reference is only used to update the shared status text and to schedule
    throttled sends on the Tk root.
    """

    def __init__(
        self,
        app,
        *,
        label,
        pub_target,
        zero_client,
        live_getter,
    ):
        self.app = app
        self.label = label
        self.pub_target = pub_target
        self.zero_client = zero_client
        self._live_getter = live_getter  # () -> [x, y, z, d]

        self.axis_step = IntVar(value=DEFAULT_AXIS_STEP)
        self.speed = IntVar(value=DEFAULT_SPEED)

        self.x = IntVar(value=0)
        self.y = IntVar(value=0)
        self.z = IntVar(value=0)
        self.d = IntVar(value=0)

        self.live_x = StringVar(value="—")
        self.live_y = StringVar(value="—")
        self.live_z = StringVar(value="—")
        self.live_d = StringVar(value="—")

        self._send_job_id = None

    # Order matches the on-screen rows. Used by the layout helpers.
    def axis_iter(self):
        return (
            ("X", self.x, self.live_x),
            ("Y", self.y, self.live_y),
            ("Z", self.z, self.live_z),
            ("D", self.d, self.live_d),
        )

    def _resolved_speed(self):
        """Clamp the speed entry, push it back to the var, and return it."""
        speed = clamp(int(self.speed.get()), SPEED_MIN, SPEED_MAX)
        self.speed.set(speed)
        return speed

    def _axis_var(self, axis_name):
        return {"X": self.x, "Y": self.y, "Z": self.z, "D": self.d}[axis_name]

    def bump_axis(self, axis_name, sign):
        """Increment the local target by ±step on one axis, then publish."""
        var = self._axis_var(axis_name)
        new_val = clamp(int(var.get()) + sign * int(self.axis_step.get()), AXIS_MIN, AXIS_MAX)
        var.set(new_val)
        self.send_now()

    def send_now(self):
        self._send_job_id = None
        x, y, z, d = (int(v.get()) for v in (self.x, self.y, self.z, self.d))
        speed = self._resolved_speed()

        msg = Int32MultiArray()
        msg.data = [x, y, z, d, speed]
        self.pub_target.publish(msg)
        self.app.status.set(f"{self.label} target: X={x}, Y={y}, Z={z}, D={d} @ {speed}")

    def schedule_send(self):
        # Coalesce rapid entry-box edits into one publish per SEND_THROTTLE_MS.
        if self._send_job_id is None:
            self._send_job_id = self.app.root.after(SEND_THROTTLE_MS, self.send_now)

    def home(self):
        for v in (self.x, self.y, self.z, self.d):
            v.set(0)
        self.send_now()

    def sync_to_live(self):
        lx, ly, lz, ld = self._live_getter()
        self.x.set(lx); self.y.set(ly); self.z.set(lz); self.d.set(ld)
        self.app.status.set(f"{self.label} targets synced to live.")

    def update_live_display(self):
        lx, ly, lz, ld = self._live_getter()
        self.live_x.set(f"{lx:d}")
        self.live_y.set(f"{ly:d}")
        self.live_z.set(f"{lz:d}")
        self.live_d.set(f"{ld:d}")

    def calibrate_zero(self):
        ok, msg = self.app.node.call_trigger(self.zero_client)
        self.app.status.set(f"{self.label} Zero: {ok} ({msg})")


# ─────────────────────────────────────────────────────────────────────────────
# Tk application
# ─────────────────────────────────────────────────────────────────────────────
class UMPGuiApp:
    def __init__(self, node: GuiNode):
        self.node = node

        self.root = tk.Tk()
        self.root.title("Sensapex UMP1 + UMP2 + Blackfly S + ODrive (ROS2)")
        self.root.geometry("1200x950")

        self.status = StringVar(value="Ready")
        self.acq_status = StringVar(value="Data acquisition: STOPPED")

        self.panel1 = _UmpPanel(
            self,
            label="UMP1",
            pub_target=node.pub_ump_target,
            zero_client=node.cli_zero,
            live_getter=lambda: node.latest_live_ump,
        )
        self.panel2 = _UmpPanel(
            self,
            label="UMP2",
            pub_target=node.pub_ump2_target,
            zero_client=node.cli_zero2,
            live_getter=lambda: node.latest_live_ump2,
        )

        # Motor / shared widgets.
        self.motor_step = IntVar(value=DEFAULT_MOTOR_STEP)
        self.motor_target = IntVar(value=0)
        self.live_motor = StringVar(value="—")

        self.sol1_state_text = StringVar(value="S1: —")
        self.sol2_state_text = StringVar(value="S2: —")

        self._tkimg = None  # keep a ref so Tk doesn't garbage-collect the image

        self._build_ui()
        self._bind_keys()

        self.root.after(LIVE_POLL_MS, self._poll_live_to_gui)
        self.root.after(CAM_UPDATE_MS, self._update_camera_view)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Layout ─────────────────────────────────────────────────────────────
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

        ttk.Label(
            self.left,
            text="Sensapex UMP1 + UMP2 + ODrive Controller",
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, sticky=W, pady=(0, 8))
        ttk.Label(
            self.left,
            text="Buttons/keys nudge the target by 'Axis step'. All commands are absolute.",
            foreground="#555",
        ).grid(row=1, column=0, sticky=W, pady=(0, 6))

        self._build_ump_frame(
            row=2, panel=self.panel1, title="UMP 1  (keys: WASD / arrows / <> )"
        )
        self._build_ump_frame(row=3, panel=self.panel2, title="UMP 2")
        self._build_motor_frame(row=4)
        self._build_pressure_frame(row=5)

        ttk.Button(self.left, text="Start Data Acquisition", command=self._acq_start).grid(
            row=6, column=0, sticky=W, pady=(4, 2)
        )
        ttk.Button(self.left, text="Stop Data Acquisition", command=self._acq_stop).grid(
            row=7, column=0, sticky=W, pady=(2, 2)
        )
        ttk.Label(
            self.left,
            textvariable=self.acq_status,
            foreground="#006400",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=8, column=0, sticky=W, pady=(2, 4))

        ttk.Label(self.left, textvariable=self.status, foreground="#333").grid(
            row=9, column=0, sticky=W, pady=(4, 0)
        )

        # Camera preview on the right.
        ttk.Label(self.right, text=CAM_TEXT, font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky=W, pady=(0, 6)
        )
        self.cam_label = ttk.Label(self.right)
        self.cam_label.grid(row=1, column=0, sticky=(N, S, E, W))

    def _build_ump_frame(self, row, panel: _UmpPanel, title: str):
        frame = ttk.LabelFrame(self.left, text=title, padding=8)
        frame.grid(row=row, column=0, sticky=(N, S, E, W), pady=(0, 8))

        ttk.Label(frame, text="Axis step:").grid(row=0, column=0, sticky=W)
        ttk.Entry(frame, width=7, textvariable=panel.axis_step, justify="right").grid(
            row=0, column=1, sticky=W, padx=(4, 12)
        )
        ttk.Label(frame, text="Speed:").grid(row=0, column=2, sticky=W)
        ttk.Entry(frame, width=7, textvariable=panel.speed, justify="right").grid(
            row=0, column=3, sticky=W, padx=(4, 0)
        )

        ttk.Label(frame, text="Target (absolute)", font=("Segoe UI", 9, "bold")).grid(
            row=1, column=3, sticky=W, pady=(4, 0)
        )
        ttk.Label(frame, text="Live", font=("Segoe UI", 9, "bold")).grid(
            row=1, column=4, sticky=W, pady=(4, 0)
        )

        for i, (axis_name, var, live_var) in enumerate(panel.axis_iter()):
            self._build_axis_row(frame, row=2 + i, panel=panel, axis_name=axis_name,
                                 var=var, live_var=live_var)

        ttk.Button(frame, text="Send Now", command=panel.send_now).grid(
            row=6, column=0, sticky=W, pady=(8, 0)
        )
        ttk.Button(frame, text="Home (0,0,0,0)", command=panel.home).grid(
            row=6, column=1, sticky=W, pady=(8, 0)
        )
        ttk.Button(frame, text="Sync to Live", command=panel.sync_to_live).grid(
            row=6, column=2, sticky=W, pady=(8, 0)
        )
        ttk.Button(frame, text="Calibrate Zero", command=panel.calibrate_zero).grid(
            row=6, column=3, sticky=W, pady=(8, 0)
        )

    def _build_axis_row(self, parent, *, row, panel, axis_name, var, live_var):
        ttk.Label(parent, text=axis_name, width=8).grid(row=row, column=0, padx=(0, 6), sticky=W)
        ttk.Button(parent, text="▲", command=lambda: panel.bump_axis(axis_name, +1)).grid(
            row=row, column=1, padx=2, sticky=W
        )
        ttk.Button(parent, text="▼", command=lambda: panel.bump_axis(axis_name, -1)).grid(
            row=row, column=2, padx=2, sticky=W
        )

        entry = ttk.Entry(parent, width=10, textvariable=var, justify="right")
        entry.grid(row=row, column=3, padx=(6, 6), sticky=W)
        ttk.Label(parent, textvariable=live_var, width=12, anchor="e").grid(
            row=row, column=4, padx=(12, 0), sticky=W
        )

        def on_commit(_e=None):
            try:
                v = int(var.get())
            except Exception:
                v = 0
            var.set(clamp(v, AXIS_MIN, AXIS_MAX))
            panel.schedule_send()

        entry.bind("<FocusOut>", on_commit)
        entry.bind("<Return>", on_commit)

    def _build_motor_frame(self, row):
        frame = ttk.LabelFrame(self.left, text="Motor (ODrive)", padding=8)
        frame.grid(row=row, column=0, sticky=(N, S, E, W), pady=(0, 8))

        ttk.Label(frame, text="Motor step:").grid(row=0, column=0, sticky=W)
        ttk.Entry(frame, width=8, textvariable=self.motor_step, justify="right").grid(
            row=0, column=1, sticky=W, padx=(4, 0)
        )

        ttk.Label(frame, text="Motor", width=8).grid(row=1, column=0, padx=(0, 6), sticky=W)
        ttk.Button(frame, text="▲", command=lambda: self._bump_motor(+1)).grid(
            row=1, column=1, padx=2, sticky=W
        )
        ttk.Button(frame, text="▼", command=lambda: self._bump_motor(-1)).grid(
            row=1, column=2, padx=2, sticky=W
        )

        entry = ttk.Entry(frame, width=10, textvariable=self.motor_target, justify="right")
        entry.grid(row=1, column=3, padx=(6, 6), sticky=W)
        ttk.Label(frame, textvariable=self.live_motor, width=12, anchor="e").grid(
            row=1, column=4, padx=(12, 0), sticky=W
        )

        def on_commit(_e=None):
            try:
                v = int(self.motor_target.get())
            except Exception:
                v = 0
            self.motor_target.set(clamp(v, MOTOR_MIN, MOTOR_MAX))
            self._publish_motor_target()

        entry.bind("<FocusOut>", on_commit)
        entry.bind("<Return>", on_commit)

    def _build_pressure_frame(self, row):
        frame = ttk.LabelFrame(self.left, text="Pressure (Solenoids)", padding=8)
        frame.grid(row=row, column=0, sticky=(N, S, E, W), pady=(0, 8))

        ttk.Label(frame, text="Solenoid 1", width=10).grid(row=0, column=0, sticky=W)
        ttk.Button(frame, text="ON",  command=lambda: self._set_solenoid(1, True)).grid(
            row=0, column=1, padx=2, sticky=W
        )
        ttk.Button(frame, text="OFF", command=lambda: self._set_solenoid(1, False)).grid(
            row=0, column=2, padx=2, sticky=W
        )
        ttk.Label(frame, textvariable=self.sol1_state_text, width=10, anchor="w").grid(
            row=0, column=3, padx=(12, 0), sticky=W
        )

        ttk.Label(frame, text="Solenoid 2", width=10).grid(row=1, column=0, sticky=W, pady=(4, 0))
        ttk.Button(frame, text="ON",  command=lambda: self._set_solenoid(2, True)).grid(
            row=1, column=1, padx=2, pady=(4, 0), sticky=W
        )
        ttk.Button(frame, text="OFF", command=lambda: self._set_solenoid(2, False)).grid(
            row=1, column=2, padx=2, pady=(4, 0), sticky=W
        )
        ttk.Label(frame, textvariable=self.sol2_state_text, width=10, anchor="w").grid(
            row=1, column=3, padx=(12, 0), pady=(4, 0), sticky=W
        )

    def _set_solenoid(self, which, on):
        pub = self.node.pub_sol1_cmd if which == 1 else self.node.pub_sol2_cmd
        pub.publish(Bool(data=bool(on)))
        self.status.set(f"Solenoid {which}: {'ON' if on else 'OFF'}")

    # ── Motor button actions ──────────────────────────────────────────────
    def _publish_motor_target(self):
        target = int(self.motor_target.get())
        self.node.pub_motor_tgt.publish(Int32(data=target))
        self.status.set(f"Motor target: {target}")

    def _bump_motor(self, sign):
        new_val = clamp(
            int(self.motor_target.get()) + sign * int(self.motor_step.get()),
            MOTOR_MIN, MOTOR_MAX,
        )
        self.motor_target.set(new_val)
        self._publish_motor_target()

    # ── Keyboard shortcuts (UMP1 only) ─────────────────────────────────────
    def _bind_keys(self):
        self.root.bind_all("<KeyPress>", self._on_key_press)

    _KEY_BINDINGS = {
        "up":      ("Z", +1),
        "down":    ("Z", -1),
        "a":       ("X", -1),
        "d":       ("X", +1),
        "s":       ("Y", -1),
        "w":       ("Y", +1),
        "less":    ("D", -1),
        "comma":   ("D", -1),
        "greater": ("D", +1),
        "period":  ("D", +1),
    }

    def _on_key_press(self, event):
        ks = (event.keysym or "").lower()
        binding = self._KEY_BINDINGS.get(ks)
        if binding is not None:
            axis, sign = binding
            self.panel1.bump_axis(axis, sign)

    # ── Acquisition service buttons ────────────────────────────────────────
    def _acq_start(self):
        ok, msg = self.node.call_trigger(self.node.cli_acq_start)
        self.acq_status.set(
            "Data acquisition: RUNNING" if ok else "Data acquisition: START FAILED"
        )
        self.status.set(f"Acq start: {ok} ({msg})")

    def _acq_stop(self):
        ok, msg = self.node.call_trigger(self.node.cli_acq_stop)
        self.acq_status.set("Data acquisition: STOPPED")
        self.status.set(f"Acq stop: {ok} ({msg})")

    # ── Periodic Tk callbacks ──────────────────────────────────────────────
    def _poll_live_to_gui(self):
        self.panel1.update_live_display()
        self.panel2.update_live_display()
        self.live_motor.set(f"{int(self.node.latest_live_motor):d}")
        self.sol1_state_text.set(f"S1: {'ON' if self.node.latest_sol1_state else 'OFF'}")
        self.sol2_state_text.set(f"S2: {'ON' if self.node.latest_sol2_state else 'OFF'}")
        self.root.after(LIVE_POLL_MS, self._poll_live_to_gui)

    def _update_camera_view(self):
        frame = self.node.latest_frame_bgr
        if frame is not None and PIL_AVAILABLE:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PILImage.fromarray(frame_rgb)

            w = self.right.winfo_width() or 640
            h = max(50, (self.right.winfo_height() or 480) - 30)

            img = img.copy()
            img.thumbnail((w, h))
            self._tkimg = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=self._tkimg)

        self.root.after(CAM_UPDATE_MS, self._update_camera_view)

    def _on_close(self):
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

"""
Modern Qt control panel for the UMP suite.

The ROS surface is:
  * publishes absolute UMP1 / UMP2 targets
  * publishes ODrive motor targets
  * publishes the exact commanded pressure in mbar
  * subscribes to live robot, pressure, camera, and HEKA voltage/current topics
  * calls acquisition and UMP zeroing services
  * publishes the injection settings and calls the injection service

The injection settings are saved in ~/.config/ump_suite/gui.ini, so they are
still there after a restart.

The panel is mouse-only; there are deliberately no keyboard shortcuts.

Only the presentation layer changes. PyQt5 is used because it is available in
the ROS environment on this machine and gives the app a more polished desktop
feel without requiring a separate web server.
"""

import json
import math
import os
import sys
import threading
import time
import uuid
from collections import deque

import cv2
import numpy as np
import rclpy
from PyQt5.QtCore import QSettings, Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, Float32MultiArray, Int32, Int32MultiArray
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .injection import (
    AXES as INJECT_AXES,
    DEFAULT_AXIS as DEFAULT_INJECT_AXIS,
    DEFAULT_DURATION_MS,
    DEFAULT_PRESSURE_MBAR as DEFAULT_INJECT_PRESSURE_MBAR,
    DEFAULT_SPEED_UM_S,
    DEFAULT_STEP_UM,
    DURATION_MAX_MS,
    DURATION_MIN_MS,
    PRESSURE_LIMIT_MBAR as INJECT_PRESSURE_LIMIT_MBAR,
    SPEED_MAX_UM_S,
    SPEED_MIN_UM_S,
    STEP_LIMIT_UM,
    encode_params_message,
    validate_params,
)
from .ros_interfaces import (
    SRV_ACQ_START,
    SRV_ACQ_STOP,
    SRV_INJECT_START,
    SRV_UMP_STOP,
    SRV_ZERO,
    SRV_ZERO2,
    TOPIC_CAM_IMAGE_COMPRESSED,
    TOPIC_HEKA_CURRENT_PA,
    TOPIC_HEKA_RESISTANCE,
    TOPIC_HEKA_VOLTAGE_RAW,
    TOPIC_INJECT_PARAMS,
    TOPIC_INJECT_STATUS,
    TOPIC_MOTOR_LIVE,
    TOPIC_MOTOR_TGT,
    SRV_PRESSURE_RESET,
    TOPIC_PRESSURE_STATUS,
    TOPIC_PRESSURE_MBAR,
    TOPIC_PRESSURE_MEASURED,
    TOPIC_PRESSURE_TARGET,
    TOPIC_UMP_LIVE,
    TOPIC_UMP_TARGET,
    TOPIC_UMP2_LIVE,
    TOPIC_UMP2_TARGET,
    latched_qos,
)


AXIS_MIN, AXIS_MAX = 0, 20000
SPEED_MIN, SPEED_MAX = 10, 2000
MOTOR_MIN, MOTOR_MAX = -1_000_000, 1_000_000

# Matches the Fluigent LineUP push-pull channel range (+/-1000 mbar). The
# pressure node clamps to whatever the connected controller actually reports.
PRESSURE_LIMIT_MBAR = 1000.0

# ── Pressure preset buttons ────────────────────────────────────────────────
# Clicking a preset only fills the pressure box; press Send to apply it.
# Edit this tuple to change which shortcuts appear, in this order.
PRESSURE_PRESETS_MBAR = (50, 20, 0, -10, -20, -30, -100)

DEFAULT_AXIS_STEP = 50
DEFAULT_AXIS_TARGET = 10000
DEFAULT_SPEED = 1000
DEFAULT_MOTOR_STEP = 500
DEFAULT_PRESSURE_MBAR = 0.0

LIVE_POLL_MS = 50
SEND_THROTTLE_MS = 60
CAM_UPDATE_MS = 30
HEKA_PLOT_UPDATE_MS = 100
HEKA_PLOT_WINDOW_S = 0.25
HEKA_MAX_DRAW_POINTS = 2200

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".config", "ump_suite", "gui.ini")
# How long Inject waits for the driver to confirm it holds the values in the
# boxes. Normally it already does: the boxes publish whenever they change.
INJECT_PARAMS_ECHO_TIMEOUT_S = 1.0


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


class GuiNode(Node):
    """All ROS publishers/subscribers/clients used by the GUI."""

    def __init__(self):
        super().__init__("gui_node")

        self._heka_lock = threading.Lock()
        self.heka_voltage_history = deque()
        self.heka_current_history = deque()

        self.pub_ump_target = self.create_publisher(
            Int32MultiArray, TOPIC_UMP_TARGET, 10
        )
        self.pub_ump2_target = self.create_publisher(
            Int32MultiArray, TOPIC_UMP2_TARGET, 10
        )
        self.pub_motor_tgt = self.create_publisher(Int32, TOPIC_MOTOR_TGT, 10)
        # Applied readbacks are latched. The pressure driver deliberately
        # requests volatile commands, so restart never restores an old request.
        self.pub_pressure_mbar = self.create_publisher(
            Float32, TOPIC_PRESSURE_MBAR, latched_qos()
        )

        self.create_subscription(Int32MultiArray, TOPIC_UMP_LIVE, self._on_ump_live, 10)
        self.create_subscription(
            Int32MultiArray, TOPIC_UMP2_LIVE, self._on_ump2_live, 10
        )
        self.create_subscription(Int32, TOPIC_MOTOR_LIVE, self._on_motor_live, 10)
        self.create_subscription(
            CompressedImage, TOPIC_CAM_IMAGE_COMPRESSED, self._on_cam_image, 10
        )
        self.create_subscription(
            Float32, TOPIC_PRESSURE_MEASURED, self._on_pressure_measured, 10
        )
        self.create_subscription(
            Float32, TOPIC_PRESSURE_TARGET, self._on_pressure_target, latched_qos()
        )
        self.create_subscription(
            Float32MultiArray, TOPIC_HEKA_VOLTAGE_RAW, self._on_heka_voltage, 10
        )
        self.create_subscription(
            Float32MultiArray, TOPIC_HEKA_CURRENT_PA, self._on_heka_current, 10
        )
        self.create_subscription(
            Float32, TOPIC_HEKA_RESISTANCE, self._on_heka_resistance, 10
        )

        self.create_subscription(String, TOPIC_PRESSURE_STATUS, self._on_pressure_status,
                                 latched_qos())
        self.cli_pressure_reset = self.create_client(Trigger, SRV_PRESSURE_RESET)
        self.latest_pressure_status = None
        self.latest_pressure_status_stamp = 0.0
        self.latest_pressure_measured_stamp = 0.0
        self.latest_pressure_target_stamp = 0.0

        self.cli_acq_start = self.create_client(Trigger, SRV_ACQ_START)
        self.cli_acq_stop = self.create_client(Trigger, SRV_ACQ_STOP)
        self.cli_zero = self.create_client(Trigger, SRV_ZERO)
        self.cli_zero2 = self.create_client(Trigger, SRV_ZERO2)

        # Injection: latched settings for whoever triggers, the trigger itself,
        # and UMP 1's stop service, which also aborts a running injection.
        self.pub_inject_params = self.create_publisher(
            String, TOPIC_INJECT_PARAMS, latched_qos()
        )
        self.create_subscription(
            String, TOPIC_INJECT_STATUS, self._on_inject_status, latched_qos(depth=10)
        )
        self.cli_inject = self.create_client(Trigger, SRV_INJECT_START)
        self.cli_ump_stop = self.create_client(Trigger, SRV_UMP_STOP)
        self.latest_inject_status = None

        self.latest_live_ump = [0, 0, 0, 0]
        self.latest_live_ump2 = [0, 0, 0, 0]
        self.latest_live_motor = 0
        self.live_ump_stamp = None
        self.live_ump2_stamp = None
        self.live_motor_stamp = None
        self.latest_frame_bgr = None
        self.latest_pressure_mbar = None
        self.latest_pressure_target = None
        self.latest_heka_voltage = None
        self.latest_heka_current = None
        self.latest_heka_resistance = None

    def _on_ump_live(self, msg: Int32MultiArray):
        if len(msg.data) >= 4:
            self.latest_live_ump = [int(v) for v in msg.data[:4]]
            self.live_ump_stamp = time.monotonic()

    def _on_ump2_live(self, msg: Int32MultiArray):
        if len(msg.data) >= 4:
            self.latest_live_ump2 = [int(v) for v in msg.data[:4]]
            self.live_ump2_stamp = time.monotonic()

    def _on_motor_live(self, msg: Int32):
        self.latest_live_motor = int(msg.data)
        self.live_motor_stamp = time.monotonic()

    def _on_pressure_status(self, msg):
        try:
            status = json.loads(msg.data)
            if not isinstance(status, dict) or not isinstance(status.get('ready'), bool):
                return
        except (TypeError, ValueError):
            return
        self.latest_pressure_status = status
        self.latest_pressure_status_stamp = time.monotonic()

    def _on_inject_status(self, msg: String):
        try:
            status = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        if isinstance(status, dict) and isinstance(status.get('active'), bool):
            self.latest_inject_status = status

    def _on_pressure_measured(self, msg: Float32):
        value = float(msg.data)
        if math.isfinite(value):
            self.latest_pressure_mbar = value
            self.latest_pressure_measured_stamp = time.monotonic()

    def _on_pressure_target(self, msg: Float32):
        value = float(msg.data)
        if math.isfinite(value):
            self.latest_pressure_target = value
            self.latest_pressure_target_stamp = time.monotonic()

    def _on_cam_image(self, msg: CompressedImage):
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is not None:
                self.latest_frame_bgr = frame
        except Exception:
            pass

    def _append_heka_samples(self, history, latest_attr, msg: Float32MultiArray):
        data = list(msg.data)
        if len(data) < 2:
            return
        sample_rate_hz = float(data[0])
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            return

        samples = [float(v) for v in data[1:]]
        now = time.monotonic()
        first_t = now - (len(samples) - 1) / sample_rate_hz
        cutoff = now - HEKA_PLOT_WINDOW_S
        latest = None
        with self._heka_lock:
            for i, value in enumerate(samples):
                if not math.isfinite(value):
                    continue
                latest = value
                history.append((first_t + i / sample_rate_hz, value))
            if latest is not None:
                setattr(self, latest_attr, latest)
            while history and history[0][0] < cutoff:
                history.popleft()

    def _on_heka_voltage(self, msg: Float32MultiArray):
        self._append_heka_samples(
            self.heka_voltage_history, "latest_heka_voltage", msg
        )

    def _on_heka_current(self, msg: Float32MultiArray):
        self._append_heka_samples(
            self.heka_current_history, "latest_heka_current", msg
        )

    def _on_heka_resistance(self, msg: Float32):
        value = float(msg.data)
        if math.isfinite(value):
            self.latest_heka_resistance = value

    def get_heka_signal_snapshot(self):
        now = time.monotonic()
        cutoff = now - HEKA_PLOT_WINDOW_S
        with self._heka_lock:
            while self.heka_voltage_history and self.heka_voltage_history[0][0] < cutoff:
                self.heka_voltage_history.popleft()
            while self.heka_current_history and self.heka_current_history[0][0] < cutoff:
                self.heka_current_history.popleft()
            latest_voltage = self.latest_heka_voltage
            latest_current = self.latest_heka_current
            voltage_points = list(self.heka_voltage_history)
            current_points = list(self.heka_current_history)
        return now, latest_voltage, latest_current, voltage_points, current_points

    def call_trigger(self, client):
        if not client.wait_for_service(timeout_sec=1.0):
            return False, "service not available"

        fut = client.call_async(Trigger.Request())
        deadline = time.time() + 2.0
        while time.time() < deadline and not fut.done():
            time.sleep(0.01)

        if not fut.done() or fut.result() is None:
            return False, "no response"
        return bool(fut.result().success), str(fut.result().message)


class StatusPill(QLabel):
    """Small colored state badge used for acquisition state."""

    def __init__(self, text="--", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumWidth(74)
        self.set_on(False, text)

    def set_on(self, on, text):
        bg = "#dcfce7" if on else "#f1f5f9"
        fg = "#166534" if on else "#475569"
        border = "#86efac" if on else "#cbd5e1"
        self.setText(text)
        self.setStyleSheet(
            f"""
            QLabel {{
                color: {fg};
                background: {bg};
                border: 1px solid {border};
                border-radius: 12px;
                padding: 3px 8px;
                font-weight: 700;
            }}
            """
        )


class HekaPlot(QWidget):
    """Lightweight rolling HEKA signal plot drawn with QPainter."""

    def __init__(self, *, unit, waiting_topic, line_color, parent=None):
        super().__init__(parent)
        self.unit = unit
        self.waiting_topic = waiting_topic
        self.line_color = line_color
        self.now = time.monotonic()
        self.latest = None
        self.points = []
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_data(self, now, latest, points):
        self.now = now
        self.latest = latest
        self.points = points
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.white)

        width = max(320, self.width())
        height = max(110, self.height())
        left, right, top, bottom = 72, 24, 20, 42
        plot_w = max(1, width - left - right)
        plot_h = max(1, height - top - bottom)

        visible_all = [
            (t, v)
            for t, v in self.points
            if 0.0 <= self.now - t <= HEKA_PLOT_WINDOW_S
        ]
        visible = self._decimate(visible_all, HEKA_MAX_DRAW_POINTS)

        if visible_all:
            values = [v for _, v in visible_all]
            ymin = min(values)
            ymax = max(values)
            if abs(ymax - ymin) < 1e-9:
                pad = max(1.0, abs(ymax) * 0.05)
            else:
                pad = (ymax - ymin) * 0.12
            ymin -= pad
            ymax += pad
        else:
            ymin, ymax = 0.0, 1.0

        def x_from_time(t):
            age = self.now - t
            return left + ((HEKA_PLOT_WINDOW_S - age) / HEKA_PLOT_WINDOW_S) * plot_w

        def y_from_value(v):
            return top + (1.0 - ((v - ymin) / (ymax - ymin))) * plot_h

        grid = QPen(Qt.GlobalColor.lightGray)
        grid.setColor(Qt.GlobalColor.lightGray)
        axis = QPen(Qt.GlobalColor.black, 2)
        line = QPen(self.line_color, 3)

        painter.setPen(grid)
        for i in range(6):
            frac = i / 5.0
            x = left + frac * plot_w
            painter.drawLine(int(x), top, int(x), top + plot_h)
            milliseconds = HEKA_PLOT_WINDOW_S * 1000.0 * (frac - 1.0)
            painter.setPen(Qt.GlobalColor.darkGray)
            painter.drawText(
                int(x - 18),
                top + plot_h + 24,
                42,
                16,
                Qt.AlignCenter,
                f"{milliseconds:.0f}",
            )
            painter.setPen(grid)

        for i in range(5):
            frac = i / 4.0
            y = top + frac * plot_h
            value = ymax - frac * (ymax - ymin)
            painter.drawLine(left, int(y), left + plot_w, int(y))
            painter.setPen(Qt.GlobalColor.darkGray)
            painter.drawText(6, int(y - 8), left - 14, 16, Qt.AlignRight, f"{value:.3g}")
            painter.setPen(grid)

        painter.setPen(axis)
        painter.drawLine(left, top, left, top + plot_h)
        painter.drawLine(left, top + plot_h, left + plot_w, top + plot_h)

        painter.setPen(Qt.GlobalColor.darkGray)
        painter.drawText(left, height - 24, plot_w, 18, Qt.AlignCenter, "Time (ms)")
        painter.drawText(left + 4, 2, 120, 16, Qt.AlignLeft, self.unit)

        if len(visible) >= 2:
            painter.setPen(line)
            for i in range(len(visible) - 1):
                x1, y1 = x_from_time(visible[i][0]), y_from_value(visible[i][1])
                x2, y2 = x_from_time(visible[i + 1][0]), y_from_value(visible[i + 1][1])
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        elif len(visible) == 1:
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.line_color)
            x = x_from_time(visible[0][0])
            y = y_from_value(visible[0][1])
            painter.drawEllipse(int(x - 4), int(y - 4), 8, 8)
        else:
            painter.setPen(Qt.GlobalColor.darkGray)
            painter.drawText(
                left,
                top,
                plot_w,
                plot_h,
                Qt.AlignCenter,
                f"Waiting for {self.waiting_topic}",
            )

    @staticmethod
    def _decimate(points, max_points):
        if len(points) <= max_points:
            return points
        stride = max(1, math.ceil(len(points) / max_points))
        decimated = points[::stride]
        if decimated[-1] != points[-1]:
            decimated.append(points[-1])
        return decimated


class UmpPanel(QGroupBox):
    """Qt controls for one Sensapex UMP."""

    def __init__(self, app, *, label, pub_target, zero_client, live_getter,
                 live_stamp_getter, subtitle):
        super().__init__(label)
        self.app = app
        self.label = label
        self.pub_target = pub_target
        self.zero_client = zero_client
        self._live_getter = live_getter
        self._live_stamp_getter = live_stamp_getter
        self._updating = False

        self.axis_step = self._spin(DEFAULT_AXIS_STEP, 1, 5000)
        self.speed = self._spin(DEFAULT_SPEED, SPEED_MIN, SPEED_MAX)
        self.target_spins = {
            "X": self._spin(DEFAULT_AXIS_TARGET, AXIS_MIN, AXIS_MAX),
            "Y": self._spin(DEFAULT_AXIS_TARGET, AXIS_MIN, AXIS_MAX),
            "Z": self._spin(DEFAULT_AXIS_TARGET, AXIS_MIN, AXIS_MAX),
            "D": self._spin(DEFAULT_AXIS_TARGET, AXIS_MIN, AXIS_MAX),
        }
        self.live_labels = {axis: QLabel("--") for axis in self.target_spins}
        # The target boxes start at a placeholder. Until they have adopted the
        # real pose, nudging one axis would publish the placeholder on the other
        # three - a large, unrequested move on first use.
        self._adopted_live = False

        self._send_timer = QTimer(self)
        self._send_timer.setSingleShot(True)
        self._send_timer.timeout.connect(self.send_now)

        self._build(subtitle)

    @staticmethod
    def _spin(value, vmin, vmax):
        spin = QSpinBox()
        spin.setRange(vmin, vmax)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        spin.setAlignment(Qt.AlignRight)
        spin.setButtonSymbols(QSpinBox.NoButtons)
        spin.setFixedHeight(28)
        spin.setMaximumWidth(92)
        return spin

    def _build(self, subtitle):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)

        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("hint")
            layout.addWidget(sub)

        top = QGridLayout()
        top.setHorizontalSpacing(7)
        top.setVerticalSpacing(4)
        top.addWidget(QLabel("Axis step"), 0, 0)
        top.addWidget(self.axis_step, 0, 1)
        top.addWidget(QLabel("Speed"), 0, 2)
        top.addWidget(self.speed, 0, 3)
        layout.addLayout(top)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        grid.addWidget(QLabel("Axis"), 0, 0)
        grid.addWidget(QLabel("Nudge"), 0, 1, 1, 2)
        grid.addWidget(QLabel("Target"), 0, 3)
        grid.addWidget(QLabel("Live"), 0, 4)

        for row, axis in enumerate(("X", "Y", "Z", "D"), start=1):
            live = self.live_labels[axis]
            live.setObjectName("liveValue")
            live.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            live.setMinimumWidth(72)

            up = self._tool_button("+", lambda _checked=False, a=axis: self.bump_axis(a, +1))
            down = self._tool_button("-", lambda _checked=False, a=axis: self.bump_axis(a, -1))

            grid.addWidget(QLabel(axis), row, 0)
            grid.addWidget(up, row, 1)
            grid.addWidget(down, row, 2)
            grid.addWidget(self.target_spins[axis], row, 3)
            grid.addWidget(live, row, 4)
            self.target_spins[axis].editingFinished.connect(self.schedule_send)

        layout.addLayout(grid)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        commands = (
            ("Send", "Send Now", self.send_now, "primary"),
            ("Home", "Home (0,0,0,0)", self.home, "secondary"),
            ("Sync", "Sync to Live", self.sync_to_live, "secondary"),
            ("Zero", "Calibrate Zero", self.calibrate_zero, "secondary"),
        )
        for text, tooltip, slot, kind in commands:
            btn = QPushButton(text)
            btn.setToolTip(tooltip)
            btn.setProperty("kind", kind)
            btn.clicked.connect(slot)
            buttons.addWidget(btn)
        layout.addLayout(buttons)

    @staticmethod
    def _tool_button(text, slot):
        button = QToolButton()
        button.setText(text)
        button.setAutoRaise(False)
        button.setFixedSize(30, 26)
        button.clicked.connect(slot)
        return button

    def _resolved_speed(self):
        speed = clamp(int(self.speed.value()), SPEED_MIN, SPEED_MAX)
        self.speed.setValue(speed)
        return speed

    def _axis_value(self, axis):
        return int(self.target_spins[axis].value())

    def bump_axis(self, axis, sign):
        spin = self.target_spins[axis]
        new_val = clamp(
            int(spin.value()) + sign * int(self.axis_step.value()), AXIS_MIN, AXIS_MAX
        )
        spin.setValue(new_val)
        self.send_now()

    def send_now(self):
        if not self._adopted_live or not self._live_is_fresh():
            self.app.set_status(
                f"{self.label}: waiting for live position before commanding "
                "(targets still hold their placeholder)"
            )
            return
        self._send_timer.stop()
        x, y, z, d = (self._axis_value(axis) for axis in ("X", "Y", "Z", "D"))
        speed = self._resolved_speed()

        msg = Int32MultiArray()
        msg.data = [x, y, z, d, speed]
        self.pub_target.publish(msg)
        self.app.set_status(f"{self.label} target: X={x}, Y={y}, Z={z}, D={d} @ {speed}")

    def schedule_send(self):
        if not self._updating:
            self._send_timer.start(SEND_THROTTLE_MS)

    def home(self):
        self._updating = True
        for spin in self.target_spins.values():
            spin.setValue(0)
        self._updating = False
        self.send_now()

    def sync_to_live(self):
        self._updating = True
        for axis, value in zip(("X", "Y", "Z", "D"), self._live_getter()):
            self.target_spins[axis].setValue(int(value))
        self._updating = False
        self.app.set_status(f"{self.label} targets synced to live")

    def _live_is_fresh(self):
        stamp = self._live_stamp_getter()
        return (stamp is not None and stamp > getattr(self, "_live_after", float("-inf"))
                and 0 <= time.monotonic() - stamp <= 2.0)

    def update_live_display(self):
        if not self._live_is_fresh():
            self._adopted_live = False
            return
        values = list(self._live_getter())
        for axis, value in zip(("X", "Y", "Z", "D"), values):
            self.live_labels[axis].setText(f"{int(value):d}")
        # First real feedback: adopt it, so the first nudge is relative to where
        # the stage actually is rather than to the placeholder.
        if not self._adopted_live:
            self._updating = True
            for axis, value in zip(("X", "Y", "Z", "D"), values):
                self.target_spins[axis].setValue(int(value))
            self._updating = False
            self._adopted_live = True
            self.app.set_status(f"{self.label} targets adopted from live position")

    def calibrate_zero(self):
        self._send_timer.stop()
        self._adopted_live = False
        ok, msg = self.app.node.call_trigger(self.zero_client)
        if ok:
            self._live_after = time.monotonic()
        self.app.set_status(f"{self.label} zero: {ok} ({msg})")


class InjectionPanel(QGroupBox):
    """
    Inject button and the values the injection macro uses (see injection.py).

    The values are published on /inject/params whenever they change, so a
    trigger from anywhere uses what these boxes show. The UMP 1 driver runs the
    sequence; this panel only asks it to start, and shows its progress.
    """

    SETTINGS_KEYS = {
        "speed": ("injection/speed_um_s", DEFAULT_SPEED_UM_S, int),
        "step": ("injection/step_um", DEFAULT_STEP_UM, int),
        "pressure": ("injection/pressure_mbar", DEFAULT_INJECT_PRESSURE_MBAR, float),
        "duration": ("injection/duration_ms", DEFAULT_DURATION_MS, int),
    }
    AXIS_KEY = "injection/axis"

    def __init__(self, app, settings):
        super().__init__("Injection")
        self.app = app
        self.settings = settings
        self._token = uuid.uuid4().hex
        self._seq = 0
        self._published = None

        self.speed = self._int_box(SPEED_MIN_UM_S, SPEED_MAX_UM_S, " \u00b5m/s")
        self.step = self._int_box(-STEP_LIMIT_UM, STEP_LIMIT_UM, " \u00b5m")
        self.pressure = QDoubleSpinBox()
        self.pressure.setRange(-INJECT_PRESSURE_LIMIT_MBAR, INJECT_PRESSURE_LIMIT_MBAR)
        self.pressure.setDecimals(1)
        self.pressure.setSuffix(" mbar")
        self._style_box(self.pressure)
        self.duration = self._int_box(DURATION_MIN_MS, DURATION_MAX_MS, " ms")
        self.boxes = {"speed": self.speed, "step": self.step,
                      "pressure": self.pressure, "duration": self.duration}

        # One checkable button per axis; exactly one is selected (highlighted).
        self.axis_group = QButtonGroup(self)
        self.axis_group.setExclusive(True)
        self.axis_buttons = {}
        for axis in INJECT_AXES:
            button = QPushButton(axis)
            button.setCheckable(True)
            button.setProperty("kind", "axis")
            button.setFixedSize(38, 26)
            button.setToolTip(f"Inject along the {axis} axis")
            self.axis_group.addButton(button)
            self.axis_buttons[axis] = button
        self._load_settings()

        self.inject_button = QPushButton("Inject")
        self.inject_button.setProperty("kind", "danger")
        self.inject_button.setToolTip(
            "Move the selected axis in by Step at Speed, hold Pressure for Time, "
            "vent, move back")
        self.inject_button.setFixedSize(84, 62)
        self.inject_button.clicked.connect(self.inject)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setProperty("kind", "secondary")
        self.stop_button.setToolTip("Stop UMP 1 now; a running injection vents and halts")
        self.stop_button.clicked.connect(self.stop)
        self.state_label = QLabel("Waiting for the UMP 1 driver")
        self.state_label.setWordWrap(True)
        self.state_label.setMinimumHeight(40)  # room for two wrapped lines
        self.state_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self._build()
        for name, box in self.boxes.items():
            box.valueChanged.connect(lambda _value, n=name: self._on_value_changed(n))
        for axis, button in self.axis_buttons.items():
            button.toggled.connect(
                lambda checked, a=axis: checked and self._on_axis_selected(a))
        self.publish_params()

    @staticmethod
    def _style_box(box):
        box.setKeyboardTracking(False)
        box.setAlignment(Qt.AlignRight)
        box.setButtonSymbols(QAbstractSpinBox.NoButtons)
        box.setFixedHeight(28)
        box.setMinimumWidth(116)
        box.setMaximumWidth(124)

    @classmethod
    def _int_box(cls, vmin, vmax, suffix):
        box = QSpinBox()
        box.setRange(vmin, vmax)
        box.setSuffix(suffix)
        cls._style_box(box)
        return box

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)

        axis_row = QHBoxLayout()
        axis_row.setSpacing(4)
        axis_row.addWidget(QLabel("Axis"))
        for button in self.axis_buttons.values():
            axis_row.addWidget(button)
        axis_row.addStretch(1)
        layout.addLayout(axis_row)

        # The button with its four values beside it, one per row.
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.addWidget(self.inject_button, 0, 0, 4, 1, Qt.AlignVCenter)
        rows = (("Speed", self.speed), ("Step", self.step),
                ("Pressure", self.pressure), ("Time", self.duration))
        for row, (text, box) in enumerate(rows):
            grid.addWidget(QLabel(text), row, 1)
            grid.addWidget(box, row, 2)
        grid.addWidget(self.stop_button, 0, 3, 4, 1, Qt.AlignVCenter)
        grid.setColumnStretch(4, 1)
        layout.addLayout(grid)

        hint = QLabel("Step: + moves the axis up first, - down. Pressure: '-' pulls. "
                      "Time starts when the pressure driver acknowledges.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.state_label)

    # ── Settings ───────────────────────────────────────────────────────────
    def _load_settings(self):
        for name, (key, default, kind) in self.SETTINGS_KEYS.items():
            box = self.boxes[name]
            try:
                # INI files return text, and may hold "50.0" for an integer box.
                value = float(self.settings.value(key, default))
            except (TypeError, ValueError):
                value = float(default)
            if not math.isfinite(value) or not box.minimum() <= value <= box.maximum():
                value = float(default)
            box.setValue(kind(round(value)) if kind is int else value)
        axis = str(self.settings.value(self.AXIS_KEY, DEFAULT_INJECT_AXIS)).strip().upper()
        if axis not in self.axis_buttons:
            axis = DEFAULT_INJECT_AXIS
        self.axis_buttons[axis].setChecked(True)

    def current_axis(self):
        checked = self.axis_group.checkedButton()
        return checked.text() if checked is not None else DEFAULT_INJECT_AXIS

    def _on_axis_selected(self, axis):
        self.settings.setValue(self.AXIS_KEY, axis)
        self.settings.sync()
        self.publish_params()

    def _on_value_changed(self, name):
        key, _default, kind = self.SETTINGS_KEYS[name]
        self.settings.setValue(key, kind(self.boxes[name].value()))
        self.settings.sync()
        self.publish_params()

    # ── ROS ────────────────────────────────────────────────────────────────
    def current_params(self):
        """Return the boxes as InjectionParams; raise ValueError if they are invalid."""
        return validate_params({
            "axis": self.current_axis(),
            "speed_um_s": self.speed.value(),
            "step_um": self.step.value(),
            "pressure_mbar": self.pressure.value(),
            "duration_ms": self.duration.value(),
        })

    def publish_params(self, force=False):
        """Publish the boxes if they changed; return the sequence number, or None."""
        try:
            params = self.current_params()
        except ValueError as exc:
            self.state_label.setText(f"Not published: {exc}")
            return None
        if force or self._published is None or self._published[0] != params:
            self._seq += 1
            self.app.node.pub_inject_params.publish(
                String(data=encode_params_message(params, self._token, self._seq)))
            self._published = (params, self._seq)
        return self._published[1]

    def _driver_has(self, seq):
        status = self.app.node.latest_inject_status
        return (status is not None and status.get("params_token") == self._token
                and status.get("params_seq") == seq)

    def inject(self):
        for box in self.boxes.values():
            box.interpretText()
        seq = self.publish_params()
        if seq is not None and not self._driver_has(seq):
            # Someone else published values since; make these the current ones.
            seq = self.publish_params(force=True)
        if seq is None:
            self.app.set_status("Injection not started: " + self.state_label.text())
            return
        # The trigger carries no values; make sure the driver holds these ones.
        deadline = time.monotonic() + INJECT_PARAMS_ECHO_TIMEOUT_S
        while not self._driver_has(seq):
            if time.monotonic() > deadline:
                self.app.set_status(
                    "Injection not started: the UMP 1 driver has not confirmed the "
                    "values (is it running?)")
                return
            time.sleep(0.01)
        ok, message = self.app.node.call_trigger(self.app.node.cli_inject)
        self.app.set_status(message if message else f"Inject: {ok}")

    def stop(self):
        ok, message = self.app.node.call_trigger(self.app.node.cli_ump_stop)
        self.app.set_status(f"UMP 1 stop: {'OK' if ok else 'FAILED'} ({message})")

    def update_status(self):
        status = self.app.node.latest_inject_status
        if status is None:
            self.inject_button.setEnabled(True)
            self.state_label.setText("Waiting for the UMP 1 driver")
            return
        active = bool(status.get("active"))
        self.inject_button.setEnabled(not active)
        stage = str(status.get("stage", ""))
        message = str(status.get("message", ""))
        count = status.get("count", 0)
        if active:
            text = f"#{count} running ({stage}): {message}"
        elif stage in ("done", "aborted"):
            text = f"#{count} {stage}: {message}"
        else:
            text = message or "ready"
        self.state_label.setText(text)


class UMPGuiApp(QMainWindow):
    """Main Qt application window."""

    def __init__(self, node: GuiNode, settings=None):
        super().__init__()
        self.node = node
        self.settings = settings or QSettings(SETTINGS_PATH, QSettings.IniFormat)
        self.setWindowTitle("Patch Clamping Robot")
        self.resize(1280, 900)
        self.setMinimumSize(900, 650)

        self.status = QLabel("Ready")
        self.acq_pill = StatusPill("STOPPED")
        self.live_pressure = QLabel("-- mbar")
        self.live_pressure.setObjectName("liveValue")
        self.live_pressure.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # What the node actually applied, so a clamped value is visible.
        self.live_pressure_target = QLabel("-- mbar")
        self.live_pressure_target.setObjectName("liveValue")
        self.live_pressure_target.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # One box for the exact pressure; type a leading '-' for suction.
        self.pressure_mbar = self._double_spin(
            DEFAULT_PRESSURE_MBAR, -PRESSURE_LIMIT_MBAR, PRESSURE_LIMIT_MBAR
        )
        self.live_motor = QLabel("--")
        self.live_motor.setObjectName("liveValue")
        self.live_motor.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.live_resistance = QLabel("-- MOhm")
        self.live_resistance.setObjectName("resistanceValue")
        self.live_resistance.setAlignment(Qt.AlignCenter)
        self.heka_voltage = QLabel("Voltage: -- V")
        self.heka_voltage.setObjectName("sectionTitle")
        self.heka_current = QLabel("Current: -- pA")
        self.heka_current.setObjectName("sectionTitle")

        self.motor_step = self._spin(DEFAULT_MOTOR_STEP, 1, 100_000)
        self.motor_target = self._spin(0, MOTOR_MIN, MOTOR_MAX)

        self.camera_label = QLabel("Blackfly S Live")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setObjectName("cameraView")
        self.camera_label.setMinimumSize(360, 500)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.heka_voltage_plot = HekaPlot(
            unit="Voltage (V)",
            waiting_topic=TOPIC_HEKA_VOLTAGE_RAW,
            line_color=Qt.GlobalColor.red,
        )
        self.heka_current_plot = HekaPlot(
            unit="Current (pA)",
            waiting_topic=TOPIC_HEKA_CURRENT_PA,
            line_color=Qt.GlobalColor.black,
        )

        self.panel1 = UmpPanel(
            self,
            label="UMP 1",
            pub_target=node.pub_ump_target,
            zero_client=node.cli_zero,
            live_getter=lambda: node.latest_live_ump,
            live_stamp_getter=lambda: node.live_ump_stamp,
            subtitle=None,
        )
        self.panel2 = UmpPanel(
            self,
            label="UMP 2",
            pub_target=node.pub_ump2_target,
            zero_client=node.cli_zero2,
            live_getter=lambda: node.latest_live_ump2,
            live_stamp_getter=lambda: node.live_ump2_stamp,
            subtitle=None,
        )
        self.injection_panel = InjectionPanel(self, self.settings)

        self._build_ui()

        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self._poll_live_to_gui)
        self.live_timer.start(LIVE_POLL_MS)

        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self._update_camera_view)
        self.camera_timer.start(CAM_UPDATE_MS)

        self.heka_timer = QTimer(self)
        self.heka_timer.timeout.connect(self._update_heka_plots)
        self.heka_timer.start(HEKA_PLOT_UPDATE_MS)

    @staticmethod
    def _spin(value, vmin, vmax):
        spin = QSpinBox()
        spin.setRange(vmin, vmax)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        spin.setAlignment(Qt.AlignRight)
        spin.setButtonSymbols(QSpinBox.NoButtons)
        spin.setFixedHeight(28)
        spin.setMaximumWidth(92)
        return spin

    @staticmethod
    def _double_spin(value, vmin, vmax):
        spin = QDoubleSpinBox()
        spin.setRange(vmin, vmax)
        spin.setDecimals(1)
        spin.setSingleStep(1.0)
        spin.setSuffix(" mbar")
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        spin.setAlignment(Qt.AlignRight)
        spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        spin.setFixedHeight(28)
        spin.setMaximumWidth(110)
        return spin

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(8)
        root_layout.addLayout(self._header())

        outer = QHBoxLayout()
        outer.setSpacing(14)
        root_layout.addLayout(outer, 1)

        left = QFrame()
        left.setObjectName("panelColumn")
        left.setMinimumWidth(390)
        left.setMaximumWidth(410)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 4, 8, 4)
        left_layout.setSpacing(3)

        left_layout.addWidget(self.panel1)
        left_layout.addWidget(self.panel2)
        left_layout.addWidget(self.injection_panel)
        left_layout.addWidget(self._motor_group())
        left_layout.addWidget(self._pressure_group())
        left_layout.addWidget(self._acquisition_group())
        left_layout.addWidget(self._resistance_group())
        left_layout.addStretch(1)
        left_layout.addWidget(self.status)

        right = QFrame()
        right.setObjectName("panelColumn")
        right.setMinimumWidth(420)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        camera_title = QLabel("Blackfly S Live")
        camera_title.setObjectName("sectionTitle")
        right_layout.addWidget(camera_title)
        right_layout.addWidget(self.camera_label, 2)
        right_layout.addWidget(self.heka_voltage)
        right_layout.addWidget(self.heka_voltage_plot, 1)
        right_layout.addWidget(self.heka_current)
        right_layout.addWidget(self.heka_current_plot, 1)

        left_scroll = QScrollArea()
        left_scroll.setObjectName("controlScroll")
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFixedWidth(430)
        left_scroll.setWidget(left)

        outer.addWidget(left_scroll)
        outer.addWidget(right, 1)
        self._apply_style()

    def _header(self):
        bar = QHBoxLayout()
        bar.setSpacing(10)

        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel("Patch Clamping Robot")
        title.setObjectName("appTitle")
        subtitle = QLabel(
            "Dual UMP control with focusing knob, pressure control, camera, "
            "and HEKA voltage/current monitoring"
        )
        subtitle.setObjectName("hint")
        text.addWidget(title)
        text.addWidget(subtitle)

        bar.addLayout(text)
        bar.addStretch(1)
        return bar

    def _motor_group(self):
        group = QGroupBox("Motor (ODrive)")
        layout = QGridLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)

        layout.addWidget(QLabel("Step"), 0, 0)
        layout.addWidget(self.motor_step, 0, 1)
        layout.addWidget(QLabel("Target"), 0, 2)
        layout.addWidget(self.motor_target, 0, 3)
        layout.addWidget(QLabel("Live"), 1, 0)
        layout.addWidget(self.live_motor, 1, 1)

        up = QToolButton()
        up.setText("+")
        up.setFixedSize(30, 26)
        up.clicked.connect(lambda: self._bump_motor(+1))
        down = QToolButton()
        down.setText("-")
        down.setFixedSize(30, 26)
        down.clicked.connect(lambda: self._bump_motor(-1))
        layout.addWidget(up, 1, 2)
        layout.addWidget(down, 1, 3)

        send = QPushButton("Send")
        send.setProperty("kind", "primary")
        send.clicked.connect(self._publish_motor_target)
        layout.addWidget(send, 1, 4)

        self.motor_target.editingFinished.connect(self._publish_motor_target)
        return group

    def _pressure_group(self):
        group = QGroupBox("Pressure (Fluigent)")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hint = QLabel("Type a value in mbar (use a leading '-' to pull), then Send.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        entry = QHBoxLayout()
        entry.setSpacing(6)
        entry.addWidget(QLabel("Pressure"))
        entry.addWidget(self.pressure_mbar)
        send = QPushButton("Send")
        send.setProperty("kind", "primary")
        send.setToolTip("Apply the value in the box")
        send.clicked.connect(self._send_pressure)
        entry.addWidget(send)
        entry.addStretch(1)
        layout.addLayout(entry)

        # Preset buttons only fill the box; the operator still presses Send.
        presets = QGridLayout()
        presets.setHorizontalSpacing(4)
        presets.setVerticalSpacing(4)
        for i, value in enumerate(PRESSURE_PRESETS_MBAR):
            presets.addWidget(self._preset_button(value), i // 4, i % 4)
        layout.addLayout(presets)

        readback = QHBoxLayout()
        readback.setSpacing(6)
        readback.addWidget(QLabel("SDK target"))
        readback.addWidget(self.live_pressure_target)
        readback.addWidget(QLabel("Measured"))
        readback.addWidget(self.live_pressure)
        readback.addStretch(1)
        layout.addLayout(readback)
        self.pressure_health = QLabel("Waiting for pressure driver status")
        self.pressure_health.setWordWrap(True)
        layout.addWidget(self.pressure_health)
        reset = QPushButton("Reconnect / reset (0 mbar)")
        reset.setToolTip("Reconnect after resolving a fault; requests zero pressure")
        reset.clicked.connect(self._reset_pressure)
        layout.addWidget(reset)
        return group

    def _preset_button(self, value):
        # Signed label, except 0 which reads better unsigned.
        label = f"{value:g}" if value == 0 else f"{value:+g}"
        button = QPushButton(label)
        button.setToolTip(f"Fill the box with {label} mbar")
        button.setProperty("kind", "secondary")
        button.setFixedHeight(26)
        button.clicked.connect(lambda _checked=False, v=value: self._fill_pressure(v))
        return button

    def _acquisition_group(self):
        group = QGroupBox("Data Acquisition")
        layout = QGridLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(6)

        start = QPushButton("Start")
        start.setProperty("kind", "primary")
        start.clicked.connect(self._acq_start)
        stop = QPushButton("Stop")
        stop.setProperty("kind", "secondary")
        stop.clicked.connect(self._acq_stop)

        layout.addWidget(start, 0, 0)
        layout.addWidget(stop, 0, 1)
        layout.addWidget(self.acq_pill, 0, 2)
        return group

    def _resistance_group(self):
        group = QGroupBox("Live Resistance")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.live_resistance)
        return group

    def set_status(self, text):
        self.status.setText(text)

    def _motor_is_fresh(self):
        stamp = self.node.live_motor_stamp
        return stamp is not None and 0 <= time.monotonic() - stamp <= 2.0

    def _publish_motor_target(self):
        if not getattr(self, "_motor_adopted", False) or not self._motor_is_fresh():
            self.set_status("Motor: waiting for fresh live position")
            return
        target = int(self.motor_target.value())
        self.node.pub_motor_tgt.publish(Int32(data=target))
        self.set_status(f"Motor target: {target}")

    def _bump_motor(self, sign):
        new_val = clamp(
            int(self.motor_target.value()) + sign * int(self.motor_step.value()),
            MOTOR_MIN,
            MOTOR_MAX,
        )
        self.motor_target.setValue(new_val)
        self._publish_motor_target()

    def _fill_pressure(self, value):
        """Preset click: load the box only, so nothing reaches the device yet."""
        self.pressure_mbar.setValue(float(value))
        self.set_status(f"Pressure box set to {value:+g} mbar (press Send to apply)")

    def _send_pressure(self):
        self.pressure_mbar.interpretText()
        value = float(self.pressure_mbar.value())
        status = self.node.latest_pressure_status
        fresh = time.monotonic() - self.node.latest_pressure_status_stamp <= 3.0
        if value != 0.0 and (not fresh or not status or not status.get('ready')):
            reason = status.get('message') if fresh and status else 'driver status unavailable'
            self.set_status(
                f"Pressure not sent: {reason}. Use Reconnect / reset after resolving it."
            )
            return
        self._pending_pressure = (value, time.monotonic())
        self.node.pub_pressure_mbar.publish(Float32(data=value))
        self.set_status(f"Pressure request sent: {value:+.1f} mbar; awaiting SDK acknowledgment")

    def _reset_pressure(self):
        ok, message = self.node.call_trigger(self.node.cli_pressure_reset)
        self._pending_pressure = None
        self.set_status(f"Pressure reset: {'OK' if ok else 'FAILED'} ({message})")

    def _acq_start(self):
        ok, msg = self.node.call_trigger(self.node.cli_acq_start)
        self.acq_pill.set_on(ok, "RUNNING" if ok else "FAILED")
        self.set_status(f"Acq start: {ok} ({msg})")

    def _acq_stop(self):
        ok, msg = self.node.call_trigger(self.node.cli_acq_stop)
        self.acq_pill.set_on(False, "STOPPED")
        self.set_status(f"Acq stop: {ok} ({msg})")

    def _poll_live_to_gui(self):
        self.panel1.update_live_display()
        self.panel2.update_live_display()
        self.injection_panel.update_status()
        self.live_motor.setText(f"{int(self.node.latest_live_motor):d}")
        if not self._motor_is_fresh():
            self._motor_adopted = False
        elif not getattr(self, "_motor_adopted", False):
            self.motor_target.setValue(int(self.node.latest_live_motor))
            self._motor_adopted = True

        measured = self.node.latest_pressure_mbar
        measured_fresh = time.monotonic() - self.node.latest_pressure_measured_stamp <= 2.0
        self.live_pressure.setText(
            "-- mbar" if measured is None else
            f"{measured:+.1f} mbar" + ("" if measured_fresh else " (stale)")
        )
        applied = self.node.latest_pressure_target
        self.live_pressure_target.setText(
            "-- mbar" if applied is None else f"{applied:+.1f} mbar"
        )

        status = self.node.latest_pressure_status
        fresh = time.monotonic() - self.node.latest_pressure_status_stamp <= 3.0
        if not fresh or not status:
            health = "Pressure driver offline or status unavailable"
        elif not status.get('ready'):
            health = "FAULT: " + status.get('message', 'not ready')
        else:
            health = "Pressure driver ready; compare SDK target with measured pressure"
        self.pressure_health.setText(health)
        pending = getattr(self, '_pending_pressure', None)
        if pending is not None:
            value, sent = pending
            if fresh and status and not status.get('ready'):
                self.set_status("Pressure request failed: " + status.get('message', 'not ready'))
                self._pending_pressure = None
            elif self.node.latest_pressure_target_stamp >= sent:
                self.set_status(f"SDK acknowledged {applied:+.1f} mbar; verify measured pressure")
                self._pending_pressure = None
            elif time.monotonic() - sent > 3.0:
                self.set_status(f"No SDK acknowledgment for {value:+.1f} mbar pressure request")
                self._pending_pressure = None

        resistance = self.node.latest_heka_resistance
        if resistance is None:
            self.live_resistance.setText("-- MOhm")
        elif resistance >= 1000.0:
            self.live_resistance.setText(f"{resistance / 1000.0:.3f} GOhm")
        else:
            self.live_resistance.setText(f"{resistance:.2f} MOhm")

    def _update_camera_view(self):
        frame = self.node.latest_frame_bgr
        if frame is None:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled)

    def _update_heka_plots(self):
        now, latest_v, latest_i, voltage_points, current_points = (
            self.node.get_heka_signal_snapshot()
        )

        if latest_v is None:
            self.heka_voltage.setText("Voltage: -- V")
        else:
            self.heka_voltage.setText(f"Voltage: {latest_v:.6g} V")

        if latest_i is None:
            self.heka_current.setText("Current: -- pA")
        else:
            self.heka_current.setText(f"Current: {latest_i:.6g} pA")

        self.heka_voltage_plot.set_data(now, latest_v, voltage_points)
        self.heka_current_plot.set_data(now, latest_i, current_points)

    def _apply_style(self):
        QApplication.instance().setStyleSheet(
            """
            QWidget#root {
                background: #eef2f7;
                color: #172033;
                font-family: "Segoe UI", "Inter", "Noto Sans", sans-serif;
                font-size: 12px;
            }
            QFrame#panelColumn {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 14px;
            }
            QScrollArea#controlScroll {
                background: transparent;
                border: none;
            }
            QScrollArea#controlScroll > QWidget > QWidget {
                background: transparent;
            }
            QScrollBar:vertical {
                background: #e2e8f0;
                width: 10px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #94a3b8;
                border-radius: 5px;
                min-height: 42px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }
            QLabel#appTitle {
                font-size: 24px;
                font-weight: 800;
                color: #111827;
            }
            QLabel#sectionTitle {
                font-size: 17px;
                font-weight: 800;
                color: #111827;
            }
            QLabel#hint {
                color: #64748b;
                font-size: 12px;
            }
            QLabel#liveValue {
                background: #f8fafc;
                border: 1px solid #dbe4ef;
                border-radius: 8px;
                padding: 5px 8px;
                color: #0f172a;
                font-weight: 700;
            }
            QLabel#resistanceValue {
                background: #ecfeff;
                border: 1px solid #67e8f9;
                border-radius: 10px;
                padding: 8px 10px;
                color: #155e75;
                font-size: 18px;
                font-weight: 800;
            }
            QLabel#cameraView {
                background: #0b1120;
                color: #cbd5e1;
                border-radius: 14px;
                border: 1px solid #1f2937;
                font-size: 18px;
                font-weight: 700;
            }
            QGroupBox {
                background: #f8fafc;
                border: 1px solid #dbe4ef;
                border-radius: 10px;
                margin-top: 10px;
                padding: 5px;
                font-weight: 800;
                color: #111827;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                left: 10px;
            }
            QSpinBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 7px;
                padding: 3px 6px;
                min-height: 18px;
                color: #0f172a;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #2563eb;
            }
            QPushButton, QToolButton {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 7px;
                padding: 4px 8px;
                color: #172033;
                font-weight: 700;
            }
            QPushButton:hover, QToolButton:hover {
                background: #f1f5f9;
                border-color: #94a3b8;
            }
            QPushButton[kind="primary"] {
                background: #2563eb;
                border-color: #2563eb;
                color: #ffffff;
            }
            QPushButton[kind="primary"]:hover {
                background: #1d4ed8;
            }
            QPushButton[kind="danger"] {
                background: #fff7ed;
                border-color: #fdba74;
                color: #9a3412;
            }
            QPushButton[kind="danger"]:hover {
                background: #ffedd5;
            }
            QPushButton[kind="secondary"] {
                background: #ffffff;
            }
            QPushButton[kind="axis"] {
                background: #ffffff;
                padding: 2px;
            }
            QPushButton[kind="axis"]:checked {
                background: #2563eb;
                border-color: #2563eb;
                color: #ffffff;
                font-weight: 700;
            }
            """
        )


def main():
    from .runtime_guard import acquire_process_lock
    acquire_process_lock("gui")

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    rclpy.init()
    node = GuiNode()

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def spin():
        try:
            executor.spin()
        except ExternalShutdownException:
            pass

    spin_thread = threading.Thread(target=spin, daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    window = UMPGuiApp(node)
    window.showMaximized()
    rc = app.exec_()

    executor.shutdown()
    spin_thread.join()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()

"""
Modern Qt control panel for the UMP suite.

The ROS surface is intentionally the same as the original Tk GUI:
  * publishes absolute UMP1 / UMP2 targets
  * publishes ODrive motor targets
  * publishes solenoid commands
  * subscribes to live robot, pressure, camera, and HEKA voltage/current topics
  * calls acquisition and UMP zeroing services

Only the presentation layer changes. PyQt5 is used because it is available in
the ROS environment on this machine and gives the app a more polished desktop
feel without requiring a separate web server.
"""

import math
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from PyQt5.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QShortcut,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32, Int32MultiArray
from std_srvs.srv import Trigger

from .ros_interfaces import (
    SRV_ACQ_START,
    SRV_ACQ_STOP,
    SRV_ZERO,
    SRV_ZERO2,
    TOPIC_CAM_IMAGE_COMPRESSED,
    TOPIC_HEKA_CURRENT_PA,
    TOPIC_HEKA_RESISTANCE,
    TOPIC_HEKA_VOLTAGE_RAW,
    TOPIC_MOTOR_LIVE,
    TOPIC_MOTOR_TGT,
    TOPIC_SOL1_CMD,
    TOPIC_SOL1_STATE,
    TOPIC_UMP_LIVE,
    TOPIC_UMP_TARGET,
    TOPIC_UMP2_LIVE,
    TOPIC_UMP2_TARGET,
)


AXIS_MIN, AXIS_MAX = 0, 20000
SPEED_MIN, SPEED_MAX = 10, 2000
MOTOR_MIN, MOTOR_MAX = -1_000_000, 1_000_000

DEFAULT_AXIS_STEP = 50
DEFAULT_AXIS_TARGET = 10000
DEFAULT_SPEED = 1000
DEFAULT_MOTOR_STEP = 500

LIVE_POLL_MS = 50
SEND_THROTTLE_MS = 60
CAM_UPDATE_MS = 30
HEKA_PLOT_UPDATE_MS = 100
HEKA_PLOT_WINDOW_S = 0.25
HEKA_MAX_DRAW_POINTS = 2200


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
        self.pub_sol1_cmd = self.create_publisher(Bool, TOPIC_SOL1_CMD, 10)

        self.create_subscription(Int32MultiArray, TOPIC_UMP_LIVE, self._on_ump_live, 10)
        self.create_subscription(
            Int32MultiArray, TOPIC_UMP2_LIVE, self._on_ump2_live, 10
        )
        self.create_subscription(Int32, TOPIC_MOTOR_LIVE, self._on_motor_live, 10)
        self.create_subscription(
            CompressedImage, TOPIC_CAM_IMAGE_COMPRESSED, self._on_cam_image, 10
        )
        self.create_subscription(Bool, TOPIC_SOL1_STATE, self._on_sol1_state, 10)
        self.create_subscription(
            Float32MultiArray, TOPIC_HEKA_VOLTAGE_RAW, self._on_heka_voltage, 10
        )
        self.create_subscription(
            Float32MultiArray, TOPIC_HEKA_CURRENT_PA, self._on_heka_current, 10
        )
        self.create_subscription(
            Float32, TOPIC_HEKA_RESISTANCE, self._on_heka_resistance, 10
        )

        self.cli_acq_start = self.create_client(Trigger, SRV_ACQ_START)
        self.cli_acq_stop = self.create_client(Trigger, SRV_ACQ_STOP)
        self.cli_zero = self.create_client(Trigger, SRV_ZERO)
        self.cli_zero2 = self.create_client(Trigger, SRV_ZERO2)

        self.latest_live_ump = [0, 0, 0, 0]
        self.latest_live_ump2 = [0, 0, 0, 0]
        self.latest_live_motor = 0
        self.latest_frame_bgr = None
        self.latest_sol1_state = False
        self.latest_heka_voltage = None
        self.latest_heka_current = None
        self.latest_heka_resistance = None

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
    """Small colored state badge used for solenoid and acquisition state."""

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


class KeyRouter(QObject):
    """Global key handler for UMP1 shortcuts."""

    bump = pyqtSignal(str, int)

    KEY_BINDINGS = {
        Qt.Key_Up: ("Z", +1),
        Qt.Key_Down: ("Z", -1),
        Qt.Key_A: ("X", -1),
        Qt.Key_D: ("X", +1),
        Qt.Key_S: ("Y", -1),
        Qt.Key_W: ("Y", +1),
        Qt.Key_Comma: ("D", -1),
        Qt.Key_Less: ("D", -1),
        Qt.Key_Period: ("D", +1),
        Qt.Key_Greater: ("D", +1),
    }

    def eventFilter(self, _obj, event):
        if event.type() != QEvent.KeyPress:
            return False
        binding = self.KEY_BINDINGS.get(event.key())
        if binding is None:
            return False
        axis, sign = binding
        self.bump.emit(axis, sign)
        return True


class UmpPanel(QGroupBox):
    """Qt controls for one Sensapex UMP."""

    def __init__(self, app, *, label, pub_target, zero_client, live_getter, subtitle):
        super().__init__(label)
        self.app = app
        self.label = label
        self.pub_target = pub_target
        self.zero_client = zero_client
        self._live_getter = live_getter
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

    def update_live_display(self):
        for axis, value in zip(("X", "Y", "Z", "D"), self._live_getter()):
            self.live_labels[axis].setText(f"{int(value):d}")

    def calibrate_zero(self):
        ok, msg = self.app.node.call_trigger(self.zero_client)
        self.app.set_status(f"{self.label} zero: {ok} ({msg})")


class UMPGuiApp(QMainWindow):
    """Main Qt application window."""

    def __init__(self, node: GuiNode):
        super().__init__()
        self.node = node
        self.setWindowTitle("Patch Clamping Robot")
        self.resize(1280, 900)
        self.setMinimumSize(900, 650)

        self.status = QLabel("Ready")
        self.acq_pill = StatusPill("STOPPED")
        self.sol1_pill = StatusPill("OFF")
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
            subtitle=None,
        )
        self.panel2 = UmpPanel(
            self,
            label="UMP 2",
            pub_target=node.pub_ump2_target,
            zero_client=node.cli_zero2,
            live_getter=lambda: node.latest_live_ump2,
            subtitle=None,
        )

        self._build_ui()
        self._install_shortcuts()

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
        group = QGroupBox("Pressure (Solenoid)")
        layout = QGridLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)

        layout.addWidget(QLabel("Solenoid"), 0, 0)
        layout.addWidget(self._pressure_button("ON", True), 0, 1)
        layout.addWidget(self._pressure_button("OFF", False), 0, 2)
        layout.addWidget(self.sol1_pill, 0, 3)
        return group

    def _pressure_button(self, text, on):
        button = QPushButton(text)
        button.setProperty("kind", "danger" if on else "secondary")
        button.clicked.connect(lambda _checked=False: self._set_solenoid(on))
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

    def _install_shortcuts(self):
        # QShortcut catches common letter keys even when widgets have focus.
        for key, axis, sign in (
            ("W", "Y", +1),
            ("S", "Y", -1),
            ("A", "X", -1),
            ("D", "X", +1),
            (",", "D", -1),
            (".", "D", +1),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda a=axis, s=sign: self.panel1.bump_axis(a, s))

        self.key_router = KeyRouter(self)
        self.key_router.bump.connect(self.panel1.bump_axis)
        QApplication.instance().installEventFilter(self.key_router)

    def set_status(self, text):
        self.status.setText(text)

    def _publish_motor_target(self):
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

    def _set_solenoid(self, on):
        self.node.pub_sol1_cmd.publish(Bool(data=bool(on)))
        self.set_status(f"Solenoid: {'ON' if on else 'OFF'}")

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
        self.live_motor.setText(f"{int(self.node.latest_live_motor):d}")
        sol1_text = "ON" if self.node.latest_sol1_state else "OFF"
        self.sol1_pill.set_on(self.node.latest_sol1_state, sol1_text)
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

    def closeEvent(self, event):
        QApplication.instance().removeEventFilter(self.key_router)
        event.accept()

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
            QSpinBox {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 7px;
                padding: 3px 6px;
                min-height: 18px;
                color: #0f172a;
            }
            QSpinBox:focus {
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
            """
        )


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    rclpy.init()
    node = GuiNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 9))
    window = UMPGuiApp(node)
    window.showMaximized()
    rc = app.exec_()

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()

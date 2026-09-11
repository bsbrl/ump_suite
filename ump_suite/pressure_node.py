"""
ROS2 driver for the Fluigent push-pull pressure controller (LineUP).

The pressure is commanded as an exact value in mbar on a single topic:

    /pressure/mbar = -20.0  ->  fgt_set_pressure(channel, -20.0)   (pull)
    /pressure/mbar =  50.0  ->  fgt_set_pressure(channel,  50.0)   (push)
    /pressure/mbar =   0.0  ->  vented

Whatever arrives is clamped to the range the controller reports for its channel,
so a mistyped or out-of-range value cannot exceed the hardware limits.

Command subscriptions are volatile: restarting this driver never restores an
old publisher's cached pressure. Startup vents and only fresh commands are
accepted. Applied-target readbacks remain latched for late-joining loggers.

Two readbacks are published:

  * /pressure/target_mbar   the value actually written to the device, i.e. the
                            request after clamping. The logger records this, so
                            a dataset never claims a pressure the controller
                            never received.
  * /pressure/measured_mbar the controller's own pressure sensor.

Requires the Fluigent Python SDK (`fluigent_sdk`), which bundles its own
libfgt_SDK.so, so no system library setup is needed.
"""

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32

from Fluigent.SDK import (
    fgt_close,
    fgt_detect,
    fgt_get_pressure,
    fgt_get_pressureRange,
    fgt_init,
    fgt_set_pressure,
)

from .ros_interfaces import (
    TOPIC_PRESSURE_MBAR,
    TOPIC_PRESSURE_MEASURED,
    TOPIC_PRESSURE_TARGET,
    latched_qos,
)


# Pressure applied on connect and on shutdown.
IDLE_MBAR = 0.0


def require_sdk_ok(status, operation):
    """Raise on SDK error codes, including errors the SDK only prints."""
    if int(status) != 0:
        raise RuntimeError(f"{operation} returned SDK error {status}")


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


class PressureNode(Node):
    def __init__(self):
        super().__init__("pressure_node")

        self.declare_parameter("channel", 0)
        self.declare_parameter("poll_ms", 100)
        # Optional safety envelope, intersected with the device range below.
        # Tighten these to keep well inside what the pipette can take.
        self.declare_parameter("max_mbar", 1000.0)
        self.declare_parameter("min_mbar", -1000.0)
        self.channel = int(self.get_parameter("channel").value)
        poll_ms = int(self.get_parameter("poll_ms").value)

        self.enabled = False
        self.commanded_mbar = IDLE_MBAR
        self.pressure_min = self.pressure_max = IDLE_MBAR
        self._sdk_initialized = False
        self._faulted = False

        self.pub_measured = self.create_publisher(Float32, TOPIC_PRESSURE_MEASURED, 10)
        # Latched: the logger must see the applied pressure even if it starts
        # after the command that set it.
        self.pub_target = self.create_publisher(
            Float32, TOPIC_PRESSURE_TARGET, latched_qos()
        )

        self._connect()

        self.create_subscription(
            Float32, TOPIC_PRESSURE_MBAR, self._on_pressure_cmd, 10
        )

        self.timer = self.create_timer(poll_ms / 1000.0, self._poll_measured)

    # ── Device setup ───────────────────────────────────────────────────────
    def _connect(self):
        try:
            self.get_logger().info("Detecting Fluigent controller...")
            serials, types = fgt_detect()
            if not serials:
                raise RuntimeError("no Fluigent controller detected")

            # Init can allocate a partial SDK session before reporting failure.
            self._sdk_initialized = True
            require_sdk_ok(fgt_init(serials), "fgt_init")
            status, lower, upper = fgt_get_pressureRange(self.channel, get_error=True)
            require_sdk_ok(status, "fgt_get_pressureRange")
            if (not all(math.isfinite(v) for v in (lower, upper))
                    or not lower <= 0 <= upper or lower == upper):
                raise RuntimeError(f"invalid device pressure range: {lower}, {upper}")
            self.pressure_min, self.pressure_max = float(lower), float(upper)
            self._safe_limits()  # Reject malformed parameter envelopes on connect.
            self.enabled = True
            if not self._write_pressure(IDLE_MBAR):
                raise RuntimeError("controller did not acknowledge startup vent")
            self.get_logger().info(
                f"Pressure channel {self.channel} range: {lower:.1f} .. {upper:.1f} mbar"
            )
        except Exception as exc:
            self.enabled = False
            self._faulted = True
            self.get_logger().error(f"Fluigent controller unavailable: {exc}")
            self._close_sdk()

    # ── Command handling ───────────────────────────────────────────────────
    def _safe_limits(self):
        """Device range, tightened by the optional parameter envelope."""
        configured = (float(self.get_parameter("min_mbar").value),
                      float(self.get_parameter("max_mbar").value))
        if not all(math.isfinite(v) for v in configured) or configured[0] > configured[1]:
            raise RuntimeError(f"invalid pressure envelope: {configured}")
        lower = max(configured[0], self.pressure_min)
        upper = min(configured[1], self.pressure_max)
        if lower > upper:
            raise RuntimeError("pressure envelope does not intersect device range")
        return lower, upper

    def _on_pressure_cmd(self, msg: Float32):
        requested = float(msg.data)
        if not math.isfinite(requested):
            self.get_logger().warn(f"Ignoring non-finite pressure {requested}")
            return

        self._apply_pressure(requested)

    def _apply_pressure(self, requested: float):
        try:
            lower, upper = self._safe_limits()
        except Exception as exc:
            self.get_logger().error(str(exc))
            self._faulted = True
            self._write_pressure(IDLE_MBAR)
            return
        # Venting is always permitted, whatever the envelope. An operator may
        # legitimately configure a wholly negative window (say -80..-20 mbar for
        # a seal); clamping a vent into that window would leave the pipette
        # pressurised at exactly the moment it must not be.
        if requested == IDLE_MBAR:
            target = IDLE_MBAR
        else:
            target = clamp(requested, lower, upper)
        if target != requested:
            self.get_logger().warn(
                f"Pressure {requested:+.1f} mbar clamped to {target:+.1f} mbar "
                f"(limits {lower:+.1f} .. {upper:+.1f})"
            )

        if self._write_pressure(target):
            self.get_logger().info(f"Pressure set to {target:+.1f} mbar")

    # ── Device I/O ─────────────────────────────────────────────────────────
    def _write_pressure(self, mbar):
        """Write to the device and announce what was applied. False on failure."""
        if not self.enabled or (self._faulted and mbar != IDLE_MBAR):
            self.get_logger().warn(
                f"Fluigent not connected; dropping {mbar:+.1f} mbar command"
            )
            return False
        try:
            target = float(mbar)
            if not math.isfinite(target):
                raise ValueError("non-finite setpoint")
            require_sdk_ok(fgt_set_pressure(self.channel, target), "fgt_set_pressure")
        except Exception as e:
            self._faulted = True
            self.get_logger().error(f"fgt_set_pressure failed; nonzero commands latched off: {e}")
            if mbar != IDLE_MBAR:
                self._write_pressure(IDLE_MBAR)
            return False

        # Only announce a target the device really accepted.
        self.commanded_mbar = target
        self.pub_target.publish(Float32(data=target))
        return True

    def _poll_measured(self):
        if not self.enabled:
            return
        try:
            status, measured = fgt_get_pressure(self.channel, get_error=True)
            require_sdk_ok(status, "fgt_get_pressure")
            measured = float(measured)
            if not math.isfinite(measured):
                raise RuntimeError("non-finite pressure readback")
            self.pub_measured.publish(Float32(data=measured))
        except Exception as exc:
            self.get_logger().error(f"Pressure readback failed; venting and latching fault: {exc}")
            if not self._faulted:
                self._faulted = True
                self._write_pressure(IDLE_MBAR)

    def _close_sdk(self):
        if not self._sdk_initialized:
            return
        try:
            require_sdk_ok(fgt_set_pressure(self.channel, IDLE_MBAR), "shutdown vent")
            time.sleep(0.5)
        except Exception as exc:
            self.get_logger().error(f"Could not confirm pressure vent: {exc}")
        try:
            require_sdk_ok(fgt_close(), "fgt_close")
        except Exception as exc:
            self.get_logger().error(f"Could not close Fluigent SDK: {exc}")
        self._sdk_initialized = False
        self.enabled = False

    def destroy_node(self):
        self._close_sdk()
        super().destroy_node()


def main():
    rclpy.init()
    node = PressureNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl+C, or SIGTERM from `ros2 launch` tearing the system down.
        pass
    finally:
        # destroy_node() vents to 0 mbar and closes the SDK, so it must run even
        # on Ctrl+C. rclpy's SIGINT handler may already have shut the context
        # down; calling shutdown() again would raise and mask the vent.
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

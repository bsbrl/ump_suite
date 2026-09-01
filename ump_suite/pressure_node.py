"""ROS2 driver for the Fluigent push-pull pressure controller (LineUP).

The pressure is commanded as an exact value in mbar on a single topic:

    /pressure/mbar = -20.0  ->  fgt_set_pressure(channel, -20.0)   (pull)
    /pressure/mbar =  50.0  ->  fgt_set_pressure(channel,  50.0)   (push)
    /pressure/mbar =   0.0  ->  vented

Whatever arrives is clamped to the range the controller reports for its channel,
so a mistyped or out-of-range value cannot exceed the hardware limits.

The topic is latched, so this node picks up the last commanded pressure even if
it restarts. Because `/pressure/mbar` has more than one publisher (the GUI and
the rollout client), a restart delivers the last sample from EACH of them in an
order ROS does not define. Startup therefore collects that history for
`startup_grace_s` and only restores it when every publisher agrees; a conflict
holds at 0 mbar and says so, rather than resurrecting whichever stale command
happened to arrive last.

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

# Fallback device limits, used only if the controller will not report its range.
FALLBACK_RANGE_MBAR = (-1000.0, 1000.0)

# How long to collect latched command history at startup before acting on it.
# See _resolve_startup_commands for why a window is needed at all.
DEFAULT_STARTUP_GRACE_S = 1.0


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
        # Startup window during which latched commands are collected rather than
        # applied immediately. 0 disables the window and restores the historical
        # apply-whatever-arrives-last behaviour.
        self.declare_parameter("startup_grace_s", DEFAULT_STARTUP_GRACE_S)

        self.channel = int(self.get_parameter("channel").value)
        poll_ms = int(self.get_parameter("poll_ms").value)

        self.enabled = False
        self.commanded_mbar = IDLE_MBAR
        self.pressure_min, self.pressure_max = FALLBACK_RANGE_MBAR

        # `/pressure/mbar` is latched and has more than one publisher (the GUI
        # and the rollout client). On startup this node therefore receives the
        # last sample from EACH of them, in an order ROS does not define, so
        # "apply whatever arrives last" can resurrect a stale command. Collect
        # them instead and only act once the picture is unambiguous.
        grace = float(self.get_parameter("startup_grace_s").value)
        self._startup_grace_s = grace if math.isfinite(grace) and grace > 0 else 0.0
        self._startup_open = self._startup_grace_s > 0.0
        self._startup_commands = []
        self._startup_timer = None

        self.pub_measured = self.create_publisher(Float32, TOPIC_PRESSURE_MEASURED, 10)
        # Latched: the logger must see the applied pressure even if it starts
        # after the command that set it.
        self.pub_target = self.create_publisher(
            Float32, TOPIC_PRESSURE_TARGET, latched_qos()
        )

        self._connect()

        self.create_subscription(
            Float32, TOPIC_PRESSURE_MBAR, self._on_pressure_cmd, latched_qos()
        )

        if self._startup_open:
            self._startup_timer = self.create_timer(
                self._startup_grace_s, self._resolve_startup_commands
            )

        self.timer = self.create_timer(poll_ms / 1000.0, self._poll_measured)

    # ── Device setup ───────────────────────────────────────────────────────
    def _connect(self):
        try:
            self.get_logger().info("Detecting Fluigent controller...")
            serials, types = fgt_detect()
            if not serials:
                raise RuntimeError("no Fluigent controller detected")

            fgt_init(serials)
            self.get_logger().info(
                f"Fluigent initialized: serials={serials}, types={types}"
            )

            try:
                self.pressure_min, self.pressure_max = fgt_get_pressureRange(
                    self.channel
                )
            except Exception as e:
                self.get_logger().warn(
                    f"Could not read pressure range, using "
                    f"{FALLBACK_RANGE_MBAR} mbar: {e}"
                )
                self.pressure_min, self.pressure_max = FALLBACK_RANGE_MBAR

            self.enabled = True
            self.get_logger().info(
                f"Pressure channel {self.channel} range: "
                f"{self.pressure_min:.1f} .. {self.pressure_max:.1f} mbar"
            )

            # Start from a known, harmless pressure.
            self._write_pressure(IDLE_MBAR)
        except Exception as e:
            self.enabled = False
            self.get_logger().error(f"Fluigent controller not available: {e}")

    # ── Command handling ───────────────────────────────────────────────────
    def _safe_limits(self):
        """Device range, tightened by the optional parameter envelope."""
        lower = max(float(self.get_parameter("min_mbar").value), self.pressure_min)
        upper = min(float(self.get_parameter("max_mbar").value), self.pressure_max)
        # Guard against a reversed envelope leaving no valid pressure at all.
        if lower > upper:
            return 0.0, 0.0
        return lower, upper

    def _resolve_startup_commands(self):
        """Decide what the latched startup history actually means.

        Unanimous history is the intended single-publisher restore and is
        applied. Conflicting history is genuinely ambiguous - it is exactly the
        case where applying the wrong one re-pressurises a pipette the operator
        believes is vented - so this fails closed at the idle pressure and says
        which values it saw.
        """
        if self._startup_timer is not None:
            self._startup_timer.cancel()
            self._startup_timer = None
        self._startup_open = False

        seen = list(self._startup_commands)
        self._startup_commands.clear()
        if not seen:
            return

        distinct = sorted(set(seen))
        if len(distinct) == 1:
            self.get_logger().info(
                f"Restoring latched pressure {distinct[0]:+.1f} mbar from startup history"
            )
            self._apply_pressure(distinct[0])
            return

        self.get_logger().warn(
            "Conflicting latched pressure commands at startup "
            f"({', '.join(f'{v:+.1f}' for v in distinct)} mbar) - `/pressure/mbar` has "
            "more than one publisher and their order is undefined. Holding "
            f"{IDLE_MBAR:+.1f} mbar; send the intended pressure explicitly."
        )

    def _on_pressure_cmd(self, msg: Float32):
        requested = float(msg.data)
        if not math.isfinite(requested):
            self.get_logger().warn(f"Ignoring non-finite pressure {requested}")
            return

        if self._startup_open:
            # Collected, not applied. _resolve_startup_commands decides.
            self._startup_commands.append(requested)
            self.get_logger().info(
                f"Deferring pressure {requested:+.1f} mbar until the startup "
                "window closes"
            )
            return

        self._apply_pressure(requested)

    def _apply_pressure(self, requested: float):
        lower, upper = self._safe_limits()
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
        if not self.enabled:
            self.get_logger().warn(
                f"Fluigent not connected; dropping {mbar:+.1f} mbar command"
            )
            return False
        try:
            target = float(mbar)
            fgt_set_pressure(self.channel, target)
        except Exception as e:
            self.get_logger().error(f"fgt_set_pressure failed: {e}")
            return False

        # Only announce a target the device really accepted.
        self.commanded_mbar = target
        self.pub_target.publish(Float32(data=target))
        return True

    def _poll_measured(self):
        if not self.enabled:
            return
        try:
            self.pub_measured.publish(
                Float32(data=float(fgt_get_pressure(self.channel)))
            )
        except Exception as e:
            self.get_logger().warn(f"fgt_get_pressure failed: {e}")

    # ── Shutdown ───────────────────────────────────────────────────────────
    def destroy_node(self):
        if self.enabled:
            try:
                # Vent before closing so the pipette is not left pressurized.
                fgt_set_pressure(self.channel, IDLE_MBAR)
                time.sleep(0.5)
            except Exception:
                pass
            try:
                fgt_close()
            except Exception:
                pass
            self.enabled = False
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

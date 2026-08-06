"""ROS2 driver for the Fluigent push-pull pressure controller (LineUP).

Pressure is commanded as a *binary state* rather than a value:

    /pressure/state_cmd = True   ->  apply the positive setpoint
    /pressure/state_cmd = False  ->  apply the negative setpoint

The two setpoints arrive on their own latched topics (published by the GUI),
so whoever is driving the rig -- operator or policy -- only has to decide
"push" or "pull" and this node resolves that against the values currently
dialed in. That keeps the logged/predicted pressure channel binary while the
actual mbar values stay adjustable at run time.

The node publishes the state it actually applied on /pressure/state -- positive,
negative, or vented -- and the measured pressure on /pressure/measured_mbar.

Venting (0 mbar) is exposed as the /pressure/vent service rather than a third
value on the command topic, so the binary action space a policy drives stays
binary while an operator still has a one-click way back to neutral.

Requires the Fluigent Python SDK (`fluigent_sdk`), which bundles its own
libfgt_SDK.so, so no system library setup is needed.
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int8
from std_srvs.srv import Trigger

from Fluigent.SDK import (
    fgt_close,
    fgt_detect,
    fgt_get_pressure,
    fgt_get_pressureRange,
    fgt_init,
    fgt_set_pressure,
)

from .ros_interfaces import (
    PRESSURE_STATE_NEGATIVE,
    PRESSURE_STATE_POSITIVE,
    PRESSURE_STATE_VENTED,
    SRV_PRESSURE_VENT,
    TOPIC_PRESSURE_MEASURED,
    TOPIC_PRESSURE_NEG_MBAR,
    TOPIC_PRESSURE_POS_MBAR,
    TOPIC_PRESSURE_STATE,
    TOPIC_PRESSURE_STATE_CMD,
    latched_qos,
)


# Pressure held while the node is connected but has not been told a state yet,
# and on shutdown.
IDLE_MBAR = 0.0

# Fallback device limits, used only if the controller will not report its range.
FALLBACK_RANGE_MBAR = (-1000.0, 1000.0)


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


class PressureNode(Node):
    def __init__(self):
        super().__init__("pressure_node")

        self.declare_parameter("channel", 0)
        self.declare_parameter("poll_ms", 100)
        # Setpoints used until the GUI publishes its own. Deliberately small:
        # these are applied to a patch pipette.
        self.declare_parameter("positive_mbar", 20.0)
        self.declare_parameter("negative_mbar", -20.0)
        # Optional safety envelope, tightened against the device range below.
        # Leave at the device limits to allow the controller's full travel.
        self.declare_parameter("max_positive_mbar", 1000.0)
        self.declare_parameter("min_negative_mbar", -1000.0)

        self.channel = int(self.get_parameter("channel").value)
        poll_ms = int(self.get_parameter("poll_ms").value)

        self.enabled = False
        # Starts vented: 0 mbar is applied on connect, which is neither push
        # nor pull, so it must not be logged as either.
        self.state = PRESSURE_STATE_VENTED
        self.pressure_min, self.pressure_max = FALLBACK_RANGE_MBAR

        self.positive_mbar = 0.0
        self.negative_mbar = 0.0

        self.pub_state = self.create_publisher(
            Int8, TOPIC_PRESSURE_STATE, latched_qos()
        )
        self.pub_measured = self.create_publisher(Float32, TOPIC_PRESSURE_MEASURED, 10)

        self._connect()

        # Clamp the parameter defaults only once the device range is known.
        self.positive_mbar = self._clamp_positive(
            float(self.get_parameter("positive_mbar").value)
        )
        self.negative_mbar = self._clamp_negative(
            float(self.get_parameter("negative_mbar").value)
        )

        self.create_subscription(
            Bool, TOPIC_PRESSURE_STATE_CMD, self._on_state_cmd, 10
        )
        self.create_subscription(
            Float32, TOPIC_PRESSURE_POS_MBAR, self._on_positive_mbar, latched_qos()
        )
        self.create_subscription(
            Float32, TOPIC_PRESSURE_NEG_MBAR, self._on_negative_mbar, latched_qos()
        )
        self.create_service(Trigger, SRV_PRESSURE_VENT, self._on_vent)

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

            # Start from a known, harmless pressure and say so, so subscribers
            # that join later know the channel is vented rather than unknown.
            self._write_pressure(IDLE_MBAR)
            self._publish_state(PRESSURE_STATE_VENTED)
        except Exception as e:
            self.enabled = False
            self.get_logger().error(f"Fluigent controller not available: {e}")

    # ── Setpoint handling ──────────────────────────────────────────────────
    def _clamp_positive(self, value):
        """Positive setpoints are clamped to [0, max]: never pull on 'push'."""
        upper = min(
            float(self.get_parameter("max_positive_mbar").value), self.pressure_max
        )
        return clamp(float(value), 0.0, max(0.0, upper))

    def _clamp_negative(self, value):
        """Negative setpoints are clamped to [min, 0]: never push on 'pull'."""
        lower = max(
            float(self.get_parameter("min_negative_mbar").value), self.pressure_min
        )
        return clamp(float(value), min(0.0, lower), 0.0)

    def _on_positive_mbar(self, msg: Float32):
        self.positive_mbar = self._clamp_positive(msg.data)
        # Editing the value while 'positive' is already applied should take
        # effect immediately, not on the next button press. Venting is not
        # disturbed by a setpoint edit.
        if self.state == PRESSURE_STATE_POSITIVE:
            self._apply_state(True)

    def _on_negative_mbar(self, msg: Float32):
        self.negative_mbar = self._clamp_negative(msg.data)
        if self.state == PRESSURE_STATE_NEGATIVE:
            self._apply_state(False)

    def _on_state_cmd(self, msg: Bool):
        self._apply_state(bool(msg.data))

    def _publish_state(self, state):
        self.state = state
        self.pub_state.publish(Int8(data=state))

    def _apply_state(self, positive):
        target = self.positive_mbar if positive else self.negative_mbar
        if not self._write_pressure(target):
            return

        self._publish_state(
            PRESSURE_STATE_POSITIVE if positive else PRESSURE_STATE_NEGATIVE
        )
        self.get_logger().info(
            f"Pressure state {'POSITIVE' if positive else 'NEGATIVE'}: "
            f"{target:+.1f} mbar"
        )

    def _on_vent(self, _req, res):
        if self._write_pressure(IDLE_MBAR):
            self._publish_state(PRESSURE_STATE_VENTED)
            res.success = True
            res.message = f"Vented to {IDLE_MBAR:.1f} mbar."
            self.get_logger().info(res.message)
        else:
            res.success = False
            res.message = "Vent failed: Fluigent controller not available."
        return res

    # ── Device I/O ─────────────────────────────────────────────────────────
    def _write_pressure(self, mbar):
        if not self.enabled:
            self.get_logger().warn(
                f"Fluigent not connected; dropping {mbar:+.1f} mbar command"
            )
            return False
        try:
            fgt_set_pressure(self.channel, clamp(
                float(mbar), self.pressure_min, self.pressure_max
            ))
            return True
        except Exception as e:
            self.get_logger().error(f"fgt_set_pressure failed: {e}")
            return False

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
    except KeyboardInterrupt:
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

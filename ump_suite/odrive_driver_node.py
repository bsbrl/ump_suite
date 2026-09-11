"""
ROS2 driver for an ODrive single-axis motor used as the manipulator's focusing knob.

The driver accepts absolute target encoder counts on /motor/target_counts and
publishes the current encoder counts on /motor/live_counts. Position control
is implemented in software here as a bang-bang velocity command on top of the
ODrive's velocity controller.
"""

import time

import odrive
import rclpy
from odrive.enums import (
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    AXIS_STATE_IDLE,
    CONTROL_MODE_VELOCITY_CONTROL,
)
from rclpy.node import Node
from std_msgs.msg import Int32

from .ros_interfaces import TOPIC_MOTOR_LIVE, TOPIC_MOTOR_TGT


class ODriveDriverNode(Node):
    def __init__(self):
        super().__init__("odrive_driver_node")

        self.declare_parameter("poll_ms", 50)
        self.declare_parameter("goto_speed_turns_s", 0.5)
        self.declare_parameter("deadband_counts", 200)

        poll_ms = int(self.get_parameter("poll_ms").value)

        self.motor_target = 0
        self.axis = None
        self.enabled = False

        self.pub_live = self.create_publisher(Int32, TOPIC_MOTOR_LIVE, 10)
        self.sub_tgt = self.create_subscription(Int32, TOPIC_MOTOR_TGT, self.on_target, 10)

        self._connect()
        self.timer = self.create_timer(poll_ms / 1000.0, self.loop)

    def _connect(self):
        try:
            self.get_logger().info("Connecting to ODrive...")
            self.odrv = odrive.find_any(timeout=10)
            self.axis = self.odrv.axis0
            # Set the mode and zero input while idle, before energizing the axis.
            self.axis.requested_state = AXIS_STATE_IDLE
            time.sleep(0.3)
            self.odrv.clear_errors()
            self.axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            self.axis.controller.input_vel = 0.0
            self.motor_target = int(self.axis.encoder.shadow_count)
            self.axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
            time.sleep(0.7)
            if self.axis.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                raise RuntimeError("axis did not enter closed-loop control")
            self.enabled = True
            self.get_logger().info(
                f"ODrive connected. axis0 current_counts={self.motor_target}"
            )
        except Exception as e:
            self.enabled = False
            self._stop_axis()
            self.get_logger().error(f"ODrive not available: {e}")

    def _shadow_count(self):
        return int(self.axis.encoder.shadow_count)

    def on_target(self, msg: Int32):
        self.motor_target = int(msg.data)

    def loop(self):
        if not self.enabled or self.axis is None:
            return

        goto_speed = float(self.get_parameter("goto_speed_turns_s").value)
        deadband = int(self.get_parameter("deadband_counts").value)

        try:
            pos = self._shadow_count()
            self.pub_live.publish(Int32(data=pos))

            err = self.motor_target - pos
            if abs(err) <= deadband:
                self.axis.controller.input_vel = 0.0
            else:
                self.axis.controller.input_vel = goto_speed * (1.0 if err > 0 else -1.0)
        except Exception as e:
            # Do not leave a previously commanded velocity running. If the read
            # or write failed we no longer know the state, so command zero and
            # latch out of the loop rather than coasting on a stale setpoint.
            self.get_logger().error(f"ODrive loop error, commanding zero: {e}")
            self._stop_axis()
            self.enabled = False

    def _stop_axis(self):
        if self.axis is None:
            return
        # An input write can fail independently of the state request. Always
        # attempt both, including after partial initialization.
        try:
            self.axis.controller.input_vel = 0.0
        except Exception as exc:
            self.get_logger().error(f"ODrive zero-velocity write failed: {exc}")
        try:
            self.axis.requested_state = AXIS_STATE_IDLE
        except Exception as exc:
            self.get_logger().error(f"ODrive idle request failed: {exc}")

    def destroy_node(self):
        self.enabled = False
        self._stop_axis()
        super().destroy_node()


def main():
    rclpy.init()
    node = ODriveDriverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

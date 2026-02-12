import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

import odrive
from odrive.enums import AXIS_STATE_CLOSED_LOOP_CONTROL, AXIS_STATE_IDLE, CONTROL_MODE_VELOCITY_CONTROL

from .ros_interfaces import TOPIC_MOTOR_TGT, TOPIC_MOTOR_JOG, TOPIC_MOTOR_LIVE

class ODriveDriverNode(Node):
    def __init__(self):
        super().__init__("odrive_driver_node")

        self.declare_parameter("poll_ms", 50)
        self.declare_parameter("jog_speed_turns_s", 0.5)
        self.declare_parameter("goto_speed_turns_s", 0.5)
        self.declare_parameter("deadband_counts", 200)

        poll_ms = int(self.get_parameter("poll_ms").value)

        self.motor_target = 0
        self.jog_dir = 0

        self.pub_live = self.create_publisher(Int32, TOPIC_MOTOR_LIVE, 10)
        self.sub_tgt = self.create_subscription(Int32, TOPIC_MOTOR_TGT, self.on_target, 10)
        self.sub_jog = self.create_subscription(Int32, TOPIC_MOTOR_JOG, self.on_jog, 10)

        self.axis = None
        self.enabled = False

        self._connect()
        self.timer = self.create_timer(poll_ms / 1000.0, self.loop)

    def _connect(self):
        try:
            self.get_logger().info("Connecting to ODrive...")
            self.odrv = odrive.find_any()
            self.axis = self.odrv.axis0
            self.odrv.clear_errors()
            time.sleep(0.3)

            self.axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
            time.sleep(0.7)

            self.axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            self.axis.controller.input_vel = 0.0

            cur = int(self.axis.encoder.shadow_count)
            self.motor_target = cur
            self.enabled = True
            self.get_logger().info(f"ODrive connected. axis0 current_counts={cur}")
        except Exception as e:
            self.enabled = False
            self.axis = None
            self.get_logger().error(f"ODrive not available: {e}")

    def on_target(self, msg: Int32):
        self.motor_target = int(msg.data)

    def on_jog(self, msg: Int32):
        d = int(msg.data)
        self.jog_dir = -1 if d < 0 else (1 if d > 0 else 0)
        if self.enabled and self.axis is not None:
            try:
                self.motor_target = int(self.axis.encoder.shadow_count)
            except Exception:
                pass

    def loop(self):
        if not self.enabled or self.axis is None:
            return

        jog_speed = float(self.get_parameter("jog_speed_turns_s").value)
        goto_speed = float(self.get_parameter("goto_speed_turns_s").value)
        deadband = int(self.get_parameter("deadband_counts").value)

        try:
            pos = int(self.axis.encoder.shadow_count)
            self.pub_live.publish(Int32(data=pos))

            if self.jog_dir != 0:
                self.axis.controller.input_vel = self.jog_dir * jog_speed
                self.motor_target = pos
            else:
                err = self.motor_target - pos
                if abs(err) <= deadband:
                    self.axis.controller.input_vel = 0.0
                else:
                    self.axis.controller.input_vel = goto_speed * (1.0 if err > 0 else -1.0)
        except Exception as e:
            self.get_logger().warn(f"ODrive loop error: {e}")

    def destroy_node(self):
        try:
            if self.axis is not None:
                self.axis.controller.input_vel = 0.0
                self.axis.requested_state = AXIS_STATE_IDLE
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = ODriveDriverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


# Add focal length calibration support in this node later if necessary
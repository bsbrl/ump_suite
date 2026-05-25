"""ROS2 driver for one Sensapex UMP micromanipulator stage.

The Sensapex SDK reports positions in unsigned device units. This node exposes
those same absolute device coordinates on the ROS topics and forwards absolute
targets directly to the device. Topic names are built from a `topic_prefix`
parameter, allowing one process per device.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import Trigger

from sensapex import UMP


class UMPDriverNode(Node):
    def __init__(self, node_name="ump_driver_node", *, parameter_overrides=None):
        super().__init__(node_name, parameter_overrides=parameter_overrides or [])

        self.declare_parameter("device_id", 1)
        self.declare_parameter("poll_ms", 50)
        self.declare_parameter("topic_prefix", "ump")

        device_id = int(self.get_parameter("device_id").value)
        poll_ms = int(self.get_parameter("poll_ms").value)
        prefix = self.get_parameter("topic_prefix").value

        self.get_logger().info(
            f"Connecting to Sensapex UMP device {device_id} (prefix: /{prefix})..."
        )
        self.ump = UMP.get_ump()
        self.stage = self.ump.get_device(device_id)
        self.get_logger().info(f"Connected to UMP device {device_id}")

        self.pub_live = self.create_publisher(Int32MultiArray, f"/{prefix}/live", 10)
        self.sub_target = self.create_subscription(
            Int32MultiArray, f"/{prefix}/target", self.on_target, 10
        )
        self.srv_zero = self.create_service(
            Trigger, f"/{prefix}/calibrate_zero", self.on_zero
        )
        self.timer = self.create_timer(poll_ms / 1000.0, self.poll_live)

    def _read_absolute_pos(self):
        """Return the current [x, y, z, d] position in absolute device counts."""
        pos = self.stage.get_pos()
        return [int(pos[i]) for i in range(4)]

    def poll_live(self):
        try:
            msg = Int32MultiArray()
            msg.data = self._read_absolute_pos()
            self.pub_live.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"UMP live read error: {e}")

    def on_target(self, msg: Int32MultiArray):
        if len(msg.data) < 5:
            self.get_logger().warn("UMP target msg requires [x,y,z,d,speed]")
            return
        try:
            x, y, z, d, speed = (int(v) for v in msg.data[:5])
            self.stage.goto_pos([x, y, z, d], speed=speed)
        except Exception as e:
            self.get_logger().error(f"UMP goto_pos error: {e}")

    def on_zero(self, _req, res):
        try:
            self.stage.calibrate_zero_position()
            res.success = True
            res.message = "Zero calibrated at current position."
        except Exception as e:
            res.success = False
            res.message = f"Calibrate zero error: {e}"
        return res


def main():
    rclpy.init()
    node = UMPDriverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main_dual():
    """Run two UMP driver nodes in a single process so they share one SDK instance."""
    rclpy.init()

    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter

    node1 = UMPDriverNode("ump_driver_node", parameter_overrides=[
        Parameter("device_id", value=1),
        Parameter("poll_ms", value=50),
        Parameter("topic_prefix", value="ump"),
    ])
    node2 = UMPDriverNode("ump2_driver_node", parameter_overrides=[
        Parameter("device_id", value=2),
        Parameter("poll_ms", value=50),
        Parameter("topic_prefix", value="ump2"),
    ])

    executor = MultiThreadedExecutor()
    executor.add_node(node1)
    executor.add_node(node2)
    try:
        executor.spin()
    finally:
        node1.destroy_node()
        node2.destroy_node()
        rclpy.shutdown()

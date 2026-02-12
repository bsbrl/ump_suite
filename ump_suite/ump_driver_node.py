import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import Trigger

from sensapex import UMP
from .ros_interfaces import TOPIC_UMP_TARGET, TOPIC_UMP_LIVE, SRV_ZERO

CENTER_OFFSET = 10000

def device_to_center(v: int) -> int:
    return int(v - CENTER_OFFSET)

def center_to_device(v: int) -> int:
    return int(v + CENTER_OFFSET)

class UMPDriverNode(Node):
    def __init__(self):
        super().__init__("ump_driver_node")

        self.declare_parameter("device_id", 1)
        self.declare_parameter("poll_ms", 50)

        device_id = int(self.get_parameter("device_id").value)
        poll_ms = int(self.get_parameter("poll_ms").value)

        self.get_logger().info("Connecting to Sensapex UMP...")
        self.ump = UMP.get_ump()
        self.stage = self.ump.get_device(device_id)
        self.get_logger().info(f"Connected to UMP device {device_id}")

        self.pub_live = self.create_publisher(Int32MultiArray, TOPIC_UMP_LIVE, 10)
        self.sub_target = self.create_subscription(Int32MultiArray, TOPIC_UMP_TARGET, self.on_target, 10)

        self.srv_zero = self.create_service(Trigger, SRV_ZERO, self.on_zero)

        self.timer = self.create_timer(poll_ms / 1000.0, self.poll_live)

    def poll_live(self):
        try:
            pos = self.stage.get_pos()  # [X,Y,Z,D,...]
            centered = [device_to_center(int(pos[i])) for i in range(4)]
            msg = Int32MultiArray()
            msg.data = centered
            self.pub_live.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"UMP live read error: {e}")

    def on_target(self, msg: Int32MultiArray):
        try:
            if len(msg.data) < 5:
                self.get_logger().warn("UMP target msg requires [x,y,z,d,speed]")
                return
            x, y, z, d, speed = [int(v) for v in msg.data[:5]]
            dev = [center_to_device(v) for v in (x, y, z, d)]
            self.stage.goto_pos(dev, speed=int(speed))
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
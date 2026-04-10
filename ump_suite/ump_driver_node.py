import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Int32
from std_srvs.srv import Trigger

from sensapex import UMP

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
        self.declare_parameter("default_speed", 1000)
        self.declare_parameter("topic_prefix", "ump")

        device_id = int(self.get_parameter("device_id").value)
        poll_ms = int(self.get_parameter("poll_ms").value)
        self.current_speed = int(self.get_parameter("default_speed").value)
        prefix = self.get_parameter("topic_prefix").value

        topic_live   = f"/{prefix}/live"
        topic_target = f"/{prefix}/target"
        topic_delta  = f"/{prefix}/delta"
        topic_speed  = f"/{prefix}/target_speed"
        srv_zero     = f"/{prefix}/calibrate_zero"

        self.get_logger().info(f"Connecting to Sensapex UMP device {device_id} (prefix: /{prefix})...")
        self.ump = UMP.get_ump()
        self.stage = self.ump.get_device(device_id)
        self.get_logger().info(f"Connected to UMP device {device_id}")

        self.pub_live = self.create_publisher(Int32MultiArray, topic_live, 10)

        self.sub_target = self.create_subscription(
            Int32MultiArray, topic_target, self.on_target, 10
        )
        self.sub_delta = self.create_subscription(
            Int32MultiArray, topic_delta, self.on_delta, 10
        )
        self.sub_speed = self.create_subscription(
            Int32, topic_speed, self.on_speed, 10
        )

        self.srv_zero = self.create_service(Trigger, srv_zero, self.on_zero)
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

    def on_speed(self, msg: Int32):
        self.current_speed = int(msg.data)

    def on_target(self, msg: Int32MultiArray):
        try:
            if len(msg.data) < 5:
                self.get_logger().warn("UMP target msg requires [x,y,z,d,speed]")
                return

            x, y, z, d, speed = [int(v) for v in msg.data[:5]]
            self.current_speed = int(speed)

            dev = [center_to_device(v) for v in (x, y, z, d)]
            self.stage.goto_pos(dev, speed=int(speed))
        except Exception as e:
            self.get_logger().error(f"UMP goto_pos error: {e}")

    def on_delta(self, msg: Int32MultiArray):
        try:
            if len(msg.data) < 4:
                self.get_logger().warn("UMP delta msg requires [dx,dy,dz,dd]")
                return

            dx, dy, dz, dd = [int(v) for v in msg.data[:4]]

            pos = self.stage.get_pos()
            cur_centered = [device_to_center(int(pos[i])) for i in range(4)]

            tgt_centered = [
                cur_centered[0] + dx,
                cur_centered[1] + dy,
                cur_centered[2] + dz,
                cur_centered[3] + dd,
            ]

            dev = [center_to_device(v) for v in tgt_centered]
            self.stage.goto_pos(dev, speed=int(self.current_speed))

        except Exception as e:
            self.get_logger().error(f"UMP delta goto_pos error: {e}")

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
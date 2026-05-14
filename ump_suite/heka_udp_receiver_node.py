# heka_udp_receiver_node.py
import math
import socket

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class HekaUdpReceiverNode(Node):
    def __init__(self):
        super().__init__("heka_udp_receiver_node")

        self.declare_parameter("port", 5005)
        port = int(self.get_parameter("port").value)

        self.pub_res = self.create_publisher(Float32, "/heka/resistance_mohm", 10)
        self.pub_mon = self.create_publisher(Float32, "/heka/monitor_v", 10)
        self.pub_step = self.create_publisher(Float32, "/heka/monitor_step_v", 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)

        self.timer = self.create_timer(0.01, self.loop)
        self.get_logger().info(f"Listening for HEKA UDP on port {port}")

    def loop(self):
        try:
            data, addr = self.sock.recvfrom(1024)
        except BlockingIOError:
            return

        try:
            text = data.decode("utf-8").strip()
            parts = text.split(",")

            # Expected:
            # timestamp, mean_voltage_V, monitor_step_V, resistance_MOhm
            if len(parts) < 4:
                self.get_logger().warn(f"Short HEKA packet from {addr}: {text}")
                return

            mean_v = float(parts[1])
            step_v = float(parts[2])
            resistance_mohm = float(parts[3])

            self.pub_mon.publish(Float32(data=mean_v))
            self.pub_step.publish(Float32(data=step_v))

            if math.isfinite(resistance_mohm):
                self.pub_res.publish(Float32(data=resistance_mohm))

        except Exception as e:
            self.get_logger().warn(f"Bad HEKA UDP packet from {addr}: {e}")


def main():
    rclpy.init()
    node = HekaUdpReceiverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
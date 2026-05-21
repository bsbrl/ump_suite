# heka_udp_receiver_node.py
import math
import socket
import struct

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray

from .ros_interfaces import (
    TOPIC_HEKA_CURRENT_PA,
    TOPIC_HEKA_MONITOR_STEP_V,
    TOPIC_HEKA_MONITOR_V,
    TOPIC_HEKA_RESISTANCE,
    TOPIC_HEKA_VOLTAGE_RAW,
)


PACKET_MAGIC = b"HEKA1"
PACKET_HEADER = struct.Struct("<5sdfH")


class HekaUdpReceiverNode(Node):
    def __init__(self):
        super().__init__("heka_udp_receiver_node")

        self.declare_parameter("port", 5005)
        port = int(self.get_parameter("port").value)

        self.pub_res = self.create_publisher(Float32, TOPIC_HEKA_RESISTANCE, 10)
        self.pub_mon = self.create_publisher(Float32, TOPIC_HEKA_MONITOR_V, 10)
        self.pub_step = self.create_publisher(Float32, TOPIC_HEKA_MONITOR_STEP_V, 10)
        self.pub_voltage = self.create_publisher(
            Float32MultiArray, TOPIC_HEKA_VOLTAGE_RAW, 10
        )
        self.pub_current = self.create_publisher(
            Float32MultiArray, TOPIC_HEKA_CURRENT_PA, 10
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)

        self.timer = self.create_timer(0.01, self.loop)
        self.get_logger().info(f"Listening for HEKA UDP on port {port}")

    def loop(self):
        try:
            data, addr = self.sock.recvfrom(65535)
        except BlockingIOError:
            return

        try:
            if data.startswith(PACKET_MAGIC):
                self._handle_binary_packet(data, addr)
            else:
                self._handle_legacy_text_packet(data, addr)

        except Exception as e:
            self.get_logger().warn(f"Bad HEKA UDP packet from {addr}: {e}")

    def _handle_binary_packet(self, data, addr):
        if len(data) < PACKET_HEADER.size:
            self.get_logger().warn(f"Short HEKA binary packet from {addr}")
            return

        magic, _first_sample_time, sample_rate_hz, sample_count = (
            PACKET_HEADER.unpack_from(data, 0)
        )
        if magic != PACKET_MAGIC:
            self.get_logger().warn(f"Bad HEKA magic from {addr}: {magic!r}")
            return
        if sample_count <= 0:
            return

        expected_bytes = PACKET_HEADER.size + sample_count * 2 * 4
        if len(data) < expected_bytes:
            self.get_logger().warn(
                f"Short HEKA payload from {addr}: got {len(data)}, "
                f"expected {expected_bytes}"
            )
            return

        samples = np.frombuffer(
            data,
            dtype="<f4",
            count=sample_count * 2,
            offset=PACKET_HEADER.size,
        ).reshape(sample_count, 2)

        voltage_raw_v = samples[:, 0]
        current_pa = samples[:, 1]

        voltage_msg = Float32MultiArray()
        voltage_msg.data = [float(sample_rate_hz)] + voltage_raw_v.astype(float).tolist()
        self.pub_voltage.publish(voltage_msg)

        current_msg = Float32MultiArray()
        current_msg.data = [float(sample_rate_hz)] + current_pa.astype(float).tolist()
        self.pub_current.publish(current_msg)

        if voltage_raw_v.size:
            latest_voltage = float(voltage_raw_v[-1])
            if math.isfinite(latest_voltage):
                self.pub_mon.publish(Float32(data=latest_voltage))

    def _handle_legacy_text_packet(self, data, addr):
        text = data.decode("utf-8").strip()
        parts = text.split(",")

        # Legacy format:
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

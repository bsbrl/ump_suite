"""ROS2 driver for the Arduino-based pressure controller (two solenoids).

The Arduino firmware listens on a serial port for newline-terminated commands:
    S11 -> solenoid 1 ON      S10 -> solenoid 1 OFF
    S21 -> solenoid 2 ON      S20 -> solenoid 2 OFF

This node bridges those commands to ROS2:
  * subscribes to /pressure/solenoid{1,2}/cmd (std_msgs/Bool)
  * publishes the echoed state on /pressure/solenoid{1,2}/state (std_msgs/Bool)

The serial port defaults to /dev/ttyACM1 but can be overridden via the
`port` ROS parameter (see launch file).
"""

import threading

import rclpy
import serial
from rclpy.node import Node
from std_msgs.msg import Bool

from .ros_interfaces import (
    TOPIC_SOL1_CMD,
    TOPIC_SOL1_STATE,
    TOPIC_SOL2_CMD,
    TOPIC_SOL2_STATE,
)


class PressureNode(Node):
    def __init__(self):
        super().__init__("pressure_node")

        self.declare_parameter("port", "/dev/ttyACM1")
        self.declare_parameter("baud", 9600)
        self.declare_parameter("reconnect_s", 2.0)

        self.port = str(self.get_parameter("port").value)
        self.baud = int(self.get_parameter("baud").value)

        self._ser = None
        self._ser_lock = threading.Lock()
        self._stop = threading.Event()

        self.pub_sol1_state = self.create_publisher(Bool, TOPIC_SOL1_STATE, 10)
        self.pub_sol2_state = self.create_publisher(Bool, TOPIC_SOL2_STATE, 10)

        self.create_subscription(Bool, TOPIC_SOL1_CMD, self._on_sol1_cmd, 10)
        self.create_subscription(Bool, TOPIC_SOL2_CMD, self._on_sol2_cmd, 10)

        self._connect()

        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _connect(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
            self.get_logger().info(f"Arduino connected on {self.port} @ {self.baud}")
        except Exception as e:
            self._ser = None
            self.get_logger().error(f"Failed to open {self.port}: {e}")

    def _send(self, cmd: str):
        with self._ser_lock:
            if self._ser is None or not self._ser.is_open:
                self.get_logger().warn(f"Serial not open; dropping {cmd!r}")
                return
            try:
                self._ser.write((cmd + "\n").encode("ascii"))
                self._ser.flush()
            except Exception as e:
                self.get_logger().error(f"Serial write failed: {e}")
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None

    def _on_sol1_cmd(self, msg: Bool):
        self._send("S11" if msg.data else "S10")

    def _on_sol2_cmd(self, msg: Bool):
        self._send("S21" if msg.data else "S20")

    def _read_loop(self):
        reconnect_s = float(self.get_parameter("reconnect_s").value)
        while not self._stop.is_set():
            if self._ser is None or not self._ser.is_open:
                self._stop.wait(reconnect_s)
                if self._stop.is_set():
                    return
                self._connect()
                continue
            try:
                line = self._ser.readline().decode("ascii", errors="ignore").strip()
            except Exception as e:
                self.get_logger().warn(f"Serial read failed: {e}")
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
                continue

            if not line:
                continue

            self.get_logger().debug(f"Arduino: {line}")

            if line == "S1 ON":
                self.pub_sol1_state.publish(Bool(data=True))
            elif line == "S1 OFF":
                self.pub_sol1_state.publish(Bool(data=False))
            elif line == "S2 ON":
                self.pub_sol2_state.publish(Bool(data=True))
            elif line == "S2 OFF":
                self.pub_sol2_state.publish(Bool(data=False))

    def destroy_node(self):
        self._stop.set()
        try:
            if self._ser is not None and self._ser.is_open:
                self._send("S10")
                self._send("S20")
                self._ser.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = PressureNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

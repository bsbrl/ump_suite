# heka_udp_receiver_node.py
import math
import socket
import struct
from collections import deque

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
DEFAULT_VOLTAGE_MONITOR_SCALE = 10.0
# Upper bound on packets handled per timer tick. Large enough to absorb
# ordinary jitter at 100 packets/s, small enough that a flooded socket
# cannot block the executor.
MAX_PACKETS_PER_TICK = 64


class HekaUdpReceiverNode(Node):
    def __init__(self):
        super().__init__("heka_udp_receiver_node")

        self.declare_parameter("port", 5005)
        self.declare_parameter("voltage_monitor_scale", DEFAULT_VOLTAGE_MONITOR_SCALE)
        self.declare_parameter("resistance_window_s", 0.25)
        self.declare_parameter("min_voltage_step_raw_v", 0.005)
        self.declare_parameter("min_pulse_ms", 0.8)
        self.declare_parameter("baseline_window_ms", 2.0)
        self.declare_parameter("min_current_delta_pa", 0.05)
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
        # Absorb bursts while the executor is busy elsewhere. Best effort: the
        # kernel may cap this below what is requested.
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)

        self._res_voltage = deque()
        self._res_current = deque()
        self._backlog_ticks = 0

        self.timer = self.create_timer(0.01, self.loop)
        self.get_logger().info(f"Listening for HEKA UDP on port {port}")

    def loop(self):
        """Drain the socket each tick rather than taking a single packet.

        The Windows sender emits one packet per 10 ms, and this timer also fires
        every 10 ms, so handling exactly one packet per tick leaves zero rate
        headroom: any scheduling jitter accumulates in the socket buffer as
        growing latency and then as silent drops. Draining is bounded so one
        tick can never starve the executor.
        """
        for _ in range(MAX_PACKETS_PER_TICK):
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
        else:
            # Reaching the cap (rather than emptying the socket) means packets
            # are arriving faster than they can be processed, which shows up as
            # a lagging resistance estimate.
            self._backlog_ticks += 1
            if self._backlog_ticks % 100 == 1:
                self.get_logger().warn(
                    f"HEKA receiver hit its per-tick cap of {MAX_PACKETS_PER_TICK} "
                    f"packets ({self._backlog_ticks} tick(s) so far); samples may be "
                    "arriving faster than they can be processed"
                )

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

        self._update_resistance(voltage_raw_v, current_pa, float(sample_rate_hz))

    def _update_resistance(self, voltage_raw_v, current_pa, sample_rate_hz):
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            return

        for voltage, current in zip(voltage_raw_v, current_pa):
            voltage = float(voltage)
            current = float(current)
            if math.isfinite(voltage) and math.isfinite(current):
                self._res_voltage.append(voltage)
                self._res_current.append(current)

        window_s = max(0.02, float(self.get_parameter("resistance_window_s").value))
        max_samples = max(16, int(sample_rate_hz * window_s))
        while len(self._res_voltage) > max_samples:
            self._res_voltage.popleft()
            self._res_current.popleft()

        if len(self._res_voltage) < max(16, int(sample_rate_hz * 0.004)):
            return

        voltage = np.asarray(self._res_voltage, dtype=np.float64)
        current = np.asarray(self._res_current, dtype=np.float64)
        resistance_mohm = self._estimate_resistance_mohm(
            voltage, current, sample_rate_hz
        )
        if resistance_mohm is not None:
            self.pub_res.publish(Float32(data=float(resistance_mohm)))

    def _estimate_resistance_mohm(self, voltage_raw_v, current_pa, sample_rate_hz):
        baseline_voltage = float(np.median(voltage_raw_v))
        voltage_delta = voltage_raw_v - baseline_voltage
        peak_delta = float(np.max(np.abs(voltage_delta)))
        min_step_raw = float(self.get_parameter("min_voltage_step_raw_v").value)
        if peak_delta < min_step_raw:
            return None

        threshold = max(min_step_raw, peak_delta * 0.25)
        pulse_mask = np.abs(voltage_delta) >= threshold
        segments = self._true_segments(pulse_mask)

        min_pulse_len = max(
            3, int(sample_rate_hz * float(self.get_parameter("min_pulse_ms").value) / 1000.0)
        )
        baseline_len = max(
            3,
            int(
                sample_rate_hz
                * float(self.get_parameter("baseline_window_ms").value)
                / 1000.0
            ),
        )
        min_current_delta_pa = float(self.get_parameter("min_current_delta_pa").value)
        voltage_monitor_scale = float(self.get_parameter("voltage_monitor_scale").value)
        if not math.isfinite(voltage_monitor_scale) or voltage_monitor_scale <= 0:
            voltage_monitor_scale = DEFAULT_VOLTAGE_MONITOR_SCALE

        latest_resistance = None
        for start, end in segments:
            pulse_len = end - start
            if pulse_len < min_pulse_len:
                continue

            baseline_start = max(0, start - baseline_len)
            baseline_end = start
            if baseline_end - baseline_start < 3:
                continue

            pulse_start = start + pulse_len // 2
            pulse_end = end
            if pulse_end - pulse_start < 3:
                continue

            baseline_i = float(np.median(current_pa[baseline_start:baseline_end]))
            pulse_i = float(np.median(current_pa[pulse_start:pulse_end]))
            baseline_v = float(np.median(voltage_raw_v[baseline_start:baseline_end]))
            pulse_v = float(np.median(voltage_raw_v[pulse_start:pulse_end]))

            delta_i_pa = pulse_i - baseline_i
            if abs(delta_i_pa) < min_current_delta_pa:
                continue

            # EPC 10 voltage monitor is scaled relative to pipette voltage.
            delta_v_pipette = (pulse_v - baseline_v) / voltage_monitor_scale
            resistance_mohm = abs(delta_v_pipette / delta_i_pa) * 1e6
            if math.isfinite(resistance_mohm) and resistance_mohm > 0:
                latest_resistance = resistance_mohm

        return latest_resistance

    @staticmethod
    def _true_segments(mask):
        segments = []
        start = None
        for i, value in enumerate(mask):
            if value and start is None:
                start = i
            elif not value and start is not None:
                segments.append((start, i))
                start = None
        if start is not None:
            segments.append((start, len(mask)))
        return segments

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

"""Simulated Sensapex stage and Fluigent driver shared by the ROS-level tests."""

import json
import threading
import time

from rclpy.node import Node
from std_msgs.msg import Float32, String

from ump_suite.ros_interfaces import (
    TOPIC_PRESSURE_MBAR,
    TOPIC_PRESSURE_MEASURED,
    TOPIC_PRESSURE_STATUS,
    TOPIC_PRESSURE_TARGET,
    latched_qos,
)

START = [5000.0, 6000.0, 7000.0, 8000.0]


class SimMove:
    """A straight move at the commanded speed, like the SDK's MoveRequest."""

    def __init__(self, stage, dest, speed):
        self.finished_event = threading.Event()
        self.interrupted = False
        self.interrupt_reason = None
        self.last_pos = None
        self._stage, self._dest, self._speed = stage, list(dest), float(speed)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        start = list(self._stage.pos)
        target = [float(v) for v in self._dest]
        duration = max(abs(t - s) for s, t in zip(start, target)) / self._speed
        t0 = time.monotonic()
        while not self.interrupted:
            fraction = 1.0 if duration == 0 else min(1.0, (time.monotonic() - t0) / duration)
            for axis in range(4):
                self._stage.pos[axis] = start[axis] + fraction * (target[axis] - start[axis])
            if fraction >= 1.0:
                self.last_pos = list(self._stage.pos)
                self.finished_event.set()
                return
            time.sleep(0.002)

    def interrupt(self, reason):
        self.interrupted, self.interrupt_reason = True, reason
        self.finished_event.set()


class SimStage:
    def __init__(self):
        self.pos = list(START)
        self.moves = []
        self.current = None

    def get_pos(self, timeout=None):
        return list(self.pos)

    def goto_pos(self, pos, speed, **_kwargs):
        self.moves.append((list(pos), speed))
        self.current = SimMove(self, pos, speed)
        return self.current

    def is_busy(self):
        return self.current is not None and not self.current.finished_event.is_set()

    def stop(self):
        if self.current is not None and not self.current.finished_event.is_set():
            self.current.interrupt("stop requested before move finished")


class SimUMP:
    stage = None

    @classmethod
    def get_ump(cls):
        return cls()

    def get_device(self, _device_id):
        return SimUMP.stage


class FakePressure(Node):
    """Acknowledge every request like pressure_node, and report readiness."""

    def __init__(self):
        super().__init__("fake_pressure")
        self.ready = True
        self.commands = []
        self.pub_target = self.create_publisher(Float32, TOPIC_PRESSURE_TARGET, latched_qos())
        self.pub_measured = self.create_publisher(Float32, TOPIC_PRESSURE_MEASURED, 10)
        self.pub_status = self.create_publisher(String, TOPIC_PRESSURE_STATUS, latched_qos())
        self.create_subscription(Float32, TOPIC_PRESSURE_MBAR, self._on_command, 10)
        self.pub_target.publish(Float32(data=0.0))
        self.current = 0.0
        self.create_timer(0.2, self._tick)
        self._tick()

    def _on_command(self, msg):
        self.commands.append((time.monotonic(), float(msg.data)))
        self.current = float(msg.data)
        self.pub_target.publish(Float32(data=self.current))

    def _tick(self):
        self.pub_status.publish(String(data=json.dumps({"ready": self.ready})))
        self.pub_measured.publish(Float32(data=self.current))

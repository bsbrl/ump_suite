"""
ROS-level injection tests: real driver and logger nodes, simulated hardware.

The Sensapex SDK is replaced by a simulated stage that moves at the commanded
speed, and the Fluigent driver by a node that acknowledges every request. All
traffic stays on an isolated, localhost-only ROS domain, so these tests can
never reach a running rig.
"""

import csv
import json
import os
import threading
import time

import pytest

os.environ["ROS_DOMAIN_ID"] = "87"
os.environ["ROS_LOCALHOST_ONLY"] = "1"

import rclpy  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import Int32MultiArray, String  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

from ump_suite import ump_driver_node  # noqa: E402
from ump_suite.injection import validate_params, encode_params_message  # noqa: E402
from ump_suite.logger_node import CSV_HEADER, LoggerNode  # noqa: E402
from ump_suite.ros_interfaces import (  # noqa: E402
    SRV_ACQ_START,
    SRV_ACQ_STOP,
    SRV_INJECT_START,
    SRV_UMP_STOP,
    TOPIC_INJECT_PARAMS,
    TOPIC_INJECT_STATUS,
    TOPIC_UMP_TARGET,
    latched_qos,
)

from injection_sim import FakePressure, SimStage, SimUMP, START  # noqa: E402


class Client(Node):
    def __init__(self):
        super().__init__("injection_test_client")
        self.statuses = []
        self.pub_params = self.create_publisher(String, TOPIC_INJECT_PARAMS, latched_qos())
        self.pub_target = self.create_publisher(Int32MultiArray, TOPIC_UMP_TARGET, 10)
        self.create_subscription(String, TOPIC_INJECT_STATUS,
                                 lambda m: self.statuses.append(json.loads(m.data)),
                                 latched_qos(depth=10))
        names = (SRV_INJECT_START, SRV_UMP_STOP, SRV_ACQ_START, SRV_ACQ_STOP)
        self.triggers = {name: self.create_client(Trigger, name) for name in names}

    def call(self, name):
        client = self.triggers[name]
        assert client.wait_for_service(timeout_sec=5.0), name
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + 5.0
        while not future.done():
            assert time.monotonic() < deadline, f"{name} did not answer"
            time.sleep(0.005)
        return future.result()

    def set_params(self, seq, **values):
        raw = {"speed_um_s": 1000, "step_um": 200, "pressure_mbar": 30.0,
               "duration_ms": 300, **values}
        self.pub_params.publish(String(data=encode_params_message(
            validate_params(raw), "test", seq)))

    def wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.statuses and predicate(self.statuses[-1]):
                return self.statuses[-1]
            time.sleep(0.005)
        raise AssertionError(f"timed out; last status {self.statuses[-1:]}")


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    SimUMP.stage = SimStage()
    monkeypatch.setattr(ump_driver_node, "UMP", SimUMP)
    rclpy.init(args=["--ros-args", "-p", "logger_node:log_interval_ms:=50"])
    nodes, executor, thread = [], None, None
    try:
        client = Client()
        nodes.append(client)
        # Published before the driver exists: the latched value must still reach it.
        client.set_params(1)
        pressure = FakePressure()
        nodes.append(pressure)
        driver = ump_driver_node.UMPDriverNode()
        nodes.append(driver)
        nodes.append(LoggerNode())
        executor = MultiThreadedExecutor()
        for node in nodes:
            executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5.0
        while not driver._pressure_ready:  # the pressure driver has reported in
            assert time.monotonic() < deadline, "driver never saw pressure status"
            time.sleep(0.01)
        yield client, pressure, driver, SimUMP.stage
    finally:
        if executor is not None:
            executor.shutdown()
        if thread is not None:
            thread.join(timeout=2.0)
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.shutdown()


def read_trial():
    with open(os.path.join("logs", "trial_1.csv"), newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def test_injection_runs_the_sequence_and_logs_one_frozen_command(rig):
    client, pressure, _driver, stage = rig
    client.wait_for(lambda s: s["params_seq"] == 1)
    time.sleep(0.3)  # live position and pressure status are flowing
    assert client.call(SRV_ACQ_START).success
    client.pub_target.publish(Int32MultiArray(data=[5000, 6000, 7000, 8000, 1000]))
    time.sleep(0.4)
    stage.moves.clear()  # that target was a zero-length move

    response = client.call(SRV_INJECT_START)
    assert response.success, response.message
    assert "Injection #1 started" in response.message
    second = client.call(SRV_INJECT_START)
    assert not second.success and "already running" in second.message
    # Ordinary targets are ignored while it runs.
    client.pub_target.publish(Int32MultiArray(data=[9000, 6000, 7000, 8000, 1000]))
    done = client.wait_for(lambda s: s["stage"] in ("done", "aborted"))
    assert done["stage"] == "done", done["message"]
    time.sleep(0.4)
    assert client.call(SRV_ACQ_STOP).success

    assert stage.moves == [([5200.0, 6000.0, 7000.0, 8000.0], 1000), (START, 1000)]
    assert stage.pos == START
    values = [value for _, value in pressure.commands]
    assert values == [30.0, 0.0]
    held = pressure.commands[1][0] - pressure.commands[0][0]
    assert 0.29 <= held <= 0.40, held
    stages = [(s["stage"], s["active"]) for s in client.statuses if s["count"] == 1]
    assert ("forward", True) in stages and ("pressure", True) in stages
    assert ("back", True) in stages and stages[-1] == ("done", False)

    rows = read_trial()
    assert list(rows[0].keys()) == CSV_HEADER and CSV_HEADER[-1] == "Injection"
    flags = [int(row["Injection"]) for row in rows]
    assert flags.count(1) == 1, flags
    first = flags.index(1)
    assert first > 2 and len(rows) - first > 10
    # Positions and commands never show the injection moves or pulse.
    assert {row["current_x"] for row in rows} == {"5000"}
    assert {(row["target_x"], row["target_y"], row["target_z"], row["target_d"])
            for row in rows[first:]} == {("5000", "6000", "7000", "8000")}
    assert {row["target_pressure"] for row in rows} == {"0.0"}
    # The measured pressure stays live.
    assert "30.0" in {row["measured_pressure"] for row in rows}


def test_stop_during_hold_vents_and_leaves_the_needle_in_place(rig):
    client, pressure, _driver, stage = rig
    client.set_params(2, duration_ms=3000)
    client.wait_for(lambda s: s["params_seq"] == 2)
    assert client.call(SRV_INJECT_START).success
    client.wait_for(lambda s: s["stage"] == "pressure")
    time.sleep(0.2)
    assert client.call(SRV_UMP_STOP).success
    aborted = client.wait_for(lambda s: s["stage"] in ("done", "aborted"))
    assert aborted["stage"] == "aborted" and "vented" in aborted["message"]
    assert [value for _, value in pressure.commands] == [30.0, 0.0]
    assert len(stage.moves) == 1 and stage.pos[0] == 5200.0


def test_nonzero_pressure_is_refused_while_the_pressure_driver_is_not_ready(rig):
    client, pressure, _driver, stage = rig
    client.wait_for(lambda s: s["params_seq"] == 1)
    pressure.ready = False
    time.sleep(0.5)
    refused = client.call(SRV_INJECT_START)
    assert not refused.success and "pressure driver" in refused.message
    assert stage.moves == [] and pressure.commands == []
    assert client.statuses[-1]["count"] == 0


def test_injection_leaving_the_stage_range_is_refused_before_moving(rig):
    client, _pressure, _driver, stage = rig
    stage.pos[0] = 19900.0
    client.set_params(3, step_um=200)
    client.wait_for(lambda s: s["params_seq"] == 3)
    refused = client.call(SRV_INJECT_START)
    assert not refused.success and "outside the stage range" in refused.message
    assert stage.moves == []


def test_an_injection_stopped_after_a_finished_one_leaves_the_driver_usable(rig):
    client, pressure, driver, stage = rig
    client.wait_for(lambda s: s["params_seq"] == 1)
    assert client.call(SRV_INJECT_START).success
    client.wait_for(lambda s: s["count"] == 1 and s["stage"] == "done")
    client.set_params(2, duration_ms=3000)
    client.wait_for(lambda s: s["params_seq"] == 2)
    assert client.call(SRV_INJECT_START).success
    client.wait_for(lambda s: s["count"] == 2 and s["stage"] == "pressure")
    time.sleep(0.1)
    assert client.call(SRV_UMP_STOP).success
    client.wait_for(lambda s: s["count"] == 2 and s["stage"] == "aborted")
    time.sleep(0.2)
    # Neither outcome, nor logging them, may latch a fault.
    assert not driver._faulted
    client.pub_target.publish(Int32MultiArray(data=[5000, 6000, 7000, 8000, 1000]))
    deadline = time.monotonic() + 3.0
    while stage.pos[0] != 5000.0:
        assert time.monotonic() < deadline, stage.pos
        time.sleep(0.01)
    assert stage.moves[-1] == ([5000, 6000, 7000, 8000], 1000)
    assert [value for _, value in pressure.commands] == [30.0, 0.0, 30.0, 0.0]


def test_injection_on_the_z_axis_moves_only_z_and_logs_it_frozen(rig):
    client, pressure, _driver, stage = rig
    client.set_params(2, axis="Z", step_um=-150)
    client.wait_for(lambda s: s["params_seq"] == 2 and s["params"]["axis"] == "Z")
    time.sleep(0.3)
    assert client.call(SRV_ACQ_START).success
    time.sleep(0.3)
    response = client.call(SRV_INJECT_START)
    assert response.success and "Z -150 um" in response.message
    done = client.wait_for(lambda s: s["stage"] in ("done", "aborted"))
    assert done["stage"] == "done", done["message"]
    time.sleep(0.3)
    assert client.call(SRV_ACQ_STOP).success
    assert stage.moves == [([5000.0, 6000.0, 6850.0, 8000.0], 1000), (START, 1000)]
    assert stage.pos == START
    rows = read_trial()
    assert [int(row["Injection"]) for row in rows].count(1) == 1
    for column, value in (("current_x", "5000"), ("current_y", "6000"),
                          ("current_z", "7000"), ("current_d", "8000")):
        assert {row[column] for row in rows} == {value}, column

"""
ROS-level tests of the orient segments with the real UMP driver and a simulated stage.

A segment only marks when orienting began and ended; the needle is moved by the
ordinary targets meanwhile. Isolated, localhost-only ROS domain.
"""

import json
import os
import threading
import time

import pytest

os.environ["ROS_DOMAIN_ID"] = "88"
os.environ["ROS_LOCALHOST_ONLY"] = "1"

import rclpy  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import Int32MultiArray, String  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

from ump_suite import ump_driver_node  # noqa: E402
from ump_suite.ros_interfaces import (  # noqa: E402
    SRV_INJECT_START,
    SRV_ORIENT_ABORT,
    SRV_ORIENT_DONE,
    SRV_ORIENT_START,
    TOPIC_ORIENT_STATUS,
    TOPIC_UMP_LIVE,
    TOPIC_UMP_TARGET,
    latched_qos,
)

from injection_sim import FakePressure, SimStage, SimUMP, START  # noqa: E402


class Client(Node):
    def __init__(self):
        super().__init__("orient_test_client")
        self.statuses = []
        self.pub_target = self.create_publisher(Int32MultiArray, TOPIC_UMP_TARGET, 10)
        self.create_subscription(
            String,
            TOPIC_ORIENT_STATUS,
            lambda m: self.statuses.append(json.loads(m.data)),
            latched_qos(depth=10),
        )
        self.triggers = {
            n: self.create_client(Trigger, n)
            for n in (SRV_ORIENT_START, SRV_ORIENT_DONE, SRV_ORIENT_ABORT, SRV_INJECT_START)
        }

    def call(self, name):
        client = self.triggers[name]
        assert client.wait_for_service(timeout_sec=5.0), name
        future = client.call_async(Trigger.Request())
        deadline = time.monotonic() + 5.0
        while not future.done():
            assert time.monotonic() < deadline, f"{name} did not answer"
            time.sleep(0.005)
        return future.result()

    def wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.statuses and predicate(self.statuses[-1]):
                return self.statuses[-1]
            time.sleep(0.01)
        raise AssertionError(f"timed out; last status {self.statuses[-1:]}")


@pytest.fixture
def rig(monkeypatch):
    SimUMP.stage = SimStage()
    monkeypatch.setattr(ump_driver_node, "UMP", SimUMP)
    rclpy.init()
    nodes, executor, thread = [], None, None
    try:
        client = Client()
        nodes.append(client)
        nodes.append(FakePressure())
        driver = ump_driver_node.UMPDriverNode()
        nodes.append(driver)
        executor = MultiThreadedExecutor()
        for node in nodes:
            executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        client.wait_for(lambda s: s["stage"] == "idle" and not s["active"])
        yield client, driver, SimUMP.stage
    finally:
        if executor is not None:
            executor.shutdown()
        if thread is not None:
            thread.join(timeout=2.0)
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.shutdown()


def test_a_segment_opens_lets_the_needle_move_and_closes_done(rig):
    client, driver, stage = rig
    res = client.call(SRV_ORIENT_START)
    assert res.success, res.message
    opened = client.wait_for(lambda s: s["active"])
    assert opened["count"] == 1 and opened["stage"] == "running"
    target = [int(START[0]) - 40, int(START[1]) + 25, int(START[2]), int(START[3])]
    client.pub_target.publish(Int32MultiArray(data=target + [500]))
    deadline = time.monotonic() + 2.0
    while not any(m[0] == target for m in stage.moves):
        assert time.monotonic() < deadline, "the target was not applied during the segment"
        time.sleep(0.01)
    res = client.call(SRV_ORIENT_DONE)
    assert res.success and "done" in res.message, res.message
    closed = client.wait_for(lambda s: not s["active"] and s["count"] == 1)
    assert closed["stage"] == "done" and closed["result"] == "done", closed
    assert not driver._orienting


def test_inject_and_a_second_orient_are_refused_while_a_segment_is_open(rig):
    client, _driver, _stage = rig
    assert client.call(SRV_ORIENT_START).success
    client.wait_for(lambda s: s["active"])
    res = client.call(SRV_INJECT_START)
    assert not res.success and "orient segment is open" in res.message
    again = client.call(SRV_ORIENT_START)
    assert not again.success and "already open" in again.message
    res = client.call(SRV_ORIENT_ABORT)
    assert res.success
    closed = client.wait_for(lambda s: not s["active"])
    assert closed["stage"] == "aborted" and closed["result"] == "aborted", closed


def test_closing_without_an_open_segment_is_refused(rig):
    client, _driver, _stage = rig
    for name in (SRV_ORIENT_DONE, SRV_ORIENT_ABORT):
        res = client.call(name)
        assert not res.success and "No orient segment" in res.message


def test_logger_flags_the_segment_logs_its_moves_and_records_the_outcome(tmp_path, monkeypatch):
    """Flag the start and the result of a segment; the positions inside it are live."""
    import csv

    from ump_suite.logger_node import CSV_HEADER, LoggerNode
    from ump_suite.ros_interfaces import SRV_ACQ_START, SRV_ACQ_STOP, TOPIC_INJECT_OUTCOME

    monkeypatch.chdir(tmp_path)
    rclpy.init(args=["--ros-args", "-p", "logger_node:log_interval_ms:=50"])
    nodes, executor, thread = [], None, None
    try:
        pub = Node("orient_status_pub")
        nodes.append(pub)
        status = pub.create_publisher(String, TOPIC_ORIENT_STATUS, latched_qos(depth=10))
        live = pub.create_publisher(Int32MultiArray, TOPIC_UMP_LIVE, 10)
        outcome = pub.create_publisher(String, TOPIC_INJECT_OUTCOME, 10)
        logger = LoggerNode()
        nodes.append(logger)
        acq = {n: pub.create_client(Trigger, n) for n in (SRV_ACQ_START, SRV_ACQ_STOP)}
        executor = MultiThreadedExecutor()
        for node in nodes:
            executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()

        def call(name):
            assert acq[name].wait_for_service(timeout_sec=5.0)
            fut = acq[name].call_async(Trigger.Request())
            while not fut.done():
                time.sleep(0.005)
            return fut.result()

        def send(count, active, stage, result=None):
            status.publish(String(data=json.dumps(
                {"count": count, "active": active, "stage": stage, "result": result})))

        def move(x, seconds):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                live.publish(Int32MultiArray(data=[x, 2000, 300, 9000]))
                time.sleep(0.02)

        send(0, False, "idle")
        assert call(SRV_ACQ_START).success
        move(1000, 0.2)
        send(1, True, "running")
        for x in (1010, 1020, 1030, 1040):
            move(x, 0.1)
        send(1, False, "done", "done")
        move(1040, 0.2)
        outcome.publish(String(data=json.dumps({"count": 1, "outcome": "cell"})))
        outcome.publish(String(data=json.dumps({"count": 1, "outcome": "nonsense"})))
        move(1040, 0.2)
        assert call(SRV_ACQ_STOP).success
        with open(os.path.join("logs", "trial_1.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
    finally:
        if executor is not None:
            executor.shutdown()
        if thread is not None:
            thread.join(timeout=2.0)
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.shutdown()
    assert list(rows[0].keys()) == CSV_HEADER
    assert [r["Orient"] for r in rows].count("1") == 1
    assert [r["orient_result"] for r in rows if r["orient_result"]] == ["done"]
    assert [r["injection_outcome"] for r in rows if r["injection_outcome"]] == ["cell"]
    start = next(i for i, r in enumerate(rows) if r["Orient"] == "1")
    end = next(i for i, r in enumerate(rows) if r["orient_result"])
    assert start < end
    inside = {int(r["current_x"]) for r in rows[start:end]}
    assert len(inside) > 1, f"positions inside the segment were frozen: {inside}"

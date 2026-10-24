"""
ROS-level orient tests with the real UMP driver and a simulated stage and embryo.

A node publishes vision scenes of a simulated embryo that turns when the needle
drags it.

Isolated, localhost-only ROS domain; the calibration file is a temporary one.
"""

import json
import math
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
from ump_suite.orient import OrientParams, encode_params_message  # noqa: E402
from ump_suite.ros_interfaces import (  # noqa: E402
    SRV_INJECT_START,
    SRV_ORIENT_START,
    SRV_UMP_STOP,
    TOPIC_ORIENT_PARAMS,
    TOPIC_ORIENT_STATUS,
    TOPIC_UMP_TARGET,
    TOPIC_VISION_SCENE,
    latched_qos,
)
from ump_suite.vision import calibration as cal_module  # noqa: E402
from ump_suite.vision.geometry import wrap_deg  # noqa: E402

from injection_sim import FakePressure, SimStage, SimUMP, START  # noqa: E402
from orient_sim import UM_PER_PX, calibration  # noqa: E402

FAST = OrientParams(
    speed_um_s=2000,
    settle_s=0.05,
    confirm_frames=2,
    stroke_max_um=250,
    gain_deg_per_um=0.15,
    timeout_s=30.0,
)


class SceneFeeder(Node):
    """Publishes scenes at 30 Hz; the embryo turns while the tip drags inside the chorion."""

    def __init__(self, stage, cell_deg=-60.0, gain=0.15):
        super().__init__("scene_feeder")
        self.stage, self.cal = stage, calibration()
        self.cell, self.gain = cell_deg, gain
        self.cx, self.cy, self.r = 500.0, 540.0, 370.0
        self.tip0, self.pos0 = (1000.0, 540.0), list(START)
        self.pub = self.create_publisher(String, TOPIC_VISION_SCENE, 5)
        self.last_tip = self.tip()
        self.create_timer(1 / 30.0, self.tick)

    def tip(self):
        jx, jy = self.cal.vx, self.cal.vy
        ux, uy = self.stage.pos[0] - self.pos0[0], self.stage.pos[1] - self.pos0[1]
        return (self.tip0[0] + jx[0] * ux + jy[0] * uy, self.tip0[1] + jx[1] * ux + jy[1] * uy)

    def tick(self):
        tip = self.tip()
        if math.hypot(self.last_tip[0] - self.cx, self.last_tip[1] - self.cy) < self.r:
            dy_px = tip[1] - self.last_tip[1]
            self.cell += self.gain * (-dy_px) * UM_PER_PX
            self.cy -= dy_px / 2.0
        self.last_tip = tip
        a = math.radians(self.cell)
        yolk = (self.cx - 60, self.cy, 180)
        scene = {
            "version": 1,
            "stamp": time.time(),
            "image_size": [1440, 1080],
            "um_per_px": None,
            "embryos": [
                {
                    "id": 0,
                    "chorion": {"cx": self.cx, "cy": self.cy, "r": self.r},
                    "complete": True,
                    "yolk": {"cx": yolk[0], "cy": yolk[1], "r": 180},
                    "cell": {
                        "cx": yolk[0] + 200 * math.cos(a),
                        "cy": yolk[1] - 200 * math.sin(a),
                        "area": 9000,
                    },
                    "cell_angle_image_deg": wrap_deg(self.cell),
                    "cell_angle_deg": wrap_deg(self.cell),
                    "cell_on_face": False,
                    "score": 1.0,
                }
            ],
            "needle": {
                "tip": list(tip),
                "base": [1439.0, tip[1]],
                "direction": [-1.0, 0.0],
                "length_px": 400,
            },
            "target": 0,
            "ray": None,
        }
        self.pub.publish(String(data=json.dumps(scene)))


class Client(Node):
    def __init__(self):
        super().__init__("orient_test_client")
        self.statuses = []
        self.pub_params = self.create_publisher(String, TOPIC_ORIENT_PARAMS, latched_qos())
        self.pub_target = self.create_publisher(Int32MultiArray, TOPIC_UMP_TARGET, 10)
        self.create_subscription(
            String,
            TOPIC_ORIENT_STATUS,
            lambda m: self.statuses.append(json.loads(m.data)),
            latched_qos(depth=10),
        )
        self.triggers = {
            n: self.create_client(Trigger, n)
            for n in (SRV_ORIENT_START, SRV_INJECT_START, SRV_UMP_STOP)
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

    def wait_for(self, predicate, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.statuses and predicate(self.statuses[-1]):
                return self.statuses[-1]
            time.sleep(0.01)
        raise AssertionError(f"timed out; last status {self.statuses[-1:]}")


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(cal_module, "DEFAULT_PATH", tmp_path / "vision_calibration.json")
    calibration().save()
    SimUMP.stage = SimStage()
    monkeypatch.setattr(ump_driver_node, "UMP", SimUMP)
    rclpy.init()
    nodes, executor, thread = [], None, None
    try:
        client = Client()
        nodes.append(client)
        client.pub_params.publish(String(data=encode_params_message(FAST, "test", 1)))
        nodes.append(FakePressure())
        driver = ump_driver_node.UMPDriverNode()
        nodes.append(driver)
        feeder = SceneFeeder(SimUMP.stage)
        nodes.append(feeder)
        executor = MultiThreadedExecutor()
        for node in nodes:
            executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        client.wait_for(lambda s: s.get("params_seq") == 1, timeout=5.0)
        yield client, driver, feeder, SimUMP.stage
    finally:
        if executor is not None:
            executor.shutdown()
        if thread is not None:
            thread.join(timeout=2.0)
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.shutdown()


def test_orient_turns_the_embryo_and_reports_done(rig):
    client, driver, feeder, stage = rig
    res = client.call(SRV_ORIENT_START)
    assert res.success, res.message
    final = client.wait_for(lambda s: s["stage"] in ("done", "aborted") and s["count"] == 1)
    assert final["stage"] == "done" and final["result"] == "done", final
    assert abs(wrap_deg(feeder.cell)) <= FAST.tolerance_deg + 1
    assert final["strokes"] >= 1 and stage.moves
    assert all(m[0][2] == START[2] and m[0][3] == START[3] for m in stage.moves)
    assert not driver._orienting


def test_inject_and_targets_are_refused_while_orienting_and_stop_aborts(rig):
    client, driver, feeder, stage = rig
    assert client.call(SRV_ORIENT_START).success
    client.wait_for(lambda s: s["active"] and s["stage"] == "stroke")
    res = client.call(SRV_INJECT_START)
    assert not res.success and "orient" in res.message
    again = client.call(SRV_ORIENT_START)
    assert not again.success and "already running" in again.message
    n = len(stage.moves)
    client.pub_target.publish(Int32MultiArray(data=[1, 2, 3, 4, 100]))
    time.sleep(0.2)
    assert all(m[0] != [1, 2, 3, 4] for m in stage.moves[n:])
    assert client.call(SRV_UMP_STOP).success
    final = client.wait_for(lambda s: s["stage"] in ("done", "aborted"))
    assert final["stage"] == "aborted" and "stop" in final["message"], final


def test_orient_is_refused_without_a_calibration(rig, tmp_path, monkeypatch):
    client, *_ = rig
    monkeypatch.setattr(cal_module, "DEFAULT_PATH", tmp_path / "missing.json")
    res = client.call(SRV_ORIENT_START)
    assert not res.success and "calibration" in res.message


def test_logger_flags_the_orient_and_records_the_outcome(tmp_path, monkeypatch):
    """Freeze state during an orient, flag its start and result, and log an outcome."""
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
            status.publish(
                String(
                    data=json.dumps(
                        {"count": count, "active": active, "stage": stage, "result": result}
                    )
                )
            )

        send(0, False, "idle")
        assert call(SRV_ACQ_START).success
        time.sleep(0.2)
        send(1, True, "starting")
        time.sleep(0.25)
        send(1, False, "done", "done")
        time.sleep(0.2)
        outcome.publish(String(data=json.dumps({"count": 1, "outcome": "cell"})))
        outcome.publish(String(data=json.dumps({"count": 1, "outcome": "nonsense"})))
        time.sleep(0.2)
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

"""
GUI orient panel, outcome recording and vision overlay.

Offscreen, against the real UMP driver with a simulated stage and a simulated
embryo publishing vision scenes; isolated localhost-only ROS domain; temporary
settings and calibration files.
"""

import json
import os
import threading
import time

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["ROS_DOMAIN_ID"] = "89"
os.environ["ROS_LOCALHOST_ONLY"] = "1"

import rclpy  # noqa: E402
from PyQt5.QtCore import QSettings  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from ump_suite import ump_driver_node  # noqa: E402
from ump_suite.gui_node import GuiNode, UMPGuiApp, draw_scene  # noqa: E402
from ump_suite.ros_interfaces import TOPIC_INJECT_OUTCOME  # noqa: E402
from ump_suite.vision import calibration as cal_module  # noqa: E402
from ump_suite.vision.geometry import wrap_deg  # noqa: E402

from injection_sim import FakePressure, SimStage, SimUMP  # noqa: E402
from orient_sim import calibration  # noqa: E402
from test_orient_ros import SceneFeeder  # noqa: E402


@pytest.fixture
def gui(tmp_path, monkeypatch):
    monkeypatch.setattr(cal_module, "DEFAULT_PATH", tmp_path / "vision_calibration.json")
    calibration().save()
    SimUMP.stage = SimStage()
    monkeypatch.setattr(ump_driver_node, "UMP", SimUMP)
    app = QApplication.instance() or QApplication([])
    rclpy.init()
    nodes, executor, thread = [], None, None
    try:
        node = GuiNode()
        nodes.append(node)
        nodes.append(FakePressure())
        driver = ump_driver_node.UMPDriverNode()
        nodes.append(driver)
        feeder = SceneFeeder(SimUMP.stage)
        nodes.append(feeder)
        listener = Node("outcome_listener")
        outcomes = []
        listener.create_subscription(
            String, TOPIC_INJECT_OUTCOME, lambda m: outcomes.append(json.loads(m.data)), 10
        )
        nodes.append(listener)
        executor = MultiThreadedExecutor()
        for item in nodes:
            executor.add_node(item)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        yield app, node, driver, feeder, outcomes, str(tmp_path / "gui.ini")
    finally:
        if executor is not None:
            executor.shutdown()
        if thread is not None:
            thread.join(timeout=2.0)
        for item in reversed(nodes):
            item.destroy_node()
        rclpy.shutdown()


def wait_until(app, window, predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        window._poll_live_to_gui()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def test_orient_button_runs_the_skill_and_resyncs_ump1(gui):
    app, node, driver, feeder, _outcomes, path = gui
    window = UMPGuiApp(node, settings=QSettings(path, QSettings.IniFormat))
    panel = window.orient_panel
    assert panel.title() == "Orient" and panel.orient_button.text() == "Orient"
    # Faster settings for the test, typed into the boxes; they persist.
    for name, value in (("speed_um_s", 2000), ("stroke_max_um", 250), ("tolerance_deg", 15)):
        panel.boxes[name].setValue(value)
    wait_until(
        app,
        window,
        lambda: (node.latest_orient_status or {}).get("params", {}).get("speed_um_s") == 2000,
    )
    feeder.gain = 0.15
    driver._orient_params = driver._orient_params.__class__(
        **{**driver._orient_params.to_dict(), "gain_deg_per_um": 0.15, "settle_s": 0.05}
    )
    panel.orient()
    wait_until(
        app,
        window,
        lambda: (node.latest_orient_status or {}).get("stage") in ("done", "aborted")
        and (node.latest_orient_status or {}).get("count") == 1,
    )
    assert node.latest_orient_status["result"] == "done", node.latest_orient_status
    assert abs(wrap_deg(feeder.cell)) <= 16
    wait_until(app, window, lambda: panel.orient_button.isEnabled())
    live = [int(v) for v in node.latest_live_ump]
    targets = [window.panel1.target_spins[a].value() for a in ("X", "Y", "Z", "D")]
    assert targets == live
    reopened = UMPGuiApp(node, settings=QSettings(path, QSettings.IniFormat))
    assert reopened.orient_panel.boxes["speed_um_s"].value() == 2000


def test_outcome_needs_an_injection_and_is_published(gui):
    app, node, _driver, _feeder, outcomes, path = gui
    window = UMPGuiApp(node, settings=QSettings(path, QSettings.IniFormat))
    panel = window.injection_panel
    panel.record_outcome()
    assert outcomes == []
    node.latest_inject_status = {"count": 3, "active": False}
    panel.outcome.setCurrentIndex(panel.outcome.findData("yolk"))
    panel.record_outcome()
    deadline = time.monotonic() + 3.0
    while not outcomes and time.monotonic() < deadline:
        time.sleep(0.01)
    assert outcomes == [{"count": 3, "outcome": "yolk"}]


def test_draw_scene_marks_the_target_and_the_distance():
    frame = np.full((1080, 1440, 3), 230, np.uint8)
    scene = {
        "embryos": [
            {
                "chorion": {"cx": 500, "cy": 540, "r": 370},
                "yolk": {"cx": 440, "cy": 540, "r": 180},
                "cell": {"cx": 700, "cy": 540, "area": 900},
                "cell_angle_deg": 0.0,
                "cell_on_face": False,
            }
        ],
        "needle": {"tip": [1000, 540], "base": [1439, 540], "direction": [-1, 0]},
        "target": 0,
        "ray": {"hits_cell": True, "cell_near_px": 290, "cell_near_um": 522},
        "test_only_model": True,
    }
    out = draw_scene(frame.copy(), scene)
    assert (out != frame).any()
    assert tuple(out[540, 710 - 0]) != (230, 230, 230)  # the line to the cell's edge

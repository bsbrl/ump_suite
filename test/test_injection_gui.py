"""
GUI injection panel: placement, typed values, persistence and the button.

Runs offscreen against the real UMP driver node with a simulated stage and
pressure driver, on an isolated localhost-only ROS domain.
"""

import os
import threading
import time

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["ROS_DOMAIN_ID"] = "87"
os.environ["ROS_LOCALHOST_ONLY"] = "1"

import rclpy  # noqa: E402
from PyQt5.QtCore import QSettings  # noqa: E402
from PyQt5.QtWidgets import QApplication, QGroupBox  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402

from ump_suite import ump_driver_node  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from ump_suite.gui_node import GuiNode, UMPGuiApp  # noqa: E402
from ump_suite.injection import encode_params_message, validate_params  # noqa: E402

from injection_sim import FakePressure, SimStage, SimUMP, START  # noqa: E402


@pytest.fixture
def gui(tmp_path, monkeypatch):
    SimUMP.stage = SimStage()
    monkeypatch.setattr(ump_driver_node, "UMP", SimUMP)
    app = QApplication.instance() or QApplication([])
    rclpy.init()
    nodes, executor, thread = [], None, None
    try:
        node = GuiNode()
        nodes.append(node)
        pressure = FakePressure()
        nodes.append(pressure)
        driver = ump_driver_node.UMPDriverNode()
        nodes.append(driver)
        executor = MultiThreadedExecutor()
        for item in nodes:
            executor.add_node(item)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5.0
        while not driver._pressure_ready:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        path = str(tmp_path / "ump_suite" / "gui.ini")
        yield app, node, pressure, path
    finally:
        if executor is not None:
            executor.shutdown()
        if thread is not None:
            thread.join(timeout=2.0)
        for item in reversed(nodes):
            item.destroy_node()
        rclpy.shutdown()


def wait_until(app, window, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        window._poll_live_to_gui()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def test_panel_sits_between_the_ump_panels_and_the_odrive(gui):
    app, node, _pressure, path = gui
    window = UMPGuiApp(node, settings=QSettings(path, QSettings.IniFormat))
    layout = window.injection_panel.parentWidget().layout()
    motor = next(g for g in window.findChildren(QGroupBox) if g.title() == "Motor (ODrive)")
    order = [layout.indexOf(w) for w in (window.panel1, window.panel2,
                                         window.injection_panel, motor)]
    assert order == sorted(order) and order[2] == order[1] + 1 and order[3] == order[2] + 1
    assert window.injection_panel.title() == "Injection"
    assert window.injection_panel.inject_button.text() == "Inject"


def test_typed_values_drive_the_injection_and_survive_a_restart(gui, tmp_path):
    app, node, pressure, path = gui
    window = UMPGuiApp(node, settings=QSettings(path, QSettings.IniFormat))
    panel = window.injection_panel
    # Type the values as an operator would, including a leading '-'.
    for box, text in ((panel.speed, "1500"), (panel.step, "-120"),
                      (panel.pressure, "-35.5"), (panel.duration, "250")):
        box.lineEdit().selectAll()
        box.lineEdit().setText(text)
        box.interpretText()
    assert (panel.speed.value(), panel.step.value(), panel.pressure.value(),
            panel.duration.value()) == (1500, -120, -35.5, 250)

    panel.inject_button.click()
    # Disabled while the driver reports the injection running, enabled after.
    wait_until(app, window, lambda: not panel.inject_button.isEnabled())
    wait_until(app, window, lambda: (node.latest_inject_status or {}).get("stage") == "done")
    wait_until(app, window, lambda: panel.inject_button.isEnabled())
    assert "#1 done" in panel.state_label.text()
    assert SimUMP.stage.moves == [([START[0] - 120, *START[1:]], 1500), (START, 1500)]
    assert [value for _, value in pressure.commands] == [-35.5, 0.0]

    # Close and reopen: a new window reading the same file shows the same values.
    window.close()
    window.deleteLater()
    reopened = UMPGuiApp(node, settings=QSettings(path, QSettings.IniFormat))
    again = reopened.injection_panel
    assert (again.speed.value(), again.step.value(), again.pressure.value(),
            again.duration.value()) == (1500, -120, -35.5, 250)
    assert os.path.isfile(path)

    reopened.resize(1500, 1000)
    reopened.show()
    wait_until(app, reopened, lambda: True, timeout=0.2)
    shot = os.environ.get("INJECTION_GUI_SCREENSHOT")
    if shot:
        reopened.grab().save(shot)


def test_invalid_box_values_are_not_published_or_triggered(gui):
    app, node, pressure, path = gui
    window = UMPGuiApp(node, settings=QSettings(path, QSettings.IniFormat))
    panel = window.injection_panel
    panel.step.setValue(0)
    assert "Not published" in panel.state_label.text()
    panel.inject_button.click()
    assert "Injection not started" in window.status.text()
    time.sleep(0.2)
    assert SimUMP.stage.moves == [] and pressure.commands == []


def test_inject_uses_the_boxes_even_after_another_client_published_values(gui):
    app, node, pressure, path = gui
    window = UMPGuiApp(node, settings=QSettings(path, QSettings.IniFormat))
    panel = window.injection_panel
    panel.duration.setValue(150)
    wait_until(app, window, lambda: (node.latest_inject_status or {}).get("params_token")
               == panel._token)
    # A different client (a policy, a script) publishes its own values.
    node.pub_inject_params.publish(String(data=encode_params_message(
        validate_params({"speed_um_s": 100, "step_um": 10, "pressure_mbar": 99.0,
                         "duration_ms": 10}), "other", 1)))
    wait_until(app, window, lambda: node.latest_inject_status.get("params_token") == "other")
    panel.inject_button.click()
    wait_until(app, window, lambda: (node.latest_inject_status or {}).get("stage") == "done")
    assert [value for _, value in pressure.commands] == [50.0, 0.0]
    assert SimUMP.stage.moves[0] == ([START[0] + 50, *START[1:]], 1000)

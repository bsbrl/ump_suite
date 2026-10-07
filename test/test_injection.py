"""Unit tests for the injection sequence; fakes only, no ROS or hardware."""

import json
import threading
import time

import pytest

from ump_suite.injection import (
    AXES,
    DURATION_MAX_MS,
    InjectionParams,
    InjectionSequence,
    STEP_LIMIT_UM,
    decode_params_message,
    encode_params_message,
    forward_target,
    validate_params,
)

GOOD = {"speed_um_s": 1000, "step_um": 40, "pressure_mbar": -25.5, "duration_ms": 80}


class FakeMove:
    def __init__(self, stage, dest, finish=True, reach=True):
        self.finished_event = threading.Event()
        self.interrupted = False
        self.interrupt_reason = None
        self.last_pos = None
        self.dest = list(dest)
        if finish:
            stage.pos[:] = [float(v) if reach else float(v) - 5.0 for v in dest]
            self.last_pos = list(stage.pos)
            self.finished_event.set()


class Rig:
    """Records every hardware call in order."""

    def __init__(self, ack=True, finish=True, reach=True, interrupt=False):
        self.pos = [1000.0, 2000.0, 3000.0, 4000.0]
        self.calls = []
        self.reports = []
        self.ack = ack
        self.finish = finish
        self.reach = reach
        self.interrupt = interrupt
        self.abort = threading.Event()
        self.ack_time = None
        self.vent_time = None

    def read_position(self):
        return list(self.pos)

    def move_to(self, position, speed):
        self.calls.append(("move", list(position), speed))
        move = FakeMove(self, position, finish=self.finish and not self.interrupt,
                        reach=self.reach)
        if self.interrupt:
            move.interrupted = True
            move.interrupt_reason = "stop requested before move finished"
            move.finished_event.set()
        return move

    def stop_stage(self):
        self.calls.append(("stop",))

    def send_pressure(self, mbar):
        self.calls.append(("pressure", mbar))
        if mbar == 0.0:
            self.vent_time = time.monotonic()

    def wait_pressure_ack(self, after, timeout):
        if not self.ack:
            return None
        self.ack_time = time.monotonic()
        return self.calls[-1][1]

    def report(self, stage, active, message):
        self.reports.append((stage, active, message))

    def sequence(self, params):
        return InjectionSequence(
            params, read_position=self.read_position, move_to=self.move_to,
            stop_stage=self.stop_stage, send_pressure=self.send_pressure,
            wait_pressure_ack=self.wait_pressure_ack, report=self.report,
            abort_event=self.abort)


def params(**changes):
    return validate_params({**GOOD, **changes})


def test_validate_accepts_signed_values_and_returns_typed_params():
    p = params(step_um=-40)
    assert p == InjectionParams(1000, -40, -25.5, 80, "X")
    assert isinstance(p.speed_um_s, int) and isinstance(p.pressure_mbar, float)


def test_axis_defaults_to_x_and_accepts_any_case():
    assert params().axis == "X"  # a message from before the axis selector
    assert params(axis="z").axis == "Z" and params(axis=" d ").axis_index == 3
    assert "Y +40 um" in params(axis="Y").describe()


@pytest.mark.parametrize("changes", [
    {"speed_um_s": 9}, {"speed_um_s": 2001}, {"step_um": 0},
    {"step_um": STEP_LIMIT_UM + 1}, {"pressure_mbar": 1000.5},
    {"pressure_mbar": float("nan")}, {"duration_ms": 0},
    {"duration_ms": DURATION_MAX_MS + 1}, {"step_um": 2.5}, {"speed_um_s": "fast"},
    {"axis": "W"}, {"axis": None}, {"axis": "XY"},
])
def test_validate_rejects_out_of_range_or_malformed_values(changes):
    with pytest.raises(ValueError):
        params(**changes)


def test_validate_names_a_missing_key():
    raw = dict(GOOD)
    del raw["duration_ms"]
    with pytest.raises(ValueError, match="duration_ms"):
        validate_params(raw)


def test_params_message_round_trip_keeps_token_and_sequence():
    text = encode_params_message(params(axis="D"), "abc", 7)
    assert json.loads(text)["axis"] == "D"
    assert decode_params_message(text) == (params(axis="D"), "abc", 7)
    with pytest.raises(ValueError):
        decode_params_message("not json")
    with pytest.raises(ValueError):
        decode_params_message(json.dumps([1, 2]))


def test_forward_target_refuses_leaving_the_stage_range():
    assert forward_target([100.0, 0, 0, 0], params(step_um=-100)) == 0.0
    with pytest.raises(ValueError):
        forward_target([50.0, 0, 0, 0], params(step_um=-100))
    with pytest.raises(ValueError):
        forward_target([19990.0, 0, 0, 0], params(step_um=20))


def test_forward_target_uses_the_selected_axis():
    start = [10000.0, 200.0, 19990.0, 50.0]
    assert forward_target(start, params(axis="Y", step_um=-100)) == 100.0
    with pytest.raises(ValueError, match="Z target"):
        forward_target(start, params(axis="Z", step_um=20))
    with pytest.raises(ValueError, match="D target"):
        forward_target(start, params(axis="D", step_um=-100))


@pytest.mark.parametrize("axis", AXES)
def test_sequence_moves_only_the_selected_axis_and_back(axis):
    rig = Rig()
    assert rig.sequence(params(axis=axis)).run() is True
    inward = [1000.0, 2000.0, 3000.0, 4000.0]
    inward[AXES.index(axis)] += 40.0
    assert [call for call in rig.calls if call[0] == "move"] == [
        ("move", inward, 1000), ("move", [1000.0, 2000.0, 3000.0, 4000.0], 1000)]
    assert rig.reports[0][2].startswith(f"moving {axis} to")


def test_arrival_is_checked_on_the_selected_axis():
    rig = Rig(reach=False)
    assert rig.sequence(params(axis="Z")).run() is False
    assert "stopped at Z 3035.0 um, not 3040.0 um" in rig.reports[-1][2]


def test_full_sequence_order_values_and_hold_time():
    rig = Rig()
    p = params()
    assert rig.sequence(p).run() is True
    assert rig.calls == [
        ("move", [1040.0, 2000.0, 3000.0, 4000.0], 1000),
        ("pressure", -25.5),
        ("pressure", 0.0),
        ("move", [1000.0, 2000.0, 3000.0, 4000.0], 1000),
    ]
    held = rig.vent_time - rig.ack_time
    assert 0.080 <= held < 0.080 + 0.05
    stages = [(stage, active) for stage, active, _ in rig.reports]
    assert stages == [("forward", True), ("pressure", True), ("back", True), ("done", False)]
    assert rig.pos[0] == 1000.0


def test_negative_step_moves_down_first_and_back_up():
    rig = Rig()
    assert rig.sequence(params(step_um=-30)).run() is True
    moves = [call[1][0] for call in rig.calls if call[0] == "move"]
    assert moves == [970.0, 1000.0]


def test_missing_pressure_ack_vents_and_retracts():
    rig = Rig(ack=False)
    assert rig.sequence(params()).run() is False
    assert rig.calls[1:] == [
        ("pressure", -25.5), ("pressure", 0.0),
        ("move", [1000.0, 2000.0, 3000.0, 4000.0], 1000),
    ]
    stage, active, message = rig.reports[-1]
    assert (stage, active) == ("aborted", False)
    assert "did not acknowledge" in message and "vented" in message and "moved back" in message


def test_stop_during_hold_vents_immediately_and_does_not_move():
    rig = Rig()
    long_hold = params(duration_ms=5000)
    timer = threading.Timer(0.05, rig.abort.set)
    timer.start()
    started = time.monotonic()
    assert rig.sequence(long_hold).run() is False
    assert time.monotonic() - started < 1.0
    assert rig.calls[-1] == ("pressure", 0.0)
    assert [c for c in rig.calls if c[0] == "move"] == [
        ("move", [1040.0, 2000.0, 3000.0, 4000.0], 1000)]
    assert rig.reports[-1][0] == "aborted"


def test_interrupted_inward_move_never_applies_pressure():
    rig = Rig(interrupt=True)
    assert rig.sequence(params()).run() is False
    assert not [c for c in rig.calls if c[0] == "pressure"]
    assert "interrupted" in rig.reports[-1][2]


def test_move_that_stops_short_is_an_abort():
    rig = Rig(reach=False)
    assert rig.sequence(params()).run() is False
    assert not [c for c in rig.calls if c[0] == "pressure"]
    assert "stopped at X" in rig.reports[-1][2]


def test_stop_request_while_moving_stops_the_stage():
    rig = Rig(finish=False)
    threading.Timer(0.05, rig.abort.set).start()
    assert rig.sequence(params()).run() is False
    assert ("stop",) in rig.calls
    assert not [c for c in rig.calls if c[0] == "pressure"]


def test_move_timeout_stops_the_stage():
    rig = Rig(finish=False)
    now = [0.0]

    def clock():
        now[0] += 1.0
        return now[0]

    sequence = rig.sequence(params(step_um=10, speed_um_s=2000))
    sequence._clock = clock
    assert sequence.run() is False
    assert ("stop",) in rig.calls
    assert "timed out" in rig.reports[-1][2]


def test_pressure_failure_vents_stops_and_reraises():
    rig = Rig()

    def failing(mbar):
        rig.calls.append(("pressure", mbar))
        if mbar != 0.0:
            raise RuntimeError("publisher gone")

    rig.send_pressure = failing
    with pytest.raises(RuntimeError, match="publisher gone"):
        rig.sequence(params()).run()
    assert ("stop",) in rig.calls
    assert rig.calls[-1] == ("pressure", 0.0)
    assert rig.reports[-1][0] == "aborted"


def test_report_failure_after_completion_does_not_fail_the_injection():
    rig = Rig()
    original = rig.report

    def report(stage, active, message):
        original(stage, active, message)
        if stage == "done":
            raise RuntimeError("context shut down")

    rig.report = report
    assert rig.sequence(params()).run() is True

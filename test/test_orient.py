"""Orient skill tests against a simulated embryo; no ROS or hardware."""

import json
import math
import threading

import pytest

from orient_sim import SimEmbryo, calibration
from ump_suite.orient import (
    OrientParams,
    OrientSequence,
    decode_params_message,
    encode_params_message,
    validate_params,
)


def run(sim, params=None, cal="default", abort=None):
    reports = []
    seq = OrientSequence(
        params or OrientParams(settle_s=0.0),
        sim.cal if cal == "default" else cal,
        read_position=sim.read_position,
        move_to=sim.move_to,
        stop_stage=sim.stop_stage,
        get_scene=sim.get_scene,
        report=lambda *a: reports.append(a),
        abort_event=abort or threading.Event(),
        wall=sim.wall,
        sleep=lambda s: None,
    )
    return seq.run(), reports, seq


def test_already_oriented_moves_nothing():
    sim = SimEmbryo(cell_deg=5.0)
    result, reports, _ = run(sim)
    assert result["result"] == "already_oriented"
    assert sim.moves == []
    assert reports[-1][0] == "done"


@pytest.mark.parametrize("start", [-90.0, -150.0, 70.0])
def test_turns_the_cell_to_face_the_needle(start):
    sim = SimEmbryo(cell_deg=start)
    result, reports, seq = run(sim)
    assert result["result"] == "done", result
    assert abs(result["angle_deg"]) <= OrientParams().tolerance_deg
    assert 1 <= result["strokes"] <= OrientParams().max_strokes
    # the needle is parked outside the chorion, on the line through its centre
    tip = sim.tip()
    assert math.hypot(tip[0] - sim.cx, tip[1] - sim.cy) > sim.r
    assert abs(tip[1] - sim.cy) < 2.0
    # Z and D never move
    assert all(t[2] == 8000.0 and t[3] == 12000.0 for t, _ in sim.moves)
    assert seq.history and all(h["turned_deg"] * h["stroke_um"] > 0 for h in seq.history)


def test_adapts_when_the_embryo_turns_less_than_expected():
    sim = SimEmbryo(cell_deg=-100.0)
    sim.gain /= 3.0
    result, _, seq = run(sim, OrientParams(settle_s=0.0, max_strokes=50))
    assert result["result"] == "done", result
    assert seq.gain < math.degrees(1.0 / (2.0 * 370 * 1.8))


def test_an_embryo_that_does_not_turn_aborts_and_backs_off():
    sim = SimEmbryo(cell_deg=-90.0, gain=0.0)
    result, reports, seq = run(sim)
    assert result["result"] == "aborted"
    assert "does not turn" in result["message"] and "backed off" not in result["message"]
    assert [h["dent_um"] for h in seq.history][:3] == [40.0, 60.0, 80.0]
    tip = sim.tip()
    assert math.hypot(tip[0] - sim.cx, tip[1] - sim.cy) > sim.r


def test_turning_the_wrong_way_aborts():
    sim = SimEmbryo(cell_deg=-90.0)
    sim.gain = -sim.gain
    result, _, _ = run(sim)
    assert result["result"] == "aborted" and "wrong way" in result["message"]


def test_cell_on_top_aborts_without_moving():
    sim = SimEmbryo(cell_deg=-90.0, cell_on_face=True)
    result, _, _ = run(sim)
    assert result["result"] == "aborted" and "press" in result["message"]
    assert sim.moves == []


def test_no_calibration_aborts():
    sim = SimEmbryo()
    result, _, _ = run(sim, cal=None)
    assert result["result"] == "aborted" and "calibration" in result["message"]


def test_dry_run_reports_the_plan_and_sends_no_motion():
    sim = SimEmbryo(cell_deg=-90.0)
    result, reports, _ = run(sim, OrientParams(settle_s=0.0, dry_run=True))
    assert result["result"] == "aborted" and "dry run" in result["message"]
    assert sim.moves == []
    assert any(r[0] == "dry_run" and "stroke" in r[2] for r in reports)


def test_stop_before_the_first_move_sends_nothing():
    sim = SimEmbryo(cell_deg=-90.0)
    stop = threading.Event()
    stop.set()
    result, _, _ = run(sim, abort=stop)
    assert result["result"] == "aborted" and "stop" in result["message"]
    assert sim.moves == []


def test_needle_too_far_away_aborts():
    sim = SimEmbryo(cell_deg=-90.0, tip=(1400.0, 540.0))
    result, _, _ = run(sim, OrientParams(settle_s=0.0, approach_max_um=300.0))
    assert result["result"] == "aborted" and "closer" in result["message"]
    assert sim.moves == []


def test_a_bent_needle_is_noticed():
    sim = SimEmbryo(cell_deg=-90.0, bend=True)
    result, _, _ = run(sim)
    assert result["result"] == "aborted" and "bent or slipped" in result["message"]


def test_params_validation_and_round_trip():
    p = validate_params({"tolerance_deg": 10, "max_strokes": 5.0, "dry_run": 1})
    assert p.tolerance_deg == 10.0 and p.max_strokes == 5 and isinstance(p.max_strokes, int)
    assert p.dry_run is True
    q, token, seq = decode_params_message(encode_params_message(p, "gui", 4))
    assert q == p and token == "gui" and seq == 4
    for bad in (
        {"tolerance_deg": 0.5},
        {"stroke_max_um": 1000},
        {"speed_um_s": "fast"},
        {"stroke_min_um": 100, "stroke_max_um": 50},
        {"dent_um": 200, "dent_max_um": 100},
    ):
        with pytest.raises(ValueError):
            validate_params(bad)
    with pytest.raises(ValueError):
        decode_params_message("[1, 2]")
    assert json.loads(encode_params_message(OrientParams()))["dent_um"] == 40.0


def test_calibration_helper_moves_the_tip_up_with_positive_y():
    cal = calibration()
    ux, uy = cal.xy_move_for_pixels(0.0, -10.0)
    assert ux == pytest.approx(0.0) and uy == pytest.approx(18.0)

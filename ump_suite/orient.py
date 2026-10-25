"""
The orient skill: turn an embryo until its cell faces the needle.

A step-and-look loop on the cell's angle phi (from the vision scene, degrees;
0 = the cell faces the needle, positive = the embryo must turn clockwise on
screen to get there):

    1. if |phi| is already within tolerance on `confirm_frames` fresh frames,
       stop without moving ("already oriented")
    2. line the needle up with the chorion's centre (sideways move)
    3. advance until the tip is `dent_um` inside the chorion's outline
    4. one stroke sideways, along the chorion's rim:
           u = clip(loop_gain * (-phi) / gain, +/-stroke_max_um)
       a stroke "up" on screen at the needle-side rim turns the embryo
       counterclockwise, which raises phi
    5. back off to `backoff_um` outside the chorion and look again
    6. update this embryo's gain (degrees turned per um of stroke) from what the
       stroke actually did, deepen the dent if it did nothing
    7. repeat until |phi| is within tolerance on `confirm_frames` frames

The needle ends parked `backoff_um` outside the chorion on the line through its
centre (a last sideways move, since the final stroke rolls the embryo along):
the standard standoff MicroVLA resumes from. Every abort backs off first
if the needle is touching the chorion.

Motion is in the stage's X and Y only (Z and D never change). Pixel moves are
turned into X and Y moves with the camera calibration (calibration.py).

Nothing here imports ROS or the Sensapex SDK; the UMP driver hands in the
hardware callables, as for the injection macro.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, fields

from .vision.geometry import nearest_embryo, wrap_deg

AXIS_MIN_UM, AXIS_MAX_UM = 0, 20000
ARRIVAL_TOLERANCE_UM = 1.5
# How far the tip may end up from where the moves should have put it (pixels)
# before the skill assumes the needle bent or slipped.
# Micrometres, so the check means the same at any camera scale (it was 30 px at
# 1.8 um/px on the first camera).
TIP_LAG_LIMIT_UM = 55.0
# Strokes that turn the embryo by less than this count as "did nothing".
NO_TURN_DEG = 2.0


@dataclass(frozen=True)
class OrientParams:
    tolerance_deg: float = 15.0
    confirm_frames: int = 3
    max_strokes: int = 25
    stroke_max_um: float = 250.0
    stroke_min_um: float = 15.0
    dent_um: float = 40.0
    dent_max_um: float = 120.0
    dent_step_um: float = 20.0
    backoff_um: float = 60.0
    approach_max_um: float = 800.0
    gain_deg_per_um: float = 0.0  # 0 = estimate from the chorion's size
    gain_blend: float = 0.5
    loop_gain: float = 0.6
    speed_um_s: int = 200
    settle_s: float = 0.4
    scene_timeout_s: float = 2.0
    timeout_s: float = 300.0
    dry_run: bool = False

    def to_dict(self):
        return asdict(self)


LIMITS = {
    "tolerance_deg": (2.0, 90.0),
    "confirm_frames": (1, 30),
    "max_strokes": (1, 50),
    "stroke_max_um": (5.0, 400.0),
    "stroke_min_um": (1.0, 200.0),
    "dent_um": (0.0, 300.0),
    "dent_max_um": (0.0, 300.0),
    "dent_step_um": (0.0, 100.0),
    "backoff_um": (10.0, 500.0),
    "approach_max_um": (50.0, 3000.0),
    "gain_deg_per_um": (0.0, 5.0),
    "gain_blend": (0.0, 1.0),
    "loop_gain": (0.05, 1.5),
    "speed_um_s": (10, 2000),
    "settle_s": (0.0, 5.0),
    "scene_timeout_s": (0.2, 10.0),
    "timeout_s": (5.0, 600.0),
}


def validate_params(raw) -> OrientParams:
    """Return OrientParams from a mapping (missing keys take defaults); ValueError says why not."""
    if not isinstance(raw, dict):
        raise ValueError("orient parameters must be a JSON object")
    values = {}
    for f in fields(OrientParams):
        if f.name not in raw:
            continue
        value = raw[f.name]
        if f.name == "dry_run":
            values[f.name] = bool(value)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{f.name} must be a number, got {value!r}") from None
        if not math.isfinite(number):
            raise ValueError(f"{f.name} must be finite")
        lo, hi = LIMITS[f.name]
        if not lo <= number <= hi:
            raise ValueError(f"{f.name} {number:g} is outside {lo:g}..{hi:g}")
        values[f.name] = (
            int(round(number))
            if isinstance(f.default, int) and not isinstance(f.default, bool)
            else number
        )
    params = OrientParams(**values)
    if params.stroke_min_um > params.stroke_max_um:
        raise ValueError("stroke_min_um is larger than stroke_max_um")
    if params.dent_um > params.dent_max_um:
        raise ValueError("dent_um is larger than dent_max_um")
    return params


def encode_params_message(params, token="", seq=0):
    return json.dumps({"token": str(token), "seq": int(seq), **params.to_dict()})


def decode_params_message(text):
    try:
        raw = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"orient parameters are not JSON: {exc}") from None
    if not isinstance(raw, dict):
        raise ValueError("orient parameters must be a JSON object")
    return validate_params(raw), raw.get("token"), raw.get("seq")


def _yolk_in_view(embryo, scene, margin=2.0):
    """
    Return True when the yolk is fully inside the frame (else the chorion centre).

    The chorion's rim may be cut by the frame edge (it often is at this
    magnification); the angle is measured around the yolk, so that is what
    must stay in view.
    """
    w, h = scene.get("image_size") or (float("inf"), float("inf"))
    yolk = embryo.get("yolk")
    if yolk:
        cx, cy, r = yolk["cx"], yolk["cy"], yolk["r"]
    else:
        cx, cy, r = embryo["chorion"]["cx"], embryo["chorion"]["cy"], 0.0
    return margin <= cx - r and cx + r <= w - margin and margin <= cy - r and cy + r <= h - margin


class OrientAborted(Exception):
    pass


class OrientSequence:
    """
    Run the orient skill through injected callables.

    read_position() -> [x, y, z, d] um
    move_to(position, speed) -> MoveRequest-like (finished_event, interrupted,
        interrupt_reason, last_pos), as for the injection macro
    stop_stage()
    get_scene(after_wall_time, timeout) -> the first scene whose image was taken
        after `after_wall_time`, or None on timeout
    report(stage, active, message, info) -> publish progress
    abort_event: threading.Event set by stop requests
    calibration: vision.calibration.Calibration
    """

    def __init__(
        self,
        params,
        calibration,
        *,
        read_position,
        move_to,
        stop_stage,
        get_scene,
        report,
        abort_event,
        clock=time.monotonic,
        wall=time.time,
        sleep=time.sleep,
    ):
        self.p = params
        self.cal = calibration
        self._read_position = read_position
        self._move_to = move_to
        self._stop_stage = stop_stage
        self._get_scene = get_scene
        self._report = report
        self._abort = abort_event
        self._clock = clock
        self._wall = wall
        self._sleep = sleep
        self.strokes = 0
        self.gain = None
        self.angle = None
        self.history = []
        self._contact = False
        self._t0 = None

    # ── public ────────────────────────────────────────────────────────────
    def run(self):
        """Return a result dict; result is "done", "already_oriented" or "aborted"."""
        self._t0 = self._clock()
        try:
            result = self._run()
            self._safe(
                self._report, "done", False, result["message"], self._info(result["result"])
            )
            return result
        except OrientAborted as exc:
            reason = str(exc)
        except Exception as exc:  # SDK or ROS failure: stop, back off, re-raise
            self._safe(self._stop_stage)
            self._safe(self._report, "aborted", False, f"error: {exc}", self._info("aborted"))
            raise
        if self._contact and not self._abort.is_set():
            try:
                self._release()
                reason += "; backed off"
            except OrientAborted as exc:
                reason += f"; back-off failed: {exc}"
        self._safe(self._report, "aborted", False, reason, self._info("aborted"))
        return {
            "result": "aborted",
            "message": reason,
            "strokes": self.strokes,
            "angle_deg": self.angle,
            "gain_deg_per_um": self.gain,
        }

    # ── the loop ──────────────────────────────────────────────────────────
    def _run(self):
        p, cal = self.p, self.cal
        if cal is None:
            raise OrientAborted("no camera calibration; run the calibration first")
        scene = self._scene(self._wall())
        target = self._target(scene)
        embryo = scene["embryos"][target]
        self._check_embryo(embryo, scene)
        self._track = (embryo["chorion"]["cx"], embryo["chorion"]["cy"])
        self.angle = embryo["cell_angle_deg"]
        self._report(
            "measuring",
            True,
            f"cell at {self.angle:+.0f} deg from the needle",
            self._info("running"),
        )
        if self._confirm(self.p.tolerance_deg, first=scene):
            return {
                "result": "already_oriented",
                "message": f"already oriented: cell at {self.angle:+.0f} deg",
                "strokes": 0,
                "angle_deg": self.angle,
                "gain_deg_per_um": None,
            }
        r_um = embryo["chorion"]["r"] * cal.um_per_px
        self.gain = p.gain_deg_per_um or math.degrees(1.0 / (2.0 * r_um))
        dent = p.dent_um
        no_turn = wrong_way = 0
        while True:
            self._check_time()
            if self.strokes >= p.max_strokes:
                raise OrientAborted(
                    f"not oriented after {self.strokes} strokes "
                    f"(cell still at {self.angle:+.0f} deg)"
                )
            embryo, scene = self._current(scene)
            phi = embryo["cell_angle_deg"]
            u = p.loop_gain * (-phi) / self.gain
            u = math.copysign(min(max(abs(u), p.stroke_min_um), p.stroke_max_um), u)
            self._report(
                "stroke",
                True,
                f"stroke {self.strokes + 1}: cell at {phi:+.0f} deg, {u:+.0f} um "
                f"(gain {self.gain:.3f} deg/um, dent {dent:.0f} um)",
                self._info("running"),
            )
            expected_tip = self._cycle(scene, embryo, u, dent)
            self.strokes += 1
            after = self._scene(self._wall())
            embryo_after = self._find(after)
            self._check_tip(after, expected_tip)
            self._check_embryo(embryo_after, after)
            new = embryo_after["cell_angle_deg"]
            turned = wrap_deg(new - phi)
            self.history.append(
                {
                    "stroke_um": round(u, 1),
                    "before_deg": phi,
                    "after_deg": new,
                    "turned_deg": round(turned, 1),
                    "dent_um": dent,
                }
            )
            self.angle, scene = new, after
            if abs(turned) < NO_TURN_DEG:
                no_turn += 1
                wrong_way = 0
                if dent + p.dent_step_um <= p.dent_max_um and p.dent_step_um > 0:
                    dent += p.dent_step_um
                elif no_turn >= 2:
                    raise OrientAborted(
                        f"the embryo does not turn (dent {dent:.0f} um, "
                        f"{no_turn} strokes without effect)"
                    )
            elif turned * u < 0:
                wrong_way += 1
                no_turn = 0
                if wrong_way >= 2:
                    raise OrientAborted(
                        "the embryo turns the wrong way; check the camera "
                        "calibration's Y direction"
                    )
            else:
                no_turn = wrong_way = 0
                observed = turned / u
                lo, hi = self.gain / 10.0, self.gain * 10.0
                self.gain = (1 - p.gain_blend) * self.gain + p.gain_blend * min(
                    max(observed, lo), hi
                )
            if abs(new) <= p.tolerance_deg and self._confirm(p.tolerance_deg, first=after):
                self._park()
                return {
                    "result": "done",
                    "message": (f"oriented in {self.strokes} stroke(s): "
                                f"cell at {self.angle:+.0f} deg"),
                    "strokes": self.strokes,
                    "angle_deg": self.angle,
                    "gain_deg_per_um": self.gain,
                }

    def _cycle(self, scene, embryo, u, dent):
        """Align, touch, stroke, back off. Returns where the tip should end (pixels)."""
        cal, p = self.cal, self.p
        upx = cal.um_per_px
        tip = scene["needle"]["tip"]
        dx, dy = scene["needle"]["direction"]
        cx, cy, r = embryo["chorion"]["cx"], embryo["chorion"]["cy"], embryo["chorion"]["r"]
        nx, ny = -dy, dx
        offset = (cx - tip[0]) * nx + (cy - tip[1]) * ny
        along = (cx - tip[0]) * dx + (cy - tip[1]) * dy
        gap = along - r  # tip to the chorion's rim along the (aligned) needle line
        if gap < -dent / upx:
            raise OrientAborted("the tip is already inside the chorion; back the needle off first")
        if gap * upx > p.approach_max_um:
            raise OrientAborted(
                f"the needle is {gap * upx:.0f} um from the chorion "
                f"(limit {p.approach_max_um:.0f}); bring it closer first"
            )
        forward = gap + dent / upx
        # The rim point facing the needle and the screen direction that turns
        # the embryo counterclockwise there.
        a = math.atan2(dy, -dx)  # screen angle of the side the needle comes from
        tx, ty = -math.sin(a), -math.cos(a)
        stroke_px = u / upx
        back = dent / upx + p.backoff_um / upx
        moves = [
            ("align", offset * nx, offset * ny),
            ("touch", forward * dx, forward * dy),
            ("stroke", stroke_px * tx, stroke_px * ty),
            ("release", -back * dx, -back * dy),
        ]
        end = [tip[0], tip[1]]
        for label, mx, my in moves:
            end[0] += mx
            end[1] += my
        if p.dry_run:
            self._report(
                "dry_run",
                True,
                "dry run: "
                + ", ".join(f"{label} ({mx:+.0f}, {my:+.0f}) px" for label, mx, my in moves),
                self._info("running"),
            )
            raise OrientAborted("dry run: no motion sent")
        for label, mx, my in moves:
            self._move_pixels(mx, my, label)
            if label == "touch":
                self._contact = True
            elif label == "release":
                self._contact = False
        if p.settle_s:
            self._sleep(p.settle_s)
        return end

    # ── helpers ───────────────────────────────────────────────────────────
    def _move_pixels(self, mx, my, label):
        if abs(mx) < 1e-6 and abs(my) < 1e-6:
            return
        if self._abort.is_set():
            raise OrientAborted(f"stop requested before the {label} move")
        ux, uy = self.cal.xy_move_for_pixels(mx, my)
        pos = [float(v) for v in self._read_position()]
        target = list(pos)
        target[0] += ux
        target[1] += uy
        for i in (0, 1):
            if not AXIS_MIN_UM <= target[i] <= AXIS_MAX_UM:
                raise OrientAborted(f"the {label} move would leave the stage range")
        move = self._move_to(target, self.p.speed_um_s)
        dist = math.hypot(ux, uy)
        deadline = self._clock() + 3.0 * dist / self.p.speed_um_s + 2.0
        while not move.finished_event.wait(0.005):
            if self._abort.is_set():
                self._safe(self._stop_stage)
                raise OrientAborted(f"stop requested during the {label} move")
            if self._clock() > deadline:
                self._safe(self._stop_stage)
                raise OrientAborted(f"the {label} move timed out")
        if getattr(move, "interrupted", False):
            raise OrientAborted(
                f"the {label} move was interrupted: {getattr(move, 'interrupt_reason', '')}"
            )
        final = getattr(move, "last_pos", None) or self._read_position()
        if max(abs(float(final[i]) - target[i]) for i in (0, 1)) > ARRIVAL_TOLERANCE_UM:
            raise OrientAborted(f"the {label} move stopped short")

    def _park(self):
        """Line the needle up with the chorion's centre again; the last stroke rolled it."""
        scene = self._scene(self._wall())
        embryo = self._find(scene)
        tip = scene["needle"]["tip"]
        dx, dy = scene["needle"]["direction"]
        nx, ny = -dy, dx
        offset = (embryo["chorion"]["cx"] - tip[0]) * nx + (embryo["chorion"]["cy"] - tip[1]) * ny
        self._move_pixels(offset * nx, offset * ny, "park")

    def _release(self):
        scene = self._get_scene(self._wall(), self.p.scene_timeout_s)
        back = (self.p.dent_max_um + self.p.backoff_um) / self.cal.um_per_px
        if scene and scene.get("needle"):
            dx, dy = scene["needle"]["direction"]
        else:  # no fresh needle: back off along -X, the approach axis
            vx = self.cal.vx
            norm = math.hypot(*vx)
            dx, dy = vx[0] / norm, vx[1] / norm
        self._move_pixels(-back * dx, -back * dy, "back-off")
        self._contact = False

    def _scene(self, after):
        scene = self._get_scene(after, self.p.scene_timeout_s)
        if scene is None:
            raise OrientAborted("no fresh vision scene (is the vision node running with a model?)")
        return scene

    def _target(self, scene):
        if not scene.get("needle"):
            raise OrientAborted("no needle in view")
        if scene.get("target") is None:
            raise OrientAborted("the needle does not point at an embryo")
        return scene["target"]

    def _find(self, scene):
        idx = nearest_embryo(scene, *self._track)
        if idx is None:
            raise OrientAborted("lost the embryo (outline not found near where it was)")
        e = scene["embryos"][idx]
        self._track = (e["chorion"]["cx"], e["chorion"]["cy"])
        return e

    def _current(self, scene):
        return self._find(scene), scene

    def _check_embryo(self, embryo, scene):
        if not scene.get("needle"):
            raise OrientAborted("no needle in view")
        if embryo.get("cell") is None or embryo.get("cell_angle_deg") is None:
            raise OrientAborted("no cell found on this embryo")
        if embryo.get("cell_on_face"):
            raise OrientAborted(
                "the cell is on top of or under the yolk; it needs a press "
                "from above, which this skill does not do"
            )
        if not _yolk_in_view(embryo, scene):
            raise OrientAborted(
                "the embryo is leaving the view (its yolk is cut by the frame edge)")

    def _check_tip(self, scene, expected):
        tip = scene["needle"]["tip"] if scene.get("needle") else None
        if tip is None:
            raise OrientAborted("lost the needle")
        lag_um = math.hypot(tip[0] - expected[0], tip[1] - expected[1]) * self.cal.um_per_px
        if lag_um > TIP_LAG_LIMIT_UM:
            raise OrientAborted(
                f"the tip is {lag_um:.0f} um from where it was sent (needle bent or slipped?)"
            )

    def _confirm(self, tol, first):
        """Return True when `confirm_frames` consecutive fresh scenes keep |phi| <= tol."""
        count = 0
        scene = first
        while True:
            e = self._find(scene)
            phi = e.get("cell_angle_deg")
            if phi is None or abs(phi) > tol:
                return False
            self.angle = phi
            count += 1
            if count >= self.p.confirm_frames:
                return True
            scene = self._scene(float(scene.get("stamp") or self._wall()) + 1e-6)

    def _check_time(self):
        if self._abort.is_set():
            raise OrientAborted("stop requested")
        if self._clock() - self._t0 > self.p.timeout_s:
            raise OrientAborted(f"timed out after {self.p.timeout_s:.0f} s")

    def _info(self, result):
        return {
            "result": result,
            "strokes": self.strokes,
            "angle_deg": self.angle,
            "gain_deg_per_um": self.gain,
            "history": self.history[-5:],
        }

    @staticmethod
    def _safe(function, *args):
        try:
            function(*args)
        except Exception:
            pass

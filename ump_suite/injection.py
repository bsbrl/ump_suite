"""
The injection macro: move in along one axis, apply pressure briefly, vent, move back.

    1. move `axis` (X, Y, Z or D) by `step_um` at `speed_um_s` and wait for the
       SDK to report arrival
    2. request `pressure_mbar` and wait for the pressure driver to acknowledge it
    3. hold for `duration_ms`, measured from that acknowledgment
    4. vent (0 mbar)
    5. move that axis back to where it started, at the same speed; the other
       three axes are never commanded to change

The UMP driver runs this in a worker thread when `/inject/start` is called. The
parameters come from the GUI on `/inject/params`, so whoever triggers an
injection (the Inject button now, a policy later) uses the values the GUI shows
at that moment.

Nothing here imports ROS or the Sensapex SDK. Hardware access goes through the
callables handed to `InjectionSequence`, which keeps the sequence testable.
"""

import json
import math
import time
from dataclasses import asdict, dataclass

# Limits shared by the GUI boxes and the driver's validation. Speed matches the
# UMP panels; the pressure driver additionally clamps to the device range.
SPEED_MIN_UM_S, SPEED_MAX_UM_S = 10, 2000
STEP_LIMIT_UM = 2000
PRESSURE_LIMIT_MBAR = 1000.0
DURATION_MIN_MS, DURATION_MAX_MS = 1, 10000
AXIS_MIN_UM, AXIS_MAX_UM = 0, 20000
# Sensapex axis order, as in [x, y, z, d] positions and targets.
AXES = ("X", "Y", "Z", "D")

DEFAULT_SPEED_UM_S = 1000
DEFAULT_STEP_UM = 50
DEFAULT_PRESSURE_MBAR = 50.0
DEFAULT_DURATION_MS = 100
DEFAULT_AXIS = "X"

VENT_MBAR = 0.0
# A move counts as arrived within this distance of its target. The SDK itself
# retries until it is within 0.4 um, and the ROS readback is whole micrometres.
ARRIVAL_TOLERANCE_UM = 1.0
PRESSURE_ACK_TIMEOUT_S = 0.5


@dataclass(frozen=True)
class InjectionParams:
    """One injection's settings, in the units the GUI shows."""

    speed_um_s: int
    step_um: int
    pressure_mbar: float
    duration_ms: int
    axis: str = DEFAULT_AXIS

    @property
    def axis_index(self):
        return AXES.index(self.axis)

    def to_dict(self):
        return asdict(self)

    def describe(self):
        return (f"{self.axis} {self.step_um:+d} um at {self.speed_um_s} um/s, "
                f"{self.pressure_mbar:+.1f} mbar for {self.duration_ms} ms")


def _integer(raw, name):
    value = float(raw[name])
    if not math.isfinite(value) or value != int(value):
        raise ValueError(f"{name} must be a whole number, got {raw[name]!r}")
    return int(value)


def validate_params(raw):
    """
    Return InjectionParams from a mapping, or raise ValueError saying why not.

    A mapping without "axis" means X, the only axis before the selector existed.
    """
    try:
        axis = str(raw.get("axis", DEFAULT_AXIS)).strip().upper()
        speed = _integer(raw, "speed_um_s")
        step = _integer(raw, "step_um")
        duration = _integer(raw, "duration_ms")
        pressure = float(raw["pressure_mbar"])
    except KeyError as exc:
        raise ValueError(f"missing injection parameter {exc.args[0]!r}") from None
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid injection parameter: {exc}") from None
    if axis not in AXES:
        raise ValueError(f"axis must be one of {', '.join(AXES)}, got {raw.get('axis')!r}")
    if not SPEED_MIN_UM_S <= speed <= SPEED_MAX_UM_S:
        raise ValueError(
            f"speed {speed} um/s is outside {SPEED_MIN_UM_S}..{SPEED_MAX_UM_S}")
    if step == 0 or abs(step) > STEP_LIMIT_UM:
        raise ValueError(f"step must be nonzero and within +/-{STEP_LIMIT_UM} um, got {step}")
    if not math.isfinite(pressure) or abs(pressure) > PRESSURE_LIMIT_MBAR:
        raise ValueError(
            f"pressure must be within +/-{PRESSURE_LIMIT_MBAR:g} mbar, got {pressure}")
    if not DURATION_MIN_MS <= duration <= DURATION_MAX_MS:
        raise ValueError(
            f"time must be {DURATION_MIN_MS}..{DURATION_MAX_MS} ms, got {duration}")
    return InjectionParams(speed, step, pressure, duration, axis)


def encode_params_message(params, token, seq):
    """JSON for /inject/params. token and seq let the GUI see its own echo."""
    return json.dumps({"token": str(token), "seq": int(seq), **params.to_dict()})


def decode_params_message(text):
    """Return (params, token, seq) from /inject/params JSON; raise ValueError."""
    try:
        raw = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"injection parameters are not JSON: {exc}") from None
    if not isinstance(raw, dict):
        raise ValueError("injection parameters must be a JSON object")
    return validate_params(raw), raw.get("token"), raw.get("seq")


def forward_target(start, params):
    """Target of the inward move on the chosen axis; refuse leaving the stage range."""
    target = float(start[params.axis_index]) + params.step_um
    if not AXIS_MIN_UM <= target <= AXIS_MAX_UM:
        raise ValueError(
            f"{params.axis} target {target:.1f} um is outside the stage range "
            f"{AXIS_MIN_UM}..{AXIS_MAX_UM} um")
    return target


def move_timeout_s(params):
    """Generous allowance for one move of |step| at speed, including acceleration."""
    return 3.0 * abs(params.step_um) / params.speed_um_s + 2.0


class InjectionAborted(Exception):
    """The sequence stopped early. `retract` asks for the return move anyway."""

    def __init__(self, reason, retract=False):
        super().__init__(reason)
        self.retract = retract


class InjectionSequence:
    """
    Run one injection through injected hardware callables.

    read_position() -> [x, y, z, d] in um (floats)
    move_to(position, speed) -> an SDK MoveRequest-like object with
        finished_event, interrupted, interrupt_reason and last_pos
    stop_stage() -> stop motion now
    send_pressure(mbar) -> publish a pressure request
    wait_pressure_ack(after, timeout) -> applied mbar, or None if no
        acknowledgment arrived after monotonic time `after`
    report(stage, active, message) -> publish progress
    abort_event: a threading.Event that stop requests set
    """

    def __init__(self, params, *, read_position, move_to, stop_stage, send_pressure,
                 wait_pressure_ack, report, abort_event, clock=time.monotonic):
        self.params = params
        self._read_position = read_position
        self._move_to = move_to
        self._stop_stage = stop_stage
        self._send_pressure = send_pressure
        self._wait_pressure_ack = wait_pressure_ack
        self._report = report
        self._abort = abort_event
        self._clock = clock
        self.timings = {}

    def run(self, start=None):
        """Run to completion; return True if every step finished as planned."""
        params = self.params
        axis, index = params.axis, params.axis_index
        pressure_on = vented = arrived = False
        start = list(self._read_position() if start is None else start)
        inward = list(start)
        inward[index] = forward_target(start, params)
        t0 = self._clock()
        try:
            self._report("forward", True, f"moving {axis} to {inward[index]:.1f} um")
            self._move(inward, "inward")
            arrived = True
            self.timings["inward_s"] = self._clock() - t0

            self._report("pressure", True, f"applying {params.pressure_mbar:+.1f} mbar")
            sent = self._clock()
            pressure_on = True
            self._send_pressure(params.pressure_mbar)
            applied = self._wait_pressure_ack(sent, PRESSURE_ACK_TIMEOUT_S)
            if applied is None:
                raise InjectionAborted(
                    "the pressure driver did not acknowledge the request", retract=True)
            acknowledged = self._clock()
            self.timings["ack_s"] = acknowledged - sent
            self.timings["applied_mbar"] = float(applied)
            if self._abort.wait(params.duration_ms / 1000.0):
                raise InjectionAborted("stop requested while pressure was applied")
            self._send_pressure(VENT_MBAR)
            vented = True
            self.timings["held_s"] = self._clock() - acknowledged

            self._report("back", True, f"moving {axis} back to {start[index]:.1f} um")
            back_started = self._clock()
            self._move(start, "return")
            self.timings["return_s"] = self._clock() - back_started
            self.timings["total_s"] = self._clock() - t0
            self._safe(self._report, "done", False, self._summary())
            return True
        except InjectionAborted as exc:
            reason = str(exc)
            retract = exc.retract
        except Exception as exc:  # SDK or ROS failure: stop and vent, then re-raise
            self._safe(self._stop_stage)
            if pressure_on and not vented:
                self._safe(self._send_pressure, VENT_MBAR)
            self._safe(self._report, "aborted", False, f"error: {exc}")
            raise

        if pressure_on and not vented:
            self._safe(self._send_pressure, VENT_MBAR)
            reason += "; vented"
        if retract and arrived and not self._abort.is_set():
            try:
                self._move(start, "return")
                reason += "; moved back"
            except InjectionAborted as exc:
                reason += f"; return move failed: {exc}"
        self._safe(self._report, "aborted", False, reason)
        return False

    def _move(self, position, label):
        if self._abort.is_set():
            raise InjectionAborted(f"stop requested before the {label} move")
        move = self._move_to(position, self.params.speed_um_s)
        deadline = self._clock() + move_timeout_s(self.params)
        while not move.finished_event.wait(0.005):
            if self._abort.is_set():
                self._safe(self._stop_stage)
                raise InjectionAborted(f"stop requested during the {label} move")
            if self._clock() > deadline:
                self._safe(self._stop_stage)
                raise InjectionAborted(f"the {label} move timed out")
        if getattr(move, "interrupted", False):
            raise InjectionAborted(
                f"the {label} move was interrupted: {getattr(move, 'interrupt_reason', '')}")
        final = getattr(move, "last_pos", None)
        if final is None:
            final = self._read_position()
        axis, index = self.params.axis, self.params.axis_index
        if abs(float(final[index]) - float(position[index])) > ARRIVAL_TOLERANCE_UM:
            raise InjectionAborted(
                f"the {label} move stopped at {axis} {float(final[index]):.1f} um, "
                f"not {float(position[index]):.1f} um")

    def _summary(self):
        t = self.timings
        ms = {key: round(value * 1000) for key, value in t.items() if key.endswith("_s")}
        return (f"{ms['total_s']} ms total (in {ms['inward_s']}, ack {ms['ack_s']}, "
                f"held {ms['held_s']}, back {ms['return_s']} ms) at "
                f"{t['applied_mbar']:+.1f} mbar")

    @staticmethod
    def _safe(function, *args):
        try:
            function(*args)
        except Exception:
            pass

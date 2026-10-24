"""A simulated embryo, needle and stage for the orient skill; no ROS, no hardware."""

import math
import threading

from ump_suite.vision.calibration import Calibration
from ump_suite.vision.geometry import wrap_deg

UM_PER_PX = 1.8


def calibration(flip_y=False, um_per_px=UM_PER_PX):
    s = 1.0 / um_per_px
    return Calibration(
        um_per_px=um_per_px,
        vx=[-s, 0.0],
        vy=[0.0, s if flip_y else -s],
        vd=[-s * math.cos(math.radians(30)), 0.0],
        theta_deg=30.0,
        image_size=[1440, 1080],
    )


class Move:
    def __init__(self, pos):
        self.finished_event = threading.Event()
        self.finished_event.set()
        self.interrupted = False
        self.interrupt_reason = None
        self.last_pos = list(pos)


class SimEmbryo:
    """
    One embryo in a trough, a horizontal needle from the right.

    While the tip is inside the chorion, a sideways move turns the embryo by
    `gain` degrees per um of movement along the rim (counterclockwise for a
    stroke up the needle-side rim) and moves it along by `roll` of the stroke
    (0 = turns in place; 0.5 = rolls along a trough wall).
    """

    def __init__(
        self,
        cell_deg=-90.0,
        gain=None,
        tip=(1000.0, 540.0),
        centre=(500.0, 540.0),
        r=370.0,
        bend=False,
        cell_on_face=False,
        roll=0.0,
    ):
        self.cal = calibration()
        self.pos = [10000.0, 10000.0, 8000.0, 12000.0]
        self.pos0 = list(self.pos)
        self.tip0 = tip
        self.cx, self.cy, self.r = centre[0], centre[1], r
        self.cell = cell_deg
        r_um = r * UM_PER_PX
        self.gain = math.degrees(1.0 / (2.0 * r_um)) if gain is None else gain
        self.bend = bend
        self.cell_on_face = cell_on_face
        self.roll = roll  # fraction of each stroke the embryo moves along (0.5 = against a wall)
        self.moves = []
        self.clock = 1000.0

    # stage
    def tip(self):
        if self.bend:
            return self.tip0
        jx, jy = self.cal.vx, self.cal.vy
        ux, uy = self.pos[0] - self.pos0[0], self.pos[1] - self.pos0[1]
        return (self.tip0[0] + jx[0] * ux + jy[0] * uy, self.tip0[1] + jx[1] * ux + jy[1] * uy)

    def read_position(self):
        return list(self.pos)

    def move_to(self, target, speed):
        before = self.tip()
        self.pos[:] = [float(v) for v in target]
        after = self.tip()
        self.moves.append((list(target), speed))
        inside = math.hypot(before[0] - self.cx, before[1] - self.cy) < self.r
        if inside and not self.bend:
            # contact point faces the needle (right); CCW tangent there is "up" (-y)
            tangential_px = -(after[1] - before[1])
            tangential_um = tangential_px * UM_PER_PX
            self.cell += self.gain * tangential_um
            self.cy -= tangential_px * self.roll
        return Move(self.pos)

    def stop_stage(self):
        pass

    # vision
    def scene(self):
        tip = self.tip()
        a = math.radians(self.cell)
        yolk = (self.cx - 60, self.cy, 180)
        cell = (yolk[0] + 200 * math.cos(a), yolk[1] - 200 * math.sin(a))
        hits = abs(tip[1] - self.cy) < self.r and tip[0] > self.cx - self.r
        return {
            "version": 1,
            "stamp": self.clock,
            "image_size": [1440, 1080],
            "um_per_px": None,
            "embryos": [
                {
                    "id": 0,
                    "chorion": {"cx": self.cx, "cy": self.cy, "r": self.r},
                    "complete": True,
                    "yolk": {"cx": yolk[0], "cy": yolk[1], "r": yolk[2]},
                    "cell": {"cx": cell[0], "cy": cell[1], "area": 9000},
                    "cell_angle_image_deg": wrap_deg(self.cell),
                    "cell_angle_deg": wrap_deg(self.cell),
                    "cell_on_face": self.cell_on_face,
                    "score": 1.0,
                }
            ],
            "needle": {
                "tip": list(tip),
                "base": [1439.0, tip[1]],
                "direction": [-1.0, 0.0],
                "length_px": 400,
            },
            "target": 0 if hits else None,
            "ray": None,
        }

    def get_scene(self, after, timeout):
        self.clock = max(self.clock, after) + 0.1
        return self.scene()

    def wall(self):
        return self.clock

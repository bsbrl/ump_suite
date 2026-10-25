"""Vision geometry, needle and calibration tests on synthetic frames with known answers."""

import math

import numpy as np
import pytest

from ump_suite.vision import calibration as cal
from ump_suite.vision.geometry import (
    EMBRYO_FEATURE_DIM,
    Instance,
    build_scene,
    embryo_features,
    fit_circle,
    injection_step_um,
    nearest_embryo,
    wrap_deg,
)
from ump_suite.vision.needle import refine_tip, tip_from_difference

H, W = 1080, 1440


def disc(cx, cy, r, shape=(H, W)):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def needle_mask(tip, shape=(H, W), width=6):
    """Return a horizontal needle mask from the right edge to `tip`."""
    m = np.zeros(shape, dtype=bool)
    x0, y = int(tip[0]), int(tip[1])
    half = width // 2
    m[y - half:y + half + 1, x0:] = True
    return m


def embryo(cx, cy, cell_angle_deg, r=370, yolk_r=180, cell_r=70, yolk_off=(-60, 0)):
    """Instances for one embryo whose cell sits at `cell_angle_deg` (screen) around the yolk."""
    yx, yy = cx + yolk_off[0], cy + yolk_off[1]
    a = math.radians(cell_angle_deg)
    ccx, ccy = yx + (yolk_r + 20) * math.cos(a), yy - (yolk_r + 20) * math.sin(a)
    return [
        Instance("chorion", disc(cx, cy, r)),
        Instance("yolk", disc(yx, yy, yolk_r)),
        Instance("cell", disc(ccx, ccy, cell_r) & ~disc(yx, yy, yolk_r)),
    ]


def test_fit_circle_recovers_centre_and_radius():
    t = np.linspace(0, 2 * np.pi, 200)
    pts = np.column_stack([300 + 120 * np.cos(t), 200 + 120 * np.sin(t)])
    cx, cy, r = fit_circle(pts)
    assert (cx, cy, r) == pytest.approx((300, 200, 120), abs=1e-6)


@pytest.mark.parametrize(
    "angle,expected", [(0, 0), (-90, -90), (90, 90), (150, 150), (-150, -150)]
)
def test_cell_angle_is_measured_from_the_needle_side(angle, expected):
    inst = embryo(500, 540, angle) + [Instance("needle", needle_mask((1100, 540)))]
    scene = build_scene(inst, (H, W))
    e = scene["embryos"][0]
    assert e["cell_angle_image_deg"] == pytest.approx(angle, abs=2.0)
    assert e["cell_angle_deg"] == pytest.approx(expected, abs=2.0)
    assert e["cell_on_face"] is False


def test_cell_over_the_yolk_middle_is_flagged():
    yx, yy = 440, 540
    inst = [
        Instance("chorion", disc(500, 540, 370)),
        Instance("yolk", disc(yx, yy, 180)),
        Instance("cell", disc(yx + 30, yy, 60)),
    ]
    assert build_scene(inst, (H, W))["embryos"][0]["cell_on_face"] is True


def test_needle_target_and_distances_along_the_line():
    inst = embryo(500, 540, 0) + [Instance("needle", needle_mask((1000, 540)))]
    scene = build_scene(inst, (H, W), um_per_px=2.0)
    assert scene["target"] == 0
    tip = scene["needle"]["tip"]
    assert tip[0] == pytest.approx(1000, abs=2)
    assert scene["needle"]["direction"][0] == pytest.approx(-1.0, abs=0.01)
    ray = scene["ray"]
    # chorion edge at x = 870, cell near edge at 440 + 200 + 70 = 710
    assert ray["chorion_px"] == pytest.approx(130, abs=3)
    assert ray["hits_cell"] is True
    assert ray["cell_near_px"] == pytest.approx(290, abs=3)
    assert ray["cell_near_um"] == pytest.approx(580, abs=6)
    assert ray["cell_far_px"] > ray["cell_near_px"]


def test_line_that_misses_the_cell_reports_no_hit():
    inst = embryo(500, 540, -90) + [Instance("needle", needle_mask((1000, 540)))]
    ray = build_scene(inst, (H, W))["ray"]
    assert ray["hits_cell"] is False and ray["cell_near_px"] is None


def test_embryos_are_ordered_top_to_bottom_and_the_first_hit_is_the_target():
    top = embryo(700, 200, 0, r=150, yolk_r=70, cell_r=30, yolk_off=(-20, 0))
    low = embryo(700, 800, 0, r=150, yolk_r=70, cell_r=30, yolk_off=(-20, 0))
    needle = Instance("needle", needle_mask((1100, 800)))
    scene = build_scene(low + top + [needle], (H, W))
    ys = [e["chorion"]["cy"] for e in scene["embryos"]]
    assert ys == sorted(ys)
    assert scene["target"] == 1  # the lower embryo is the one in line with the needle
    assert nearest_embryo(scene, 705, 205) == 0
    assert nearest_embryo(scene, 100, 100) is None


def test_chorion_cut_by_the_frame_edge_is_not_complete():
    inst = embryo(200, 540, 0)  # radius 370 crosses x = 0
    assert build_scene(inst, (H, W))["embryos"][0]["complete"] is False


def test_no_needle_means_no_target_and_no_relative_angle():
    scene = build_scene(embryo(500, 540, 30), (H, W))
    assert scene["needle"] is None and scene["target"] is None
    assert scene["embryos"][0]["cell_angle_deg"] is None


def test_embryo_features_shape_and_mask():
    scene = build_scene(embryo(500, 540, 90), (H, W))
    feats, valid = embryo_features(scene, n_max=4)
    assert feats.shape == (4, EMBRYO_FEATURE_DIM)
    assert valid.tolist() == [True, False, False, False]
    assert feats[0, 0] == pytest.approx(500 / W, abs=0.01)
    assert feats[0, 4] == pytest.approx(1.0, abs=0.05)  # sin(90 deg)


def test_injection_step_formula_and_clamp():
    step, raw = injection_step_um(
        300, depth_um=50, margin_um=40, theta_deg=60, min_um=100, max_um=2000
    )
    assert raw == pytest.approx(350 / 0.5 + 40)
    assert step == 740
    assert (
        injection_step_um(10, depth_um=0, margin_um=0, theta_deg=0, min_um=100, max_um=500)[0]
        == 100
    )
    assert (
        injection_step_um(900, depth_um=0, margin_um=0, theta_deg=0, min_um=100, max_um=500)[0]
        == 500
    )
    with pytest.raises(ValueError):
        injection_step_um(None, depth_um=0, margin_um=0, theta_deg=0, min_um=0, max_um=1)
    with pytest.raises(ValueError):
        injection_step_um(100, depth_um=0, margin_um=0, theta_deg=89, min_um=0, max_um=2000)


def test_wrap_deg():
    assert wrap_deg(190) == pytest.approx(-170)
    assert wrap_deg(-180) == 180
    assert wrap_deg(360) == 0


def test_refine_tip_follows_a_faint_tip_past_the_mask():
    gray = np.full((200, 400), 220, np.uint8)
    gray[98:103, 150:] = 120  # shank
    gray[99:102, 110:150] = 205  # faint tip, 15 grey levels darker
    tip = refine_tip(gray, (150, 100), (-1.0, 0.0), contrast=6)
    assert tip[0] == pytest.approx(110, abs=2)


def test_tip_from_difference_finds_the_needle_entering_from_the_edge():
    bg = np.full((300, 500), 225, np.uint8)
    bg[50:80, 40:70] = 90  # a static object stays out of the difference
    frame = bg.copy()
    frame[148:153, 260:] = 120
    found = tip_from_difference(frame, bg)
    assert found is not None
    (tx, ty), (dx, dy) = found
    assert tx == pytest.approx(260, abs=3) and ty == pytest.approx(150, abs=2)
    assert dx == pytest.approx(-1.0, abs=0.02)
    assert tip_from_difference(bg, bg) is None


def _samples(um_per_px, theta_deg, rot_deg, flip_y=False):
    """Return moves seen by a camera rotated by rot_deg; +X goes left on screen like the rig."""
    s = 1.0 / um_per_px
    r = math.radians(rot_deg)
    vx = np.array([-math.cos(r), math.sin(r)]) * s
    vy = np.array([math.sin(r), math.cos(r)]) * s * (-1 if flip_y else 1)
    vd = vx * math.cos(math.radians(theta_deg))
    base = np.array([700.0, 500.0])
    out = []
    for axis, v in (("X", vx), ("Y", vy), ("D", vd)):
        for delta in (100.0, -100.0):
            out.append(
                (axis, delta, tuple(base), tuple(base + v * delta + np.random.normal(0, 0.3, 2)))
            )
    return out


def test_calibration_fit_recovers_scale_tilt_and_axes(tmp_path):
    np.random.seed(0)
    c = cal.fit(_samples(1.8, 30, 3.0), (1440, 1080))
    assert c.um_per_px == pytest.approx(1.8, rel=0.01)
    assert c.theta_deg == pytest.approx(30, abs=1.0)
    # +X moves the tip left on screen
    assert c.vx[0] < 0
    mx, my = c.xy_move_for_pixels(-55.5, 0.0)
    assert math.hypot(mx, my) == pytest.approx(100, rel=0.02)
    path = c.save(tmp_path / "cal.json")
    again = cal.load(path)
    assert again.um_per_px == pytest.approx(c.um_per_px)
    assert cal.load(tmp_path / "missing.json") is None


def test_calibration_rejects_mismatched_axes():
    samples = [
        ("X", 100, (0, 0), (-55, 0)),
        ("Y", 100, (0, 0), (0, -30)),
        ("D", 100, (0, 0), (-40, 0)),
    ]
    with pytest.raises(ValueError, match="disagree"):
        cal.fit(samples, (1440, 1080))
    with pytest.raises(ValueError, match="no usable D"):
        cal.fit(samples[:2], (1440, 1080))


def test_eval_compare_scores_identical_and_shifted_frames():
    """A perfect prediction scores IoU 1 and zero errors; a shifted cell costs angle."""
    from ump_suite.vision.tools.eval_segmenter import compare, summarise

    truth = embryo(500, 540, 0) + [Instance("needle", needle_mask((1000, 540)))]
    same = compare(truth, truth, (H, W))
    assert same["iou_chorion"] == 1.0 and same["iou_cell"] == 1.0
    assert same["cell_angle_deg"] == 0.0 and same["tip_px"] == 0.0
    turned = embryo(500, 540, 30) + [Instance("needle", needle_mask((1000, 540)))]
    row = compare(truth, turned, (H, W))
    assert row["cell_angle_deg"] == pytest.approx(30, abs=3)
    assert row["iou_cell"] < 1.0
    stats = summarise([same, row])
    assert stats["cell_angle_deg"]["n"] == 2


def _needle_frames(px_per_um=0.5, tip0=200.0, width=500, start_x=10000.0):
    """Grey frames of a needle from the right edge whose tip moves right as X is pulled back."""
    state = {"x": start_x}

    def goto(pos):
        state["x"] = pos[0]

    def grab():
        img = np.full((120, width), 220, np.uint8)
        tip = tip0 + (start_x - state["x"]) * px_per_um
        if tip < width:
            img[58:63, int(tip):] = 120
        return img

    return goto, grab, state


def test_find_background_pulls_back_until_the_needle_is_gone():
    """The first step that changes nothing gives a needle-free background."""
    from ump_suite.calibrate_camera import find_background

    goto, grab, state = _needle_frames()
    background, pulled = find_background(goto, grab, [10000.0, 0, 0, 0], step_um=200, max_um=6000)
    assert (background == 220).all()
    # the tip leaves the 500 px view after 600 um (300 px / 0.5 px per um); one more step confirms
    assert pulled == 800
    assert state["x"] == 10000.0 - 800


def test_find_background_gives_up_and_respects_the_stage_range():
    """A needle that never leaves, or a pull-back past X = 0, raises instead of guessing."""
    from ump_suite.calibrate_camera import find_background

    goto, grab, _ = _needle_frames(px_per_um=0.01)
    with pytest.raises(RuntimeError, match="still in view"):
        find_background(goto, grab, [10000.0, 0, 0, 0], step_um=500, max_um=1500)
    goto, grab, _ = _needle_frames()
    with pytest.raises(RuntimeError, match="stage range"):
        find_background(goto, grab, [300.0, 0, 0, 0], step_um=500, max_um=6000)


def test_a_needle_sliding_along_its_own_axis_still_counts_as_change():
    """Only the tip region changes when the needle slides; that is still change."""
    from ump_suite.vision.needle import changed_area

    goto, grab, _ = _needle_frames()
    before = grab()
    goto([10000.0 - 200, 0, 0, 0])
    assert changed_area(grab(), before) > 150


def test_coco_conversion_skips_unlabelled_frames_and_brush_masks(tmp_path):
    import json

    from ump_suite.vision.tools.labels_to_yolo import from_coco

    coco = {
        "images": [
            {"id": 1, "file_name": "a__t1__f0.png", "width": 100, "height": 50},
            {"id": 2, "file_name": "a__t1__f4.png", "width": 100, "height": 50},
        ],
        # CVAT's order, not ours: the written class numbers must follow the names.
        "categories": [{"id": 7, "name": "Needle"}, {"id": 8, "name": "chorion"},
                       {"id": 9, "name": "bubble"}],
        "annotations": [
            {"image_id": 1, "category_id": 7, "segmentation": [[90, 20, 99, 20, 99, 30]]},
            {"image_id": 1, "category_id": 8, "segmentation": {"counts": "x", "size": [50, 100]}},
            {"image_id": 1, "category_id": 9, "segmentation": [[0, 0, 5, 0, 5, 5]]},
        ],
    }
    (tmp_path / "export.json").write_text(json.dumps(coco))
    written, unlabelled, skipped = from_coco(tmp_path / "export.json", tmp_path / "labels")
    assert (written, unlabelled) == (1, 1)
    assert skipped["<brush mask, not a polygon>"] == 1 and skipped["bubble"] == 1
    assert not (tmp_path / "labels" / "a__t1__f4.txt").exists()
    line = (tmp_path / "labels" / "a__t1__f0.txt").read_text().split()
    assert line[0] == "3" and line[1:3] == ["0.90000", "0.40000"]

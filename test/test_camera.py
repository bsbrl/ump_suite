"""Raw-Bayer conversion in the camera node, checked without a camera."""

import sys
import types

import numpy as np
import pytest

pytest.importorskip("cv2")
# The node imports PySpin at module level but only uses it on a live camera.
sys.modules.setdefault("PySpin", types.ModuleType("PySpin"))
from ump_suite.camera_node import _BAYER_TO_BGR, bayer_to_bgr  # noqa: E402

LAYOUTS = {"BayerRG8": "RGGB", "BayerGB8": "GBRG", "BayerGR8": "GRBG", "BayerBG8": "BGGR"}


def _mosaic(layout, rgb, shape=(8, 8)):
    """Sample a uniform colour through a 2x2 Bayer `layout` such as 'RGGB'."""
    value = dict(zip("RGB", rgb))
    raw = np.zeros(shape, np.uint8)
    for k, channel in enumerate(layout):
        raw[k // 2::2, k % 2::2] = value[channel]
    return raw


@pytest.mark.parametrize("fmt", sorted(LAYOUTS))
def test_each_bayer_layout_puts_each_colour_in_its_channel(fmt):
    assert set(_BAYER_TO_BGR) == set(LAYOUTS)
    raw = _mosaic(LAYOUTS[fmt], (200, 100, 50))
    bgr = bayer_to_bgr(raw, _BAYER_TO_BGR[fmt])
    assert bgr.shape == (8, 8, 3)
    assert (bgr[2:-2, 2:-2] == (50, 100, 200)).all()


def test_the_colour_matrix_is_applied_in_bgr_order():
    raw = _mosaic("RGGB", (200, 100, 50))
    halve_red = np.diag([1.0, 1.0, 0.5]).astype(np.float32)
    bgr = bayer_to_bgr(raw, _BAYER_TO_BGR["BayerRG8"], halve_red)
    assert (bgr[2:-2, 2:-2] == (50, 100, 100)).all()
    saturate = np.diag([1.0, 1.0, 2.0]).astype(np.float32)
    assert (bayer_to_bgr(raw, _BAYER_TO_BGR["BayerRG8"], saturate)[2:-2, 2:-2, 2] == 255).all()

"""
Embryo vision: segmentation outlines -> geometry the rig and MicroVLA act on.

    segmenter.py    YOLO-seg wrapper (ultralytics, imported lazily)
    geometry.py     outlines -> a scene: embryos, cell angle, needle, the
                    embryo at the needle and the needle-to-cell distance
    needle.py       needle line, tip refinement, tip from a frame difference
    calibration.py  µm per pixel, image <-> uMp axes, D-axis tilt
    tools/          labelling, training and evaluation scripts

Nothing in this package imports ROS, so MicroVLA and the tests use it directly.
"""

CLASS_NAMES = ("chorion", "yolk", "cell", "needle")

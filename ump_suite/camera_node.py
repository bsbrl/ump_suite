"""
ROS2 driver for a FLIR Blackfly camera via PySpin.

Each frame is grabbed in a worker thread, then:
  * a JPEG-compressed copy is published at `publish_hz` for the GUI / VLA client
  * the FPS achieved by the grabber is published on `/camera/fps`
  * if recording is active, the raw frame is appended to an mp4 video file

Recording is controlled by a String message on /camera/record_cmd: a non-empty
path starts recording to that file, an empty string stops it.

=== Resolution and frame rate ===
``binning`` combines b x b pixels in the camera, so the image keeps the whole
field of view at 1/b the width and height. The 20 MP BFS-U3-200S6C (5472 x 3648)
runs with 2: full-resolution frames are 60 MB in BGR8, and every consumer (JPEG
encode, the GUI, a 3 Hz PNG logger) falls behind on them.
``roi_width``/``roi_height``/``roi_offset_*`` then crop the binned image in the
camera. The 20 MP sensor sees out to the edge of the microscope's light circle,
which sits left of the sensor's centre, so the launch file crops to the lit 4:3
part.

While the camera's image processor (ISP) is on, which every colour format
needs, this model runs at 6 fps whatever the binning or crop. ``raw_bayer``
turns the ISP off and takes the 8-bit Bayer mosaic instead (FLIR's table: 16 fps
at full resolution, 32 fps binned 2x2); this node then demosaics it and applies
the colour matrix the ISP would have, read from the camera at startup (measured
against the ISP's own output: within 4 grey levels). White balance and exposure
stay in the camera. With the ISP off the camera averages binned pixels instead
of summing them, so ``gain_db`` (12 dB = x4) restores the brightness the ISP's
"Sum" gave at the same exposure. ``link_limit_mb_s`` raises the USB bandwidth
the camera may use (it powers up at 60 MB/s) and ``frame_rate`` caps the rate;
keep it below ``publish_hz`` or the preview throttle skips every other frame.

USB3 cameras this size also need more USB buffer memory than Linux's 16 MB
default (``/sys/module/usbcore/parameters/usbfs_memory_mb``; FLIR recommends
1000). With the default, ``Init()`` fails with Spinnaker error -1016.

=== Image brightness ===
The camera powers up with ``ExposureAuto = Continuous`` and an automatically
chosen target grey level, which renders a bright brightfield background at
roughly mid grey. That is why the live view looks much dimmer than the eyepiece
even when the light path is fine. ``target_grey_percent`` sets that target
explicitly; ``exposure_time_us`` overrides the loop entirely.

Auto exposure also has a subtler cost for dataset collection. With average
metering, a dark pipette entering the frame lowers the average, so the loop
brightens the whole scene: background brightness ends up encoding manipulator
position. Measured on this rig, frame mean correlates with pipette x at
r = -0.99. ``lock_exposure_while_recording`` freezes exposure and gain for the
duration of each trial so recorded frames are photometrically consistent.
"""

import threading
import time

import cv2
import numpy as np
import PySpin
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, String

from .ros_interfaces import (
    TOPIC_CAM_FPS,
    TOPIC_CAM_IMAGE_COMPRESSED,
    TOPIC_CAM_REC_CMD,
)


CAM_GET_TIMEOUT_MS = 1000


# The camera's 8-bit Bayer layouts and OpenCV's demosaic codes. OpenCV names a
# layout by the second row's second and third pixels, so RGGB is its "BG"
# (checked on the BFS-U3-200S6C: the (0, 0) phase follows the red balance ratio).
_BAYER_TO_BGR = {
    "BayerRG8": cv2.COLOR_BayerBG2BGR,
    "BayerGB8": cv2.COLOR_BayerGR2BGR,
    "BayerGR8": cv2.COLOR_BayerGB2BGR,
    "BayerBG8": cv2.COLOR_BayerRG2BGR,
}


def _ensure_bgr(frame):
    """Convert a PySpin frame to a 3-channel BGR image (no-op if already BGR)."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def bayer_to_bgr(raw, code, ccm=None):
    """Demosaic an 8-bit Bayer frame, then apply a 3x3 colour matrix (BGR order)."""
    bgr = cv2.cvtColor(raw, code)
    if ccm is not None:
        bgr = cv2.transform(bgr, ccm)
    return bgr


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        self.declare_parameter("publish_hz", 30.0)
        # Frame rate written into the mp4; 0 = the camera's own frame rate, so
        # the video plays at real speed.
        self.declare_parameter("record_fps", 0.0)
        self.declare_parameter("jpeg_quality", 80)
        # Combine binning x binning pixels in the camera (see the module docstring).
        self.declare_parameter("binning", 1)
        self.declare_parameter("binning_mode", "Sum")  # Sum | Average
        # Crop in the camera, in binned pixels (see the module docstring).
        # 0 keeps the whole width/height; a negative offset centres the crop.
        self.declare_parameter("roi_width", 0)
        self.declare_parameter("roi_height", 0)
        self.declare_parameter("roi_offset_x", -1)
        self.declare_parameter("roi_offset_y", -1)
        # Take the raw Bayer mosaic with the camera's ISP off and convert it
        # here; USB bandwidth in MB/s (0 = leave the camera's value); frame-rate
        # cap (0 = as fast as the camera goes). See the module docstring.
        self.declare_parameter("raw_bayer", False)
        self.declare_parameter("link_limit_mb_s", 0.0)
        self.declare_parameter("frame_rate", 0.0)

        # --- Image quality ------------------------------------------------
        # Target for the auto-exposure loop, in percent of full scale. The
        # camera default (~50%) is what makes a white field look mid grey.
        # 0 or less leaves whatever the camera is already doing.
        # Default path: measure the delivered image and solve for the exposure
        # that puts its mean grey level here, then hold it for the session.
        # Deterministic, and independent of the vendor auto loop, whose
        # ExposureTime readback on this model is cached and untrustworthy.
        self.declare_parameter("target_mean_grey", 200.0)
        # Set > 0 to state the exposure outright and skip calibration.
        self.declare_parameter("exposure_time_us", 0.0)
        self.declare_parameter("gain_db", 0.0)
        self.declare_parameter("exposure_search_max_us", 15000.0)
        # Opt back in to the camera's own auto-exposure loop. Useful when the
        # illumination changes during a session; not recommended while
        # recording, because it couples brightness to what is in frame.
        self.declare_parameter("use_auto_exposure", False)
        self.declare_parameter("target_grey_percent", 80.0)
        # Raise the ceiling the auto loop may use; 0 leaves it alone.
        self.declare_parameter("auto_exposure_max_us", 0.0)
        # White balance is the colour analogue of exposure: left on
        # Continuous it keeps re-adapting to whatever is in frame, so the same
        # scene yields different colours over a trial. "Once" converges on the
        # empty field at startup and then holds. Pin balance_ratio_* to
        # reproduce an exact previous session.
        self.declare_parameter("white_balance", "Once")   # Once | Continuous | Off
        self.declare_parameter("balance_ratio_red", 0.0)  # 0 = leave to the above
        self.declare_parameter("balance_ratio_blue", 0.0)
        self.declare_parameter("gamma", 0.0)  # 0 = leave the camera's value
        # Freeze exposure/gain for the duration of a recorded trial so the
        # pipette cannot modulate background brightness.
        self.declare_parameter("lock_exposure_while_recording", True)

        self._exposure_locked = False
        self._held_exposure_us = 0.0
        # Set by _configure_pixel_format when frames arrive as a Bayer mosaic.
        self._bayer_code = None
        self._ccm = None
        # PySpin image acquisition and node-map access are not thread safe, and
        # two threads reach the camera: the grab loop below, and the ROS
        # executor thread whenever a record command triggers a recalibration.
        # Every camera access is therefore taken under this lock. It is held per
        # access rather than for a whole calibration, so a bisection cannot
        # stall the preview stream for seconds at a time.
        self._cam_lock = threading.Lock()
        self._record_lock = threading.RLock()

        self.pub_img = self.create_publisher(CompressedImage, TOPIC_CAM_IMAGE_COMPRESSED, 10)
        self.pub_fps = self.create_publisher(Float32, TOPIC_CAM_FPS, 10)
        self.sub_rec = self.create_subscription(String, TOPIC_CAM_REC_CMD, self.on_rec_cmd, 10)

        self.system = None
        self.cams = None
        self.cam = None

        self.running = True
        self.recording = False
        self.record_path = None
        self.video_writer = None

        self._init_camera()

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    # ── Camera setup ───────────────────────────────────────────────────────
    def _init_camera(self):
        self.get_logger().info("Initializing PySpin camera...")
        self.system = PySpin.System.GetInstance()
        self.cams = self.system.GetCameras()
        if self.cams.GetSize() == 0:
            raise RuntimeError("No PySpin cameras detected.")

        self.cam = self.cams[0]
        try:
            self.cam.Init()
        except PySpin.SpinnakerException as exc:
            # Drop every SDK reference before raising: a System released while a
            # camera still holds its interface aborts the whole process.
            self.cam = None
            self.cams.Clear()
            self.cams = None
            self.system.ReleaseInstance()
            self.system = None
            hint = ""
            if "-1016" in str(exc):
                hint = (
                    " The USB buffer memory is too small for this camera: run "
                    "sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb' "
                    "(and make it permanent, see the ump_suite README)."
                )
            raise RuntimeError(f"camera initialisation failed: {exc}.{hint}") from exc

        self._set_stream_newest_only(self.cam)
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        limit = float(self.get_parameter("link_limit_mb_s").value)
        if limit > 0:
            self._set_int("DeviceLinkThroughputLimit", int(limit * 1e6))
        # Pixel format first: binning and the frame-rate limits depend on it.
        self._configure_pixel_format()
        self._configure_binning()
        self._configure_frame_rate()

        self._configure_image_quality()

        self.cam.BeginAcquisition()
        self.get_logger().info("Camera acquisition started.")
        # Calibration needs live frames, so it has to follow BeginAcquisition.
        # Exposure first so white balance sees a sanely exposed image, then
        # exposure again because the white-balance gains move the mean.
        self._calibrate_exposure()
        self._calibrate_white_balance()
        self._calibrate_exposure()
        self._log_exposure("after startup")

    def _configure_pixel_format(self):
        """
        Choose BGR8 from the camera's ISP, or with `raw_bayer` the Bayer mosaic.

        The ISP state is set either way, so a raw session does not leave it off.
        """
        self._set_bool("IspEnable", True)
        # Prefer BGR8 so we don't have to debayer manually. Cameras that
        # don't support it stay on whatever default they advertise.
        self._set_enum("PixelFormat", "BGR8")
        if not bool(self.get_parameter("raw_bayer").value):
            return
        formats = self._enum_entries("PixelFormat")
        fmt = next((f for f in _BAYER_TO_BGR if f in formats), None)
        if fmt is None:
            self.get_logger().warn("raw_bayer: this camera offers no 8-bit Bayer format")
            return
        # Read the colour matrix while the ISP that owns it is still on.
        ccm = self._read_color_matrix()
        if not (self._set_enum("PixelFormat", fmt) and self._set_bool("IspEnable", False)):
            self.get_logger().warn("raw_bayer: could not switch the ISP off; staying on BGR8")
            self._set_bool("IspEnable", True)
            self._set_enum("PixelFormat", "BGR8")
            return
        self._bayer_code = _BAYER_TO_BGR[fmt]
        self._ccm = ccm
        self.get_logger().info(
            f"Raw {fmt} with the camera's ISP off; demosaic"
            f"{' and colour matrix' if ccm is not None else ''} on this PC"
        )

    def _read_color_matrix(self):
        """Return the ISP's RGB-to-RGB colour matrix in BGR order, or None if it is off."""
        if not self._get_bool("ColorTransformationEnable"):
            return None
        self._set_enum("ColorTransformationSelector", "RGBtoRGB")
        matrix = np.zeros((3, 3), np.float32)
        for i in range(3):
            for j in range(3):
                if not self._set_enum("ColorTransformationValueSelector", f"Gain{i}{j}"):
                    return None
                matrix[i, j] = self._get_float("ColorTransformationValue")
        if not np.isfinite(matrix).all():
            return None
        return matrix[::-1, ::-1].copy()  # RGB to BGR on both sides

    def _configure_frame_rate(self):
        """Cap the frame rate at `frame_rate`, or let the camera run free (0)."""
        fps = float(self.get_parameter("frame_rate").value)
        self._set_bool("AcquisitionFrameRateEnable", fps > 0)
        if fps > 0:
            self._set_float("AcquisitionFrameRate", fps)
        self.get_logger().info(
            f"Frame rate {self._get_float('AcquisitionResultingFrameRate'):.1f} fps"
            f"{f' (capped at {fps:g})' if fps > 0 else ''}"
        )

    def _to_bgr(self, frame):
        if self._bayer_code is not None:
            return bayer_to_bgr(frame, self._bayer_code, self._ccm)
        return _ensure_bgr(frame)

    def _configure_binning(self):
        """
        Apply `binning` and the `roi_*` crop.

        Both are applied every start, so values left in the camera by an earlier
        session are undone.
        """
        b = max(1, int(self.get_parameter("binning").value))
        mode = str(self.get_parameter("binning_mode").value)
        if self._bayer_code is not None:
            mode = "Average"  # what the camera does with the ISP off, whatever is asked
        self._set_enum("BinningSelector", "All")
        if b > 1:
            self._set_enum("BinningHorizontalMode", mode)
            self._set_enum("BinningVerticalMode", mode)
        applied = [self._set_int(name, b) for name in ("BinningHorizontal", "BinningVertical")]
        # The image area shrinks with binning; grow it back to the whole sensor.
        for name in ("OffsetX", "OffsetY"):
            self._set_int(name, 0)
        full_w = self._set_int_to_max("Width")
        full_h = self._set_int_to_max("Height")
        if b > 1 and not all(applied):
            self.get_logger().warn(f"this camera did not accept {b}x{b} binning")
        area = ""
        if full_w and full_h:
            area = self._configure_roi(full_w, full_h)
        self.get_logger().info(
            f"Image {self._get_int('Width')}x{self._get_int('Height')}{area} "
            f"(binning {b}x{b}{', ' + mode if b > 1 else ''})"
        )

    def _configure_roi(self, full_w, full_h):
        """Crop to the `roi_*` parameters; return a note for the log ('' when uncropped)."""
        want_w = int(self.get_parameter("roi_width").value) or full_w
        want_h = int(self.get_parameter("roi_height").value) or full_h
        if want_w >= full_w and want_h >= full_h:
            return ""
        fits = [self._set_int("Width", want_w), self._set_int("Height", want_h)]
        w, h = self._get_int("Width"), self._get_int("Height")
        offsets = []
        for name, size, full in (("roi_offset_x", w, full_w), ("roi_offset_y", h, full_h)):
            offset = int(self.get_parameter(name).value)
            offsets.append((full - size) // 2 if offset < 0 else offset)
        fits += [self._set_int("OffsetX", offsets[0]), self._set_int("OffsetY", offsets[1])]
        x, y = self._get_int("OffsetX"), self._get_int("OffsetY")
        if not all(fits) or (w, h, x, y) != (want_w, want_h, *offsets):
            self.get_logger().warn(
                f"crop {want_w}x{want_h} at ({offsets[0]}, {offsets[1]}) adjusted by the "
                f"camera to {w}x{h} at ({x}, {y})"
            )
        return f" at ({x}, {y}) of {full_w}x{full_h}"

    def _set_int(self, name, value) -> bool:
        """
        Set one integer node, returning False instead of raising.

        The value is clamped to the node's range and rounded down to its step
        (crop sizes and offsets come in steps of a few pixels).
        """
        try:
            with self._cam_lock:
                node = PySpin.CIntegerPtr(self._node_map().GetNode(name))
                if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
                    return False
                lo, hi, inc = node.GetMin(), node.GetMax(), max(1, node.GetInc())
                fits = lo <= int(value) <= hi
                value = min(max(int(value), lo), hi)
                node.SetValue(lo + (value - lo) // inc * inc)
            return fits
        except PySpin.SpinnakerException as exc:
            self.get_logger().warn(f"could not set {name}={value}: {exc}")
            return False

    def _enum_entries(self, name):
        """Names of the entries an enumeration node currently offers."""
        try:
            with self._cam_lock:
                node = PySpin.CEnumerationPtr(self._node_map().GetNode(name))
                if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
                    return []
                return [PySpin.CEnumEntryPtr(e).GetSymbolic()
                        for e in node.GetEntries() if PySpin.IsAvailable(e)]
        except PySpin.SpinnakerException:
            return []

    def _set_bool(self, name, value) -> bool:
        """Set one boolean node, returning False instead of raising."""
        try:
            with self._cam_lock:
                node = PySpin.CBooleanPtr(self._node_map().GetNode(name))
                if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
                    return False
                node.SetValue(bool(value))
            return True
        except PySpin.SpinnakerException as exc:
            self.get_logger().warn(f"could not set {name}={value}: {exc}")
            return False

    def _get_bool(self, name):
        try:
            with self._cam_lock:
                node = PySpin.CBooleanPtr(self._node_map().GetNode(name))
                if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
                    return bool(node.GetValue())
        except PySpin.SpinnakerException:
            pass
        return None

    def _get_int(self, name):
        try:
            with self._cam_lock:
                node = PySpin.CIntegerPtr(self._node_map().GetNode(name))
                if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
                    return node.GetValue()
        except PySpin.SpinnakerException:
            pass
        return None

    def _set_int_to_max(self, name):
        """Set an integer node to its maximum and return the value, or None."""
        try:
            with self._cam_lock:
                node = PySpin.CIntegerPtr(self._node_map().GetNode(name))
                if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                    node.SetValue(node.GetMax())
                return node.GetValue() if PySpin.IsReadable(node) else None
        except PySpin.SpinnakerException as exc:
            self.get_logger().warn(f"could not set {name} to its maximum: {exc}")
            return None

    # ── Exposure / brightness ──────────────────────────────────────────────
    def _node_map(self):
        return self.cam.GetNodeMap()

    def _set_enum(self, name, entry) -> bool:
        """Set one enumeration node, returning False instead of raising."""
        try:
            with self._cam_lock:
                node = PySpin.CEnumerationPtr(self._node_map().GetNode(name))
                if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
                    return False
                value = node.GetEntryByName(entry)
                if not PySpin.IsAvailable(value) or not PySpin.IsReadable(value):
                    return False
                node.SetIntValue(value.GetValue())
            return True
        except PySpin.SpinnakerException as exc:
            self.get_logger().warn(f"could not set {name}={entry}: {exc}")
            return False

    def _set_float(self, name, value) -> bool:
        """Set one float node, clamped to the range the camera reports."""
        try:
            with self._cam_lock:
                node = PySpin.CFloatPtr(self._node_map().GetNode(name))
                if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
                    return False
                node.SetValue(
                    float(min(max(float(value), node.GetMin()), node.GetMax()))
                )
            return True
        except PySpin.SpinnakerException as exc:
            self.get_logger().warn(f"could not set {name}={value}: {exc}")
            return False

    def _get_float(self, name):
        try:
            with self._cam_lock:
                node = PySpin.CFloatPtr(self._node_map().GetNode(name))
                if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
                    return node.GetValue()
        except PySpin.SpinnakerException:
            pass
        return float("nan")

    def _get_enum(self, name):
        try:
            with self._cam_lock:
                node = PySpin.CEnumerationPtr(self._node_map().GetNode(name))
                if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
                    return node.GetCurrentEntry().GetSymbolic()
        except PySpin.SpinnakerException:
            pass
        return "?"

    def _configure_image_quality(self):
        """Apply the brightness parameters. Every step degrades gracefully."""
        manual_us = float(self.get_parameter("exposure_time_us").value)
        gain_db = float(self.get_parameter("gain_db").value)
        target = float(self.get_parameter("target_grey_percent").value)
        ceiling = float(self.get_parameter("auto_exposure_max_us").value)

        if ceiling > 0:
            self._set_float("AutoExposureExposureTimeUpperLimit", ceiling)

        if manual_us > 0:
            # Fixed exposure: the same scene always produces the same pixels.
            self._apply_fixed_exposure(manual_us)
            self.get_logger().info(
                f"Exposure fixed at {manual_us:.0f} us, gain {gain_db:.2f} dB"
            )
        elif bool(self.get_parameter("use_auto_exposure").value):
            # The order matters: the target is only honoured once both loops
            # have been handed authority and the target's own auto is off.
            self._set_enum("ExposureAuto", "Continuous")
            self._set_enum("GainAuto", "Continuous")
            if target > 0:
                # The target is only writable once its own 'auto' is off.
                self._set_enum("AutoExposureTargetGreyValueAuto", "Off")
                if self._set_float("AutoExposureTargetGreyValue", target):
                    self.get_logger().info(
                        f"Auto exposure targeting {target:.0f}% grey "
                        "(camera default is ~50%, which looks dim on a bright field)"
                    )

        gamma = float(self.get_parameter("gamma").value)
        if gamma > 0:
            self._set_enum("GammaEnable", "On")
            self._set_float("Gamma", gamma)

        # White balance is handled after acquisition starts; see
        # _calibrate_white_balance, which needs live frames to converge.

    def _apply_fixed_exposure(self, microseconds: float) -> None:
        """Pin exposure and gain, recording what we asked for."""
        self._set_enum("ExposureAuto", "Off")
        self._set_enum("ExposureMode", "Timed")
        self._set_float("ExposureTime", microseconds)
        self._set_enum("GainAuto", "Off")
        self._set_float("Gain", float(self.get_parameter("gain_db").value))
        self._held_exposure_us = float(microseconds)

    def _frame_mean(self, discard: int = 4) -> float:
        """Mean grey of a freshly grabbed frame, after flushing the pipeline."""
        value = float("nan")
        for _ in range(max(1, discard)):
            with self._cam_lock:
                try:
                    image = self.cam.GetNextImage(2000)
                except PySpin.SpinnakerException:
                    return value
                if not image.IsIncomplete():
                    # Measure what is delivered, after any conversion on this PC.
                    value = float(self._to_bgr(image.GetNDArray()).mean())
                image.Release()
        return value

    def _solve_exposure_for(self, target_mean: float) -> float:
        """
        Bisect exposure time until the delivered frame mean matches.

        Brightness is monotonic in exposure, so this converges quickly. It is
        driven entirely by measurement because the camera's ExposureTime
        readback is cached and can disagree with what is really in effect.
        """
        low = 12.0
        high = float(self.get_parameter("exposure_search_max_us").value)
        for _ in range(10):
            middle = 0.5 * (low + high)
            self._apply_fixed_exposure(middle)
            mean = self._frame_mean()
            if mean != mean:  # NaN: grabbing failed, stop rather than thrash
                return float("nan")
            if mean > target_mean:
                high = middle
            else:
                low = middle
            if abs(mean - target_mean) <= 1.0:
                break
        return self._frame_mean()

    def _calibrate_exposure(self):
        """Hold the exposure that delivers ``target_mean_grey``."""
        target = float(self.get_parameter("target_mean_grey").value)
        if target <= 0:
            return
        if float(self.get_parameter("exposure_time_us").value) > 0:
            return
        if bool(self.get_parameter("use_auto_exposure").value):
            return

        ceiling = float(self.get_parameter("exposure_search_max_us").value)
        self._apply_fixed_exposure(ceiling)
        brightest = self._frame_mean()
        if brightest == brightest and brightest < target:
            # Even the longest allowed exposure falls short: the light really is
            # insufficient, which is a rig problem rather than a settings one.
            self.get_logger().warn(
                f"cannot reach mean grey {target:.0f} even at {ceiling:.0f} us "
                f"(best {brightest:.1f}); holding the longest allowed exposure. "
                "Check the light path, the beam splitter and any ND filter."
            )
            return

        achieved = self._solve_exposure_for(target)
        if achieved != achieved:
            self.get_logger().warn("exposure calibration could not read frames")
            return
        self.get_logger().info(
            f"Exposure calibrated to {self._held_exposure_us:.0f} us for mean grey "
            f"{achieved:.1f} (target {target:.0f}); held fixed for this session"
        )

    def _balance_ratio(self, channel, value=None):
        """Read or write one BalanceRatio channel ("Red" or "Blue")."""
        if not self._set_enum("BalanceRatioSelector", channel):
            return float("nan")
        if value is not None:
            self._set_float("BalanceRatio", value)
        return self._get_float("BalanceRatio")

    def _calibrate_white_balance(self):
        """
        Fix the colour balance for the session.

        Continuous white balance re-adapts to frame content, so a dark pipette
        or a stained sample shifts the colour of the whole image over a trial -
        the same kind of content-dependent drift that auto exposure causes.
        """
        red = float(self.get_parameter("balance_ratio_red").value)
        blue = float(self.get_parameter("balance_ratio_blue").value)
        mode = str(self.get_parameter("white_balance").value).strip().capitalize()

        if red > 0 and blue > 0:
            self._set_enum("BalanceWhiteAuto", "Off")
            self._balance_ratio("Red", red)
            self._balance_ratio("Blue", blue)
            self.get_logger().info(
                f"White balance pinned at red={red:.3f} blue={blue:.3f}"
            )
            return

        if mode == "Continuous":
            self._set_enum("BalanceWhiteAuto", "Continuous")
            self.get_logger().warn(
                "white_balance=Continuous keeps adapting to frame content; "
                "colours will drift within a trial. Prefer Once for datasets."
            )
            return

        if mode == "Once":
            # Let the camera's own algorithm converge on the current field,
            # then it latches itself to Off and holds those gains.
            self._set_enum("BalanceWhiteAuto", "Continuous")
            self._frame_mean(discard=25)
            self._set_enum("BalanceWhiteAuto", "Off")
        else:
            self._set_enum("BalanceWhiteAuto", "Off")

        got_red = self._balance_ratio("Red")
        got_blue = self._balance_ratio("Blue")
        self.get_logger().info(
            f"White balance held at red={got_red:.3f} blue={got_blue:.3f} "
            "(set balance_ratio_red/blue to reproduce this exactly)"
        )

    def _log_exposure(self, when):
        self.get_logger().info(
            f"Exposure {when}: ExposureAuto={self._get_enum('ExposureAuto')} "
            f"time={self._get_float('ExposureTime'):.0f}us "
            f"GainAuto={self._get_enum('GainAuto')} gain={self._get_float('Gain'):.2f}dB"
        )

    def _exposure_is_already_deterministic(self) -> bool:
        """
        Report whether exposure is fixed and cannot drift during a trial.

        Three configurations pin it: an explicit ``exposure_time_us``, the
        startup ``target_mean_grey`` bisection, and simply having the vendor auto
        loop switched off. In all three the brightness is already independent of
        frame content, so there is nothing to freeze.
        """
        # Match _configure_image_quality: a positive manual exposure takes
        # precedence over the auto flag. Otherwise only continuous auto needs
        # freezing; auto-off is already fixed, even without a target grey level.
        if float(self.get_parameter("exposure_time_us").value) > 0:
            return True
        return not bool(self.get_parameter("use_auto_exposure").value)

    def _lock_exposure(self):
        """
        Freeze the converged auto values so a trial is photometrically stable.

        This is only meaningful when the vendor auto-exposure loop is actually
        running, which is what the parameter documentation has always said. When
        exposure is already deterministic, re-solving would be actively harmful:
        the bisection re-targets the CURRENT frame mean, which at the start of a
        trial may already contain the pipette, partially undoing the
        content-independence the startup calibration exists to guarantee.
        """
        if self._exposure_locked or self.cam is None:
            return
        if not bool(self.get_parameter("lock_exposure_while_recording").value):
            return
        if self._exposure_is_already_deterministic():
            self._exposure_locked = True   # already fixed; nothing to freeze
            return
        # Reproduce the brightness the auto loop had reached, by measurement.
        # The ExposureTime readback cannot be used: it reports a cached value,
        # and writing it back visibly darkens the image.
        before = self._frame_mean()
        achieved = self._solve_exposure_for(before)
        self._exposure_locked = True
        self.get_logger().info(
            f"Exposure locked for this trial at {self._held_exposure_us:.0f} us "
            f"(mean grey {achieved:.1f}, was {before:.1f}) so the pipette cannot "
            "modulate background brightness"
        )

    def _unlock_exposure(self):
        """Hand exposure back to the auto loop between trials."""
        if not self._exposure_locked or self.cam is None:
            return
        self._exposure_locked = False
        # Nothing was taken away in a deterministic configuration, so nothing is
        # handed back. Saying otherwise used to log a plainly false statement on
        # every trial stop.
        if self._exposure_is_already_deterministic():
            return
        self._configure_image_quality()
        self.get_logger().info("Exposure returned to auto between trials")

    @staticmethod
    def _set_stream_newest_only(cam):
        # Drop stale frames so the GUI / policy always see the freshest image.
        s_nm = cam.GetTLStreamNodeMap()
        handling = PySpin.CEnumerationPtr(s_nm.GetNode("StreamBufferHandlingMode"))
        if PySpin.IsAvailable(handling) and PySpin.IsWritable(handling):
            newest = handling.GetEntryByName("NewestOnly")
            if PySpin.IsAvailable(newest) and PySpin.IsReadable(newest):
                handling.SetIntValue(newest.GetValue())

    # ── Recording control ──────────────────────────────────────────────────
    def _close_writer(self):
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

    def on_rec_cmd(self, msg: String):
        with self._record_lock:
            self._handle_rec_cmd(msg)

    def _handle_rec_cmd(self, msg: String):
        path = (msg.data or "").strip()

        if path == "":
            self.recording = False
            self.record_path = None
            self._close_writer()
            self._unlock_exposure()
            self.get_logger().info("Recording stopped.")
            return

        # Switching to a new file: drop any previous writer first.
        self._close_writer()
        self._lock_exposure()
        self.recording = True
        self.record_path = path
        self.get_logger().info(f"Recording started: {self.record_path}")

    # ── Frame loop ─────────────────────────────────────────────────────────
    def _publish_jpeg(self, frame_bgr, fps):
        q = int(self.get_parameter("jpeg_quality").value)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(5, min(95, q))]
        ok, enc = cv2.imencode(".jpg", frame_bgr, encode_params)
        if ok:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera"
            msg.format = "jpeg"
            msg.data = enc.tobytes()
            self.pub_img.publish(msg)

        self.pub_fps.publish(Float32(data=float(fps)))

    def _record_frame(self, frame_bgr):
        with self._record_lock:
            if self.recording and self.record_path:
                self._write_record_frame(frame_bgr)

    def _write_record_frame(self, frame_bgr):
        if self.video_writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            rec_fps = float(self.get_parameter("record_fps").value)
            if rec_fps <= 0:
                rec_fps = self._get_float("AcquisitionResultingFrameRate")
                if not rec_fps > 0:  # NaN when the camera does not report it
                    rec_fps = 10.0
            self.video_writer = cv2.VideoWriter(self.record_path, fourcc, rec_fps, (w, h))

        if self.video_writer.isOpened():
            self.video_writer.write(frame_bgr)

    def _loop(self):
        publish_period = 1.0 / max(1e-3, float(self.get_parameter("publish_hz").value))
        last_pub = 0.0
        last = time.time()

        while self.running and rclpy.ok():
            try:
                with self._cam_lock:
                    img = self.cam.GetNextImage(CAM_GET_TIMEOUT_MS)
                    try:
                        if img.IsIncomplete():
                            continue
                        # SDK memory must be copied before releasing this image.
                        frame = img.GetNDArray().copy()
                    finally:
                        img.Release()

                now = time.time()
                fps = 1.0 / max(1e-6, (now - last))
                last = now

                # Throttle the published preview to publish_hz; recording
                # always sees every captured frame so the video stays smooth.
                publish_due = (now - last_pub) >= publish_period
                if publish_due or (self.recording and self.record_path):
                    frame_bgr = self._to_bgr(frame)

                    if publish_due:
                        last_pub = now
                        self._publish_jpeg(frame_bgr, fps)

                    if self.recording and self.record_path:
                        self._record_frame(frame_bgr)

            except PySpin.SpinnakerException:
                time.sleep(0.01)
            except Exception as e:
                if not self.running or not rclpy.ok():
                    break
                self.get_logger().warn(f"Camera loop error: {e}")
                time.sleep(0.01)

    # ── Shutdown ───────────────────────────────────────────────────────────
    def destroy_node(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=max(2.0, CAM_GET_TIMEOUT_MS / 1000.0 + 1.0))
            if self.thread.is_alive():
                raise RuntimeError(
                    "Camera worker did not stop; refusing to release SDK buffers in use"
                )
        self._close_writer()

        try:
            if self.cam is not None:
                try:
                    self.cam.EndAcquisition()
                except Exception:
                    pass
                self.cam.DeInit()
        except Exception as exc:
            self.get_logger().error(f"Camera deinitialization failed: {exc}")
        finally:
            # A CameraPtr retains its interface even after DeInit. Drop it
            # before clearing the list or releasing the System singleton.
            self.cam = None

        try:
            if self.cams is not None:
                self.cams.Clear()
        except Exception as exc:
            self.get_logger().error(f"Camera list cleanup failed: {exc}")
        finally:
            self.cams = None

        try:
            if self.system is not None:
                self.system.ReleaseInstance()
        except Exception as exc:
            self.get_logger().error(f"Spinnaker system release failed: {exc}")
        finally:
            self.system = None

        super().destroy_node()


def main():
    from .runtime_guard import acquire_process_lock
    acquire_process_lock("camera")
    # OpenCV starts one worker per core (32 here) for every small call and they
    # spin: about a core per process at 25 fps. Two do the same work.
    cv2.setNumThreads(2)

    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

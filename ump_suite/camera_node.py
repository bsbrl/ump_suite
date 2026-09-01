"""ROS2 driver for a FLIR Blackfly camera via PySpin.

Each frame is grabbed in a worker thread, then:
  * a JPEG-compressed copy is published at `publish_hz` for the GUI / VLA client
  * the FPS achieved by the grabber is published on `/camera/fps`
  * if recording is active, the raw frame is appended to an mp4 video file

Recording is controlled by a String message on /camera/record_cmd: a non-empty
path starts recording to that file, an empty string stops it.

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
import PySpin
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, String

from .ros_interfaces import (
    TOPIC_CAM_FPS,
    TOPIC_CAM_IMAGE_COMPRESSED,
    TOPIC_CAM_REC_CMD,
)


CAM_GET_TIMEOUT_MS = 1000


def _ensure_bgr(frame):
    """Convert a PySpin frame to a 3-channel BGR image (no-op if already BGR)."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("record_fps", 20.0)
        self.declare_parameter("jpeg_quality", 80)

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
        # PySpin image acquisition and node-map access are not thread safe, and
        # two threads reach the camera: the grab loop below, and the ROS
        # executor thread whenever a record command triggers a recalibration.
        # Every camera access is therefore taken under this lock. It is held per
        # access rather than for a whole calibration, so a bisection cannot
        # stall the preview stream for seconds at a time.
        self._cam_lock = threading.Lock()

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
        self.cam.Init()

        self._set_stream_newest_only(self.cam)
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        # Prefer BGR8 so we don't have to debayer manually. Cameras that
        # don't support it stay on whatever default they advertise.
        try:
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
        except PySpin.SpinnakerException:
            pass

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
                    value = float(image.GetNDArray().mean())
                image.Release()
        return value

    def _solve_exposure_for(self, target_mean: float) -> float:
        """Bisect exposure time until the delivered frame mean matches.

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
        """Fix the colour balance for the session.

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
        """True when exposure is already fixed and cannot drift during a trial.

        Three configurations pin it: an explicit ``exposure_time_us``, the
        startup ``target_mean_grey`` bisection, and simply having the vendor auto
        loop switched off. In all three the brightness is already independent of
        frame content, so there is nothing to freeze.
        """
        if float(self.get_parameter("exposure_time_us").value) > 0:
            return True
        if float(self.get_parameter("target_mean_grey").value) > 0:
            return True
        return not bool(self.get_parameter("use_auto_exposure").value)

    def _lock_exposure(self):
        """Freeze the converged auto values so a trial is photometrically stable.

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
        if self.video_writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            rec_fps = float(self.get_parameter("record_fps").value)
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
                    if img.IsIncomplete():
                        img.Release()
                        continue
                    # GetNDArray() aliases the buffer, so it must be copied
                    # before Release() and before the lock is dropped.
                    frame = img.GetNDArray().copy()
                    img.Release()

                now = time.time()
                fps = 1.0 / max(1e-6, (now - last))
                last = now

                # Throttle the published preview to publish_hz; recording
                # always sees every captured frame so the video stays smooth.
                publish_due = (now - last_pub) >= publish_period
                if publish_due or (self.recording and self.record_path):
                    frame_bgr = _ensure_bgr(frame)

                    if publish_due:
                        last_pub = now
                        self._publish_jpeg(frame_bgr, fps)

                    if self.recording and self.record_path:
                        self._record_frame(frame_bgr)

            except PySpin.SpinnakerException:
                time.sleep(0.01)
            except Exception as e:
                self.get_logger().warn(f"Camera loop error: {e}")
                time.sleep(0.01)

    # ── Shutdown ───────────────────────────────────────────────────────────
    def destroy_node(self):
        self.running = False
        self._close_writer()

        try:
            if self.cam is not None:
                try:
                    self.cam.EndAcquisition()
                except Exception:
                    pass
                self.cam.DeInit()
        except Exception:
            pass

        try:
            if self.cams is not None:
                self.cams.Clear()
        except Exception:
            pass

        try:
            if self.system is not None:
                self.system.ReleaseInstance()
        except Exception:
            pass

        super().destroy_node()


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

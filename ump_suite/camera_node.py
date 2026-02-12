import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from sensor_msgs.msg import CompressedImage

import PySpin
import cv2

from .ros_interfaces import TOPIC_CAM_FPS, TOPIC_CAM_REC_CMD, TOPIC_CAM_IMAGE_COMPRESSED


CAM_GET_TIMEOUT_MS = 1000

class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("record_fps", 20.0)
        self.declare_parameter("jpeg_quality", 80)

        self.pub_img = self.create_publisher(CompressedImage, TOPIC_CAM_IMAGE_COMPRESSED, 10)
        self.pub_fps = self.create_publisher(Float32, TOPIC_CAM_FPS, 10)
        self.sub_rec = self.create_subscription(String, TOPIC_CAM_REC_CMD, self.on_rec_cmd, 10)

        self.system = None
        self.cams = None
        self.cam = None

        self.running = True

        # Recording
        self.recording = False
        self.record_path = None
        self.video_writer = None

        self._init_camera()

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _set_stream_newest_only(self, cam):
        s_nm = cam.GetTLStreamNodeMap()
        handling = PySpin.CEnumerationPtr(s_nm.GetNode("StreamBufferHandlingMode"))
        if PySpin.IsAvailable(handling) and PySpin.IsWritable(handling):
            newest = handling.GetEntryByName("NewestOnly")
            if PySpin.IsAvailable(newest) and PySpin.IsReadable(newest):
                handling.SetIntValue(newest.GetValue())

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

        try:
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
        except PySpin.SpinnakerException:
            pass

        self.cam.BeginAcquisition()
        self.get_logger().info("Camera acquisition started.")

    def on_rec_cmd(self, msg: String):
        path = (msg.data or "").strip()

        if path == "":
            self.recording = False
            self.record_path = None
            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    pass
                self.video_writer = None
            self.get_logger().info("Recording stopped.")
            return

        self.recording = True
        self.record_path = path
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None

        self.get_logger().info(f"Recording started: {self.record_path}")

    def _loop(self):
        publish_period = 1.0 / max(1e-3, float(self.get_parameter("publish_hz").value))
        last_pub = 0.0
        last = time.time()

        while self.running and rclpy.ok():
            try:
                img = self.cam.GetNextImage(CAM_GET_TIMEOUT_MS)
                if img.IsIncomplete():
                    img.Release()
                    continue

                frame = img.GetNDArray()
                img.Release()

                now = time.time()
                fps = 1.0 / max(1e-6, (now - last))
                last = now

                # Publish JPEG compressed
                if (now - last_pub) >= publish_period:
                    last_pub = now

                    if frame.ndim == 2:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        frame_bgr = frame

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

                # Recording (mp4)
                if self.recording and self.record_path:
                    rec_fps = float(self.get_parameter("record_fps").value)

                    if frame.ndim == 2:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        frame_bgr = frame

                    if self.video_writer is None:
                        h, w = frame_bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        self.video_writer = cv2.VideoWriter(self.record_path, fourcc, rec_fps, (w, h))

                    if self.video_writer is not None and self.video_writer.isOpened():
                        self.video_writer.write(frame_bgr)

            except PySpin.SpinnakerException:
                time.sleep(0.01)
            except Exception as e:
                self.get_logger().warn(f"Camera loop error: {e}")
                time.sleep(0.01)

    def destroy_node(self):
        self.running = False
        try:
            if self.video_writer is not None:
                self.video_writer.release()
        except Exception:
            pass

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
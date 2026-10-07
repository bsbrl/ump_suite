"""
ROS2 driver for one Sensapex UMP micromanipulator stage.

The Sensapex SDK reports positions in MICROMETRES as floats, and speeds in um/s.
This node exposes those absolute coordinates on the ROS topics, truncated to
whole micrometres by the Int32MultiArray message type, and forwards absolute
targets directly to the device. Topic names are built from a `topic_prefix`
parameter, allowing one process per device.

Units matter for safety: a step cap of 50 is 50 um per tick, which is several
cell diameters. Do not read these as unspecified encoder counts.

The UMP 1 node also runs the injection macro (`injection.py`): `/inject/start`
moves X in, pulses pressure, vents and moves back, using the values the GUI last
published on `/inject/params`. While it runs, ordinary targets are ignored, and
`/ump/stop` or a fault aborts it, venting if pressure was applied.
"""

import json
import math
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32, Int32MultiArray, String
from std_srvs.srv import Trigger

from sensapex import UMP

from .injection import (
    InjectionSequence,
    decode_params_message,
    forward_target,
)
from .ros_interfaces import (
    SRV_INJECT_START,
    TOPIC_INJECT_PARAMS,
    TOPIC_INJECT_STATUS,
    TOPIC_PRESSURE_MBAR,
    TOPIC_PRESSURE_STATUS,
    TOPIC_PRESSURE_TARGET,
    latched_qos,
)

# A nonzero injection pressure needs a ready report from the pressure driver
# no older than this; it publishes one every second.
PRESSURE_STATUS_MAX_AGE_S = 3.0


class UMPDriverNode(Node):
    def __init__(self, node_name="ump_driver_node", *, parameter_overrides=None):
        super().__init__(node_name, parameter_overrides=parameter_overrides or [])

        self.declare_parameter("device_id", 1)
        self.declare_parameter("poll_ms", 50)
        self.declare_parameter("topic_prefix", "ump")
        self.declare_parameter("injection_enabled", True)

        device_id = int(self.get_parameter("device_id").value)
        poll_ms = int(self.get_parameter("poll_ms").value)
        prefix = self.get_parameter("topic_prefix").value

        self.get_logger().info(
            f"Connecting to Sensapex UMP device {device_id} (prefix: /{prefix})..."
        )
        self.ump = UMP.get_ump()
        self.stage = self.ump.get_device(device_id)
        self.get_logger().info(f"Connected to UMP device {device_id}")

        self.pub_live = self.create_publisher(Int32MultiArray, f"/{prefix}/live", 10)
        self.sub_target = self.create_subscription(
            Int32MultiArray, f"/{prefix}/target", self.on_target, 10
        )
        self.srv_zero = self.create_service(
            Trigger, f"/{prefix}/calibrate_zero", self.on_zero
        )
        self.srv_stop = self.create_service(Trigger, f"/{prefix}/stop", self.on_stop)
        self._faulted = False

        self._inject_lock = threading.Lock()
        self._injecting = False
        self._inject_abort = threading.Event()
        self._inject_thread = None
        self._inject_count = 0
        self._inject_params = None
        self._inject_params_error = "no parameters received on " + TOPIC_INJECT_PARAMS
        self._inject_token = self._inject_seq = None
        self._pressure_ack = threading.Condition()
        self._pressure_ack_value = None
        self._pressure_ack_stamp = float("-inf")
        self._pressure_ready = False
        self._pressure_status_stamp = float("-inf")
        if bool(self.get_parameter("injection_enabled").value):
            self._setup_injection()

        self.timer = self.create_timer(poll_ms / 1000.0, self.poll_live)

    # ── Injection macro ────────────────────────────────────────────────────
    def _setup_injection(self):
        self.pub_pressure = self.create_publisher(Float32, TOPIC_PRESSURE_MBAR, 10)
        self.pub_inject_status = self.create_publisher(
            String, TOPIC_INJECT_STATUS, latched_qos(depth=10)
        )
        # Latched: parameters the GUI published before this driver started count.
        self.create_subscription(
            String, TOPIC_INJECT_PARAMS, self._on_inject_params, latched_qos()
        )
        self.create_subscription(
            Float32, TOPIC_PRESSURE_TARGET, self._on_pressure_target, latched_qos()
        )
        self.create_subscription(
            String, TOPIC_PRESSURE_STATUS, self._on_pressure_status, latched_qos()
        )
        self.srv_inject = self.create_service(Trigger, SRV_INJECT_START, self._on_inject)
        self._publish_inject_status("idle", False, "ready")

    def _publish_inject_status(self, stage, active, message):
        if stage in ("done", "aborted"):
            # Publish the final position first: the logger stops holding its
            # pre-injection values on this status, and must not then log a
            # sample taken while the stage was still moving.
            self.poll_live()
        params = self._inject_params
        self.pub_inject_status.publish(String(data=json.dumps({
            "count": self._inject_count,
            "active": bool(active),
            "stage": stage,
            "message": message,
            "params": None if params is None else params.to_dict(),
            "params_token": self._inject_token,
            "params_seq": self._inject_seq,
            "stamp": time.time(),
        })))

    def _on_inject_params(self, msg: String):
        try:
            params, token, seq = decode_params_message(msg.data)
        except ValueError as exc:
            self._inject_params, self._inject_params_error = None, str(exc)
            self._inject_token = self._inject_seq = None
            self.get_logger().warn(f"Injection parameters rejected: {exc}")
            self._publish_inject_status("idle", self._injecting, f"parameters rejected: {exc}")
            return
        # A running injection keeps the values it started with; this only
        # affects the next trigger.
        self._inject_params, self._inject_params_error = params, ""
        self._inject_token, self._inject_seq = token, seq
        self._publish_inject_status(
            "running" if self._injecting else "idle", self._injecting,
            f"parameters: {params.describe()}")

    def _on_pressure_target(self, msg: Float32):
        value = float(msg.data)
        if math.isfinite(value):
            with self._pressure_ack:
                self._pressure_ack_value = value
                self._pressure_ack_stamp = time.monotonic()
                self._pressure_ack.notify_all()

    def _on_pressure_status(self, msg: String):
        try:
            status = json.loads(msg.data)
            ready = status["ready"]
        except (TypeError, ValueError, KeyError):
            return
        if isinstance(ready, bool):
            self._pressure_ready = ready
            self._pressure_status_stamp = time.monotonic()

    def _wait_pressure_ack(self, after, timeout):
        """Return the mbar acknowledged after monotonic time `after`, else None."""
        deadline = time.monotonic() + timeout
        with self._pressure_ack:
            while self._pressure_ack_stamp < after:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._pressure_ack.wait(remaining)
            return self._pressure_ack_value

    def _send_pressure(self, mbar):
        self.pub_pressure.publish(Float32(data=float(mbar)))

    def _refuse_injection(self, res, reason):
        res.success = False
        res.message = f"Injection refused: {reason}"
        self.get_logger().warn(res.message)
        return res

    def _on_inject(self, _req, res):
        with self._inject_lock:
            if self._injecting:
                return self._refuse_injection(res, "an injection is already running")
            if self._faulted:
                return self._refuse_injection(res, "UMP fault latched; restart the driver")
            params = self._inject_params
            if params is None:
                return self._refuse_injection(res, self._inject_params_error)
            if params.pressure_mbar != 0.0:
                age = time.monotonic() - self._pressure_status_stamp
                if not self._pressure_ready or age > PRESSURE_STATUS_MAX_AGE_S:
                    return self._refuse_injection(
                        res, "the pressure driver is not reporting ready")
            try:
                if self.stage.is_busy():
                    return self._refuse_injection(res, "the stage is still moving")
                start = [float(v) for v in self.stage.get_pos()[:4]]
                forward_target(start, params)
            except ValueError as exc:
                return self._refuse_injection(res, str(exc))
            except Exception as exc:
                return self._refuse_injection(res, f"could not read the stage: {exc}")
            self._inject_abort.clear()
            self._injecting = True
            self._inject_count += 1
            count = self._inject_count
            sequence = InjectionSequence(
                params,
                read_position=lambda: [float(v) for v in self.stage.get_pos()[:4]],
                move_to=lambda pos, speed: self.stage.goto_pos(pos, speed=speed),
                stop_stage=self.stage.stop,
                send_pressure=self._send_pressure,
                wait_pressure_ack=self._wait_pressure_ack,
                report=self._publish_inject_status,
                abort_event=self._inject_abort,
            )
            self._inject_thread = threading.Thread(
                target=self._run_injection, args=(sequence, start, count),
                name=f"injection-{count}", daemon=True)
            self._inject_thread.start()
        res.success = True
        res.message = f"Injection #{count} started: {params.describe()}"
        self.get_logger().info(res.message)
        return res

    def _run_injection(self, sequence, start, count):
        finished = False
        try:
            finished = sequence.run(start)
        except Exception as exc:
            # Only a hardware or ROS failure inside the sequence latches a fault;
            # an abort (stop, timeout, missing acknowledgment) does not.
            self._faulted = True
            self.get_logger().error(
                f"Injection #{count} failed; stopping and latching fault: {exc}")
            return
        finally:
            with self._inject_lock:
                self._injecting = False
        # rclpy refuses two severities from one call site, so keep them apart.
        if finished:
            self.get_logger().info(f"Injection #{count} finished")
        else:
            self.get_logger().warn(f"Injection #{count} aborted")

    def _abort_injection(self):
        """Ask a running injection to stop; it vents if pressure was applied."""
        self._inject_abort.set()

    def _read_absolute_pos(self):
        """
        Return the current [x, y, z, d] absolute position in MICROMETRES.

        The Sensapex SDK documents positions in um and returns floats. They are
        truncated to whole micrometres here because the ROS message is an
        Int32MultiArray; sub-micrometre feedback is lost. Anything that needs it
        must change the message type, not just this function.
        """
        pos = self.stage.get_pos()
        return [int(pos[i]) for i in range(4)]

    def poll_live(self):
        try:
            msg = Int32MultiArray()
            msg.data = self._read_absolute_pos()
            self.pub_live.publish(msg)
        except Exception as e:
            self.get_logger().error(f"UMP live read failed; stopping and latching fault: {e}")
            self._faulted = True
            self._stop_stage()

    def on_target(self, msg: Int32MultiArray):
        if self._faulted:
            self.get_logger().error("UMP fault latched; restart driver after resolving the fault")
            return
        if self._injecting:
            self.get_logger().warn("Target ignored: an injection is running")
            return
        if len(msg.data) < 5:
            self.get_logger().warn("UMP target msg requires [x,y,z,d,speed]")
            return
        try:
            x, y, z, d, speed = (int(v) for v in msg.data[:5])
            if speed <= 0:
                raise ValueError("speed must be positive")
            self.stage.goto_pos([x, y, z, d], speed=speed)
        except Exception as e:
            self.get_logger().error(f"UMP goto_pos error: {e}")
            self._faulted = True
            self._stop_stage()

    def _stop_stage(self):
        self._abort_injection()
        try:
            self.stage.stop()
            return True, "SDK acknowledged stop"
        except Exception as exc:
            self.get_logger().error(f"UMP stop failed: {exc}")
            return False, str(exc)

    def on_stop(self, _req, res):
        res.success, res.message = self._stop_stage()
        return res

    def destroy_node(self):
        self._stop_stage()
        thread = self._inject_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        super().destroy_node()

    def on_zero(self, _req, res):
        try:
            self.stage.calibrate_zero_position()
            res.success = True
            res.message = "Zero calibrated at current position."
        except Exception as e:
            res.success = False
            res.message = f"Calibrate zero error: {e}"
        return res


def main():
    from .runtime_guard import acquire_process_lock
    acquire_process_lock("sensapex")

    rclpy.init()
    node = UMPDriverNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main_dual():
    """Run two UMP driver nodes in a single process so they share one SDK instance."""
    from .runtime_guard import acquire_process_lock
    acquire_process_lock("sensapex")

    rclpy.init()

    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter

    node1 = UMPDriverNode("ump_driver_node", parameter_overrides=[
        Parameter("device_id", value=1),
        Parameter("poll_ms", value=50),
        Parameter("topic_prefix", value="ump"),
    ])
    node2 = UMPDriverNode("ump2_driver_node", parameter_overrides=[
        Parameter("device_id", value=2),
        Parameter("poll_ms", value=50),
        Parameter("topic_prefix", value="ump2"),
        # The injection macro and its topics belong to UMP 1 only.
        Parameter("injection_enabled", value=False),
    ])

    executor = MultiThreadedExecutor()
    executor.add_node(node1)
    executor.add_node(node2)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node1.destroy_node()
        node2.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

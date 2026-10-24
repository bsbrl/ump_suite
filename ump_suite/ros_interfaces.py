"""Topic and service names shared by every node in the package."""

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

# UMP 1 ── primary Sensapex micromanipulator
TOPIC_UMP_TARGET = "/ump/target"          # Int32MultiArray [x,y,z,d,speed] absolute
TOPIC_UMP_LIVE = "/ump/live"            # Int32MultiArray [x,y,z,d] absolute

# UMP 2 ── secondary Sensapex micromanipulator
TOPIC_UMP2_TARGET = "/ump2/target"        # Int32MultiArray [x,y,z,d,speed] absolute
TOPIC_UMP2_LIVE = "/ump2/live"          # Int32MultiArray [x,y,z,d] absolute

# ODrive motor (manual GUI control; not part of policy rollout/logged CSV state)
TOPIC_MOTOR_TGT = "/motor/target_counts"   # Int32 absolute encoder counts
TOPIC_MOTOR_LIVE = "/motor/live_counts"     # Int32 current counts

# Pressure control (Fluigent LineUP push-pull controller)
#
# Commands carry the exact pressure in mbar; negative values pull, positive
# values push, 0 vents.
#
#   /pressure/mbar         <- what a client asks for (GUI, policy)
#   /pressure/target_mbar  -> what the node actually wrote to the device, after
#                             clamping to the hardware range. This is the
#                             logged/learned target after SDK acknowledgment;
#                             it does not verify physical pressure settling.
#   /pressure/measured_mbar-> what the controller's sensor reads back
#
# Publishers use transient-local durability. Applied-target subscribers may
# receive history; the pressure driver requests volatile commands and therefore
# ignores cached requests when it starts.
TOPIC_PRESSURE_MBAR = "/pressure/mbar"             # Float32 requested mbar, latched
TOPIC_PRESSURE_TARGET = "/pressure/target_mbar"      # Float32 applied mbar, latched
TOPIC_PRESSURE_STATUS = "/pressure/status"  # String JSON health/fault status, latched
SRV_PRESSURE_RESET = "/pressure/reset_fault"  # Trigger: reconnect and request 0 mbar
TOPIC_PRESSURE_MEASURED = "/pressure/measured_mbar"    # Float32 measured pressure

# Camera (Blackfly via PySpin)
TOPIC_CAM_IMAGE_COMPRESSED = "/camera/image/compressed"   # CompressedImage (jpeg)
TOPIC_CAM_FPS = "/camera/fps"                # Float32
TOPIC_CAM_REC_CMD = "/camera/record_cmd"         # String: path to start, "" to stop

# Services (all std_srvs/Trigger)
SRV_ACQ_START = "/acq/start"
SRV_ACQ_STOP = "/acq/stop"
SRV_ZERO = "/ump/calibrate_zero"
SRV_ZERO2 = "/ump2/calibrate_zero"

SRV_UMP_STOP = "/ump/stop"
SRV_UMP2_STOP = "/ump2/stop"

# Injection macro, run by the UMP 1 driver (see injection.py):
#   /inject/params  <- GUI: latched JSON {token, seq, axis, speed_um_s, step_um,
#                      pressure_mbar, duration_ms}; the values any trigger uses.
#                      axis is X, Y, Z or D; a message without it means X
#   /inject/start   -> std_srvs/Trigger: start one injection with those values
#   /inject/status  -> latched JSON {count, active, stage, message, params,
#                      params_token, params_seq, stamp}; `count` increments
#                      once per injection that actually starts
# /ump/stop aborts a running injection (and vents if pressure was applied).
TOPIC_INJECT_PARAMS = "/inject/params"
TOPIC_INJECT_STATUS = "/inject/status"
SRV_INJECT_START = "/inject/start"
# The operator's verdict on the last injection, from the GUI:
#   JSON {count, outcome} with outcome one of INJECTION_OUTCOMES
TOPIC_INJECT_OUTCOME = "/inject/outcome"
INJECTION_OUTCOMES = ("cell", "yolk", "chorion_space", "no_puncture")

# Orient skill, run by the UMP 1 driver (see orient.py): turns the embryo the
# needle points at until its cell faces the needle.
#   /orient/params  <- GUI: latched JSON of OrientParams fields (+ token, seq)
#   /orient/start   -> std_srvs/Trigger
#   /orient/status  -> latched JSON {count, active, stage, message, result,
#                      strokes, angle_deg, gain_deg_per_um, history, params, stamp}
# /ump/stop aborts a running orient.
TOPIC_ORIENT_PARAMS = "/orient/params"
TOPIC_ORIENT_STATUS = "/orient/status"
SRV_ORIENT_START = "/orient/start"

# Vision node (vision_node.py): the scene from the segmentation model.
#   /vision/scene   -> JSON scene (vision/geometry.py) for each processed frame
#   /vision/status  -> latched JSON {ready, model, message, fps, inference_ms, stamp}
TOPIC_VISION_SCENE = "/vision/scene"
TOPIC_VISION_STATUS = "/vision/status"

TOPIC_HEKA_RESISTANCE = "/heka/resistance_mohm"      # std_msgs/Float32 computed/live
TOPIC_HEKA_MONITOR_V = "/heka/monitor_v"            # std_msgs/Float32
TOPIC_HEKA_MONITOR_STEP_V = "/heka/monitor_step_v"  # std_msgs/Float32
TOPIC_HEKA_VOLTAGE_RAW = "/heka/voltage_raw_v"      # Float32MultiArray [rate_hz, samples...]
TOPIC_HEKA_CURRENT_PA = "/heka/current_pa"          # Float32MultiArray [rate_hz, samples...]


def latched_qos(depth=1):
    """
    Return a latched QoS profile for command publishers and applied readbacks.

    Transient-local durability means a node that starts *after* the value was
    published still receives the last one when its subscription also requests
    transient-local durability. Applied-pressure loggers do so; command-consuming
    pressure drivers deliberately use volatile subscriptions to exclude history.
    """
    return QoSProfile(
        depth=depth,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

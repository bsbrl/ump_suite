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

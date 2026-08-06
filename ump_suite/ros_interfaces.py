"""Topic and service names shared by every node in the package."""

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

# UMP 1 ── primary Sensapex micromanipulator
TOPIC_UMP_TARGET = "/ump/target"          # Int32MultiArray [x,y,z,d,speed] absolute
TOPIC_UMP_LIVE   = "/ump/live"            # Int32MultiArray [x,y,z,d] absolute

# UMP 2 ── secondary Sensapex micromanipulator
TOPIC_UMP2_TARGET = "/ump2/target"        # Int32MultiArray [x,y,z,d,speed] absolute
TOPIC_UMP2_LIVE   = "/ump2/live"          # Int32MultiArray [x,y,z,d] absolute

# ODrive motor (manual GUI control; not part of policy rollout/logged CSV state)
TOPIC_MOTOR_TGT  = "/motor/target_counts"   # Int32 absolute encoder counts
TOPIC_MOTOR_LIVE = "/motor/live_counts"     # Int32 current counts

# Pressure control (Fluigent LineUP push-pull controller)
#
# The *commanded* state is binary: True = apply the positive setpoint,
# False = apply the negative setpoint. The two setpoints live on their own
# latched topics, so a policy only has to emit the binary state and the
# pressure node resolves it against whatever the GUI has dialed in.
#
# The *reported* state adds a third value for "vented" (0 mbar), which is how
# the node starts up and where the vent service puts it. Venting is an operator
# action exposed as a service, deliberately not on the command topic, so a
# policy's action space stays strictly binary.
TOPIC_PRESSURE_STATE_CMD = "/pressure/state_cmd"       # Bool: True=positive, False=negative
TOPIC_PRESSURE_STATE     = "/pressure/state"           # Int8 PRESSURE_STATE_*, latched
TOPIC_PRESSURE_POS_MBAR  = "/pressure/positive_mbar"   # Float32 setpoint (>= 0), latched
TOPIC_PRESSURE_NEG_MBAR  = "/pressure/negative_mbar"   # Float32 setpoint (<= 0), latched
TOPIC_PRESSURE_MEASURED  = "/pressure/measured_mbar"   # Float32 measured pressure

# Values published on TOPIC_PRESSURE_STATE. Positive/negative are logged as the
# binary 1/0 dataset column; vented is logged as blank because it is neither.
PRESSURE_STATE_NEGATIVE = 0
PRESSURE_STATE_POSITIVE = 1
PRESSURE_STATE_VENTED = -1

# Camera (Blackfly via PySpin)
TOPIC_CAM_IMAGE_COMPRESSED = "/camera/image/compressed"   # CompressedImage (jpeg)
TOPIC_CAM_FPS              = "/camera/fps"                # Float32
TOPIC_CAM_REC_CMD          = "/camera/record_cmd"         # String: path to start, "" to stop

# Services (all std_srvs/Trigger)
SRV_ACQ_START = "/acq/start"
SRV_ACQ_STOP  = "/acq/stop"
SRV_ZERO      = "/ump/calibrate_zero"
SRV_ZERO2     = "/ump2/calibrate_zero"
SRV_PRESSURE_VENT = "/pressure/vent"   # Drop to 0 mbar (operator safety control)

TOPIC_HEKA_RESISTANCE = "/heka/resistance_mohm"      # std_msgs/Float32 computed/live
TOPIC_HEKA_MONITOR_V = "/heka/monitor_v"            # std_msgs/Float32
TOPIC_HEKA_MONITOR_STEP_V = "/heka/monitor_step_v"  # std_msgs/Float32
TOPIC_HEKA_VOLTAGE_RAW = "/heka/voltage_raw_v"      # Float32MultiArray [rate_hz, samples...]
TOPIC_HEKA_CURRENT_PA = "/heka/current_pa"          # Float32MultiArray [rate_hz, samples...]


def latched_qos(depth=1):
    """QoS for set-and-forget values (pressure setpoints, pressure state).

    Transient-local durability means a node that starts *after* the value was
    published still receives the last one. Without it, restarting the pressure
    node would leave it using its parameter defaults until the GUI happened to
    republish, and the logger would miss the pressure state of a trial that
    started before it subscribed.
    """
    return QoSProfile(
        depth=depth,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

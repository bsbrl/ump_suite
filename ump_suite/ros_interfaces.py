"""Topic and service names shared by every node in the package."""

# UMP 1 ── primary Sensapex micromanipulator
TOPIC_UMP_TARGET = "/ump/target"          # Int32MultiArray [x,y,z,d,speed] absolute
TOPIC_UMP_LIVE   = "/ump/live"            # Int32MultiArray [x,y,z,d]

# UMP 2 ── secondary Sensapex micromanipulator
TOPIC_UMP2_TARGET = "/ump2/target"
TOPIC_UMP2_LIVE   = "/ump2/live"

# ODrive motor (single axis)
TOPIC_MOTOR_TGT  = "/motor/target_counts"   # Int32 absolute encoder counts
TOPIC_MOTOR_LIVE = "/motor/live_counts"     # Int32 current counts

# Pressure control (Arduino-driven solenoids)
TOPIC_SOL1_CMD   = "/pressure/solenoid1/cmd"     # Bool: True=ON, False=OFF
TOPIC_SOL2_CMD   = "/pressure/solenoid2/cmd"     # Bool: True=ON, False=OFF
TOPIC_SOL1_STATE = "/pressure/solenoid1/state"   # Bool: echoed actual state
TOPIC_SOL2_STATE = "/pressure/solenoid2/state"   # Bool: echoed actual state

# Camera (Blackfly via PySpin)
TOPIC_CAM_IMAGE_COMPRESSED = "/camera/image/compressed"   # CompressedImage (jpeg)
TOPIC_CAM_FPS              = "/camera/fps"                # Float32
TOPIC_CAM_REC_CMD          = "/camera/record_cmd"         # String: path to start, "" to stop

# Services (all std_srvs/Trigger)
SRV_ACQ_START = "/acq/start"
SRV_ACQ_STOP  = "/acq/stop"
SRV_ZERO      = "/ump/calibrate_zero"
SRV_ZERO2     = "/ump2/calibrate_zero"

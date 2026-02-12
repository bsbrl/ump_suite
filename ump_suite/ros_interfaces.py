# ump_suite/ros_interfaces.py

# Topics
TOPIC_UMP_TARGET = "/ump/target"           # std_msgs/Int32MultiArray: [x,y,z,d,speed]
TOPIC_UMP_LIVE   = "/ump/live"             # std_msgs/Int32MultiArray: [x,y,z,d]

TOPIC_MOTOR_TGT  = "/motor/target_counts"  # std_msgs/Int32
TOPIC_MOTOR_JOG  = "/motor/jog_dir"        # std_msgs/Int32: -1,0,+1
TOPIC_MOTOR_LIVE = "/motor/live_counts"    # std_msgs/Int32

# Camera topics
TOPIC_CAM_IMAGE_COMPRESSED = "/camera/image/compressed"   # sensor_msgs/CompressedImage
TOPIC_CAM_FPS = "/camera/fps"                             # std_msgs/Float32
TOPIC_CAM_REC_CMD = "/camera/record_cmd"                  # std_msgs/String

# Services
SRV_ACQ_START = "/acq/start"               # std_srvs/Trigger
SRV_ACQ_STOP  = "/acq/stop"                # std_srvs/Trigger
SRV_ZERO      = "/ump/calibrate_zero"      # std_srvs/Trigger
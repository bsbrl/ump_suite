# ump_suite/ros_interfaces.py

# UMP topics
TOPIC_UMP_TARGET = "/ump/target"          # std_msgs/Int32MultiArray: [x,y,z,d,speed] absolute
TOPIC_UMP_DELTA  = "/ump/delta"           # std_msgs/Int32MultiArray: [dx,dy,dz,dd]
TOPIC_UMP_LIVE   = "/ump/live"            # std_msgs/Int32MultiArray: [x,y,z,d]
TOPIC_UMP_SPEED  = "/ump/target_speed"    # std_msgs/Int32: speed used for delta commands

# Motor topics
TOPIC_MOTOR_TGT   = "/motor/target_counts"   # std_msgs/Int32 absolute
TOPIC_MOTOR_DELTA = "/motor/delta_counts"    # std_msgs/Int32 delta
TOPIC_MOTOR_JOG   = "/motor/jog_dir"         # std_msgs/Int32: -1,0,+1
TOPIC_MOTOR_LIVE  = "/motor/live_counts"     # std_msgs/Int32

# Camera topics
TOPIC_CAM_IMAGE_COMPRESSED = "/camera/image/compressed"   # sensor_msgs/CompressedImage
TOPIC_CAM_FPS = "/camera/fps"                             # std_msgs/Float32
TOPIC_CAM_REC_CMD = "/camera/record_cmd"                  # std_msgs/String

# UMP2 topics
TOPIC_UMP2_TARGET = "/ump2/target"         # std_msgs/Int32MultiArray: [x,y,z,d,speed] absolute
TOPIC_UMP2_DELTA  = "/ump2/delta"          # std_msgs/Int32MultiArray: [dx,dy,dz,dd]
TOPIC_UMP2_LIVE   = "/ump2/live"           # std_msgs/Int32MultiArray: [x,y,z,d]
TOPIC_UMP2_SPEED  = "/ump2/target_speed"   # std_msgs/Int32: speed used for delta commands

# Services
SRV_ACQ_START = "/acq/start"              # std_srvs/Trigger
SRV_ACQ_STOP  = "/acq/stop"               # std_srvs/Trigger
SRV_ZERO      = "/ump/calibrate_zero"     # std_srvs/Trigger
SRV_ZERO2     = "/ump2/calibrate_zero"    # std_srvs/Trigger
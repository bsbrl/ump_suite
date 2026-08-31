"""Brings up every node in the ump_suite at once.

Most nodes are launched normally; the camera is started via ExecuteProcess
because PySpin needs the system Spinnaker libraries and a dedicated venv
(see the bash one-liner below) rather than the colcon Python environment.
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


# PySpin requires the bundled Spinnaker .so files and is installed in a
# separate virtualenv. Activate it before running the camera node.
# Brightness parameters are passed on the command line because the camera runs
# as an ExecuteProcess (for the PySpin venv) rather than a launch_ros Node, so
# it cannot take a `parameters=[{...}]` block.
#
#   target_mean_grey     at startup the node measures the delivered image and
#                        solves for the exposure whose mean grey level is this,
#                        then holds it. The camera's own auto-exposure aims at
#                        roughly mid grey, which is why the live view looks far
#                        dimmer than the eyepiece. 200 of 255 is bright without
#                        clipping; raise toward 235 for a brighter field, and
#                        keep an eye on saturation.
#   exposure_time_us     set > 0 to state the exposure outright and skip the
#                        calibration entirely.
#   use_auto_exposure    hand brightness back to the camera's own loop. Handy
#                        when the illumination changes mid-session, but it
#                        couples background brightness to what is in frame.
#   lock_exposure_while_recording
#                        only meaningful with use_auto_exposure; freezes
#                        exposure for each logged trial.
CAMERA_PARAMS = (
    "--ros-args"
    " -p publish_hz:=30.0"
    " -p target_mean_grey:=200.0"
    " -p exposure_time_us:=0.0"
    " -p gain_db:=0.0"
    " -p use_auto_exposure:=false"
    " -p lock_exposure_while_recording:=true"
)

CAMERA_BOOTSTRAP = (
    "export PYTHONNOUSERSITE=1; "
    "export LD_LIBRARY_PATH=/opt/spinnaker/lib:$LD_LIBRARY_PATH; "
    "source ~/venvs/pyspin_cam/bin/activate; "
    f"python -m ump_suite.camera_node {CAMERA_PARAMS}"
)


def generate_launch_description():
    return LaunchDescription([
        # Both UMP devices run in one process to share the Sensapex SDK
        # singleton (UDP socket). Separate processes cause port conflicts
        # and timeouts on the second device.
        Node(
            package="ump_suite",
            executable="ump_dual_driver_node",
            output="screen",
        ),

        Node(
            package="ump_suite",
            executable="pressure_node",
            output="screen",
            parameters=[{
                "channel": 0,
                "poll_ms": 100,
                # Safety envelope, intersected with the range the controller
                # reports. Tighten these to protect the pipette.
                "max_mbar": 1000.0,
                "min_mbar": -1000.0,
            }],
        ),

        Node(
            package="ump_suite",
            executable="odrive_driver_node",
            output="screen",
            parameters=[{
                "poll_ms": 50,
                "jog_speed_turns_s": 0.5,
                "goto_speed_turns_s": 0.5,
                "deadband_counts": 200,
            }],
        ),

        ExecuteProcess(cmd=["bash", "-lc", CAMERA_BOOTSTRAP], output="screen"),

        Node(
            package="ump_suite",
            executable="logger_node",
            output="screen",
            parameters=[{"log_interval_ms": 200}],
        ),

        Node(
            package="ump_suite",
            executable="gui_node",
            output="screen",
        ),

        Node(
            package="ump_suite",
            executable="heka_udp_receiver_node",
            output="screen",
            parameters=[{"port": 5005}],
        ),
    ])

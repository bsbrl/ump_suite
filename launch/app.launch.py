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
CAMERA_BOOTSTRAP = (
    "export PYTHONNOUSERSITE=1; "
    "export LD_LIBRARY_PATH=/opt/spinnaker/lib:$LD_LIBRARY_PATH; "
    "source ~/venvs/pyspin_cam/bin/activate; "
    "python -m ump_suite.camera_node"
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
    ])

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():

    camera = ExecuteProcess(
        cmd=[
            "bash", "-lc",
            "export PYTHONNOUSERSITE=1; "
            "export LD_LIBRARY_PATH=/opt/spinnaker/lib:$LD_LIBRARY_PATH; "
            "source ~/venvs/pyspin_cam/bin/activate; "
            "python -m ump_suite.camera_node"
        ],
        output="screen",
    )

    return LaunchDescription([
        Node(
            package="ump_suite",
            executable="ump_driver_node",
            output="screen",
            parameters=[{"device_id": 1, "poll_ms": 50}],
        ),

        Node(
            package="ump_suite",
            executable="odrive_driver_node",
            output="screen",
            parameters=[{
                "poll_ms": 50,
                "jog_speed_turns_s": 0.5,
                "goto_speed_turns_s": 0.5,
                "deadband_counts": 200
            }],
        ),

        camera,

        Node(
            package="ump_suite",
            executable="logger_node",
            output="screen",
            parameters=[{"log_interval_ms": 500}],
        ),

        Node(
            package="ump_suite",
            executable="gui_node",
            output="screen",
        ),
    ])
"""
Brings up every node in the ump_suite at once.

Most nodes are launched normally; the camera is started via ExecuteProcess
because PySpin needs the system Spinnaker libraries and a dedicated venv
(see the bash one-liner below) rather than the colcon Python environment.
"""

from launch import LaunchDescription
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

from ump_suite.runtime_guard import acquire_process_lock


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
#   binning              combine b x b pixels in the camera, keeping the whole
#                        field of view. 2 for the 20 MP BFS-U3-200S6C (2736 x
#                        1824); 1 for a small sensor such as the old 1.6 MP one.
#                        Changing it changes um/px: recalibrate the camera.
#   roi_width, roi_height, roi_offset_x, roi_offset_y
#                        crop in the camera, in binned pixels (0 = whole width
#                        or height, offset -1 = centred). The microscope's light
#                        circle sits left of the BFS-U3-200S6C's centre, leaving
#                        its right edge dark; 2304 x 1728 at (0, 0) is the 4:3
#                        part that stays lit with the 4x objective. 4:3 also
#                        matches the policy's 320 x 240 input, so frames are
#                        neither padded in training nor stretched in rollout.
#                        Once the camera is centred, 2432 x 1824 centred is the
#                        full-height 4:3 crop.
#   raw_bayer            take the raw Bayer mosaic with the camera's image
#                        processor off and convert colours on this PC. With the
#                        processor on, the BFS-U3-200S6C runs at 6 fps in every
#                        mode; off, it reaches 32 fps binned. false for the old
#                        camera (its processor never limited it).
#   link_limit_mb_s      USB bandwidth the camera may use. It powers up at 60
#                        MB/s; 380 is what FLIR rates it at.
#   frame_rate           cap on the camera's frame rate. 25 rather than its 32:
#                        it must stay below publish_hz or the preview throttle
#                        skips every other frame, and 25 fps is plenty for a
#                        3 Hz policy at a third less CPU.
#   gain_db              with the processor off the camera averages binned
#                        pixels instead of summing them; 12 dB (x4) gives back
#                        the brightness at the same exposure (measured: within
#                        3% of the processor's Sum, and no noisier).
CAMERA_PARAMS = (
    "--ros-args"
    " -p publish_hz:=30.0"
    " -p binning:=2"
    " -p roi_width:=2304"
    " -p roi_height:=1728"
    " -p roi_offset_x:=0"
    " -p roi_offset_y:=0"
    " -p raw_bayer:=true"
    " -p link_limit_mb_s:=380.0"
    " -p frame_rate:=25.0"
    " -p target_mean_grey:=200.0"
    " -p exposure_time_us:=0.0"
    " -p gain_db:=12.0"
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
    acquire_process_lock("suite")
    gui = Node(package="ump_suite", executable="gui_node", output="screen")
    return LaunchDescription([
        RegisterEventHandler(OnProcessExit(
            target_action=gui,
            on_exit=[EmitEvent(event=Shutdown(reason="Rig control window closed"))],
        )),
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

        # The logging period, the dataset fps recorded by the converter, and the
        # policy's CONTROL_HZ all describe the same quantity and must agree. At
        # 200 ms (5 Hz) a chunk whose steps are 200 ms apart in the demonstration
        # was replayed at the policy's 3 Hz, i.e. ~40% slower than it was
        # performed. 333 ms is 3 Hz, matching `--fps 3` and CONTROL_HZ = 3.0.
        # If you change this, change those two as well.
        Node(
            package="ump_suite",
            executable="logger_node",
            output="screen",
            parameters=[{"log_interval_ms": 333}],
        ),

        gui,

        # Segmentation of the camera feed for the orient skill and MicroVLA's
        # vision step. Without a model file it stays idle and says so on
        # /vision/status; the rest of the rig is unaffected.
        Node(
            package="ump_suite",
            executable="vision_node",
            output="screen",
            parameters=[{
                "max_hz": 10.0,
                "conf": 0.35,
                "imgsz": 1024,
                "device": "0",
            }],
        ),

        Node(
            package="ump_suite",
            executable="heka_udp_receiver_node",
            output="screen",
            parameters=[{"port": 5005}],
        ),
    ])

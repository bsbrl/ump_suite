# ump_suite

A ROS2 (Humble, `ament_python`) package for collecting datasets and running closed-loop VLA policies on a rig built around **Sensapex UMP micromanipulators**, an **ODrive**-driven height stage and a **FLIR Blackfly S** camera.

The package wraps every device behind a small ROS2 node, ships a Tk GUI for manual teleop, a logger that writes synchronized image / video / CSV trials, and a thin client that lets an [OpenPI](https://github.com/Physical-Intelligence/openpi) policy server drive the rig in closed loop.

---

## Hardware

| Device | Driver / SDK | Node |
|---|---|---|
| Sensapex UMP (×2) | `sensapex` Python SDK + `libum.so` | [ump_driver_node.py](ump_suite/ump_driver_node.py) |
| ODrive single-axis motor (height stage) | `odrive` Python SDK | [odrive_driver_node.py](ump_suite/odrive_driver_node.py) |
| FLIR Blackfly S camera | PySpin (Spinnaker) | [camera_node.py](ump_suite/camera_node.py) |

A copy of the Sensapex shared library used during development is bundled at [InstallationFiles/libum.so](InstallationFiles/libum.so).

---

## What's in the package

```
ump_suite/
├── launch/app.launch.py         # Brings up every node at once
├── ump_suite/
│   ├── ros_interfaces.py        # Topic / service name constants shared by all nodes
│   ├── ump_driver_node.py       # Sensapex UMP driver (one per device)
│   ├── odrive_driver_node.py    # ODrive height-stage driver
│   ├── camera_node.py           # PySpin camera publisher + mp4 recorder
│   ├── logger_node.py           # CSV + frame + video dataset logger
│   ├── gui_node.py              # Tk control panel
│   ├── sensapex_env.py          # Synchronous ROS2 client used by VLA rollouts
│   ├── main.py                  # Closed-loop OpenPI rollout (absolute targets)
│   └── _rollout_common.py       # Shared CLI args / E-stop / SIGINT helpers
└── InstallationFiles/libum.so   # Sensapex shared library
```

### ROS topics & services

All names live in [ros_interfaces.py](ump_suite/ros_interfaces.py).

| Name | Type | Direction | Notes |
|---|---|---|---|
| `/ump/live`, `/ump2/live` | `std_msgs/Int32MultiArray` | publish | Current `[x, y, z, d]` in **centered counts** (0 = middle of travel) |
| `/ump/target`, `/ump2/target` | `std_msgs/Int32MultiArray` | subscribe | Absolute target `[x, y, z, d, speed]` |
| `/motor/live_counts` | `std_msgs/Int32` | publish | Current ODrive shadow encoder count |
| `/motor/target_counts` | `std_msgs/Int32` | subscribe | Absolute target encoder count |
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | publish | JPEG preview from PySpin grabber |
| `/camera/fps` | `std_msgs/Float32` | publish | Effective grabber FPS |
| `/camera/record_cmd` | `std_msgs/String` | subscribe | Path = start mp4 recording, `""` = stop |
| `/ump/calibrate_zero`, `/ump2/calibrate_zero` | `std_srvs/Trigger` | service | Calibrate zero at the current pose |
| `/acq/start`, `/acq/stop` | `std_srvs/Trigger` | service | Begin / end a logged trial |

The UMP driver translates between the Sensapex SDK's unsigned device units and a symmetric "centered counts" frame (`CENTER_OFFSET = 10000`), so the GUI, logger and policies always see signed values around zero.

---

## Nodes

### `ump_driver_node`
One process per Sensapex stage. Connects to the UMP at the configured `device_id`, publishes the live pose at `poll_ms`, and forwards `[x, y, z, d, speed]` targets to `stage.goto_pos`. Topic names are derived from the `topic_prefix` parameter so two instances can be launched side by side (`ump`, `ump2`).

### `odrive_driver_node`
Connects via `odrive.find_any()`, puts axis 0 into closed-loop velocity control, and implements a software bang-bang position controller on top: every tick it diffs the latest target against `encoder.shadow_count` and commands `±goto_speed_turns_s` until inside `deadband_counts`. The axis is returned to idle on shutdown.

### `camera_node`
Initializes the first PySpin camera, prefers `BGR8` and `NewestOnly` stream buffering so the policy / GUI always see the freshest frame. A worker thread:
- publishes a JPEG preview (`jpeg_quality`) at `publish_hz` on `/camera/image/compressed`
- publishes the actual grab rate on `/camera/fps`
- writes every captured frame to an mp4 (`record_fps`) when recording is active

Recording is toggled by sending a path on `/camera/record_cmd` (empty string to stop).

PySpin needs the system Spinnaker `.so` libraries plus a dedicated virtualenv, so the launch file starts the camera node via `ExecuteProcess` with the venv activated rather than as a normal `ament_python` executable. Edit the `CAMERA_BOOTSTRAP` string in [launch/app.launch.py](launch/app.launch.py) to match your setup.

### `logger_node`
Builds a synchronized dataset:
1. Subscribes to **live** topics (UMP1, UMP2, motor) and to **target** topics published by the GUI / policy.
2. On `/acq/start`, picks the next free `trial_N` ID under `logs/`, opens `logs/trial_N.csv`, creates `saved_frames/trial_N/`, and tells the camera to record `saved_videos/trial_N.mp4`.
3. Every `log_interval_ms` it saves the latest JPEG to `saved_frames/trial_N/frame_NNNNNN.png` and appends one CSV row with the live pose, the most-recent commanded target, and the saved image's path.
4. On `/acq/stop` it closes the file and sends an empty record command to the camera.

The latest target is **not cleared** between ticks, so even if the user stops issuing commands the most recent target keeps appearing in the log and `(target − current)` is always meaningful.

CSV columns:

```
timestep,
current_x, current_y, current_z, current_d, current_motor,
target_x,  target_y,  target_z,  target_d,  target_motor,
current_x2, current_y2, current_z2, current_d2,
target_x2,  target_y2,  target_z2,  target_d2,
image_path
```

### `gui_node`
A Tkinter control panel split into a controls column and a live camera preview. Two `_UmpPanel` instances drive UMP1 and UMP2 (each with X / Y / Z / D entries, ▲▼ bump buttons, axis step, speed, **Send Now**, **Home**, **Sync to Live**, **Calibrate Zero**), plus a row for the ODrive motor and Start / Stop buttons that call `/acq/start` and `/acq/stop`.

Keyboard shortcuts (UMP1 only):

| Key | Axis |
|---|---|
| `W` / `S` | Y +/− |
| `A` / `D` | X −/+ |
| `↑` / `↓` | Z +/− |
| `,` `<` / `.` `>` | D −/+ |

All commands are absolute targets — the bump buttons mutate the locally-held target and republish the full vector.

---

## Closed-loop VLA rollouts

[main.py](ump_suite/main.py) and [sensapex_env.py](ump_suite/sensapex_env.py) connect this rig to an [OpenPI](https://github.com/Physical-Intelligence/openpi) policy server.

`SensapexEnv` spins its own `rclpy` node in a background thread, subscribes to `/camera/image/compressed`, `/ump/live` and `/motor/live_counts`, and exposes a synchronous interface:

```python
env = SensapexEnv(default_speed=100)
obs = env.get_observation()      # SensapexObs(image_rgb, state=[x,y,z,d,h_ticks])
env.step_absolute(action_5d)     # publishes /ump/target + /motor/target_counts
env.close()
```

`main.py` runs the rollout loop:

1. Asks the user for an instruction.
2. Each tick, grabs `(image, state)` from `SensapexEnv`.
3. Whenever the open-loop chunk is exhausted, packages `image` (resized) + `state` + `prompt` and calls `policy_client.infer(...)` against the OpenPI websocket server.
4. Pops the next 5-dim absolute action from the chunk and runs it through:
   - `clamp_action_5d` — workspace safety box (`X_MIN/MAX`, `Y_MIN/MAX`, `Z_MIN/MAX`, `D_MIN/MAX`, `H_MIN/MAX`)
   - `limit_step` — per-tick max delta on each axis (`MAX_DX/Y/Z/D/H`)
   - optional first-order EMA smoothing (`USE_EMA_SMOOTHING`, `EMA_ALPHA`)
5. Sends the result via `env.step_absolute(...)` and sleeps to hold `CONTROL_FREQUENCY_HZ` (default 3 Hz, matching the dataset).

Two safety paths are wired in:

- **E-stop** — press `q` + Enter at any time. The watcher thread (`start_estop_listener`) flips a flag the loop polls; on the next tick the env is commanded to hold its current state and the loop exits.
- **SIGINT shielding around inference** — `prevent_keyboard_interrupt()` blocks `SIGINT` for the duration of the websocket round-trip so a Ctrl+C in the middle of inference cannot leave the websocket in a half-open state. The interrupt is re-raised cleanly after the call returns.

CLI args (see [`RolloutArgs`](ump_suite/_rollout_common.py)):

```
--remote-host             OpenPI server host (default 127.0.0.1)
--remote-port             OpenPI server port (default 8000)
--max-timesteps           Maximum rollout length (default 600)
--open-loop-horizon       Actions consumed per inference (default 8)
--resize-h / --resize-w   Image size sent to the policy (default 224)
--default-speed           UMP move speed (default 100)
--save-preview            Periodically save the latest frame to disk
--preview-path            Where to write that PNG
--preview-every-n-frames  Throttle for the preview save
--debug-every             Print one state/cmd line every N steps (0 = silent)
```

> ⚠️ The safety limits in [main.py](ump_suite/main.py) (`X_MIN`, `X_MAX`, ..., `H_MIN`, `H_MAX`, `MAX_DX`, ...) are tied to a specific physical setup. **Edit them for your stage before running a rollout.**

---

## Build & install

```bash
# In your ROS2 workspace
cd ~/ros2_ws/src
git clone git@github.com:bsbrl/ump_suite.git

cd ~/ros2_ws
colcon build --packages-select ump_suite
source install/setup.bash
```

### Python dependencies

The driver and rollout code import several non-`rosdep` packages:

- `sensapex` — Sensapex Python SDK (point it at the bundled `libum.so` if needed)
- `odrive` — ODrive Python SDK
- `PySpin` — Spinnaker Python wheel (install into a dedicated venv, see below)
- `opencv-python`, `numpy`, `pillow`
- `tyro`, `openpi-client` — only for the rollout scripts

Because PySpin is picky about the host Python and Spinnaker `.so` paths, the launch file expects a separate virtualenv for the camera node:

```bash
python3.10 -m venv ~/venvs/pyspin_cam
source ~/venvs/pyspin_cam/bin/activate
# install spinnaker_python wheel from FLIR + numpy + opencv-python + rclpy bindings
```

Then update the `CAMERA_BOOTSTRAP` string at the top of [launch/app.launch.py](launch/app.launch.py) so it activates *your* venv and exports the right `LD_LIBRARY_PATH` for `libSpinnaker`.

---

## Running

### Bring everything up

```bash
ros2 launch ump_suite app.launch.py
```

This starts both UMP drivers (`device_id=1` and `device_id=2`), the ODrive driver, the camera (via the bootstrap venv), the logger, and the GUI.

### Collect a dataset trial

1. Launch the suite as above.
2. Use the GUI (or publish on `/ump/target`, `/ump2/target`, `/motor/target_counts` directly) to drive the rig.
3. Click **Start Data Acquisition** — this calls `/acq/start`, which opens `logs/trial_N.csv`, creates `saved_frames/trial_N/`, and asks the camera to record `saved_videos/trial_N.mp4`.
4. Perform the trial. The logger writes one row per `log_interval_ms` (default 200 ms).
5. Click **Stop Data Acquisition** — this calls `/acq/stop`, closes the CSV, and stops the mp4.

Output layout:

```
logs/trial_1.csv
saved_frames/trial_1/frame_000000.png
saved_frames/trial_1/frame_000001.png
...
saved_videos/trial_1.mp4
```

### Run a closed-loop policy rollout

1. Start the OpenPI websocket policy server somewhere reachable.
2. Launch this suite (the rollout needs `/camera/image/compressed`, `/ump/live`, `/motor/live_counts`).
3. **Edit the safety limits in [main.py](ump_suite/main.py) for your stage.**
4. Run:

   ```bash
   ros2 run ump_suite sensapex_rollout \
     --remote-host 127.0.0.1 \
     --remote-port 8000 \
     --max-timesteps 600
   ```

5. Type the natural-language instruction at the prompt.
6. Press `q` + Enter at any time to E-stop and hold the current pose.

---

## Console scripts

Defined in [setup.py](setup.py):

| Script | Module |
|---|---|
| `gui_node` | `ump_suite.gui_node:main` |
| `ump_driver_node` | `ump_suite.ump_driver_node:main` |
| `odrive_driver_node` | `ump_suite.odrive_driver_node:main` |
| `camera_node` | `ump_suite.camera_node:main` |
| `logger_node` | `ump_suite.logger_node:main` |
| `sensapex_rollout` | `ump_suite.main:main_entry` |

---

## Maintainer

Raian Chowdhury — `chowd207@umn.edu`

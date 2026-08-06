# ump_suite

A ROS2 (Humble, `ament_python`) package for collecting datasets and running closed-loop VLA policies on a rig built around **two Sensapex UMP micromanipulators**, an **ODrive**-driven focusing knob and a **FLIR Blackfly S** camera.

The package wraps every device behind a small ROS2 node, ships a Qt GUI for manual teleop, and a logger that writes synchronized image / video / CSV trials. It is the **robot side** only: closed-loop VLA policy clients live in separate repos and drive this rig over the ROS topics below.

---

## Hardware

| Device | Driver / SDK | Node |
|---|---|---|
| Sensapex UMP (×2) | `sensapex` Python SDK + `libum.so` | [ump_driver_node.py](ump_suite/ump_driver_node.py) |
| ODrive single-axis motor (focusing knob) | `odrive` Python SDK | [odrive_driver_node.py](ump_suite/odrive_driver_node.py) |
| FLIR Blackfly S camera | PySpin (Spinnaker) | [camera_node.py](ump_suite/camera_node.py) |
| Fluigent LineUP push-pull pressure controller | Fluigent Python SDK (`fluigent_sdk`) | [pressure_node.py](ump_suite/pressure_node.py) |
| HEKA / patch-clamp monitor stream | UDP packets | [heka_udp_receiver_node.py](ump_suite/heka_udp_receiver_node.py) |

A copy of the Sensapex shared library used during development is bundled at [InstallationFiles/libum.so](InstallationFiles/libum.so).

---

## What's in the package

```
ump_suite/
├── launch/app.launch.py         # Brings up every node at once
├── ump_suite/
│   ├── ros_interfaces.py        # Topic / service name constants shared by all nodes
│   ├── ump_driver_node.py       # Sensapex UMP driver (one per device)
│   ├── odrive_driver_node.py    # ODrive focusing-knob driver
│   ├── camera_node.py           # PySpin camera publisher + mp4 recorder
│   ├── pressure_node.py         # Fluigent push-pull pressure controller
│   ├── logger_node.py           # CSV + frame + video dataset logger
│   ├── gui_node.py              # PyQt control panel
│   └── heka_udp_receiver_node.py # HEKA UDP bridge for voltage/current samples
├── WindowsCode/
│   └── windows_send_heka_data.py # NI-DAQ sender that streams HEKA monitors over UDP
└── InstallationFiles/libum.so   # Sensapex shared library
```

### ROS topics & services

All names live in [ros_interfaces.py](ump_suite/ros_interfaces.py).

| Name | Type | Direction | Notes |
|---|---|---|---|
| `/ump/live`, `/ump2/live` | `std_msgs/Int32MultiArray` | publish | Current `[x, y, z, d]` in absolute Sensapex device counts |
| `/ump/target`, `/ump2/target` | `std_msgs/Int32MultiArray` | subscribe | Absolute Sensapex target `[x, y, z, d, speed]` |
| `/motor/live_counts` | `std_msgs/Int32` | publish | Current ODrive shadow encoder count |
| `/motor/target_counts` | `std_msgs/Int32` | subscribe | Absolute target encoder count |
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | publish | JPEG preview from PySpin grabber |
| `/camera/fps` | `std_msgs/Float32` | publish | Effective grabber FPS |
| `/camera/record_cmd` | `std_msgs/String` | subscribe | Path = start mp4 recording, `""` = stop |
| `/pressure/state_cmd` | `std_msgs/Bool` | subscribe | Binary pressure state: `True` = apply positive setpoint, `False` = apply negative setpoint |
| `/pressure/state` | `std_msgs/Int8` | publish | State actually applied (latched): `1` = positive, `0` = negative, `-1` = vented |
| `/pressure/positive_mbar` | `std_msgs/Float32` | subscribe | Positive setpoint in mbar, clamped to `[0, device max]` (latched) |
| `/pressure/negative_mbar` | `std_msgs/Float32` | subscribe | Negative setpoint in mbar, clamped to `[device min, 0]` (latched) |
| `/pressure/measured_mbar` | `std_msgs/Float32` | publish | Pressure measured by the controller |
| `/heka/voltage_raw_v` | `std_msgs/Float32MultiArray` | publish | HEKA voltage sample packet: `[sample_rate_hz, v0, v1, ...]` |
| `/heka/current_pa` | `std_msgs/Float32MultiArray` | publish | HEKA current sample packet: `[sample_rate_hz, i0, i1, ...]` |
| `/heka/monitor_v` | `std_msgs/Float32` | publish | Latest voltage sample from binary packets; mean monitor voltage for legacy packets |
| `/heka/monitor_step_v` | `std_msgs/Float32` | publish | Legacy monitor step voltage |
| `/heka/resistance_mohm` | `std_msgs/Float32` | publish | Live resistance estimate in MOhm |
| `/ump/calibrate_zero`, `/ump2/calibrate_zero` | `std_srvs/Trigger` | service | Calibrate zero at the current pose |
| `/pressure/vent` | `std_srvs/Trigger` | service | Drop the channel to 0 mbar |
| `/acq/start`, `/acq/stop` | `std_srvs/Trigger` | service | Begin / end a logged trial |

The UMP driver publishes and accepts raw absolute Sensapex device coordinates. There is no `10000` count centering offset in the ROS topics.

---

## Nodes

### `ump_driver_node`
Connects to the UMP at the configured `device_id`, publishes the live absolute pose at `poll_ms`, and forwards `[x, y, z, d, speed]` targets directly to `stage.goto_pos`. Topic names are derived from the `topic_prefix` parameter so devices can expose `/ump/*` and `/ump2/*`.

The `ump_dual_driver_node` entry point runs both devices in one process so they share the Sensapex SDK singleton / UDP socket. This is what [launch/app.launch.py](launch/app.launch.py) uses, because separate UMP processes can conflict on the SDK socket.

### `odrive_driver_node`
Connects via `odrive.find_any()`, puts axis 0 into closed-loop velocity control, and implements a software bang-bang position controller on top: every tick it diffs the latest target against `encoder.shadow_count` and commands `±goto_speed_turns_s` until inside `deadband_counts`. The axis is returned to idle on shutdown.

The ODrive remains available for manual focusing-knob control from the GUI, but it is not included in the policy rollout action vector and is not written into the CSV logger.

### `camera_node`
Initializes the first PySpin camera, prefers `BGR8` and `NewestOnly` stream buffering so the policy / GUI always see the freshest frame. A worker thread:
- publishes a JPEG preview (`jpeg_quality`) at `publish_hz` on `/camera/image/compressed`
- publishes the actual grab rate on `/camera/fps`
- writes every captured frame to an mp4 (`record_fps`) when recording is active

Recording is toggled by sending a path on `/camera/record_cmd` (empty string to stop).

PySpin needs the system Spinnaker `.so` libraries plus a dedicated virtualenv, so the launch file starts the camera node via `ExecuteProcess` with the venv activated rather than as a normal `ament_python` executable. Edit the `CAMERA_BOOTSTRAP` string in [launch/app.launch.py](launch/app.launch.py) to match your setup.

### `pressure_node`
Drives a **Fluigent push-pull pressure controller** (LineUP) through the Fluigent Python SDK. `fgt_detect()` → `fgt_init()` on startup, then the channel's real range is read with `fgt_get_pressureRange()` and used as the hard clamp.

Pressure is commanded as a **binary state**, not a value:

```
/pressure/state_cmd = True   ->  fgt_set_pressure(channel, positive_mbar)
/pressure/state_cmd = False  ->  fgt_set_pressure(channel, negative_mbar)
```

The two mbar setpoints arrive on their own latched topics (published by the GUI), so an operator *or* a policy only has to decide push vs. pull and the node resolves that against whatever values are currently dialed in. Editing a setpoint while that state is active re-applies it immediately; editing one while vented does not re-pressurize.

**Venting** (0 mbar) is the third condition the channel can be in, and it is exposed as the `/pressure/vent` **service** rather than a third value on the command topic. That keeps a policy's action space strictly binary — it cannot vent — while an operator always has one click back to neutral. The reported state on `/pressure/state` is therefore tri-state (`1` / `0` / `-1`), even though the commanded state is binary.

Sign is enforced on both setpoints — the positive setpoint is clamped to `[0, max]` and the negative one to `[min, 0]` — so a mistyped value can never turn a push into a pull on a pipette. The channel is set to **0 mbar on connect**, and vented to 0 mbar before `fgt_close()` on shutdown.

Parameters:

| Name | Default | Notes |
|---|---|---|
| `channel` | `0` | Fluigent pressure channel index. |
| `poll_ms` | `100` | How often `/pressure/measured_mbar` is published. |
| `positive_mbar` | `20.0` | Setpoint used until the GUI publishes one. |
| `negative_mbar` | `-20.0` | Setpoint used until the GUI publishes one. |
| `max_positive_mbar` | `1000.0` | Safety ceiling, intersected with the device range. |
| `min_negative_mbar` | `-1000.0` | Safety floor, intersected with the device range. |

If no controller is detected the node logs an error and stays inert rather than killing the launch, matching the ODrive driver's behaviour.

### `heka_udp_receiver_node`
Listens for UDP packets on `port` (default `5005`). The current Windows sender emits binary packets:

```
header = "<5sdfH": magic=b"HEKA1", first_sample_time, sample_rate_hz, sample_count
payload = repeated float32 pairs: voltage_raw_v, current_pA
```

It republishes voltage packets on `/heka/voltage_raw_v` and current packets on `/heka/current_pa` as `Float32MultiArray` messages whose first element is the sample rate and remaining elements are samples. It also republishes the latest voltage sample on `/heka/monitor_v`, estimates resistance from the test pulse response, and publishes that value on `/heka/resistance_mohm`.

The previous comma-separated packet format is still accepted for compatibility:

```
timestamp, mean_voltage_V, monitor_step_V, resistance_MOhm
```

For now, the GUI plots voltage and current and shows the live resistance estimate in the left control column. The logger includes the same value in the `resistance_mohm` CSV column.

### `logger_node`
Builds a synchronized dataset:
1. Subscribes to **live** topics (UMP1, UMP2) and to **target** topics published by the GUI / policy.
2. Subscribes to `/heka/resistance_mohm` so each row can include the latest finite HEKA resistance value when available, and to `/pressure/state` for the binary pressure column.
3. On `/acq/start`, picks the next free `trial_N` ID under `logs/`, opens `logs/trial_N.csv`, creates `saved_frames/trial_N/`, and tells the camera to record `saved_videos/trial_N.mp4`.
4. Every `log_interval_ms` it saves the latest JPEG to `saved_frames/trial_N/frame_NNNNNN.png` and appends one CSV row with the live pose, the most-recent commanded target, the saved image's path, and the latest resistance when available.
5. On `/acq/stop` it closes the file and sends an empty record command to the camera.

The latest target is **not cleared** between ticks, so even if the user stops issuing commands the most recent target keeps appearing in the log and `(target − current)` is always meaningful.

The ODrive motor is intentionally excluded from the CSV rows. It can still be driven from the GUI, but the dataset state/target columns below are UMP-only.

CSV columns:

```
timestep,
current_x, current_y, current_z, current_d,
target_x,  target_y,  target_z,  target_d,
current_x2, current_y2, current_z2, current_d2,
target_x2,  target_y2,  target_z2,  target_d2,
image_path,
resistance_mohm,
pressure_state
```

`pressure_state` is binary — `1` = positive pressure, `0` = negative pressure — never the mbar value, so the column stays valid when the setpoints are re-dialed between trials.

It is **blank** whenever the channel is vented (including before the first state is commanded, since the node starts vented), because "vented" is neither push nor pull and must not be mislabelled as either. So **set a pressure state before starting a trial** if you want every row populated, and expect blanks for any stretch where you vented mid-trial.

### `gui_node`
A PyQt5 control panel split into a controls column and a live camera / HEKA preview column. Two `UmpPanel` instances drive UMP1 and UMP2 (each with X / Y / Z / D controls, nudge buttons, axis step, speed, **Send Now**, **Home**, **Sync Live**, **Calibrate Zero**), a row for the ODrive motor, a **Pressure (Fluigent)** panel, Start / Stop buttons that call `/acq/start` and `/acq/stop`, rolling voltage/current plots from `/heka/voltage_raw_v` and `/heka/current_pa`, and a live resistance readout.

The pressure panel has a positive and a negative mbar spin box (range ±1000 mbar, sign-locked per box) and one button each:

- **Positive** — publishes `/pressure/state_cmd = True`, so the controller goes to the positive setpoint
- **Negative** — publishes `/pressure/state_cmd = False`, so it goes to the negative setpoint
- **Vent (0 mbar)** — calls `/pressure/vent` to drop the channel to neutral

A colored badge shows the state the node actually applied (`POSITIVE` / `NEGATIVE` / `VENTED` / `--`), next to the live measured pressure. Editing a setpoint republishes it, and the node re-applies it on the spot if that state is currently active. Both setpoints are published once at startup so a state command works before you touch the boxes.

The panel is **mouse-only** — there are deliberately no keyboard shortcuts, so keystrokes always go to the widget you are editing.

All UMP commands are absolute Sensapex targets. The bump buttons mutate the locally-held target and republish the full vector; the GUI spin boxes use the raw device range (`0` to `20000` counts).

---

## Closed-loop VLA rollouts

> ⓘ **The rollout client no longer lives in this package.** `main.py` and `sensapex_env.py` were deleted in commit `699bd1f`, along with the `sensapex_rollout` console script. The policy code now sits in the separate training / inference repos (`~/SmolVLA`, `~/MicroVLA*/rollout`).
>
> This package is now purely the **robot side**. What follows is the interface a policy client has to speak — not documentation of code in this repo.

### What the rig publishes (observation)

| Topic | Use |
|---|---|
| `/camera/image/compressed` | JPEG frame, resize as the policy requires |
| `/ump/live`, `/ump2/live` | `[x, y, z, d]` absolute counts per manipulator → the 8-value state |

### What the rig accepts (action)

| Topic | Use |
|---|---|
| `/ump/target`, `/ump2/target` | `[x, y, z, d, speed]` absolute counts |
| `/pressure/state_cmd` | `Bool` binary pressure state |

So the action is **8 motion values + 1 binary pressure state**:

- The 8 motion values are the two 4-axis UMPs only. The ODrive focusing knob is driven separately through `/motor/target_counts` and is not part of the action vector.
- Pressure is the extra binary value: the model predicts push vs. pull, publishes it on `/pressure/state_cmd`, and the pressure node converts it to whichever mbar setpoint the GUI has dialed in at that moment. The policy never emits mbar — that is why `pressure_state` is logged binary. Venting is a service, not a topic, so a policy cannot vent; that stays an operator control.

This matches the dataset columns the logger writes, so state/action shapes line up between training and inference.

### Client-side responsibilities

These lived in the deleted `main.py` and are now the client's job — worth re-checking whichever repo you roll out from:

- **Workspace clamping** — per-stage min/max boxes on each of the 8 axes. ⚠️ These are tied to one physical setup; they must be set for *your* stage before any rollout.
- **Per-tick step limiting** — cap the delta on each axis so a bad prediction cannot command a large jump.
- **Control rate** — the dataset was collected at ~2.5 Hz and stored at 3 Hz; running much faster tends to overshoot on real hardware.
- **E-stop** — a way to stop sending actions and hold the current pose.
- **Optional EMA smoothing** on the action stream to reduce jitter.

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

The driver nodes import several non-`rosdep` packages:

- `sensapex` — Sensapex Python SDK (point it at the bundled `libum.so` if needed)
- `odrive` — ODrive Python SDK
- `PySpin` — Spinnaker Python wheel (install into a dedicated venv, see below)
- `fluigent_sdk` — Fluigent pressure controller. Install into the same Python that runs the ROS nodes:
  ```bash
  pip install fluigent_sdk
  # or from the bundled SDK release:
  # pip install ~/fluigent_test/sdk_release/fgt-SDK-23.0.0/SDK-23.0.0/Python/fluigent_sdk-23.0.0.zip
  ```
  The wheel ships its own `libfgt_SDK.so`, so unlike PySpin it needs no system libraries and no separate virtualenv.
- `PyQt5` — modern desktop GUI (`apt install python3-pyqt5`)
- `opencv-python`, `numpy`

(`tyro` / `openpi-client` are no longer needed — they were only used by the rollout client that has since moved out of this package.)

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

This starts the dual UMP driver (`device_id=1` and `device_id=2` in one process), the ODrive driver, the camera (via the bootstrap venv), the Fluigent pressure controller, the HEKA UDP receiver, the logger, and the GUI.

### Collect a dataset trial

1. Launch the suite as above.
2. Use the GUI (or publish on `/ump/target`, `/ump2/target`, `/motor/target_counts` and `/pressure/state_cmd` directly) to drive the rig. The CSV logger records UMP state/targets, HEKA resistance and the binary pressure state, but not the ODrive motor.
   Set the pressure state before step 3 so the `pressure_state` column is populated from the first row.
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

The rollout client is **not part of this package** — run it from whichever policy repo you are using (`~/SmolVLA`, `~/MicroVLA*/rollout`). From this side:

1. Launch this suite, so the client has `/camera/image/compressed`, `/ump/live` and `/ump2/live` to read and `/ump/target`, `/ump2/target`, `/pressure/state_cmd` to write.
2. Set the positive / negative mbar setpoints in the GUI. The policy only emits the binary pressure state, so these values decide what that state physically means for the whole rollout.
3. **Check the client's workspace limits and per-tick step caps for this stage before starting.**
4. Start the policy client (and its policy server, if it uses one).

Keep the GUI up during a rollout: the **Vent (0 mbar)** button and the manual UMP controls stay live, so you can intervene without stopping the client.

---

## Console scripts

Defined in [setup.py](setup.py):

| Script | Module |
|---|---|
| `gui_node` | `ump_suite.gui_node:main` |
| `ump_driver_node` | `ump_suite.ump_driver_node:main` |
| `ump_dual_driver_node` | `ump_suite.ump_driver_node:main_dual` |
| `odrive_driver_node` | `ump_suite.odrive_driver_node:main` |
| `camera_node` | `ump_suite.camera_node:main` |
| `pressure_node` | `ump_suite.pressure_node:main` |
| `logger_node` | `ump_suite.logger_node:main` |
| `heka_udp_receiver_node` | `ump_suite.heka_udp_receiver_node:main` |

---

## Maintainer

Raian Haider Chowdhury — `chowd207@umn.edu`

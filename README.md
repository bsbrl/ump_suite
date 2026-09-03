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
| `/pressure/mbar` | `std_msgs/Float32` | subscribe | Requested pressure in mbar; negative pulls, positive pushes, `0` vents (latched) |
| `/pressure/target_mbar` | `std_msgs/Float32` | publish | Pressure actually written to the device, i.e. the request after clamping (latched) |
| `/pressure/measured_mbar` | `std_msgs/Float32` | publish | Pressure measured by the controller's sensor |
| `/heka/voltage_raw_v` | `std_msgs/Float32MultiArray` | publish | HEKA voltage sample packet: `[sample_rate_hz, v0, v1, ...]` |
| `/heka/current_pa` | `std_msgs/Float32MultiArray` | publish | HEKA current sample packet: `[sample_rate_hz, i0, i1, ...]` |
| `/heka/monitor_v` | `std_msgs/Float32` | publish | Latest voltage sample from binary packets; mean monitor voltage for legacy packets |
| `/heka/monitor_step_v` | `std_msgs/Float32` | publish | Legacy monitor step voltage |
| `/heka/resistance_mohm` | `std_msgs/Float32` | publish | Live resistance estimate in MOhm |
| `/ump/calibrate_zero`, `/ump2/calibrate_zero` | `std_srvs/Trigger` | service | Calibrate zero at the current pose |
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

#### Brightness and exposure

The camera powers up with `ExposureAuto = Continuous` aiming at roughly **mid
grey**, and with `BalanceWhiteAuto = Continuous`. On a brightfield scope under
white light that renders a bright field at about half scale, which is why the
live view can look far dimmer than the eyepiece even when the light path is
perfectly fine. Measured here: the auto loop settled at 3.5 ms and 0 dB gain
while the sensor fully saturates at ~6 ms — the light was never the limit.

Both auto loops are also **content-dependent**, which is a problem for dataset
collection. With average metering, a dark pipette entering the frame lowers the
average and the loop brightens the whole scene, so background brightness encodes
manipulator position. Measured on the recorded trials: **r = −0.99** between
frame mean and pipette x, a swing of ~15 grey levels. Continuous white balance
drifts the same way, in colour.

The node therefore calibrates once at startup and then holds both fixed:

| Parameter | Default | Notes |
|---|---|---|
| `target_mean_grey` | `200.0` | Measures the delivered image and bisects exposure until its mean grey matches, then holds it. `0` disables. |
| `exposure_time_us` | `0.0` | `> 0` states the exposure outright and skips calibration. |
| `gain_db` | `0.0` | Prefer exposure over gain; gain amplifies noise. |
| `exposure_search_max_us` | `15000.0` | Upper bound for the search. |
| `use_auto_exposure` | `false` | Hand brightness back to the camera's own loop. |
| `target_grey_percent` | `80.0` | Target for that loop. Only used when `use_auto_exposure` is true. |
| `lock_exposure_while_recording` | `true` | Freezes exposure per trial. Applies **only** when `use_auto_exposure` is true — with `target_mean_grey` or an explicit `exposure_time_us`, exposure is already deterministic and re-solving would re-target the current frame mean, which at the start of a trial may already contain the pipette. |
| `white_balance` | `Once` | `Once` converges on the field then holds; `Continuous` keeps adapting; `Off` freezes as-is. |
| `balance_ratio_red` / `_blue` | `0.0` | `> 0` pins exact gains, reproducing a previous session. |

Startup then reports what it settled on, and those numbers are what you pin to
reproduce a session later:

```
Exposure calibrated to 4110 us for mean grey 200.8 (target 200); held fixed
White balance held at red=1.469 blue=2.876
```

Under white light that yields R/G/B = 200.8 / 200.8 / 200.7, no saturation, and
a frame-to-frame drift of 0.02–0.06 grey levels.

> **Filters change everything.** The 153-episode `OocyteTargetting` dataset was
> shot through a **green filter**: its frames are R≈10, G≈217, B≈48, effectively
> single-channel. That green channel was itself well exposed (mean 216, no
> clipping) — the low *grey* mean of 130 was the filter, not under-exposure. But
> the exposure/position coupling above is present in that data regardless, and
> a model trained on green-filtered frames will not transfer to white light.
> Re-calibrate whenever the filter or illumination changes; the startup
> calibration does this for you automatically.

Every camera access — the grab loop and any recalibration triggered from the ROS
executor thread — is serialized behind one lock, because PySpin acquisition is not
thread safe. The lock is held per access rather than for a whole calibration, so a
bisection cannot stall the preview stream.

Calibration is driven entirely by measuring frames. The camera's `ExposureTime`
readback is cached on this model and cannot be trusted — it reported a constant
2223 µs while the delivered image swung between mean 123 and 235 — so any logic
that reads it back and writes it somewhere else silently corrupts the setting.

If even the longest allowed exposure cannot reach the target, the node says so
and names the likely causes rather than quietly under-exposing:

```
[WARN] cannot reach mean grey 250 even at 600 us (best 43.4); holding the longest
       allowed exposure. Check the light path, the beam splitter and any ND filter.
```

That message is the dividing line between a settings problem and a real optical
one. Note also that full-resolution BGR8 is capped near **12.9 fps** by
`DeviceLinkThroughputLimit` (60 MB/s), regardless of `publish_hz`.

PySpin needs the system Spinnaker `.so` libraries plus a dedicated virtualenv, so the launch file starts the camera node via `ExecuteProcess` with the venv activated rather than as a normal `ament_python` executable. Edit the `CAMERA_BOOTSTRAP` string in [launch/app.launch.py](launch/app.launch.py) to match your setup.

### `pressure_node`
Drives a **Fluigent push-pull pressure controller** (LineUP) through the Fluigent Python SDK. `fgt_detect()` → `fgt_init()` on startup, then the channel's real range is read with `fgt_get_pressureRange()` and used as the hard clamp.

Pressure is commanded as an **exact value in mbar** on one topic:

```
/pressure/mbar = -20.0  ->  fgt_set_pressure(channel, -20.0)   (pull)
/pressure/mbar =  50.0  ->  fgt_set_pressure(channel,  50.0)   (push)
/pressure/mbar =   0.0  ->  vented
```

One number, sign carries the direction. The topic is latched, so the node picks up the last commanded pressure even if it restarts.

Because `/pressure/mbar` has **more than one publisher** (the GUI and the rollout client), a restart delivers the last latched sample from *each* of them, in an order ROS does not define — so "apply whatever arrives last" could resurrect a stale command over a vent. Startup therefore collects that history for `startup_grace_s` (default 1.0 s) and only restores it when every publisher agrees. A conflict holds at 0 mbar and names the values it saw:

```
Conflicting latched pressure commands at startup (-30.0, +0.0 mbar) -
`/pressure/mbar` has more than one publisher and their order is undefined.
Holding +0.0 mbar; send the intended pressure explicitly.
```

Set `startup_grace_s: 0.0` to restore the historical apply-whatever-arrives-last behaviour.

Incoming values are clamped to the range the controller reports for its channel (intersected with the `min_mbar` / `max_mbar` parameters), and non-finite values are rejected outright, both with a warning. The channel is set to **0 mbar on connect**, and vented to 0 mbar before `fgt_close()` on shutdown — including on Ctrl+C and on the SIGTERM `ros2 launch` sends.

Two readbacks come back out:

- **`/pressure/target_mbar`** — the value actually written to the device, published only after `fgt_set_pressure` succeeded. Because it is the post-clamp value, the dataset can never claim a pressure the controller never received.
- **`/pressure/measured_mbar`** — the controller's own sensor, polled every `poll_ms`.

Comparing the two is how you see the channel settling, or spot a request that got clamped.

Parameters:

| Name | Default | Notes |
|---|---|---|
| `channel` | `0` | Fluigent pressure channel index. |
| `poll_ms` | `100` | How often `/pressure/measured_mbar` is published. |
| `max_mbar` | `1000.0` | Safety ceiling, intersected with the device range. |
| `min_mbar` | `-1000.0` | Safety floor, intersected with the device range. |
| `startup_grace_s` | `1.0` | Window for collecting latched commands at startup before acting on them. `0` disables it. |

If no controller is detected the node logs an error and stays inert rather than killing the launch, matching the ODrive driver's behaviour.

### `heka_udp_receiver_node`
Listens for UDP packets on `port` (default `5005`), draining the socket each tick
(bounded at 64 packets) rather than taking one packet per 10 ms timer tick. The
sender emits 100 packets/s and the timer fires at the same rate, so handling a
single packet per tick left zero headroom: any jitter accumulated as latency and
then as silent drops. It warns if it keeps hitting the per-tick cap. The current Windows sender emits binary packets:

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
2. Subscribes to `/heka/resistance_mohm` so each row can include the latest finite HEKA resistance value when available, and to `/pressure/mbar` for the pressure column.
3. On `/acq/start`, picks the next free `trial_N` ID by inspecting **`logs/`, `saved_frames/` and `saved_videos/` together**, opens `logs/trial_N.csv`, creates `saved_frames/trial_N/`, and tells the camera to record `saved_videos/trial_N.mp4`. Scanning all three matters: deleting a CSV while its frame directory survives would otherwise hand the number back out and the new run would overwrite the old frames.
4. Every `log_interval_ms` it saves the latest JPEG to `saved_frames/trial_N/frame_NNNNNN.png` and appends one CSV row with the live pose, the most-recent commanded target, the saved image's path, the latest resistance when available, and the timing columns below.
5. On `/acq/stop` it closes the file, sends an empty record command to the camera, and reports the logging rate it actually achieved.

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
target_pressure,
measured_pressure,
wall_time,
image_stamp,
state_stamp,
image_age_s
```

The four timing columns exist so a late tick or a stalled camera is detectable
after the fact. Without them a frozen camera silently writes the same frame into
many rows and the dataset still looks perfectly well formed:

- **`wall_time`** — POSIX time the row was written.
- **`image_stamp`** — POSIX time the camera stamped this frame, from the
  `CompressedImage` header. Blank if unavailable.
- **`state_stamp`** — POSIX time the newest manipulator state arrived.
- **`image_age_s`** — `wall_time − image_stamp`, i.e. how stale the saved frame
  is. The node also warns live once this exceeds `stale_image_warn_s`
  (default 0.5 s).

The two pressure columns, both in mbar with negative meaning pull:

- **`target_pressure`** — from `/pressure/target_mbar`: the pressure actually applied to the device. This is the action label to train on.
- **`measured_pressure`** — from `/pressure/measured_mbar`: the controller's sensor reading. Expect it to lag `target_pressure` by a poll tick or two while the channel settles, and to sit slightly off the target.

Both are blank until the pressure node publishes, which it does as soon as it connects (it applies 0 mbar on startup), so in practice they are populated from the first row of any trial where the pressure node is running.

**Logging rate.** `log_interval_ms` defaults to **333 ms (3 Hz)** in the launch
file, matching the converter's `--fps 3` and the policy's `CONTROL_HZ`. These
three describe the same quantity and must agree: at 200 ms a chunk whose steps
were 200 ms apart in the demonstration got replayed at 3 Hz, about 40% slower
than it was performed. Because a ROS timer is best effort and the PNG encode runs
inside the callback, the node measures what actually happened and reports it at
`/acq/stop`, warning if it fell below 90% of the configured rate.

### `gui_node`
A PyQt5 control panel split into a controls column and a live camera / HEKA preview column. Two `UmpPanel` instances drive UMP1 and UMP2 (each with X / Y / Z / D controls, nudge buttons, axis step, speed, **Send Now**, **Home**, **Sync Live**, **Calibrate Zero**), a row for the ODrive motor, a **Pressure (Fluigent)** panel, Start / Stop buttons that call `/acq/start` and `/acq/stop`, rolling voltage/current plots from `/heka/voltage_raw_v` and `/heka/current_pa`, and a live resistance readout.

The pressure panel is one box plus a Send button:

- **Pressure box** — type the exact value in mbar, with a leading `-` to pull. Range ±1000 mbar, one decimal.
- **Send** — publishes the box's value on `/pressure/mbar`. Nothing reaches the device until you press it.
- **Preset buttons** — `+50`, `+20`, `0`, `-10`, `-20`, `-30`, `-100`. These only **fill the box**; press Send to apply. Use the `0` preset plus Send to vent.

Underneath, **Applied** shows `/pressure/target_mbar` (what the node actually wrote, so a clamped value is visible) and **Measured** shows `/pressure/measured_mbar` (the sensor), so you can confirm the controller reached the value you asked for.

To change which presets appear, edit the `PRESSURE_PRESETS_MBAR` tuple near the top of [gui_node.py](ump_suite/gui_node.py) — the buttons and their layout are generated from it, so adding or removing entries is all that is needed.

The panel is **mouse-only** — there are deliberately no keyboard shortcuts, so keystrokes always go to the widget you are editing.

All UMP commands are absolute Sensapex targets. The bump buttons mutate the locally-held target and republish the full vector; the GUI spin boxes use the raw device range (`0` to `20000` counts).

---

## Closed-loop VLA rollouts

> ⓘ **The rollout client no longer lives in this package.** `main.py` and `sensapex_env.py` were deleted in commit `699bd1f`, along with the `sensapex_rollout` console script. The policy code now sits in the separate training / inference repo (`~/MicroVLA2/rollout`; `~/SmolVLA` for the older SmolVLA experiments).
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
| `/pressure/mbar` | `Float32` exact pressure in mbar |

So the action is **8 motion values + 1 pressure value**:

- The 8 motion values are the two 4-axis UMPs only. The ODrive focusing knob is driven separately through `/motor/target_counts` and is not part of the action vector.
- Pressure is the 9th value, in mbar, matching the `target_pressure` column the logger writes — so the policy predicts the same quantity it was trained on. Negative pulls, positive pushes, `0` vents.

This matches the dataset columns the logger writes, so state/action shapes line up between training and inference.

### Client-side responsibilities

These lived in the deleted `main.py` and are now the client's job — worth re-checking whichever repo you roll out from:

- **Workspace clamping** — per-stage min/max boxes on each of the 8 axes. ⚠️ These are tied to one physical setup; they must be set for *your* stage before any rollout.
- **Pressure clamping** — keep the predicted mbar inside what the pipette tolerates. The node clamps to the device range (±1000 on this LineUP) as a backstop, and `target_pressure` logs the post-clamp value, so the dataset stays honest either way — but the pipette does not care that the clamp saved the controller.
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
2. Use the GUI (or publish on `/ump/target`, `/ump2/target`, `/motor/target_counts` and `/pressure/mbar` directly) to drive the rig. The CSV logger records UMP state/targets, HEKA resistance and the commanded pressure, but not the ODrive motor.
   The pressure columns populate as soon as the pressure node is up, since it publishes its startup 0 mbar.
3. Click **Start Data Acquisition** — this calls `/acq/start`, which opens `logs/trial_N.csv`, creates `saved_frames/trial_N/`, and asks the camera to record `saved_videos/trial_N.mp4`.
4. Perform the trial. The logger writes one row per `log_interval_ms` (default 333 ms = 3 Hz).
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

The rollout client is **not part of this package** — run it from the policy repo (`~/MicroVLA2/rollout`, or `~/SmolVLA` for the older experiments). From this side:

1. Launch this suite, so the client has `/camera/image/compressed`, `/ump/live` and `/ump2/live` to read and `/ump/target`, `/ump2/target`, `/pressure/mbar` to write.
2. **Check the client's workspace limits, per-tick step caps and pressure range for this stage before starting.**
3. Start the policy client (and its policy server, if it uses one).

Keep the GUI up during a rollout: the pressure box and the manual UMP controls stay live, so you can intervene without stopping the client — hit the `0` preset and **Send** to vent.

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

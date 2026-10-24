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
│   ├── heka_udp_receiver_node.py # HEKA UDP bridge for voltage/current samples
│   ├── injection.py             # The injection macro (run by the UMP 1 driver)
│   ├── orient.py                # The orient skill (run by the UMP 1 driver)
│   ├── vision_node.py           # YOLO-seg segmentation of the camera -> /vision/scene
│   ├── calibrate_camera.py      # um/px, image <-> stage axes and D tilt, by moving UMP 1
│   └── vision/                  # Scene geometry, needle tip, calibration, segmenter (no ROS)
│       └── tools/               # Frame export, label conversion, SAM 2 pre-labels, train, evaluate
├── WindowsCode/
│   └── windows_send_heka_data.py # NI-DAQ sender that streams HEKA monitors over UDP
└── InstallationFiles/libum.so   # Sensapex shared library
```

### ROS topics & services

Shared names live in [ros_interfaces.py](ump_suite/ros_interfaces.py); the UMP
driver also constructs service names from its configured prefix.

| Name | Type | Direction | Notes |
|---|---|---|---|
| `/ump/live`, `/ump2/live` | `std_msgs/Int32MultiArray` | publish | Current `[x, y, z, d]` in absolute micrometres (integer ROS fields) |
| `/ump/target`, `/ump2/target` | `std_msgs/Int32MultiArray` | subscribe | Absolute Sensapex target `[x, y, z, d, speed]` |
| `/motor/live_counts` | `std_msgs/Int32` | publish | Current ODrive shadow encoder count |
| `/motor/target_counts` | `std_msgs/Int32` | subscribe | Absolute target encoder count |
| `/camera/image/compressed` | `sensor_msgs/CompressedImage` | publish | JPEG preview from PySpin grabber |
| `/camera/fps` | `std_msgs/Float32` | publish | Effective grabber FPS |
| `/camera/record_cmd` | `std_msgs/String` | subscribe | Path = start mp4 recording, `""` = stop |
| `/pressure/mbar` | `std_msgs/Float32` | subscribe | Requested mbar; negative pulls, positive pushes, `0` requests vent. Driver subscription is volatile and ignores history |
| `/pressure/target_mbar` | `std_msgs/Float32` | publish | SDK-acknowledged setpoint after clamping (latched); physical settling requires measurement |
| `/pressure/measured_mbar` | `std_msgs/Float32` | publish | Pressure measured by the controller's sensor |
| `/pressure/status` | `std_msgs/String` | publish | Latched JSON health (`ready`, `connected`, `faulted`, `message`, `target_mbar`), refreshed every second |
| `/pressure/reset_fault` | `std_srvs/Trigger` | service | Explicit reconnect at 0 mbar; never restores an old nonzero target |
| `/heka/voltage_raw_v` | `std_msgs/Float32MultiArray` | publish | HEKA voltage sample packet: `[sample_rate_hz, v0, v1, ...]` |
| `/heka/current_pa` | `std_msgs/Float32MultiArray` | publish | HEKA current sample packet: `[sample_rate_hz, i0, i1, ...]` |
| `/heka/monitor_v` | `std_msgs/Float32` | publish | Latest voltage sample from binary packets; mean monitor voltage for legacy packets |
| `/heka/monitor_step_v` | `std_msgs/Float32` | publish | Legacy monitor step voltage |
| `/heka/resistance_mohm` | `std_msgs/Float32` | publish | Live resistance estimate in MOhm |
| `/ump/stop`, `/ump2/stop` | `std_srvs/Trigger` | service | Request SDK stop; does not latch out future publishers |
| `/ump/calibrate_zero`, `/ump2/calibrate_zero` | `std_srvs/Trigger` | service | Calibrate zero at the current pose |
| `/acq/start`, `/acq/stop` | `std_srvs/Trigger` | service | Begin / end a logged trial |
| `/inject/params` | `std_msgs/String` | subscribe (UMP 1 driver) | Latched JSON `{token, seq, axis, speed_um_s, step_um, pressure_mbar, duration_ms}`, published by the GUI whenever its Injection axis or boxes change; `axis` is `X`, `Y`, `Z` or `D`, and a message without it means `X` |
| `/inject/start` | `std_srvs/Trigger` | service (UMP 1 driver) | Run one injection with the current `/inject/params`; refused with a reason if it cannot start |
| `/inject/status` | `std_msgs/String` | publish (UMP 1 driver) | Latched JSON `{count, active, stage, message, params, params_token, params_seq, injection_params, stamp}`; `count` rises once per injection that starts; `params` are the values the next trigger will use, `injection_params` the values injection number `count` actually ran with (`null` before the first) |
| `/inject/outcome` | `std_msgs/String` | subscribe (logger) | JSON `{count, outcome}` from the GUI's Record button; `outcome` is `cell`, `yolk`, `chorion_space` or `no_puncture` |
| `/orient/params` | `std_msgs/String` | subscribe (UMP 1 driver) | Latched JSON of the orient skill's settings (`OrientParams` in [orient.py](ump_suite/orient.py)) plus `token`, `seq`; published by the GUI's Orient panel; missing keys take defaults |
| `/orient/start` | `std_srvs/Trigger` | service (UMP 1 driver) | Run the orient skill once on the embryo the needle points at; refused with a reason (no calibration, injection running, stage moving, fault) |
| `/orient/status` | `std_msgs/String` | publish (UMP 1 driver) | Latched JSON `{count, active, stage, message, result, strokes, angle_deg, gain_deg_per_um, history, params, params_token, params_seq, stamp}`; `result` is `done`, `already_oriented` or `aborted` |
| `/vision/scene` | `std_msgs/String` | publish (vision node) | JSON scene per processed frame: embryos (chorion, yolk, cell, cell angle), needle tip/direction, the embryo the needle points at, distances along the needle line; `stamp` is the camera frame's own timestamp |
| `/vision/status` | `std_msgs/String` | publish (vision node) | Latched JSON `{ready, message, model, test_only, fps, inference_ms, calibrated, stamp}`, once a second |

The UMP driver publishes and accepts raw absolute Sensapex device coordinates. There is no `10000` count centering offset in the ROS topics.

---

## Nodes

### `ump_driver_node`
Connects to the UMP at the configured `device_id`, publishes the live absolute pose at `poll_ms`, and forwards `[x, y, z, d, speed]` targets directly to `stage.goto_pos`. Topic names are derived from the `topic_prefix` parameter so devices can expose `/ump/*` and `/ump2/*`.

The `/ump/stop` and `/ump2/stop` Trigger services call the SDK's actual stop API,
without depending on camera images or cached coordinates. Read/move failures stop
the affected stage and latch further targets off until the driver is restarted.
Driver shutdown also attempts a stop. Updated MicroVLA live rollout requires
these services before starting. Acknowledgment confirms the SDK call, not a
measurement that physical motion has ceased.

**Injection macro (UMP 1 only).** `/inject/start` runs the sequence in
[injection.py](ump_suite/injection.py) in a worker thread, with the values last
received on `/inject/params`:

1. move the selected `axis` by `step_um` (positive = increasing that axis) at `speed_um_s`, and wait for the SDK to report arrival within 1 µm; the other three axes are not commanded to change;
2. request `pressure_mbar` on `/pressure/mbar` and wait up to 0.5 s for `/pressure/target_mbar` to acknowledge it;
3. hold for `duration_ms`, counted from that acknowledgment;
4. vent (`0` mbar);
5. move that axis back to where it started, at the same speed.

It is refused while another injection runs, while the stage is still moving,
after a latched fault, when the axis target would leave 0–20000 µm, or (for a
nonzero pressure) unless `/pressure/status` reported ready in the last 3 s.
While it runs, ordinary `/ump/target` commands are ignored. `/ump/stop` aborts
it: the stage stops, the pressure vents if it was applied, and the needle stays
where it stopped. A missing acknowledgment vents and moves back. A move that
times out, is interrupted, or ends more than 1 µm from its target aborts too.
Limits: speed 10–2000 µm/s, |step| 1–2000 µm, pressure ±1000 mbar, time
1–10000 ms. Measured on the rig (7 Oct 2026): +10 mbar reached ~95 % about
170 ms after acknowledgment, so much shorter pulses do not reach the set value.

Any client may publish `/inject/params`; the driver uses the last message it
received. MicroVLA does this for one injection when its policy has learned the
step: it copies the GUI's values, replaces only `step_um`, waits until
`/inject/status` echoes its own token and sequence number, triggers, and then
republishes the GUI's message unchanged. The GUI itself republishes its boxes
before every Inject press, so a value another client left behind is never used
by the button. `injection_params` in the status says which step really ran.

The `ump_dual_driver_node` entry point runs both devices in one process so they share the Sensapex SDK singleton / UDP socket. This is what [launch/app.launch.py](launch/app.launch.py) uses, because separate UMP processes can conflict on the SDK socket.

### `odrive_driver_node`
Connects via `odrive.find_any()`, puts axis 0 into closed-loop velocity control, and implements a software bang-bang position controller on top: every tick it diffs the latest target against `encoder.shadow_count` and commands `±goto_speed_turns_s` until inside `deadband_counts`. The mode and zero velocity are configured while idle before entering closed-loop control. Read/write failures and partial initialization attempt both zero velocity and idle independently; faults disable further control until restart.

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
/pressure/mbar =   0.0  ->  requests 0 mbar (verify measured pressure)
```

One number, sign carries the direction. The command subscriber uses volatile
QoS: startup requests 0 mbar and waits for a fresh command. Cached commands from GUI or
rollout publishers are never restored after a driver restart. The former
`startup_grace_s` restoration mechanism has been removed.

SDK return codes are checked on initialization, range lookup, writes, reads, and
close. A failed write is not reported as applied; a failed read is not published
as a sensor measurement. Read/write failures attempt to vent and latch nonzero
commands off until an explicit reconnect/reset or driver restart. Failure to confirm a vent is logged
as an error. An unreadable or invalid device range disables the driver.

Incoming values are clamped to the range the controller reports for its channel (intersected with the `min_mbar` / `max_mbar` parameters), and non-finite values are rejected outright, both with a warning. The channel requests **0 mbar on connect** and again before `fgt_close()` on shutdown — including on Ctrl+C and on the SIGTERM `ros2 launch` sends.

Two readbacks come back out:

- **`/pressure/target_mbar`** — the value actually written to the device, published only after `fgt_set_pressure` succeeded. Because it is the post-clamp value, the dataset can never claim a pressure the controller never received.
- **`/pressure/measured_mbar`** — the controller's own sensor, polled every `poll_ms`.

Comparing the two is how you see the channel settling, or spot a request that got clamped.

The GUI displays **SDK target** separately from measured pressure, waits for a new target acknowledgement after Send, marks measurements stale after 2 seconds, and shows driver faults/offline status. Nonzero GUI requests require a fresh ready status (3-second timeout). **Reconnect / reset (0 mbar)** calls `/pressure/reset_fault`; it reconnects and requests zero without replaying the old target. Success means the SDK accepted zero, not that the measured pressure reached zero. A device protection error must still be resolved at the controller/supply level. Faulted read polling is limited to 1 Hz; nonzero commands remain disabled even if sensor readings resume.

Parameters:

| Name | Default | Notes |
|---|---|---|
| `channel` | `0` | Fluigent pressure channel index. |
| `poll_ms` | `100` | How often `/pressure/measured_mbar` is published. |
| `max_mbar` | `1000.0` | Safety ceiling, intersected with the device range. |
| `min_mbar` | `-1000.0` | Safety floor, intersected with the device range. |

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
Records the latest received observations and targets with timing diagnostics:
1. Subscribes to **live** topics (UMP1, UMP2) and to **target** topics published by the GUI / policy.
2. Subscribes to `/heka/resistance_mohm` so each row can include the latest finite HEKA resistance value when available, and to `/pressure/target_mbar` and `/pressure/measured_mbar` for applied setpoints and sensor values.
3. On `/acq/start`, picks the next free `trial_N` ID by inspecting **`logs/`, `saved_frames/` and `saved_videos/` together**, atomically reserves `saved_frames/trial_N/`, exclusively creates `logs/trial_N.csv`, and tells the camera to record `saved_videos/trial_N.mp4`. Scanning all three matters: deleting a CSV while its frame directory survives would otherwise hand the number back out and the new run would overwrite the old frames.
4. Every `log_interval_ms` it saves the latest JPEG to `saved_frames/trial_N/frame_NNNNNN.png` and appends one CSV row with the live pose, the most-recent commanded target, the saved image's path, the latest resistance when available, and the timing columns below.
5. On `/acq/stop` it closes the file, sends an empty record command to the camera, and reports the logging rate it actually achieved.

The latest UMP target is **not cleared** between ticks. Validity fields distinguish
missing commands from intentional zero targets. Receipt does not establish driver
acceptance or freshness; stale values can persist until new feedback arrives.

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
image_age_s,
target_valid, target_valid2, state_valid, state_valid2,
Injection, injection_axis, injection_step_um,
Orient, orient_result, injection_outcome
```

The current CSV has **35 columns**. `Injection` was added on 7 October 2026,
`injection_axis` / `injection_step_um` on 22 October 2026, and `Orient`,
`orient_result` and `injection_outcome` on 24 October 2026 (branch
`orientation-skill`); older trials have 29, 30 or 32 and simply lack them.

**`Orient`** is `1` on the first row written after the orient skill starts.
**`orient_result`** (`done`, `already_oriented` or `aborted`) is written on the
first row after it ends. As for an injection, the position, target, pressure
command and state-stamp columns hold their pre-orient values while it runs, but
unlike an injection the skill leaves the needle somewhere new, so the result row
shows a jump in position. The GUI then re-syncs UMP 1's targets to the live
position (see `gui_node`), so the next logged target is the new position.
**`injection_outcome`** is the operator's verdict on the last injection,
written on the next row after it was recorded with the GUI's Record button.

**`Injection`** is `1` on the first row written after an injection starts and `0`
on every other row. On that row only, **`injection_axis`** (`X`, `Y`, `Z` or `D`)
and **`injection_step_um`** (signed, whole µm) record the axis and step the
injection ran with, taken from the driver's `injection_params`; they are blank
on every other row, and blank on the flagged row if the driver did not report
them. Speed, pressure and time are not logged. While the
injection runs, the position, target, `target_pressure` and `state_stamp`
columns keep the values they had just before it started, so the dataset shows
one command rather than the moves and pressure pulse inside it. UMP 1 targets
sent during an injection are not logged, because the driver ignores them.
Images, resistance and `measured_pressure` stay live.

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

- **`target_pressure`** — from `/pressure/target_mbar`: the setpoint acknowledged by the SDK, not a measurement of achieved pressure. This is the action label to train on.
- **`measured_pressure`** — from `/pressure/measured_mbar`: the controller's sensor reading. Expect it to lag `target_pressure` by a poll tick or two while the channel settles, and to sit slightly off the target.

Each pressure column stays blank until valid data arrives. The target also requires
a ready `/pressure/status` received within 3 seconds; the measured value requires
a successful reading within 2 seconds. Startup attempts a zero setpoint;
connection or SDK faults can prevent publication. Resistance still does not expire.

**Logging rate.** `log_interval_ms` defaults to **333 ms (approximately 3 Hz)** standalone and in the launch
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

Underneath, **SDK target** shows `/pressure/target_mbar` and **Measured** shows `/pressure/measured_mbar`, with stale readings marked. Faults/offline status appear separately. The reconnect/reset button attempts recovery at zero; physical settling must be checked in the measured value.

To change which presets appear, edit the `PRESSURE_PRESETS_MBAR` tuple near the top of [gui_node.py](ump_suite/gui_node.py) — the buttons and their layout are generated from it, so adding or removing entries is all that is needed.

The **Injection** panel, between UMP 2 and the ODrive, has a row of **X Y Z D**
axis buttons (the selected one is highlighted blue; exactly one is always
selected) and an **Inject** button with four boxes beside it: **Speed** (µm/s),
**Step** (µm; `-` moves the axis down first), **Pressure** (mbar; `-` pulls) and
**Time** (ms). The injection runs on the selected axis. The boxes publish
`/inject/params` whenever they change, and Inject calls `/inject/start` once the
driver has echoed those values back. **Stop** calls `/ump/stop`. The line
underneath shows the driver's progress and timings. The axis and boxes are saved
in `~/.config/ump_suite/gui.ini` and restored on the next start.

Below the Injection boxes, **Went into** (cell / yolk / chorion space / no
puncture) and **Record** publish `/inject/outcome` for the latest injection, so
each trial records whether the injection reached the cell.

The **Orient** panel, under Injection, has the orient skill's settings
(Tolerance, Strokes, Stroke, Dent, Standoff, Speed; saved in `gui.ini` under
`orient/`), a **Dry run** box (the skill plans and reports its moves but sends
none), **Orient** and **Stop**. A line shows the vision node's state (model
ready or not, test-only, fps, calibrated or not) and the skill's progress. When
an orient run ends, the panel syncs UMP 1's target boxes to the live position and
sends it: the skill moved the stage through the SDK, not `/ump/target`, so the
boxes would otherwise still hold the old target and the next nudge would drive
the needle back there.

The camera view draws the vision scene when **Show vision** is ticked: chorion
circles (the embryo the needle points at thicker), yolk circle, cell centroid
and its angle relative to the needle, the needle line, the point where it meets
the cell and the distance to it, and a `TEST-ONLY MODEL` banner for a model
trained on unchecked labels.

The panel is **mouse-only** — there are deliberately no keyboard shortcuts, so keystrokes always go to the widget you are editing.

All UMP commands are absolute Sensapex targets. The bump buttons mutate the locally-held target and republish the full vector; the GUI spin boxes use the raw device range (`0` to `20000` micrometres); this UI range is not calibrated workspace protection.

---

## Embryo vision and the orient skill (branch `orientation-skill`, 24 October 2026)

### `vision_node`
Runs a YOLO-seg model on the newest camera frame (frames that arrive while it
runs are skipped; at most `max_hz`, default 10) and publishes a **scene** on
`/vision/scene`. The model segments four classes, **chorion**, **yolk**,
**cell** and **needle**; [vision/geometry.py](ump_suite/vision/geometry.py)
turns the masks into:

- every embryo (one per chorion, sorted top to bottom): the chorion circle,
  the yolk circle, the cell's centroid, the cell's **screen angle** around the
  yolk, whether the cell sits over the yolk's middle (on top or underneath) and
  whether the chorion is fully in view;
- the needle: a line fitted to its mask, with the tip extended along that line
  while the image stays darker than its sides (the tip is a few micrometres
  wide and the mask stops short of it);
- `target`: the embryo whose chorion the needle line meets first;
- `ray`: along the needle line from the tip, the distance to that chorion, to
  the cell's near edge and to its far edge, and whether the line meets the cell
  at all (micrometre copies when a camera calibration exists).

`cell_angle_deg` is measured from the side the needle comes from: `0` means
the cell faces the needle, positive means the embryo must turn clockwise on
screen. Angles follow the screen: 0° is 3 o'clock, 90° is 12 o'clock.

Parameters: `model_path` (default
`~/.local/share/ump_suite/models/embryo_seg.pt`), `conf`, `imgsz`, `max_hz`,
`device`, `calibration_path`. Without a model file the node stays up, publishes
no scenes and says why on `/vision/status`. A JSON note beside the weights
(`embryo_seg.json`, written by the trainer) says what the model was trained
on; `test_only: true` marks one trained on unchecked labels, and the GUI and
every scene say so.

ultralytics is installed without its dependencies (`pip install --user
--no-deps ultralytics==8.4.175` plus `ultralytics-thop`, `nvidia-ml-py`,
`polars`), because it would otherwise pull a second OpenCV over the headless
one. The package sets `YOLO_AUTOINSTALL=false` and `YOLO_OFFLINE=true` before
importing it, so it never pip-installs at run time.

### The orient skill (`orient.py`, run by the UMP 1 driver)
`/orient/start` turns the embryo the needle points at until its cell faces the
needle, with a step-and-look loop on the cell angle φ:

1. if |φ| is within the tolerance on `confirm_frames` fresh scenes, stop
   without moving (`already_oriented`);
2. move sideways so the needle line passes through the chorion's centre;
3. advance until the tip is `dent_um` inside the chorion's outline;
4. stroke sideways along the rim by `u = clip(loop_gain · (−φ) / ĝ, ±stroke_max_um)`
   (at least `stroke_min_um`); a stroke "up" on screen at the needle-side rim
   turns the embryo counterclockwise;
5. back off to `backoff_um` outside the chorion and look again;
6. update this embryo's gain ĝ (degrees per µm of stroke) from what the
   stroke did; if it did nothing, deepen the dent by `dent_step_um` up to
   `dent_max_um`;
7. when |φ| is within tolerance on `confirm_frames` scenes, re-centre
   sideways and stop at the standard standoff.

The starting gain is 1 / (2R) rad/µm (R the chorion radius), the value for a
chorion rolling against a trough wall; the loop then learns the real one. Only
X and Y move; Z and D never change. Pixel moves become X/Y moves through the
camera calibration.

It aborts (backing off first if the needle is in the chorion) when: there is no
calibration, no fresh scene, no needle, the needle points at no embryo, the
embryo has no cell, the cell is over the yolk's middle (it would need a press
from above, which this skill does not do), the yolk is leaving the view, the
needle is more than `approach_max_um` from the chorion or already inside it,
the tip ends more than 30 px from where the moves should have put it (needle
bent or slipped), two strokes in a row turn the embryo the wrong way (check the
calibration's Y), two strokes do nothing at the deepest dent, `max_strokes` is
reached, or `timeout_s` passes. `/ump/stop` aborts it at once, without backing
off. While it runs the driver ignores `/ump/target` and refuses `/inject/start`;
an injection running refuses `/orient/start`.

Rolling the chorion against a wall moves the embryo along by about half of each
stroke, so a long turn can carry the yolk out of view; the skill then stops.
Whether rolling the chorion turns the embryo *inside* it, and which method
(needle against the trough wall, a second tool on the far side, or a deeper
dent onto the yolk) works best, has to be found by hand on the rig first. The
skill implements the single-needle methods; a two-tool version is not written.

Defaults (`OrientParams`): tolerance 15°, confirm 3 frames, 25 strokes, stroke
15–250 µm, dent 40 µm (max 120, step 20), standoff 60 µm, approach at most
800 µm, loop gain 0.6, gain blend 0.5, 200 µm/s, settle 0.4 s, timeout 300 s.

### Camera calibration (`calibrate_camera`)
The vision step and the orient skill need micrometres per pixel, which way each
stage axis moves the tip on screen, and the D axis's tilt. With the needle tip
in view, in clear liquid with nothing within a few hundred micrometres:

```bash
ros2 run ump_suite calibrate_camera --confirm --save-frames ~/calibration_frames
```

It pulls X back by `--background-um` (1500) so the needle leaves the view and
takes a background frame, returns, then moves X and Y by ±`--step` (100 µm) and
D back by `step` and `2·step` (D only ever retracts), returning to the start
after each move. The tip is found by differencing each frame against the
background (no model needed); `--tip-source vision` uses `/vision/scene`
instead. It fits µm/px (X and Y must agree within 10%), the 2×2 image↔X/Y
mapping and the D tilt θ (|vD|·µm/px = cos θ), prints the residuals and the
return drift, and saves `~/.config/ump_suite/vision_calibration.json`
(`--dry-run` only prints). The needle always ends where it started, also
after an error or Ctrl-C. It refuses to move without `--confirm`.

### Training the segmentation model
Label by hand; that is what a model you trust is trained on.

1. **Export frames**, grouped by trial so validation can hold whole trials
   (embryos) out:
   ```bash
   python -m ump_suite.vision.tools.export_frames --trials ~/Backup/New\ 3 \
       --trials ~ --every 4 --out ~/embryo_labels
   ```
   Every 4th frame plus every frame an injection or orient was decided on goes
   to `images/`, listed with its trial in `manifest.csv`.
2. **Label** in any polygon tool (CVAT, Label Studio, Roboflow, labelme, µSAM):
   one polygon per object, classes `chorion`, `yolk`, `cell`, `needle`; one
   chorion, yolk and cell per embryo, every embryo in the frame (an unlabelled
   embryo teaches the model that embryos are background). Tools that export
   YOLO segmentation write straight into `labels/`; for COCO or labelme:
   ```bash
   python -m ump_suite.vision.tools.labels_to_yolo --coco export.json \
       --images ~/embryo_labels/images --out ~/embryo_labels/labels
   ```
   It reports objects per class and frames without labels.
3. **Optional head start**: `bootstrap_labels.py --sam sam2.1_b.pt` writes
   SAM 2 pre-annotations (seeded by the dark yolk) to a separate folder for
   correcting in your tool. They are wrong often enough that a model trained
   on them alone must be treated as test-only.
4. **Train** (YOLO11-seg from pretrained weights, whole trials held out):
   ```bash
   python -m ump_suite.vision.tools.train_segmenter --data ~/embryo_labels \
       --epochs 80 --install
   ```
   `--install` copies the best weights to the vision node's path with a note
   beside them; add `--test-only` for unchecked labels.
5. **Evaluate** on the held-out trials, in the units the skills use (IoU per
   class, chorion centre, cell angle, tip and needle-to-cell distance errors):
   ```bash
   python -m ump_suite.vision.tools.eval_segmenter --data ~/embryo_labels \
       --groups-from ~/embryo_seg_runs/<run>/model_note.json --um-per-px 1.8
   ```

**State on 24 October 2026.** The installed model is a **test-only** YOLO11s-seg
trained on SAM 2 pre-annotations of 654 frames from 22 trials (40 epochs).
Against those same unchecked labels on 4 held-out trials: IoU chorion 0.92,
yolk 0.90, cell 0.49, needle 0.46; cell angle error median 6°, mean 21°; tip
error median 7 px but 90th percentile 387 px. It runs at about 18 ms per frame
on the RTX 3080 Ti. Good enough to exercise the pipeline, not to orient or
inject. No camera calibration exists yet, and the orient skill, the
calibration and the measured step have not yet been run on a needle in view on
the rig; on 24 October the needle was out of the field (UMP 1 at Z 200) and was
not moved blind. Checked live that day without moving the needle: the vision
node at 10 fps on real frames, `/orient/start` refused without a calibration,
a dry-run orient through the real driver aborting on "no needle in view", the
35-column log with an outcome, the GUI panels, and a MicroVLA dry run.

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
| `/ump/target`, `/ump2/target` | `[x, y, z, d, speed]` absolute micrometres plus speed |
| `/pressure/mbar` | `Float32` exact pressure in mbar |

A dual-arm client controls **8 motion coordinates and a separate pressure channel**:

- The 8 motion values are the two 4-axis UMPs only. The ODrive focusing knob is driven separately through `/motor/target_counts` and is not part of the action vector.
- Pressure has a separate scalar head/chunk and normalization in MicroVLA; it is
  not appended to the Sensapex action vector. Its mbar target matches the `target_pressure` column the logger writes — so the policy predicts the same quantity it was trained on. Negative pulls, positive pushes, `0` vents.

This matches the dataset columns the logger writes, so state/action shapes line up between training and inference.

### Client-side responsibilities

These lived in the deleted `main.py` and are now the client's job — worth re-checking whichever repo you roll out from:

- **Workspace clamping** — per-stage min/max boxes on each of the 8 axes. ⚠️ These are tied to one physical setup; they must be set for *your* stage before any rollout.
- **Pressure clamping** — keep the predicted mbar inside what the pipette tolerates. The node reads device limits and intersects them with its configured envelope.
  `target_pressure` records the SDK-acknowledged setpoint. Device range is not a
  substitute for a calibrated pipette envelope.
- **Per-tick step limiting** — cap the delta on each axis so a bad prediction cannot command a large jump.
- **Control rate** — inspect actual timestamps and match training/inference timing.
  The requested logger interval is 333 ms; irregular cadence is not repaired by
  merely relabeling frames with an average FPS.
- **Stop** — stop client publication and call the UMP SDK stop services. Physical
  stop/vent behavior and process-death protection still require validation.
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
   Pressure targets populate only while the driver reports ready, and measured pressure requires fresh successful readings; faults leave the relevant cells blank.
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

Manual and policy publishers can compete; the package has no exclusive command
owner or takeover lease. Stop the policy before manual intervention. The `0`
preset and **Send** request a vent, but a competing publisher can overwrite it
and physical pressure must be checked independently.

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
| `vision_node` | `ump_suite.vision_node:main` |
| `calibrate_camera` | `ump_suite.calibrate_camera:main` |

---

## Maintainer

Raian Haider Chowdhury — `chowd207@umn.edu`

### Log validity and manual feedback

New CSVs append `target_valid`, `target_valid2`, `state_valid`, and `state_valid2`.
A valid zero target means Home; an absent target is recorded as invalid. Each arm
is handled independently during conversion. Frames whose PNG write fails have
no reported image path, so conversion detects the missing frame. The standalone
logger defaults to 333 ms, matching the launch's requested rate; actual achieved
rate still depends on acquisition and disk throughput.

The manual GUI waits for received, fresh feedback, including an all-zero pose,
before issuing UMP or motor commands. Coordinate zeroing invalidates the cached
UMP target until feedback from the new coordinate frame arrives. The camera's
recording lock follows manual exposure precedence, recording writes are serialized,
and shutdown joins the capture worker before releasing SDK buffers.


## Session ownership and recording integrity (September 2026)

Run one local rig session per user. `runtime_guard.py` holds process-lifetime
filesystem locks for the suite and each hardware/logger/GUI entry point. A second
launch fails before it starts hardware processes. Individual duplicate nodes are
also rejected. Closing the control GUI shuts down the entire launch and waits for
its children; it no longer leaves invisible loggers and hardware drivers behind.
Locks release automatically when the owning process exits. Do not delete lock
files while processes are running; file existence alone does not mean a lock is
held.

The logger reserves each trial with an atomic frame-directory creation and opens
its CSV exclusively (`x` mode). Concurrent reservations cannot share a trial,
and an existing CSV cannot be truncated. Every row has the same named columns
as the header (32 since the injection columns), is flushed after writing, and a write failure stops acquisition.
CSV cleanup still runs if ROS has already shut down.

`target_pressure` is blank while pressure-driver status is faulted, absent, or
older than 3 seconds. `measured_pressure` is blank once the last valid read is
older than 2 seconds. Zero remains a real measurement/command, not a missing
value. A target is the last SDK-acknowledged request, not proof of physical
settling or of a front-panel command. Direct front-panel changes have no SDK
setpoint acknowledgement and cannot be reconstructed as pressure action labels.

On September 18, two old complete launches were found writing the same trial
files concurrently. Existing `~/logs/trial_28.csv`, `trial_29.csv`, and
`trial_30.csv` contain overwritten/merged rows; `trial_25.csv` also has structural
corruption. Those originals were preserved, not repaired by guessing. Do not use
corrupt trials for training. The fixes prevent future collisions and do not
recover overwritten measurements or frames.

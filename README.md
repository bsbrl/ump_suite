# ump_suite

A ROS2 (Humble, `ament_python`) package for collecting datasets and running closed-loop VLA policies on a rig built around **two Sensapex UMP micromanipulators**, an **ODrive**-driven focusing knob and a **FLIR Blackfly S BFS-U3-200S6C** camera (20 MP colour, USB3, installed 27 October 2026 in place of the 1.6 MP BFS-U3-16S2C; see [`camera_node`](#camera_node)).

The package wraps every device behind a small ROS2 node, ships a Qt GUI for manual teleop, and a logger that writes synchronized image / video / CSV trials. It also runs the rig's fixed skills: the **injection macro**, the **embryo vision** (segmentation of chorion, yolk, cell and needle) and the **orient skill** that turns an embryo until its cell faces the needle. It is the **robot side** only: the policy client (MicroVLA, in `~/MicroVLA2`) drives this rig over the ROS topics below.

---

## Hardware

| Device | Driver / SDK | Node |
|---|---|---|
| Sensapex UMP (×2) | `sensapex` Python SDK + `libum.so` | [ump_driver_node.py](ump_suite/ump_driver_node.py) |
| ODrive single-axis motor (focusing knob) | `odrive` Python SDK | [odrive_driver_node.py](ump_suite/odrive_driver_node.py) |
| FLIR Blackfly S BFS-U3-200S6C camera (20 MP colour, USB3) | PySpin (Spinnaker 4.2) | [camera_node.py](ump_suite/camera_node.py) |
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
Initializes the first PySpin camera with `NewestOnly` stream buffering so the policy / GUI always see the freshest frame. It takes either `BGR8` from the camera's image processor or, with `raw_bayer` (the launch setting for the 20 MP camera), the raw Bayer mosaic, which it converts to colour itself (see [Frame rate: raw Bayer mode](#frame-rate-raw-bayer-mode)). A worker thread:
- publishes a JPEG preview (`jpeg_quality`) at `publish_hz` on `/camera/image/compressed`
- publishes the actual grab rate on `/camera/fps`
- writes every captured frame to an mp4 when recording is active, declared at the
  camera's own frame rate (`record_fps` 0) so it plays at real speed

Recording is toggled by sending a path on `/camera/record_cmd` (empty string to stop).

#### Resolution and USB memory (20 MP camera)

The BFS-U3-200S6C's full frame is 5472 × 3648. At that size every consumer falls
behind (measured 27 October 2026: the logger reached 1.6 Hz instead of 3 Hz with
15 MB PNGs, saved frames went 3 s stale while recording, and the GUI used five
CPU cores). The launch file therefore sets **`binning:=2`**: the camera combines
2 × 2 pixels, keeping the whole field of view at **2736 × 1824** (cropped further
below). Measured on 27 October with the camera's image processor doing the
binning (`binning_mode` `Sum`, so each pixel gets 4× the light and the exposure
dropped from about 13.6 ms to 3.8 ms): 6.1 fps, JPEG 0.3 MB, the logger at 3.00
Hz with 3.9 MB PNGs and frames at most 0.33 s old, the video at real speed, the
GUI at about 28% CPU. The camera now runs with that processor off and does the
same binning on the raw data, at 25 fps (see
[Frame rate: raw Bayer mode](#frame-rate-raw-bayer-mode)). `binning` is applied
on every start; changing it changes µm per pixel, so recalibrate the camera
(`calibrate_camera`) afterwards. Use `binning:=1` for a small sensor.

USB3 cameras this size also need more USB buffer memory than Linux's default
16 MB. With the default, Spinnaker fails at `Init()` with **-1016 "Not enough
available memory to allocate buffers for streaming"**, the camera node exits and
the GUI's camera view stays empty. FLIR recommends 1000 MB:

```bash
sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'   # until the next reboot
```

To make it permanent, add a udev rule that sets it whenever a FLIR camera
appears (including at boot):

```bash
echo 'ACTION=="add", SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="1e10", RUN+="/bin/sh -c '"'"'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'"'"'"' \
  | sudo tee /etc/udev/rules.d/41-flir-usbfs-memory.rules
sudo udevadm control --reload-rules
```

(or add `usbcore.usbfs_memory_mb=1000` to `GRUB_CMDLINE_LINUX_DEFAULT` in
`/etc/default/grub` and run `sudo update-grub`, FLIR's documented way). Check with
`cat /sys/module/usbcore/parameters/usbfs_memory_mb`.

**The crop.** The new sensor is about 2.6 times as wide as the old one and sees
out to the edge of the microscope's circular field. On 27 October that circle sat
off centre, about 12% of the frame width to the left and 6% up (radius about 1616
binned pixels with the 4× objective), so the right of the frame went dark: from
about 85% of the width in the corners, black at the edge. The 2× objective
showed less of it. The launch file therefore crops in the camera, in binned
pixels, to the part that stays lit:

```
roi_width:=2304  roi_height:=1728  roi_offset_x:=0  roi_offset_y:=0
```

Its darkest corner is about 70% of the centre's brightness with the 4× objective
(1% without the crop). The crop is 4:3 on purpose: MicroVLA's image input is
320 × 240, the dataset converter letterboxes frames into it and the rollout
stretches them, and the two only agree for 4:3 frames. It also keeps the exposure
calibration, which measures the whole delivered frame, off the dark area. It does
not change the frame rate (with the image processor on, 6.1 fps at every crop
measured from 1368 to 2736 px wide; with it off, 32.2 fps binned with or without
the crop), but every frame is 20% smaller: PNGs 3.1 MB instead of 3.9 MB. The
camera calibration is unaffected
(it uses how far the tip moves in pixels, not where the frame starts); changing
the crop's size or position does not need a recalibration, changing `binning`
does. The node logs the result at startup, e.g. `Image 2304x1728 at (0, 0) of
2736x1824 (binning 2x2, Average)`, and warns if the camera had to adjust it
(sizes come in steps of 4 px across and 2 px down).

To use more of the sensor, centre the light circle on it. Close the field
diaphragm: if its edge appears off centre, centre it with the condenser's
centring screws and open it until it just clears the frame. If the circle does
not respond to the diaphragm, the camera adapter is off centre. Once centred,
the full-height 4:3 crop (`roi_width:=2432 roi_height:=1824`, offsets `-1` for
centred) fits inside the circle; the whole 3:2 frame would still have dim
corners, as the circle only just reaches them.

#### Frame rate: raw Bayer mode

With its image processor (ISP) on, which every colour pixel format needs, the
BFS-U3-200S6C runs at **6 fps in every mode**. FLIR's frame-rate table lists 6
fps for every ISP-on row, and here binning, cropping, the ADC bit depth and the
USB limit made no difference (measured 28 October). At 6 fps a frame can already
be 165 ms old, and MicroVLA's rollout, which refuses an observation whose frame
and positions arrived more than 0.25 s apart, failed that check about once a
minute.

The launch file therefore sets **`raw_bayer:=true`**. The camera sends its 8-bit
Bayer mosaic (`BayerRG8`) with the ISP off, still binned 2 × 2 and cropped in the
camera, and the node converts each frame:

1. demosaic with OpenCV (`COLOR_BayerBG2BGR`, which is OpenCV's name for the
   camera's RGGB layout);
2. the ISP's own colour matrix, read from the camera at startup.

White balance and exposure stay in the camera; both work with the ISP off.
Against the ISP's output of the same view at the same exposure, the result is
within 3% in brightness and within 4 grey levels per pixel on average, and it is
no noisier (frame-to-frame noise 3.4 / 2.5 / 2.2 grey levels in B / G / R
against 4.8 / 2.5 / 3.2). The ISP applied no gamma on top of the matrix, so the
node does not either. The two steps take about 3 ms per frame.

| Parameter | Launch value | Why |
|---|---|---|
| `raw_bayer` | `true` | ISP off; the camera then reaches 32 fps binned. `false` for the old camera, whose ISP never limited it. |
| `link_limit_mb_s` | `380` | USB bandwidth the camera may use. It powers up at 60 MB/s, which caps raw frames at 3–4 fps; 380 is FLIR's reference value. |
| `frame_rate` | `25` | Cap. It must stay below `publish_hz` (30): the preview throttle skips a frame that comes sooner than 1/`publish_hz` after the last one, so at 32 fps every other frame would go. 25 fps also costs a quarter less CPU. |
| `gain_db` | `12` | With the ISP off the camera averages binned pixels instead of summing them; 12 dB (× 4) restores the brightness at the same exposure. |

Measured on 28 October with the whole rig running:

- 25.0 fps steady (frame gaps 36–43 ms), including while recording;
- MicroVLA's own observation code, polled at 3 Hz for a minute, passed 180 of
  180 checks: frame and positions at most 51 ms apart, frames at most 52 ms old;
- the logger at 3.01 Hz with saved frames at most 0.19 s old, videos at 25 fps
  and real speed;
- mean grey 200.2, a neutral white field, no clipped pixels;
- CPU: the camera node about 70% of one core (about 105% while recording video),
  the GUI about 40%.

OpenCV is limited to two threads in the camera and GUI nodes. By default it
starts a worker per core (32 on this PC) for every small call and they spin,
which cost about a core per process at 25 fps.

On 28 October the field needed 10.6 ms of exposure for mean grey 200, against
4.9 ms the day before; the ISP mode needed the same 10.3 ms that day, so the light
was dimmer, not the mode. Raise the illumination if the moving needle blurs.

Unbinned raw frames would run at 16.6 fps (23.4 fps cropped to 4608 × 3456) but
send 16–20 MB per frame over USB and leave the binning to the PC.

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
| `gain_db` | `0.0` | Prefer exposure over gain; gain amplifies noise. The launch file sets `12` with `raw_bayer` (see [Frame rate](#frame-rate-raw-bayer-mode)). |
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
one. The frame rate is a property of the camera, not `publish_hz`: the old 1.6 MP
camera's full-resolution BGR8 was capped near 12.9 fps by
`DeviceLinkThroughputLimit` (60 MB/s); the 20 MP camera runs at 25 fps in raw
mode (see [Frame rate](#frame-rate-raw-bayer-mode)).

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
3. On `/acq/start`, picks the next free `trial_N` ID (all paths are relative to the folder the rig was launched from) by inspecting **`logs/`, `saved_frames/` and `saved_videos/` together**, atomically reserves `saved_frames/trial_N/`, exclusively creates `logs/trial_N.csv`, and tells the camera to record `saved_videos/trial_N.mp4`. Scanning all three matters: deleting a CSV while its frame directory survives would otherwise hand the number back out and the new run would overwrite the old frames.
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
`orient_result` and `injection_outcome` on 24 October 2026; older trials have
29, 30 or 32 and simply lack them.

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

The camera view draws the vision scene when **Show vision** is ticked (the
default): chorion circles (the embryo the needle points at thicker), yolk
circle, cell centroid and its angle relative to the needle (`(on top/under)`
when the cell sits over the yolk's middle), the needle line, the point where it
meets the cell and the distance to it (µm with a calibration, px without), and a
`TEST-ONLY MODEL` banner for a model trained on unchecked labels. It draws on a
copy of the displayed frame only: recordings, the skills and MicroVLA are
unaffected, and it shows nothing while no model is installed or the newest scene
is more than 1 s old.

The panel is **mouse-only** — there are deliberately no keyboard shortcuts, so keystrokes always go to the widget you are editing.

All UMP commands are absolute Sensapex targets. The bump buttons mutate the locally-held target and republish the full vector; the GUI spin boxes use the raw device range (`0` to `20000` micrometres); this UI range is not calibrated workspace protection.

---

## Embryo vision and the orient skill (added 24–25 October 2026)

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
  while the image stays darker than the pixels just beyond the needle's own
  width on either side (the tip is a few micrometres wide and the mask stops
  short of it; the width is measured from the mask, so this adapts to the
  camera's scale);
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
6. if the stroke turned the embryo at least 2° the way it was sent, update
   this embryo's gain ĝ (degrees per µm of stroke):
   ĝ ← (1 − `gain_blend`)·ĝ + `gain_blend`·(turned / stroke), the measured
   value clamped to ĝ/10…10ĝ; if it turned less than 2°, deepen the dent by
   `dent_step_um` up to `dent_max_um` instead; a turn the wrong way is counted
   but not learned from;
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
the tip ends more than 55 µm from where the moves should have put it (needle
bent or slipped), two strokes in a row turn the embryo the wrong way (check the
calibration's Y), two strokes in a row do nothing and the dent cannot go
deeper, `max_strokes` is reached, or `timeout_s` passes. `/ump/stop` aborts it at once, without backing
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

It pulls X back in `--background-step-um` steps (500 µm) until the whole needle
has left the view: a step that changes almost nothing on screen, and a frame
that differs from the starting one by a needle-shaped region reaching the frame
edge. That frame is the background, so it works for any field of view; it gives
up after `--max-background-um` (6000) or if X would leave the stage range. It
then returns, moves X and Y by ±`--step` (100 µm) and
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
   (embryos) out. `--trials` is a folder holding `logs/` and `saved_frames/`
   (the logger writes both into the folder the rig was launched from) and can
   be repeated:
   ```bash
   python3 -m ump_suite.vision.tools.export_frames --trials ~/newcam \
       --every 30 --out ~/embryo_labels
   ```
   Every Nth frame (`--every`, default 4) plus every frame an injection or
   orient was decided on goes to `images/`, listed with its trial in
   `manifest.csv`. Frames of one trial at 3 Hz barely differ, so for labelling
   a large N (30 = one frame per 10 s) gives more variety for the effort.
2. **Label** in any polygon tool (CVAT, Label Studio, Roboflow, labelme, µSAM):
   one polygon per object, classes `chorion`, `yolk`, `cell`, `needle`; one
   chorion, yolk and cell per embryo, every embryo in the frame, including
   ones cut by the frame edge (an unlabelled embryo teaches the model that
   embryos are background). All embryos get the same class names; which one
   is "topmost" is worked out later from the positions. Outlines overlap: the
   chorion is the whole circle, with the yolk, cell and any needle crossing it
   drawn on top. Draw polygons, not brush masks. Export **COCO 1.0** (CVAT) and
   convert, which numbers the classes by name:
   ```bash
   python3 -m ump_suite.vision.tools.labels_to_yolo --coco export.json \
       --images ~/embryo_labels/images --out ~/embryo_labels/labels
   ```
   It reports objects per class; frames you have not labelled yet get no
   label file, so the trainer leaves them out.

   What to label: start with about 150–200 frames from 20 or more trials
   (different embryos), then train, evaluate and label more of what the model
   gets wrong. Have the needle in about three quarters of the frames, in every
   state (far, approaching, touching or denting the chorion, inside the embryo,
   slightly out of focus), and vary the number of embryos (1–4), their layout,
   the cells' clock positions, focus, bubbles and debris. Keep the camera
   settings the ones used for real.
3. **Optional head start**: `python3 -m ump_suite.vision.tools.bootstrap_labels
   --frames-dir ~/embryo_labels/images --out-dir ~/embryo_labels/labels_bootstrap
   --sam ~/.local/share/ump_suite/models/sam2.1_b.pt` writes SAM 2
   pre-annotations (seeded by the dark yolk) as YOLO labels in a separate
   folder, for correcting in your tool. They are wrong often enough that a
   model trained on them alone must be treated as test-only, and its
   thresholds were tuned to the old camera.
4. **Train** (YOLO11-seg from pretrained weights, whole trials held out):
   ```bash
   python3 -m ump_suite.vision.tools.train_segmenter --data ~/embryo_labels \
       --epochs 80 --install
   ```
   Runs go to `~/embryo_seg_runs/`. `--install` copies the best weights to the
   vision node's path with a note beside them (the vision node loads it on its
   next start); add `--test-only` for unchecked labels. Training mirrors frames
   top to bottom and tilts them up to 10° on the fly, labels included, so the
   needle still enters from the right and cells appear at the mirrored clock
   positions; there is no need to rotate or relabel frames by hand.
5. **Evaluate** on the held-out trials, in the units the skills use (IoU per
   class, chorion centre, cell angle, tip and needle-to-cell distance errors):
   ```bash
   python3 -m ump_suite.vision.tools.eval_segmenter --data ~/embryo_labels \
       --groups-from ~/embryo_seg_runs/<run>/model_note.json --um-per-px <from calibration>
   ```

**State on 27 October 2026: no model is installed.** The frames recorded before
25 October are obsolete: the camera has a larger sensor now (installed 27
October), which shows more embryos per view and makes each embryo and the
needle smaller on screen. The test-only
model trained on the old frames, the exported frames and its training runs
were deleted on 25 October; the vision node reports "no model" until a model is
trained on the new camera's frames. No camera calibration exists yet either.
The generic starting weights (`yolo11s-seg.pt`, `yolo11n-seg.pt`, `sam2.1_b.pt`)
are kept in `~/.local/share/ump_suite/models/`.

For the record, on the old camera (24 October): a test-only YOLO11s-seg trained on
SAM 2 pre-annotations of 654 frames reached, against those same unchecked labels
on 4 held-out trials, IoU chorion 0.92, yolk 0.90, cell 0.49, needle 0.46, and
ran at about 18 ms per frame on the RTX 3080 Ti. Checked live that day without
moving the needle (it was out of the field, UMP 1 at Z 200): the vision node at
10 fps, `/orient/start` refused without a calibration, a dry-run orient through
the real driver aborting on "no needle in view", the 35-column log with an
outcome, the GUI panels, and a MicroVLA dry run. The orient skill on an
embryo, the calibration and the measured step have not been run on the rig.

**For the new camera:**
- Launch the rig from a new folder so the new trials do not mix with the
  obsolete ones in `~/logs`; the logger numbers trials per folder. The desktop
  **My Robot** icon starts the rig in the home folder, so start it from a
  terminal instead: `mkdir -p ~/newcam && cd ~/newcam && ~/start_robot.sh`. Use
  that folder for `export_frames --trials` and MicroVLA's `--data-root`.
- The BFS-U3-200S6C runs through `camera_node` as raw Bayer data, binned 2 × 2
  and cropped to the lit 2304 × 1728, at 25 fps (see
  [Frame rate: raw Bayer mode](#frame-rate-raw-bayer-mode)), and the logger
  keeps 3 Hz with it. The machine needs the larger USB buffer memory described in
  [Resolution and USB memory](#resolution-and-usb-memory-20-mp-camera); until
  the udev rule there is installed, it falls back to 16 MB at every reboot and
  the camera view stays empty.
- The segmentation input size (`imgsz`, 1024 in the launch file and the trainer)
  may need raising, or tiling, for smaller embryos and a thinner needle; decide
  from the first frames.
- The skills' limits are in micrometres through the calibration, and the tip
  finder adapts to the needle's width, so they carry over; recalibrate whenever
  the camera, objective or adapter changes.
- `bootstrap_labels.py`'s thresholds were tuned to the old frames; expect to
  retune it, or label entirely by hand.

## Closed-loop VLA rollouts

> ⓘ **The rollout client no longer lives in this package.** `main.py` and `sensapex_env.py` were deleted in commit `699bd1f`, along with the `sensapex_rollout` console script. The policy code now sits in the separate training / inference repo (`~/MicroVLA2/rollout`; `~/SmolVLA` holds the older SmolVLA experiments).
>
> This package is now purely the **robot side**. What follows is the interface a policy client has to speak — not documentation of code in this repo.

### What the rig publishes (observation)

| Topic | Use |
|---|---|
| `/camera/image/compressed` | JPEG frame, resize as the policy requires |
| `/ump/live`, `/ump2/live` | `[x, y, z, d]` absolute micrometres per manipulator → the 4- or 8-value state |
| `/heka/resistance_mohm`, `/pressure/measured_mbar` | optional sensor inputs |
| `/vision/scene` | embryos, needle and distances (embryo tokens, the measured step, the prompt check) |
| `/inject/status`, `/orient/status` | progress and counts of the injection macro and the orient skill |

### What the rig accepts (action)

| Topic | Use |
|---|---|
| `/ump/target`, `/ump2/target` | `[x, y, z, d, speed]` absolute micrometres plus speed |
| `/pressure/mbar` | `Float32` exact pressure in mbar |
| `/inject/params`, `/inject/start` | run the injection macro (optionally with the client's own step) |
| `/orient/start` | run the orient skill |
| `/ump/stop`, `/ump2/stop` | SDK stop |

A dual-arm client controls **8 motion coordinates and a separate pressure channel**:

- The 8 motion values are the two 4-axis UMPs only. The ODrive focusing knob is driven separately through `/motor/target_counts` and is not part of the action vector.
- Pressure has a separate scalar head/chunk and normalization in MicroVLA; it is
  not appended to the Sensapex action vector. Its mbar target matches the `target_pressure` column the logger writes — so the policy predicts the same quantity it was trained on. Negative pulls, positive pushes, `0` vents.

This matches the dataset columns the logger writes, so state/action shapes line up between training and inference.

### Client-side responsibilities

These lived in the deleted `main.py` and are now the client's job (MicroVLA's rollout does all of them) — worth re-checking whichever client you roll out from:

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
- `ultralytics` and `torch` — the vision node and the segmentation tools (see
  [`vision_node`](#vision_node) for why ultralytics is installed `--no-deps`)

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

1. Launch the suite as above, from the folder the trials should go into (the
   desktop **My Robot** icon uses the home folder).
2. Use the GUI (or publish on `/ump/target`, `/ump2/target`, `/motor/target_counts` and `/pressure/mbar` directly) to drive the rig. The CSV logger records UMP state/targets, HEKA resistance and the commanded pressure, but not the ODrive motor.
   Pressure targets populate only while the driver reports ready, and measured pressure requires fresh successful readings; faults leave the relevant cells blank.
3. Click **Start Data Acquisition** — this calls `/acq/start`, which opens `logs/trial_N.csv`, creates `saved_frames/trial_N/`, and asks the camera to record `saved_videos/trial_N.mp4`.
4. Perform the trial. The logger writes one row per `log_interval_ms` (default 333 ms = 3 Hz).
5. Click **Stop Data Acquisition** — this calls `/acq/stop`, closes the CSV, and stops the mp4.

For microinjection: press **Orient** when the cell does not face the needle,
**Inject** when the needle is in place, and after each injection choose **Went
into** and press **Record**. The logger records all three.

Output layout (inside the launch folder):

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

## Log validity and manual feedback

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
as the header (35 since 24 October 2026), is flushed after writing, and a
write failure stops acquisition.
CSV cleanup still runs if ROS has already shut down.

`target_pressure` is blank while pressure-driver status is faulted, absent, or
older than 3 seconds. `measured_pressure` is blank once the last valid read is
older than 2 seconds. Zero remains a real measurement/command, not a missing
value. A target is the last SDK-acknowledged request, not proof of physical
settling or of a front-panel command. Direct front-panel changes have no SDK
setpoint acknowledgement and cannot be reconstructed as pressure action labels.

On September 18, two old complete launches were found writing the same trial
files concurrently. That day's `~/logs/trial_28.csv`, `trial_29.csv`, and
`trial_30.csv` contained overwritten/merged rows and `trial_25.csv` had
structural corruption (those files are no longer in `~/logs`). Those originals were preserved, not repaired by guessing. Do not use
corrupt trials for training. The fixes prevent future collisions and do not
recover overwritten measurements or frames.

---

## Maintainer

Raian Haider Chowdhury — `chowd207@umn.edu`

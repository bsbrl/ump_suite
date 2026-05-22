# ruff: noqa
import contextlib
import dataclasses
import signal
import time
import sys
import threading

import numpy as np
import tyro

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from .sensapex_env import SensapexEnv


# === Control rate ===
# Dataset was collected around ~2.5 Hz and stored as 3 Hz for LeRobot.
# 10 Hz can cause overshoot/jitter on real hardware.
CONTROL_FREQUENCY_HZ = 3


# === Safety limits ===
# Units are absolute Sensapex counts (matching /ump/live and /ump2/live in your ROS nodes).
# EDIT THESE for your workspace before running on hardware. `_clamp` below tolerates
# reversed (min > max) ordering, so encoding "deeper z is the higher count" works either way.

# uMp 1
X1_MIN, X1_MAX = 4600, 5700
Y1_MIN, Y1_MAX = 4900, 5500
Z1_MIN, Z1_MAX = 8250, 8750
D1_MIN, D1_MAX = 5900, 6100

# uMp 2
X2_MIN, X2_MAX = 4600, 5700
Y2_MIN, Y2_MAX = 4900, 5500
Z2_MIN, Z2_MAX = 8250, 8750
D2_MIN, D2_MAX = 5900, 6100

# Max step per control tick (prevents sudden jumps)
MAX_DX1 = MAX_DY1 = MAX_DZ1 = MAX_DD1 = 250.0
MAX_DX2 = MAX_DY2 = MAX_DZ2 = MAX_DD2 = 250.0

# Optional EMA smoothing (reduces jitter)
USE_EMA_SMOOTHING = True
EMA_ALPHA = 0.35  # higher = less smoothing, lower = more smoothing


def _clamp(v, lo, hi):
    """Clamp a scalar, accepting bounds in either order."""
    lower = min(float(lo), float(hi))
    upper = max(float(lo), float(hi))
    return lower if v < lower else (upper if v > upper else float(v))


def clamp_action_8d(action_8d: np.ndarray) -> np.ndarray:
    """Clamp absolute action [x1,y1,z1,d1,x2,y2,z2,d2] to safe workspace limits."""
    a = np.asarray(action_8d, dtype=np.float32).reshape(8,)
    return np.array(
        [
            _clamp(a[0], X1_MIN, X1_MAX),
            _clamp(a[1], Y1_MIN, Y1_MAX),
            _clamp(a[2], Z1_MIN, Z1_MAX),
            _clamp(a[3], D1_MIN, D1_MAX),
            _clamp(a[4], X2_MIN, X2_MAX),
            _clamp(a[5], Y2_MIN, Y2_MAX),
            _clamp(a[6], Z2_MIN, Z2_MAX),
            _clamp(a[7], D2_MIN, D2_MAX),
        ],
        dtype=np.float32,
    )


def limit_step(prev_state_8d: np.ndarray, target_action_8d: np.ndarray) -> np.ndarray:
    """
    prev_state_8d: current [x1,y1,z1,d1,x2,y2,z2,d2] from observation
    target_action_8d: absolute desired [x1,y1,z1,d1,x2,y2,z2,d2]
    returns: absolute command with per-step delta caps
    """
    prev = np.asarray(prev_state_8d, dtype=np.float32).reshape(8,)
    tgt = np.asarray(target_action_8d, dtype=np.float32).reshape(8,)
    caps = (MAX_DX1, MAX_DY1, MAX_DZ1, MAX_DD1, MAX_DX2, MAX_DY2, MAX_DZ2, MAX_DD2)

    out = np.empty(8, dtype=np.float32)
    for i, cap in enumerate(caps):
        out[i] = prev[i] + _clamp(tgt[i] - prev[i], -cap, cap)
    return out


def _fmt8(v: np.ndarray) -> str:
    return (
        f"[{v[0]:.0f},{v[1]:.0f},{v[2]:.0f},{v[3]:.0f}|"
        f"{v[4]:.0f},{v[5]:.0f},{v[6]:.0f},{v[7]:.0f}]"
    )


def start_estop_listener():
    """
    Type:  q  then Enter   to stop rollout.
    Works over SSH as long as stdin is attached.
    """
    flag = {"stop": False}

    def _worker():
        while True:
            s = sys.stdin.readline()
            if not s:
                continue
            if s.strip().lower() == "q":
                flag["stop"] = True
                break

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return flag


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


@dataclasses.dataclass
class Args:
    # Policy server (OpenPI server)
    remote_host: str = "127.0.0.1"
    remote_port: int = 8000

    # Rollout
    max_timesteps: int = 600
    open_loop_horizon: int = 8

    # Camera preprocessing
    resize_h: int = 224
    resize_w: int = 224

    # Robot params
    default_speed: int = 100

    # Live preview (writes a file on robot pc)
    save_preview: bool = True
    preview_path: str = "sensapex_live.png"
    preview_every_n_frames: int = 5

    # Debug prints every N steps
    debug_every: int = 10


def main(args: Args):
    # Connect env (ROS)
    env = SensapexEnv(
        save_preview=args.save_preview,
        preview_path=args.preview_path,
        preview_every_n_frames=args.preview_every_n_frames,
        default_speed=args.default_speed,
    )

    if args.save_preview:
        print(f"[sensapex] Live preview will be saved to: {args.preview_path}")

    # Connect to policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    instruction = input("Enter instruction: ").strip()
    if not instruction:
        instruction = "Move the needles towards the bead"

    print("Running rollout...")
    print("  - Press Ctrl+C to stop early")
    print("  - Type 'q' + Enter to E-STOP (stop sending actions + hold position)")

    stop_flag = start_estop_listener()

    actions_from_chunk_completed = 0
    pred_action_chunk = None

    ema_action = None
    period = 1.0 / float(CONTROL_FREQUENCY_HZ)

    for t in range(int(args.max_timesteps)):
        start_time = time.time()
        try:
            if stop_flag["stop"]:
                # Hold current position once then exit
                obs = env.get_observation()
                hold = obs.state.astype(np.float32).copy()
                print("[E-STOP] Holding current position and exiting.")
                env.step_absolute(hold)
                break

            obs = env.get_observation()
            img = obs.image_rgb  # RGB uint8
            state = obs.state.astype(np.float32)  # (8,) [x1,y1,z1,d1, x2,y2,z2,d2]

            # Query policy server when needed
            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                actions_from_chunk_completed = 0

                # NOTE: keys here must match what `SensapexInputs` reads
                # (see src/openpi/policies/sensapex_policy.py). The repack transform
                # in `LeRobotSensapexDataConfig` only runs at training time.
                request_data = {
                    "observation/image": image_tools.resize_with_pad(img, args.resize_h, args.resize_w),
                    "observation/state": state,
                    "prompt": instruction,
                }

                with prevent_keyboard_interrupt():
                    resp = policy_client.infer(request_data)

                if "actions" not in resp:
                    raise RuntimeError(f"Policy response missing 'actions' key. Keys={list(resp.keys())}")

                pred_action_chunk = np.asarray(resp["actions"], dtype=np.float32)

                if pred_action_chunk.ndim != 2 or pred_action_chunk.shape[1] != 8:
                    raise RuntimeError(f"Expected actions shape (T,8), got {pred_action_chunk.shape}")

            # Execute one action from the chunk
            action = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1

            # --- Safety + smoothing pipeline ---
            action = clamp_action_8d(action)
            action = limit_step(state, action)

            if USE_EMA_SMOOTHING:
                if ema_action is None:
                    ema_action = action.copy()
                else:
                    ema_action = (EMA_ALPHA * action) + ((1.0 - EMA_ALPHA) * ema_action)
                cmd = ema_action
            else:
                cmd = action

            # Send to robot (absolute targets for both uMps)
            env.step_absolute(cmd)

            if args.debug_every > 0 and (t % int(args.debug_every) == 0):
                print(f"[t={t:04d}] state={_fmt8(state)} cmd={_fmt8(cmd)}")

            # Sleep to match control frequency
            elapsed = time.time() - start_time
            if elapsed < period:
                time.sleep(period - elapsed)

        except KeyboardInterrupt:
            print("Stopped early (Ctrl+C).")
            break

    env.close()


def main_entry():
    args: Args = tyro.cli(Args)
    main(args)


if __name__ == "__main__":
    main_entry()

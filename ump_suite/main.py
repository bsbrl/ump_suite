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
# Units: "centered counts" (same units as /ump/live and /ump/target in the ROS nodes).
X_MIN, X_MAX = 2040, 3360
Y_MIN, Y_MAX = 5180, 6120
Z_MIN, Z_MAX = 8650, 8730
D_MIN, D_MAX = 5845, 5850


# Motor ticks safety (EDIT for the stage)
H_MIN, H_MAX = -200, 100

# Max step per control tick (prevents sudden jumps)
MAX_DX = 250.0
MAX_DY = 250.0
MAX_DZ = 250.0
MAX_DD = 250.0
MAX_DH = 5000.0

# Optional EMA smoothing (reduces jitter)
USE_EMA_SMOOTHING = True
EMA_ALPHA = 0.35  # higher = less smoothing, lower = more smoothing


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def clamp_action_5d(action_5d: np.ndarray) -> np.ndarray:
    """Clamp absolute action [x,y,z,d,h_ticks] to safe workspace limits."""
    a = np.asarray(action_5d, dtype=np.float32).reshape(5,)
    x, y, z, d, h = [float(v) for v in a]
    x = _clamp(x, X_MIN, X_MAX)
    y = _clamp(y, Y_MIN, Y_MAX)
    z = _clamp(z, Z_MIN, Z_MAX)
    d = _clamp(d, D_MIN, D_MAX)
    h = _clamp(h, H_MIN, H_MAX)
    return np.array([x, y, z, d, h], dtype=np.float32)


def limit_step(prev_state_5d: np.ndarray, target_action_5d: np.ndarray) -> np.ndarray:
    """
    prev_state_5d: current [x,y,z,d,h] from observation
    target_action_5d: absolute desired [x,y,z,d,h]
    returns: absolute command with per-step delta caps
    """
    prev = np.asarray(prev_state_5d, dtype=np.float32).reshape(5,)
    tgt = np.asarray(target_action_5d, dtype=np.float32).reshape(5,)

    dx = _clamp(tgt[0] - prev[0], -MAX_DX, MAX_DX)
    dy = _clamp(tgt[1] - prev[1], -MAX_DY, MAX_DY)
    dz = _clamp(tgt[2] - prev[2], -MAX_DZ, MAX_DZ)
    dd = _clamp(tgt[3] - prev[3], -MAX_DD, MAX_DD)
    dh = _clamp(tgt[4] - prev[4], -MAX_DH, MAX_DH)

    out = np.array(
        [prev[0] + dx, prev[1] + dy, prev[2] + dz, prev[3] + dd, prev[4] + dh],
        dtype=np.float32,
    )
    return out


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
        instruction = "Move the needle towards the bead"

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
            state = obs.state.astype(np.float32)  # (5,) [x,y,z,d,h_ticks]

            # Query policy server when needed
            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                actions_from_chunk_completed = 0

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

                if pred_action_chunk.ndim != 2 or pred_action_chunk.shape[1] != 5:
                    raise RuntimeError(f"Expected actions shape (T,5), got {pred_action_chunk.shape}")

            # Execute one action from the chunk
            action = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1

            # --- Safety + smoothing pipeline ---
            action = clamp_action_5d(action)
            action = limit_step(state, action)

            if USE_EMA_SMOOTHING:
                if ema_action is None:
                    ema_action = action.copy()
                else:
                    ema_action = (EMA_ALPHA * action) + ((1.0 - EMA_ALPHA) * ema_action)
                cmd = ema_action
            else:
                cmd = action

            # Send to robot (absolute targets)
            env.step_absolute(cmd)

            if args.debug_every > 0 and (t % int(args.debug_every) == 0):
                print(
                    f"[t={t:04d}] state=[{state[0]:.0f},{state[1]:.0f},{state[2]:.0f},{state[3]:.0f},{state[4]:.0f}] "
                    f"cmd=[{cmd[0]:.0f},{cmd[1]:.0f},{cmd[2]:.0f},{cmd[3]:.0f},{cmd[4]:.0f}]"
                )

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

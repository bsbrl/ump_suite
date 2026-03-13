# ruff: noqa
import contextlib
import dataclasses
import signal
import sys
import threading
import time

import numpy as np
import tyro

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from .sensapex_env import SensapexEnv


CONTROL_FREQUENCY_HZ = 3


# Absolute workspace limits
X_MIN, X_MAX = -10000, 10000
Y_MIN, Y_MAX = -10000, 10000
Z_MIN, Z_MAX = -10000, 10000
D_MIN, D_MAX = -10000, 10000
H_MIN, H_MAX = -1_000_000, 1_000_000


# Delta limits per step
DX_MIN, DX_MAX = -300.0, 300.0
DY_MIN, DY_MAX = -300.0, 300.0
DZ_MIN, DZ_MAX = -300.0, 300.0
DD_MIN, DD_MAX = -300.0, 300.0
DH_MIN, DH_MAX = -5000.0, 5000.0


# Optional scaling on model deltas
DELTA_SCALE_X = 1.0
DELTA_SCALE_Y = 1.0
DELTA_SCALE_Z = 1.0
DELTA_SCALE_D = 1.0
DELTA_SCALE_H = 1.0


# Smoothing
USE_EMA_SMOOTHING = True
EMA_ALPHA = 0.35


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def clamp_absolute_target(target_5d: np.ndarray) -> np.ndarray:
    t = np.asarray(target_5d, dtype=np.float32).reshape(5,)
    x, y, z, d, h = [float(v) for v in t]

    x = _clamp(x, X_MIN, X_MAX)
    y = _clamp(y, Y_MIN, Y_MAX)
    z = _clamp(z, Z_MIN, Z_MAX)
    d = _clamp(d, D_MIN, D_MAX)
    h = _clamp(h, H_MIN, H_MAX)

    return np.array([x, y, z, d, h], dtype=np.float32)


def clamp_delta(delta_5d: np.ndarray) -> np.ndarray:
    d = np.asarray(delta_5d, dtype=np.float32).reshape(5,)
    dx, dy, dz, dd, dh = [float(v) for v in d]

    dx = _clamp(dx * DELTA_SCALE_X, DX_MIN, DX_MAX)
    dy = _clamp(dy * DELTA_SCALE_Y, DY_MIN, DY_MAX)
    dz = _clamp(dz * DELTA_SCALE_Z, DZ_MIN, DZ_MAX)
    dd = _clamp(dd * DELTA_SCALE_D, DD_MIN, DD_MAX)
    dh = _clamp(dh * DELTA_SCALE_H, DH_MIN, DH_MAX)

    return np.array([dx, dy, dz, dd, dh], dtype=np.float32)


def start_estop_listener():
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
    remote_host: str = "127.0.0.1"
    remote_port: int = 8000

    max_timesteps: int = 600
    open_loop_horizon: int = 8

    resize_h: int = 224
    resize_w: int = 224

    default_speed: int = 1000

    save_preview: bool = True
    preview_path: str = "sensapex_live.png"
    preview_every_n_frames: int = 5

    debug_every: int = 10


def main(args: Args):
    env = SensapexEnv(
        save_preview=args.save_preview,
        preview_path=args.preview_path,
        preview_every_n_frames=args.preview_every_n_frames,
        default_speed=args.default_speed,
    )

    if args.save_preview:
        print(f"[sensapex] Live preview will be saved to: {args.preview_path}")

    policy_client = websocket_client_policy.WebsocketClientPolicy(
        args.remote_host,
        args.remote_port,
    )

    instruction = input("Enter instruction: ").strip()
    if not instruction:
        instruction = "Move the needle towards the bead"

    print("Running delta rollout...")
    print("  - Press Ctrl+C to stop early")
    print("  - Type 'q' + Enter to E-STOP (hold current position and exit)")

    stop_flag = start_estop_listener()

    actions_from_chunk_completed = 0
    pred_action_chunk = None
    ema_target = None
    period = 1.0 / float(CONTROL_FREQUENCY_HZ)

    for t in range(int(args.max_timesteps)):
        start_time = time.time()
        try:
            obs = env.get_observation()
            img = obs.image_rgb
            state = obs.state.astype(np.float32)

            if stop_flag["stop"]:
                print("[E-STOP] Holding current position and exiting.")
                env.step_absolute(state)
                break

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

            raw_delta = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1

            delta = clamp_delta(raw_delta)

            # delta -> absolute target
            target = state + delta
            target = clamp_absolute_target(target)

            if USE_EMA_SMOOTHING:
                if ema_target is None:
                    ema_target = target.copy()
                else:
                    ema_target = (EMA_ALPHA * target) + ((1.0 - EMA_ALPHA) * ema_target)
                cmd = ema_target
            else:
                cmd = target

            cmd = clamp_absolute_target(cmd)
            env.step_absolute(cmd)

            if args.debug_every > 0 and (t % int(args.debug_every) == 0):
                print(
                    f"[t={t:04d}] "
                    f"state=[{state[0]:.0f},{state[1]:.0f},{state[2]:.0f},{state[3]:.0f},{state[4]:.0f}] "
                    f"raw_delta=[{raw_delta[0]:.0f},{raw_delta[1]:.0f},{raw_delta[2]:.0f},{raw_delta[3]:.0f},{raw_delta[4]:.0f}] "
                    f"delta=[{delta[0]:.0f},{delta[1]:.0f},{delta[2]:.0f},{delta[3]:.0f},{delta[4]:.0f}] "
                    f"cmd=[{cmd[0]:.0f},{cmd[1]:.0f},{cmd[2]:.0f},{cmd[3]:.0f},{cmd[4]:.0f}]"
                )

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
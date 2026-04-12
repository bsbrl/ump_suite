"""Closed-loop rollout where the policy emits *absolute* target poses.

Pipeline per tick:
    obs ──► policy.infer ──► clamp_action_9d ──► limit_step (vs. current state)
                                           │
                                           ▼
                            optional EMA smoothing ──► env.step_absolute

The rig carries **two** Sensapex micromanipulators plus a single ODrive
motor, so every state / action vector is 9-dim:
``[x1, y1, z1, d1, x2, y2, z2, d2, h_ticks]``.
"""
# ruff: noqa
import time

import numpy as np
import tyro
from openpi_client import image_tools, websocket_client_policy

from .rollout import (
    RolloutArgs,
    clamp,
    prevent_keyboard_interrupt,
    start_estop_listener,
)
from .sensapex_env import SensapexEnv


# === Control rate ===
# Dataset was collected at ~2.5 Hz and stored as 3 Hz for LeRobot.
# 10 Hz can cause overshoot/jitter on real hardware.
CONTROL_FREQUENCY_HZ = 3


# === Safety limits ===
# Units are "centered counts" (same coordinates the ROS nodes use on
# /ump/live, /ump/target, /ump2/live, /ump2/target).

# --- UMP1 workspace box ---
X1_MIN, X1_MAX = 4600, 5700
Y1_MIN, Y1_MAX = 4900, 5500
Z1_MIN, Z1_MAX = 8750, 8250
D1_MIN, D1_MAX = 5900, 6100

# --- UMP2 workspace box ---
# NOTE: duplicated from UMP1 as placeholders — edit for your second stage.
X2_MIN, X2_MAX = 4600, 5700
Y2_MIN, Y2_MAX = 4900, 5500
Z2_MIN, Z2_MAX = 8750, 8250
D2_MIN, D2_MAX = 5900, 6100

# ODrive motor ticks safety window (edit per stage).
H_MIN, H_MAX = -6000, 1000

# Max absolute step per control tick — caps single-tick jumps even if the
# policy spits out a target far from where we are right now.
MAX_DX1 = 250.0
MAX_DY1 = 250.0
MAX_DZ1 = 250.0
MAX_DD1 = 250.0
MAX_DX2 = 250.0
MAX_DY2 = 250.0
MAX_DZ2 = 250.0
MAX_DD2 = 250.0
MAX_DH  = 5000.0

# Optional first-order smoothing on the commanded action.
USE_EMA_SMOOTHING = True
EMA_ALPHA = 0.35  # 1.0 = no smoothing, →0 = heavy smoothing


def clamp_action_9d(action_9d: np.ndarray) -> np.ndarray:
    """Clamp absolute action [x1,y1,z1,d1, x2,y2,z2,d2, h_ticks] to safe box."""
    a = np.asarray(action_9d, dtype=np.float32).reshape(9,)
    return np.array(
        [
            clamp(float(a[0]), X1_MIN, X1_MAX),
            clamp(float(a[1]), Y1_MIN, Y1_MAX),
            clamp(float(a[2]), Z1_MIN, Z1_MAX),
            clamp(float(a[3]), D1_MIN, D1_MAX),
            clamp(float(a[4]), X2_MIN, X2_MAX),
            clamp(float(a[5]), Y2_MIN, Y2_MAX),
            clamp(float(a[6]), Z2_MIN, Z2_MAX),
            clamp(float(a[7]), D2_MIN, D2_MAX),
            clamp(float(a[8]), H_MIN,  H_MAX),
        ],
        dtype=np.float32,
    )


def limit_step(prev_state_9d: np.ndarray, target_action_9d: np.ndarray) -> np.ndarray:
    """Cap each axis' per-tick movement so a far-away target ramps in safely."""
    prev = np.asarray(prev_state_9d,    dtype=np.float32).reshape(9,)
    tgt  = np.asarray(target_action_9d, dtype=np.float32).reshape(9,)

    caps = (
        MAX_DX1, MAX_DY1, MAX_DZ1, MAX_DD1,
        MAX_DX2, MAX_DY2, MAX_DZ2, MAX_DD2,
        MAX_DH,
    )
    out = np.empty(9, dtype=np.float32)
    for i, cap in enumerate(caps):
        out[i] = prev[i] + clamp(tgt[i] - prev[i], -cap, cap)
    return out


def main(args: RolloutArgs):
    env = SensapexEnv(
        save_preview=args.save_preview,
        preview_path=args.preview_path,
        preview_every_n_frames=args.preview_every_n_frames,
        default_speed=args.default_speed,
    )
    if args.save_preview:
        print(f"[sensapex] Live preview will be saved to: {args.preview_path}")

    policy_client = websocket_client_policy.WebsocketClientPolicy(
        args.remote_host, args.remote_port
    )

    instruction = input("Enter instruction: ").strip() or "Move the needle towards the bead"

    print("Running rollout...")
    print("  - Press Ctrl+C to stop early")
    print("  - Type 'q' + Enter to E-STOP (stop sending actions + hold position)")

    stop_flag = start_estop_listener()

    actions_completed_in_chunk = 0
    pred_action_chunk = None
    ema_action = None
    period = 1.0 / float(CONTROL_FREQUENCY_HZ)

    for t in range(int(args.max_timesteps)):
        start_time = time.time()
        try:
            if stop_flag["stop"]:
                # Hold the most recent state once, then exit.
                obs = env.get_observation()
                hold = obs.state.astype(np.float32).copy()
                print("[E-STOP] Holding current position and exiting.")
                env.step_absolute(hold)
                break

            obs = env.get_observation()
            img = obs.image_rgb
            state = obs.state.astype(np.float32)

            # Fetch a fresh chunk of actions from the policy whenever the
            # previous chunk has been fully consumed.
            need_new_chunk = (
                actions_completed_in_chunk == 0
                or actions_completed_in_chunk >= args.open_loop_horizon
            )
            if need_new_chunk:
                actions_completed_in_chunk = 0
                request = {
                    "observation/image": image_tools.resize_with_pad(
                        img, args.resize_h, args.resize_w
                    ),
                    "observation/state": state,
                    "prompt": instruction,
                }
                with prevent_keyboard_interrupt():
                    resp = policy_client.infer(request)

                if "actions" not in resp:
                    raise RuntimeError(
                        f"Policy response missing 'actions' key. Keys={list(resp.keys())}"
                    )
                pred_action_chunk = np.asarray(resp["actions"], dtype=np.float32)
                if pred_action_chunk.ndim != 2 or pred_action_chunk.shape[1] != 9:
                    raise RuntimeError(
                        f"Expected actions shape (T,9), got {pred_action_chunk.shape}"
                    )

            # Pop the next action from the open-loop chunk and run it through
            # the safety + smoothing pipeline.
            action = pred_action_chunk[actions_completed_in_chunk]
            actions_completed_in_chunk += 1

            action = clamp_action_9d(action)
            action = limit_step(state, action)

            if USE_EMA_SMOOTHING:
                if ema_action is None:
                    ema_action = action.copy()
                else:
                    ema_action = EMA_ALPHA * action + (1.0 - EMA_ALPHA) * ema_action
                cmd = ema_action
            else:
                cmd = action

            env.step_absolute(cmd)

            if args.debug_every > 0 and (t % int(args.debug_every) == 0):
                print(
                    f"[t={t:04d}] "
                    f"state=[{state[0]:.0f},{state[1]:.0f},{state[2]:.0f},{state[3]:.0f}|"
                    f"{state[4]:.0f},{state[5]:.0f},{state[6]:.0f},{state[7]:.0f}|"
                    f"{state[8]:.0f}] "
                    f"cmd=[{cmd[0]:.0f},{cmd[1]:.0f},{cmd[2]:.0f},{cmd[3]:.0f}|"
                    f"{cmd[4]:.0f},{cmd[5]:.0f},{cmd[6]:.0f},{cmd[7]:.0f}|"
                    f"{cmd[8]:.0f}]"
                )

            elapsed = time.time() - start_time
            if elapsed < period:
                time.sleep(period - elapsed)

        except KeyboardInterrupt:
            print("Stopped early (Ctrl+C).")
            break

    env.close()


def main_entry():
    main(tyro.cli(RolloutArgs))


if __name__ == "__main__":
    main_entry()

"""Shared building blocks for the OpenPI rollout scripts.

`main.py` and `main_delta.py` both run a closed-loop policy on the Sensapex
hardware. They differ only in how policy outputs are interpreted (absolute
targets vs. per-step deltas), so the boring scaffolding lives here.
"""

import contextlib
import dataclasses
import signal
import sys
import threading


def clamp(v, lo, hi):
    """Branchless clamp that works for both python ints and numpy scalars."""
    return lo if v < lo else (hi if v > hi else v)


def start_estop_listener():
    """Watch stdin for `q` + Enter and flip a flag the rollout loop polls.

    Works over SSH as long as stdin is attached. The returned dict is mutated
    in place by the worker thread.
    """
    flag = {"stop": False}

    def _worker():
        while True:
            line = sys.stdin.readline()
            if not line:
                continue
            if line.strip().lower() == "q":
                flag["stop"] = True
                break

    threading.Thread(target=_worker, daemon=True).start()
    return flag


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Block SIGINT inside the with-block, then re-raise it once on exit.

    Used to wrap the synchronous policy-server call: a Ctrl+C right in the
    middle of an inference round-trip would otherwise corrupt the websocket
    state and crash the loop instead of stopping cleanly.
    """
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(_signum, _frame):
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
class RolloutArgs:
    """CLI arguments shared by both rollout scripts."""

    # Policy server (OpenPI websocket)
    remote_host: str = "127.0.0.1"
    remote_port: int = 8000

    # Rollout
    max_timesteps: int = 600
    open_loop_horizon: int = 8

    # Camera preprocessing (resize before sending to the policy)
    resize_h: int = 224
    resize_w: int = 224

    # Robot params
    default_speed: int = 100

    # Live preview file (handy when SSH'ed in)
    save_preview: bool = True
    preview_path: str = "sensapex_live.png"
    preview_every_n_frames: int = 5

    # Print one debug line every N steps (0 disables)
    debug_every: int = 10

"""Async RTC control client for the Piper arm (local Windows / robot side).

Full-version companion to `scripts/serve_policy_rtc.py`. It reuses the camera rig, robot
interface and observation builder from `scripts/piper_remote_client.py` UNMODIFIED, and
adds the two things that make RTC actually hide latency:

  1. Async prefetch: a background thread requests the NEXT chunk from the RTC server while
     the arm keeps executing the CURRENT chunk -> no stall at chunk boundaries.
  2. Timeline alignment: the arm executes exactly `execution_horizon` steps per chunk (so
     the server's `prev_chunk[execution_horizon:]` prefix lines up), and the measured
     inference latency (in control steps) is fed back as `inference_delay` so the server
     hard-commits exactly the steps that elapse during a fetch.

Loop shape (control runs at --control-hz):
    execute chunk[0 : execution_horizon]; when `prefetch_lead` steps remain, fire a
    background fetch of the next chunk; at the boundary, swap to the prefetched chunk.

Requires the RTC server (serve_policy_rtc.py) on the GPU box; point --server-host at it
(or at the forwarded localhost if you SSH-tunnel port 8000).

Run (local, robot connected):
    python scripts\piper_rtc_client.py --server-host 127.0.0.1 --server-port 8000 \
        --bci-piper-root D:\hxf\code\piper\bci_piper --real \
        --prompt "Pick up the blue pen and place it in the large container" \
        --control-hz 30 --execution-horizon 8 --prefetch-lead 4 --no-disable-gripper
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time

import numpy as np
import tyro

from openpi_client import websocket_client_policy

try:
    import piper_remote_client as prc
except ImportError:
    from scripts import piper_remote_client as prc


@dataclasses.dataclass
class Args(prc.Args):
    # How many actions to execute from each chunk before swapping to the next one.
    # MUST equal the server's --execution-horizon so the RTC prefix aligns.
    execution_horizon: int = 8
    # Fire the background prefetch when this many steps remain in the current chunk.
    # Set >= your typical inference latency in control steps so the next chunk arrives in time.
    prefetch_lead: int = 4
    # Initial guess for inference latency (in control steps) before the first real measurement.
    default_inference_delay: int = 2
    # Disable async prefetch (synchronous: stall each boundary). For A/B debugging only.
    synchronous: bool = False


class RTCFetcher:
    """Owns the websocket call + shared state between the main loop and the fetch thread."""

    def __init__(self, client: websocket_client_policy.WebsocketClientPolicy, control_dt: float, default_delay: int):
        self._client = client
        self._dt = control_dt
        self._lock = threading.Lock()
        self._chunk: np.ndarray | None = None
        self._busy = False
        self.delay_steps = int(default_delay)  # last measured inference latency, in control steps
        self.last_infer_ms = 0.0

    def _do_fetch(self, obs: dict, *, reset: bool) -> None:
        payload = dict(obs)
        payload["rtc_reset"] = reset
        payload["inference_delay"] = int(self.delay_steps)
        t0 = time.monotonic()
        result = self._client.infer(payload)
        elapsed = time.monotonic() - t0
        chunk = np.asarray(result["actions"], dtype=np.float32)
        with self._lock:
            self._chunk = chunk
            self.delay_steps = max(1, math.ceil(elapsed / self._dt))
            self.last_infer_ms = elapsed * 1000.0
            self._busy = False

    def fetch_blocking(self, obs: dict, *, reset: bool) -> np.ndarray:
        self._do_fetch(obs, reset=reset)
        return self.take()

    def start_fetch(self, obs: dict, *, reset: bool = False) -> None:
        with self._lock:
            if self._busy or self._chunk is not None:
                return
            self._busy = True
        threading.Thread(target=self._do_fetch, args=(obs,), kwargs={"reset": reset}, daemon=True).start()

    def status(self) -> tuple[bool, bool]:
        with self._lock:
            return self._busy, self._chunk is not None

    def take(self) -> np.ndarray | None:
        with self._lock:
            chunk, self._chunk = self._chunk, None
            return chunk


def main(args: Args) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(args.server_host, args.server_port)
    print("Connected to RTC policy server:", client.get_server_metadata())

    cameras = prc.CameraRig(args)
    robot = prc.PiperRobotClient(args)
    if args.home_on_start:
        robot.move_to_home(args)

    dt = 1.0 / args.control_hz
    eh = int(args.execution_horizon)
    fetcher = RTCFetcher(client, dt, args.default_inference_delay)

    def observe() -> dict:
        obs, _, _ = prc.build_observation(cameras, robot, args.prompt, args.image_size, show_preview=args.show_preview)
        return obs

    try:
        # First chunk: blocking, reset the server's RTC state for a fresh episode.
        chunk = fetcher.fetch_blocking(observe(), reset=True)
        print(f"step=0 first chunk shape={chunk.shape} infer={fetcher.last_infer_ms:.0f}ms")
        exec_idx = 0

        for step in range(args.max_steps):
            start = time.perf_counter()

            # Prefetch the next chunk once we are within `lead` steps of the boundary.
            # `lead` adapts to the measured inference latency (delay_steps) so a slow/laggy
            # fetch starts earlier and still arrives before the swap; floored by --prefetch-lead.
            if not args.synchronous:
                busy, ready = fetcher.status()
                lead = min(eh, max(args.prefetch_lead, fetcher.delay_steps + 1))
                if not busy and not ready and exec_idx >= eh - lead:
                    fetcher.start_fetch(observe(), reset=False)

            # Execute one action from the current chunk (clamp index to chunk length).
            model_action = chunk[min(exec_idx, len(chunk) - 1)]
            obs_state = robot.get_state()
            current_state, model_action_dbg, sent_action = robot.send_action(model_action)
            if args.print_state_debug and step % max(1, args.print_every) == 0:
                prc.print_debug(step, obs_state, current_state, model_action_dbg, sent_action)
            exec_idx += 1

            # At the execution horizon, swap to the next chunk.
            if exec_idx >= eh:
                if args.synchronous:
                    chunk = fetcher.fetch_blocking(observe(), reset=False)
                else:
                    nxt = fetcher.take()
                    while nxt is None:  # prefetch not back yet -> unavoidable stall
                        time.sleep(dt)
                        nxt = fetcher.take()
                    chunk = nxt
                print(f"step={step} swap chunk delay_steps={fetcher.delay_steps} infer={fetcher.last_infer_ms:.0f}ms")
                exec_idx = 0

            elapsed = time.perf_counter() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    finally:
        if args.show_preview:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        cameras.close()
        robot.close()


if __name__ == "__main__":
    main(tyro.cli(Args))

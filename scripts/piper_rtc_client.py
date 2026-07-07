r"""Async RTC control client for the Piper arm (local Windows / robot side).

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
    # Print the raw VLA-predicted action curve before client smoothing/clipping.
    print_action_curve: bool = False
    # Rolling window used for per-step VLA action jitter stats.
    action_curve_window: int = 30
    # Print rolling stats every N control steps when --print-action-curve is enabled.
    action_curve_every: int = 10
    # Optional CSV path for raw VLA action samples actually selected for execution.
    action_curve_csv: str | None = None


@dataclasses.dataclass
class FetchResult:
    chunk: np.ndarray
    next_exec_idx: int


class ActionCurveLogger:
    def __init__(self, args: Args):
        self._enabled = bool(args.print_action_curve)
        self._window = max(2, int(args.action_curve_window))
        self._every = max(1, int(args.action_curve_every))
        self._history: list[np.ndarray] = []
        self._csv = None
        if args.action_curve_csv is not None:
            self._enabled = True
            self._csv = open(args.action_curve_csv, "w", encoding="utf-8")
            self._csv.write("step,source,chunk_index,a0,a1,a2,a3,a4,a5,a6\n")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def close(self) -> None:
        if self._csv is not None:
            self._csv.close()
            self._csv = None

    def _write_csv(self, step: int, source: str, chunk_index: int, action: np.ndarray) -> None:
        if self._csv is None:
            return
        vals = ",".join(f"{float(x):.8f}" for x in np.asarray(action, dtype=np.float32)[:7])
        self._csv.write(f"{step},{source},{chunk_index},{vals}\n")
        self._csv.flush()

    def log_chunk(self, *, step: int, source: str, chunk: np.ndarray, start_idx: int, horizon: int) -> None:
        if not self._enabled:
            return
        chunk = np.asarray(chunk, dtype=np.float32)
        end = min(len(chunk), start_idx + horizon)
        segment = chunk[start_idx:end, :7]
        print(
            f"[vla_curve] step={step} {source} start_idx={start_idx} "
            f"rows={segment.shape[0]} dims=joint1..joint6,gripper"
        )
        print(np.array2string(segment, precision=4, suppress_small=False, max_line_width=160))
        for i, action in enumerate(segment):
            self._write_csv(step, source, start_idx + i, action)
        if segment.shape[0] >= 2:
            delta = np.diff(segment, axis=0)
            print(
                f"[vla_curve] step={step} {source} delta_mean_abs="
                f"{np.round(np.mean(np.abs(delta), axis=0), 5).tolist()} "
                f"delta_max_abs={np.round(np.max(np.abs(delta), axis=0), 5).tolist()} "
                f"delta_l2_mean={float(np.linalg.norm(delta[:, :6], axis=1).mean()):.5f} "
                f"delta_l2_max={float(np.linalg.norm(delta[:, :6], axis=1).max()):.5f}"
            )

    def log_step(self, *, step: int, chunk_index: int, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float32)[:7]
        self._write_csv(step, "executed", chunk_index, action)
        if not self._enabled:
            return
        self._history.append(action.copy())
        if len(self._history) > self._window:
            self._history = self._history[-self._window :]
        if step % self._every != 0 or len(self._history) < 2:
            return
        arr = np.stack(self._history, axis=0)
        delta = np.diff(arr, axis=0)
        print(
            f"[vla_curve] step={step} rolling_window={arr.shape[0]} "
            f"selected_delta_mean_abs={np.round(np.mean(np.abs(delta), axis=0), 5).tolist()} "
            f"selected_delta_max_abs={np.round(np.max(np.abs(delta), axis=0), 5).tolist()} "
            f"selected_delta_l2_mean={float(np.linalg.norm(delta[:, :6], axis=1).mean()):.5f} "
            f"selected_delta_l2_max={float(np.linalg.norm(delta[:, :6], axis=1).max()):.5f}"
        )


class RTCFetcher:
    """Owns the websocket call + shared state between the main loop and the fetch thread."""

    def __init__(self, client: websocket_client_policy.WebsocketClientPolicy, control_dt: float, default_delay: int):
        self._client = client
        self._dt = control_dt
        self._lock = threading.Lock()
        self._result: FetchResult | None = None
        self._error: BaseException | None = None
        self._busy = False
        self.delay_steps = int(default_delay)  # last measured inference latency, in control steps
        self.last_infer_ms = 0.0

    def _do_fetch(self, obs: dict, *, reset: bool, executed: int, next_exec_idx: int) -> None:
        try:
            payload = dict(obs)
            payload["rtc_reset"] = reset
            payload["rtc_executed"] = int(executed)
            payload["inference_delay"] = int(self.delay_steps)
            t0 = time.monotonic()
            result = self._client.infer(payload)
            elapsed = time.monotonic() - t0
            chunk = np.asarray(result["actions"], dtype=np.float32)
            with self._lock:
                self._result = FetchResult(chunk=chunk, next_exec_idx=int(next_exec_idx))
                self.delay_steps = max(1, math.ceil(elapsed / self._dt))
                self.last_infer_ms = elapsed * 1000.0
                self._busy = False
        except BaseException as exc:
            with self._lock:
                self._error = exc
                self._busy = False

    def fetch_blocking(self, obs: dict, *, reset: bool, executed: int, next_exec_idx: int = 0) -> FetchResult:
        self._do_fetch(obs, reset=reset, executed=executed, next_exec_idx=next_exec_idx)
        result = self.take()
        if result is None:
            raise RuntimeError("RTC fetch failed without returning a chunk")
        return result

    def start_fetch(self, obs: dict, *, reset: bool = False, executed: int, next_exec_idx: int) -> None:
        with self._lock:
            if self._busy or self._result is not None:
                return
            self._busy = True
            self._error = None
        threading.Thread(
            target=self._do_fetch,
            args=(obs,),
            kwargs={"reset": reset, "executed": executed, "next_exec_idx": next_exec_idx},
            daemon=True,
        ).start()

    def status(self) -> tuple[bool, bool]:
        with self._lock:
            if self._error is not None:
                raise RuntimeError("RTC background fetch failed") from self._error
            return self._busy, self._result is not None

    def take(self) -> FetchResult | None:
        with self._lock:
            if self._error is not None:
                error, self._error = self._error, None
                raise RuntimeError("RTC fetch failed") from error
            result, self._result = self._result, None
            return result


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
    action_curve = ActionCurveLogger(args)

    def observe() -> dict:
        obs, _, _ = prc.build_observation(cameras, robot, args.prompt, args.image_size, show_preview=args.show_preview)
        return obs

    try:
        # First chunk: blocking, reset the server's RTC state for a fresh episode.
        first = fetcher.fetch_blocking(observe(), reset=True, executed=0)
        chunk = first.chunk
        exec_idx = first.next_exec_idx
        steps_since_swap = 0
        print(f"step=0 first chunk shape={chunk.shape} infer={fetcher.last_infer_ms:.0f}ms")
        action_curve.log_chunk(step=0, source="first_chunk", chunk=chunk, start_idx=exec_idx, horizon=eh)

        for step in range(args.max_steps):
            start = time.perf_counter()

            # Prefetch the next chunk once we are within `lead` steps of the boundary.
            # The request tells the server how much of this full chunk has already been
            # executed, and records where the returned chunk should start at the swap.
            if not args.synchronous:
                busy, ready = fetcher.status()
                max_safe_lead = max(0, len(chunk) - eh)
                requested_lead = min(eh, max(args.prefetch_lead, fetcher.delay_steps + 1))
                lead = min(requested_lead, max_safe_lead)
                if lead > 0 and not busy and not ready and steps_since_swap >= eh - lead:
                    skip_at_swap = eh - steps_since_swap
                    fetcher.start_fetch(
                        observe(), reset=False, executed=exec_idx, next_exec_idx=skip_at_swap
                    )

            # Execute one action from the current full server chunk.
            action_idx = min(exec_idx, len(chunk) - 1)
            model_action = chunk[action_idx]
            action_curve.log_step(step=step, chunk_index=action_idx, action=model_action)
            obs_state = robot.get_state()
            current_state, model_action_dbg, sent_action = robot.send_action(model_action)
            if args.print_state_debug and step % max(1, args.print_every) == 0:
                prc.print_debug(step, obs_state, current_state, model_action_dbg, sent_action)
            exec_idx += 1
            steps_since_swap += 1

            # At the execution horizon, swap to the next chunk.
            if steps_since_swap >= eh:
                if args.synchronous:
                    nxt = fetcher.fetch_blocking(observe(), reset=False, executed=exec_idx)
                else:
                    nxt = fetcher.take()
                    if nxt is None:
                        busy, _ = fetcher.status()
                        if not busy:
                            nxt = fetcher.fetch_blocking(observe(), reset=False, executed=exec_idx)
                    while nxt is None:  # prefetch not back yet -> unavoidable stall
                        time.sleep(dt)
                        nxt = fetcher.take()
                chunk = nxt.chunk
                exec_idx = min(max(0, nxt.next_exec_idx), len(chunk) - 1)
                steps_since_swap = 0
                print(
                    f"step={step} swap chunk start_idx={exec_idx} "
                    f"delay_steps={fetcher.delay_steps} infer={fetcher.last_infer_ms:.0f}ms"
                )
                action_curve.log_chunk(step=step, source="swap_chunk", chunk=chunk, start_idx=exec_idx, horizon=eh)

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
        action_curve.close()
        cameras.close()
        robot.close()


if __name__ == "__main__":
    main(tyro.cli(Args))

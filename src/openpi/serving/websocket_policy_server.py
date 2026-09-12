import asyncio
from datetime import datetime, timezone
import http
import io
import json
import logging
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        capture_dir: str | Path | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._capture_dir = Path(capture_dir).expanduser().resolve() if capture_dir else None
        self._capture_index = 0
        if self._capture_dir is not None:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Capturing decoded policy requests in: %s", self._capture_dir)
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                wire_request = await websocket.recv()
                if not isinstance(wire_request, bytes):
                    raise TypeError(
                        f"Expected a binary websocket request, got {type(wire_request).__name__}"
                    )
                received_wall_ns = time.time_ns()
                received_monotonic_ns = time.monotonic_ns()
                obs = msgpack_numpy.unpackb(wire_request)
                obs = _decode_jpeg_observation(obs)

                infer_started_wall_ns = time.time_ns()
                infer_time = time.monotonic()
                # bci-piper jpeg_v1 compatibility
                wire_format = obs.pop("_openpi_image_wire_format", "raw_rgb")
                if wire_format == "jpeg_v1":
                    import io
                    import numpy as np
                    from PIL import Image
                    for jpeg_key, image_key in (
                        ("observation/image_jpeg", "observation/image"),
                        ("observation/wrist_image_jpeg", "observation/wrist_image"),
                    ):
                        payload = obs.pop(jpeg_key, None)
                        if not isinstance(payload, (bytes, bytearray, memoryview)):
                            raise ValueError(f"jpeg_v1 request missing byte payload {jpeg_key!r}")
                        with Image.open(io.BytesIO(bytes(payload))) as image:
                            obs[image_key] = np.asarray(
                                image.convert("RGB"), dtype=np.uint8
                            ).copy()
                elif wire_format != "raw_rgb":
                    raise ValueError(f"unsupported image wire format: {wire_format!r}")

                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time
                infer_finished_wall_ns = time.time_ns()

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                wire_response = packer.pack(action)
                if self._capture_dir is not None:
                    try:
                        self._capture_request(
                            obs=obs,
                            action=action,
                            wire_request=wire_request,
                            wire_response=wire_response,
                            remote_address=websocket.remote_address,
                            received_wall_ns=received_wall_ns,
                            received_monotonic_ns=received_monotonic_ns,
                            infer_started_wall_ns=infer_started_wall_ns,
                            infer_finished_wall_ns=infer_finished_wall_ns,
                        )
                    except Exception:
                        logger.exception("Failed to capture policy request; inference response will still be sent")
                await websocket.send(wire_response)
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _capture_request(
        self,
        *,
        obs: dict,
        action: dict,
        wire_request: bytes,
        wire_response: bytes,
        remote_address,
        received_wall_ns: int,
        received_monotonic_ns: int,
        infer_started_wall_ns: int,
        infer_finished_wall_ns: int,
    ) -> None:
        """Write one self-contained, lossless capture after JPEG decoding and inference."""
        assert self._capture_dir is not None
        capture_index = self._capture_index
        self._capture_index += 1
        timestamp = datetime.fromtimestamp(received_wall_ns / 1e9, tz=timezone.utc)
        capture_name = f"request_{capture_index:06d}_{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}"
        capture_path = self._capture_dir / capture_name
        capture_path.mkdir()

        (capture_path / "request.msgpack").write_bytes(wire_request)
        (capture_path / "response.msgpack").write_bytes(wire_response)

        arrays = {}
        array_metadata = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
                array_metadata[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                arrays[f"output/{key}"] = value
                array_metadata[f"output/{key}"] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
        np.savez(capture_path / "arrays.npz", **arrays)

        image_files = {}
        for key, filename in (
            ("observation/image", "external_decoded.png"),
            ("observation/wrist_image", "wrist_decoded.png"),
        ):
            image = obs.get(key)
            if isinstance(image, np.ndarray):
                Image.fromarray(image).save(capture_path / filename, format="PNG")
                image_files[key] = filename

        prompt = obs.get("prompt")
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8", errors="replace")
        (capture_path / "prompt.txt").write_text(str(prompt or ""), encoding="utf-8")
        metadata = {
            "schema_version": 1,
            "capture_index": capture_index,
            "remote_address": list(remote_address) if remote_address is not None else None,
            "received_at_utc": timestamp.isoformat(),
            "received_wall_ns": received_wall_ns,
            "received_monotonic_ns": received_monotonic_ns,
            "infer_started_wall_ns": infer_started_wall_ns,
            "infer_finished_wall_ns": infer_finished_wall_ns,
            "infer_duration_ms": (infer_finished_wall_ns - infer_started_wall_ns) / 1e6,
            "prompt": prompt,
            "arrays": array_metadata,
            "images": image_files,
            "observation_keys": sorted(obs),
            "output_keys": sorted(action),
        }
        (capture_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Captured policy request %d: %s", capture_index, capture_path)


def _decode_jpeg_observation(obs):
    """Decode the versioned compressed image wire format before policy transforms.

    Requests without the marker retain the original raw-RGB protocol.  A marked
    request is validated strictly so a mixed Windows/Linux deployment fails
    clearly instead of feeding malformed observations to the policy.
    """
    if not isinstance(obs, dict) or "_openpi_image_wire_format" not in obs:
        return obs
    wire = obs.pop("_openpi_image_wire_format")
    if wire != "jpeg_v1":
        raise ValueError(f"Unsupported OpenPI image wire format: {wire!r}")
    pairs = (
        ("observation/image_jpeg", "observation/image"),
        ("observation/wrist_image_jpeg", "observation/wrist_image"),
    )
    for encoded_key, image_key in pairs:
        payload = obs.pop(encoded_key, None)
        if not isinstance(payload, (bytes, bytearray)):
            raise ValueError(f"jpeg_v1 request missing byte payload {encoded_key!r}")
        with Image.open(io.BytesIO(payload)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"decoded {encoded_key} must be HWC RGB, got {rgb.shape}")
        obs[image_key] = np.ascontiguousarray(rgb)
    return obs


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None

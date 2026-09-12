import io
import json

import numpy as np
from PIL import Image

from openpi.serving.websocket_policy_server import WebsocketPolicyServer, _decode_jpeg_observation


def _jpeg(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=91)
    return buffer.getvalue()


def test_jpeg_v1_decodes_before_policy_and_preserves_other_fields():
    source = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(12, 10, 3)
    payload = _jpeg(source)
    expected = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.uint8)
    result = _decode_jpeg_observation({
        "_openpi_image_wire_format": "jpeg_v1",
        "observation/image_jpeg": payload,
        "observation/wrist_image_jpeg": payload,
        "prompt": "pick",
    })
    np.testing.assert_array_equal(result["observation/image"], expected)
    np.testing.assert_array_equal(result["observation/wrist_image"], expected)
    assert result["prompt"] == "pick"


def test_legacy_raw_rgb_request_is_unchanged():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    request = {"observation/image": image}
    assert _decode_jpeg_observation(request) is request


def test_capture_request_writes_replayable_decoded_inputs(tmp_path):
    server = WebsocketPolicyServer(policy=None, capture_dir=tmp_path)
    external = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    wrist = np.flip(external, axis=1).copy()
    state = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)

    server._capture_request(
        obs={
            "observation/image": external,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": "pour water",
        },
        action={"actions": actions, "server_timing": {"infer_ms": 12.5}},
        wire_request=b"request bytes",
        wire_response=b"response bytes",
        remote_address=("127.0.0.1", 12345),
        received_wall_ns=1_700_000_000_000_000_000,
        received_monotonic_ns=123,
        infer_started_wall_ns=1_700_000_000_001_000_000,
        infer_finished_wall_ns=1_700_000_000_013_500_000,
    )

    capture_path = next(tmp_path.iterdir())
    with np.load(capture_path / "arrays.npz") as captured:
        np.testing.assert_array_equal(captured["observation/image"], external)
        np.testing.assert_array_equal(captured["observation/wrist_image"], wrist)
        np.testing.assert_array_equal(captured["observation/state"], state)
        np.testing.assert_array_equal(captured["output/actions"], actions)
    np.testing.assert_array_equal(np.asarray(Image.open(capture_path / "external_decoded.png")), external)
    assert (capture_path / "prompt.txt").read_text(encoding="utf-8") == "pour water"
    assert (capture_path / "request.msgpack").read_bytes() == b"request bytes"
    assert (capture_path / "response.msgpack").read_bytes() == b"response bytes"
    metadata = json.loads((capture_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["infer_duration_ms"] == 12.5
    assert metadata["arrays"]["observation/state"] == {"shape": [7], "dtype": "float32"}

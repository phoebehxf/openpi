import numpy as np
import pytest

from openpi.models import lora_adapters


def test_save_and_apply_adapter(tmp_path):
    trained = {
        "block": {
            "kernel": np.array([1.0, 2.0], dtype=np.float32),
            "lora_a": np.array([3.0, 4.0], dtype=np.float32),
        },
        "lora_b": np.array([5.0], dtype=np.float32),
    }
    base = {
        "block": {
            "kernel": np.array([-1.0, -2.0], dtype=np.float32),
            "lora_a": np.zeros(2, dtype=np.float32),
        },
        "lora_b": np.zeros(1, dtype=np.float32),
    }

    lora_adapters.save_adapter(trained, tmp_path, source_checkpoint="checkpoint/5000")
    merged = lora_adapters.apply_adapter(base, tmp_path)

    np.testing.assert_array_equal(merged["block"]["kernel"], base["block"]["kernel"])
    np.testing.assert_array_equal(merged["block"]["lora_a"], trained["block"]["lora_a"])
    np.testing.assert_array_equal(merged["lora_b"], trained["lora_b"])


def test_apply_adapter_rejects_non_lora_parameter(tmp_path):
    np.savez(tmp_path / lora_adapters.ADAPTER_FILENAME, kernel=np.ones(1))

    with pytest.raises(ValueError, match="non-LoRA"):
        lora_adapters.apply_adapter({"kernel": np.zeros(1)}, tmp_path)


def test_apply_adapter_rejects_incompatible_shape(tmp_path):
    np.savez(tmp_path / lora_adapters.ADAPTER_FILENAME, lora_a=np.ones(2))

    with pytest.raises(ValueError, match="Shape mismatch"):
        lora_adapters.apply_adapter({"lora_a": np.zeros(1)}, tmp_path)


def test_apply_adapter_rejects_missing_lora_parameter(tmp_path):
    np.savez(tmp_path / lora_adapters.ADAPTER_FILENAME, lora_a=np.ones(1))

    with pytest.raises(ValueError, match="missing 1 LoRA"):
        lora_adapters.apply_adapter({"lora_a": np.zeros(1), "block": {"lora_b": np.zeros(1)}}, tmp_path)

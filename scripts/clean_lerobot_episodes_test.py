import numpy as np
import pyarrow as pa

from scripts import clean_lerobot_episodes as cleaner


def test_choose_trim_bounds_removes_only_endpoint_plateaus():
    home = np.zeros((20, 7), dtype=np.float32)
    motion = np.zeros((50, 7), dtype=np.float32)
    motion[:, 1] = np.linspace(0.0, 1.0, len(motion))
    terminal = np.repeat(motion[-1:], 25, axis=0)
    actions = np.concatenate([home, motion, terminal])

    start, stop = cleaner.choose_trim_bounds(
        actions,
        motion_threshold_rad=0.02,
        gripper_threshold=0.1,
        sustain_frames=3,
        terminal_reference_frames=10,
        terminal_postroll_frames=5,
        min_episode_frames=30,
    )

    assert start == 21
    assert stop == 74


def test_rebuild_episode_table_resets_all_indices():
    table = pa.table(
        {
            "observation.state": [[float(i)] * 7 for i in range(10)],
            "action": [[float(i)] * 7 for i in range(10)],
            "timestamp": np.arange(10, dtype=np.float32) / 30,
            "frame_index": np.arange(10, dtype=np.int64),
            "episode_index": np.zeros(10, dtype=np.int64),
            "index": np.arange(10, dtype=np.int64),
            "task_index": np.zeros(10, dtype=np.int64),
        }
    )

    result = cleaner.rebuild_episode_table(table, 3, 8, episode_index=4, global_start=20, fps=30)

    assert result["frame_index"].to_pylist() == [0, 1, 2, 3, 4]
    assert result["episode_index"].to_pylist() == [4] * 5
    assert result["index"].to_pylist() == [20, 21, 22, 23, 24]
    np.testing.assert_allclose(result["timestamp"].to_numpy(), np.arange(5) / 30)

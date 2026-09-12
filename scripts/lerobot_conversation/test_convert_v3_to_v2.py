from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from convert_v3_to_v2 import load_episode_records


def _write(path: Path, episodes: list[int], file_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": episodes,
                "length": [10] * len(episodes),
                "data/chunk_index": [0] * len(episodes),
                "data/file_index": [file_index] * len(episodes),
                "dataset_from_index": [episode * 10 for episode in episodes],
                "dataset_to_index": [(episode + 1) * 10 for episode in episodes],
            }
        ),
        path,
    )


class LoadEpisodeRecordsTest(unittest.TestCase):
    def test_loads_all_episode_metadata_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / "meta/episodes/chunk-000/file-000.parquet", list(range(10)), 0)
            _write(root / "meta/episodes/chunk-000/file-001.parquet", list(range(10, 20)), 1)

            with mock.patch("convert_v3_to_v2.EPISODES_DIR", "meta/episodes"):
                records = load_episode_records(root)

            self.assertEqual([record["episode_index"] for record in records], list(range(20)))


if __name__ == "__main__":
    unittest.main()

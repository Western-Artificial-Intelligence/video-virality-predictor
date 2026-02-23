import tempfile
import unittest
from pathlib import Path

from Data.common.horizon_delta import DeltaItem
from Data.common.stage_state import StageStateDB, compute_stage_delta


class StageStateTests(unittest.TestCase):
    def test_compute_delta_respects_success_terminal_status(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db = StageStateDB(Path(tmp_dir) / "state.sqlite")
            item = DeltaItem(
                video_id="abc123xyz00",
                video_url="https://youtube.com/watch?v=abc123xyz00",
                captured_at="2026-01-01T00:00:00+00:00",
                source_hash="h1",
                row={},
            )
            delta = compute_stage_delta([item], db, terminal_statuses=("success",))
            self.assertEqual(len(delta), 1)

            db.upsert(item.video_id, item.source_hash, status="success")
            delta = compute_stage_delta([item], db, terminal_statuses=("success",))
            self.assertEqual(len(delta), 0)

            db.upsert(item.video_id, item.source_hash, status="fail")
            delta = compute_stage_delta([item], db, terminal_statuses=("success",))
            self.assertEqual(len(delta), 1)
            db.close()


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Super_Predict.train_suite_from_horizon import get_feature_columns, load_cluster_frame


class TrainSuiteFromHorizonTests(unittest.TestCase):
    def test_load_cluster_frame_dedups_by_latest_capture(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cluster_csv = Path(tmp_dir) / "cluster_results.csv"
            pd.DataFrame(
                [
                    {"video_id": "vid1", "cluster": 1, "captured_at": "2026-01-02T00:00:00+00:00"},
                    {"video_id": "vid1", "cluster": 4, "captured_at": "2026-01-05T00:00:00+00:00"},
                    {"video_id": "vid2", "cluster_id": 2, "captured_at": "2026-01-04T00:00:00+00:00"},
                ]
            ).to_csv(cluster_csv, index=False)

            out = load_cluster_frame(cluster_csv)
            self.assertEqual(len(out), 2)
            row_vid1 = out[out["video_id"] == "vid1"].iloc[0]
            self.assertEqual(int(row_vid1["cluster_id"]), 4)
            self.assertEqual(int(row_vid1["cluster"]), 4)

    def test_get_feature_columns_marks_cluster_fields_as_categorical(self):
        df = pd.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "target_log": [1.0, 1.5],
                "target_raw": [1.7, 3.5],
                "cluster_id": pd.Series([0, 1], dtype="Int64"),
                "cluster": pd.Series([0, 1], dtype="Int64"),
                "fusion_strategy": ["concat", "concat"],
                "k_selected": pd.Series([8, 8], dtype="Int64"),
                "channel_country": ["US", "CA"],
                "duration_seconds": [10.0, 20.0],
            }
        )

        numeric_cols, categorical_cols, dropped_cols = get_feature_columns(df)
        self.assertIn("cluster_id", categorical_cols)
        self.assertIn("cluster", categorical_cols)
        self.assertIn("fusion_strategy", categorical_cols)
        self.assertIn("k_selected", categorical_cols)
        self.assertIn("duration_seconds", numeric_cols)
        self.assertEqual(dropped_cols, [])


if __name__ == "__main__":
    unittest.main()

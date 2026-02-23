import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Super_Predict.train_from_horizon import FEATURE_WHITELIST, build_training_features


class TrainFromHorizonTests(unittest.TestCase):
    def test_build_training_features_filters_horizon_and_whitelist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "meta.csv"
            df = pd.DataFrame(
                [
                    {
                        "video_id": "vid1",
                        "horizon_days": 7,
                        "horizon_view_count": 100,
                        "captured_at": "2026-01-02T00:00:00+00:00",
                        "published_at": "2026-01-01T10:00:00+00:00",
                        "title_length": 10,
                        "channel_country": "US",
                        "view_count": 99999,  # leakage feature, should not survive
                    },
                    {
                        "video_id": "vid1",
                        "horizon_days": 7,
                        "horizon_view_count": 200,
                        "captured_at": "2026-01-03T00:00:00+00:00",
                        "published_at": "2026-01-01T10:00:00+00:00",
                        "title_length": 11,
                        "channel_country": "US",
                    },
                    {
                        "video_id": "vid2",
                        "horizon_days": 30,
                        "horizon_view_count": 300,
                        "captured_at": "2026-01-04T00:00:00+00:00",
                        "published_at": "2026-01-01T09:00:00+00:00",
                        "title_length": 12,
                        "channel_country": "CA",
                    },
                ]
            )
            df.to_csv(csv_path, index=False)

            target_col = "log_view_count_h7d"
            out = build_training_features(csv_path, target_horizon_days=7, target_col=target_col)
            self.assertEqual(len(out), 1)
            self.assertIn("video_id", out.columns)
            self.assertIn(target_col, out.columns)
            self.assertIn("published_hour", out.columns)
            self.assertIn("published_dayofweek", out.columns)
            self.assertIn("title_length", out.columns)
            self.assertNotIn("view_count", out.columns)
            for c in out.columns:
                self.assertTrue(c in {"video_id", target_col} or c in FEATURE_WHITELIST)


if __name__ == "__main__":
    unittest.main()

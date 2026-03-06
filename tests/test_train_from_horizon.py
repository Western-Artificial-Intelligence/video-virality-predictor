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

    def test_build_training_features_joins_cluster_csv_by_video_id(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "meta.csv"
            cluster_csv = Path(tmp_dir) / "cluster_results.csv"

            meta_df = pd.DataFrame(
                [
                    {
                        "video_id": "vid1",
                        "horizon_days": 7,
                        "horizon_view_count": 100,
                        "captured_at": "2026-01-02T00:00:00+00:00",
                        "published_at": "2026-01-01T10:00:00+00:00",
                        "title_length": 10,
                    },
                    {
                        "video_id": "vid2",
                        "horizon_days": 7,
                        "horizon_view_count": 200,
                        "captured_at": "2026-01-03T00:00:00+00:00",
                        "published_at": "2026-01-01T11:00:00+00:00",
                        "title_length": 11,
                    },
                ]
            )
            meta_df.to_csv(csv_path, index=False)

            cluster_df = pd.DataFrame(
                [
                    {"video_id": "vid1", "cluster_id": 1, "captured_at": "2026-01-01T00:00:00+00:00"},
                    {"video_id": "vid1", "cluster_id": 3, "captured_at": "2026-01-05T00:00:00+00:00"},
                    {"video_id": "vid3", "cluster_id": 2, "captured_at": "2026-01-04T00:00:00+00:00"},
                ]
            )
            cluster_df.to_csv(cluster_csv, index=False)

            target_col = "log_view_count_h7d"
            out = build_training_features(
                csv_path,
                target_horizon_days=7,
                target_col=target_col,
                cluster_csv=cluster_csv,
            )
            self.assertIn("cluster_id", out.columns)
            self.assertIn("cluster", out.columns)

            row_vid1 = out[out["video_id"] == "vid1"].iloc[0]
            row_vid2 = out[out["video_id"] == "vid2"].iloc[0]
            self.assertEqual(int(row_vid1["cluster_id"]), 3)
            self.assertEqual(int(row_vid1["cluster"]), 3)
            self.assertTrue(pd.isna(row_vid2["cluster_id"]))


if __name__ == "__main__":
    unittest.main()

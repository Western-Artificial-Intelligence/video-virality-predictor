import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Interpretation.build_interpretation import build_interpretation_rows
from Interpretation.combine_cluster_and_links import build_video_url_map, join_cluster_with_urls


class InterpretationRefactorTests(unittest.TestCase):
    def test_build_video_url_map_supports_video_url_and_url_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            for col in ("video_url", "url"):
                csv_path = tmp_dir / f"meta_{col}.csv"
                rows = [
                    {
                        "video_id": "abc123def45",
                        col: "https://www.youtube.com/watch?v=abc123def45",
                        "captured_at": "2026-01-01T00:00:00+00:00",
                    },
                    {
                        "video_id": "abc123def45",
                        col: "https://www.youtube.com/watch?v=abc123def45_new",
                        "captured_at": "2026-01-02T00:00:00+00:00",
                    },
                    {
                        "video_id": "zzz987yyy65",
                        col: "https://www.youtube.com/watch?v=zzz987yyy65",
                        "captured_at": "2026-01-03T00:00:00+00:00",
                    },
                ]

                fieldnames = ["video_id", col, "captured_at"]
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

                mapping = build_video_url_map(csv_path)
                self.assertIn("abc123def45", mapping)
                self.assertIn("zzz987yyy65", mapping)
                self.assertEqual(mapping["abc123def45"], "https://www.youtube.com/watch?v=abc123def45_new")

    def test_join_cluster_with_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            clusters_csv = tmp_dir / "cluster_results.csv"
            out_csv = tmp_dir / "cluster_links.csv"

            pd.DataFrame(
                [
                    {"video_id": "bbb222ccc33", "cluster": 1, "cluster_id": 1},
                    {"video_id": "aaa111bbb22", "cluster": 0, "cluster_id": 0},
                ]
            ).to_csv(clusters_csv, index=False)

            rows = join_cluster_with_urls(
                clusters_csv=clusters_csv,
                video_id_to_url={
                    "aaa111bbb22": "https://www.youtube.com/watch?v=aaa111bbb22",
                    "bbb222ccc33": "https://www.youtube.com/watch?v=bbb222ccc33",
                },
                out_csv=out_csv,
            )

            self.assertEqual(len(rows), 2)
            self.assertTrue(out_csv.exists())
            self.assertEqual(rows[0]["video_id"], "aaa111bbb22")
            self.assertIn("url", rows[0])

    def test_missing_media_rows_are_kept_with_zero_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            video_dir = tmp_dir / "videos"
            audio_dir = tmp_dir / "audio"
            video_dir.mkdir(parents=True, exist_ok=True)
            audio_dir.mkdir(parents=True, exist_ok=True)

            clusters = pd.DataFrame(
                [
                    {
                        "video_id": "abc123def45",
                        "cluster": 2,
                        "cluster_id": 2,
                        "fusion_strategy": "concat",
                        "k_selected": 10,
                    }
                ]
            )

            rows, stats = build_interpretation_rows(
                clusters=clusters,
                video_id_to_url={"abc123def45": "https://www.youtube.com/watch?v=abc123def45"},
                video_dir=video_dir,
                audio_dir=audio_dir,
                s3=None,
                raw_prefix="clipfarm/raw",
                fetch_missing=False,
                fps_sample=2,
                diff_thresh=25.0,
                edge_thresh=20.0,
            )

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["video_id"], "abc123def45")
            self.assertEqual(int(row["video_available"]), 0)
            self.assertEqual(int(row["audio_available"]), 0)
            self.assertEqual(float(row["motion_mean"]), 0.0)
            self.assertEqual(float(row["cut_rate_per_min"]), 0.0)
            self.assertEqual(float(row["audio_rms_mean"]), 0.0)
            self.assertEqual(float(row["audio_rms_std"]), 0.0)
            self.assertEqual(float(row["visual_density"]), 0.0)
            self.assertEqual(stats["video_missing"], 1)
            self.assertEqual(stats["audio_missing"], 1)

    def test_end_to_end_smoke_interpretation_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            clusters = pd.DataFrame(
                [
                    {"video_id": "aaa111bbb22", "cluster": 0, "cluster_id": 0, "fusion_strategy": "concat", "k_selected": 2},
                    {"video_id": "bbb222ccc33", "cluster": 1, "cluster_id": 1, "fusion_strategy": "concat", "k_selected": 2},
                ]
            )

            rows, _stats = build_interpretation_rows(
                clusters=clusters,
                video_id_to_url={
                    "aaa111bbb22": "https://www.youtube.com/watch?v=aaa111bbb22",
                    "bbb222ccc33": "https://www.youtube.com/watch?v=bbb222ccc33",
                },
                video_dir=tmp_dir / "videos",
                audio_dir=tmp_dir / "audio",
                s3=None,
                raw_prefix="clipfarm/raw",
                fetch_missing=False,
                fps_sample=2,
                diff_thresh=25.0,
                edge_thresh=20.0,
            )

            out_csv = tmp_dir / "interpretation.csv"
            pd.DataFrame(rows).to_csv(out_csv, index=False)

            self.assertTrue(out_csv.exists())
            out_df = pd.read_csv(out_csv)
            self.assertEqual(len(out_df), 2)
            self.assertTrue(
                {
                    "video_id",
                    "cluster",
                    "cluster_id",
                    "fusion_strategy",
                    "k_selected",
                    "url",
                    "video_available",
                    "audio_available",
                    "motion_mean",
                    "cut_rate_per_min",
                    "audio_rms_mean",
                    "audio_rms_std",
                    "visual_density",
                }.issubset(out_df.columns)
            )


if __name__ == "__main__":
    unittest.main()

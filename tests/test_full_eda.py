import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


class FullEDATests(unittest.TestCase):
    def test_run_full_eda_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            metadata_csv = tmp_dir / "metadata.csv"
            cluster_csv = tmp_dir / "cluster.csv"
            interpretation_csv = tmp_dir / "interpretation.csv"
            out_dir = tmp_dir / "eda_out"

            pd.DataFrame(
                [
                    {
                        "video_id": "aaa111bbb22",
                        "url": "https://www.youtube.com/watch?v=aaa111bbb22",
                        "query": "shorts",
                        "category_type": "seed",
                        "captured_at": "2026-01-01T00:00:00+00:00",
                        "published_at": "2025-12-30T00:00:00+00:00",
                        "view_count": 1000,
                        "like_count": 100,
                        "comment_count": 10,
                        "duration_seconds": 30,
                        "horizon_days": 7,
                        "horizon_view_count": 1200,
                        "virality_score": 0.5,
                        "likes_per_view": 0.1,
                        "comments_per_view": 0.01,
                    },
                    {
                        "video_id": "bbb222ccc33",
                        "url": "https://www.youtube.com/watch?v=bbb222ccc33",
                        "query": "gaming",
                        "category_type": "seed",
                        "captured_at": "2026-01-02T00:00:00+00:00",
                        "published_at": "2025-12-29T00:00:00+00:00",
                        "view_count": 2000,
                        "like_count": 180,
                        "comment_count": 20,
                        "duration_seconds": 25,
                        "horizon_days": 30,
                        "horizon_view_count": 2800,
                        "virality_score": 1.2,
                        "likes_per_view": 0.09,
                        "comments_per_view": 0.01,
                    },
                ]
            ).to_csv(metadata_csv, index=False)

            pd.DataFrame(
                [
                    {"video_id": "aaa111bbb22", "cluster": 0, "cluster_id": 0, "fusion_strategy": "concat", "k_selected": 2},
                    {"video_id": "bbb222ccc33", "cluster": 1, "cluster_id": 1, "fusion_strategy": "concat", "k_selected": 2},
                ]
            ).to_csv(cluster_csv, index=False)

            pd.DataFrame(
                [
                    {
                        "video_id": "aaa111bbb22",
                        "cluster": 0,
                        "cluster_id": 0,
                        "fusion_strategy": "concat",
                        "k_selected": 2,
                        "url": "https://www.youtube.com/watch?v=aaa111bbb22",
                        "video_available": 1,
                        "audio_available": 1,
                        "motion_mean": 1.0,
                        "cut_rate_per_min": 2.0,
                        "audio_rms_mean": 0.5,
                        "audio_rms_std": 0.1,
                        "visual_density": 0.2,
                    },
                    {
                        "video_id": "bbb222ccc33",
                        "cluster": 1,
                        "cluster_id": 1,
                        "fusion_strategy": "concat",
                        "k_selected": 2,
                        "url": "https://www.youtube.com/watch?v=bbb222ccc33",
                        "video_available": 0,
                        "audio_available": 1,
                        "motion_mean": 0.0,
                        "cut_rate_per_min": 0.0,
                        "audio_rms_mean": 0.4,
                        "audio_rms_std": 0.2,
                        "visual_density": 0.0,
                    },
                ]
            ).to_csv(interpretation_csv, index=False)

            cmd = [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                str(REPO_ROOT / "scripts" / "run_full_eda.py"),
                "--metadata_csv",
                str(metadata_csv),
                "--cluster_csv",
                str(cluster_csv),
                "--interpretation_csv",
                str(interpretation_csv),
                "--out_dir",
                str(out_dir),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")

            self.assertTrue((out_dir / "eda_report.md").exists())
            self.assertTrue((out_dir / "overview.json").exists())
            self.assertTrue((out_dir / "tables" / "missingness.csv").exists())
            self.assertTrue((out_dir / "tables" / "numeric_summary.csv").exists())
            self.assertTrue((out_dir / "plots" / "missingness_top20.png").exists())


if __name__ == "__main__":
    unittest.main()

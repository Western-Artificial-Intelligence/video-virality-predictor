import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Unsup_Cluster.cluster import (
    StrategyDataset,
    _build_reduced_space,
    _candidate_sort_key,
    _canonicalize_labels,
    _evaluate_strategy_candidates,
    _load_strategy_dataset,
    run_clustering,
)


class FakeS3:
    def __init__(self, key_to_local: dict[str, Path]):
        self.key_to_local = {k: Path(v) for k, v in key_to_local.items()}

    def exists(self, key: str) -> bool:
        return key in self.key_to_local and self.key_to_local[key].exists()

    def download_file(self, key: str, local_path: Path) -> None:
        if key not in self.key_to_local:
            raise FileNotFoundError(key)
        src = self.key_to_local[key]
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() == local.resolve():
            return
        shutil.copyfile(src, local)


class ClusterRefactorTests(unittest.TestCase):
    def test_load_strategy_dataset_dedupes_latest_video_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            shard_a = tmp_dir / "part-0001.npz"
            np.savez(shard_a, vectors=np.asarray([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32))
            shard_b = tmp_dir / "part-0002.npz"
            np.savez(shard_b, vectors=np.asarray([[3.0, 3.0]], dtype=np.float32))

            manifest = pd.DataFrame(
                [
                    {
                        "video_id": "vidA",
                        "source_hash": "h_old",
                        "captured_at": "2026-01-01T00:00:00+00:00",
                        "fused_key": "clipfarm/fused/concat/shards/date=2026-01-01/part-0001.npz",
                        "shard_idx": 0,
                        "fusion_status": "success_full",
                    },
                    {
                        "video_id": "vidA",
                        "source_hash": "h_new",
                        "captured_at": "2026-01-02T00:00:00+00:00",
                        "fused_key": "clipfarm/fused/concat/shards/date=2026-01-02/part-0002.npz",
                        "shard_idx": 0,
                        "fusion_status": "success_full",
                    },
                    {
                        "video_id": "vidB",
                        "source_hash": "h_b",
                        "captured_at": "2026-01-01T00:00:00+00:00",
                        "fused_key": "clipfarm/fused/concat/shards/date=2026-01-01/part-0001.npz",
                        "shard_idx": 1,
                        "fusion_status": "success_text_placeholder",
                    },
                    {
                        "video_id": "vidC",
                        "source_hash": "h_c",
                        "captured_at": "2026-01-01T00:00:00+00:00",
                        "fused_key": "clipfarm/fused/concat/shards/date=2026-01-01/part-0001.npz",
                        "shard_idx": 0,
                        "fusion_status": "fail_terminal",
                    },
                ]
            )
            manifest_path = tmp_dir / "fused_manifest.parquet"
            manifest.to_parquet(manifest_path, index=False)

            fake_s3 = FakeS3(
                {
                    "clipfarm/fused/concat/fused_manifest.parquet": manifest_path,
                    "clipfarm/fused/concat/shards/date=2026-01-01/part-0001.npz": shard_a,
                    "clipfarm/fused/concat/shards/date=2026-01-02/part-0002.npz": shard_b,
                }
            )

            ds = _load_strategy_dataset(
                s3=fake_s3,
                strategy="concat",
                fused_prefix_base="clipfarm/fused",
                statuses={"success_full", "success_text_placeholder"},
                metadata_video_ids=None,
                tmp_dir=tmp_dir,
            )

            self.assertEqual(ds.video_ids, ["vidA", "vidB"])
            np.testing.assert_allclose(ds.vectors, np.asarray([[3.0, 3.0], [2.0, 2.0]], dtype=np.float32))
            self.assertEqual(ds.manifest_rows_raw, 4)
            self.assertEqual(ds.manifest_rows_filtered, 3)

    def test_canonicalize_labels_is_deterministic(self):
        labels = np.asarray([10, 10, 20, 20, 30, 30])
        features = np.asarray(
            [
                [5.0, 0.0],
                [5.2, 0.1],
                [0.0, 5.0],
                [0.1, 5.3],
                [-5.0, -5.0],
                [-5.2, -4.9],
            ],
            dtype=np.float32,
        )

        c1 = _canonicalize_labels(labels, features)
        c2 = _canonicalize_labels(labels, features)

        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(c1, np.asarray([2, 2, 1, 1, 0, 0]))

    def test_auto_selection_prefers_clear_three_cluster_structure(self):
        rng = np.random.default_rng(7)
        c1 = rng.normal(loc=[-4.0, -4.0], scale=0.2, size=(40, 2))
        c2 = rng.normal(loc=[0.0, 4.0], scale=0.2, size=(40, 2))
        c3 = rng.normal(loc=[4.0, -2.0], scale=0.2, size=(40, 2))
        x = np.vstack([c1, c2, c3]).astype(np.float32)

        ds = StrategyDataset(
            strategy="concat",
            video_ids=[f"vid{i:03d}" for i in range(x.shape[0])],
            vectors=x,
            manifest_rows_raw=x.shape[0],
            manifest_rows_filtered=x.shape[0],
        )
        reduced = _build_reduced_space(
            dataset=ds,
            random_seed=42,
            enable_umap_cluster=False,
            umap_cluster_dim=15,
            umap_viz_neighbors=15,
        )

        candidates = _evaluate_strategy_candidates(
            reduced=reduced,
            k_values=[2, 3, 4, 5],
            random_seeds=[11, 23, 37],
            min_cluster_fraction=0.01,
            max_silhouette_samples=0,
        )
        best = max(candidates, key=_candidate_sort_key)
        self.assertEqual(int(best["k"]), 3)

    def test_end_to_end_smoke_generates_cluster_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            out_csv = tmp_dir / "cluster_results.csv"
            diag_json = tmp_dir / "cluster_diagnostics.json"
            emb_dir = tmp_dir / "embeddings"
            plot_dir = tmp_dir / "plots"

            ids = [f"vid{i:03d}" for i in range(20)]
            concat_vectors = np.vstack(
                [
                    np.repeat([[-3.0, -3.0, 0.0, 0.0]], repeats=10, axis=0),
                    np.repeat([[3.0, 3.0, 0.0, 0.0]], repeats=10, axis=0),
                ]
            ).astype(np.float32)
            sum_vectors = np.random.default_rng(99).normal(loc=0.0, scale=1.5, size=(20, 4)).astype(np.float32)

            concat_shard = tmp_dir / "concat_part.npz"
            np.savez(concat_shard, vectors=concat_vectors)
            sum_shard = tmp_dir / "sum_part.npz"
            np.savez(sum_shard, vectors=sum_vectors)

            concat_manifest = pd.DataFrame(
                [
                    {
                        "video_id": vid,
                        "source_hash": f"h_{vid}",
                        "captured_at": "2026-01-02T00:00:00+00:00",
                        "fused_key": "clipfarm/fused/concat/shards/date=2026-01-02/part-0000.npz",
                        "shard_idx": i,
                        "fusion_status": "success_full",
                    }
                    for i, vid in enumerate(ids)
                ]
            )
            sum_manifest = pd.DataFrame(
                [
                    {
                        "video_id": vid,
                        "source_hash": f"h_{vid}",
                        "captured_at": "2026-01-02T00:00:00+00:00",
                        "fused_key": "clipfarm/fused/sum_pool/shards/date=2026-01-02/part-0000.npz",
                        "shard_idx": i,
                        "fusion_status": "success_full",
                    }
                    for i, vid in enumerate(ids)
                ]
            )

            concat_manifest_path = tmp_dir / "concat_manifest.parquet"
            sum_manifest_path = tmp_dir / "sum_manifest.parquet"
            concat_manifest.to_parquet(concat_manifest_path, index=False)
            sum_manifest.to_parquet(sum_manifest_path, index=False)

            fake_s3 = FakeS3(
                {
                    "clipfarm/fused/concat/fused_manifest.parquet": concat_manifest_path,
                    "clipfarm/fused/sum_pool/fused_manifest.parquet": sum_manifest_path,
                    "clipfarm/fused/concat/shards/date=2026-01-02/part-0000.npz": concat_shard,
                    "clipfarm/fused/sum_pool/shards/date=2026-01-02/part-0000.npz": sum_shard,
                }
            )

            payload = run_clustering(
                s3=fake_s3,
                metadata_video_ids=set(ids),
                fused_prefix_base="clipfarm/fused",
                strategies=["concat", "sum_pool"],
                statuses={"success_full", "success_text_placeholder"},
                k_values=[2, 3],
                random_seeds=[13, 23, 37],
                enable_umap_cluster=False,
                umap_cluster_dim=15,
                umap_viz_neighbors=15,
                min_cluster_fraction=0.01,
                max_silhouette_samples=0,
                output_csv=out_csv,
                diagnostics_json=diag_json,
                embeddings_dir=emb_dir,
                plots_dir=plot_dir,
            )

            self.assertTrue(out_csv.exists())
            self.assertTrue(diag_json.exists())
            self.assertTrue((plot_dir / "pca_2d.png").exists())

            out_df = pd.read_csv(out_csv)
            self.assertEqual(len(out_df), len(ids))
            self.assertTrue({"video_id", "cluster", "cluster_id", "fusion_strategy", "k_selected"}.issubset(out_df.columns))
            self.assertTrue((out_df["fusion_strategy"] == "concat").all())
            self.assertEqual(int(payload["row_counts"]["clustered_rows"]), len(ids))


if __name__ == "__main__":
    unittest.main()

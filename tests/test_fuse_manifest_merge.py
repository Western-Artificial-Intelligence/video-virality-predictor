import unittest

import pandas as pd

from Data.common.fuse_embeddings_delta import _merge_manifest


class FuseManifestMergeTests(unittest.TestCase):
    def test_upsert_by_video_id_and_source_hash(self):
        existing = pd.DataFrame(
            [
                {
                    "video_id": "v1",
                    "source_hash": "h1",
                    "captured_at": "2026-01-01T00:00:00+00:00",
                    "fused_key": "old1",
                    "shard_idx": 0,
                }
            ]
        )
        new = pd.DataFrame(
            [
                {
                    "video_id": "v1",
                    "source_hash": "h1",
                    "captured_at": "2026-01-02T00:00:00+00:00",
                    "fused_key": "new1",
                    "shard_idx": 3,
                },
                {
                    "video_id": "v1",
                    "source_hash": "h2",
                    "captured_at": "2026-01-03T00:00:00+00:00",
                    "fused_key": "new2",
                    "shard_idx": 1,
                },
            ]
        )

        merged = _merge_manifest(existing, new)
        self.assertEqual(len(merged), 2)

        row_h1 = merged[(merged["video_id"] == "v1") & (merged["source_hash"] == "h1")].iloc[0]
        row_h2 = merged[(merged["video_id"] == "v1") & (merged["source_hash"] == "h2")].iloc[0]
        self.assertEqual(row_h1["fused_key"], "new1")
        self.assertEqual(int(row_h1["shard_idx"]), 3)
        self.assertEqual(row_h2["fused_key"], "new2")


if __name__ == "__main__":
    unittest.main()

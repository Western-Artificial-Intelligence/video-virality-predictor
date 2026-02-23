import unittest

from Data.common.pinecone_client import build_full_metadata


class PineconeMetadataTests(unittest.TestCase):
    def test_metadata_truncation_and_mandatory_override(self):
        row = {
            "title": "x" * 5000,
            "description": "desc",
            "horizon_days": 30,
        }
        meta = build_full_metadata(
            row=row,
            mandatory={"video_id": "vid1", "modality": "video", "title": "must_win"},
            max_value_len=50,
        )
        self.assertEqual(meta["video_id"], "vid1")
        self.assertEqual(meta["modality"], "video")
        self.assertEqual(meta["title"], "must_win")
        self.assertLessEqual(len(meta["description"]), 50)


if __name__ == "__main__":
    unittest.main()

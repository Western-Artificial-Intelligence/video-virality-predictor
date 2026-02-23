import unittest

from Data.common.s3_artifact_store import S3ArtifactStore

try:  # pragma: no cover
    import boto3  # noqa: F401

    HAS_BOTO3 = True
except Exception:  # pragma: no cover
    HAS_BOTO3 = False


@unittest.skipIf(not HAS_BOTO3, "boto3 not installed")
class S3ArtifactStoreTests(unittest.TestCase):
    def test_key_join(self):
        store = S3ArtifactStore(bucket="dummy-bucket", region="", prefix="clipfarm/raw")
        self.assertEqual(store.key("video/a.mp4"), "clipfarm/raw/video/a.mp4")
        self.assertEqual(store.key("/video/a.mp4"), "clipfarm/raw/video/a.mp4")


if __name__ == "__main__":
    unittest.main()

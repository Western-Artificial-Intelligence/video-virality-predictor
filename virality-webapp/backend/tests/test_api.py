import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import AppSettings


class FakeService:
    def load_models(self) -> None:
        return None

    def get_schema_payload(self):
        return {
            "modes": ["fast", "full"],
            "limits": {"max_upload_mb": 64, "allowed_extensions": [".mp4"]},
            "fields": [
                {"name": "title", "type": "string", "required": False, "default": "", "options": None, "description": ""},
                {"name": "duration_seconds", "type": "number", "required": False, "default": None, "options": None, "description": ""},
            ],
        }

    def predict(self, video_path, mode, metadata):
        assert Path(video_path).exists()
        if mode == "fast":
            return {
                "mode": "fast",
                "predictions_7d": {"prediction_raw": 100.0, "model": "gbdt", "strategy": "concat"},
                "predictions_30d": {"prediction_raw": 300.0, "model": "gbdt", "strategy": "concat"},
                "artifact_provenance": {"7d": [], "30d": []},
                "transcript": {"text_present": 1, "source": "mock", "model": "mock", "language": "en", "error": ""},
            }
        return {
            "mode": "full",
            "predictions_7d": [
                {"prediction_raw": 80.0, "model": "gbdt", "strategy": "concat"},
                {"prediction_raw": 120.0, "model": "ridge", "strategy": "sum_pool"},
            ],
            "predictions_30d": [
                {"prediction_raw": 250.0, "model": "gbdt", "strategy": "concat"},
                {"prediction_raw": 500.0, "model": "ridge", "strategy": "sum_pool"},
            ],
            "range_7d": {"min_raw": 80.0, "max_raw": 120.0, "min_log": 1.0, "max_log": 2.0},
            "range_30d": {"min_raw": 250.0, "max_raw": 500.0, "min_log": 2.0, "max_log": 3.0},
            "artifact_provenance": {"7d": [], "30d": []},
            "transcript": {"text_present": 0, "source": "", "model": "", "language": "", "error": "mock"},
        }


def _settings() -> AppSettings:
    return AppSettings(
        repo_root=Path(__file__).resolve().parents[3],
        model_s3_bucket="",
        model_s3_region="",
        model_snapshot_prefix="clipfarm/models/snapshots",
        model_cache_dir=Path("/tmp/virality-webapp-tests"),
        asr_backend="auto",
        asr_model="small",
        max_upload_mb=64,
        text_dim=768,
        auto_load_models_on_startup=False,
        cors_allow_origins=("*",),
    )


def test_api_predict_fast_success():
    app = create_app(service=FakeService(), settings=_settings(), auto_load_models=False)
    client = TestClient(app)

    response = client.post(
        "/api/predict",
        data={"mode": "fast", "metadata_json": json.dumps({"title": "hello"})},
        files={"video_file": ("clip.mp4", b"fake-mp4", "video/mp4")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "fast"
    assert payload["predictions_7d"]["prediction_raw"] == 100.0


def test_api_predict_full_success():
    app = create_app(service=FakeService(), settings=_settings(), auto_load_models=False)
    client = TestClient(app)

    response = client.post(
        "/api/predict",
        data={"mode": "full", "metadata_json": json.dumps({"title": "hello"})},
        files={"video_file": ("clip.mp4", b"fake-mp4", "video/mp4")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "full"
    assert payload["range_7d"]["min_raw"] == 80.0
    assert len(payload["predictions_30d"]) == 2


def test_api_predict_rejects_non_mp4_upload():
    app = create_app(service=FakeService(), settings=_settings(), auto_load_models=False)
    client = TestClient(app)

    response = client.post(
        "/api/predict",
        data={"mode": "fast", "metadata_json": "{}"},
        files={"video_file": ("clip.mov", b"fake", "video/quicktime")},
    )

    assert response.status_code == 400
    assert "mp4" in response.json()["detail"].lower()


def test_api_predict_rejects_invalid_json():
    app = create_app(service=FakeService(), settings=_settings(), auto_load_models=False)
    client = TestClient(app)

    response = client.post(
        "/api/predict",
        data={"mode": "fast", "metadata_json": "{invalid"},
        files={"video_file": ("clip.mp4", b"fake-mp4", "video/mp4")},
    )

    assert response.status_code == 400
    assert "invalid json" in response.json()["detail"].lower()

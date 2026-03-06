import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import AppSettings


class SmokeService:
    def load_models(self):
        return None

    def get_schema_payload(self):
        return {
            "modes": ["fast", "full"],
            "limits": {"max_upload_mb": 128, "allowed_extensions": [".mp4"]},
            "fields": [
                {"name": "title", "type": "string", "required": False, "default": "", "options": None, "description": ""},
                {"name": "query", "type": "string", "required": False, "default": "", "options": None, "description": ""},
            ],
        }

    def predict(self, video_path, mode, metadata):
        base = {
            "mode": mode,
            "artifact_provenance": {"7d": [], "30d": []},
            "transcript": {"text_present": 1, "source": "mock", "model": "mock", "language": "en", "error": ""},
        }
        if mode == "fast":
            base["predictions_7d"] = {"prediction_raw": 12.0, "prediction_log": 2.0, "model": "gbdt", "strategy": "concat"}
            base["predictions_30d"] = {"prediction_raw": 34.0, "prediction_log": 3.0, "model": "gbdt", "strategy": "concat"}
            return base
        base["predictions_7d"] = [
            {"prediction_raw": 10.0, "prediction_log": 2.0, "model": "gbdt", "strategy": "concat"},
            {"prediction_raw": 18.0, "prediction_log": 2.5, "model": "ridge", "strategy": "sum_pool"},
        ]
        base["predictions_30d"] = [
            {"prediction_raw": 30.0, "prediction_log": 3.0, "model": "gbdt", "strategy": "concat"},
            {"prediction_raw": 50.0, "prediction_log": 3.7, "model": "ridge", "strategy": "sum_pool"},
        ]
        base["range_7d"] = {"min_raw": 10.0, "max_raw": 18.0, "min_log": 2.0, "max_log": 2.5}
        base["range_30d"] = {"min_raw": 30.0, "max_raw": 50.0, "min_log": 3.0, "max_log": 3.7}
        return base


def _settings() -> AppSettings:
    return AppSettings(
        repo_root=Path(__file__).resolve().parents[3],
        model_s3_bucket="",
        model_s3_region="",
        model_snapshot_prefix="clipfarm/models/snapshots",
        model_cache_dir=Path("/tmp/virality-webapp-tests-smoke"),
        asr_backend="auto",
        asr_model="small",
        max_upload_mb=128,
        text_dim=768,
        auto_load_models_on_startup=False,
        cors_allow_origins=("*",),
    )


def test_smoke_end_to_end_with_mocked_service_outputs_non_negative_predictions():
    app = create_app(service=SmokeService(), settings=_settings(), auto_load_models=False)
    client = TestClient(app)

    schema_res = client.get("/api/schema")
    assert schema_res.status_code == 200
    schema = schema_res.json()
    assert schema["modes"] == ["fast", "full"]

    predict_res = client.post(
        "/api/predict",
        data={"mode": "full", "metadata_json": json.dumps({"title": "demo", "query": "sports"})},
        files={"video_file": ("demo.mp4", b"synthetic", "video/mp4")},
    )
    assert predict_res.status_code == 200

    payload = predict_res.json()
    assert payload["mode"] == "full"
    assert payload["range_7d"]["min_raw"] >= 0
    assert payload["range_30d"]["min_raw"] >= 0
    assert len(payload["predictions_7d"]) == 2
    assert len(payload["predictions_30d"]) == 2

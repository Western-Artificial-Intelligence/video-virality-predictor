from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .constants import is_valid_mode
from .service import PredictionService
from .settings import AppSettings, load_settings


class PredictServiceProtocol:
    def load_models(self) -> None: ...

    def get_schema_payload(self) -> dict[str, Any]: ...

    def predict(self, video_path: Path, mode: str, metadata: dict[str, Any]) -> dict[str, Any]: ...


def create_app(
    service: PredictServiceProtocol | None = None,
    settings: AppSettings | None = None,
    auto_load_models: bool | None = None,
) -> FastAPI:
    app_settings = settings or load_settings()
    app_service = service or PredictionService(app_settings)
    startup_load = app_settings.auto_load_models_on_startup if auto_load_models is None else bool(auto_load_models)

    app = FastAPI(title="MP4-to-Virality Predictor", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.cors_allow_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        if startup_load:
            app_service.load_models()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/schema")
    def get_schema() -> dict[str, Any]:
        try:
            return app_service.get_schema_payload()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"schema_load_failed: {exc}") from exc

    @app.post("/api/predict")
    def predict(
        video_file: UploadFile = File(...),
        mode: str = Form(...),
        metadata_json: str = Form("{}"),
    ) -> dict[str, Any]:
        filename = (video_file.filename or "upload.mp4").strip()
        if not filename.lower().endswith(".mp4"):
            raise HTTPException(status_code=400, detail="video_file must be an .mp4")

        if not is_valid_mode(mode):
            raise HTTPException(status_code=400, detail="mode must be one of: fast, full")

        try:
            parsed = json.loads(metadata_json or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"metadata_json is invalid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="metadata_json must decode to a JSON object")

        max_bytes = app_settings.max_upload_mb * 1024 * 1024

        with tempfile.TemporaryDirectory(prefix="virality_predict_") as tmp_dir:
            tmp_path = Path(tmp_dir) / "input.mp4"
            total = 0
            try:
                with tmp_path.open("wb") as fh:
                    while True:
                        chunk = video_file.file.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_bytes:
                            raise HTTPException(
                                status_code=413,
                                detail=f"video_file exceeds max_upload_mb={app_settings.max_upload_mb}",
                            )
                        fh.write(chunk)
            finally:
                video_file.file.close()

            try:
                return app_service.predict(video_path=tmp_path, mode=mode, metadata=parsed)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"prediction_failed: {exc}") from exc

    return app


app = create_app()

from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .settings import AppSettings

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class PipelineResult:
    video_vec: np.ndarray
    audio_vec: np.ndarray
    text_vec: np.ndarray
    text_present: int
    transcript: str
    transcript_meta: dict[str, Any]
    transcript_error: str


class OnlineFeaturePipeline:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._video_embedder: Any | None = None
        self._audio_embedder: Any | None = None
        self._text_embedder: Any | None = None

    def run(self, video_path: Path, metadata: dict[str, Any]) -> PipelineResult:
        from Data.embeddings.text.embed_text_delta import build_meta_text  # local import
        from Data.raw.Text.text_collect import transcribe_audio  # local import

        wav_path = video_path.with_suffix(".wav")
        self.extract_wav_from_video(video_path, wav_path)

        transcript_text = ""
        transcript_meta: dict[str, Any] = {}
        transcript_error = ""

        try:
            transcript_text, transcript_meta = transcribe_audio(
                audio_path=wav_path,
                backend=self.settings.asr_backend,
                model_name=self.settings.asr_model,
            )
        except Exception as exc:
            transcript_text = ""
            transcript_meta = {}
            transcript_error = str(exc)

        video_vec = self._get_video_embedder().embed(video_path, num_frames=16)
        audio_vec = self._get_audio_embedder().embed(
            wav_path,
            sample_rate=16000,
            max_audio_seconds=90.0,
        )

        text_present = 1 if transcript_text.strip() else 0
        if text_present == 1:
            meta_text = build_meta_text(metadata)
            try:
                text_vec = self._get_text_embedder().embed(meta_text, transcript_text)
            except Exception as exc:
                text_vec = np.zeros((self.settings.text_dim,), dtype=np.float32)
                text_present = 0
                if transcript_error:
                    transcript_error = f"{transcript_error} | text_embed_fail:{exc}"
                else:
                    transcript_error = f"text_embed_fail:{exc}"
        else:
            text_vec = np.zeros((self.settings.text_dim,), dtype=np.float32)

        return PipelineResult(
            video_vec=np.asarray(video_vec, dtype=np.float32).reshape(-1),
            audio_vec=np.asarray(audio_vec, dtype=np.float32).reshape(-1),
            text_vec=np.asarray(text_vec, dtype=np.float32).reshape(-1),
            text_present=int(text_present),
            transcript=transcript_text,
            transcript_meta=transcript_meta,
            transcript_error=transcript_error,
        )

    @staticmethod
    def extract_wav_from_video(video_path: Path, wav_path: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ffmpeg failed")

    def _get_video_embedder(self) -> Any:
        with self._lock:
            if self._video_embedder is None:
                from Data.embeddings.video.embed_video_delta import VideoEmbedder  # local import

                self._video_embedder = VideoEmbedder(model_name="MCG-NJU/videomae-base", device="auto")
            return self._video_embedder

    def _get_audio_embedder(self) -> Any:
        with self._lock:
            if self._audio_embedder is None:
                from Data.embeddings.audio.embed_audio_delta import AudioEmbedder  # local import

                self._audio_embedder = AudioEmbedder(model_name="facebook/wav2vec2-base-960h")
            return self._audio_embedder

    def _get_text_embedder(self) -> Any:
        with self._lock:
            if self._text_embedder is None:
                from Data.embeddings.text.embed_text_delta import TextEmbedder  # local import

                self._text_embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
            return self._text_embedder

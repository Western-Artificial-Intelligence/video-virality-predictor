# Raw Text Downloader (Caption + ASR)

This stage writes transcript JSON files for each `video_id` and uploads them to:
- `s3://<bucket>/clipfarm/raw/text/<video_id>.json`

Primary script:
- `Data/raw/Text/text_collect.py`

Local runner:
- `scripts/run_text_pipeline_local.sh`

Colab GPU runner:
- `scripts/run_text_pipeline_colab.py`

## ASR Backends
- `whisper_cpp` (recommended local default)
- `faster_whisper`
- `openai_api`
- `auto` (tries `faster_whisper` -> `whisper_cpp` -> `openai_api`)

`whisper` is kept as a compatibility alias and routes to `whisper_cpp`.

## whisper.cpp Setup (Apple Silicon, CPU)

1. Install binary:
```bash
brew install whisper-cpp
```

2. Download multilingual model (`small` default):
```bash
mkdir -p ~/.cache/whisper.cpp
curl -L \
  -o ~/.cache/whisper.cpp/ggml-small.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
```

3. Export runtime env:
```bash
export WHISPER_CPP_BIN="$(command -v whisper-cli)"
export WHISPER_CPP_MODEL_DIR="$HOME/.cache/whisper.cpp"
export WHISPER_CPP_THREADS=0   # 0 = auto
```

4. Run local text pipeline:
```bash
TEXT_ASR_BACKEND=whisper_cpp TEXT_ASR_MODEL=small ./scripts/run_text_pipeline_local.sh
```

## Notes
- No `openai-whisper` Python package is required for `whisper_cpp`.
- The raw text stage does not require PyTorch when using `whisper_cpp`.
- ASR payload includes:
  - `transcript`
  - `transcript_language`
  - `timestamps` (if returned by backend)
- `subtitle_srt` / `subtitle_vtt` (if produced)

## Google Colab (GPU)

Use `faster_whisper` on GPU with S3 state checkpoints:

```bash
python scripts/run_text_pipeline_colab.py \
  --s3_bucket "$S3_BUCKET" \
  --s3_region "$AWS_REGION" \
  --asr_backend faster_whisper \
  --asr_model small \
  --max_workers 1 \
  --checkpoint_seconds 60
```

Recommended for Colab:
- Runtime: GPU (T4/L4/A100)
- `max_workers=1` to avoid GPU memory contention
- Leave `--faster_whisper_batch_size=0` (auto-tunes by GPU memory)

## Optional CLI knobs
`Data/raw/Text/text_collect.py` supports:
- `--whisper_cpp_bin`
- `--whisper_cpp_model_dir`
- `--whisper_cpp_threads`
- `--whisper_cpp_emit_subtitles / --no-whisper_cpp_emit_subtitles`

"""
Download audio-only files for all URLs in the wide links CSV.
"""

import csv
import re
from pathlib import Path

import yt_dlp


def safe_name(text: str) -> str:
    text = re.sub(r"[\\/]+", "_", text)
    text = re.sub(r"[^A-Za-z0-9 _-]", "", text).strip()
    return text or "Shorts"


def download_audio(url: str, out_stem: Path) -> None:
    ydl_opts = {
        "outtmpl": str(out_stem) + ".%(ext)s",
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    csv_path = root / "Data" / "Links" / "shorts_data" / "shorts_links_wide.csv"
    out_dir = root / "Interpretation" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = (row.get("url") or "").strip()
            if not url:
                continue
            category = (row.get("query") or row.get("category_type") or "Shorts").strip()
            counters[category] = counters.get(category, 0) + 1
            name = f"{safe_name(category)}{counters[category]}"
            out_stem = out_dir / name
            out_mp3 = out_stem.with_suffix(".mp3")
            if out_mp3.exists():
                print(f"SKIP  {out_mp3.name}")
                continue
            print(f"GET   {url} -> {out_mp3.name}")
            try:
                download_audio(url, out_stem)
            except Exception as exc:
                print(f"FAIL  {url}: {exc}")


if __name__ == "__main__":
    main()

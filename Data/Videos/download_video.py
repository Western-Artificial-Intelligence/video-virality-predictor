"""
download_video.py
-----------------
Reads video IDs from videos.csv and downloads each YouTube video (Short)
as an MP4 into data/raw/.
"""

import csv
from pathlib import Path
import yt_dlp
import re


def read_url_categories(csv_path: Path):
    """
    Read URL + category pairs from the CSV (headers: query, category_type, url).
    Uses `query` as the human-readable category (e.g., "gaming Shorts");
    falls back to `category_type` if `query` is missing.
    Returns a list of tuples: (url, category).
    """
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("url") or "").strip()
            if not url:
                continue
            category = (row.get("query") or row.get("category_type") or "Shorts").strip()
            rows.append((url, category))
    return rows


def download_video(url: str, out_stem: Path):
    """
    Download a YouTube video (Short) as MP4 using yt-dlp.
    """
    out_tpl = str(out_stem.with_suffix(".%(ext)s"))

    # format options: https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#format-selection-examples
    # dependency: ffmpeg
    ydl_opts = {
        "outtmpl": out_tpl,
        "format": "bv*[height<=480]/bv*[height>=480]",  # video only with “closest to 480p” (e.g. 360p or 720p fallback)
        #"format": "bv*[filesize<50M]/b[filesize<50M]",  # video-only <50MB
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Normalize to .mp4 if yt-dlp used another format
    for ext in (".mkv", ".webm"):
        alt = out_stem.with_suffix(ext)
        if alt.exists():
            alt.rename(out_stem.with_suffix(".mp4"))


def main():
    # Directory of this script
    script_dir = Path(__file__).parent

    # Path to the CSV file
    csv_path = script_dir.parent / "Links" / "shorts_data" / "shorts_links_wide.csv"

    # Use the same folder as the download target
    out_dir = script_dir 

    rows = read_url_categories(csv_path)
    if not rows:
        print("No URLs found in CSV.")
        return

    print(f"Found {len(rows)} videos in {csv_path}. Starting downloads...\n")

    successes, failures = 0, 0
    counters = {}
    for url, category in rows:
        # Increment per-category counter
        counters[category] = counters.get(category, 0) + 1
        idx = counters[category]

        # Build a safe filename stem: allow spaces; remove path separators and odd chars
        safe_category = re.sub(r"[\\/]+", "_", category)
        safe_category = re.sub(r"[^A-Za-z0-9 _-]", "", safe_category).strip()
        if not safe_category:
            safe_category = "Shorts"
        filename_stem = f"{safe_category}{idx}"

        out_stem = out_dir / filename_stem
        out_mp4 = out_stem.with_suffix(".mp4")

        if out_mp4.exists():
            print(f"SKIP  {url} → {out_mp4} (already exists)")
            continue

        try:
            print(f"GET   {url} → {out_mp4}")
            download_video(url, out_stem)
            print(f"OK    {out_mp4.name}")
            successes += 1
        except Exception as e:
            print(f"FAIL  {url}: {e}")
            failures += 1

    print("\nSummary:")
    print(f"  Success: {successes}")
    print(f"  Failed : {failures}")


if __name__ == "__main__":
    main()

"""
Join cluster assignments with the source video URLs.
"""

import csv
import re
from pathlib import Path


def build_video_id_map(links_csv: Path) -> dict:
    counters = {}
    video_id_to_url = {}
    with links_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = (row.get("url") or "").strip()
            if not url:
                continue
            category = (row.get("query") or row.get("category_type") or "Shorts").strip()
            counters[category] = counters.get(category, 0) + 1
            idx = counters[category]
            safe = re.sub(r"[\\/]+", "_", category)
            safe = re.sub(r"[^A-Za-z0-9 _-]", "", safe).strip() or "Shorts"
            video_id = f"{safe}{idx}"
            video_id_to_url[video_id] = url
    return video_id_to_url


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    clusters_csv = root / "Unsup_Cluster" / "cluster_results.csv"
    links_csv = root / "Data" / "Links" / "shorts_data" / "shorts_links_wide.csv"
    out_csv = root / "Interpretation" / "cluster_links.csv"

    video_id_to_url = build_video_id_map(links_csv)

    rows = []
    with clusters_csv.open(newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        for row in reader:
            video_id = (row.get("video_id") or "").strip()
            rows.append(
                {
                    "video_id": video_id,
                    "cluster": (row.get("cluster") or "").strip(),
                    "url": video_id_to_url.get(video_id, ""),
                }
            )

    rows.sort(
        key=lambda r: (
            int(r["cluster"]) if r["cluster"].isdigit() else 10**9,
            r["video_id"],
        )
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=["video_id", "cluster", "url"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()

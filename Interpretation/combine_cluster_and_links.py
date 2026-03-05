"""Join cluster assignments with source URLs using canonical video_id."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Data.common.horizon_delta import DEFAULT_METADATA_CSV, load_latest_horizon_rows  # noqa: E402


def build_video_url_map(metadata_csv: Path) -> dict[str, str]:
    items = load_latest_horizon_rows(csv_path=metadata_csv)
    return {item.video_id: item.video_url for item in items if item.video_id}


def _cluster_sort_key(row: dict) -> tuple[int, str]:
    raw = str(row.get("cluster", "")).strip()
    if raw.isdigit():
        return int(raw), str(row.get("video_id", ""))
    return 10**9, str(row.get("video_id", ""))


def join_cluster_with_urls(
    *,
    clusters_csv: Path,
    video_id_to_url: dict[str, str],
    out_csv: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with clusters_csv.open(newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        required = {"video_id", "cluster"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required columns in {clusters_csv}: {missing}")

        for row in reader:
            video_id = str(row.get("video_id") or "").strip()
            out_row = dict(row)
            out_row["video_id"] = video_id
            out_row["cluster"] = str(row.get("cluster") or "").strip()
            out_row["url"] = video_id_to_url.get(video_id, "")
            rows.append(out_row)

    rows.sort(key=_cluster_sort_key)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["video_id", "cluster", "url"]
    if "url" not in fieldnames:
        fieldnames.append("url")

    with out_csv.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join cluster_results.csv with metadata URLs by video_id")
    parser.add_argument("--clusters_csv", default=str(REPO_ROOT / "Unsup_Cluster" / "cluster_results.csv"))
    parser.add_argument("--metadata_csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--out_csv", default=str(REPO_ROOT / "Interpretation" / "cluster_links.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    clusters_csv = Path(args.clusters_csv)
    metadata_csv = Path(args.metadata_csv)
    out_csv = Path(args.out_csv)

    if not clusters_csv.exists():
        raise FileNotFoundError(f"Cluster file not found: {clusters_csv}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")

    video_id_to_url = build_video_url_map(metadata_csv)

    rows = join_cluster_with_urls(
        clusters_csv=clusters_csv,
        video_id_to_url=video_id_to_url,
        out_csv=out_csv,
    )

    matched = sum(1 for r in rows if str(r.get("url", "")).strip())
    print("cluster_links summary")
    print(f"rows: {len(rows)}")
    print(f"urls_matched: {matched}")
    print(f"out_csv: {out_csv}")


if __name__ == "__main__":
    main()

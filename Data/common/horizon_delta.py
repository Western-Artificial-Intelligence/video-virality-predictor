import csv
import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

DEFAULT_METADATA_CSV = Path("Data/raw/Metadata/shorts_metadata_horizon.csv")
DEFAULT_URL_COLUMNS = ("video_url", "url", "youtube_url")

YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def parse_video_id(raw_url: str) -> Optional[str]:
    if not raw_url:
        return None

    value = raw_url.strip()
    if YOUTUBE_ID_RE.match(value):
        return value

    parsed = urlparse(value)
    host = (parsed.netloc or "").lower().replace("www.", "")
    path = parsed.path or ""

    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        qs = parse_qs(parsed.query or "")
        if "v" in qs and qs["v"]:
            candidate = qs["v"][0]
            if YOUTUBE_ID_RE.match(candidate):
                return candidate
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "v"}:
            candidate = parts[1]
            if YOUTUBE_ID_RE.match(candidate):
                return candidate

    if host == "youtu.be":
        candidate = (path or "").lstrip("/").split("/")[0]
        if YOUTUBE_ID_RE.match(candidate):
            return candidate

    match = re.search(r"([A-Za-z0-9_-]{11})", value)
    if match:
        return match.group(1)
    return None


def parse_captured_at(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def resolve_url_column(fieldnames: List[str], candidates: List[str]) -> str:
    lowered = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]

    url_like = [f for f in fieldnames if "url" in f.lower() and "thumb" not in f.lower()]
    if url_like:
        return url_like[0]

    raise ValueError("No URL column found. Tried explicit candidates and URL-like fallback columns.")


def build_source_hash(url: str, captured_at: Optional[str] = None, include_captured_at: bool = False) -> str:
    normalized_url = (url or "").strip()
    payload = normalized_url if not include_captured_at else f"{normalized_url}|{captured_at or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class DeltaItem:
    video_id: str
    video_url: str
    captured_at: str
    source_hash: str
    row: Dict[str, str]


class ScriptStateDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_items (
                video_id TEXT PRIMARY KEY,
                source_hash TEXT NOT NULL,
                processed_at TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT
            )
            """
        )
        self.conn.commit()

    def get(self, video_id: str):
        cur = self.conn.execute(
            "SELECT video_id, source_hash, processed_at, status, error FROM processed_items WHERE video_id = ?",
            (video_id,),
        )
        return cur.fetchone()

    def upsert(self, video_id: str, source_hash: str, processed_at: str, status: str, error: str = "") -> None:
        self.conn.execute(
            """
            INSERT INTO processed_items (video_id, source_hash, processed_at, status, error)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
              source_hash = excluded.source_hash,
              processed_at = excluded.processed_at,
              status = excluded.status,
              error = excluded.error
            """,
            (video_id, source_hash, processed_at, status, error),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def load_latest_horizon_rows(
    csv_path: Path,
    url_columns: Optional[List[str]] = None,
    captured_at_col: str = "captured_at",
    include_captured_at_in_hash: bool = False,
) -> List[DeltaItem]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []

        url_col = resolve_url_column(reader.fieldnames, url_columns or list(DEFAULT_URL_COLUMNS))

        latest: Dict[str, Dict[str, str]] = {}
        for row in reader:
            raw_video_id = (row.get("video_id") or "").strip()
            video_url = (row.get(url_col) or "").strip()
            video_id = raw_video_id or parse_video_id(video_url)
            if not video_id:
                continue

            current = dict(row)
            current["video_id"] = video_id
            current["_resolved_video_url"] = video_url
            current_ts = parse_captured_at(current.get(captured_at_col, "") or "")

            prev = latest.get(video_id)
            if prev is None:
                latest[video_id] = current
                continue

            prev_ts = parse_captured_at(prev.get(captured_at_col, "") or "")
            if prev_ts is None and current_ts is not None:
                latest[video_id] = current
            elif prev_ts is not None and current_ts is not None and current_ts >= prev_ts:
                latest[video_id] = current

    items: List[DeltaItem] = []
    for row in latest.values():
        video_id = row["video_id"]
        video_url = row.get("_resolved_video_url", "")
        captured_at = row.get(captured_at_col, "") or ""
        source_hash = build_source_hash(
            video_url,
            captured_at=captured_at,
            include_captured_at=include_captured_at_in_hash,
        )
        items.append(
            DeltaItem(
                video_id=video_id,
                video_url=video_url,
                captured_at=captured_at,
                source_hash=source_hash,
                row=row,
            )
        )

    return items


def compute_delta(items: List[DeltaItem], state_db: ScriptStateDB, max_items: Optional[int] = None) -> List[DeltaItem]:
    # Only treat these statuses as terminal/processed for a given source hash.
    # Failed states must stay in delta so the next run retries them.
    terminal_statuses = {"success", "no_captions"}

    delta: List[DeltaItem] = []
    for item in items:
        existing = state_db.get(item.video_id)
        if existing is None:
            delta.append(item)
        else:
            _, existing_hash, _, existing_status, _ = existing
            if existing_hash != item.source_hash or existing_status not in terminal_statuses:
                delta.append(item)

        if max_items is not None and len(delta) >= max_items:
            break
    return delta

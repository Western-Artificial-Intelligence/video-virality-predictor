"""SQLite state tracking for stage-level idempotency and delta retries."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from Data.common.horizon_delta import DeltaItem


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageStateRow:
    video_id: str
    source_hash: str
    processed_at: str
    status: str
    error: str
    artifact_key: str
    vector_id: str
    retry_count: int = 0


class StageStateDB:
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
                error TEXT,
                artifact_key TEXT,
                vector_id TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._migrate_schema()
        self.conn.commit()

    def _migrate_schema(self) -> None:
        columns = {
            str(row[1]) for row in self.conn.execute("PRAGMA table_info(processed_items)")
        }
        if "retry_count" not in columns:
            self.conn.execute(
                "ALTER TABLE processed_items ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        self.conn.close()

    def get(self, video_id: str) -> Optional[StageStateRow]:
        cur = self.conn.execute(
            """
            SELECT video_id, source_hash, processed_at, status, error, artifact_key, vector_id, retry_count
            FROM processed_items
            WHERE video_id = ?
            """,
            (video_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return StageStateRow(
            video_id=row[0],
            source_hash=row[1],
            processed_at=row[2],
            status=row[3],
            error=row[4] or "",
            artifact_key=row[5] or "",
            vector_id=row[6] or "",
            retry_count=int(row[7] or 0),
        )

    def upsert(
        self,
        video_id: str,
        source_hash: str,
        status: str,
        error: str = "",
        artifact_key: str = "",
        vector_id: str = "",
        retry_count: Optional[int] = None,
    ) -> None:
        existing = self.get(video_id)
        if retry_count is None:
            if existing and existing.source_hash == source_hash:
                retry_count = int(existing.retry_count)
            else:
                retry_count = 0
        self.conn.execute(
            """
            INSERT INTO processed_items (
                video_id, source_hash, processed_at, status, error, artifact_key, vector_id, retry_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                source_hash = excluded.source_hash,
                processed_at = excluded.processed_at,
                status = excluded.status,
                error = excluded.error,
                artifact_key = excluded.artifact_key,
                vector_id = excluded.vector_id,
                retry_count = excluded.retry_count
            """,
            (
                video_id,
                source_hash,
                utc_now_iso(),
                status,
                error,
                artifact_key,
                vector_id,
                int(retry_count),
            ),
        )
        self.conn.commit()

    def upsert_with_retry(
        self,
        video_id: str,
        source_hash: str,
        max_fail_retries: int,
        error: str = "",
        artifact_key: str = "",
        vector_id: str = "",
        fail_status: str = "fail",
        terminal_status: str = "fail_terminal",
    ) -> str:
        existing = self.get(video_id)
        if existing and existing.source_hash == source_hash:
            retry_count = int(existing.retry_count) + 1
        else:
            retry_count = 1

        status = fail_status if retry_count <= int(max_fail_retries) else terminal_status
        self.upsert(
            video_id=video_id,
            source_hash=source_hash,
            status=status,
            error=error,
            artifact_key=artifact_key,
            vector_id=vector_id,
            retry_count=retry_count,
        )
        return status


def compute_stage_delta(
    items: Sequence[DeltaItem],
    state_db: StageStateDB,
    terminal_statuses: Iterable[str] = ("success",),
    max_items: Optional[int] = None,
) -> list[DeltaItem]:
    terminal = set(terminal_statuses)
    delta: list[DeltaItem] = []
    for item in items:
        existing = state_db.get(item.video_id)
        if existing is None:
            delta.append(item)
        else:
            if existing.source_hash != item.source_hash or existing.status not in terminal:
                delta.append(item)
        if max_items is not None and len(delta) >= max_items:
            break
    return delta

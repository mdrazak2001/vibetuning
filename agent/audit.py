"""Audit trail: SingleStore when live, SQLite when the flag is off or the wire fails."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from agent import ROOT
from agent.flags import status

SQLITE_PATH = ROOT / "data" / "audit.sqlite"

RUNS_SQL = """
CREATE TABLE IF NOT EXISTS runs (
  run_id VARCHAR(64) PRIMARY KEY,
  phase VARCHAR(32),
  base_model VARCHAR(128),
  status VARCHAR(32),
  predicted_inr DOUBLE,
  actual_inr DOUBLE,
  wer_before DOUBLE,
  wer_after DOUBLE,
  created_at VARCHAR(64)
)
"""

EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id VARCHAR(64),
  ts VARCHAR(64),
  event VARCHAR(64),
  payload TEXT,
  cost_inr DOUBLE,
  wer DOUBLE
)
"""

ROWS_SQL = """
CREATE TABLE IF NOT EXISTS training_rows (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id VARCHAR(64),
  dataset VARCHAR(128),
  source_url VARCHAR(512),
  license VARCHAR(64),
  split VARCHAR(16),
  audio_path VARCHAR(256),
  text TEXT,
  connector_version VARCHAR(64),
  sync_id VARCHAR(64),
  ingested_at VARCHAR(64)
)
"""

# SingleStore wants AUTO_INCREMENT not AUTOINCREMENT.
RUNS_SS = RUNS_SQL
EVENTS_SS = EVENTS_SQL.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGINT PRIMARY KEY AUTO_INCREMENT")
ROWS_SS = ROWS_SQL.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGINT PRIMARY KEY AUTO_INCREMENT")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ss_kwargs() -> dict[str, Any] | None:
    host = os.getenv("SINGLESTORE_HOST") or os.getenv("DB_HOST")
    if not host:
        return None
    return {
        "host": host,
        "port": int(os.getenv("SINGLESTORE_PORT") or os.getenv("DB_PORT") or "3306"),
        "user": os.getenv("SINGLESTORE_USER") or os.getenv("SINGLESTORE_USERNAME") or os.getenv("DB_USER") or "",
        "password": os.getenv("SINGLESTORE_PASSWORD") or os.getenv("DB_PASSWORD") or "",
        "database": os.getenv("SINGLESTORE_DATABASE") or os.getenv("DB_NAME") or "vibetuning",
        "connection_timeout": 8,
    }


class Store:
    def __init__(self) -> None:
        self.backend = "sqlite"
        self.error = ""
        flags = status()
        if flags.singlestore and _ss_kwargs():
            try:
                import mysql.connector  # noqa: F401

                self.backend = "singlestore"
            except Exception as exc:
                self.backend = "sqlite"
                self.error = f"mysql driver: {exc.__class__.__name__}"
                flags.notes["singlestore"] = self.error
        else:
            flags.notes["singlestore"] = "sqlite fallback (no host or flag off)"

    @contextmanager
    def connect(self) -> Iterator[Any]:
        if self.backend == "singlestore":
            import mysql.connector

            conn = mysql.connector.connect(**_ss_kwargs())
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(SQLITE_PATH)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            cur = conn.cursor()
            if self.backend == "singlestore":
                cur.execute(RUNS_SS)
                cur.execute(EVENTS_SS)
                cur.execute(ROWS_SS)
            else:
                cur.execute(RUNS_SQL)
                cur.execute(EVENTS_SQL)
                cur.execute(ROWS_SQL)

    def upsert_run(self, **row: Any) -> None:
        cols = [
            "run_id",
            "phase",
            "base_model",
            "status",
            "predicted_inr",
            "actual_inr",
            "wer_before",
            "wer_after",
            "created_at",
        ]
        row.setdefault("created_at", _now())
        placeholders = ", ".join(["%s" if self.backend == "singlestore" else "?"] * len(cols))
        sql = (
            f"INSERT INTO runs ({', '.join(cols)}) VALUES ({placeholders}) "
            "ON DUPLICATE KEY UPDATE status=VALUES(status), actual_inr=VALUES(actual_inr), "
            "wer_after=VALUES(wer_after)"
            if self.backend == "singlestore"
            else f"INSERT OR REPLACE INTO runs ({', '.join(cols)}) VALUES ({placeholders})"
        )
        vals = [row.get(c) for c in cols]
        with self.connect() as conn:
            conn.cursor().execute(sql, vals)

    def emit(
        self,
        run_id: str,
        event: str,
        payload: dict | None = None,
        cost_inr: float | None = None,
        wer: float | None = None,
        ts: str | None = None,
    ) -> None:
        ph = "%s" if self.backend == "singlestore" else "?"
        sql = (
            f"INSERT INTO events (run_id, ts, event, payload, cost_inr, wer) "
            f"VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})"
        )
        with self.connect() as conn:
            conn.cursor().execute(
                sql,
                (run_id, ts or _now(), event, json.dumps(payload or {}), cost_inr, wer),
            )

    def events(self, run_id: str | None = None) -> list[dict]:
        with self.connect() as conn:
            cur = conn.cursor()
            if run_id:
                ph = "%s" if self.backend == "singlestore" else "?"
                cur.execute(
                    f"SELECT run_id, ts, event, payload, cost_inr, wer FROM events WHERE run_id={ph} ORDER BY ts, id",
                    (run_id,),
                )
            else:
                cur.execute("SELECT run_id, ts, event, payload, cost_inr, wer FROM events ORDER BY ts, id")
            rows = cur.fetchall()
        out = []
        for r in rows:
            if isinstance(r, sqlite3.Row):
                item = dict(r)
            else:
                item = {
                    "run_id": r[0],
                    "ts": r[1],
                    "event": r[2],
                    "payload": r[3],
                    "cost_inr": r[4],
                    "wer": r[5],
                }
            payload = item.get("payload")
            if isinstance(payload, str):
                try:
                    item["payload"] = json.loads(payload)
                except json.JSONDecodeError:
                    pass
            out.append(item)
        return out

    def lineage(self, run_id: str) -> dict:
        with self.connect() as conn:
            cur = conn.cursor()
            ph = "%s" if self.backend == "singlestore" else "?"
            cur.execute(
                f"SELECT COUNT(*), MIN(dataset), MIN(source_url), MIN(license), MIN(connector_version), MIN(sync_id) "
                f"FROM training_rows WHERE run_id={ph}",
                (run_id,),
            )
            row = cur.fetchone()
        if not row:
            return {"n": 0}
        n = row[0] if not isinstance(row, sqlite3.Row) else row[0]
        return {
            "n": int(n or 0),
            "dataset": row[1],
            "source_url": row[2],
            "license": row[3],
            "connector_version": row[4],
            "sync_id": row[5],
        }

    def insert_training_rows(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        ph = "%s" if self.backend == "singlestore" else "?"
        cols = [
            "run_id",
            "dataset",
            "source_url",
            "license",
            "split",
            "audio_path",
            "text",
            "connector_version",
            "sync_id",
            "ingested_at",
        ]
        sql = f"INSERT INTO training_rows ({', '.join(cols)}) VALUES ({', '.join([ph] * len(cols))})"
        vals = []
        now = _now()
        for r in rows:
            vals.append(
                (
                    r.get("run_id"),
                    r.get("dataset"),
                    r.get("source_url"),
                    r.get("license"),
                    r.get("split"),
                    r.get("audio_path"),
                    r.get("text"),
                    r.get("connector_version"),
                    r.get("sync_id"),
                    r.get("ingested_at") or now,
                )
            )
        with self.connect() as conn:
            conn.cursor().executemany(sql, vals)
        return len(vals)


_STORE: Store | None = None


def store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
        _STORE.init_schema()
    return _STORE

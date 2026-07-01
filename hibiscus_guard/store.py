"""The incident store — durable state in a tiny SQLite file.

Why this exists: the ambient agent's escalation logic needs to remember past
sightings, and the (future) daily digest agent — a SEPARATE process — needs to
read them. In-memory session state can do neither across a restart or a process
boundary. A small SQLite file does both: one writer (the ambient agent), many
readers, survives restarts.

One table, append-only. Each row is one sighting the agent decided about:
  * tier      "confident" | "candidate"  (a "maybe")
  * alerted   did the agent actually notify someone?
  * urgency   low/medium/high, or NULL for candidates it passed on

This is also a governance seam: the ambient agent has WRITE access; the digest
agent will get READ-ONLY access. Least privilege, made concrete.
"""

import os
import sqlite3
import time
from contextlib import contextmanager

_DB_PATH = os.environ.get("INCIDENT_DB", os.path.join(os.path.dirname(__file__), "incidents.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL    NOT NULL,
    camera     TEXT    NOT NULL,
    track_id   INTEGER NOT NULL,
    tier       TEXT    NOT NULL,   -- 'confident' | 'candidate'
    confidence REAL    NOT NULL,
    alerted    INTEGER NOT NULL,   -- 0 | 1
    urgency    TEXT                -- NULL for candidates
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def record(
    camera: str, track_id: int, tier: str, confidence: float, alerted: bool, urgency: str | None
) -> None:
    """Append one decided sighting."""
    with _conn() as c:
        c.execute(
            "INSERT INTO incidents (ts, camera, track_id, tier, confidence, alerted, urgency)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), camera, track_id, tier, confidence, int(alerted), urgency),
        )


def history(camera: str, window_seconds: int = 3600) -> dict:
    """Confirmed-alert history for a camera, for the escalation decision."""
    since = time.time() - window_seconds
    with _conn() as c:
        rows = c.execute(
            "SELECT ts FROM incidents WHERE camera=? AND alerted=1 AND ts>=? ORDER BY ts",
            (camera, since),
        ).fetchall()
    alerts = [r["ts"] for r in rows]
    last = max(alerts) if alerts else None
    return {
        "alerts_last_hour": len(alerts),
        "seconds_since_last_alert": (time.time() - last) if last else None,
    }


def day_summary(since_seconds: int = 86400) -> dict:
    """Read-only rollup for the daily digest agent (built next phase)."""
    since = time.time() - since_seconds
    with _conn() as c:
        rows = c.execute(
            "SELECT camera, track_id, tier, alerted, urgency, confidence, ts"
            " FROM incidents WHERE ts>=? ORDER BY ts",
            (since,),
        ).fetchall()
    confirmed = [dict(r) for r in rows if r["tier"] == "confident"]
    candidates = [dict(r) for r in rows if r["tier"] == "candidate"]
    return {
        "confirmed_count": len(confirmed),
        "candidate_count": len(candidates),
        "confirmed": confirmed,
        "candidates": candidates,
    }

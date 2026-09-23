"""SQLite persistence: the governance decisions and created places.

The Overture parquet is the read-only seed of existing POIs (held in memory as a
fast blocking cache). This SQLite file (``poi.db``) is the *write* side:

    places  : POIs the chain decided to create (auto_create), stored on disk
    audit   : every submission -> moderation/dedup/route decision, timestamped

stdlib ``sqlite3`` only, no extra dependency. Path via ``POI_DB_PATH``.
"""
import os
import sqlite3
import threading
import uuid

from . import models

DB_PATH = os.environ.get("POI_DB_PATH", "poi.db")

_lock = threading.Lock()
_connection = None


def _get_conn():
    """Return the shared connection. Callers must hold ``_lock``."""
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _init(_connection)
    return _connection


def _init(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS places (
            id TEXT PRIMARY KEY,
            name TEXT, address TEXT, category TEXT, phone TEXT, brand TEXT,
            lat REAL, lng REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT (datetime('now')),
            sub_id TEXT, name TEXT, description TEXT,
            mod_verdict TEXT, mod_conf REAL,
            dedup_verdict TEXT, dedup_backend TEXT, matched_place TEXT, dedup_conf REAL,
            action TEXT, reason TEXT
        );
    """)
    conn.commit()


def record(sub, mod, res, route):
    """Persist one submission decision to the audit log (+ place mutation)."""
    with _lock:
        conn = _get_conn()
        if route["action"] == "auto_create":
            conn.execute(
                "INSERT INTO places (id, name, address, category, phone, brand, lat, lng) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (f"poi_{uuid.uuid4().hex[:12]}",
                 models.get(sub.get("name")), models.get(sub.get("address")),
                 models.get(sub.get("category")), models.get(sub.get("phone")),
                 models.get(sub.get("brand")), models.get(sub.get("lat")), models.get(sub.get("lng"))))
        conn.execute(
            "INSERT INTO audit (sub_id, name, description, mod_verdict, mod_conf, "
            "dedup_verdict, dedup_backend, matched_place, dedup_conf, action, reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (sub.get("id"), models.get(sub.get("name")), (sub.get("description") or "")[:200],
             mod["verdict"], mod["confidence"],
             res["verdict"], res.get("backend"),
             (models.get(res["matched"].get("name")) if res["matched"] else None),
             res["confidence"],
             route["action"], (route.get("why") or "")[:200]))
        conn.commit()


def stats():
    with _lock:
        conn = _get_conn()
        return {
            "places": conn.execute("SELECT COUNT(*) c FROM places").fetchone()["c"],
            "audit": conn.execute("SELECT COUNT(*) c FROM audit").fetchone()["c"],
            "by_action": {r["action"]: r["c"] for r in
                          conn.execute("SELECT action, COUNT(*) c FROM audit GROUP BY action")},
        }


def recent(limit=50):
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT ts, name, description, mod_verdict, dedup_verdict, matched_place, "
            "action, reason FROM audit ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]


def places():
    """All locally-created places, as POI dicts for the dedup scan.

    Read fresh on every call so a place created moments ago is immediately
    visible to the next dedup -- the governance index is seed + live, not a
    frozen snapshot.
    """
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, name, address, category, phone, brand, lat, lng FROM places"
        ).fetchall()
        return [dict(r) for r in rows]

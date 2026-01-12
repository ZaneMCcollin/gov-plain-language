"""migrations.py — ultra-light SQLite migrations.

Keeps your DB schema forward-compatible on Cloud Run and Streamlit Cloud.
"""

from __future__ import annotations

import sqlite3
from typing import Callable, List, Tuple


Migration = Tuple[str, Callable[[sqlite3.Connection], None]]


def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    return col in cols


def apply_migrations(conn: sqlite3.Connection, migrations: List[Migration]) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            applied_at TEXT NOT NULL
        )
        """
    )
    conn.commit()

    applied = {r[0] for r in conn.execute("SELECT name FROM schema_migrations").fetchall()}

    for name, fn in migrations:
        if name in applied:
            continue
        fn(conn)
        conn.execute("INSERT INTO schema_migrations(name, applied_at) VALUES(?, datetime('now'))", (name,))
        conn.commit()


def default_migrations() -> List[Migration]:
    def m001_core(conn: sqlite3.Connection) -> None:
        # audit + roles tables (idempotent)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                user_email TEXT,
                role TEXT,
                event TEXT NOT NULL,
                workspace TEXT,
                doc_id TEXT,
                meta_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS role_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                workspace TEXT NOT NULL,
                email TEXT NOT NULL,
                role TEXT NOT NULL,
                updated_by TEXT
            )
            """
        )

    def m002_audit_role_backfill(conn: sqlite3.Connection) -> None:
        # older DBs might not have role column (if created pre-role)
        if not _has_column(conn, "audit_logs", "role"):
            conn.execute("ALTER TABLE audit_logs ADD COLUMN role TEXT")

    def m003_indexes(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_logs(ts DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_email, ts DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_role ON audit_logs(role, ts DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ws ON audit_logs(workspace, ts DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_doc ON audit_logs(doc_id, ts DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_roles_email_ws ON role_assignments(email, workspace)")

    return [
        ("001_core_audit_roles", m001_core),
        ("002_audit_role_column", m002_audit_role_backfill),
        ("003_audit_roles_indexes", m003_indexes),
    ]

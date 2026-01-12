"""roles.py — role resolution + permissions + role assignment persistence (SQLite).

Designed to be imported by app.py without side effects.
All DB access is done via a provided connection factory / db_path.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple


def normalize_email_list(x) -> List[str]:
    """Normalize a config value into a list of lowercase emails/domains.

    Accepts:
      - comma-separated str
      - list/tuple of values
    """
    if not x:
        return []
    if isinstance(x, str):
        items = [e.strip().lower() for e in x.split(",") if e.strip()]
    elif isinstance(x, (list, tuple)):
        items = [str(e).strip().lower() for e in x if str(e).strip()]
    else:
        items = []
    # preserve order while de-duping
    seen = set()
    out: List[str] = []
    for it in items:
        if it and it not in seen:
            out.append(it)
            seen.add(it)
    return out


def ensure_role_tables(conn: sqlite3.Connection) -> None:
    """Ensure role assignment tables exist (idempotent)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS role_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            workspace TEXT NOT NULL,      -- '*' means global
            email TEXT NOT NULL,
            role TEXT NOT NULL,           -- admin/reviewer/editor/viewer
            updated_by TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_roles_email_ws ON role_assignments(email, workspace)")
    conn.commit()


def db_role_lookup(db_path: str, email: str, workspace: str) -> Optional[str]:
    """Return most recent role for (email, workspace) or global ('*')."""
    email = (email or "").strip().lower()
    ws = (workspace or "").strip() or "default"
    if not email:
        return None
    try:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        ensure_role_tables(conn)
        cur = conn.cursor()

        cur.execute(
            "SELECT role FROM role_assignments WHERE email = ? AND workspace = ? ORDER BY id DESC LIMIT 1",
            (email, ws),
        )
        row = cur.fetchone()
        if row and row[0]:
            conn.close()
            return str(row[0])

        cur.execute(
            "SELECT role FROM role_assignments WHERE email = ? AND workspace = '*' ORDER BY id DESC LIMIT 1",
            (email,),
        )
        row = cur.fetchone()
        conn.close()
        return str(row[0]) if row and row[0] else None
    except Exception:
        return None


def set_role_assignment(db_path: str, workspace: str, email: str, role: str, updated_by: str = "") -> None:
    """Persist a role assignment (safe; never raises)."""
    ws = (workspace or "").strip() or "default"
    em = (email or "").strip().lower()
    rl = (role or "").strip().lower() or "viewer"
    if not em:
        return
    try:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        ensure_role_tables(conn)
        conn.execute(
            "INSERT INTO role_assignments(ts, workspace, email, role, updated_by) VALUES(?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), ws, em, rl, (updated_by or "").strip().lower()),
        )
        conn.commit()
        conn.close()
    except Exception:
        return


def list_role_assignments(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Return latest role assignments for admin UI."""
    try:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        ensure_role_tables(conn)
        cur = conn.cursor()
        cur.execute(
            "SELECT ts, workspace, email, role, updated_by FROM role_assignments ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
        rows = cur.fetchall() or []
        conn.close()
        return [
            {"ts": r[0], "workspace": r[1], "email": r[2], "role": r[3], "updated_by": r[4]}
            for r in rows
        ]
    except Exception:
        return []


# Permissions
ROLE_PERMS: Dict[str, Set[str]] = {
    "admin":    {"convert", "export", "approve", "edit_outputs", "rollback", "analytics", "role_admin", "audit_view"},
    "reviewer": {"export", "approve", "edit_outputs"},
    "editor":   {"convert", "export", "edit_outputs"},
    "viewer":   set(),
}

# In production we lock down high-risk actions regardless of role switching flags
PROD_HARD_BLOCK: Set[str] = {"convert", "edit_outputs", "rollback", "analytics", "role_admin", "workspace_switch"}


def can_action(role: str, prod: bool, action: str) -> bool:
    role = (role or "viewer").strip().lower() or "viewer"

    if role == "admin":
        return True

    if prod and action in PROD_HARD_BLOCK:
        return False

    return action in ROLE_PERMS.get(role, set())

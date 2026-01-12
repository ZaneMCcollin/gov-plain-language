"""audit.py — audit logging + querying + Streamlit admin viewer.

No side effects on import. All DB work uses a provided connection factory.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ConnFactory = Callable[[], sqlite3.Connection]


def ensure_audit_tables(conn: sqlite3.Connection) -> None:
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_logs(ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_email, ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_role ON audit_logs(role, ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ws ON audit_logs(workspace, ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_doc ON audit_logs(doc_id, ts DESC)")
    conn.commit()


def log_audit(
    db: ConnFactory,
    event: str,
    user_email: str = "",
    doc_id: str = "",
    workspace: str = "",
    role: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Write an audit log event to SQLite. Safe: never raises."""
    try:
        conn = db()
        ensure_audit_tables(conn)
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        ts = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO audit_logs(ts, user_email, role, event, workspace, doc_id, meta_json) VALUES(?,?,?,?,?,?,?)",
            (ts, (user_email or "").lower(), (role or ""), event, workspace or "", doc_id or "", meta_json),
        )
        conn.commit()
        conn.close()
    except Exception:
        return


def query_audit(
    db: ConnFactory,
    days: int,
    workspace: str,
    all_workspaces: bool,
    user_contains: str,
    role_contains: str,
    event_contains: str,
    limit: int,
) -> Tuple[str, List[Tuple[Any, ...]]]:
    since = (datetime.now(timezone.utc).timestamp() - int(days) * 86400)
    since_iso = datetime.fromtimestamp(since, tz=timezone.utc).isoformat()
    conn = db()
    ensure_audit_tables(conn)
    cur = conn.cursor()
    sql = "SELECT ts, user_email, role, event, workspace, doc_id, meta_json FROM audit_logs WHERE ts >= ?"
    params: List[Any] = [since_iso]
    if not all_workspaces:
        sql += " AND workspace = ?"
        params.append(workspace or "default")
    if user_contains.strip():
        sql += " AND lower(user_email) LIKE ?"
        params.append(f"%{user_contains.strip().lower()}%")
    if role_contains.strip():
        sql += " AND lower(role) LIKE ?"
        params.append(f"%{role_contains.strip().lower()}%")
    if event_contains.strip():
        sql += " AND lower(event) LIKE ?"
        params.append(f"%{event_contains.strip().lower()}%")
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(int(limit))
    cur.execute(sql, tuple(params))
    rows = cur.fetchall() or []
    conn.close()
    return since_iso, rows


def rows_to_csv(rows: Sequence[Tuple[Any, ...]]) -> bytes:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["ts", "user_email", "role", "event", "workspace", "doc_id", "meta_json"])
    for r in rows:
        w.writerow(list(r))
    return out.getvalue().encode("utf-8")


def render_audit_viewer(
    st,
    db: ConnFactory,
    current_workspace: str,
    default_days: int = 30,
    max_default: int = 200,
) -> None:
    """Streamlit UI for audit viewer (admin)."""
    st.subheader("Audit Logs")
    ws_filter_mode = st.selectbox("Workspace scope", ["Current workspace", "All workspaces"], index=0)
    audit_days = st.slider("Audit lookback (days)", 1, 180, int(default_days), key="audit_days")
    q_user = st.text_input("Filter: user email contains", value="", key="audit_user_contains")
    q_role = st.text_input("Filter: role contains", value="", key="audit_role_contains")
    q_event = st.text_input("Filter: event contains", value="", key="audit_event_contains")
    max_rows = st.slider("Rows to show", 50, 500, int(max_default), key="audit_max_rows")

    all_ws = (ws_filter_mode == "All workspaces")
    since_iso, rows = query_audit(
        db=db,
        days=audit_days,
        workspace=current_workspace or "default",
        all_workspaces=all_ws,
        user_contains=q_user,
        role_contains=q_role,
        event_contains=q_event,
        limit=max_rows,
    )
    st.caption(f"Showing up to {max_rows} rows since {since_iso}")

    if rows:
        st.dataframe(
            [{"ts": r[0], "user": r[1], "role": r[2], "event": r[3], "workspace": r[4], "doc_id": r[5], "meta": r[6]} for r in rows],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download audit CSV (shown rows)",
            data=rows_to_csv(rows),
            file_name=f"audit_logs_{'all' if all_ws else (current_workspace or 'default')}_{audit_days}d.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("No audit logs found for these filters.")

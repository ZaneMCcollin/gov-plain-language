# auth.py
# ============================================================
# Authentication + allowlists + roles helpers
# - Cloud Run friendly: reads roles from env JSON string or Streamlit Secrets
# - SUPERADMIN_EMAIL / SUPERADMIN_EMAILS: hard-pin admin (break-glass)
# - Optional DB role lookup (per-workspace scoping) via callback
# ============================================================

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

DBLookup = Callable[[str, str], Dict[str, Any]]  # (email, workspace) -> {"role": "...", "is_global_admin": bool}


def normalize_email_list(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, str):
        return [e.strip().lower() for e in x.split(",") if e.strip()]
    if isinstance(x, (list, tuple)):
        return [str(e).strip().lower() for e in x if str(e).strip()]
    return []


def _parse_roles_value(raw: Any) -> Dict[str, List[str]]:
    """
    Accepts:
      - mapping like {"admin": "...", "editor": "..."}
      - JSON string like '{"admin":["a@b.com"],"editor":["c@d.com"]}'
      - legacy-ish string like '{admin:[a@b.com]}' (best-effort)
    Returns dict role->emails list.
    """
    roles: Dict[str, Any] = {}
    if isinstance(raw, dict):
        roles = raw
    elif isinstance(raw, str) and raw.strip():
        s = raw.strip()

        # Best-effort conversion for legacy "{admin:[a@b.com]}" (missing quotes)
        if s.startswith("{") and s.endswith("}") and '"' not in s:
            # Add quotes around keys and list items
            # {admin:[a@b.com,b@c.com]} -> {"admin":["a@b.com","b@c.com"]}
            try:
                inner = s[1:-1].strip()
                # split by commas at top level
                parts = [p.strip() for p in inner.split(",") if p.strip()]
                tmp = {}
                for p in parts:
                    if ":" not in p:
                        continue
                    k, v = p.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    if v.startswith("[") and v.endswith("]"):
                        items = [i.strip() for i in v[1:-1].split(",") if i.strip()]
                        tmp[k] = items
                    else:
                        tmp[k] = [v]
                roles = tmp
            except Exception:
                roles = {}
        else:
            # JSON parse
            try:
                roles = json.loads(s)
            except Exception:
                roles = {}

    out = {"admin": [], "reviewer": [], "editor": [], "viewer": []}
    out["admin"] = normalize_email_list(roles.get("admin"))
    out["reviewer"] = normalize_email_list(roles.get("reviewer"))
    out["editor"] = normalize_email_list(roles.get("editor"))
    out["viewer"] = normalize_email_list(roles.get("viewer"))
    return out


def _superadmins() -> List[str]:
    # Support either SUPERADMIN_EMAIL or SUPERADMIN_EMAILS (comma-separated)
    one = os.environ.get("SUPERADMIN_EMAIL", "") or ""
    many = os.environ.get("SUPERADMIN_EMAILS", "") or ""
    emails = []
    if one.strip():
        emails.append(one.strip().lower())
    emails.extend([e.strip().lower() for e in many.split(",") if e.strip()])
    # de-dupe
    return sorted(set([e for e in emails if e]))


def is_allowed(email: str, safe_secret=None) -> bool:
    """Return True if email is allowed by ALLOWED_EMAILS / ALLOWED_DOMAINS.

    Reads from env vars by default; if safe_secret is provided, it will also read
    from Streamlit secrets (Cloud-friendly).
    """
    if not email or "@" not in email:
        return False
    email = email.lower().strip()

    def _get(key: str) -> str:
        if callable(safe_secret):
            try:
                v = safe_secret(key, "")
                return str(v or "")
            except Exception:
                return str(os.getenv(key, "") or "")
        return str(os.getenv(key, "") or "")

    allowed_domains = [d.strip().lower() for d in _get("ALLOWED_DOMAINS").split(",") if d.strip()]
    allowed_emails = [e.strip().lower() for e in _get("ALLOWED_EMAILS").split(",") if e.strip()]

    if allowed_emails or allowed_domains:
        if email in allowed_emails:
            return True
        domain = email.split("@", 1)[-1]
        return domain in allowed_domains

    # If allowlists not set, allow (dev-friendly); caller may hard-block in PROD.
    return True


def _user_email() -> str:
    try:
        u = getattr(st, "user", None)
        if not u:
            return ""
        return (getattr(u, "email", "") or "").strip().lower()
    except Exception:
        return ""


def require_login(*, prod: bool, safe_secret, is_streamlit_cloud) -> str:
    """
    Returns logged-in email.
    Uses Streamlit built-in auth when available; otherwise fallback allowlist login.
    """
    # A) Streamlit auth
    if hasattr(st, "login") and hasattr(st, "user"):
        try:
            email = _user_email()
            if email:
                if not is_allowed(email, safe_secret):
                    st.error("❌ You are not authorized to use this app.")
                    st.stop()
                return email

            st.info("Please sign in to continue.")
            st.login()
            st.stop()
        except Exception:
            pass

    # B) Fallback allowlist login (Cloud Run)
    st.info("Please sign in to continue.")
    email = st.text_input("Email", placeholder="you@example.com").strip().lower()
    if st.button("Log in"):
        if not email or "@" not in email:
            st.error("Enter a valid email.")
            st.stop()

        if not is_allowed(email, safe_secret):
            st.error("❌ You are not authorized to use this app.")
            st.stop()

        # If allowlists are missing in PROD, warn loudly but allow once
        allow_domains = os.getenv("ALLOWED_DOMAINS", "").strip()
        allow_emails = os.getenv("ALLOWED_EMAILS", "").strip()
        if prod and not (allow_domains or allow_emails):
            st.warning("⚠️ ALLOWED_EMAILS / ALLOWED_DOMAINS are missing. Allowing this login (bootstrap).")
            st.caption("Set ALLOWED_EMAILS or ALLOWED_DOMAINS in Cloud Run env vars to enforce access control.")

        st.session_state["manual_auth_email"] = email
        st.rerun()

    email2 = (st.session_state.get("manual_auth_email") or "").strip().lower()
    if email2:
        return email2

    st.stop()


def get_effective_role(
    *,
    email: str,
    workspace: str,
    safe_secret,
    prod: bool,
    db_lookup: Optional[DBLookup] = None,
) -> Dict[str, Any]:
    """
    Returns dict:
      role: admin|reviewer|editor|viewer
      is_global_admin: bool
      break_glass_admin: bool
      source: str
    Priority:
      1) SUPERADMIN_EMAIL(S) => admin (global)
      2) DB lookup (workspace-specific, then global '*')
      3) roles config from env/secrets
      4) default viewer
    """
    email = (email or "").strip().lower()
    ws = (workspace or "").strip().lower() or "default"

    supers = _superadmins()
    if email and email in supers:
        return {
            "role": "admin",
            "is_global_admin": True,
            "break_glass_admin": True,
            "source": "superadmin",
        }

    if db_lookup:
        r = db_lookup(email, ws) or {}
        role = (r.get("role") or "").strip().lower()
        if role in ("admin", "reviewer", "editor", "viewer"):
            return {
                "role": role,
                "is_global_admin": bool(r.get("is_global_admin")),
                "break_glass_admin": False,
                "source": "db",
            }

    roles_raw = safe_secret("roles", {})
    roles = _parse_roles_value(roles_raw)

    if email and email in roles["admin"]:
        return {"role": "admin", "is_global_admin": False, "break_glass_admin": False, "source": "config"}
    if email and email in roles["reviewer"]:
        return {"role": "reviewer", "is_global_admin": False, "break_glass_admin": False, "source": "config"}
    if email and email in roles["editor"]:
        return {"role": "editor", "is_global_admin": False, "break_glass_admin": False, "source": "config"}
    if email and email in roles["viewer"]:
        return {"role": "viewer", "is_global_admin": False, "break_glass_admin": False, "source": "config"}

    return {"role": "viewer", "is_global_admin": False, "break_glass_admin": False, "source": "default"}

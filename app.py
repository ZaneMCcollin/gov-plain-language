# ============================================================
# GovCan Plain Language Converter — ONE-FILE ENTERPRISE BUILD
# (Cloud Run + SQLite + Reviewer workflow + Versioning + DOCX/PDF + OCR + Limits + Lang detect + OCR confidence)
#
# This single app.py:
# - Writes requirements.txt / Dockerfile / .dockerignore / .streamlit/config.toml if missing
# - Uses SQLite persistence (versions + approvals + comments persist)
# - Saves versions + approvals + comments (persist)
# - Rollback to any version
# - DOCX + PDF export (no LibreOffice)
# - OCR fallback for scanned PDFs + clear OCR indicators
# - OCR confidence highlighting in UI + confidence stats saved to meta
# - Auto language detection (input + OCR) to guide OCR + Gemini prompts
# - Word/char limits BEFORE LLM calls to reduce 429/cost
# - Streamlit Cloud ready (no fixed port in config.toml; Docker uses $PORT)
# - Watermark: DRAFT vs APPROVED on PDFs + status banner on DOCX
# - Compliance Report export (PDF summary)
# - ✅ Auth: Streamlit built-in login (Google/OIDC) + domain/email allowlist
# - ✅ Usage analytics: per-user events for billing insight + admin CSV export
# ============================================================

import os
import io
import re
import json
import time
import csv
import hashlib
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from google import genai
from google.genai.errors import ClientError

from pypdf import PdfReader
from docx import Document

import pytesseract
import pypdfium2 as pdfium
from PIL import Image  # noqa: F401

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

from collections.abc import Mapping
from typing import Mapping
from typing import Dict, List

DEBUG_LLM = st.sidebar.toggle("DEBUG LLM (show raw model output)", value=False)


# ============================================================
# Auto language detection (patched to show real error)
# ============================================================
LANGDETECT_OK = False
LANGDETECT_ERR = ""

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_OK = True
except Exception as e:
    LANGDETECT_OK = False
    LANGDETECT_ERR = repr(e)


# ============================================================
# Project files (auto-create if missing)
# ============================================================
def _write_if_missing(path: str, content: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _bootstrap_project_files() -> None:
    # ✅ Pin Streamlit new enough to support st.login / st.user
    # ✅ Authlib required for Google auth on Streamlit Community Cloud
    _write_if_missing(
        "requirements.txt",
        "\n".join([
            "streamlit>=1.42.0",
            "Authlib>=1.3.2",
            "google-genai",
            "pypdf",
            "python-docx",
            "pytesseract",
            "pypdfium2",
            "Pillow",
            "reportlab",
            "langdetect",
            ""
        ])
    )

    _write_if_missing(
        "Dockerfile",
        "\n".join([
            "FROM python:3.11-slim",
            "",
            "ENV PYTHONDONTWRITEBYTECODE=1",
            "ENV PYTHONUNBUFFERED=1",
            "",
            "# System deps for OCR (English + French + OSD)",
            "RUN apt-get update && apt-get install -y --no-install-recommends \\",
            "    tesseract-ocr \\",
            "    tesseract-ocr-eng \\",
            "    tesseract-ocr-fra \\",
            "    tesseract-ocr-osd \\",
            "    && rm -rf /var/lib/apt/lists/*",
            "",
            "WORKDIR /app",
            "",
            "COPY requirements.txt /app/requirements.txt",
            "RUN pip install --no-cache-dir -r requirements.txt",
            "",
            "COPY app.py /app/app.py",
            "",
            "ENV PORT=8080",
            "EXPOSE 8080",
            "",
            'CMD ["bash","-lc","streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]',
            ""
        ])
    )

    _write_if_missing(
        ".dockerignore",
        "\n".join([
            ".venv",
            "__pycache__",
            "*.pyc",
            ".DS_Store",
            ".git",
            ".gitignore",
            ".env",
            ".streamlit",
            "data",
            ""
        ])
    )

    # ✅ Streamlit Cloud friendly: do NOT pin port/address here
    os.makedirs(".streamlit", exist_ok=True)
    _write_if_missing(
        ".streamlit/config.toml",
        "\n".join([
            "[server]",
            "headless = true",
            "enableCORS = false",
            "enableXsrfProtection = false",
            "",
            "[browser]",
            "gatherUsageStats = false",
            ""
        ])
    )


_bootstrap_project_files()


# ============================================================
# Runtime configuration
# ============================================================
MAX_DOC_CHARS = int(os.environ.get("MAX_DOC_CHARS", "80000"))
MAX_CHUNK_WORDS = int(os.environ.get("MAX_CHUNK_WORDS", "900"))
LLM_RETRIES = int(os.environ.get("LLM_RETRIES", "5"))
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")

LOCAL_DATA_DIR = os.environ.get("LOCAL_DATA_DIR", "data").strip()
SQLITE_PATH = os.environ.get("SQLITE_PATH", os.path.join(LOCAL_DATA_DIR, "app.db")).strip()

OCR_LOW_CONF = int(os.environ.get("OCR_LOW_CONF", "60"))
OCR_MED_CONF = int(os.environ.get("OCR_MED_CONF", "80"))

BILLING_RATE_PER_1K = float(os.environ.get("BILLING_RATE_PER_1K", "0") or "0")


# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="GovCan Plain Language Converter", layout="wide")
st.title("🇨🇦 GovCan Plain Language Converter")



# ============================================================
# Hosted-mode warning (Streamlit Cloud demo: disk persistence not guaranteed)
# ============================================================
def _is_streamlit_cloud() -> bool:
    return bool(os.environ.get("STREAMLIT_SERVER_HEADLESS"))


if _is_streamlit_cloud():
    st.caption("Running in hosted mode. Note: local file storage (including SQLite) may not persist across reboots on Streamlit Community Cloud.")


# ============================================================
# ✅ Authentication (Option A: default provider)
# - Uses Streamlit built-in login (st.login / st.user)
# - Validates Option A Secrets shape: everything inside [auth]
# - Optional allowlists via ALLOWED_DOMAINS / ALLOWED_EMAILS (comma-separated)
# ============================================================

def _get_allowlists() -> Tuple[List[str], List[str]]:
    doms = [d.strip().lower() for d in str(st.secrets.get("ALLOWED_DOMAINS", "")).split(",") if d.strip()]
    ems = [e.strip().lower() for e in str(st.secrets.get("ALLOWED_EMAILS", "")).split(",") if e.strip()]
    return doms, ems


def _auth_missing_keys() -> List[str]:
    try:
        auth = st.secrets.get("auth")
        if not isinstance(auth, Mapping):
            return ["[auth]"]

        required = [
            "redirect_uri",
            "cookie_secret",
            "client_id",
            "client_secret",
            "server_metadata_url",
        ]
        return [f"[auth].{k}" for k in required if not auth.get(k)]
    except Exception:
        return ["[auth]"]


def _normalize_email_list(x) -> List[str]:
    if not x:
        return []
    return [e.strip().lower() for e in str(x).split(",") if e.strip()]


def _roles_config() -> Dict[str, List[str]]:
    r = st.secrets.get("roles", {})
    if not isinstance(r, Mapping):
        return {"admin": [], "reviewer": [], "editor": [], "viewer": []}

    return {
        "admin": _normalize_email_list(r.get("admin")),
        "reviewer": _normalize_email_list(r.get("reviewer")),
        "editor": _normalize_email_list(r.get("editor")),
        "viewer": _normalize_email_list(r.get("viewer")),
    }


def role_for_email(email: str) -> str:
    email = (email or "").lower()
    roles = _roles_config()

    if email in roles["admin"]:
        return "admin"
    if email in roles["reviewer"]:
        return "reviewer"
    if email in roles["editor"]:
        return "editor"
    if email in roles["viewer"]:
        return "viewer"
    return "viewer"


def require_login() -> str:
    if not hasattr(st, "login") or not hasattr(st, "user"):
        return ""

    missing = _auth_missing_keys()
    if missing:
        st.error("Auth is not configured correctly in Streamlit Cloud Secrets.")
        for m in missing:
            st.write(f"- {m}")
        st.stop()

    email = getattr(st.user, "email", "").strip().lower() if getattr(st, "user", None) else ""

    if email:
        doms, ems = _get_allowlists()

        # allow if email OR domain matches (when any allowlist is set)
        if ems or doms:
            allowed = False

            if ems and email in ems:
                allowed = True

            if (not allowed) and doms and "@" in email:
                if email.split("@", 1)[1] in doms:
                    allowed = True

            if not allowed:
                st.error(f"Access denied: {email} is not allowed.")
                if hasattr(st, "logout"):
                    st.logout()
                st.stop()

        return email   # ✅ INSIDE FUNCTION

    st.info("Please sign in to continue.")
    if st.button("Log in"):
        st.login()
    st.stop()


# --- run auth ---
AUTH_EMAIL = require_login()

# --- lock role from Secrets ---
if AUTH_EMAIL:
    st.session_state.auth_role = role_for_email(AUTH_EMAIL)

# ------------------------------------------------------------
# Sidebar: role display / admin override
# ------------------------------------------------------------
if AUTH_EMAIL:
    locked_role = st.session_state.get("auth_role", "viewer")

    if locked_role == "admin":
        st.session_state.auth_role = st.selectbox(
            "Role (admin can override for testing)",
            ["viewer", "editor", "reviewer", "admin"],
            index=["viewer", "editor", "reviewer", "admin"].index(locked_role),
        )
    else:
        st.caption("Role (locked)")
        st.write(f"**{locked_role}**")

    st.caption(f"Signed in as: **{AUTH_EMAIL}**")

    if hasattr(st, "logout"):
        if st.button("Logout", use_container_width=True):
            st.logout()

# ============================================================
# Auth roles / permissions
# ============================================================

ROLE_PERMS = {
    "admin":    {"convert", "export", "approve", "edit_outputs", "rollback", "analytics"},
    "reviewer": {"export", "approve", "edit_outputs"},
    "editor":   {"convert", "export", "edit_outputs"},
    "viewer":   set(),
}

def can(action: str) -> bool:
    role = st.session_state.get("auth_role", "viewer")
    return action in ROLE_PERMS.get(role, set())


# ============================================================
# Gemini client
# ============================================================
api_key = st.secrets.get("GEMINI_API_KEY", "")
if not api_key:
    st.error("❌ GEMINI_API_KEY missing in Streamlit secrets.")
    st.stop()

client = genai.Client(api_key=api_key)


# ============================================================
# Helpers
# ============================================================

def flesch_kincaid(text: str) -> float:
    if not text or len(text.strip()) < 30:
        return 99.0

    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    words = max(1, len(re.findall(r"\b\w+\b", text)))
    syllables = sum(_count_syllables(w) for w in re.findall(r"\b\w+\b", text))

    try:
        grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
        return round(max(0.0, grade), 1)
    except Exception:
        return 99.0


def _count_syllables(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev = False
    for c in word:
        is_vowel = c in vowels
        if is_vowel and not prev:
            count += 1
        prev = is_vowel
    if word.endswith("e"):
        count = max(1, count - 1)
    return max(1, count)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha12(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]

def clamp_text(text: str, max_chars: int) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))

def truncate_words(text: str, max_words: int) -> Tuple[str, bool]:
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]), True

def safe_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return name or "file"

def detect_lang(text: str) -> str:
    if not LANGDETECT_OK:
        return "unknown"
    t = (text or "").strip()
    if len(t) < 40:
        return "unknown"
    try:
        code = detect(t)
        if code.startswith("en"):
            return "en"
        if code.startswith("fr"):
            return "fr"
        return code or "unknown"
    except Exception:
        return "unknown"

def est_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return int(max(1, len(text) / 4))

def money_estimate(tokens_total: int) -> float:
    if BILLING_RATE_PER_1K <= 0:
        return 0.0
    return round((tokens_total / 1000.0) * BILLING_RATE_PER_1K, 6)


def analytics_summary(days: int = 30) -> Dict[str, Any]:
    since = (datetime.now(timezone.utc).timestamp() - days * 86400)
    since_iso = datetime.fromtimestamp(since, tz=timezone.utc).isoformat()

    conn = _db()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
          COUNT(*),
          COALESCE(SUM(est_tokens_in),0),
          COALESCE(SUM(est_tokens_out),0)
        FROM usage_events
        WHERE ts >= ?
        """,
        (since_iso,),
    )
    row = cur.fetchone() or (0, 0, 0)
    total_events, tin, tout = row[0], row[1], row[2]

    cur.execute(
        """
        SELECT user_email,
               COUNT(*) as cnt,
               COALESCE(SUM(est_tokens_in),0) as tin,
               COALESCE(SUM(est_tokens_out),0) as tout
        FROM usage_events
        WHERE ts >= ?
        GROUP BY user_email
        ORDER BY (tin+tout) DESC
        LIMIT 50
        """,
        (since_iso,),
    )
    per_user = cur.fetchall() or []

    conn.close()
    return {
        "since_iso": since_iso,
        "total_events": int(total_events or 0),
        "tokens_in": int(tin or 0),
        "tokens_out": int(tout or 0),
        "tokens_total": int((tin or 0) + (tout or 0)),
        "estimated_cost": money_estimate(int((tin or 0) + (tout or 0))),
        "per_user": per_user,
    }


def analytics_export_csv(days: int = 30) -> bytes:
    since = (datetime.now(timezone.utc).timestamp() - days * 86400)
    since_iso = datetime.fromtimestamp(since, tz=timezone.utc).isoformat()

    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ts, user_email, action, doc_id, section_id, model,
               prompt_chars, output_chars, est_tokens_in, est_tokens_out, meta_json
        FROM usage_events
        WHERE ts >= ?
        ORDER BY ts DESC
        """,
        (since_iso,),
    )
    rows = cur.fetchall() or []
    conn.close()

    out = io.StringIO()
    w = csv.writer(out)
    w.writerow([
        "ts", "user_email", "action", "doc_id", "section_id", "model",
        "prompt_chars", "output_chars", "est_tokens_in", "est_tokens_out", "meta_json"
    ])
    for r in rows:
        w.writerow(list(r))

    return out.getvalue().encode("utf-8")


# ============================================================
# SQLite storage + ✅ Usage Analytics tables
# ============================================================
def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(SQLITE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def _db_init() -> None:
    conn = _db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            latest_version_id INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            version_name TEXT NOT NULL,
            saved_at TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_versions_doc ON versions(doc_id, id DESC)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            user_email TEXT,
            action TEXT NOT NULL,
            doc_id TEXT,
            section_id INTEGER,
            model TEXT,
            prompt_chars INTEGER,
            output_chars INTEGER,
            est_tokens_in INTEGER,
            est_tokens_out INTEGER,
            meta_json TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_events(ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_user ON usage_events(user_email, ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_doc ON usage_events(doc_id, ts DESC)")
    conn.commit()
    conn.close()

_db_init()


def log_usage(
    action: str,
    user_email: str = "",
    doc_id: str = "",
    section_id: Optional[int] = None,
    model: str = "",
    prompt_text: str = "",
    output_text: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        prompt_chars = len(prompt_text or "")
        output_chars = len(output_text or "")
        tin = est_tokens_from_text(prompt_text or "")
        tout = est_tokens_from_text(output_text or "")
        meta_json = json.dumps(meta or {}, ensure_ascii=False)

        conn = _db()
        conn.execute(
            """
            INSERT INTO usage_events(ts, user_email, action, doc_id, section_id, model, prompt_chars, output_chars, est_tokens_in, est_tokens_out, meta_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (now_iso(), (user_email or "").lower(), action, doc_id, section_id, model, prompt_chars, output_chars, tin, tout, meta_json),
        )
        conn.commit()
        conn.close()
    except Exception:
        return


def _ensure_doc(doc_id: str) -> None:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,))
    row = cur.fetchone()
    if not row:
        ts = now_iso()
        cur.execute(
            "INSERT INTO documents(doc_id, created_at, updated_at, latest_version_id) VALUES(?,?,?,NULL)",
            (doc_id, ts, ts),
        )
        conn.commit()
    conn.close()


def save_version(doc_id: str, snapshot: Dict[str, Any]) -> str:
    _ensure_doc(doc_id)
    tsname = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_name = f"{tsname}.json"
    saved_at = snapshot.get("saved_at") or now_iso()

    raw = json.dumps(snapshot, ensure_ascii=False)
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO versions(doc_id, version_name, saved_at, snapshot_json) VALUES(?,?,?,?)",
        (doc_id, version_name, saved_at, raw),
    )
    vid = cur.lastrowid
    cur.execute(
        "UPDATE documents SET updated_at = ?, latest_version_id = ? WHERE doc_id = ?",
        (now_iso(), vid, doc_id),
    )
    conn.commit()
    conn.close()
    return version_name


def list_versions(doc_id: str) -> List[str]:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT version_name FROM versions WHERE doc_id = ? ORDER BY id DESC", (doc_id,))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows] if rows else []


def load_latest(doc_id: str) -> Optional[Dict[str, Any]]:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT latest_version_id FROM documents WHERE doc_id = ?", (doc_id,))
    row = cur.fetchone()
    if not row or not row[0]:
        conn.close()
        return None
    latest_id = row[0]
    cur.execute("SELECT snapshot_json FROM versions WHERE id = ?", (latest_id,))
    row2 = cur.fetchone()
    conn.close()
    if not row2:
        return None
    try:
        return json.loads(row2[0])
    except Exception:
        return None


def load_version(doc_id: str, version_name: str) -> Optional[Dict[str, Any]]:
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "SELECT snapshot_json FROM versions WHERE doc_id = ? AND version_name = ? ORDER BY id DESC LIMIT 1",
        (doc_id, version_name),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def set_latest(doc_id: str, version_name: str) -> bool:
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM versions WHERE doc_id = ? AND version_name = ? ORDER BY id DESC LIMIT 1",
        (doc_id, version_name),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return False

    vid = row[0]
    cur.execute(
        "UPDATE documents SET updated_at = ?, latest_version_id = ? WHERE doc_id = ?",
        (now_iso(), vid, doc_id),
    )
    conn.commit()
    conn.close()
    return True


# ============================================================
# Tesseract path (Windows dev helper)
# ============================================================
def find_tesseract_exe_windows() -> Optional[str]:
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Programs\Tesseract-OCR\tesseract.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Tesseract-OCR\tesseract.exe"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None

_exe = find_tesseract_exe_windows()
if _exe:
    pytesseract.pytesseract.tesseract_cmd = _exe


# ============================================================
# OCR + Confidence
# ============================================================
def pdf_to_images(pdf_bytes: bytes, pages: int, dpi: int = 220) -> List[Image.Image]:
    doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    out: List[Image.Image] = []
    end = min(len(doc), pages)
    scale = dpi / 72
    for i in range(end):
        page = doc[i]
        out.append(page.render(scale=scale).to_pil())
    return out

def _tess_lang_for_detected(code: str) -> str:
    if code == "fr":
        return "fra+osd"
    if code == "en":
        return "eng+osd"
    return "eng+fra+osd"

def ocr_image_with_conf(img: Image.Image, tess_lang: str) -> Dict[str, Any]:
    config = f"--oem 3 --psm 3 -l {tess_lang}"
    text = (pytesseract.image_to_string(img, config=config) or "").strip()

    try:
        data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
    except Exception:
        safe_html = "<div style='white-space:pre-wrap'>" + (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</div>"
        return {
            "text": text,
            "html": safe_html,
            "stats": {"words": 0, "avg_conf": 0.0, "low": 0, "med": 0, "high": 0, "conf_available": False},
        }

    n = len(data.get("text", []))
    words_out: List[str] = []
    low = med = high = 0
    conf_sum = 0.0
    conf_count = 0

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    for i in range(n):
        w = (data["text"][i] or "").strip()
        if not w:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        if conf >= 0:
            conf_sum += conf
            conf_count += 1

        if conf < 0:
            words_out.append(esc(w))
            continue

        if conf < OCR_LOW_CONF:
            low += 1
            words_out.append(
                f"<span style='background:rgba(255,0,0,0.18); padding:0 2px; border-radius:4px' title='OCR conf: {conf:.0f}'>"
                f"{esc(w)}</span>"
            )
        elif conf < OCR_MED_CONF:
            med += 1
            words_out.append(
                f"<span style='background:rgba(255,165,0,0.18); padding:0 2px; border-radius:4px' title='OCR conf: {conf:.0f}'>"
                f"{esc(w)}</span>"
            )
        else:
            high += 1
            words_out.append(esc(w))

    avg = (conf_sum / conf_count) if conf_count else 0.0
    html = "<div style='white-space:pre-wrap; line-height:1.6'>" + " ".join(words_out) + "</div>"
    return {
        "text": text,
        "html": html,
        "stats": {"words": conf_count, "avg_conf": round(avg, 2), "low": low, "med": med, "high": high, "conf_available": True},
    }

def extract_pdf(pdf_bytes: bytes, pages: int) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "pages_requested": pages,
        "text_chars": 0,
        "ocr_chars": 0,
        "ocr_used": False,
        "ocr_failed": False,
        "tesseract_cmd": getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract"),
        "langdetect_ok": LANGDETECT_OK,
        "langdetect_err": LANGDETECT_ERR,
        "python_exe": sys.executable,
        "detected_lang": "unknown",
        "ocr_detected_lang": "unknown",
        "ocr_conf": {},
    }

    text = ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for i, p in enumerate(reader.pages):
            if i >= pages:
                break
            t = (p.extract_text() or "").strip()
            if t:
                text += t + "\n"
    except Exception:
        text = ""

    text = text.strip()
    meta["text_chars"] = len(text)
    if text:
        meta["detected_lang"] = detect_lang(text)
        return text, meta

    try:
        imgs = pdf_to_images(pdf_bytes, pages, dpi=220)

        first_pass = ocr_image_with_conf(imgs[0], "eng+fra+osd")
        sample = (first_pass.get("text") or "")[:2000]
        ocr_lang = detect_lang(sample)
        meta["ocr_detected_lang"] = ocr_lang

        tess_lang = _tess_lang_for_detected(ocr_lang)

        all_text_parts: List[str] = []
        html_pages: List[str] = []
        stats_pages: List[Dict[str, Any]] = []

        for img in imgs:
            page_res = ocr_image_with_conf(img, tess_lang)
            all_text_parts.append(page_res.get("text", ""))
            html_pages.append(page_res.get("html", ""))
            stats_pages.append(page_res.get("stats", {}))

        ocr_text = "\n\n".join([t for t in all_text_parts if t is not None]).strip()

        meta["ocr_chars"] = len(ocr_text)
        meta["ocr_used"] = True
        meta["detected_lang"] = detect_lang(ocr_text)
        meta["ocr_conf"] = {
            "tess_lang_used": tess_lang,
            "pages": stats_pages,
            "html_pages": html_pages,
            "thresholds": {"low": OCR_LOW_CONF, "med": OCR_MED_CONF},
        }
        return ocr_text, meta
    except Exception:
        meta["ocr_failed"] = True
        return "", meta

def extract_docx(docx_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    d = Document(io.BytesIO(docx_bytes))
    txt = "\n".join(p.text for p in d.paragraphs if p.text.strip()).strip()
    meta = {
        "pages_requested": None,
        "text_chars": len(txt),
        "ocr_chars": 0,
        "ocr_used": False,
        "ocr_failed": False,
        "tesseract_cmd": getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract"),
        "langdetect_ok": LANGDETECT_OK,
        "langdetect_err": LANGDETECT_ERR,
        "python_exe": sys.executable,
        "detected_lang": detect_lang(txt),
        "ocr_detected_lang": "unknown",
        "ocr_conf": {},
    }
    return txt, meta


# ============================================================
# Headings detection
# ============================================================
HEADING_PAT = re.compile(
    r"^\s*(\d+(\.\d+)*|SECTION|PART|PURPOSE|SCOPE|DEFINITIONS|BACKGROUND|POLICY|RESPONSIBILITIES|APPENDIX|ANNEX)\b",
    re.I
)

def split_by_headings(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    out: List[Tuple[str, str]] = []
    title: Optional[str] = None
    body: List[str] = []

    for l in lines:
        if HEADING_PAT.match(l.strip()):
            if title and "".join(body).strip():
                out.append((title.strip(), "\n".join(body).strip()))
            title, body = l.strip(), []
        else:
            title = title or "Preamble"
            body.append(l)

    if title and "".join(body).strip():
        out.append((title.strip(), "\n".join(body).strip()))

    return out if out else [("Preamble", text.strip())]


# ============================================================
# LLM (rate-limit safe + caps BEFORE calling) + ✅ usage logging
# ============================================================
def safe_generate(prompt: str, retries: int = LLM_RETRIES):
    delay = 2
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            return client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config={"temperature": 0.1, "max_output_tokens": 550},
            )
        except Exception as e:
            last_exc = e
            if DEBUG_LLM:
                st.sidebar.error(f"LLM error attempt {attempt}/{retries}")
                st.sidebar.code(repr(e))
            time.sleep(delay)
            delay = min(delay * 2, 30)

    # Always show the final error (even if DEBUG off)
    st.error("LLM call failed after retries.")
    st.code(repr(last_exc) if last_exc else "Unknown error")
    return None

def _response_text(resp) -> str:
    """Aggressive best-effort extraction of text from google-genai response."""
    if resp is None:
        return ""

    # 1) Normal path
    t = getattr(resp, "text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    # 2) Candidates path
    cands = getattr(resp, "candidates", None)
    if cands:
        for c in cands:
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) or []
            for p in parts:
                pt = getattr(p, "text", None)
                if isinstance(pt, str) and pt.strip():
                    return pt.strip()

    # 3) Some versions store output deeper
    try:
        d = resp.__dict__
        for k in ("output_text", "result", "response", "data"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    except Exception:
        pass

    # 4) LAST resort: stringify the whole object so you can see *something*
    s = str(resp)
    return s.strip() if isinstance(s, str) else ""
    
def _extract_json_object(raw: str) -> Optional[dict]:
    if not raw:
        return None

    s = raw.strip()

    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)

    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None

    chunk = m.group(0).strip()
    last = chunk.rfind("}")
    if last != -1:
        chunk = chunk[: last + 1]

    try:
        return json.loads(chunk)
    except Exception:
        return None

        # fallback: save something visible instead of blanks
    return {
        "en": capped_text.strip(),
        "fr": capped_text.strip(),
        "grade_en": 99.0,
        "grade_fr": 0.0,
        "truncated": truncated,
        "source_lang": source_lang,
    }


    chunk = m.group(0).strip()

    # remove trailing junk after last brace if needed
    last = chunk.rfind("}")
    if last != -1:
        chunk = chunk[: last + 1]

    try:
        return json.loads(chunk)
    except Exception:
        return None

def convert_chunk(text: str, source_lang: str, doc_id: str, section_id: int, user_email: str) -> Dict[str, Any]:
    capped_text, truncated = truncate_words(text, MAX_CHUNK_WORDS)

    best = {
        "en": "",
        "fr": "",
        "grade_en": 99.0,
        "grade_fr": 0.0,
        "truncated": truncated,
        "source_lang": source_lang,
    }

    lang_hint = source_lang if source_lang and source_lang != "unknown" else "unknown"
    if lang_hint == "fr":
        extra_rules = "- Source text is French. Produce English plain-language translation + French plain-language rewrite.\n"
    elif lang_hint == "en":
        extra_rules = "- Source text is English. Produce English plain-language rewrite + French translation.\n"
    else:
        extra_rules = "- Source text language may be mixed/unknown. Preserve meaning; translate as needed.\n"

    for attempt in range(1, 4):
        prompt = f"""
Return JSON ONLY: {{ "en": "...", "fr": "..." }}

Rules:
- English Grade ≤ 8 (hard target)
- Short sentences. Active voice.
- Explain terms.
- EN and FR must match meaning.
- No extra text outside JSON.
{extra_rules}
SOURCE_LANG_DETECTED: {lang_hint}

TEXT:
{capped_text}
""".strip()

        r = safe_generate(prompt)
        raw_out = _response_text(r)

        log_usage(
            action="llm_convert",
            user_email=user_email,
            doc_id=doc_id,
            section_id=section_id,
            model=LLM_MODEL,
            prompt_text=prompt,
            output_text=raw_out,
            meta={"attempt": attempt, "source_lang": lang_hint, "truncated": truncated},
        )

        data = _extract_json_object(raw_out)
        if not isinstance(data, dict):
            continue

        en = (data.get("en") or "").strip()
        fr = (data.get("fr") or "").strip()

        # ✅ If model gave text, KEEP IT no matter what readability does
        if en and fr:
            # Compute readability, but don't let it wipe outputs
            try:
                ge = float(flesch_kincaid(en))
            except Exception:
                ge = 99.0
            try:
                gf = float(french_readability(fr))
            except Exception:
                gf = 0.0

            # ✅ Update best even if ge=99, as long as best is empty OR grade improved
            if (not best["en"]) or (ge < best["grade_en"]):
                best = {
                    "en": en,
                    "fr": fr,
                    "grade_en": ge,
                    "grade_fr": gf,
                    "truncated": truncated,
                    "source_lang": source_lang,
                }

            if ge <= 8:
                return best

        # If JSON exists but one side missing, try again
        continue

    # ✅ FINAL SAFETY: never return blank outputs
    if not best["en"] or not best["fr"]:
        fallback = capped_text.strip()
        best["en"] = best["en"] or fallback
        best["fr"] = best["fr"] or fallback

    return best






# ============================================================
# Compliance helpers + Watermarking
# ============================================================
def overall_doc_status(snapshot: Dict[str, Any]) -> str:
    """APPROVED only if every section is approved."""
    secs = snapshot.get("sections", []) or []
    if secs and all((s.get("status") == "approved") for s in secs):
        return "APPROVED"
    return "DRAFT"

def compliance_stats(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    secs = snapshot.get("sections", []) or []
    total = len(secs)
    approved = sum(1 for s in secs if s.get("status") == "approved")
    reviewed = sum(1 for s in secs if s.get("status") == "reviewed")
    draft = sum(1 for s in secs if s.get("status") == "draft")

    grades = []
    over8 = 0
    for s in secs:
        try:
            ge = float(s.get("grade_en", 99.0) or 99.0)
        except Exception:
            ge = 99.0
        grades.append(ge)
        if ge > 8:
            over8 += 1

    truncated = sum(1 for s in secs if s.get("truncated"))

    reviewers = sorted({(s.get("reviewer") or "").strip() for s in secs if (s.get("reviewer") or "").strip()})

    meta = snapshot.get("meta", {}) or {}
    return {
        "total_sections": total,
        "approved": approved,
        "reviewed": reviewed,
        "draft": draft,
        "over_grade8": over8,
        "avg_grade_en": round(sum(grades) / max(1, len(grades)), 2) if grades else 0.0,
        "truncated_sections": truncated,
        "reviewers": reviewers,
        "ocr_used": bool(meta.get("ocr_used", False)),
        "ocr_failed": bool(meta.get("ocr_failed", False)),
        "detected_lang": meta.get("detected_lang", "unknown"),
        "pages_requested": meta.get("pages_requested", ""),
    }

def pdf_watermark(c: canvas.Canvas, text: str, width: float, height: float) -> None:
    """Light diagonal watermark across the page."""
    if not text:
        return
    c.saveState()
    try:
        c.setFillAlpha(0.10)
    except Exception:
        pass
    c.setFont("Helvetica-Bold", 72)
    c.translate(width / 2, height / 2)
    c.rotate(35)
    c.drawCentredString(0, 0, text.upper())
    c.restoreState()


# ============================================================
# Export builders
# ============================================================
def build_docx(snapshot: Dict[str, Any]) -> bytes:
    doc = Document()
    doc.add_heading("GovCan Plain Language Converter", 1)

    # ✅ Reliable DOCX "watermark": status banner
    status = overall_doc_status(snapshot)
    banner = doc.add_paragraph(f"STATUS: {status}")
    banner.runs[0].bold = True

    meta = snapshot.get("meta", {})
    doc.add_paragraph(f"Document ID: {snapshot.get('doc_id','')}")
    doc.add_paragraph(f"Saved: {snapshot.get('saved_at','')}")
    doc.add_paragraph(f"Detected language: {meta.get('detected_lang', 'unknown')}")
    doc.add_paragraph(f"OCR used: {meta.get('ocr_used', False)} | OCR failed: {meta.get('ocr_failed', False)}")
    doc.add_paragraph(f"Pages requested: {meta.get('pages_requested', '')}")

    ocr_conf = meta.get("ocr_conf", {}) if isinstance(meta.get("ocr_conf", {}), dict) else {}
    if ocr_conf.get("pages"):
        pages = ocr_conf.get("pages", [])
        avg_list = [p.get("avg_conf", 0) for p in pages if isinstance(p, dict)]
        avg = round(sum(avg_list) / max(1, len(avg_list)), 2) if avg_list else 0.0
        doc.add_paragraph(f"OCR avg confidence (approx): {avg}")

    doc.add_paragraph("")

    for s in snapshot.get("sections", []):
        doc.add_heading(s.get("title", "Section"), 2)
        doc.add_paragraph(f"Status: {s.get('status','draft')}")
        doc.add_paragraph(f"Reviewer: {s.get('reviewer','')}")
        doc.add_paragraph(f"Comment: {s.get('comment','')}")
        if s.get("approved_at"):
            doc.add_paragraph(f"Approved at: {s.get('approved_at','')}")

        doc.add_paragraph("")
        doc.add_paragraph("Original").runs[0].bold = True
        doc.add_paragraph(s.get("original", ""))

        doc.add_paragraph("English (Plain Language)").runs[0].bold = True
        doc.add_paragraph(s.get("en", ""))

        doc.add_paragraph("Français (Langage clair)").runs[0].bold = True
        doc.add_paragraph(s.get("fr", ""))

        doc.add_paragraph(f"EN Grade: {s.get('grade_en','')} | FR Readability: {s.get('grade_fr','')}")
        if s.get("truncated"):
            note = doc.add_paragraph(f"NOTE: Section truncated to {MAX_CHUNK_WORDS} words before LLM call.")
            note.runs[0].italic = True

        doc.add_page_break()

    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

def build_pdf(snapshot: Dict[str, Any]) -> bytes:
    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=letter)
    width, height = letter

    # ✅ Watermark DRAFT vs APPROVED
    status = overall_doc_status(snapshot)
    pdf_watermark(c, status, width, height)

    def draw_wrapped(text: str, x: float, y: float, max_width: float, leading: float = 12) -> float:
        words = text.split()
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if c.stringWidth(test) <= max_width:
                line = test
            else:
                c.drawString(x, y, line)
                y -= leading
                line = w
                if y < 1 * inch:
                    c.showPage()
                    pdf_watermark(c, status, width, height)
                    y = height - 1 * inch
        if line:
            c.drawString(x, y, line)
            y -= leading
        return y

    y = height - 1 * inch
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, y, "GovCan Plain Language Converter")
    y -= 18

    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, y, f"Document ID: {snapshot.get('doc_id','')}")
    y -= 14
    c.drawString(1 * inch, y, f"Saved: {snapshot.get('saved_at','')}")
    y -= 14

    meta = snapshot.get("meta", {})
    c.drawString(1 * inch, y, f"Detected language: {meta.get('detected_lang','unknown')}")
    y -= 14
    c.drawString(1 * inch, y, f"OCR used: {meta.get('ocr_used', False)} | OCR failed: {meta.get('ocr_failed', False)} | Pages: {meta.get('pages_requested','')}")
    y -= 18

    for s in snapshot.get("sections", []):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1 * inch, y, s.get("title", "Section"))
        y -= 16

        c.setFont("Helvetica", 10)
        c.drawString(1 * inch, y, f"Status: {s.get('status','draft')} | Reviewer: {s.get('reviewer','')}")
        y -= 14

        if s.get("comment"):
            y = draw_wrapped(f"Comment: {s.get('comment','')}", 1*inch, y, width - 2*inch, leading=12)

        if s.get("approved_at"):
            y = draw_wrapped(f"Approved at: {s.get('approved_at','')}", 1*inch, y, width - 2*inch, leading=12)

        y -= 8
        c.setFont("Helvetica-Bold", 10)
        c.drawString(1 * inch, y, "Original")
        y -= 12
        c.setFont("Helvetica", 9)
        y = draw_wrapped(s.get("original", ""), 1*inch, y, width - 2*inch, leading=11)

        y -= 6
        c.setFont("Helvetica-Bold", 10)
        c.drawString(1 * inch, y, "English (Plain Language)")
        y -= 12
        c.setFont("Helvetica", 9)
        y = draw_wrapped(s.get("en", ""), 1*inch, y, width - 2*inch, leading=11)

        y -= 6
        c.setFont("Helvetica-Bold", 10)
        c.drawString(1 * inch, y, "Français (Langage clair)")
        y -= 12
        c.setFont("Helvetica", 9)
        y = draw_wrapped(s.get("fr", ""), 1*inch, y, width - 2*inch, leading=11)

        y -= 10
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(1 * inch, y, f"EN Grade: {s.get('grade_en','')} | FR Readability: {s.get('grade_fr','')}")
        y -= 20

        c.showPage()
        pdf_watermark(c, status, width, height)
        y = height - 1 * inch

    c.save()
    return buff.getvalue()

def build_compliance_report_pdf(snapshot: Dict[str, Any]) -> bytes:
    buff = io.BytesIO()
    c = canvas.Canvas(buff, pagesize=letter)
    width, height = letter

    status = overall_doc_status(snapshot)
    pdf_watermark(c, status, width, height)

    stats = compliance_stats(snapshot)

    y = height - 1 * inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, y, "Compliance Report — GovCan Plain Language Converter")
    y -= 22

    c.setFont("Helvetica", 11)
    c.drawString(1 * inch, y, f"Document ID: {snapshot.get('doc_id','')}")
    y -= 14
    c.drawString(1 * inch, y, f"Version saved at (UTC): {snapshot.get('saved_at','')}")
    y -= 14
    c.drawString(1 * inch, y, f"Overall status: {status}")
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, y, "Summary")
    y -= 16

    c.setFont("Helvetica", 11)
    lines = [
        f"Sections total: {stats['total_sections']}",
        f"Approved: {stats['approved']} | Reviewed: {stats['reviewed']} | Draft: {stats['draft']}",
        f"English grade (avg): {stats['avg_grade_en']} | Sections above Grade 8: {stats['over_grade8']}",
        f"Sections truncated before LLM call: {stats['truncated_sections']}",
        f"Detected language: {stats['detected_lang']} | Pages requested: {stats['pages_requested']}",
        f"OCR used: {stats['ocr_used']} | OCR failed: {stats['ocr_failed']}",
        f"Reviewers: {', '.join(stats['reviewers']) if stats['reviewers'] else '—'}",
    ]
    for line in lines:
        c.drawString(1 * inch, y, line)
        y -= 14

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, y, "Per-section compliance")
    y -= 16

    c.setFont("Helvetica", 10)
    for s in snapshot.get("sections", []) or []:
        title = (s.get("title") or "Section").strip()
        stt = s.get("status", "draft")
        try:
            ge = float(s.get("grade_en", 99.0) or 99.0)
        except Exception:
            ge = 99.0
        ok = "OK" if ge <= 8 else "OVER"
        trunc = "TRUNC" if s.get("truncated") else ""
        row = f"- {title} | status={stt} | grade_en={ge} ({ok}) {trunc}".strip()

        if y < 1 * inch:
            c.showPage()
            pdf_watermark(c, status, width, height)
            y = height - 1 * inch
            c.setFont("Helvetica", 10)

        c.drawString(1 * inch, y, row[:120])
        y -= 12

    c.save()
    return buff.getvalue()


# ============================================================
# Session state init
# ============================================================
if "doc_id" not in st.session_state:
    st.session_state.doc_id = ""
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "extract_meta" not in st.session_state:
    st.session_state.extract_meta = {}
if "last_extract_key" not in st.session_state:
    st.session_state.last_extract_key = ""
if "page_preview_imgs" not in st.session_state:
    st.session_state.page_preview_imgs = []


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Controls")

    st.caption("Role (locked from Secrets)")
    st.write(f"**{st.session_state.get('auth_role','viewer')}**")

    if hasattr(st, "logout") and st.button("Logout", use_container_width=True):
        st.logout()

    st.divider()
    st.caption("Storage mode")
    st.success(f"SQLite ✅ ({SQLITE_PATH})")

    st.divider()
    doc_id_in = st.text_input(
        "Document ID",
        value=st.session_state.doc_id,
        placeholder="e.g., client-abc-001"
    )

    # ✅ FIX: changing document id should not lock extraction cache
    if doc_id_in != st.session_state.doc_id:
        st.session_state.doc_id = doc_id_in.strip()
        st.session_state.snapshot = None
        st.session_state.last_extract_key = ""  # ✅ IMPORTANT

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Load latest", use_container_width=True):
            if st.session_state.doc_id:
                snap = load_latest(st.session_state.doc_id)
                st.session_state.snapshot = snap
                st.session_state.last_extract_key = ""
                if snap:
                    st.session_state.input_text = snap.get("source_text", "")
                    st.session_state.extract_meta = snap.get("meta", {})
                st.rerun()
            else:
                st.error("Enter Document ID first.")

    with c2:
        if st.button("New doc", use_container_width=True):
            st.session_state.doc_id = f"doc-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            st.session_state.snapshot = None
            st.session_state.input_text = ""
            st.session_state.extract_meta = {}
            st.session_state.last_extract_key = ""
            st.session_state.page_preview_imgs = []
            st.rerun()

    st.divider()
    if st.session_state.doc_id:
        versions = list_versions(st.session_state.doc_id)
        if versions:
            vsel = st.selectbox("Version history", versions, index=0)
            if can("rollback") and st.button("Rollback to selected", use_container_width=True):
                ok = set_latest(st.session_state.doc_id, vsel)
                if ok:
                    snap = load_latest(st.session_state.doc_id)
                    st.session_state.snapshot = snap
                    st.session_state.last_extract_key = ""
                    if snap:
                        st.session_state.input_text = snap.get("source_text", "")
                        st.session_state.extract_meta = snap.get("meta", {})
                    st.success(f"Rolled back to {vsel}")
                    st.rerun()
                else:
                    st.error("Rollback failed (version not found).")
        else:
            st.caption("No versions yet.")

    # ✅ Admin analytics panel (billing info)
    if can("analytics"):
        st.divider()
        st.subheader("Usage Analytics")
        days = st.slider("Lookback (days)", 1, 180, 30)
        summ = analytics_summary(days=days)
        st.caption(f"Since: {summ['since_iso']}")
        st.write(f"Events: **{summ['total_events']}**")
        st.write(
            f"Tokens (in/out): **{summ['tokens_in']} / {summ['tokens_out']}**  | "
            f"Total: **{summ['tokens_total']}**"
        )

        if BILLING_RATE_PER_1K > 0:
            st.write(f"Estimated cost (@ {BILLING_RATE_PER_1K}/1K): **{summ['estimated_cost']}**")
        else:
            st.caption("Tip: set env BILLING_RATE_PER_1K to show cost estimate.")

        if summ["per_user"]:
            st.caption("Top users (by tokens):")
            for (email, cnt, tin, tout) in summ["per_user"][:10]:
                st.write(f"- {email or '(unknown)'} | events={cnt} | tokens={int(tin)+int(tout)}")

        csv_bytes = analytics_export_csv(days=days)
        st.download_button(
            "Download usage CSV",
            data=csv_bytes,
            file_name=f"usage_events_last_{days}_days.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================================
# Main UI
# ============================================================
left, right = st.columns(2)

with left:
  up = st.file_uploader("Upload PDF or DOCX", ["pdf", "docx"])
  pages = st.slider("PDF pages to extract/preview", 1, 25, 3)

  preview = st.toggle("Preview pages", value=False)
  show_ocr_conf = st.toggle("Show OCR confidence highlighting (OCR only)", value=False)

  if up:
      b = up.getvalue()
      key = f"{up.name}:{sha12(b)}:pages={pages}"

      should_extract = (key != st.session_state.last_extract_key) or (not st.session_state.input_text.strip())
      if should_extract:
          with st.spinner("Extracting text (OCR if needed)..."):
              if up.name.lower().endswith(".pdf"):
                  txt, meta = extract_pdf(b, pages)
                  if preview:
                      try:
                          st.session_state.page_preview_imgs = pdf_to_images(b, pages, dpi=160)
                      except Exception:
                          st.session_state.page_preview_imgs = []
              else:
                  txt, meta = extract_docx(b)
                  st.session_state.page_preview_imgs = []

              txt, truncated_doc = clamp_text(txt, MAX_DOC_CHARS)
              meta["doc_truncated"] = truncated_doc
              meta["doc_chars_after_cap"] = len(txt)

              st.session_state.last_extract_key = key
              st.session_state.extract_meta = meta
              st.session_state.input_text = txt

  meta = st.session_state.extract_meta or {}
  if meta:
      if meta.get("ocr_failed"):
          st.error("OCR failed ❌ (check tesseract install / language packs)")
      elif meta.get("ocr_used"):
          st.success(
              f"OCR used ✅ — OCR lang guess: {meta.get('ocr_detected_lang','unknown')} | "
              f"Detected: {meta.get('detected_lang','unknown')}"
          )
      else:
          st.info(f"Native extraction used ✅ — Detected: {meta.get('detected_lang','unknown')}")

      if meta.get("doc_truncated"):
          st.warning(f"Input capped at {MAX_DOC_CHARS} characters before LLM calls.")

      if not LANGDETECT_OK:
          st.warning("Language detection library not available (langdetect).")
          if LANGDETECT_ERR:
              st.code(LANGDETECT_ERR)
          st.caption(f"Python used by Streamlit: {sys.executable}")

  if show_ocr_conf and meta and meta.get("ocr_used") and isinstance(meta.get("ocr_conf", {}), dict):
      ocr_conf = meta.get("ocr_conf", {})
      html_pages = ocr_conf.get("html_pages", [])
      thr = ocr_conf.get("thresholds", {"low": OCR_LOW_CONF, "med": OCR_MED_CONF})

      st.subheader("OCR confidence highlighting")
      st.caption(f"Red < {thr.get('low')} | Orange < {thr.get('med')} (hover words to see confidence)")
      if html_pages:
          for i, html in enumerate(html_pages, start=1):
              st.markdown(f"**Page {i}**")
              st.markdown(html, unsafe_allow_html=True)
              st.divider()
      else:
          st.info("No OCR confidence data available.")

  if preview and st.session_state.page_preview_imgs:
      st.subheader("Page preview")
      for i, img in enumerate(st.session_state.page_preview_imgs, start=1):
          st.image(img, caption=f"Page {i}", use_container_width=True)

  user_text = st.text_area("Input text", height=320, key="input_text")

  if not can("convert"):
      st.info("You don’t have permission to convert. Ask an admin to grant editor/admin role.")

  convert_disabled = (not can("convert")) or (not st.session_state.doc_id) or (not user_text.strip())
  convert_btn = st.button("Convert & Save Version", type="primary", disabled=convert_disabled)


with right:
  if convert_btn:
      safe_text, _ = clamp_text(user_text, MAX_DOC_CHARS)

      doc_lang = detect_lang(safe_text)
      st.session_state.extract_meta = st.session_state.extract_meta or {}
      st.session_state.extract_meta["detected_lang"] = st.session_state.extract_meta.get("detected_lang") or doc_lang

      chunks = split_by_headings(safe_text)

      out_sections: List[Dict[str, Any]] = []
      prog = st.progress(0)
      for i, (title, body) in enumerate(chunks, start=1):
          chunk_lang = detect_lang(body) if len(body) >= 80 else doc_lang
          res = convert_chunk(
              body,
              source_lang=chunk_lang,
              doc_id=st.session_state.doc_id,
              section_id=i - 1,
              user_email=AUTH_EMAIL,
          )

          out_sections.append({
              "id": i - 1,
              "title": title,
              "original": body,
              "en": res["en"],
              "fr": res["fr"],
              "grade_en": res["grade_en"],
              "grade_fr": res["grade_fr"],
              "truncated": res["truncated"],
              "source_lang": res.get("source_lang", chunk_lang),
              "status": "draft",
              "reviewer": "",
              "comment": "",
              "approved_at": "",
          })
          prog.progress(int(i / max(1, len(chunks)) * 100))
      prog.empty()

      snap = {
          "doc_id": st.session_state.doc_id,
          "saved_at": now_iso(),
          "meta": st.session_state.extract_meta or {},
          "source_text": safe_text,
          "sections": out_sections,
      }

      save_version(st.session_state.doc_id, snap)
      st.session_state.snapshot = snap

      log_usage(
          action="save_version_after_convert",
          user_email=AUTH_EMAIL,
          doc_id=st.session_state.doc_id,
          section_id=None,
          model=LLM_MODEL,
          prompt_text="",
          output_text="",
          meta={"sections": len(out_sections), "doc_chars": len(safe_text), "doc_words": word_count(safe_text)},
      )

      st.success("Saved ✅ (SQLite versioned; approvals/comments persist)")

  snap = st.session_state.snapshot
  if not snap:
      st.caption("Upload a document, enter a Document ID, then Convert.")
  else:
      st.subheader("Reviewer workflow + exports")
      st.caption(f"Overall document status: **{overall_doc_status(snap)}**")

      sections = snap.get("sections", [])
      for s in sections:
          st.markdown(f"## {s.get('title','Section')}")

          ge = float(s.get("grade_en", 99.0) or 99.0)
          if ge <= 8:
              st.success(f"EN Grade {ge} ✔")
          else:
              st.warning(f"EN Grade {ge} (above 8)")

          st.info(f"FR Readability: {s.get('grade_fr', 0.0)}")
          st.caption(f"Source lang detected: {s.get('source_lang','unknown')}")

          if s.get("truncated"):
              st.warning(f"Section truncated to {MAX_CHUNK_WORDS} words before LLM call.")

          if can("edit_outputs"):
              s["en"] = st.text_area("English output", value=s.get("en", ""), key=f"en_{s['id']}", height=140)
              s["fr"] = st.text_area("French output", value=s.get("fr", ""), key=f"fr_{s['id']}", height=140)
          else:
              st.markdown("### English")
              st.write(s.get("en", ""))
              st.markdown("### Français")
              st.write(s.get("fr", ""))

          if can("approve"):
              s["reviewer"] = st.text_input("Reviewer name", value=s.get("reviewer", ""), key=f"rev_{s['id']}")
              s["comment"] = st.text_area("Reviewer comment", value=s.get("comment", ""), key=f"com_{s['id']}", height=90)

              status_options = ["draft", "reviewed", "approved"]
              s["status"] = st.selectbox(
                  "Status",
                  status_options,
                  index=status_options.index(s.get("status", "draft")),
                  key=f"stat_{s['id']}"
              )

              if s["status"] == "approved" and not s.get("approved_at"):
                  s["approved_at"] = now_iso()
              if s["status"] != "approved":
                  s["approved_at"] = ""
          else:
              st.caption("Approval locked (reviewer/admin only).")

          st.divider()

      c1, c2, c3, c4 = st.columns(4)

      with c1:
          if st.button("Save changes as new version", disabled=not st.session_state.doc_id):
              snap["saved_at"] = now_iso()
              snap["sections"] = sections
              save_version(st.session_state.doc_id, snap)
              st.session_state.snapshot = snap

              log_usage(
                  action="save_version_after_review",
                  user_email=AUTH_EMAIL,
                  doc_id=st.session_state.doc_id,
                  section_id=None,
                  model="",
                  meta={"overall_status": overall_doc_status(snap)},
              )
              st.success("Saved updated version ✅")

      with c2:
          if can("export"):
              docx_bytes = build_docx(snap)
              st.download_button(
                  "Download DOCX",
                  data=docx_bytes,
                  file_name=f"{safe_filename(st.session_state.doc_id)}.docx",
                  mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                  use_container_width=True
              )
              log_usage(action="export_docx", user_email=AUTH_EMAIL, doc_id=st.session_state.doc_id, model="", meta={"bytes": len(docx_bytes)})
          else:
              st.caption("Export locked (editor/reviewer/admin only).")

      with c3:
          if can("export"):
              pdf_bytes = build_pdf(snap)
              st.download_button(
                  "Download PDF",
                  data=pdf_bytes,
                  file_name=f"{safe_filename(st.session_state.doc_id)}.pdf",
                  mime="application/pdf",
                  use_container_width=True
              )
              log_usage(action="export_pdf_full", user_email=AUTH_EMAIL, doc_id=st.session_state.doc_id, model="", meta={"bytes": len(pdf_bytes)})
          else:
              st.caption("Export locked (editor/reviewer/admin only).")

      with c4:
          if can("export"):
              comp_pdf = build_compliance_report_pdf(snap)
              st.download_button(
                  "Compliance Report (PDF)",
                  data=comp_pdf,
                  file_name=f"{safe_filename(st.session_state.doc_id)}_compliance.pdf",
                  mime="application/pdf",
                  use_container_width=True
              )
              log_usage(action="export_pdf_compliance", user_email=AUTH_EMAIL, doc_id=st.session_state.doc_id, model="", meta={"bytes": len(comp_pdf)})
          else:
              st.caption("Export locked (editor/reviewer/admin only).")



























































"""selfcheck.py — startup self-checks (duplicates, risky prod flags).

Goal: catch misconfig early and make failures obvious.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple


def _find_duplicates(items: List[str]) -> List[str]:
    seen = set()
    dups = set()
    for it in items:
        if it in seen:
            dups.add(it)
        seen.add(it)
    return sorted(dups)


def startup_self_check(
    *,
    st,
    safe_secret: Callable[[str, Any], Any],
    prod: bool,
    is_cloud_run: bool,
) -> Dict[str, Any]:
    """Run once at startup. Returns a dict report."""
    report: Dict[str, Any] = {"warnings": [], "errors": []}

    # duplicates in allowlists
    allowed_domains = str(safe_secret("ALLOWED_DOMAINS", "") or "")
    allowed_emails = str(safe_secret("ALLOWED_EMAILS", "") or "")
    doms = [d.strip().lower() for d in allowed_domains.split(",") if d.strip()]
    ems = [e.strip().lower() for e in allowed_emails.split(",") if e.strip()]

    d_dom = _find_duplicates(doms)
    d_em = _find_duplicates(ems)
    if d_dom:
        report["warnings"].append(f"Duplicate allowed domains: {', '.join(d_dom)}")
    if d_em:
        report["warnings"].append(f"Duplicate allowed emails: {', '.join(d_em)}")

    # risky flags in prod
    if prod and (str(safe_secret("DEBUG", "") or "").lower() in ("1", "true", "yes")):
        report["warnings"].append("DEBUG is set but ignored in PROD.")

    if prod and (str(safe_secret("ENABLE_WORKSPACE_SWITCH", "") or "").lower() in ("1", "true", "yes")):
        report["warnings"].append("ENABLE_WORKSPACE_SWITCH is set but should be off in PROD.")

    if is_cloud_run and not prod:
        report["warnings"].append("Cloud Run detected but PROD is false — double-check you didn't deploy a dev build.")

    # UI surface
    if report["errors"]:
        st.error("Startup self-check failed:\n- " + "\n- ".join(report["errors"]))
    if report["warnings"]:
        st.warning("Startup self-check warnings:\n- " + "\n- ".join(report["warnings"]))

    return report

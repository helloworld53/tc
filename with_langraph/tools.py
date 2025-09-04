# tools.py
from datetime import datetime, timedelta
from typing import Dict, Any, List
from langchain.tools import tool

@tool
def calc_residency_days(entry_date: str, exit_date: str) -> Dict[str, Any]:
    """Calculate number of days stayed between two dates (YYYY-MM-DD)."""
    d1 = datetime.strptime(entry_date, "%Y-%m-%d")
    d2 = datetime.strptime(exit_date, "%Y-%m-%d")
    if d2 < d1:
        return {"error": "exit_date must be after entry_date"}
    days = (d2 - d1).days
    return {"days": days}

@tool
def schengen_90_180_check(stays: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Check Schengen 90/180 compliance.
    stays: list of {"entry":"YYYY-MM-DD","exit":"YYYY-MM-DD"} windows.
    """
    # Simplified rolling-window check: sum days in last 180 from the latest exit
    parsed = []
    for s in stays:
        a = datetime.strptime(s["entry"], "%Y-%m-%d")
        b = datetime.strptime(s["exit"], "%Y-%m-%d")
        if b < a:
            return {"error": "Invalid stay window"}
        parsed.append((a, b))
    if not parsed:
        return {"total_days_180": 0, "compliant": True}
    end = max(b for _, b in parsed)
    window_start = end - timedelta(days=180)
    total = 0
    for a, b in parsed:
        start_overlap = max(a, window_start)
        end_overlap = b
        if end_overlap >= start_overlap:
            total += (end_overlap - start_overlap).days
    return {"total_days_180": total, "compliant": total <= 90}

@tool
def fee_estimator(visa_type: str, fast_track: bool = False) -> Dict[str, Any]:
    """
    Rough fee estimate (demo values). visa_type in {"work","student","schengen"}.
    """
    base = {"work": 140, "student": 60, "schengen": 80}.get(visa_type.lower(), 100)
    total = base + (50 if fast_track else 0)
    return {"visa_type": visa_type, "fast_track": fast_track, "estimate_eur": total}

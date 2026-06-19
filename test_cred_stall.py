"""Regression: dead Copernicus creds must not out-rank fresh ones.

Reproduces the Jun 2026 oscillation bug where a credential whose account
expired (90d) kept its week-old successes, aged out its error_recency
penalty within 6h, climbed back to "warm" (~0.71), got re-preferred for
frontier work, failed, got evicted, recovered, and oscillated forever.
Run: python3 /tmp/test_stall.py
"""
import time, copernicus

now = time.time()
now_h = int(now // 3600)

def mk(s7, e7, rs, re_, last_ok_h, last_err_h):
    return {"last_status": "valid", "usage": {
        "success_7d": s7, "error_7d": e7, "rotated_7d": 0,
        "last_use": now - last_err_h*3600,
        "last_success": now - last_ok_h*3600 if last_ok_h < 1e8 else 0,
        "last_error": now - last_err_h*3600,
        "buckets": [
            {"h": now_h-1, "s": rs, "e": re_, "r": 0},
            {"h": now_h-50, "s": max(0, s7-rs), "e": max(0, e7-re_), "r": 0},
        ]}}

def sc(*a): return copernicus.score_credential_health(mk(*a), now=now)

# Dead cred stays clamped regardless of how stale the last error is.
for errage in (0.1, 6, 12, 24):
    h = sc(617, 114, 0, 54, 34, errage)
    assert h["label"] == "stalled", (errage, h)
    assert h["score"] <= 0.10, (errage, h)

# Fresh unused cred dominates.
assert sc(0, 0, 0, 0, 1e9, 1e9)["score"] == 1.0

# Transient IP-throttle (recent successes present) must NOT stall.
h = sc(300, 25, 30, 20, 3, 0.2)
assert h["signals"]["stalled"] is False, h

# A cred with a single recent error is below the evidence floor.
h = sc(100, 1, 0, 1, 5, 0.5)
assert h["signals"]["stalled"] is False, h

print("OK: stall gate behaves correctly")

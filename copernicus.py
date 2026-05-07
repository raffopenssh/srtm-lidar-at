"""Copernicus openEO interface for Sentinel-2 NDVI, ESA WorldCover, and Sentinel-1 SAR.

Provides cloud-free NDVI composites, monthly NDVI time series, land cover
classification, and SAR backscatter data via the Copernicus Data Space
openEO API.  Results are cached locally with LRU eviction (max 2 GB).

Usage::

    from copernicus import get_ndvi_composite, get_ndvi_timeseries
    result = get_ndvi_composite({"west": 16.3, "south": 48.2, "east": 16.4, "north": 48.3})
    ndvi = result["ndvi"]  # np.ndarray (H, W)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import tempfile
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import openeo
except ImportError:
    openeo = None  # type: ignore[assignment]

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
except ImportError:
    rasterio = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Silence noisy openeo client lib warnings about unparseable error bodies.
# Every 503/502 produces a 'Failed to parse API error response' WARNING
# line that doubles the signal feeding the director's copernicus EMA.
# The retry layer below already handles the response correctly.
try:
    logging.getLogger('openeo.rest._connection').setLevel(logging.ERROR)
    logging.getLogger('openeo.rest.connection').setLevel(logging.ERROR)
except Exception:
    pass

# === SECTION: Credentials & configuration (multi-account rotation) ===
# Credentials — multiple accounts for rotation when rate-limited (429) or overloaded.
# OLD (expired 2026-04): CLIENT_ID = "sh-19061cbb-c6f9-4464-bba6-006e7fa17435"
# OLD (expired 2026-04): CLIENT_SECRET = "<REDACTED_SECRET>"
# OLD (account 1, out of credits): CLIENT_ID = "sh-187c6dab-6b27-4ce8-afa8-b73f38e640f3"
# OLD (account 1, out of credits): CLIENT_SECRET = "<REDACTED_SECRET>"
# Built-in credentials (always loaded). Additional credentials can be
# added at runtime via /api/v1/credentials and are persisted to
# data/austria_processor/copernicus_credentials.json. The file is the
# single source of truth at runtime; this list is the seed/fallback.
_BUILTIN_CREDENTIALS = [
    ("sh-f36653c6-5d8c-48a1-b86d-476c50eb389c", "<REDACTED_SECRET>"),  # fresh 2026-04
    ("sh-8d8c685f-df36-4536-b949-666532d08414", "<REDACTED_SECRET>"),  # renews 2026-05-01
    ("sh-2ed25dbb-857d-4e99-b070-e1954a99a980", "<REDACTED_SECRET>"),  # renews 2026-05-01
    ("sh-07af1740-88e5-49d1-93c8-e9fca0fe2d49", "<REDACTED_SECRET>"),  # 30k credits
    ("sh-6db28e03-8090-4194-81b1-4d7db557b5aa", "<REDACTED_SECRET>"),  # added 2026-04 (slot 5)
    ("sh-9c10ed71-86af-4c72-b6f5-50c0e160128f", "<REDACTED_SECRET>"),  # added 2026-04 (slot 6)
]

# Credentials store path (instance-local; not in git)
_CRED_STORE = pathlib.Path("data/austria_processor/copernicus_credentials.json")
# Per-credential usage stats (success/error counts + per-minute buckets).
# Written by every process that uses Copernicus, read by the dashboard.
_USAGE_STORE = pathlib.Path("data/austria_processor/copernicus_credential_usage.json")
_USAGE_BUCKET_HOURS = 24 * 7  # keep last 7 days of per-hour buckets
# Per-credential metadata: label, added_at, last_validated_at, last_status,
# notes. Indexed by client_id.
_cred_meta: Dict[str, Dict[str, Any]] = {}
# Forward-declared so _load_credentials_from_disk can rehydrate it; the
# canonical declaration (with comment) appears below.
_exhausted_cred_indices: set = set()

def _load_credentials_from_disk() -> list:
    """Merge built-ins with persisted credentials. Built-ins always come first.

    Also rehydrates ``_exhausted_cred_indices`` from any ``exhausted=True``
    flags persisted in the meta, so a fresh subprocess starts with the
    same exhaustion state the parent already discovered.
    """
    global _cred_meta
    creds = list(_BUILTIN_CREDENTIALS)
    seen = {c[0] for c in creds}
    _cred_meta = {}
    for cid, _csec in _BUILTIN_CREDENTIALS:
        _cred_meta[cid] = {"source": "builtin"}
    if _CRED_STORE.exists():
        try:
            data = json.loads(_CRED_STORE.read_text())
            for entry in data.get("credentials", []):
                cid = entry.get("client_id")
                csec = entry.get("client_secret")
                if not cid or not csec or cid in seen:
                    # Update metadata even for builtins so we keep last_validated_at etc.
                    if cid in _cred_meta:
                        for k, v in entry.items():
                            if k not in ("client_secret",):
                                _cred_meta[cid][k] = v
                    continue
                creds.append((cid, csec))
                seen.add(cid)
                _cred_meta[cid] = {k: v for k, v in entry.items() if k != "client_secret"}
        except Exception as e:
            logger.warning("Failed to load credentials store %s: %s", _CRED_STORE, e)
    # Apply persisted exhaustion to in-memory set.
    for i, (cid, _) in enumerate(creds):
        meta = _cred_meta.get(cid) or {}
        if meta.get("exhausted"):
            _exhausted_cred_indices.add(i)
    return creds

# Meta keys that represent freshly-probed runtime state. When merging
# a stale on-disk snapshot into in-memory meta we MUST NOT clobber
# these — otherwise a re-save right after revalidate_all_credentials()
# (which calls _save_credentials_to_disk -> _reload_credentials_from_disk)
# overwrites the just-probed status with the stale value we're about
# to overwrite on disk anyway. That bug stuck cred #8 on "error" for
# days even though OIDC probes were succeeding.
_VOLATILE_META_KEYS = {
    "last_status", "last_validated_at", "last_error",
    "exhausted", "exhausted_at",
}

def _reload_credentials_from_disk():
    """Re-read on-disk store and merge any user creds we don't know about
    into the in-memory pool. Necessary because gunicorn runs multiple
    workers, each with its own ``_CREDENTIALS`` list — without this,
    worker B will save its stale view and overwrite worker A's additions.

    Volatile probe-result keys (see ``_VOLATILE_META_KEYS``) are only
    pulled in for credentials we don't already track in memory; for
    known credentials, in-memory values win so a freshly-probed status
    is never clobbered by a stale on-disk snapshot.
    """
    global _CREDENTIALS
    if not _CRED_STORE.exists():
        return
    try:
        data = json.loads(_CRED_STORE.read_text())
    except Exception as e:
        logger.warning("Failed to re-read credentials store: %s", e)
        return
    seen = {c[0] for c in _CREDENTIALS}
    for entry in data.get("credentials", []):
        cid = entry.get("client_id")
        csec = entry.get("client_secret")
        if not cid or not csec:
            continue
        # Pull latest non-volatile meta (label, source, notes, added_at,
        # usage, ...). For volatile keys (last_status etc.) trust the
        # in-memory copy if we already know this credential — it may
        # have just been refreshed by revalidate_all_credentials().
        if cid in _cred_meta:
            for k, v in entry.items():
                if k == "client_secret":
                    continue
                if k in _VOLATILE_META_KEYS:
                    continue
                _cred_meta[cid][k] = v
        else:
            _cred_meta[cid] = {k: v for k, v in entry.items() if k != "client_secret"}
        if cid not in seen:
            _CREDENTIALS.append((cid, csec))
            seen.add(cid)
            # Honor persisted exhaustion
            if entry.get("exhausted"):
                _exhausted_cred_indices.add(len(_CREDENTIALS) - 1)

def _save_credentials_to_disk():
    """Persist non-builtin credentials + meta for all creds to disk.

    Also persists per-credential ``exhausted`` + ``exhausted_at`` so a
    fresh subprocess (and the director) inherits the state.
    """
    # Refresh from disk first so we don't drop creds another worker added.
    _reload_credentials_from_disk()
    builtin_ids = {c[0] for c in _BUILTIN_CREDENTIALS}
    out = {"credentials": []}
    now_iso = __import__('datetime').datetime.utcnow().isoformat() + "Z"
    for i, (cid, csec) in enumerate(_CREDENTIALS):
        meta = dict(_cred_meta.get(cid, {}))
        meta["client_id"] = cid
        meta["client_secret"] = csec
        meta.setdefault("source", "builtin" if cid in builtin_ids else "user")
        was_exh = bool(meta.get("exhausted"))
        is_exh = i in _exhausted_cred_indices
        meta["exhausted"] = is_exh
        if is_exh and not was_exh:
            meta["exhausted_at"] = now_iso
        if not is_exh:
            meta.pop("exhausted_at", None)
        out["credentials"].append(meta)
    try:
        _CRED_STORE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CRED_STORE.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2))
        os.replace(tmp, _CRED_STORE)
    except Exception as e:
        logger.warning("Failed to save credentials store: %s", e)

_CREDENTIALS = _load_credentials_from_disk()
_credential_index = 0  # current credential pair
CLIENT_ID = _CREDENTIALS[0][0]
CLIENT_SECRET = _CREDENTIALS[0][1]
OPENEO_URL = "openeo.dataspace.copernicus.eu"

# Permanent cache survives /tmp cleanup and service restarts.
# LRU eviction keeps total size under CACHE_MAX_BYTES.
CACHE_DIR = pathlib.Path("/home/exedev/srtm-lidar/rf_training_data/copernicus_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB

# Maximum bbox extent in degrees.  The tile_cache grid snaps outward
# to 0.1° steps, so tiles can reach 0.2° span.  0.25° gives headroom.
MAX_BBOX_SPAN_DEG = 0.25

# Synchronous download size threshold (area in sq-degrees).
# Below this we use direct download(); above we use batch jobs.
SYNC_AREA_THRESHOLD = 0.012  # ~0.1° × 0.1° cells fit in sync path

# ESA WorldCover class legend
WORLDCOVER_CLASSES: Dict[int, str] = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse_vegetation",
    70: "snow_ice",
    80: "permanent_water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}

# === SECTION: Internal helpers (connection, cache, retry) ===

_connection: Optional[Any] = None

# Per-credential connection pool — keyed by credential index.
# Used by _get_connection_for_cred() for parallel workers that each
# need their own openEO session (1 sync download per client_id).
_connections: Dict[int, Any] = {}

# Global flag: set when Copernicus returns 402 PaymentRequired on ALL credentials.
# Callers (e.g. rf_train) can check this to pause gracefully.
credits_exhausted: bool = False
_credits_exhausted_at: Optional[str] = None  # ISO timestamp
# tracks which credential indices got 402 (rehydrated from disk on import)

# IP-level throttle: all credentials return 402 but probes pass.
# Unlike credits_exhausted (monthly cap), this recovers in ~2 hours.
ip_throttled: bool = False
_ip_throttled_at: float = 0  # monotonic timestamp
_IP_THROTTLE_COOLDOWN = 7200  # 2 hours — observed recovery time

# Short-term memory for NDVI months that recently failed with 500 (Spark timeout)
# or persistent 402.  Key = (bbox_hash, month_label), value = expiry timestamp.
# Skips re-downloading the same month for the same area within the cooldown period.
_FAILED_MONTH_COOLDOWNS: Dict[tuple, float] = {}
_FAILED_MONTH_COOLDOWN_SECS = 1800  # 30 minutes for 500 (Spark timeout)
_FAILED_MONTH_402_COOLDOWN_SECS = 1800  # 30 minutes — 402 is IP-level, recovery takes ~2h

# Threading lock for credential state — protects _credential_index,
# _exhausted_cred_indices, credits_exhausted, CLIENT_ID, CLIENT_SECRET,
# and _connection during concurrent access from parallel download workers.
_cred_lock = threading.Lock()


# Per-process credential whitelist. Set via COPERNICUS_CRED_INDICES env
# (e.g. "0,2,3") to restrict the workers in this process to a subset of
# the credential pool. The director uses this so different peers use
# disjoint credential pairs and don't collide on rate limits.
_ALLOWED_CRED_INDICES: Optional[set] = None


def _init_allowed_cred_indices():
    global _ALLOWED_CRED_INDICES, _credential_index, CLIENT_ID, CLIENT_SECRET
    raw = os.environ.get("COPERNICUS_CRED_INDICES", "").strip()
    if not raw:
        return
    try:
        idxs = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            i = int(part)
            if 0 <= i < len(_CREDENTIALS):
                idxs.append(i)
        if idxs:
            _ALLOWED_CRED_INDICES = set(idxs)
            with _cred_lock:
                _credential_index = idxs[0]
                CLIENT_ID, CLIENT_SECRET = _CREDENTIALS[_credential_index]
            logger.info("COPERNICUS_CRED_INDICES=%s -> using credentials %s (start=%d)",
                        raw, sorted(_ALLOWED_CRED_INDICES), idxs[0])
    except Exception as e:
        logger.warning("Bad COPERNICUS_CRED_INDICES=%r: %s", raw, e)

_init_allowed_cred_indices()


def FUNCTIONING_CREDENTIALS() -> list:
    """Return the list of credential indices that haven't been exhausted (402).

    If COPERNICUS_CRED_INDICES is set, the result is intersected with the
    whitelist so this process only ever uses its assigned credentials.
    """
    with _cred_lock:
        idxs = [i for i in range(len(_CREDENTIALS)) if i not in _exhausted_cred_indices]
    if _ALLOWED_CRED_INDICES is not None:
        idxs = [i for i in idxs if i in _ALLOWED_CRED_INDICES]
    return idxs


# === SECTION: Credential usage telemetry ===
# Lightweight per-credential usage tracking: success/error counts +
# per-minute buckets for a sparkline. Written atomically; multi-process
# safe via best-effort merge (concurrent writers may briefly overwrite
# each other but the buckets are coarse enough that this is fine).

_usage_lock = threading.Lock()

def _product_from_title(title: str) -> str:
    t = (title or "").lower()
    if "ndvi" in t and ("composite" in t or "compose" in t):
        return "ndvi"
    if "ndvi" in t:
        return "ndvi"
    if "worldcover" in t or "land cover" in t or "esa worldcover" in t:
        return "worldcover"
    if "sar" in t or "backscatter" in t:
        return "sar"
    if "harmonic" in t:
        return "harmonics"
    return ""

def _load_usage() -> dict:
    try:
        if _USAGE_STORE.exists():
            with _USAGE_STORE.open() as f:
                return json.load(f) or {}
    except Exception:
        pass
    return {}

def _save_usage(data: dict) -> None:
    try:
        _USAGE_STORE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _USAGE_STORE.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(data, f)
        tmp.replace(_USAGE_STORE)
    except Exception as e:
        logger.debug("usage save failed: %s", e)

def _prune_buckets(buckets: dict, now_h: int) -> dict:
    cutoff = now_h - _USAGE_BUCKET_HOURS
    return {k: v for k, v in buckets.items() if int(k) >= cutoff}

def record_credential_usage(cred_index: int, kind: str, product: str = "") -> None:
    """Record one credential request outcome.

    Parameters
    ----------
    cred_index : int
        Credential index in ``_CREDENTIALS``.
    kind : str
        ``success``, ``error``, or ``rotated`` (transient 402).
    product : str
        Optional product label (ndvi, sar, worldcover, harmonics).
    """
    if cred_index < 0 or cred_index >= len(_CREDENTIALS):
        return
    cid = _CREDENTIALS[cred_index][0]
    now = int(time.time())
    now_h = now // 3600
    with _usage_lock:
        data = _load_usage()
        entry = data.setdefault(cid, {
            "success": 0, "error": 0, "rotated": 0,
            "last_use": 0, "last_success": 0, "last_error": 0,
            "buckets": {},
            "by_product": {},
        })
        if kind in ("success", "error", "rotated"):
            entry[kind] = int(entry.get(kind, 0)) + 1
        entry["last_use"] = now
        if kind == "success":
            entry["last_success"] = now
        elif kind == "error":
            entry["last_error"] = now
        if product:
            bp = entry.setdefault("by_product", {}).setdefault(product, {"success": 0, "error": 0, "rotated": 0})
            if kind in bp:
                bp[kind] += 1
        # Per-hour bucket: "<hour_epoch>": {s, e, r}
        buckets = entry.setdefault("buckets", {})
        bk = str(now_h)
        bb = buckets.setdefault(bk, {"s": 0, "e": 0, "r": 0})
        if kind == "success":
            bb["s"] += 1
        elif kind == "error":
            bb["e"] += 1
        elif kind == "rotated":
            bb["r"] += 1
        entry["buckets"] = _prune_buckets(buckets, now_h)
        _save_usage(data)

def _read_usage_for(cid: str) -> dict:
    with _usage_lock:
        data = _load_usage()
    e = data.get(cid) or {}
    if not e:
        return {"success": 0, "error": 0, "rotated": 0, "buckets": [], "by_product": {}}
    # Return buckets as a sorted list for the last 7 days (per-hour)
    now_h = int(time.time()) // 3600
    cutoff = now_h - _USAGE_BUCKET_HOURS
    out_buckets = []
    s7 = e7 = r7 = 0
    for h in range(cutoff + 1, now_h + 1):
        bb = e.get("buckets", {}).get(str(h), {"s": 0, "e": 0, "r": 0})
        s = int(bb.get("s", 0)); er = int(bb.get("e", 0)); ro = int(bb.get("r", 0))
        out_buckets.append({"h": h, "s": s, "e": er, "r": ro})
        s7 += s; e7 += er; r7 += ro
    return {
        "success": int(e.get("success", 0)),
        "error": int(e.get("error", 0)),
        "rotated": int(e.get("rotated", 0)),
        "last_use": int(e.get("last_use", 0)),
        "last_success": int(e.get("last_success", 0)),
        "last_error": int(e.get("last_error", 0)),
        "by_product": e.get("by_product", {}),
        "buckets": out_buckets,
        "window_hours": _USAGE_BUCKET_HOURS,
        "success_7d": s7,
        "error_7d": e7,
        "rotated_7d": r7,
    }


def score_credential_health(meta: dict, *, now: float | None = None) -> dict:
    """Score a credential's recent health & freshness for assignment.

    Higher score = better candidate for *new* frontier work. Combines
    five signals into a value in [0, 1] plus a structured breakdown so
    the dashboard can explain *why* a particular score landed:

    * **status** — penalises ``invalid`` / ``error`` from the last OIDC
      probe; ``exhausted`` zeroes the score.
    * **error_recency** — recent ``last_error`` timestamps lower the
      score on a ramp from 1 h (heavy) to 24 h (mild).
    * **error_rate** — ``error_7d / (success_7d + error_7d)``; raw
      errors over the last week.
    * **rotation** — ``rotated_7d`` (in-peer credential swaps after
      402/throttle) shaves off a small penalty since the cred just
      caused a rotation event.
    * **exhausted_at** — recent exhaustion (within 7 days) leaves
      a residual penalty even after recovery.
    * **freshness** — credentials *not* used recently get a positive
      bonus, so the director rotates the warm/hot set across peers
      instead of always reloading the same indices.

    Result fields:
      * ``score`` — final value in [0, 1] used for ordering.
      * ``components`` — per-signal contribution (negative = penalty,
        positive = bonus). Frontend renders this in the tooltip.
      * ``label`` — "healthy" / "warm" / "hot" / "degraded" / "exhausted".
    """
    if now is None:
        now = time.time()
    u = meta.get("usage") or {}
    s7 = int(u.get("success_7d") or 0)
    e7 = int(u.get("error_7d") or 0)
    r7 = int(u.get("rotated_7d") or 0)
    last_use = float(u.get("last_use") or 0)
    last_err = float(u.get("last_error") or 0)

    components: dict[str, float] = {}
    score = 1.0

    # Hard zero for exhausted (cred shouldn't be picked at all).
    if meta.get("exhausted"):
        return {
            "score": 0.0,
            "label": "exhausted",
            "components": {"status": -1.0},
            "signals": {
                "success_7d": s7, "error_7d": e7, "rotated_7d": r7,
                "last_use_age_s": (now - last_use) if last_use else None,
                "last_error_age_s": (now - last_err) if last_err else None,
            },
        }

    # 1) last_status
    st = (meta.get("last_status") or "").lower()
    if st in ("invalid",):
        score -= 0.6
        components["status"] = -0.6
    elif st == "error":
        score -= 0.25
        components["status"] = -0.25
    elif st == "valid":
        components["status"] = 0.0

    # 2) last_error recency (only counts if there was one)
    if last_err and last_err > 0:
        age_h = max(0.0, (now - last_err) / 3600.0)
        if age_h < 1.0:
            pen = -0.50
        elif age_h < 6.0:
            pen = -0.30
        elif age_h < 24.0:
            pen = -0.15
        elif age_h < 72.0:
            pen = -0.05
        else:
            pen = 0.0
        if pen:
            score += pen
            components["error_recency"] = pen

    # 3) error rate over the 7d window
    denom = s7 + e7
    if denom > 0 and e7 > 0:
        er = e7 / denom
        pen = -0.4 * er
        score += pen
        components["error_rate"] = pen

    # 4) rotation churn — small per-event penalty, capped
    if r7 > 0:
        pen = -min(0.2, r7 / 200.0)
        score += pen
        components["rotation"] = pen

    # 5) recent exhaustion residual (recovers but lingers)
    exh_at = float(meta.get("exhausted_at") or 0)
    if exh_at > 0:
        age_h = (now - exh_at) / 3600.0
        if age_h < 24.0:
            pen = -0.4
        elif age_h < 24.0 * 7:
            pen = -0.15
        else:
            pen = 0.0
        if pen:
            score += pen
            components["exhausted_recently"] = pen

    # 6) freshness bonus — under-used creds rotate in.
    if last_use <= 0:
        bonus = 0.25
    else:
        age_h = (now - last_use) / 3600.0
        if age_h >= 24.0 * 7:
            bonus = 0.25
        elif age_h >= 24.0:
            bonus = 0.15
        elif age_h >= 6.0:
            bonus = 0.05
        else:
            bonus = 0.0
    if bonus:
        score += bonus
        components["freshness"] = bonus

    # 7) Hot-cred mild de-prioritisation: peers that have absorbed
    # a lot of work over the last 7d get bumped down a touch so we
    # rotate the warm/hot set across peers rather than always landing
    # on the same handful.
    if s7 > 0:
        pen = -min(0.2, s7 / 5000.0)
        if pen <= -0.02:
            score += pen
            components["workload"] = pen

    score = max(0.0, min(1.0, score))

    if score >= 0.85:
        label = "healthy"
    elif score >= 0.65:
        label = "warm"
    elif score >= 0.40:
        label = "hot"
    elif score > 0:
        label = "degraded"
    else:
        label = "unusable"

    return {
        "score": round(score, 4),
        "label": label,
        "components": {k: round(v, 4) for k, v in components.items()},
        "signals": {
            "success_7d": s7,
            "error_7d": e7,
            "rotated_7d": r7,
            "last_use_age_s": int(now - last_use) if last_use else None,
            "last_error_age_s": int(now - last_err) if last_err else None,
            "last_status": meta.get("last_status"),
            "exhausted_at_age_s": int(now - exh_at) if exh_at else None,
        },
    }


def list_credentials() -> list:
    """Return public credential metadata (client_id, source, last_validated_at,
    last_status, exhausted, label). Never returns secrets.

    Each entry also includes a ``health`` block with the score and
    components used by ``peer_director._assign_cred_indices`` — see
    :func:`score_credential_health`."""
    with _cred_lock:
        # Pick up creds added by sibling gunicorn workers.
        _reload_credentials_from_disk()
        exhausted = set(_exhausted_cred_indices)
        creds = list(_CREDENTIALS)
    out = []
    for i, (cid, _csec) in enumerate(creds):
        meta = dict(_cred_meta.get(cid, {}))
        meta.pop("client_secret", None)
        meta["index"] = i
        meta["client_id"] = cid
        meta["client_id_short"] = cid[:16] + "..."
        meta["exhausted"] = i in exhausted
        if _ALLOWED_CRED_INDICES is not None:
            meta["allowed_in_process"] = i in _ALLOWED_CRED_INDICES
        try:
            meta["usage"] = _read_usage_for(cid)
        except Exception:
            meta["usage"] = {"success": 0, "error": 0, "rotated": 0, "buckets": []}
        try:
            meta["health"] = score_credential_health(meta)
        except Exception:
            meta["health"] = {"score": 0.5, "label": "unknown",
                              "components": {}, "signals": {}}
        out.append(meta)
    return out


def validate_credential(client_id: str, client_secret: str) -> dict:
    """Probe a (client_id, client_secret) pair. Returns dict with ok, status,
    error. Performs an OIDC client-credentials auth — same shape as
    _probe_credential() but for arbitrary creds (used before adding)."""
    if openeo is None:
        return {"ok": False, "status": "openeo_missing",
                "error": "openeo package not installed"}
    try:
        conn = openeo.connect(OPENEO_URL)
        conn.authenticate_oidc_client_credentials(
            client_id=client_id, client_secret=client_secret)
        return {"ok": True, "status": "valid"}
    except Exception as e:
        msg = str(e)
        if '402' in msg and 'PaymentRequired' in msg:
            return {"ok": False, "status": "exhausted", "error": msg[:300]}
        if '401' in msg or 'invalid_client' in msg:
            return {"ok": False, "status": "invalid", "error": msg[:300]}
        return {"ok": False, "status": "error", "error": msg[:300]}


def add_credential(client_id: str, client_secret: str, label: str = "",
                   notes: str = "", validate: bool = True) -> dict:
    """Add a new credential to the runtime pool and persist to disk.

    If *validate* is True, probe before adding. Returns the new index +
    validation result. Idempotent: re-adding an existing client_id
    updates label/notes only.
    """
    global _CREDENTIALS
    client_id = (client_id or "").strip()
    client_secret = (client_secret or "").strip()
    if not client_id or not client_secret:
        return {"ok": False, "error": "client_id and client_secret required"}

    val = {"ok": True, "status": "unchecked"}
    if validate:
        val = validate_credential(client_id, client_secret)

    now = __import__('datetime').datetime.utcnow().isoformat() + "Z"
    with _cred_lock:
        # If the credential is already known, just update meta
        existing_idx = None
        for i, (cid, _) in enumerate(_CREDENTIALS):
            if cid == client_id:
                existing_idx = i
                break
        if existing_idx is None:
            _CREDENTIALS.append((client_id, client_secret))
            existing_idx = len(_CREDENTIALS) - 1
        meta = _cred_meta.setdefault(client_id, {})
        if label:
            meta["label"] = label
        if notes:
            meta["notes"] = notes
        meta.setdefault("source", "user")
        meta.setdefault("added_at", now)
        meta["last_validated_at"] = now
        meta["last_status"] = val.get("status", "unchecked")
        if not val.get("ok"):
            meta["last_error"] = val.get("error", "")
        else:
            meta.pop("last_error", None)
        _save_credentials_to_disk()
    return {"ok": True, "index": existing_idx, "client_id": client_id,
            "validation": val}


def remove_credential(client_id: str) -> dict:
    """Remove a credential by client_id. Built-ins cannot be removed via API
    (would reappear on restart) — but we still drop the in-memory entry so
    the running process stops using them."""
    global _CREDENTIALS, _credential_index, CLIENT_ID, CLIENT_SECRET
    with _cred_lock:
        idx = None
        for i, (cid, _) in enumerate(_CREDENTIALS):
            if cid == client_id:
                idx = i
                break
        if idx is None:
            return {"ok": False, "error": "not found"}
        _CREDENTIALS.pop(idx)
        _exhausted_cred_indices.discard(idx)
        # Reindex exhausted set to account for shift
        _exhausted_cred_indices_old = set(_exhausted_cred_indices)
        _exhausted_cred_indices.clear()
        for j in _exhausted_cred_indices_old:
            if j > idx:
                _exhausted_cred_indices.add(j - 1)
            elif j < idx:
                _exhausted_cred_indices.add(j)
        if _credential_index >= len(_CREDENTIALS):
            _credential_index = 0
        if _CREDENTIALS:
            CLIENT_ID, CLIENT_SECRET = _CREDENTIALS[_credential_index]
        _cred_meta.pop(client_id, None)
        _save_credentials_to_disk()
    return {"ok": True, "removed": client_id}


def revalidate_all_credentials() -> list:
    """Probe every credential, update meta. Returns a list parallel to
    list_credentials() but with fresh probe results."""
    out = []
    now = __import__('datetime').datetime.utcnow().isoformat() + "Z"
    with _cred_lock:
        creds = list(_CREDENTIALS)
    for i, (cid, csec) in enumerate(creds):
        val = validate_credential(cid, csec)
        with _cred_lock:
            meta = _cred_meta.setdefault(cid, {})
            meta["last_validated_at"] = now
            meta["last_status"] = val.get("status", "unchecked")
            if val.get("ok"):
                meta.pop("last_error", None)
                # If we previously thought it was exhausted, reset.
                _exhausted_cred_indices.discard(i)
            else:
                meta["last_error"] = val.get("error", "")
                if val.get("status") == "exhausted":
                    _exhausted_cred_indices.add(i)
        out.append({"index": i, "client_id": cid, **val})
    with _cred_lock:
        _save_credentials_to_disk()
    return out


class CreditsExhaustedError(Exception):
    """Raised when ALL Copernicus credentials return 402 PaymentRequired."""
    pass


class IPThrottledError(RuntimeError):
    """Raised when Copernicus API is IP-level rate-limited (all credentials 402)."""
    pass


class CredentialRotatedError(Exception):
    """Raised when a 402 was handled by rotating to a fresh credential.
    Callers should rebuild their connection/datacube and retry."""
    pass


def _probe_credential(cred_index: int) -> bool:
    """Lightweight probe to check if a credential is actually exhausted.

    Attempts to authenticate with the given credential.  If auth succeeds
    the credential likely still has credits and the 402 was a transient
    rate-limit.  Returns True if the credential appears healthy.
    """
    try:
        cid, csecret = _CREDENTIALS[cred_index]
        logger.info("Probing credential %d (client_id=%s) ...", cred_index + 1, cid[:16] + "...")
        conn = openeo.connect(OPENEO_URL)
        conn.authenticate_oidc_client_credentials(client_id=cid, client_secret=csecret)
        # If auth succeeds, credential is likely still valid
        logger.info("Probe OK — credential %d is still valid", cred_index + 1)
        # Refresh the cached connection
        _connections[cred_index] = conn
        return True
    except Exception as probe_exc:
        probe_msg = str(probe_exc)
        if '402' in probe_msg and 'PaymentRequired' in probe_msg:
            logger.warning("Probe confirmed credential %d is exhausted", cred_index + 1)
            return False
        # Other errors (network, 5xx) — give benefit of the doubt
        logger.info("Probe inconclusive for credential %d (%s) — treating as healthy",
                    cred_index + 1, probe_exc)
        return True


def _check_credits_error(exc: Exception, cred_index: int | None = None) -> None:
    """If *exc* is a 402 PaymentRequired, handle credential rotation.

    Parameters
    ----------
    exc : Exception
        The exception to inspect.
    cred_index : int or None
        The credential index that caused the 402.  If None, uses the
        global ``_credential_index``.

    Probes the credential to distinguish transient rate-limits from
    genuine credit exhaustion:

    - **Probe passes** (transient 402): rotate to the next credential
      and raise ``CredentialRotatedError`` so the ``@_retry_on_rotation``
      decorator rebuilds the connection + datacube with a fresh credential.
      The credential is NOT marked exhausted.
    - **Probe fails** (genuine exhaustion): mark the credential exhausted,
      rotate, and raise ``CredentialRotatedError``.
    - **All credentials exhausted**: raise ``CreditsExhaustedError``.
    - **Not a 402**: return silently.
    """
    global credits_exhausted, _credits_exhausted_at, _credential_index
    msg = str(exc)
    if '402' not in msg or 'PaymentRequired' not in msg:
        return

    # Determine which credential hit the 402
    with _cred_lock:
        idx = cred_index if cred_index is not None else _credential_index

    # Probe OUTSIDE the lock (network call — don't block other threads)
    if openeo is not None and _probe_credential(idx):
        # Credential is still healthy — transient rate-limit.
        # Rotate to next credential so the decorator retries with it.
        with _cred_lock:
            _connections.pop(idx, None)
            rotate_credentials(_locked=True)
        logger.info("Credential %d probe passed — transient 402, rotated to %d/%d",
                    idx + 1, _credential_index + 1, len(_CREDENTIALS))
        raise CredentialRotatedError(
            f"Transient 402 on credential {idx + 1}, "
            f"rotated to {_credential_index + 1}/{len(_CREDENTIALS)}"
        ) from exc

    # Probe failed — mark as exhausted under the lock
    with _cred_lock:
        _exhausted_cred_indices.add(idx)
        logger.warning("Credential %d/%d confirmed exhausted (402 PaymentRequired, client_id=%s)",
                       idx + 1, len(_CREDENTIALS),
                       _CREDENTIALS[idx][0][:16] + "...")
        # Persist so the director and other workers see it.
        meta = _cred_meta.setdefault(_CREDENTIALS[idx][0], {})
        meta["last_status"] = "exhausted"
        meta["last_error"] = msg[:300]
        try:
            _save_credentials_to_disk()
        except Exception as _se:
            logger.debug("persist exhaustion failed: %s", _se)

        if len(_exhausted_cred_indices) >= len(_CREDENTIALS):
            # ALL credentials exhausted
            credits_exhausted = True
            _credits_exhausted_at = __import__('datetime').datetime.utcnow().isoformat()
            logger.error("ALL %d Copernicus credentials exhausted", len(_CREDENTIALS))
            raise CreditsExhaustedError(msg) from exc
        else:
            # Rotate to a non-exhausted credential
            for _ in range(len(_CREDENTIALS)):
                rotate_credentials(_locked=True)
                if _credential_index not in _exhausted_cred_indices:
                    logger.info("Rotated to fresh credential %d/%d (client_id=%s)",
                                _credential_index + 1, len(_CREDENTIALS),
                                CLIENT_ID[:16] + "...")
                    raise CredentialRotatedError(
                        f"Rotated to credential {_credential_index + 1}/{len(_CREDENTIALS)}"
                    ) from exc
            # Shouldn't reach here but just in case
            credits_exhausted = True
            _credits_exhausted_at = __import__('datetime').datetime.utcnow().isoformat()
            raise CreditsExhaustedError(msg) from exc


def reset_exhausted_credentials():
    """Clear the exhausted-credential tracking.  Call after providing new
    credentials or when credits have been replenished."""
    global credits_exhausted, _credits_exhausted_at
    with _cred_lock:
        _exhausted_cred_indices.clear()
        credits_exhausted = False
        _credits_exhausted_at = None
        # Clear persisted exhaustion flags too.
        for cid, meta in _cred_meta.items():
            if meta.get("exhausted"):
                meta["exhausted"] = False
                meta.pop("exhausted_at", None)
                if meta.get("last_status") == "exhausted":
                    meta["last_status"] = "unchecked"
        try:
            _save_credentials_to_disk()
        except Exception:
            pass
    logger.info("Reset exhausted-credential tracking for %d credentials",
                len(_CREDENTIALS))


def _retry_on_rotation(fn):
    """Decorator: retry the whole function when credentials are rotated on 402.

    Catches ``CredentialRotatedError`` (meaning a fresh credential is now
    active) and reruns the function from scratch so it builds a new
    connection + datacube with the fresh credential.  Gives up after
    ``len(_CREDENTIALS)`` rotations.

    After exhausting all credentials, checks whether any were genuinely
    exhausted (probe failed).  If none were — all 402s were transient,
    meaning IP-level rate-limiting — raises ``IPThrottledError`` and
    sets the module-level ``ip_throttled`` flag.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _connection, ip_throttled, _ip_throttled_at
        for attempt in range(len(_CREDENTIALS) + 1):
            try:
                return fn(*args, **kwargs)
            except CredentialRotatedError:
                _connection = None  # force new connection on retry
                logger.info("Credential rotated — retrying %s (attempt %d/%d)",
                            fn.__name__, attempt + 2, len(_CREDENTIALS) + 1)
                continue
            except (CreditsExhaustedError, IPThrottledError):
                raise  # propagate immediately
        # All credential rotation attempts exhausted.
        # Check: were any credentials genuinely exhausted (probe-failed)?
        with _cred_lock:
            n_exhausted = len(_exhausted_cred_indices)
        if n_exhausted == 0:
            # All probes passed — every credential is healthy but 402'd.
            # This is IP-level (or account-level) rate-limiting.
            import time as _ip_time
            ip_throttled = True
            _ip_throttled_at = _ip_time.monotonic()
            logger.warning(
                "All %d credentials returned transient 402 for %s — "
                "IP-throttled, pausing for %d min",
                len(_CREDENTIALS), fn.__name__,
                _IP_THROTTLE_COOLDOWN // 60)
            raise IPThrottledError(
                f"All {len(_CREDENTIALS)} credentials returned transient 402 "
                f"for {fn.__name__} — IP-level rate limit"
            )
        raise CreditsExhaustedError("All credential rotation attempts failed")
    return wrapper


def rotate_credentials(_locked: bool = False) -> bool:
    """Switch to the next credential pair. Returns True if rotated, False if exhausted all.

    Parameters
    ----------
    _locked : bool
        If True, the caller already holds ``_cred_lock`` — skip acquiring it.
    """
    global _credential_index, _connection, CLIENT_ID, CLIENT_SECRET

    def _do_rotate():
        global _credential_index, _connection, CLIENT_ID, CLIENT_SECRET
        old_idx = _credential_index
        _credential_index = (_credential_index + 1) % len(_CREDENTIALS)
        if _credential_index == old_idx and len(_CREDENTIALS) == 1:
            return False  # only one set of credentials
        CLIENT_ID, CLIENT_SECRET = _CREDENTIALS[_credential_index]
        _connection = None  # force re-auth with new credentials
        logger.info("Rotated to credential set %d/%d (client_id=%s)",
                    _credential_index + 1, len(_CREDENTIALS), CLIENT_ID[:16] + "...")
        return _credential_index != old_idx  # True unless we wrapped all the way around

    if _locked:
        return _do_rotate()
    else:
        with _cred_lock:
            return _do_rotate()


def _get_connection() -> Any:
    """Return a cached, authenticated openEO connection."""
    global _connection
    if _connection is not None:
        return _connection

    if openeo is None:
        raise ImportError("The 'openeo' package is required. Install with: pip install openeo")

    logger.info("Connecting to openEO backend at %s", OPENEO_URL)
    conn = openeo.connect(OPENEO_URL)
    conn.authenticate_oidc_client_credentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    logger.info("Authenticated successfully (client_id=%s)", CLIENT_ID[:16] + "...")
    _connection = conn
    return conn


def _get_connection_for_cred(cred_index: int) -> Any:
    """Return a cached connection for a specific credential index.

    Each credential gets its own openEO session, allowing parallel sync
    downloads (openEO limits 1 concurrent sync job per client_id).
    """
    if cred_index in _connections:
        return _connections[cred_index]

    if openeo is None:
        raise ImportError("The 'openeo' package is required.")

    cid, csecret = _CREDENTIALS[cred_index]
    logger.info("Connecting to openEO for cred %d/%d (client_id=%s)",
                cred_index + 1, len(_CREDENTIALS), cid[:16] + "...")
    conn = openeo.connect(OPENEO_URL)
    conn.authenticate_oidc_client_credentials(
        client_id=cid, client_secret=csecret,
    )
    logger.info("Authenticated cred %d (client_id=%s)", cred_index + 1, cid[:16] + "...")
    _connections[cred_index] = conn
    return conn


def _bbox_hash(bbox: Dict[str, float], **extra: Any) -> str:
    """Deterministic hash for a bbox + extra parameters (for cache keys)."""
    payload = json.dumps({"bbox": bbox, **extra}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_path(prefix: str, bbox: Dict[str, float], **extra: Any) -> pathlib.Path:
    """Return the cache file path for a given request."""
    h = _bbox_hash(bbox, **extra)
    return CACHE_DIR / f"{prefix}_{h}.tif"


def _touch_cache(path: pathlib.Path):
    """Update atime/mtime on a cache file (for LRU tracking)."""
    try:
        path.touch()
    except Exception:
        pass


def _validate_cache(path: pathlib.Path) -> bool:
    """Return True if *path* looks like a valid (non-empty) cache file.

    Deletes the file and returns False when it is 0-byte or unreadable,
    which can happen when a previous download was interrupted mid-write.
    """
    try:
        if path.stat().st_size == 0:
            logger.warning("Removing corrupt (0-byte) cache file: %s", path)
            path.unlink(missing_ok=True)
            return False
    except OSError:
        return False
    return True


def _enforce_cache_limit():
    """Evict oldest cache files if total size exceeds CACHE_MAX_BYTES."""
    try:
        files = []
        for f in CACHE_DIR.iterdir():
            if f.is_file():
                st = f.stat()
                files.append((st.st_mtime, st.st_size, f))
        total = sum(s for _, s, _ in files)
        if total <= CACHE_MAX_BYTES:
            return
        # Sort oldest first, evict until under limit
        files.sort()
        for mtime, size, f in files:
            if total <= CACHE_MAX_BYTES:
                break
            try:
                f.unlink()
                total -= size
                logger.debug("Cache evict: %s (%d KB)", f.name, size // 1024)
            except Exception:
                pass
    except Exception:
        pass


def _validate_bbox(bbox: Dict[str, float], *, warn_large: bool = True) -> Dict[str, float]:
    """Validate and normalise a WGS-84 bounding box dict.

    Parameters
    ----------
    warn_large : bool
        Log a warning when the bbox exceeds ``MAX_BBOX_SPAN_DEG``.
        Set to False for pre-tiled requests (e.g. from tile_cache)
        where the caller already controls the extent.
    """
    required = {"west", "south", "east", "north"}
    if not required.issubset(bbox.keys()):
        raise ValueError(f"bbox must contain keys {required}, got {set(bbox.keys())}")

    w, s, e, n = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    if w >= e or s >= n:
        raise ValueError(f"Invalid bbox extents: west={w} >= east={e} or south={s} >= north={n}")

    span_lon = e - w
    span_lat = n - s
    if warn_large and (span_lon > MAX_BBOX_SPAN_DEG or span_lat > MAX_BBOX_SPAN_DEG):
        logger.warning(
            "Bbox span (%.4f° × %.4f°) exceeds recommended max %.4f°. "
            "Large requests may be slow or fail.",
            span_lon, span_lat, MAX_BBOX_SPAN_DEG,
        )

    return {"west": w, "south": s, "east": e, "north": n}


def _bbox_area_deg(bbox: Dict[str, float]) -> float:
    return (bbox["east"] - bbox["west"]) * (bbox["north"] - bbox["south"])


def _read_geotiff(path: Union[str, pathlib.Path]) -> Tuple[np.ndarray, Any, Any]:
    """Read a GeoTIFF and return (data, transform, crs).

    Returns data with shape (bands, H, W) or (H, W) if single-band.
    """
    if rasterio is None:
        raise ImportError("The 'rasterio' package is required. Install with: pip install rasterio")

    path = pathlib.Path(path)
    # If path is a directory (batch job output), find the first .tif inside
    if path.is_dir():
        tifs = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
        if not tifs:
            raise FileNotFoundError(f"No GeoTIFF files found in {path}")
        path = tifs[0]
        logger.debug("Using GeoTIFF from directory: %s", path)

    import logging as _logging
    # Suppress harmless GDAL warnings about Photometric/ExtraSamples mismatch
    # in Copernicus-produced multi-band TIFFs (VV+VH SAR, etc.)
    _rasterio_logger = _logging.getLogger('rasterio._env')
    _prev_level = _rasterio_logger.level
    _rasterio_logger.setLevel(_logging.ERROR)
    try:
        with rasterio.open(str(path)) as ds:
            data = ds.read()  # (bands, H, W)
            transform = ds.transform
            crs = ds.crs
    finally:
        _rasterio_logger.setLevel(_prev_level)

    if data.shape[0] == 1:
        data = data[0]  # squeeze to (H, W)

    return data, transform, crs


def _run_datacube(
    datacube: Any,
    output_path: pathlib.Path,
    title: str = "copernicus_job",
    format: str = "GTiff",
    bbox: Optional[Dict[str, float]] = None,
) -> pathlib.Path:
    """Download a datacube result to *output_path*.

    Uses synchronous ``download()`` for small cubes and batch-job
    ``execute_batch()`` for larger ones.  When *bbox* is provided,
    sync is skipped if the area exceeds ``SYNC_AREA_THRESHOLD``
    (avoids a 3-minute timeout that always fails for grid tiles).

    Downloads go to a temp file first and are atomically renamed on
    success.  This prevents 0-byte / partial files from poisoning
    the cache when a download times out or fails mid-write.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: download to a temp sibling, rename on success
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    def _cleanup_tmp():
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    # Bail immediately if we're IP-throttled (all creds 402, recovers in ~2h)
    global ip_throttled, _ip_throttled_at
    import time as _time_check
    if ip_throttled:
        elapsed = _time_check.monotonic() - _ip_throttled_at
        if elapsed < _IP_THROTTLE_COOLDOWN:
            remaining_min = int((_IP_THROTTLE_COOLDOWN - elapsed) / 60)
            raise IPThrottledError(
                f"Copernicus IP-throttled ({remaining_min}m remaining) — skipping")
        else:
            ip_throttled = False
            logger.info("IP throttle cooldown expired — resuming Copernicus requests")

    # Skip sync for large areas — always times out, wastes 3 minutes
    skip_sync = False
    if bbox is not None:
        area = _bbox_area_deg(bbox)
        if area > SYNC_AREA_THRESHOLD:
            logger.info("Bbox area %.4f° > threshold %.4f° — skipping sync, going straight to batch for %s",
                        area, SYNC_AREA_THRESHOLD, title)
            skip_sync = True

    # Try synchronous download first (faster for small areas).
    # On 402, _check_credits_error raises CredentialRotatedError which
    # propagates to @_retry_on_rotation — that rebuilds the connection +
    # datacube with the next credential and retries.  After cycling all
    # credentials it raises IPThrottledError (transient) or
    # CreditsExhaustedError (genuine).  No manual rotation needed here.
    import time as _time
    import concurrent.futures
    SYNC_DOWNLOAD_TIMEOUT = 180  # 3 minutes max for synchronous download

    if not skip_sync:
        logger.info("Downloading datacube synchronously → %s (timeout=%ds)",
                    output_path, SYNC_DOWNLOAD_TIMEOUT)
        try:
            _cleanup_tmp()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(datacube.download, str(tmp_path), format)
                future.result(timeout=SYNC_DOWNLOAD_TIMEOUT)
            # Verify non-empty before committing
            if tmp_path.exists() and tmp_path.stat().st_size > 0:
                tmp_path.rename(output_path)
                logger.info("Synchronous download complete: %s", output_path)
                try:
                    record_credential_usage(_credential_index, "success", _product_from_title(title))
                except Exception:
                    pass
                return output_path
            else:
                _cleanup_tmp()
                logger.warning("Synchronous download produced empty file, falling back to batch job")
        except concurrent.futures.TimeoutError:
            _cleanup_tmp()
            logger.warning("Synchronous download timed out after %ds, falling back to batch job",
                          SYNC_DOWNLOAD_TIMEOUT)
        except Exception as exc:
            _cleanup_tmp()
            try:
                record_credential_usage(_credential_index, "error", _product_from_title(title))
            except Exception:
                pass
            # 402 → CredentialRotatedError (propagates to decorator) or
            #        CreditsExhaustedError/IPThrottledError (propagates out)
            _check_credits_error(exc)
            # Not a 402 — fall back to batch
            logger.warning("Synchronous download failed (%s), falling back to batch job", exc)

    # Batch job fallback
    logger.info("Submitting batch job: %s", title)
    output_dir = output_path.parent / f"{output_path.stem}_batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        job = datacube.execute_batch(
            outputfile=str(output_dir),
            out_format=format,
            title=title,
            max_poll_interval=30,
            print=lambda msg: logger.info("[batch] %s", msg),
        )
    except Exception as batch_exc:
        _check_credits_error(batch_exc)  # raises CredentialRotatedError on 402
        # Batch job failed (status: error) — this is often transient or
        # credential-specific.  Rotate to the next credential and let
        # @_retry_on_rotation retry the whole function from scratch.
        batch_msg = str(batch_exc)
        if "didn't finish successfully" in batch_msg or "Status: error" in batch_msg:
            logger.warning("Batch job failed (%s) — rotating credential for retry", batch_msg)
            if rotate_credentials():
                raise CredentialRotatedError(
                    f"Batch job failed, rotated credential for retry: {batch_msg}"
                )
        raise
    logger.info("Batch job %s finished", job.job_id)

    # Find the result file
    tifs = sorted(output_dir.glob("*.tif")) + sorted(output_dir.glob("*.tiff"))
    if tifs:
        import shutil
        shutil.copy2(str(tifs[0]), str(output_path))
    elif output_dir.is_file():
        import shutil
        shutil.copy2(str(output_dir), str(output_path))

    _enforce_cache_limit()
    try:
        record_credential_usage(_credential_index, "success", _product_from_title(title))
    except Exception:
        pass
    return output_path


# === SECTION: Public API (NDVI, WorldCover, SAR, harmonics) ===


@_retry_on_rotation
def get_ndvi_composite(
    bbox_wgs84: Dict[str, float],
    year: int = 2023,
    _conn: Any = None,
) -> Dict[str, Any]:
    """Fetch a cloud-free NDVI composite for a bounding box.

    Uses the growing season (April–September) of *year* and computes
    the temporal **median** NDVI after cloud masking with SCL dilation.

    Parameters
    ----------
    bbox_wgs84 : dict
        ``{"west": float, "south": float, "east": float, "north": float}``
        in EPSG:4326.
    year : int
        Year to process (default 2023).

    Returns
    -------
    dict
        ``{"ndvi": np.ndarray (H,W), "transform": Affine, "crs": CRS,
        "date_range": str}``
    """
    # Short-circuit if all credentials are already exhausted
    if credits_exhausted:
        logger.warning("Skipping NDVI composite — all Copernicus credentials exhausted")
        return None

    bbox = _validate_bbox(bbox_wgs84)
    start_date = f"{year}-04-01"
    end_date = f"{year}-09-30"
    date_range = f"{start_date}/{end_date}"

    cache_file = _cache_path("ndvi_composite", bbox, year=year)
    if cache_file.exists() and _validate_cache(cache_file):
        logger.info("Cache hit for NDVI composite: %s", cache_file)
        _touch_cache(cache_file)
        data, transform, crs = _read_geotiff(cache_file)
        return {
            "ndvi": data.astype(np.float32),
            "transform": transform,
            "crs": crs,
            "date_range": date_range,
        }

    logger.info("Fetching NDVI composite for bbox=%s, year=%d", bbox, year)
    conn = _conn or _get_connection()

    # Load Sentinel-2 L2A with B04 (Red), B08 (NIR), and SCL (Scene Classification)
    s2 = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=[start_date, end_date],
        bands=["B04", "B08", "SCL"],
    )

    # Cloud masking using SCL dilation (removes clouds, cloud shadows, etc.)
    s2_masked = s2.process(
        "mask_scl_dilation",
        data=s2,
        scl_band_name="SCL",
    )

    # Compute NDVI: (B08 - B04) / (B08 + B04)
    ndvi_cube = s2_masked.ndvi(nir="B08", red="B04")

    # Temporal median composite
    ndvi_composite = ndvi_cube.reduce_dimension(
        dimension="t",
        reducer="median",
    )

    # Download
    try:
        _run_datacube(ndvi_composite, cache_file, title=f"NDVI composite {year}", bbox=bbox)
    except (CredentialRotatedError, CreditsExhaustedError, IPThrottledError):
        raise  # let @_retry_on_rotation handle these
    except Exception as exc:
        logger.error("Failed to download NDVI composite: %s", exc)
        raise RuntimeError(f"NDVI composite download failed: {exc}") from exc

    data, transform, crs = _read_geotiff(cache_file)
    return {
        "ndvi": data.astype(np.float32),
        "transform": transform,
        "crs": crs,
        "date_range": date_range,
    }


@_retry_on_rotation
def get_ndvi_timeseries(
    bbox_wgs84: Dict[str, float],
    start_date: str,
    end_date: str,
    progress_fn: Any = None,
) -> Dict[str, Any]:
    """Fetch monthly NDVI aggregates over a period.

    Downloads one NDVI composite per month and stacks them locally.
    This avoids the openEO aggregate_temporal_period bug where
    sync download collapses multi-band temporal output to 1 band.

    Parameters
    ----------
    bbox_wgs84 : dict
        Bounding box in EPSG:4326.
    start_date, end_date : str
        ISO date strings, e.g. ``"2023-01-01"`` and ``"2023-12-31"``.

    Returns
    -------
    dict
        ``{"monthly_ndvi": {"2023-01": ndarray, ...},
        "transform": Affine, "crs": CRS}``
    """
    # Short-circuit if all credentials are already exhausted
    if credits_exhausted:
        logger.warning("Skipping NDVI time series — all Copernicus credentials exhausted")
        return None

    from datetime import datetime
    import calendar

    bbox = _validate_bbox(bbox_wgs84)

    # Check for stacked cache (new format)
    cache_file = _cache_path("ndvi_ts_v2", bbox, start=start_date, end=end_date)
    if cache_file.exists() and _validate_cache(cache_file):
        logger.info("Cache hit for NDVI time series v2: %s", cache_file)
        _touch_cache(cache_file)
        return _parse_timeseries_tiff(cache_file, start_date, end_date)

    logger.info(
        "Fetching NDVI time series (per-month) for bbox=%s, %s → %s",
        bbox, start_date, end_date,
    )

    # Build month list — skip winter months (Nov-Feb) which often have
    # zero cloud-free scenes in Austria, causing openEO EmptyBounds errors
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    SKIP_MONTHS = {11, 12, 1, 2}  # Nov-Feb: snow/clouds, no usable NDVI
    months = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        if current.month not in SKIP_MONTHS:
            months.append(current)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    logger.info("NDVI months to fetch: %s", [m.strftime('%Y-%m') for m in months])

    # Fetch each month as a separate NDVI composite — sequential downloads
    monthly_ndvi: Dict[str, np.ndarray] = {}
    transform = None
    crs = None

    conn = _get_connection()

    # Build download tasks: (label, cache_path, datacube_or_None)
    tasks = []
    for m in months:
        label = m.strftime("%Y-%m")
        last_day = calendar.monthrange(m.year, m.month)[1]
        m_start = m.strftime("%Y-%m-%d")
        m_end = m.replace(day=last_day).strftime("%Y-%m-%d")
        month_cache = _cache_path("ndvi_month", bbox, start=m_start, end=m_end)
        tasks.append((label, m_start, m_end, month_cache))

    # Check which months need downloading
    to_download = []
    for label, m_start, m_end, month_cache in tasks:
        if month_cache.exists() and _validate_cache(month_cache):
            logger.debug("Cache hit for %s: %s", label, month_cache)
            _touch_cache(month_cache)
        else:
            to_download.append((label, m_start, m_end, month_cache))

    # Download missing months in parallel (one credential per worker)
    # Tracks consecutive 402 hits per credential to distinguish transient
    # rate-limits from genuine credit exhaustion.
    _consecutive_402: Dict[int, int] = {}  # cred_idx -> consecutive 402 count
    _402_THRESHOLD = 3  # mark exhausted only after this many consecutive 402s

    def _download_month_sequential(label, m_start, m_end, month_cache):
        """Download a single month's NDVI using the current credential.
        Returns (label, error_or_None).

        On transient 402 errors (credential probes OK but download fails),
        retries up to 3 times with backoff, then rotates to a different
        credential and retries again.  Gives up only after all credentials
        have been tried."""
        import time as _time
        max_retries_per_cred = 3
        tried_creds: set[int] = set()  # credentials already exhausted via retries
        attempt = 0
        last_exc: Exception | None = None
        _download_month_sequential._proxy_tried = False

        while True:
            if credits_exhausted:
                return label, CreditsExhaustedError("all exhausted")
            try:
                cred_idx = _credential_index
                c = _get_connection()
                s2 = c.load_collection(
                    "SENTINEL2_L2A",
                    spatial_extent=bbox,
                    temporal_extent=[m_start, m_end],
                    bands=["B04", "B08", "SCL"],
                )
                s2_masked = s2.process(
                    "mask_scl_dilation", data=s2, scl_band_name="SCL",
                )
                ndvi_cube = s2_masked.ndvi(nir="B08", red="B04")
                ndvi_median = ndvi_cube.reduce_dimension(
                    dimension="t", reducer="median",
                )
                month_cache.parent.mkdir(parents=True, exist_ok=True)
                tmp_month = month_cache.with_suffix(month_cache.suffix + ".tmp")
                try:
                    ndvi_median.download(str(tmp_month), format="GTiff")
                    if tmp_month.exists() and tmp_month.stat().st_size > 0:
                        tmp_month.rename(month_cache)
                    else:
                        tmp_month.unlink(missing_ok=True)
                        raise RuntimeError("download produced empty file")
                except Exception:
                    tmp_month.unlink(missing_ok=True)
                    raise
                logger.info("NDVI %s downloaded OK%s", label,
                            " (via proxy)" if getattr(_download_month_sequential, '_proxy_tried', False) else "")
                try:
                    record_credential_usage(cred_idx, "success", "ndvi_ts")
                except Exception:
                    pass
                # Clear proxy on success
                try:
                    c.session.proxies.clear()
                except Exception:
                    pass
                return label, None
            except CredentialRotatedError:
                try:
                    record_credential_usage(cred_idx, "rotated", "ndvi_ts")
                except Exception:
                    pass
                tried_creds.add(cred_idx)
                if len(tried_creds) >= len(_CREDENTIALS):
                    logger.warning("NDVI %s: 402 on all %d credentials — IP-throttled",
                                   label, len(tried_creds))
                    return label, IPThrottledError(
                        f"NDVI {label}: all {len(_CREDENTIALS)} credentials 402")
                logger.info("NDVI %s: credential rotated to %d, retrying",
                            label, _credential_index + 1)
                attempt = 0  # reset attempts for fresh credential
                continue
            except CreditsExhaustedError as exc:
                logger.error("NDVI %s: all credentials exhausted", label)
                return label, exc
            except IPThrottledError as exc:
                logger.warning("NDVI %s: IP-throttled", label)
                return label, exc
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)

                try:
                    record_credential_usage(cred_idx, "error", "ndvi_ts")
                except Exception:
                    pass

                # --- No data / overcast month detection ---
                is_nodata = False
                nodata_patterns = [
                    'EmptyBounds', 'empty collection', 'no data available',
                    'NoDataAvailable', 'ResultTooLarge',
                ]
                for pat in nodata_patterns:
                    if pat.lower() in exc_str.lower():
                        is_nodata = True
                        break
                if 'download produced empty file' in exc_str:
                    is_nodata = True

                if is_nodata:
                    logger.info(
                        "NDVI %s: no cloud-free data available (likely overcast) — skipping",
                        label,
                    )
                    return label, None

                # --- Handle 402 PaymentRequired ---
                if '402' in exc_str and 'PaymentRequired' in exc_str:
                    try:
                        # _check_credits_error probes the credential, then:
                        #   probe passes → CredentialRotatedError (transient 402)
                        #   probe fails  → CredentialRotatedError (rotated to next)
                        #   all exhausted → CreditsExhaustedError
                        _check_credits_error(exc)
                    except CredentialRotatedError:
                        tried_creds.add(cred_idx)
                        if len(tried_creds) >= len(_CREDENTIALS):
                            # All credentials tried — IP-level throttle
                            logger.warning(
                                "NDVI %s: 402 on all %d credentials — "
                                "IP-throttled", label, len(tried_creds))
                            return label, IPThrottledError(
                                f"NDVI {label}: all {len(_CREDENTIALS)} "
                                f"credentials returned 402")
                        logger.info("NDVI %s: 402, rotated to credential %d",
                                    label, _credential_index + 1)
                        attempt = 0
                        continue
                    except CreditsExhaustedError as ce:
                        return label, ce
                    # Should not reach here (402 always raises above)
                    return label, exc

                # --- Handle 429 / 500 / 503 with backoff ---
                _retryable = (
                    "429" in exc_str
                    or "500" in exc_str
                    or "502" in exc_str
                    or "503" in exc_str
                    or "504" in exc_str
                    or "Internal" in exc_str
                    or "Bad Gateway" in exc_str
                    or "Gateway Time" in exc_str
                    or "Service Unavailable" in exc_str
                    or "Server error" in exc_str
                    or "max connections" in exc_str
                )
                if _retryable and attempt < max_retries_per_cred:
                    attempt += 1
                    wait_secs = 15 * attempt
                    _reason = (
                        '429' if '429' in exc_str
                        else '502' if ('502' in exc_str or 'Bad Gateway' in exc_str)
                        else '504' if ('504' in exc_str or 'Gateway Time' in exc_str)
                        else '500' if ('500' in exc_str or 'Internal' in exc_str or 'Server error' in exc_str)
                        else '503'
                    )
                    # Interim retries log at INFO so they don't feed the
                    # fleet-wide copernicus throttle counter — one bad
                    # NDVI month would otherwise emit 3 WARNING lines
                    # plus library noise. Final-attempt exhaustion is
                    # logged as WARNING by the 'failed: ... — skipping
                    # month' path below.
                    logger.info(
                        "NDVI %s: server error (%s), retry %d/%d in %ds...",
                        label, _reason,
                        attempt, max_retries_per_cred, wait_secs,
                    )
                    _time.sleep(wait_secs)
                    continue
                # --- Other errors — skip month ---
                logger.warning("NDVI %s failed: %s — skipping month", label, exc)
                return label, exc
        # Should not reach here, but safety net
        return label, last_exc or Exception(f"NDVI {label}: retries exhausted")

    if to_download:
        import time as _time
        n_total = len(tasks)
        n_done = n_total - len(to_download)  # already cached

        # Compute a bbox hash for failed-month cooldown tracking
        _bbox_hash = f"{bbox.get('west',0):.4f}_{bbox.get('south',0):.4f}_{bbox.get('east',0):.4f}_{bbox.get('north',0):.4f}"

        # Prune expired cooldowns
        _now = _time.time()
        expired = [k for k, v in _FAILED_MONTH_COOLDOWNS.items() if v < _now]
        for k in expired:
            _FAILED_MONTH_COOLDOWNS.pop(k, None)

        # Filter out months in cooldown (recently failed with 500 or persistent 402)
        actually_download = []
        for label, m_start, m_end, month_cache in to_download:
            cooldown_key = (_bbox_hash, label)
            if cooldown_key in _FAILED_MONTH_COOLDOWNS:
                remaining = int(_FAILED_MONTH_COOLDOWNS[cooldown_key] - _now)
                logger.info(
                    "NDVI %s: skipping (failed recently, cooldown %dm remaining)",
                    label, remaining // 60,
                )
                n_done += 1  # count as processed for progress
            else:
                actually_download.append((label, m_start, m_end, month_cache))

        if actually_download:
            logger.info("Downloading %d NDVI months (sequential, single credential)...",
                        len(actually_download))
        consecutive_402 = 0  # track 402 cascade
        consecutive_5xx = 0  # track upstream-stress cascade (503/502/500)
        for label, m_start, m_end, month_cache in actually_download:
            lbl, exc = _download_month_sequential(label, m_start, m_end, month_cache)
            if exc is None:
                n_done += 1
                consecutive_402 = 0  # reset on success
                consecutive_5xx = 0
                logger.info("Month %s done (%d/%d)", lbl, n_done, n_total)
                if progress_fn:
                    try:
                        progress_fn(n_done, n_total)
                    except Exception:
                        pass
            elif isinstance(exc, (CreditsExhaustedError, IPThrottledError)):
                logger.warning("Stopping NDVI downloads — %s",
                               "IP-throttled" if isinstance(exc, IPThrottledError)
                               else "credits exhausted")
                break
            else:
                n_done += 1  # count failed months too for progress
                exc_str = str(exc)
                # Determine cooldown duration based on error type
                if '500' in exc_str or 'Spark' in exc_str or 'Server error' in exc_str:
                    cooldown = _FAILED_MONTH_COOLDOWN_SECS  # 30 min for Spark timeout
                    logger.warning("NDVI %s failed (server error) — cooldown %dm",
                                   lbl, cooldown // 60)
                elif '402' in exc_str:
                    cooldown = _FAILED_MONTH_402_COOLDOWN_SECS  # 5 min for 402
                    consecutive_402 += 1
                else:
                    cooldown = _FAILED_MONTH_402_COOLDOWN_SECS
                    consecutive_402 = 0
                # Track upstream-stress cascade (503/502/500/504) separately
                # from 402: it indicates the openeo origin is overloaded,
                # not that our credential is exhausted.
                if any(s in exc_str for s in ('500', '502', '503', '504',
                                              'Bad Gateway', 'Gateway Time',
                                              'Service Unavailable',
                                              'Server error',
                                              'too many 503',
                                              'no available server')):
                    consecutive_5xx += 1
                else:
                    consecutive_5xx = 0
                _FAILED_MONTH_COOLDOWNS[(_bbox_hash, lbl)] = _time.time() + cooldown

                if progress_fn:
                    try:
                        progress_fn(n_done, n_total)
                    except Exception:
                        pass

                # 5xx cascade breaker: if 2+ consecutive months fail with
                # upstream errors, the openeo origin is sick. Stop hammering
                # it — raise IPThrottledError so the parent process pauses
                # this peer for 15 min via the existing copernicus_paused
                # path. The director will switch to another peer.
                if consecutive_5xx >= 2:
                    logger.warning(
                        "Upstream-stress cascade detected (%d consecutive 5xx) — "
                        "aborting NDVI downloads to give openeo origin a break",
                        consecutive_5xx,
                    )
                    raise IPThrottledError(
                        f"openeo origin returned {consecutive_5xx} consecutive 5xx "
                        f"— backing off"
                    )

                # 402 cascade breaker: if 2+ consecutive months fail with 402,
                # the credential is rate-limited. Stop wasting time on remaining months.
                if consecutive_402 >= 2:
                    remaining_count = len(actually_download) - actually_download.index(
                        (label, m_start, m_end, month_cache)) - 1
                    if remaining_count > 0:
                        logger.warning(
                            "402 cascade detected (%d consecutive) — skipping "
                            "%d remaining months (cooldown %dm)",
                            consecutive_402, remaining_count,
                            _FAILED_MONTH_402_COOLDOWN_SECS // 60,
                        )
                        # Put remaining months into cooldown too
                        idx = actually_download.index(
                            (label, m_start, m_end, month_cache))
                        for rl, rm_s, rm_e, rmc in actually_download[idx + 1:]:
                            _FAILED_MONTH_COOLDOWNS[(_bbox_hash, rl)] = (
                                _time.time() + _FAILED_MONTH_402_COOLDOWN_SECS
                            )
                            n_done += 1
                    break
            _time.sleep(2)  # gentle pacing between sequential downloads

    # Read all cached months
    for label, m_start, m_end, month_cache in tasks:
        if not month_cache.exists() or not _validate_cache(month_cache):
            continue
        try:
            with rasterio.open(str(month_cache)) as ds:
                data = ds.read(1).astype(np.float32)
                if transform is None:
                    transform = ds.transform
                    crs = ds.crs
                monthly_ndvi[label] = data
        except Exception as exc:
            logger.warning("Failed to read NDVI %s: %s", label, exc)

    if not monthly_ndvi:
        raise RuntimeError("No monthly NDVI data retrieved")

    logger.info("NDVI time series: %d/%d months retrieved", len(monthly_ndvi), len(months))

    # Stack into multi-band cache for future use
    try:
        ref_shape = next(iter(monthly_ndvi.values())).shape
        sorted_labels = sorted(monthly_ndvi.keys())
        stack = np.stack([monthly_ndvi[l] for l in sorted_labels], axis=0)
        with rasterio.open(
            str(cache_file), "w", driver="GTiff",
            height=ref_shape[0], width=ref_shape[1],
            count=len(sorted_labels), dtype="float32",
            crs=crs, transform=transform,
        ) as dst:
            for i, label in enumerate(sorted_labels):
                dst.write(stack[i], i + 1)
                dst.set_band_description(i + 1, label)
        logger.info("Saved stacked NDVI TS cache: %s (%d bands)", cache_file, len(sorted_labels))
    except Exception as exc:
        logger.warning("Failed to write stacked cache: %s", exc)

    return {
        "monthly_ndvi": monthly_ndvi,
        "transform": transform,
        "crs": crs,
    }


def _parse_timeseries_tiff(
    path: pathlib.Path,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Parse a multi-band GeoTIFF where each band is a monthly aggregate."""
    if rasterio is None:
        raise ImportError("rasterio is required")

    with rasterio.open(str(path)) as ds:
        data = ds.read()  # (bands, H, W)
        transform = ds.transform
        crs = ds.crs
        band_count = ds.count
        descriptions = [ds.descriptions[i] if ds.descriptions[i] else None for i in range(band_count)]

    # Build month labels from date range
    from datetime import datetime

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    monthly_ndvi: Dict[str, np.ndarray] = {}
    for i in range(band_count):
        # Prefer band description (set by our stacked cache writer)
        if descriptions[i]:
            label = descriptions[i]
        elif i < len(months):
            label = months[i]
        else:
            label = f"band_{i+1}"
        monthly_ndvi[label] = data[i].astype(np.float32)

    return {
        "monthly_ndvi": monthly_ndvi,
        "transform": transform,
        "crs": crs,
    }


@_retry_on_rotation
def get_land_cover(
    bbox_wgs84: Dict[str, float],
    _conn: Any = None,
) -> Dict[str, Any]:
    """Fetch ESA WorldCover 10 m land-use classification.

    Parameters
    ----------
    bbox_wgs84 : dict
        Bounding box in EPSG:4326.

    Returns
    -------
    dict
        ``{"map": np.ndarray (H,W), "transform": Affine, "crs": CRS,
        "classes": dict}``
    """
    # Short-circuit if all credentials are already exhausted
    if credits_exhausted:
        logger.warning("Skipping land cover — all Copernicus credentials exhausted")
        return None

    bbox = _validate_bbox(bbox_wgs84)

    cache_file = _cache_path("landcover", bbox)
    if cache_file.exists() and _validate_cache(cache_file):
        logger.info("Cache hit for land cover: %s", cache_file)
        _touch_cache(cache_file)
        data, transform, crs = _read_geotiff(cache_file)
        return {
            "map": data.astype(np.uint8),
            "transform": transform,
            "crs": crs,
            "classes": WORLDCOVER_CLASSES.copy(),
        }

    logger.info("Fetching ESA WorldCover for bbox=%s", bbox)
    conn = _conn or _get_connection()

    # ESA WorldCover 10m 2021 v2 — single band "MAP"
    # Temporal extent is required by openEO even for static datasets
    lc = conn.load_collection(
        "ESA_WORLDCOVER_10M_2021_V2",
        spatial_extent=bbox,
        temporal_extent=["2021-01-01", "2021-12-31"],
        bands=["MAP"],
    )

    # Reduce the (trivial) temporal dimension
    lc_flat = lc.reduce_dimension(dimension="t", reducer="first")

    try:
        _run_datacube(lc_flat, cache_file, title="ESA WorldCover", bbox=bbox)
    except (CredentialRotatedError, CreditsExhaustedError, IPThrottledError):
        raise  # let @_retry_on_rotation handle these
    except Exception as exc:
        logger.error("Failed to download land cover: %s", exc)
        raise RuntimeError(f"Land cover download failed: {exc}") from exc

    data, transform, crs = _read_geotiff(cache_file)
    return {
        "map": data.astype(np.uint8),
        "transform": transform,
        "crs": crs,
        "classes": WORLDCOVER_CLASSES.copy(),
    }


@_retry_on_rotation
def get_sar_backscatter(
    bbox_wgs84: Dict[str, float],
    start_date: str,
    end_date: str,
    _conn: Any = None,
) -> Dict[str, Any]:
    """Fetch Sentinel-1 SAR VV+VH backscatter composite.

    SAR penetrates clouds and can distinguish built structures from
    vegetation regardless of season.

    Parameters
    ----------
    bbox_wgs84 : dict
        Bounding box in EPSG:4326.
    start_date, end_date : str
        ISO date strings.

    Returns
    -------
    dict
        ``{"vv": np.ndarray (H,W), "vh": np.ndarray (H,W),
        "vv_vh_ratio": np.ndarray (H,W),
        "transform": Affine, "crs": CRS,
        "date_range": str}``
    """
    # Short-circuit if all credentials are already exhausted
    if credits_exhausted:
        logger.warning("Skipping SAR backscatter — all Copernicus credentials exhausted")
        return None

    bbox = _validate_bbox(bbox_wgs84)

    cache_file = _cache_path("sar", bbox, start=start_date, end=end_date)
    if cache_file.exists() and _validate_cache(cache_file):
        try:
            logger.info("Cache hit for SAR backscatter: %s", cache_file)
            _touch_cache(cache_file)
            return _parse_sar_tiff(cache_file, start_date, end_date)
        except Exception as exc:
            logger.warning("Corrupt SAR cache %s (%s), deleting and re-fetching",
                          cache_file, exc)
            cache_file.unlink(missing_ok=True)

    logger.info(
        "Fetching SAR backscatter for bbox=%s, %s → %s",
        bbox, start_date, end_date,
    )
    conn = _conn or _get_connection()

    s1 = conn.load_collection(
        "SENTINEL1_GRD",
        spatial_extent=bbox,
        temporal_extent=[start_date, end_date],
        bands=["VV", "VH"],
    )

    # Apply SAR backscatter processing (terrain correction)
    s1_processed = s1.sar_backscatter(
        coefficient="sigma0-ellipsoid",
    )

    # Temporal median composite
    sar_composite = s1_processed.reduce_dimension(
        dimension="t",
        reducer="median",
    )

    try:
        _run_datacube(sar_composite, cache_file, title="SAR backscatter", bbox=bbox)
    except (CredentialRotatedError, CreditsExhaustedError, IPThrottledError):
        raise  # let @_retry_on_rotation handle these
    except Exception as exc:
        logger.error("Failed to download SAR backscatter: %s", exc)
        raise RuntimeError(f"SAR backscatter download failed: {exc}") from exc

    return _parse_sar_tiff(cache_file, start_date, end_date)


def _parse_sar_tiff(
    path: pathlib.Path,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Parse a 2-band SAR GeoTIFF (VV, VH)."""
    if rasterio is None:
        raise ImportError("rasterio is required")

    import logging as _logging
    _rasterio_logger = _logging.getLogger('rasterio._env')
    _prev_level = _rasterio_logger.level
    _rasterio_logger.setLevel(_logging.ERROR)
    try:
        with rasterio.open(str(path)) as ds:
            data = ds.read()  # (bands, H, W)
            transform = ds.transform
            crs = ds.crs
    finally:
        _rasterio_logger.setLevel(_prev_level)

    # Bands: VV=0, VH=1
    vv = data[0].astype(np.float32) if data.ndim == 3 and data.shape[0] >= 1 else data.astype(np.float32)
    vh = data[1].astype(np.float32) if data.ndim == 3 and data.shape[0] >= 2 else np.zeros_like(vv)

    # VV/VH ratio (useful for distinguishing land cover types)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(vh != 0, vv / vh, 0.0).astype(np.float32)

    return {
        "vv": vv,
        "vh": vh,
        "vv_vh_ratio": ratio,
        "transform": transform,
        "crs": crs,
        "date_range": f"{start_date}/{end_date}",
    }


# === SECTION: Utility helpers ===


def clear_cache() -> int:
    """Remove all cached files. Returns the number of files deleted."""
    count = 0
    for f in CACHE_DIR.iterdir():
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    logger.info("Cleared %d cached files", count)
    return count


def bbox_from_center(
    lon: float,
    lat: float,
    size_m: float = 5000.0,
) -> Dict[str, float]:
    """Create a WGS-84 bbox centred on (lon, lat) with *size_m* half-width.

    Useful for quick queries around a point of interest.
    """
    import math

    # Approximate degree offsets
    dlat = size_m / 111_320.0
    dlon = size_m / (111_320.0 * math.cos(math.radians(lat)))
    return {
        "west": lon - dlon,
        "south": lat - dlat,
        "east": lon + dlon,
        "north": lat + dlat,
    }


def ndvi_quality_mask(
    ndvi: np.ndarray,
    min_val: float = -1.0,
    max_val: float = 1.0,
) -> np.ndarray:
    """Return a boolean mask where NDVI values are within a valid range."""
    return (ndvi >= min_val) & (ndvi <= max_val) & np.isfinite(ndvi)


# === SECTION: CLI smoke test ===

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Small test area in Vienna
    test_bbox = bbox_from_center(16.37, 48.21, size_m=2000)
    print(f"Test bbox: {test_bbox}")

    print("\n--- NDVI Composite ---")
    try:
        result = get_ndvi_composite(test_bbox, year=2023)
        print(f"  Shape: {result['ndvi'].shape}")
        print(f"  NDVI range: [{np.nanmin(result['ndvi']):.3f}, {np.nanmax(result['ndvi']):.3f}]")
        print(f"  CRS: {result['crs']}")
        print(f"  Date range: {result['date_range']}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n--- Land Cover ---")
    try:
        lc = get_land_cover(test_bbox)
        print(f"  Shape: {lc['map'].shape}")
        unique, counts = np.unique(lc["map"], return_counts=True)
        for u, c in zip(unique, counts):
            name = lc["classes"].get(int(u), "unknown")
            print(f"  Class {u:3d} ({name:25s}): {c:6d} px")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n--- SAR Backscatter ---")
    try:
        sar = get_sar_backscatter(test_bbox, "2023-06-01", "2023-08-31")
        print(f"  VV shape: {sar['vv'].shape}")
        print(f"  VH shape: {sar['vh'].shape}")
    except Exception as e:
        print(f"  Error: {e}")

# Copernicus Throttle & Retry — Mental Model

**Read this before touching copernicus.py, tile_cache.py, or the Copernicus path in austria_processor.py.**

#### The Problem

Copernicus openEO returns HTTP 402 PaymentRequired when rate-limited. This is NOT always genuine credit exhaustion — often it's a transient rate-limit where the credential is still healthy (auth works, quota page shows credits remaining). We have 4 credentials. Sometimes only 1-2 are throttled; sometimes all 4 are. Recovery takes minutes to hours.

#### Three-Layer Architecture

```
copernicus.py          tile_cache.py           austria_processor.py
(API layer)            (cache layer)           (orchestrator)
─────────────          ──────────────          ────────────────────
402 detected           re-raises               writes pause file
  │                    IPThrottledError        aborts tile loop
  ├─ probe credential  and                     re-queues KG
  │  passes? → rotate  CreditsExhaustedError   polls every 15 min
  │  + CredentialRotatedError  (never returns     until probe passes
  │                             None for these)
  ├─ probe fails? → mark
  │  exhausted + rotate
  │  + CredentialRotatedError
  │
  └─ all exhausted?
     → CreditsExhaustedError

@_retry_on_rotation decorator
  catches CredentialRotatedError
  → rebuilds connection + datacube with next credential
  → retries the entire function
  → after len(_CREDENTIALS)+1 attempts:
      all probes passed → IPThrottledError (transient)
      some probes failed → CreditsExhaustedError (genuine)
```

#### Exception Types (copernicus.py)

| Exception | Meaning | Caught by |
|-----------|---------|----------|
| `CredentialRotatedError` | One credential got 402, rotated to next | `@_retry_on_rotation` decorator (retries with fresh cred) |
| `CreditsExhaustedError` | ALL credentials genuinely exhausted (probes failed) | tile_cache re-raises → austria_processor writes pause file |
| `IPThrottledError(RuntimeError)` | ALL credentials got transient 402 (probes passed) | tile_cache re-raises → austria_processor writes pause file |

**Critical rule**: `CredentialRotatedError`, `CreditsExhaustedError`, and `IPThrottledError` must NEVER be swallowed by generic `except Exception` blocks. The public functions (`get_ndvi_composite`, `get_land_cover`, `get_sar_backscatter`) have explicit `except (CredentialRotatedError, CreditsExhaustedError, IPThrottledError): raise` before their generic handler.

#### Credential Probing

`_check_credits_error(exc)` is called on every 402. It:
1. Authenticates the credential against the OIDC endpoint (not a data download)
2. If auth succeeds → "transient 402" → rotates + `CredentialRotatedError`
3. If auth fails with 402 → "genuinely exhausted" → marks credential, rotates + `CredentialRotatedError` (or `CreditsExhaustedError` if all gone)

**Why probes always pass during rate-limiting**: the auth endpoint is separate from the processing endpoint. A credential can authenticate fine but still get 402 on downloads. This is why we must try ALL credentials — the rate-limit may be per-credential or per-IP or timing-dependent.

#### Sync → Batch Fallback

Each product download tries sync first (3 min timeout), then batch job:
- Sync: `datacube.download()` — fast for 0.1° cells, often 402'd or times out
- Batch: `datacube.execute_batch()` — slower (5-15 min) but more reliable
- The `@_retry_on_rotation` decorator wraps the entire function, so a 402 on sync in credential 1 → retry the whole function with credential 2 (new sync attempt, then batch fallback)

#### Per-Product Retry Flow

**NDVI/WorldCover/SAR** (`get_ndvi_composite`, `get_land_cover`, `get_sar_backscatter`):
```
@_retry_on_rotation (up to 5 attempts with 4 creds)
  └─ build datacube with current credential
     └─ _run_datacube()
        ├─ sync download (1 attempt)
        │   402 → _check_credits_error → CredentialRotatedError → decorator retries
        │   timeout → fall through to batch
        └─ batch job
            402 → _check_credits_error → CredentialRotatedError → decorator retries
```

**NDVI Time Series** (`get_ndvi_timeseries` → `_download_month_sequential`):
- Downloads 8 months (Mar-Oct) sequentially
- Each month has its own retry loop with credential tracking (`tried_creds` set)
- After all 4 credentials fail for one month → returns `IPThrottledError` for that month
- Download loop: if `IPThrottledError` or `CreditsExhaustedError` returned → breaks immediately (cascade breaker)
- Also has per-month cooldown (`_FAILED_MONTH_COOLDOWNS`) — skips months that failed recently

#### tile_cache.py Bridge

All 4 `_fetch_*_cell` methods follow the same pattern:
```python
try:
    result = copernicus.get_*(cell_bbox, ...)
except (CreditsExhaustedError, IPThrottledError):
    raise   # NEVER swallowed — propagates to austria_processor
except server_error:
    retry with backoff
except other:
    return None  # soft failure for non-throttle errors
```

Additional safety: if `last_exc` contains "IP-throttled", raises `IPThrottledError` even from the `return None` path.

#### austria_processor.py Response

**`_try_fetch_single(bbox)`**: Early-bails if `copernicus.ip_throttled` flag is set.

**`_fetch_copernicus_for_tile()`**: On `IPThrottledError`/`CreditsExhaustedError`, re-raises immediately — no quadrant fallback. Quadrant fallback only triggers on timeouts/server errors.

**Tile loop** (inside `process_one_kg`):
```
try:
    copernicus_data = _fetch_copernicus_for_tile(...)
except (CreditsExhaustedError, IPThrottledError):
    result["copernicus_exhausted"] = True
    result["success"] = False
    COPERNICUS_PAUSE_FILE.write_text(...)   # data/austria_processor/copernicus_paused
    break   # ABORT tile loop
```

After tile loop, if `copernicus_exhausted + success=False`: return early (skip GPKG/JSON build).

**Parent process** (`main()`):
1. Subprocess returns with `copernicus_exhausted=True` → `is_credits_issue` check
2. KG added to `retry_queue.json` (tile checkpoints preserved)
3. Enters pause loop: sleeps 15 min → `_copernicus_probe()` → if OK, deletes pause file + resumes
4. `_copernicus_probe()` resets both `ip_throttled` and `credits_exhausted` flags, clears all cached connections, then tries a tiny NDVI download
5. On resume, the re-queued KG is processed next (tile checkpoints restore completed tiles)

#### Proxies — NOT Useful for Copernicus

`bev_proxy.py` manages a pool of free HTTPS proxies from GitHub lists. These are useful for BEV GeoTIFF range reads but **do not help with Copernicus 402s** because:
- openEO authentication is per-credential (OAuth client_credentials), not per-IP
- Rate-limiting is tied to the credential's account, not the source IP
- Free proxies are unreliable and slow for the data volumes openEO returns

Historical note: proxy rotation for Copernicus was tried and removed. The solution is credential rotation (try all 4), not IP rotation.

#### 4 Credentials (copernicus.py line ~48)

| Index | Client ID prefix | Notes |
|-------|-----------------|-------|
| 1 | `sh-f36653c6` | Fresh 2026-04 |
| 2 | `sh-8d8c685f` | Renews 2026-05-01 |
| 3 | `sh-2ed25dbb` | Renews 2026-05-01 |
| 4 | `sh-07af1740` | 30k credits |

All share the same CDSE quota pools (openEO, Sentinel Hub, COG, S3). Currently only openEO is used. Each account has 10k openEO credits/month.

#### Key Files & Flags

| File/Flag | Location | Purpose |
|-----------|----------|--------|
| `copernicus_paused` | `data/austria_processor/` | Pause file — parent polls every 15 min when present |
| `openeo_circuit.json` | `data/austria_processor/` | Circuit breaker — backs off on consecutive failures |
| `copernicus.ip_throttled` | Module global (per-process) | Fast-bail flag — set by decorator after all creds fail |
| `copernicus._exhausted_cred_indices` | Module global (per-process) | Set of credential indices confirmed genuinely exhausted |
| `copernicus._IP_THROTTLE_COOLDOWN` | 7200 (2 hours) | How long `ip_throttled` stays True before auto-reset |
| `_FAILED_MONTH_COOLDOWNS` | Module global dict | Per-(bbox,month) cooldown timestamps for NDVI TS |

**Process architecture note**: Module globals (`ip_throttled`, `_exhausted_cred_indices`, etc.) live in the subprocess (one per KG). They reset when a new KG starts in a fresh subprocess. The pause file is the cross-process communication mechanism.

#### Operational Commands

```bash
# Check if paused
cat data/austria_processor/copernicus_paused

# Force resume (probe will re-validate on next KG)
rm -f data/austria_processor/copernicus_paused

# Reset all throttle state
rm -f data/austria_processor/copernicus_paused data/austria_processor/openeo_circuit.json
sudo systemctl restart austria_processor

# Check which credential is active in the subprocess
grep 'Authenticated successfully\|Rotated to credential\|IP-throttled\|transient 402' \
  data/austria_processor/logs/processor.log | tail -20
```

---

---

*See `AGENTS.md` for the project map.*

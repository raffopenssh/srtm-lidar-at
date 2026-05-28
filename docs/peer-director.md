# Peer Director — Multi-Instance Orchestration

**Read this section before touching peer_director.py, deploy.sh, or any director API endpoint.**

#### Architecture: One Frontier + Many Cache-Only Peers

The system uses a **single director** (the primary instance, `srtm-lidar-at`) to
orchestrate processing across multiple exe.dev VMs. Each VM has 100 GB/month
bandwidth.

**Two roles**:
- **Frontier** (one at a time): runs full processing including Copernicus + Hansen
  fetches.  Touches the shared Copernicus credentials, so only one frontier
  may run.  All credential rotation happens here.
- **Cache-only** (many in parallel): processor started with `--cache-only` /
  `COPERNICUS_FORBIDDEN=1`.  Refuses any Copernicus/Hansen API call — if a
  tile isn't in the local + Zenodo cache, it raises `CacheMissError` and
  the KG is re-queued for the frontier.  Peer is fed an explicit whitelist
  of fully-cached KGs computed by the director.

**Key invariants**:
- exactly one frontier peer running at any time (credential safety)
- up to `max_cache_only_peers` (default 8) cache-only peers running in parallel
- at least `min_reserve_peers` (default 0 — was 5; with cheap unlimited
  peers we no longer keep a reserve) enabled peers stay idle
- all Zenodo writes (KG uploads + tile-cache flushes) serialise through a
  single mutex broker on the primary (`/api/v1/zenodo/lock`)

```
┌─────────────────────────────────────────────────────┐
│  PRIMARY (srtm-lidar-at)                            │
│  ────────────────────────────────────                │
│  • Runs the Peer Director loop (peer_director.py)    │
│  • Has data/austria_processor/is_director flag        │
│  • Monitors bandwidth via vnstat on all peers         │
│  • Starts/stops processors on peers via REST API      │
│  • Syncs priority queue to the active peer             │
│  • Switches active peer when bandwidth < 2 GB         │
│  • Hosts the search index + combined dashboard         │
├─────────────────────────────────────────────────────┤
│  PEER at2 (srtm-lidar-at2)                          │
│  PEER at3 (srtm-lidar-at3)                          │
│  ────────────────────────────────────                │
│  • NO director loop (no is_director flag)             │
│  • NO systemd autostart for austria_processor         │
│  • Processor started/stopped ONLY by the director     │
│  • Shares Zenodo tile-cache deposit with primary        │
│  • Same codebase, same Copernicus credentials          │
└─────────────────────────────────────────────────────┘
```

**Critical invariant**: Only the primary runs the director loop. Peers are
passive workers. If a peer were to run its own director loop, it would
start/stop processors in conflict with the primary. This is enforced by the
`data/austria_processor/is_director` flag file (only exists on the primary).

#### Cache-only peers (parallel processing)

A peer started with `--cache-only` activates `tile_cache.FORBID_REMOTE`.
All `_fetch_*_cell` methods on `CopernicusTileCache` and `HansenTileCache.get_raw`
raise `CacheMissError` instead of calling the API.  The processor's tile loop
catches it, sets `result['cache_incomplete']=True`, and aborts the KG.  The
parent (`main()`) then re-queues the KG via `_append_retry_queue()` without
marking it failed — the frontier peer will pick it up later.

The director computes the cache-ready whitelist via `_compute_cache_ready_kgs()`:
1. Read `cache_manifest.json` and intersect lat-strip availability across
   `ndvi`, `sar`, `harmonics`, `worldcover`, `hansen` (cheap dict lookup).
2. For each candidate KG (bbox falls inside a covered strip), call
   `tile_cache.is_kg_fully_cached(bbox)` — walks per-cell index, no downloads.
3. Cache result for 5 minutes; the cache extends as the frontier fetches more.

When starting a cache-only peer, the director PUTs a slice of the whitelist
as the peer's priority queue and starts the processor with `cache_only=True`.
Different cache-only peers get different slices to reduce overlap.

#### Zenodo upload mutex

All Zenodo writes serialise through `/api/v1/zenodo/lock` on the primary:
- `upload_kg_to_zenodo()` wraps the entire upload in `zenodo_upload_lock()`
- `flush_tile_cache_to_zenodo()` wraps `ZenodoCache.upload_all()` similarly
- Lease has TTL 120s; a daemon thread renews via `/heartbeat` every 30s
- Stale leases (no heartbeat for >TTL) are auto-released
- Peers point at the broker via env `ZENODO_LOCK_URL`, set from
  `data/austria_processor/zenodo_lock_url.txt` (`deploy.sh` writes the
  primary's URL there)
- Primary uses `http://127.0.0.1:8000` automatically (set by app.py when
  spawning the processor on a host with the `is_director` flag)
- If the broker is unreachable, peers fail open (proceed without lease)
  to avoid deadlocking the fleet on a network blip

#### Per-peer config (`peers.json`)

New/relevant fields:
- `role: "frontier" | "cache_only"` — hint to the director.  If absent,
  treated as frontier (the director may still borrow it for cache-only work
  when the frontier is elsewhere).
- `min_reserve_peers` (top-level) — default 5.  Director never starts a
  cache-only peer if it would push the idle count below this.
- `max_cache_only_peers` (top-level) — default 8.  Cap on concurrent
  cache-only peers.

#### How the Director Works

`peer_director.py` (`PeerDirector` class) runs a background thread every 30s:

1. **Re-reads `peers.json`** from disk (handles cross-worker/cross-process updates)
2. **Polls bandwidth** on all peers via `GET /api/v1/bandwidth` (vnstat)
3. **Checks active peer**:
   - Budget exhausted (< 2 GB)? → Stop it, pick next peer with most bandwidth
   - Scheduled (`not_before` in future)? → Skip it
   - Stopped unexpectedly? → Restart it (if bandwidth remains)
   - Unreachable? → Deactivate, pick another
4. **Enforces single-active**: Stops any non-active peer found running
5. **Syncs priority queue** to the active peer every ~2.5 min
6. **Saves state** to `director_state.json`

**File lock**: Only one gunicorn worker runs the director loop (fcntl file lock
on `data/austria_processor/director.lock`). The other worker skips it.

#### Bandwidth Management

- Each exe.dev VM has 100 GB/month (resets on the 17th)
- Budget set to 95 GB (5 GB headroom) in `peers.json`
- When active peer drops below 2 GB remaining → director switches to the peer
  with the most remaining bandwidth
- When ALL peers are exhausted → director logs "no peers available" and waits
- After bandwidth reset (17th) → vnstat reports drop, peers become eligible again

#### Admin Token (cluster auth)

All mutating admin/director/processing/zenodo endpoints require
`X-Admin-Token: <token>`. Loopback (127.0.0.1, no XFF) is exempt so
the in-process director and on-box CLI work without plumbing it.

- Token lives in `data/admin_token` (gitignored, mode 0600). Auto-generated
  on first start of the primary.
- Peers must have the **same** token. `deploy.sh` accepts `ADMIN_TOKEN=...`
  to install it. Without it, peer registration and director-driven
  start/stop will 401.
- Dashboard prompts for the token on first 401 and stores it in
  `localStorage` (key `srtm_admin_token`). Reset with
  `srtmResetAdminToken()` in the JS console.
- The director re-reads `data/admin_token` on every outbound peer call
  (`peer_director._admin_headers()`), so token rotations propagate
  without a director restart.
- Rotate by writing a new value to `data/admin_token` on every peer +
  primary (e.g. via `for p in peers; do scp ...; done`); no service
  restart required.

```bash
# Get the token (on primary)
cat data/admin_token

# Use it from CLI
curl -H "X-Admin-Token: $(cat data/admin_token)" \
  -X POST https://srtm-lidar-at3.exe.xyz:8000/api/v1/admin/update
```

#### Deploying a New Peer

```bash
# On the new exe.dev VM:
#   ADMIN_TOKEN required so the peer can register with the director.
#   Get it via `cat data/admin_token` on the primary.
SELF_URL=https://srtm-lidar-at4.exe.xyz:8000 \
PEER_URL=https://srtm-lidar-at.exe.xyz:8000 \
ADMIN_TOKEN=<paste from primary> \
bash deploy.sh
```

`deploy.sh` does:
1. Clones repo, installs deps, decompresses RF model
2. Installs `srv.service` (gunicorn) and `austria_processor.service`
3. Starts the web server (`srv`) but does **NOT** enable/start the processor
4. Throttle mode is OFF by default (every peer uploads full + light GPKGs)
5. Auto-registers with the director via `POST /api/v1/director/peers/add`
6. The director will start the processor when it’s this peer’s turn

**What the peer does NOT have**:
- No `data/austria_processor/is_director` flag → no director loop
- `austria_processor.service` is disabled → no systemd auto-restart
- No `peers.json` with remote peers → default config only

**After deploy, make the VM public** (from the exe.dev shell):
```bash
share set-public srtm-lidar-at4
```

#### Updating All Peers

Click **⬆ Update Peers** on the dashboard, or:
```bash
curl -X POST http://localhost:8000/api/v1/director/update_peers
```

This calls `POST /api/v1/admin/update` on each peer, which does `git pull --ff-only`
then `sudo systemctl restart srv`. The timeout on the restart is expected (the
process serving the request dies). Peers come back up in ~10s.

**Important**: The update restarts `srv` (gunicorn) but does NOT restart the
processor. If the processor is running on the active peer, it keeps running —
only the web server restarts. Code changes to `austria_processor.py` take
effect when the processor finishes the current KG and is restarted by the
director.

To force a processor restart on the active peer:
```bash
curl -X POST https://<peer>/api/v1/processing/stop
# Director will restart it automatically on next tick (~30s)
```

#### Removing a Peer

Click the ✕ button on the peer card in the dashboard, or:
```bash
curl -X DELETE http://localhost:8000/api/v1/director/peers/<peer_id>
```

This stops the peer’s processor, removes it from `peers.json` and
`peer_urls.txt`, and clears its bandwidth state. The peer VM continues
running its web server but won’t receive any work.

#### Throttle Propagation

The 🔋 Throttle button in the dashboard toggles locally AND propagates
to all remote peers via `POST /api/v1/director/throttle`. This ensures
consistent throttle state across all instances.

```bash
# Set throttle on all peers
curl -X POST http://localhost:8000/api/v1/director/throttle \
  -H 'Content-Type: application/json' -d '{"throttle": true}'
```

#### Zenodo Cache on Peers (Shared Deposit)

All peers share the same Zenodo tile-cache deposit (depo 19650075) via
`cache_manifest.json` sync. The manifest is:
- **Pushed to the active peer** when the director starts its processor
  (via `PUT /api/v1/processing/cache_manifest`)
- **Synced bidirectionally** every 5 minutes by the peer-sync thread
  in `app.py` (`_sync_peer_data`)

This means a peer processing KGs near tiles the primary already cached
will fetch them from Zenodo (HTTP range reads, ~2-3 requests per tile)
instead of re-downloading from Copernicus. Saves openEO credits.

Concurrency safety: the director enforces single-active processing, so
only one peer writes to the deposit at a time. Tile cache uploads
(`flush_tile_cache_to_zenodo`) proceed regardless of the upload throttle
(throttle only blocks big GPKG uploads).

**If a peer has a stale manifest**: `rm -rf data/austria_processor/zenodo_zip_index/`
to force re-fetch of ZIP central directories. The next sync cycle will
push the latest manifest.

#### Director API Endpoints

| Method | Path | Purpose |
|--------|------|----------|
| GET | `/api/v1/bandwidth` | Local vnstat bandwidth for this instance |
| GET | `/api/v1/director/status` | Full director state: mode, active peer, bandwidth per peer |
| POST | `/api/v1/director/mode` | Set mode: `auto`, `manual`, `paused` |
| POST | `/api/v1/director/activate` | Manually activate a specific peer |
| POST | `/api/v1/director/stop` | Stop all peers and pause the director |
| GET\|POST | `/api/v1/director/peers` | Get/update full peers config |
| POST | `/api/v1/director/peers/add` | Add a new peer dynamically |
| DELETE | `/api/v1/director/peers/<id>` | Remove a peer (stops its processor) |
| GET\|POST | `/api/v1/director/throttle` | Get/propagate throttle state to all peers |
| GET | `/api/v1/director/proxy/status` | Proxy active peer’s processing status |
| GET | `/api/v1/director/proxy/log` | Proxy active peer’s processor log |
| POST | `/api/v1/director/update_peers` | Git pull + restart srv on all remote peers |
| POST | `/api/v1/admin/update` | Git pull + restart srv (called BY director) |
| POST | `/api/v1/admin/restart_processor` | Restart processor via systemd (fallback) |
| POST | `/api/v1/admin/disable_autostart` | Disable austria_processor systemd unit |

#### Director Files (all instance-specific, NOT in git)

| File | Purpose |
|------|----------|
| `data/austria_processor/is_director` | Flag file — director loop only runs if this exists |
| `data/austria_processor/peers.json` | Peer config: IDs, URLs, enabled, not_before |
| `data/austria_processor/director_state.json` | Runtime state: active peer, bandwidth, mode |
| `data/austria_processor/director.lock` | fcntl lock — ensures single director loop across workers |
| `data/austria_processor/peer_urls.txt` | Peer URLs for the data sync thread |

#### Server-Friendliness Throttle

When BEV (`data.bev.gv.at`), Zenodo, or Copernicus servers start emitting
warnings (HTTP 0 range-read drops, 429 / 503, openEO 402s) the director
automatically reduces the number of concurrent peers so we don't hammer
them. The mechanism:

1. `ProgressTracker.add_log()` (in `austria_processor.py`) classifies every
   warning/error into `bev` / `zenodo` / `copernicus` based on substring
   tokens, and keeps a 10-min sliding window. Per-minute rates are
   exposed in `progress.json → warning_rates` and propagate to the
   director via `/api/v1/processing/status`.
2. Each director tick (~30 s) `_capacity_factor()` takes the **max**
   per-kind 5-minute rate across all peers and maps it linearly to a
   sub-factor: 1.0 at zero warnings, `THROTTLE_MIN_FACTOR` (0.30) at
   `THROTTLE_SATURATION_RATE` warnings/min. The minimum sub-factor wins.
3. An EMA (`THROTTLE_EMA_ALPHA = 0.25`, half-life ~3 ticks) smooths the
   raw value, then a slow sinusoidal drift (±10 % over a 2-hour period,
   phase derived from the hostname) overlays an organic wobble.
4. `_orchestrate_parallel_frontiers()` and `_orchestrate_cache_only()`
   multiply their caps (`max_parallel_frontiers` and
   `max_cache_only_peers`) by the factor each tick. Frontiers always
   keep at least one slot — the active frontier is never pre-empted by
   the throttle. Cache-only count *can* drop to zero on sustained
   pressure.
5. Status payload exposes `capacity_factor` and `capacity_components`;
   dashboard shows a 🌿 pill (green/yellow/red) with the rates as
   tooltip.

Saturation thresholds (warnings per minute) live in
`peer_director.py → THROTTLE_SATURATION_RATE` (re-tuned 2026-04-23 for
the ~50-peer fleet — the previous values were too tight at this scale
and were causing the throttle to bite on ambient noise):
- bev: 5.0  (a few range-read retries are normal noise)
- zenodo: 1.5  (Zenodo rate-limits aggressively)
- copernicus: 0.4  (402s should be near zero in steady state)

A dead-zone (`THROTTLE_DEAD_ZONE_FRAC = 0.10`, i.e. 10 % of saturation)
is applied before the linear ramp — fleet-max rates below the dead-zone
read as zero so a single chatty peer can't drag capacity off 100 %.
Drift amplitude reduced from ±10 % to ±4 % (cosmetic-only, was loud
enough to flip dashboard colors on its own).

The fleet aggregator also filters out peers whose state is `stopped`
(clean exit — their warning window is stale immediately) and peers
that have been `unreachable` for more than 30 minutes. A peer that
just became unreachable is still trusted: it may be doing a long
GPKG upload and briefly not responding to `/processing/status`.
Last-seen-live timestamps live in `director_state.json` under
`peer_last_live_ts`.

Backoff timings are also more tender:
- `BANDWIDTH_BACKOFF_SECONDS = 900` (15 min after 3 failed bandwidth polls)
- `ZENODO_NETWORK_COOLDOWN_MIN = 60` (was 30)
- `HOLD_TENDENCY_WINDOW_HOURS = 6` (was 3) — longer memory for repeat offenders
- `HOLD_TENDENCY_MAX_MIN = 24 * 60` (was 12 h) — a bad peer can sit out a full day
- `THROTTLE_EMA_ALPHA = 0.15` (was 0.25) — slower recovery, kinder to upstreams
- `THROTTLE_MIN_FACTOR = 0.20` (was 0.30) — deeper cuts under heavy pressure

Tuning knobs: `THROTTLE_MIN_FACTOR`, `THROTTLE_EMA_ALPHA`,
`THROTTLE_DRIFT_PERIOD_S`, `THROTTLE_DRIFT_AMPLITUDE` at the top of
`peer_director.py`.

**Per-kind sub-factor EMAs**: `capacity_components.sub_factor_emas`
holds smoothed `bev` / `zenodo` / `copernicus` sub-factors (same
α = THROTTLE_EMA_ALPHA, persisted across worker swaps). Used by
`_effective_creds_per_frontier` with hysteresis (per=1 above 0.92,
per=floor below 0.78, sticky band) so the per=1↔per=2 decision
doesn't flap on a single 402 inside the 5-min window. The flap was
causing ~8 frontier restarts/hour because each flip re-derived
`max_parallel_frontiers` and the cred-slice plan.

**Frontier restart cooldown** (`FRONTIER_RESTART_COOLDOWN_S = 180`):
each KG ends with a clean subprocess exit, so the active frontier
reports `state in ('idle','stopped')` between KGs. The director
used to fire a fresh start on every 30 s tick. Now it only restarts
when the cred slice, lat strips, or excluded-KG set actually
changed; otherwise it waits at least 180 s. Last decisions are
tracked in `_frontier_restart_log[peer_id] = {ts, fp}`.

**Constrained starts skip systemd fallback**: when
`start_peer_processor` carries `cache_only` / `cred_indices` /
`lat_strips` / `queue_whitelist`, an API 500 is retried once after
1.5 s. If still failing, we return
`{method: 'no_fallback_constrained'}` instead of calling
`/api/v1/admin/restart_processor` (which restarts
`austria_processor.service` and ignores the per-peer contract).
Without this guard, a cache-only start was turning into a frontier
start, the director then killed it as 'non-active running
frontier', and the loop repeated forever.

**Silent-peer surfacing**: `capacity_components.peers_silent_ids`
lists peers in the fleet that didn't contribute warning signal in
the latest tick (scheduled, stopped, or unreachable past the 30-min
grace). Useful when capacity_factor looks low but you can't tell
which peer is the source.

#### BEV outage handling (multi-pool parks)

When `data.bev.gv.at` becomes unreachable for some egress pools but
not others (the common failure mode — AWS NAT routes flap independently),
the director uses `_check_bev_outage` + `_classify_outage_multi_pool`
to park *the affected /24 pools only*, never the whole fleet on first
trigger.

Trigger conditions (all three):
* Fleet BEV warns/min ≥ `BEV_OUTAGE_TRIGGER_WPM` (12.0)
* over ≥ `BEV_OUTAGE_MIN_PEERS` (4) distinct peers
* sustained for ≥ `BEV_OUTAGE_TRIGGER_PERSIST_S` (180 s)

Classification:
* `≥ OUTAGE_POOL_KNOWN_FRAC` (0.60) of warns must come from peers
  with a known `outbound_24` (else most pressure is in `'?'` and we
  can't attribute — fall through to fleet scope).
* Sort known pools by 1m sum descending. Take the smallest prefix
  whose cumulative share ≥ `OUTAGE_POOL_DOMINANCE` (0.70).
* If `len(prefix) > OUTAGE_POOL_MAX_SET` (3): fleet scope.
* Otherwise: pool scope, park every peer in each prefix pool.

What "pool scope" does (per `_check_bev_outage`, the `scope == 'pools'`
branch):
* Sets `not_before` on every peer in the affected /24s with a
  `bev_pool_park` canary_note (event, level, cooldown_until).
* Gracefully stops any running peer in those pools.
* Appends a per-pool entry to `bev_pause_history` (scope='pool',
  ended=None until cooldown / manual clear).
* Bumps `bev_pool_escalation[pool]` so the per-pool escalation
  ladder ticks independently of fleet level.
* **Does NOT set `bev_pause.active=True`** — the fleet-wide pause
  flag is reserved for the fallback fleet-scope path. Peers in
  healthy pools keep working.

Escalation: 1h → 4h → 12h → 24h cooldowns (`BEV_OUTAGE_LEVELS_S`).
A pool that re-triggers within `BEV_OUTAGE_RESET_S` (48 h) of its
last unpause moves up one level. Across the prefix the director
uses the max prev_level so a fresh-co-affected pool inherits the
repeat-offender's escalation for this round.

Manual clear: `POST /api/v1/director/bev_pause/clear` (admin-token
in header). Body `{reason: str}` optional. Records an event with
`end_reason='manual'` in `bev_pause_history`, drops the pool
escalation entry (so next legit trigger starts at L1), busts the
status cache. The dashboard's `BEV PAUSED` chip in the Service
card is wired to the same endpoint (click → confirm → POST).

**Cross-worker visibility gotcha** (fixed 8ce44cd): the director
tick loop must reload `bev_pause` + `bev_pool_escalation` from
disk on every tick. The clear endpoint runs in whichever gunicorn
worker the request lands on; without the disk-merge, the
director-loop worker keeps its stale in-mem `active=True` forever
and the clear has no effect on actual scheduling. Both keys live
in the same disk-merge block as `active_peer`/`mode`/`last_switch`
at the top of `_loop()`.

Warning-classification subtlety (`austria_processor._classify_warning`):
intermediate `bev_retry: ... attempt N/M ... retrying in Ns...`
warnings are filtered out of the `bev` bucket (proxy-lane noise,
not a BEV-outage signal). Only the final `all N attempts exhausted`
line counts. Without this filter every direct-first timeout on a
broken pool emits a wpm of fleet-side `bev` warns, which by itself
is enough to clear the 12 wpm trigger even when the wrapper is
successfully falling back to proxy.

Why egress-pool parking (not per-peer): every VM in an outbound
/24 shares the same NAT gateway and same route to BEV. If TCP
times out for one peer in `109.94.96.0/24`, it'll time out for
every other peer in that /24 too. Per-peer parking would whack-
a-mole — the director would activate an idle peer in the broken
pool, watch it warn for 5 min, park it, activate the next, repeat
— burning Copernicus minutes during each trigger streak. Pool
parking parks the whole /24 in one shot, all `not_before`
expiring together so the pool gets one clean retry per cooldown
level.

#### Director Modes

| Mode | Behaviour |
|------|----------|
| `auto` | Director picks peer with most bandwidth, starts/stops automatically |
| `manual` | Director keeps the manually-activated peer running, no auto-switch |
| `paused` | Director does nothing — all peers stay in current state |

#### Credential revalidation — director-only

`_valid_credentials()` is a **pure cache read**. It never triggers an
OIDC probe. The dashboard hot path (`get_status` running in any
gunicorn worker) and every other reader sees the cached `last_status`
values from `copernicus.list_credentials()` (refreshed on disk via
`_save_credentials_to_disk()`).

The *only* call site permitted to refresh credential health is
`_refresh_credentials_if_due()` invoked from the director loop (once
per tick, gated to `_REVALIDATE_INTERVAL_S = 600s`). It uses a
process-wide `_REVALIDATE_LOCK` for single-flight semantics, so even if
a future caller wires a second invocation in, only one OIDC probe per
process runs at a time.

Why this is strict: in 2026-05-06 we wedged the primary's worker pool
with ~50 OIDC requests/s after a srv restart. The bug was that
`_valid_credentials` had a per-call `if now-last > 600` guard reading
an in-memory timestamp — which non-director workers never updated
from disk, and which 4 racing dashboard threads all passed
simultaneously. Each cache miss fanned out into 8 OIDC token requests
(one per credential), the listen backlog overflowed, and process.html
became unreachable. The architectural fix (this section) is that the
request path *cannot* probe — only the director loop can. To force a
fresh probe (e.g. after credential renewals), call
`POST /api/v1/credentials/validate` (no body).

#### Director link on peer dashboards

`/api/v1/director/status` returns `self_url`, `director_url`, and
`is_director_local`. When the dashboard is open against a non-director
VM, the Peer Director header renders a `⇗ director: <id>` pill linking
to the current director's `process.html`. This works even when the
local `self.json` hasn't been updated by the latest announce: the JS
resolves the id by matching `director_url` against peer URLs in
`d.peers`. On the director itself the pill is hidden (the existing
`★ director` peer-card badge already marks who is in charge).

#### Director High-Availability (`director_ha.py`)

Failover is automatic. Every VM (primary + peers) runs a watchdog
thread that pings the director's `GET /api/v1/director/heartbeat`
every 30 s. The director elects a *shadow* each tick — the most-reliable
peer that is **enabled, reachable, on the same git commit, has ≥ 5 GB
free disk and ≥ 10 GB remaining bandwidth**. Sticky: keeps the current
shadow unless the noise-score gap to the best alternative exceeds 0.3.
The director PUTs a full state snapshot to the shadow every 30 s.

**Snapshot contents** (small JSON, ~200 KB total): `director_state.json`,
`kg_strikes.json`, `failure_counts.json`, `cache_miss_kgs.json`,
`deferred_kgs.json`, `retry_queue.json`, `failed_kgs.json`,
`manifest_tombstones.json`, `copernicus_credentials.json`,
`peers.json`, `cache_manifest.json`, `peer_urls.txt`. Staged under
`data/austria_processor/shadow/`.

**Auto-failover**: shadow misses 6 consecutive heartbeats (3 min) →
promotes itself: installs staged snapshot, writes `is_director`,
restarts director loop in-process (singleton replaced so EMA /
capacity_history reload), broadcasts `POST /api/v1/director/announce`
to every peer. The threshold was previously 3 misses (90 s) but that
triggered spurious takeovers during code pushes, long uploads, and
any tick where the director worker briefly stalled. 3 min still
fails over a genuinely-dead director quickly while tolerating
transient saturation. Peers flip `data/austria_processor/zenodo_lock_url.txt`
and `self.json:director_url`. Old director, if it ever comes back,
finds `stepped_down` flag and refuses to start its director loop —
lives on as a regular peer until manually re-promoted.

**Manual handover**: dashboard `⇋ Hand Over` button (next to `+ Add
Peer`) → `POST /api/v1/director/handover?to=<peer_id>` on the current
director. Director ships fresh snapshot inline to target via
`/api/v1/director/takeover`, target promotes itself, broadcasts
announce. Old director steps down proactively. Reload the dashboard
against the new director's URL afterwards.

**Identity** (`data/austria_processor/self.json`): `{id, url,
director_url}`. On the primary, `director_url=null` (it *is* the
director). Peers learn their identity at registration time (deploy.sh
or `+ Add Peer`); the director also broadcasts identity to all peers
at startup, and self-heals one peer per tick.

**HA endpoints** (all admin-token protected except heartbeat):

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/director/heartbeat` | Liveness probe (public). 200 if director, 410 otherwise. |
| GET\|PUT | `/api/v1/director/snapshot` | Director GETs snapshot, shadow accepts staged PUT. |
| POST | `/api/v1/director/announce` | New director claims authority — peers flip pointer / step down. |
| POST | `/api/v1/director/step_down` | Voluntarily relinquish director role. |
| POST | `/api/v1/director/takeover` | Inbound takeover (manual handover or watchdog promotion). |
| POST | `/api/v1/director/handover?to=<id>` | Initiated by current director; ships state + steps down. |
| GET\|POST | `/api/v1/director/identity` | Read/set self.json. |

**Files**: `is_director` (this VM is director), `stepped_down`
(refused promotion, written by step-down), `self.json` (identity),
`zenodo_lock_url.txt` (peer's pointer to current director),
`shadow/` (staged snapshot), `shadow/meta.json` (origin + shadow_id
stamp — watchdog only takes over if `meta.shadow_id == self_id`).

**Auto-handback to primary**: every director tick, if `self_id != 'primary'`
and the primary is reachable, on the same git commit, enabled and not
scheduled, the director hands the role back via `do_handover('primary',
primary_url)`. The throttle has two scales:
* `HANDBACK_RETRY_S = 60` after a transient failure (primary unreachable,
  stepped_down clear in flight, etc.) so handback fires within ~1–2 min
  of conditions becoming favourable.
* `HANDBACK_BACKOFF_S = 300` after a hard failure (primary commit behind,
  commits unrelated) which needs operator intervention.

`_handback_last_attempt` is advanced on **every** entry past the basic
gates — not just on success. The 2026-05-07 wedge happened because a
prior bug returned early without setting the timestamp on the
`stepped_down: true` path, so the function silently re-fired every
30 s tick for hours, hammering primary's `clear_stepped_down`
endpoint without ever reaching the actual handover. `_handback_last_reason`
in `director_state.json` records why the most recent attempt didn't
hand over.

**Index build deferral on freshly-promoted peers** (`app.py:
_index_build_deferred`): when a non-primary peer becomes director or
shadow it stamps `data/austria_processor/role_promoted_at`. For
`ROLE_INDEX_BUILD_DELAY_S = 1800` (30 min) after promotion the peer
* skips the initial `SearchIndex.build()` (which loads ~8000 KG JSONs
  into memory), and
* skips the JSON download phase of `_sync_peer_data` (manifest merge
  still runs — cheap).

This prevents the index build's CPU+memory spike from landing on top
of director-takeover load. On 2026-05-07 we observed a 3.4 GB worker
on at40 + load avg 6.87 + every non-heartbeat request timing out for
~30 min after promotion. Primary is unaffected (always keeps the
index). If a peer is demoted again the stamp is cleared.

**Identity hardening** (after the 2026-05-03 split-brain incident where
at2/at37/at39/at49 all believed themselves director and self.json files
were cross-corrupted):

* `POST /api/v1/director/identity` refuses to override `id`/`url` from a
  remote address. Only loopback may. Receivers derive identity from
  hostname — the broadcaster cannot lie. Bad inputs are logged at
  WARN with `_rejected={...}` in the response.
* `app.py` startup heals `self.json` from the hostname on every srv
  restart (id and url). The hostname is authoritative.
* `_normalise_local_identity_in_peers` (called during takeover) only
  rewires the *single* entry whose id == snapshot origin (read from
  `shadow/meta.json`). Other peers' URLs are never touched. Earlier
  versions stamped `prev_director_url` onto every `url=None` entry,
  which is how the cluster ended up with multiple peers pointing at
  the same URL after a few cascading handovers.

**Recovery from split-brain**: if multiple peers claim director, write
`is_director` on the primary, restart srv, then broadcast a manual
announce from the primary to every peer:
```python
payload = {'new_director_id': 'primary',
           'new_director_url': 'https://srtm-lidar-at.exe.xyz:8000',
           'reason': 'manual_consolidation'}
# POST to /api/v1/director/announce on every peer’s URL
```
`accept_announce` will step down any competing director and update
pointers on the rest. Survey afterwards via `/api/v1/director/identity`.

**Disaster recovery**: press `⇋ Hand Over` and pick a peer. Or stop
the primary and wait 90 s; the shadow takes over automatically. After
recovery, auto-handback returns the role to primary as soon as primary
is healthy and on the same commit.

#### Role-data eviction (free disk on demoted peers)

The per-KG JSON corpus (`data/austria_processor/json/*.json`, ~1 GB on a
fully-built fleet) and `data/search_index.db` (~120 MB) are only useful
to:
* the **primary** (canonical home for search index + dashboard) — always keep
* the **current director** (peer_director consults the index for cache-ready
  KGs, KG-split lookups)
* the **current shadow** (must be ready to take over)

A peer that *was* director (e.g. after handback to primary) accumulates
the full corpus. Without eviction, that data crowds out expensive
Copernicus tile caches and trips the disk-pressure threshold (1.6 GB
free on at17 was the trigger for adding this).

**Loop** (`app.py:_role_data_eviction_loop`, 10 min tick):
1. Classify role via `_is_keep_role_data()` — reads `is_director`,
   `self.json:id`, and `shadow/meta.json:shadow_id`.
2. If demoted, stamp `data/austria_processor/role_demoted_at`.
3. After `ROLE_EVICT_GRACE_SECONDS` (1 h) of continuous demotion,
   delete `json/*.json` + `search_index.db{,-wal,-shm,-journal}`.
4. Promotion (becoming director or designated shadow) clears the marker.
5. `_sync_peer_data` and `_init_search_index` short-circuit on demoted
   peers so we don't immediately re-fetch what we just freed.

**Primary special-case**: `id == 'primary'` is always keep-role, even
when not director, so the search index has a stable canonical home that
survives any failover.

**Endpoints**:
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/admin/diskstat` | Per-VM disk + role status (sizes_mb, role.demoted_at, role.grace_remaining_s) |
| POST | `/api/v1/admin/role_evict` | Force-run a tick. `{"force": true}` purges regardless of role (decommission). |

**Tuning** (top of `app.py`):
* `ROLE_EVICT_GRACE_SECONDS = 3600`
* `ROLE_EVICT_TICK_SECONDS = 600`

**When NOT to enable**: never delete on uncertainty — if the
keep-role check raises, `_is_keep_role_data()` returns True (fail-safe).

#### Processor uid (must be exedev, never root)

**Background**: the processor is launched via
`sudo systemd-run --scope -p User=exedev ...`. **`-p User=` is silently
ignored on `--scope` units** (only `--service` units honour it). For
months the processor ran as uid=0, with two consequences:
1. `/proc/<pid>/environ` was unreadable from gunicorn (uid=exedev) —
   the dashboard couldn't display per-process credential indices,
   masking director-assigned `COPERNICUS_CRED_INDICES` /
   `KG_LAT_STRIP_FILTER`.
2. `pkill` from gunicorn couldn't kill a root-owned processor; only the
   sudo escalation chain (`sudo systemctl kill --signal=SIGKILL` →
   `sudo pkill -9`) succeeded.

**Fix** (`app.py:start_peer_processor`): keep the scope + `-E env`
forwarding, but invoke the python child via
`runuser --user exedev --preserve-environment --` *inside* the scope.
Result: scope = uid 0 (sudo), runuser = uid 0, python3 = uid 1000.
The sudo and runuser shims are tiny supervisors with no env of their
own; the python child is the only thing whose `/proc/<pid>/environ`
matters, and it's now readable.

Verify with `GET /api/v1/admin/proc_env`. Look for the python3 row —
`uid` should start with `1000`, and `env` should include
`COPERNICUS_CRED_INDICES`, `KG_LAT_STRIP_FILTER`, `ZENODO_LOCK_URL`,
`HOME=/home/exedev`, `USER=exedev`. The sudo and runuser rows show
`env_err: Permission denied` (expected — they're still uid 0 / sudo).

#### Cross-Cutting Concerns (director changes)

**Adding a peer**: Use the dashboard "+ Add Peer" button or API. This updates
`peers.json` on the primary only. The director re-reads it each tick.

**Removing a peer**: Use the ✕ button or DELETE API. Stops the processor,
removes from config and peer_urls.txt.

**Changing bandwidth budget**: Edit `peers.json` directly:
```bash
python3 -c "
import json
cfg = json.load(open('data/austria_processor/peers.json'))
cfg['budget_gb'] = 90  # more conservative
json.dump(cfg, open('data/austria_processor/peers.json', 'w'), indent=2)
"
```
Director picks it up on next tick (re-reads from disk).

**Scheduling a peer** (e.g. don’t use primary until next billing cycle):
```bash
python3 -c "
import json
cfg = json.load(open('data/austria_processor/peers.json'))
for p in cfg['peers']:
    if p['id'] == 'primary':
        p['not_before'] = '2026-05-17'  # skip until bandwidth resets
json.dump(cfg, open('data/austria_processor/peers.json', 'w'), indent=2)
"
```

**Disaster recovery**: If the primary goes down, peers stop receiving work but
don’t crash. To make a peer the new director:
1. Create `data/austria_processor/is_director` on the peer
2. Copy/create `peers.json` with all peer URLs
3. Restart `srv` on the peer

---


---

*See `AGENTS.md` for the project map.*

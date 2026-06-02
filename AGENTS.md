# AGENTS.md — srtm-lidar-at

> **Token discipline**: this file is a navigation hub. Deep docs live in `docs/`.
> Don't load a `docs/*.md` unless your task touches that subsystem.

## ⚡ Fast read for agents: `/process.txt`

**Always start here when investigating cluster state.** Token-cheap,
text-only mirror of `/process.html` — full director / peer / log / Zenodo /
bandwidth snapshot. Per-worker render cache (10 s TTL, X-Cache header) +
director status cache (30 s, cross-worker via disk) so even concurrent
agent polls cost almost nothing on gunicorn. Add to your context with one
curl:

```bash
curl -s https://srtm-lidar-at.exe.xyz:8000/process.txt          # default 60 log lines, last 24h
curl -s 'https://srtm-lidar-at.exe.xyz:8000/process.txt?warn=1' # warnings + errors only
curl -s 'https://srtm-lidar-at.exe.xyz:8000/process.txt?peer=at3&log=200'
curl -s 'https://srtm-lidar-at.exe.xyz:8000/process.txt?q=cred&log=300'
curl -s 'https://srtm-lidar-at.exe.xyz:8000/process.txt?hidden=1' # also list stopped/idle peers
curl -s 'https://srtm-lidar-at.exe.xyz:8000/process.txt?hours=168&log=500&q=Diendorf' # 7d back, archive
```

Query params:
- `log=N` (default 60, max 500) — merged-log line count, newest first
- `warn=1` — restrict log to warnings + errors
- `peer=<substr>` — filter peer roster + log lines by id substring
- `q=<substr>` — substring filter on log message body
- `hidden=1` — also include stopped/idle/complete peers in roster
  (default: hidden; attention-state peers are always shown)
- **`hours=H`** — look back H hours into the merged log. Default 24
  (live ring only). Higher values transparently dip into the
  **long-term archive** at `data/log_archive/YYYY-MM-DD.jsonl.gz`
  (per-UTC-day gzipped JSONL, written when the live ring prunes).
  Use this to mine the full ~200-day forensic record — e.g.
  `?hours=2400&q=cred` for credential-rotation history.

For structured access, pair with `/api/v1/director/status` and
`/api/v1/director/log/history?hours=H` (both documented in
`/api/v1/docs/llm.txt`). The history endpoint also reads from the
per-day archive when `hours` exceeds the live ring — returns the
*most-recent* `limit` matches in range (sets `truncated:true` if more
exist), avoiding the prior bug that truncated the tail of the day.
Director orchestration events — peer auto-updates, Copernicus credential
revalidation/add/remove, frontier credential & cache-cell plan changes,
**bandwidth-wall auto-park / park-until-renewal** events — are emitted via
`app.director_event(…)` and appear inline in the merged 24h log
(`peer=director` for fleet-wide events, otherwise the affected peer id).

### Tile-checkpoint registry fields in `/process.txt`

The `chkpt_registry:` line (when present) summarises the cross-peer tile-
tar registry used by the dd08d74 BEV-cost-recovery mechanism: when a
peer aborts a KG mid-flight (BEV exhaustion / cred rotation / role
eviction) it uploads its tile metadata pickles as a gzipped tarball to
the shared Zenodo cache deposit; the next peer that picks up the KG
pulls them and skips re-doing those tiles. Format:

```
chkpt_registry: kgs=N tiles=T bytes=X.XMB oldest=Yh
```

Fleet visibility piggybacks on the existing `cache_manifest.json` sync
(`chkpt_*` filenames, size=0 = tombstone), so primary sees every peer's
upload/delete within one 5-min sync tick — no new traffic or endpoints.
Every upload / restore / delete also emits an INFO line into the merged
log so the operator can audit eviction fleet-wide:

```bash
curl -s 'https://srtm-lidar-at.exe.xyz:8000/process.txt?hours=24&log=500&q=chkpt'
```

Look for `chkpt: uploaded <kg>` (parent, post-defer), `chkpt: restored N
tile pickle(s) for <kg>` (subprocess, init step), and `chkpt: deleted
<kg>` (parent, post-completion). Warning variants surface on failure.

Local raster sidecars (Phase 1, dd08d74) under `tile_checkpoints/<kg>/
tile_N/raster/*.npy` are *not* part of the cross-peer registry — they're
per-peer-disk only (gated on `SIDECAR_MIN_FREE_GB` free, default 1 GB
since be7d155 — was 4 GB; lowered after at23/at21 lost DTM tiles in
`gpkg_full` when transient direct-egress flaps coincided with missing
sidecars) and let the `gpkg_full` step mmap DTM/DSM/ortho instead of
re-reading BEV. The `gpkg_full` stitcher also has a deferred-retry pass
(also be7d155): tiles that fail their initial DTM read are queued, then
`bev_proxy._refresh_pool()` is forced once and the queue re-painted, so
a 21h `direct`-slot cooldown can't cause a NaN hole. See
`docs/austria-processor.md` for the full lifecycle.

### Surgical eviction of a single KG's checkpoints

Two steps — stop the peer first so it can't re-upload while you delete:

```bash
TOKEN=$(cat data/admin_token)
# 1) stop the peer that holds the (possibly tainted) checkpoints
curl -s -X POST -H "X-Admin-Token: $TOKEN" \
  "http://localhost:8000/api/v1/processing/stop?peer_id=at23"
# 2) clear local pickles + raster sidecars on that peer
curl -s -X POST -H "X-Admin-Token: $TOKEN" \
  "https://srtm-lidar-at23.exe.xyz:8000/api/v1/admin/clear_tile_checkpoints?kg=62133"
# 3) drop the cross-peer Zenodo bundle + tombstone cache_manifest
python3 -c 'import tile_checkpoint_registry as r; print(r.delete_kg("62133"))'
```

Step 3 writes a size=0 entry into `cache_manifest.json` so every peer
drops its mirror view on the next 5-min sync tick; safe to run from
the primary even if no local registry entry exists.

### Bandwidth fields in `/process.txt`

The text dashboard surfaces all the bandwidth telemetry needed to debug a
fleet that's canary-by-default (every peer sampled, slowdown auto-park
on throughput collapse, park-until-renewal on near/over budget):

* **Top-line `fleet_bw`** — `used_gb`, `budget_nominal_gb`, peers,
  parked, soonest `next_renew_in_days`, plus `wall~Xgb(min=Y,n=Z)`
  distilled from peers that auto-parked on throughput collapse.
* **Per-peer columns** — `bw%` (used/effective_budget), `used/bud`
  absolute GB, plus a free-form `bw_extras` tail with optional tokens:
  - `r=0.42` — canary recent/baseline throughput ratio (red <0.30)
  - `CANARY` — has an explicit `budget_gb` override
  - `cap=80G` — peer's persisted `observed_cap_gb` (it has hit a wall)
  - `parked→3.1d` / `parked→4.5h` — active `not_before` cooldown
  - `rd=15` — effective renew_day (override > **observed BW drop** >
    stored > first_seen day-of-month). Learned values are persisted
    as `renew_day_learned` in peers.json.
* **`fleet_bw` JSON block** in `/api/v1/director/status` for structured
  consumers — fields above plus `observed_cap_gb_min/median`,
  `peers_enabled`, `peers_parked`.
* **`bw_learn` line** (below `fleet_bw:`) — empirical billing-cycle
  learning: `peers_with_history`, `learned`, `by_renew_day=[d1=50 …]`
  histogram, plus the 5 most recent observed drops
  (`peerX@dN(prev→newGB)`). Structured at
  `/api/v1/director/status.bw_learn`. Detection: cumulative
  `used_bytes` only decreases at a cycle renewal, so a one-step diff
  in `_sample_canary_history` is enough; we persist just `bw_cycle =
  {peer: {last:[ts,used], events:[…]}}` (~few KB fleet-wide, **not** a
  30-day-of-samples ring — same persist-only-the-shape discipline as
  `fleet_proxy_history`). Throttled hourly per peer + hourly on the
  loop worker; `_bw_learn_summary` aggregate TTL-cached 5 min.

**Critical invariant — eligibility is evidence-only.** Every gate
that asks "does this peer have bandwidth headroom?" goes through
`_peer_bw_depleted(peer, bw, cfg)`. A peer is depleted iff it has an
`observed_cap_gb` AND `used_gb` is within headroom of that cap. Peers
without a measured cap are NEVER disqualified by eligibility — the
canary slowdown detector + `_enforce_peer_bandwidth_walls` will park
them when their real wall hits. Do **not** re-introduce
`_peer_budget_bytes(p,cfg) - used < 2 GB`-style gates: the nominal
budget (95/200/250 GB) is a guess and a per-peer vnstat cycle that
doesn't match `renew_day` (e.g. fleet rolled in on day 1, vnstat
still aggregating from day 17) will strand the peer for weeks. See
the Jun 2026 incident where 50 peers vanished from `cache_only_eligible`.

Thresholds (peer_director.py):
* `BANDWIDTH_LOW_WATER_GB = 4` — mid-KG triggers graceful stop +
  park-until-renewal so the in-flight KG finishes & uploads.
* `BANDWIDTH_HARD_DEPLETED_GB = 1` — hard stop, even mid-KG.
* `CANARY_SLOWDOWN_RATIO = 0.30`, baseline 30 min, 500 MB minimum bytes
  before a peer can be parked for throughput collapse ("soft" park).
* `CANARY_BASELINE_NETWORK_MBPS = 5.0` and `CANARY_RECENT_PARKED_MBPS
  = 0.5` — a soft park is upgraded to a **quality observation**
  (sets `observed_cap_gb` + counts toward fleet wall) only when the
  pre-collapse baseline was network-grade AND the peer truly stalled.
  Otherwise the peer was probably just Zenodo-upload-bound.
* `CANARY_QUALITY_PERSIST_S = 15 min` — the slowdown must be
  continuously observed for 15 min (streak resets above ratio 0.60)
  before counting as quality. Defends against transient network
  blips and short upstream outages.
* `FLEET_CONCURRENT_SLOWDOWN_FRAC = 0.30` — if ≥0.30 of canary-eligible
  peers are simultaneously in slowdown, treat as a fleet-wide upstream
  event (BEV/Zenodo/internet hiccup); soft park still fires defensively
  but no quality observation is recorded for any peer that tick.
  Surfaces in `/process.txt` as `slowdown N/M [FLEET-WIDE]`.
* `FLEET_WALL_MIN_QUALITY_OBS = 5` — fleet wall (`observed_cap_gb_min/median`)
  is suppressed in `fleet_bw` until ≥5 distinct quality obs across
  the fleet. Until then `process.txt` shows `wall=? (gathering, N
  quality obs)` instead of inventing a confident wall.
* `CANARY_PARK_COOLDOWN_S = 6h` — quality-grade parks (real shaping).
* `CANARY_PARK_COOLDOWN_SOFT_S = 1h` — soft parks (defensive, not
  network-grade or not persistent). Almost always upstream blips
  (Zenodo / openEO / transient internet); the short cooldown lets the
  peer rejoin quickly. If the issue is real, the peer will soft-park
  again next tick and eventually graduate to quality-grade.
* Park-until-renewal cooldown = peer's effective `renew_day` next
  occurrence (no need to track per-peer budgets separately; the existing
  `_peer_is_scheduled` gate makes the scheduler skip the peer).

### Throttle / director-efficiency sparkline

The Service-card sparkline (`/process.html` → `renderCapacityHistory`)
plots the rolling `capacity_history` ring (240 ticks ≈ 2h). Each tick
carries:

* `f`   — EMA-smoothed capacity factor in `[THROTTLE_MIN_FACTOR, 1.0]`
* `bev` / `zen` / `cop` — fleet warnings/min per upstream
* `stl` — **fleet CPU-steal median** across *running* peers (same
  population the cache-only ramp brake uses)
* `cpu` — derived `cpu_factor` ramp brake: `1.0` below 30% steal,
  gentle linear ramp, floor `0.55` at 60%+ steal. Mirrors the damping
  curve in `_max_cache_only_peers` so the chart shows *why* the
  director may be running fewer cache-only peers than the warning-rate
  ceiling alone would allow.

Low capacity_factor with steal high → hypervisor pool congested
(reducing peers won't recover cycles; LPT partition does the real
balancing). Low capacity_factor with steal low → a real upstream
is pushing back — check `B / Z / C` chips for which one.

History persists to `director_state.json` every tick so it survives
HA handover and gunicorn's two-worker swap (just like the BEV / Zen /
Cop sub-EMAs). Schema is a 9-tuple (`t,f,bev,zen,cop,stl,cpu,fk,pk`);
the load path tolerates legacy 7-tuples (no `fk`/`pk`) and the
v1-marker 5-tuples written by pre-2026-05-19 directors. `fk`/`pk`
are **cumulative** KG-failure / partial-completion counters read
from `failed_kgs.json` and `partial_kgs.json` (both already
replicated to shadow via `director_ha.SNAPSHOT_FILES`), so the rate
per hour is derived as `(newest - oldest)/window` over any
sub-window of the 2 h ring.

`/process.txt` carries a `throttle:` block (window / cap_factor /
steal_med / cpu_factor / per-upstream warns-per-min, min/med/max over
the window) so forensic mining over the long-term archive can
correlate director efficiency with steal trends without parsing JSON.
Line 2 of the throttle block adds a `kg_outcome:` field with
cumulative + per-hour failed/partial KG rates so a future BEV
outage's downstream impact is visible inline. The Service-card
sparkline shows the same data: `F x.xx/h` / `P x.xx/h` chips and
red/yellow vertical tick marks on the chart at every tick where the
cumulative counter increased.

### Role-based parks (who is parked, and why)

Only five reasons a peer is parked (`not_before` in the future):

1. **Primary** — `_enforce_primary_park`, every tick. Floor
   `not_before=2027-01-01`, `pinned_role=idle`. Manual reset won't
   survive (next tick re-extends). Primary hosts the public
   dashboard and director state; it must never process.
2. **Active director + shadow** — `_enforce_director_self_park`,
   every tick. **Rolling 2 h cooldown**, refreshed while the role
   is held. As soon as the peer stops being director/shadow, the
   stamp expires within 2 h and the peer rejoins rotation — no
   explicit release needed. (Primary is exempt; covered by #1.)
3. **Canary-evidenced BW wall** — `_park_peer_until_renewal`,
   only when the peer has an `observed_cap_gb` set by a quality-
   grade canary slowdown AND used_gb has reached that cap. We do
   NOT park on the nominal 95 GB budget; exe.dev's real limits and
   billing anchors are unknown. Soft canary auto-parks use a short
   1 h cooldown (`auto_park` event); quality parks 6 h.
4. **Rolling-steal park** — `_check_steal_health`, every tick.
   30 min cooldown (`steal_park` event) when the per-peer
   `peer_steal_ema` (~15 min half-life) sits ≥ `STEAL_PARK_THRESHOLD_PCT`
   (70 %%) for ≥ `STEAL_PARK_PERSIST_S` (10 min). Streak resets when
   the EMA falls below `STEAL_PARK_RECOVERY_PCT` (50 %%). Goal: stop
   burning Copernicus credits / KG wall-time on peers stuck on a
   structurally over-subscribed hypervisor host. exe.dev sometimes
   re-places idle VMs onto quieter hosts; either way, 30 min off
   is cheap. Frontier scheduling additionally biases against
   high-steal peers via `FRONTIER_HIGH_STEAL_BIAS_PCT` (50 %%) so
   credentials always land on the lowest-steal eligible peer.
5. **BEV egress-pool park** — `_check_bev_outage` (since 2026-05-28,
   commit 8ce44cd). When fleet BEV warns are sustained AND the
   warning peers cluster into ≤ `OUTAGE_POOL_MAX_SET` (default 3)
   /24 egress pools whose combined share ≥ `OUTAGE_POOL_DOMINANCE`
   (70 %%), the director parks every peer in those /24s with
   escalating 1h/4h/12h/24h cooldowns (`bev_pool_park` event,
   level tracked per pool in `bev_pool_escalation`). Peers in
   healthy pools keep working — no fleet-wide pause. Root cause is
   per-egress reachability: AWS/cloud pools like `109.94.96.0/24`
   and `162.43.189.0/24` can have TCP timeouts to `data.bev.gv.at`
   while other pools (and the primary itself) reach it in <1s.
   Fleet-scope pause is the fallback when attribution fails (>3
   bad pools or `known_frac < OUTAGE_POOL_KNOWN_FRAC = 0.60`).
   **Canary unpark probes**: every 5 min the director polls one
   peer in each parked /24 via its `/api/v1/bev_probe` endpoint
   (single range request to `_BEV_TEST_URL` + TIFF magic check).
   On success it clears `not_before` on every `bev_pool_park`-tagged
   peer in that /24 early so a 20-min route flap doesn't waste the
   full cooldown. Event: `bev_pool_unpark` in `canary_notes`.
   See `docs/peer-director.md → BEV outage handling` for details.

The one-shot `_release_unverified_bw_parks` rescues peers that got
parked-until-renewal *without* an `observed_cap_gb` (legacy budget-
guess parks). It looks for our own `park_until_renewal` note tag,
so primary's 2027 stamp and canary `auto_park` cooldowns are
preserved.

**Operator note**: if you see an unexpected long `parked→Xd` on a
peer, check `peers.json` → `canary_notes` for the most recent
`event`. `role_park`/`park_until_renewal` are director-written;
`auto_park` is canary-written; anything else is manual.

### BEV proxy pool (`bev_proxy.py`)

The processor reads BEV TIFFs via `raster_io` → `bev_retry.open()`,
which rotates through `bev_proxy.next_proxy()`. The pool is
**direct slots + free HTTP proxies** — a defensive fallback for when
BEV rate-limits a peer's IP. Direct egress is the primary path; proxies
buy us per-peer resilience when individual IPs get shaped (different
failure mode from a `/24`-wide BEV outage, which is handled by
`_check_bev_outage` pool-parking instead).

**No paid proxies.** Total throughput is ~5 GB/h per peer × ~50 peers,
far beyond any affordable paid plan. We live entirely off free
aggregator lists.

**Tiered source funnel** (2026-05-29, commit bda1a39):
* `_PROXY_SOURCES_VERIFIED` (1 entry: `elliottophellia/yakumo`) — a
  pre-checked, continuously-validated list. Live phase-1 yield ~22 %
  (vs ~0.04 % for raw aggregators). We validate **every** entry from
  these each refresh, no sampling.
* `_PROXY_SOURCES` (~30 raw aggregators, ~160k unique entries) —
  random-sampled to fill the remaining slots up to a 5000-entry
  validation budget per refresh tick.
* `_fetch_candidates()` returns `(verified, raw)`. `raw -= verified`
  keeps the two disjoint so budget isn't wasted re-testing the same
  entries.

**Two-phase validation** (`_refresh_pool`, every 30 min in a background
thread per process):
1. **Phase 1**: HTTPS CONNECT to `httpbin.org/ip` (fast, ~3s/proxy,
   ~80 workers). Filters out dead / non-HTTPS proxies cheaply.
2. **Phase 2**: range-request a real BEV BigTIFF and verify the TIFF
   magic bytes. Two random tiles per proxy (± layer / dataset / N×E)
   picked via `_random_bev_test_urls()` from the 55-tile × 2-layer ×
   3-dataset grid (330 distinct URLs). Spreads load so one BEV CDN
   object isn't hit 10 000× per refresh.

Proxies that pass go into `_pool` interleaved with `DIRECT_WEIGHT=3`
direct slots. `report_failure()` puts proxies on exponentially-
increasing cooldown (`BASE_COOLDOWN=300s` → cap `3d`); `report_success()`
decays fail_score by 0.5. State persists to `data/proxy_history.json`
across restarts.

**Steady-state pool size** (sampled on primary 2026-06-01, refresh
took 100 s):
```
verified: 1377 candidates, raw: 160 107 candidates
phase-2 survivors: 19 proxies + 3 direct slots = 22
```
Fleet-side: peers don't currently NEED proxies (BEV direct egress is
working), and the merged 48h log has **zero** `bev_proxy` warnings
across all peers. The pool exists as insurance.

### Fleet proxy view in `/process.txt`

Each peer ships a slim `proxy_pool` summary (`bev_proxy.summary()`, ~200 B)
on every `/api/v1/director/peer_status` push (full **and** slim heartbeat),
so the director can render an aggregate without any extra fanout traffic.
Only peers whose processor is actively `running` / `processing` are counted
(stopped peers carry stale pool state). Surfaces as a single line:

```
fleet_proxy: peers=N proxies(min/med/max)=X/Y/Z total=T \
             phase2_med=P refresh_age_med=Am [max=Bm] \
             [no_proxies=K] [stale=S] \
             hist[Hh n=M med_min=A med_med=B med_max=C peers_med=P]
```

Fields:
- `peers` — running peers reporting a `proxy_pool`
- `proxies(min/med/max)` — across the fleet's pools right now
- `total` — sum of healthy + cooling proxies across all peers
- `phase2_med` — median of each peer's last-refresh phase-2 survivor count
  (proxies that passed both HTTPS-CONNECT and BEV-magic validation)
- `refresh_age_med` / `max` — minutes since peers last refreshed their
  pool. `max` is only printed when > 60 min (well past the 30 min
  `REFRESH_INTERVAL`); a high value means at least one peer's refresh
  thread has hung.
- `no_proxies` — running peers operating with **zero** validated proxies
  (direct slots only). Fine while BEV direct egress is healthy; load-bearing
  signal when BEV starts shaping per-IP.
- `stale` — peers whose last refresh is > 90 min old (3× `REFRESH_INTERVAL`).
- `hist[...]` — 24h sliding-window strip from `fleet_proxy_history`
  (sampled every 10 min on the director-loop worker, persisted to
  `director_state.json` so the **other** gunicorn worker and HA shadow
  see it). 144 entries max. Lets a forensic check tell `"this is normal"`
  from `"the fleet's proxy pool has been collapsing for hours."`

Structured view in `/api/v1/director/status` under `fleet_proxy`
(same fields) + `fleet_proxy_history` (the ring as a list of compact
entries: `{t,n,tot,mn,md,mx,no,st}`).

**Cross-worker correctness**: only the director-loop worker (lock holder)
appends to the in-memory ring; the other worker reads it from disk via
`load_director_state` and surfaces the same history.

**Debug recipes**:
```bash
# Live pool state on primary's gunicorn worker:
curl -s http://localhost:8000/api/v1/info | jq .proxy_pool

# Force a refresh from a python shell (takes ~100s):
python3 -c 'import bev_proxy;bev_proxy._refresh_pool();print(bev_proxy.status())'

# Inspect a peer's processor-side bev_proxy log (via admin proxy):
curl -s 'http://localhost:8000/api/v1/director/proxy/log?peer_id=at87&lines=500' \
  | jq -r '.lines[]' | grep -E 'bev_proxy.*(Fetched|Pool refreshed|Phase|verified)'

# Validate a single proxy against BEV manually:
python3 -c 'import bev_proxy;print(bev_proxy._validate_proxy("1.2.3.4:8080"))'
```

**When to add more sources**: if `Fetched` lines show < 100k unique
candidates for several days, or if peers start logging `"No proxies
passed BEV validation"` AND we're seeing concurrent per-peer BEV
throttling (look for clustered `bev_pool_park` events in director
log), check whether yakumo-style continuous validators have replaced
the one we use. Live-test new candidates with `curl --proxy http://...
--connect-timeout 3 https://httpbin.org/ip` against 50 random samples
before adding — phase-1 yield must be ≥10 % to count as "verified";
else add to the raw funnel.

**Critical invariant**: the proxy validator must NEVER hammer a
single BEV CDN object. The randomised tile selection in
`_random_bev_test_urls()` is load-bearing — reverting to a fixed
two-URL test would mean 10 000 range-requests against the same files
per 30 min per peer, which BEV would absolutely rate-limit and which
would defeat the whole point of having proxies.

## TL;DR

Flask + Leaflet app that segments Austrian landscape from BEV LiDAR + BEV ortho
+ Sentinel-2 + ESA WorldCover + Sentinel-1 + Cadastre. A background processor
(`austria_processor`) runs all ~8440 KGs and uploads to Zenodo. A peer director
on the primary VM orchestrates processing across multiple exe.dev VMs.

- **Live**: https://srtm-lidar-at.exe.xyz:8000/
- **Stack**: Python 3.12 / Flask / gunicorn / Leaflet
- **Dashboard**: `/process.html`  •  **Query UI**: `/query.html`  •  **API docs**: `/api/v1/docs/llm.txt`
- **Cadastre API**: https://cadastre-process-api.exe.xyz/api/v1/docs/llm.txt

## Deep-dive index (load on demand)

| Topic | File | When to read |
|---|---|---|
| `app.py` mental model | `docs/app.md` | touching `app.py` — section map, async task lifecycle, background threads, auth |
| Austria Processor mental model | `docs/austria-processor.md` | touching `austria_processor.py`, KG pipeline, tile checkpoints, GPKG/JSON builders, Zenodo upload |
| Peer Director (multi-VM orchestration) | `docs/peer-director.md` | touching `peer_director.py`, `deploy.sh`, parallel frontiers, HA, throttle, admin token, role eviction |
| Copernicus throttle & retry | `docs/copernicus-throttle.md` | touching `copernicus.py`, `tile_cache.py`, 402 handling, credential rotation |
| Search index | `docs/search-index.md` | touching `search_index.py`, schema, compound query, `kg_parcels`, auto-classification |
| RF training | `docs/rf-training.md` | touching `train_rf_4000kg.py`, ground-truth filters, retraining triggers |
| Zenodo persistent tile cache | `docs/zenodo-cache.md` | touching `zenodo_cache.py`, tile manifest, ZIP indices |
| Cross-cutting concerns | `docs/cross-cutting-concerns.md` | adding object types, RF features, tile grid, credential pool, navigation cheatsheet |
| Planned refactor + speed optimisation | `docs/planned-refactor.md` | next maintenance window work |
| Reference algorithms summary | `docs/reference_algorithms_summary.md` | segmentation/RF internals |

## Quick ops

```bash
sudo systemctl restart srv                # web app (gunicorn + director thread)
sudo systemctl restart rf_train           # RF training job
sudo systemctl kill -s SIGKILL austria_processor && sleep 2 && sudo systemctl start austria_processor
journalctl -u srv -f
tail -f data/austria_processor/logs/processor.log
tail -f /tmp/rf_train_4000kg.log
```

## Services

| Unit | Role | Notes |
|---|---|---|
| `srv.service` | gunicorn (2w × 4t, :8000) + director thread | MemoryMax=3G, on-failure |
| `austria_processor.service` | KG pipeline | **Disabled** on primary (director manages it) and on peers (director starts via API) |
| `rf_train.service` | background RF training (4000 KGs) | on-failure |

## File layout

### Core
| File | ~Lines | Purpose |
|---|---:|---|
| `app.py` | 5900 | Flask API, async tasks, progress, director API, share storage |
| `austria_processor.py` | 5100 | KG pipeline (parent + subprocess). See `docs/austria-processor.md` |
| `peer_director.py` | 770 | Multi-VM orchestration. See `docs/peer-director.md` |
| `object_segmentation.py` | 2200 | Felzenszwalb+RAG → per-object classify (44 features) |
| `learned_classifier.py` | 560 | RF classifier (`FEATURE_KEYS`, cadastre-trained) |
| `static/index.html` | 3100 | Single-file Leaflet UI |
| `static/process.html` | 2100 | Processor + director dashboard. Peer Director list uses unified compact strip (`.peer-card` + `.pm-card`) on desktop+mobile: donut · id (color-coded for attention) · role-tagged bar (FRONTIER/CACHE/PRIMARY/STOPPED/INTERRUPTED…) with KG inside · ⋯ menu · ▸ chevron expands legacy detail. `primary` aliased to `at1`. Sort: elapsed-on-current-KG (oldest first), running peers only. Live Log has range chip cycling `live`/`4h`/`24h` (default `4h` to keep payload small) → `/api/v1/director/log/history?hours=N`; warning filter re-renders from cache. |
| `static/query.html` | 600 | Query Explorer over `/api/v1/query*`, `/feedback` |
| `static/flag.js` | 620 | Flag widget (text-selection chip → `/api/v1/flags/match`) |

### Search & cross-API
| File | Purpose |
|---|---|
| `search_index.py` | SQLite FTS5 + R-tree over 8440 KGs + `kg_parcels`. Auto-rebuild on new JSONs. |
| `cadastre_bridge.py` | Joins cadastre API with landscape index (compound queries, nature scoring) |
| `parcel_compact.py` | Compact per-parcel layout (`frav`, `top_objs`, `top_trees`) + `classify_parcel` (15-class) |

### Data I/O
| File | Purpose |
|---|---|
| `raster_io.py` | Windowed reads from BEV via `/vsicurl/` |
| `ortho_io.py` | BEV orthophoto (RGBI, 47 operates + DOP fallback) |
| `copernicus.py` | openEO: NDVI / WC / SAR / harmonics (4-cred rotation). See `docs/copernicus-throttle.md` |
| `cadastre.py` | Building footprints + parcels |
| `hansen.py` | Hansen GFC (forest change) |
| `osm_features.py` | OSM via Overpass |
| `bev_retry.py` | Backoff + proxy rotation for `rasterio.open()` |
| `tile_index.py` | 55-tile grid, EPSG 4326 ↔ 3035 |
| `tile_cache.py` | Grid-snapped 0.1° caches, Zenodo fallback |
| `zenodo_cache.py` | Persistent tile cache on Zenodo. See `docs/zenodo-cache.md` |
| `zenodo_client.py` | Zenodo API + `Manifest` |
| `gpkg_streamed.py` | Strip-streamed full-GPKG for >100 Mpx KGs |
| `kg_splitter.py` | Splits KGs >28 tiles into directional blocks |
| `geo_parse.py` | KML/GeoJSON/Shapefile/GPX/WKT |

### Feature extraction & training
`terrain_analysis.py`, `temporal_analysis.py`, `texture_features.py`,
`ndvi_harmonics.py`, `train_rf_4000kg.py`, `calibrate.py`.

### Deprecated (kept for reference)
`landscape_classifier.py`, `object_classifier.py`, `scene_adaptive_classifier_patches.py`.

## API surface (high level)

Canonical machine-readable list: `GET /api/v1/docs/llm.txt`.
Groupings (see endpoint comments in `app.py`, search `# === SECTION:`):

- **Analysis**: `/api/v1/segment` (async), `/elevation`, `/terrain`, `/changes`, `/changes/trees`
- **Async tasks**: `/segment/progress|result|abort`, auto-saved as `auto-<task>` share
- **Overlays/exports**: `/segment/overlay`, `/{dtm,lidar,ortho,cir,hansen}/overlay`, `/export/{geopackage,kml,mbtiles}`, `/{lidar,ortho}/geotiff`
- **Shares**: `/shares`, `/share`, `/share/<id>`, `/share/<id>/rename` (1 GB cap, LRU, `data/shares/`)
- **One-stop**: `/api/v1/onestop?bbox=&format=`
- **Search index**: `/api/v1/query`, `/query/parcels`, `/kg/<code>`, `/parcel/<id>`, `/index/{status,rebuild}`
- **Cadastre bridge**: `/lookup`, `/parcels/batch` (ids / cadastre query / **compound**), `/parcels/landscape`, `/query/nature`, `/parcel/<id>/detail`, `/kg/<code>/profile`, `/cadastre/*` proxies
- **Processor (proxied)**: `/processing/{status,start,stop,pause,resume,single,retry,throttle,peers,peers/status,log,manifest,cache_manifest,cache_misses}`
- **Director (admin-token)**: `/director/{status,mode,activate,stop,peers,peers/add,peers/<id>,throttle,proxy/status,proxy/log,update_peers,heartbeat,snapshot,announce,step_down,takeover,handover,identity}`
- **Admin**: `/admin/{update,restart_processor,disable_autostart,run_backfill,backfill_status,diskstat,role_evict,proc_env}`
- **Credentials**: `/credentials` (POST add, list usage), `/credentials/validate`
- **Misc**: `/layers?bbox=`, `/info`, `/parse-geometry`, `/zenodo/lock` (mutex broker)

**Compound query** (the power query) — landscape-first:
`POST /api/v1/parcels/batch {"compound":{...}, "parcel_filters":{...}}`. 70+
numeric min/max filters across terrain, area, buildings, trees, vegetation,
NDVI harmonics, SAR, temporal change, classification quality. Per-parcel filters
cover vegetation, terrain, cadastre, auto_class. See `app.py` `# === SECTION:`
and `cadastre_bridge.py` for the full set.

## Frontend cheatsheet

- `getPostArgs()` → `{ct, body}` for fetch
- `showResultOnMap(data)` → renders features + legend + overlay
- `restoreShareResult(data)` → restore full state incl. overlays
- `clearEverything()` → reset to blank
- `checkLayerAvailability()` → debounced, hides unavailable layers
- State: `lastResult`, `allFeatureData`, `overlays`, `drawnItems`,
  `currentShareId/Name`, `_activeTaskId/_aborted`, `hiddenTypes`, `selectedTypes`
- `localStorage`: active `taskId` (resume after refresh), `srtm_admin_token`
  (admin-token for dashboard mutations, reset via `srtmResetAdminToken()`)

## Analysis pipeline (one-liner)

1. read DTM+DSM (`raster_io`) → 2. ortho/NDVI/SAR/Hansen/cadastre →
3. fused gradient → 4. Felzenszwalb (scale=150) + RAG merge (0.12) →
5. extract 44 features per segment → 6. RF classify (or rule-based) →
7. group adjacent compatible segments (tree→forest, roof→building) →
8. GeoJSON features. Details in `object_segmentation.py`.

## 25 object types (4 letters = man-made when capital)

```
Vegetation:    tree shrub grass hedge
Water:         water
Buildings:     roof greenhouse solar_panel
Infrastructure: fence wall mast (+ wind_turbine, substation)
Transport:     road path parking bridge
Agricultural:  crop orchard vineyard garden
Terrain:       bare_soil rock
Disturbance:   excavation fill tree_loss construction earthwork
```

Letter mapping for compact per-parcel arrays: see `parcel_compact.py`
(lowercase = natural, uppercase = man-made).

## External data sources

| Source | Resolution | Access |
|---|---|---|
| BEV ALS DTM/DSM | 1m, 3 dates (2022/23/24) | HTTP range on remote GeoTIFF |
| BEV DOP RGBI | 0.2m, 47 operates | HTTP range on remote GeoTIFF |
| Sentinel-2 NDVI / WorldCover / S1 SAR | 10m | openEO (4 CDSE creds) |
| Hansen GFC | 30m | `/vsicurl/` UMD |
| Austrian cadastre | mm | REST API |
| OSM | varies | Overpass |

Caches: `/tmp/copernicus_cache/`, `/tmp/hansen_cache/`. Persistent: Zenodo
depo 19650075 (see `docs/zenodo-cache.md`).

## Where to look (debug fast-path)

```bash
# Section markers
grep -n '# === SECTION' app.py austria_processor.py peer_director.py
# Project-wide section index
grep -rn '# ===' *.py | sed 's/# === SECTION: //' | sed 's/ ===//' | column -t -s:
# Color/type sync (must match across files)
grep -rl 'SEGMENT_COLORS' *.py static/*.html
# RF feature list
grep -A60 'FEATURE_KEYS = \[' learned_classifier.py
# Live processor step
cat data/austria_processor/current_step.json | python3 -m json.tool
```

More in `docs/cross-cutting-concerns.md`.

## Critical invariants (read before editing)

- **Object types & colors** are duplicated across `app.py`, `austria_processor.py`,
  `static/index.html` — must stay in sync. See `docs/cross-cutting-concerns.md`.
- **RF feature order** in `learned_classifier.py:FEATURE_KEYS` must match
  `object_segmentation.extract_object_features()`. Changing count invalidates
  the saved `.joblib`.
- **Only one director** runs at a time. Gated by `data/austria_processor/is_director`.
  Single-flight via `director.lock` (fcntl). HA failover in `director_ha.py`.
- **Only one frontier peer** at a time (Copernicus credential safety). Many
  cache-only peers in parallel are fine.
- **All Zenodo writes** serialise through `/api/v1/zenodo/lock` on the primary.
- **`CredentialRotatedError` / `CreditsExhaustedError` / `IPThrottledError`**
  must NEVER be swallowed by generic `except Exception`. See `docs/copernicus-throttle.md`.
- **Admin token** required for mutating endpoints. Loopback exempt. Lives in
  `data/admin_token` (gitignored, mode 0600). Same on all peers.
- **Process uid**: the processor MUST run as `exedev` (uid 1000), not root.
  See `docs/peer-director.md` (Processor uid section).
- **Don't probe Copernicus credentials from the request path.** Only
  `peer_director._refresh_credentials_if_due()` may. See `docs/copernicus-throttle.md`.

## Persistent state files (all `data/austria_processor/` unless noted)

| File | Purpose |
|---|---|
| `progress.json` | Live processor state (dashboard reads this) |
| `current_step.json` | Subprocess→parent IPC |
| `subprocess_warnings.jsonl` | Warning relay |
| `in_progress_kg.txt` | Crash-recovery marker |
| `tile_checkpoints/<kg>/tile_N.pkl` | Per-tile metadata checkpoints (resume-on-retry; cross-peer via Zenodo chkpt registry — see `docs/austria-processor.md`) |
| `tile_checkpoints/<kg>/tile_N/raster/*.npy` | Per-tile local raster sidecars (DTM/DSM/nDSM + multi-date + ortho RGB+NIR) so `gpkg_full` skips BEV re-reads (Phase 1, dd08d74) |
| `checkpoint_registry.json` | Local view of *which* tile-tars this peer uploaded to Zenodo (only the uploading peer writes it; the fleet-wide view lives in `cache_manifest.json` under `chkpt_*` keys, synced primary↔peers every 5 min) |
| `zenodo_manifest.json` | KG product uploads (schema: `{entries: {key: {depo_id, size, ...}}}` — *not* `{kgs: …}`. `*_error` keys are retry markers; exclude them from totals.) |
| `cache_manifest.json` | Zenodo tile-cache deposit (schema: `{depo_id, files: {fname: {size, tile_count, updated_at, ...}}}`). Doubles as the fleet-wide transport for chkpt registry deltas (`chkpt_*` filenames, size=0 = tombstone). Synced via existing `_sync_peer_data` GET+PUT cycle — no new traffic. |
| `zenodo_zip_index/*.json` | Cached central directories of remote ZIPs |
| `failed_kgs.json` / `retry_queue.json` / `deferred_kgs.json` | Queue mgmt |
| `peers.json` / `director_state.json` / `director.lock` / `is_director` | Director |
| `peer_urls.txt` | Peer list for data-sync thread |
| `self.json` | Identity (id, url, director_url) |
| `copernicus_credentials.json` / `copernicus_credential_usage.json` | Creds + telemetry |
| `copernicus_paused` / `openeo_circuit.json` | Throttle state |
| `admin_token` (in `data/`) | Cluster auth |
| `shadow/` + `shadow/meta.json` | HA: staged state for shadow promotion |
| `zenodo_lock_url.txt` | Peer→broker pointer |
| `kg_strikes.json` / `cache_miss_kgs.json` | Reliability tracking |
| `data/search_index.db` | SQLite FTS5+R-tree (~5 MB, auto-rebuild) |
| `data/combined_log_24h.jsonl` | Live 24h merged log ring (pruned every ~10 min). NOT replicated to shadow — rebuilt from peers' own `recent_log` after takeover via `_combined_log_bootstrap_once`. |
| `data/log_archive/YYYY-MM-DD.jsonl.gz` | Long-term per-day archive (full 200-day run). Replicated to shadow via `PUT /api/v1/director/log_archive` on the shadow loop — today-only every hour, full sweep on shadow change, sha256-cached so steady-state traffic is ~0. |
| `data/shares/` | Share storage (1 GB cap, LRU) |

## Conventions for editing

- **Read the relevant `docs/*.md` first** when touching a subsystem. They
  contain hard-won invariants, failure modes, and recovery procedures that
  aren't obvious from the code.
- **Section markers**: every long file (`app.py`, `austria_processor.py`,
  `peer_director.py`) uses `# === SECTION: name ===`. Add markers when
  introducing new logical groupings.
- **Keep this file short.** New deep content → new file in `docs/` and link
  it here. Aim: AGENTS.md fits in a single screen of indexed content.
- **Restart discipline**: changes to `austria_processor.py` need a processor
  restart at the next KG boundary (or kill it). Changes to `app.py` /
  `peer_director.py` need `sudo systemctl restart srv` (the director thread
  reloads automatically — singleton replaced).
- **Push → restart → rollout ordering**. `_LOCAL_GIT_COMMIT` is frozen
  at `srv` import time and is what the director sends to peers as the
  rollout target (see `versions:` / `(target=...)` line in `/process.txt`).
  Commit *before* restarting srv, then `git push origin main`, then
  `sudo systemctl restart srv`. If you restart first and commit after,
  the director keeps pushing the old commit as target until the next
  restart and the fleet never picks up your change. (`_ensure_origin_synced`
  pushes local main to origin every tick — so a forgotten `git push` is
  recoverable, but a forgotten restart is not.)

# v2 rollout / upgrade (segv2 → the product)

Load when touching `v2_ingest.py`, `kg_v2_store.py`, `kg_docs.py`, `kg_log_harvest.py`,
`v2_verify.py`, the `--v2-upgrade` path in `austria_processor.py`, or the
`_v2_upgrade_fill` logic in `peer_director.py`. Model / training internals: `segv2/README.md`,
`segv2/HANDOVER.md`.

## Products (Zenodo manifest keys)

| key | file | who writes it | notes |
|---|---|---|---|
| `_json` | `<code>.json` | fresh KGs (v1 or v2 processor) | fleet-wide **completion marker** — 44 call sites; never rename. Fresh v2 KGs write it with `version: v2`, compact. |
| `_full_gpkg` | `<code>_full.gpkg` | fresh KGs only | rasters identical in v1/v2 → no `_full_v2`. Upgrade INPUT. |
| `_light_gpkg` | `<code>_light.gpkg` | fresh v1 KGs | |
| `_json_v2` | `<code>_v2.json.gz` | upgrades + fresh v2 | `kgjson/2` (`kg_json_v2.py`), lossless, 18–22× smaller. Primary ingests it. |
| `_light_gpkg_v2` | `<code>_light_v2.gpkg` | upgrades + fresh v2 | light leg of the zen_stall triple (accepted as `light`). |

Codes are product codes (parent `19570` or split block `49006-north`).

## Fleet side (peer, no credential, no BEV/openEO)

`austria_processor.py --cache-only --v2-upgrade` (director start payload
`{cache_only:true, v2_upgrade:true}` → app.py appends `--v2-upgrade` and writes the marker
`data/austria_processor/v2_upgrade_mode`). At startup the processor reads `retry_queue.json`
(the whitelist the director PUT) and appends **upgrade units** for codes that pass
`v2_upgrade_eligible`: `_json` + committed `_full_gpkg`, no `_json_v2`, < 2 strikes in
`v2_upgrade_failed.json`. Per unit: `_v2_fetch_inputs` (v1 JSON → bbox + verify baseline; full
GPKG, 4 streams, disk-checked, md5) into `data/austria_processor/v2_source/`, then
`process_one_kg(source_gpkg=…, v1_doc=…)`: `v2_source` shim replaces BEV reads, no gpkg_full,
`_light_v2.gpkg` + `_v2.json.gz`, `v2_verify.verify_before_upload` (37 checks) → on FAIL
`aborted_v2_verify_failed` (files deleted, strike, **no** failed_kgs entry), on PASS ordered
upload light_v2 → json_v2 under the fleet Zenodo lock. GPKG deleted afterwards.
`progress.v2_upgraded` counts; log prefix `v2up:`.

`MODEL_VERSION` defaults to v2 iff `segv2.model_v2.available()` (lightgbm importable + model
present or fetchable from Zenodo depo 22824068/22824064). `admin_update` pip-installs lightgbm
(`_ensure_v2_deps`) after every git pull, so the fleet gains v2 on the normal rollout wave. A peer
without lightgbm logs `--v2-upgrade requested but MODEL_VERSION=v1` and just runs v1 cache-only.

## Director side (`peer_director._orchestrate_cache_only`)

**Priority (2.3)**: `_compute_v2_upgrade_candidates` ranks `(-1 half-landed pair, then
older-v2 re-upgrades, then never-upgraded v1)` while `V2_REUPGRADE_FIRST=True` — the
2.3 top-N change rewrites every v2 JSON, so the live v2 line converges first. Set it
False to go back to never-upgraded-first.

`_v2_upgrade_fill(whitelist, cfg)`: when `len(cache_ready_kgs) < V2_UPGRADE_FILL_BELOW_READY`
(40) append up to `MAX_V2_UPGRADE_PEERS` (24) codes from `_compute_v2_upgrade_candidates()`
(manifest scan, smallest `_full_gpkg` first, 5-min cache, skips codes whose primary ingest struck
out). Zero injection while fleet Zenodo warns > `V2_UPGRADE_ZEN_WARN_MAX` (2/min). The codes skip
block expansion (they are already product codes; the completed-block guard would drop them) and
join the LPT partition. Config keys in `director_config`: `v2_upgrade` (bool, default true),
`v2_upgrade_fill_below_ready`, `max_v2_upgrade_peers`. State: `v2_upgrade_assigned`,
`_v2_cand_cache`. Status: `/api/v1/director/status.v2.dispatch`.

`processing_queue_get` (GET `/processing/queue`, which prunes "complete" codes) keeps
upgrade-eligible codes while the `v2_upgrade_mode` marker exists — otherwise the whitelist the
director just PUT would be pruned before the processor read it.

**Peer-claim check is in-progress-only for upgrade units.** The processor's per-KG
`_get_peer_claimed_kgs` folds every reachable peer's manifest-derived `completed` list into the
claimed set. An upgrade unit is v1-complete *by definition*, so with that check every unit was
skipped as "claimed by peer" and the peer exited seconds after start; the director then re-issued
the same 24 codes every tick (Sep 2026, right after the fleet doubled — the new peers were the
first reachable ones advertising the full completed set). Units tagged `_v2_upgrade` now only
honour live `blocks` / `parents_unsplit` claims. Fresh-KG semantics are unchanged.

## Primary side

* `kg_v2_store.py` — `data/kg_v2_store.db` (WAL). `kg_v2` (verified gz blob per code, `bbox`,
  `generated_at`, `v1_bytes_freed`), `kg_log` (per-code history rings, cap 600 rows), `meta`.
  NOT the search index (that one is rebuildable and DROPs tables); the store is source data.
  Test with `kg_v2_store.STORE_PATH = Path('/tmp/x.db')` **before importing** anything else.
* `kg_docs.py` — the only doc loader: store first, `json/<code>.json` second. Used by the search
  index selector, `/api/v1/kg/<code>`, coverage oracle, quality flags.
* `v2_ingest.py` — daemon thread in `app.py` (`_v2_ingest_thread`, primary only, fcntl
  single-flight across gunicorn workers, 5-min tick, ≤20 codes/tick, `V2_INGEST=0|1` env
  override). Per code: download (+size/md5 check) → `verify_for_ingest` (v1 doc = file or older
  store copy; flat index row fallback) → `store.put` **first** → `search_index.update_kg` →
  delete `json/<code>.json` + `add_freed` → `quality_flags.scan_doc`. Then
  `kg_log_harvest.harvest_live()` + `harvest_archive_step(1)`. Failures: `meta.ingest_failed`
  `{code:{n,reason,uploaded_at}}`, 3 strikes per upload (a newer upload resets).
* Peer-sync (`_sync_peer_data`) skips `_json` downloads for codes in the store.
* `kg_log_harvest.fold_lines` is hooked into `app._archive_lines` — every merged-log line is
  folded into the per-code ring once, when it leaves the live ring.

## Telemetry

**Source of truth for every progress indicator is the search index**
(`search_index.progress_summary(manifest)`): one row per *parent* KG, `product_version` = the
manifest version tag (`v1`/`v2`/`v2.1`/`v2.2`) taken as the **minimum across split blocks**,
plus `json_uploaded_at` / `json_v2_uploaded_at` (from the manifest; `sync_manifest_links()` is
run by the app watcher whenever `zenodo_manifest.json` changes). Consequences:
* `progress:` / health banner / `processing/status.manifest_rate_*` / daily sparkline count
  fresh `_json` completions of parents only — never split blocks, never v2.x upgrade
  re-uploads (Sep 2026: `done=3508` was product codes and the `13.9/h` was the upgrade rate).
* `v2:` / `status.v2.upgraded_parents` = parents whose every block is at
  `v21_products.MANIFEST_VERSION`; `stale_total` / `stale_versions` = parents on an older v2.x
  (re-upgrade pending); `upgraded_24h` / `rate_per_h` / `eta_days` see only current-version
  uploads, so a parent that went v2 → v2.1 → v2.2 is counted once. `by_version` = parents,
  `codes_by_version` = product codes (manifest). ETA falls back to manifest `_json` timestamps
  when the index has none (`source: "manifest"`).

`/process.txt` `v2:` line:
```
v2: at_v2.2=N/8440 (p%) +N/24h @r/h eta=Dd parents_by_version[v1=… v2=… v2.1=… v2.2=…] stale=N any_v2=N ·
    fresh_v2=N · verify_fail=N pending=N ·
    store=Ncodes/MB freed=GB avg/kg=KB · json_files=N disk=GB · kg_log=codes/rows/KB
    archive_remaining=Dd · dispatch=N/Tcand (fill<40 ready, cap 24)
```
Structured: `/api/v1/director/status.v2`, `/api/v1/model/v2?rollout=1`. Grep the merged log
for `v2up:` (peers), `v2verify` (gate verdicts), `v2_ingest` (primary).

`/process.html` mirrors this as the **v2 Rollout** card (`renderV2`, fed by
`director/status?slim=1 → .v2`; click the card to filter the Live Log to `v2`). Per-peer:
progress.json carries `model_version` / `v2_upgrade` / `v2_upgraded` and `current_kg.v2_upgrade`
(written in `austria_processor.main()` / `set_current_kg`); `peer_director.get_status` forwards
them as `model_version`, `v2_upgrade_mode`, `v2_upgraded`, `current_kg_v2_upgrade`. The peer strip
shows a `v2` badge (`v2↑` = upgrade mode; yellow `v1` = running peer without lightgbm) and a
`V2-UP` role tag while a peer is on an upgrade unit. The Zenodo badge appends `v2: N json + M light`
(from `/processing/manifest?summary=1 → by_product`).

## Ops

```bash
python3 v2_ingest.py status                 # v2: line + JSON
python3 v2_ingest.py                        # one ingest tick now (primary)
python3 v2_ingest.py 19570                  # dry ingest one code (no delete / index / flags)
python3 -c 'import v2_ingest;print(v2_ingest.failed())'
python3 kg_log_harvest.py 19570             # targeted history fold + last rows
curl -s 'localhost:8000/api/v1/kg/19570?history=1' | jq '.history[-5:]'
```
Re-ingest a code: delete its store row (`kg_v2_store.delete(code)`) — next tick re-downloads.
Reprocess an upgrade: drop `<code>_json_v2` + `<code>_light_gpkg_v2` from the manifest
(the requeue/tombstone path already includes both suffixes).

## Strikes, deferrals, and the v2_regen auto-requeue (2026-09-21)

Peer file `v2_upgrade_failed.json` = `{code: {n, defer, reason, ts}}`; read it with
`GET /api/v1/admin/v2_strikes` (admin token), clear with `POST …/v2_strikes/clear
{"codes":[…]}` or `{"checks":[…]}`. Fleet union at `data/austria_processor/v2_strikes_fleet.json`
(`v2:` line → `strikes_fleet=… struck_out=…`).

* **Verify FAIL** → `n += 1`, struck out at `V2_UPGRADE_MAX_STRIKES = 2`.
* **Deferral** (input download 5xx / timeout / disk pressure, or a 404 while the fleet
  `zenodo_degraded` circuit is tripped) → `defer += 1`, **no strike**; struck out only after
  `V2_UPGRADE_MAX_DEFERS = 8`. (Before: a deferral was a strike, so one Zenodo outage struck
  out every code a peer touched twice.)
* **Stale v1-JSON md5** in the manifest → accepted if the bytes parse as this KG (baseline
  only), `v1 JSON manifest md5 stale` warning.
* **`_full_gpkg` gone** (404 with circuit clear) → fatal strike on the peer + reported as
  `status.v2_gone`. The director (`_check_v2_regen`, every 5 min, `data/austria_processor/
  v2_regen.json`) then verifies it against the deposition API — **never while the Zenodo
  circuit is degraded**, control probe must be 200, **two "gone" verdicts ≥30 min apart** —
  and requeues the KG for a fresh v2.2 run through the canonical
  `POST /processing/queue {skip_processed:false, keep_products:true, gone_keys:[<code>_full_gpkg]}`.
  `keep_products` stamps the `_requeue` tombstone and invalidates *only* the gone key: the v1
  `_json` / `_light_gpkg` (and any `*_v2`) stay referenced and served until the fresh run
  replaces them. `sync_queue_to_peer` forwards the same `gone_keys`, so peers never tombstone
  more than the primary did. Caps: 3 requeues/tick, 20 outstanding. States:
  `candidate → confirming → requeued → done` (or `dismissed` when the file turns out to be
  present). Follow on `/process.txt` (`v2_regen:` line) and `?q=v2regen` (director events).

**Zenodo replace is PUT-then-DELETE** (`zenodo_client._replace_in_bucket`): same filename →
one in-place PUT, different filename → PUT new, then DELETE old. A failed PUT can no longer
leave a deposition empty (how 18127 / 01504 / 19102 lost their full GPKG), and a v2.2
re-upgrade keeps the v2 / v2.1 product live until the new one is committed.

## Invariants

* `_json` stays the completion marker; upgrades never touch `_json`/`_light_gpkg`/`_full_gpkg`.
* `store.put` before `update_kg` before deleting the v1 file (selector reads store first).
* Only the primary ingests / deletes v1 files. A temporary director elsewhere keeps its file
  corpus; nothing is lost.
* `CredentialRotatedError`-class rules from `docs/copernicus-throttle.md` are irrelevant to the
  upgrade path (no upstream fetches) but still apply to fresh v2 KGs.

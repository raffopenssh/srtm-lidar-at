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

`/process.txt` `v2:` line:
```
v2: upgraded=N/8440 (p%) +N/24h @r/h eta=Dd · fresh_v2=N · verify_fail=N pending=N ·
    store=Ncodes/MB freed=GB avg/kg=KB · json_files=N disk=GB · kg_log=codes/rows/KB
    archive_remaining=Dd · dispatch=N/Tcand (fill<40 ready, cap 24)
```
Structured: `/api/v1/director/status.v2`, `/api/v1/model/v2?rollout=1`. Grep the merged log
for `v2up:` (peers), `v2verify` (gate verdicts), `v2_ingest` (primary).

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

## Invariants

* `_json` stays the completion marker; upgrades never touch `_json`/`_light_gpkg`/`_full_gpkg`.
* `store.put` before `update_kg` before deleting the v1 file (selector reads store first).
* Only the primary ingests / deletes v1 files. A temporary director elsewhere keeps its file
  corpus; nothing is lost.
* `CredentialRotatedError`-class rules from `docs/copernicus-throttle.md` are irrelevant to the
  upgrade path (no upstream fetches) but still apply to fresh v2 KGs.

# Zenodo Persistent Cache (`zenodo_cache.py`)

Local Copernicus/Hansen tile caches get evicted when disk approaches 5GB.
The Zenodo cache module persists these tiles on Zenodo so they survive eviction.

**Architecture**: Each local NPZ tile has a `.meta.json` sidecar (written by
`tile_cache._write_tile_meta()`) that records product type and grid coordinates.
The uploader uses these sidecars to group tiles into ZIP archives by product ×
0.5° latitude strip, then uploads to a single Zenodo deposit (depo 19650075).

**Upload flow** (`ZenodoCache.upload_all()`):
1. `_build_reverse_index()` reads `.meta.json` sidecars (+ fallback from `tile_bbox_index.json`)
2. Groups tiles by (product, lat strip)
3. For each group, compares local entry names vs cached remote ZIP central directory
4. If local ⊆ remote → skip. Otherwise, merge local + remote-only into new ZIP and upload.
   If the remote central directory can't be listed for a *non-404* reason (5xx/timeout)
   the ZIP is **deferred**, never rebuilt local-only (that would overwrite and lose every
   remote-only tile).
5. Invalidates cached ZIP index after upload.

## Upload failure semantics (2026-09-20)

`_upload_file` is **PUT-in-place → verify via deposit listing**. There is no
delete-old step: Zenodo's bucket API replaces a same-name object atomically
(probed 2026-09-20 on depo 19650075: `PUT bucket/<existing-name>` → 201, new
`version_id`, `is_head=true`, deposit file count unchanged). A failed PUT
therefore leaves the previous copy — and the manifest entry pointing at it —
intact. The old delete-then-PUT cycle turned every transient 5xx into
permanent cache loss (18 tombstones during the Sep 2026 weekend maintenance,
10 of them Copernicus cells that cost openEO credits to rebuild).

After **any** PUT failure the deposit is re-listed once (`_list_deposit_files`,
one GET) and `_handle_upload_failure` decides:

| listing says | error class | action |
|---|---|---|
| target present, size == what we sent | anything | **success** (504-after-landing) — manifest updated from listing |
| target present, other size (`present_old`) | anything | keep entry untouched, WARN |
| absent / listing failed | 5xx, 429, timeout, SSL, connection | `unverified: true, unverified_since, last_error` — size/url kept, readers still serve it, local tiles kept for retry |
| absent | definitive 4xx (not 408/429), **first** attempt | `unverified` + `upload_4xx_history` |
| absent | definitive 4xx, second attempt ≥10 min after an earlier one | **tombstone** (`prev_size/prev_checksum/prev_tile_count` preserved) |
| any | fleet `zenodo_circuit.degraded` set | never tombstone → `unverified` |

A read-side 404 on the ZIP index (`_confirm_404_then_tombstone`) also re-lists
first: present → invalidate the cached index and defer; absent → tombstone
("…confirmed absent by deposit listing"). The legitimate "deposit is at the
100-file cap → 400 on a *new* name" case still works: there's no live entry to
act on, the failure is logged and retried next flush (the `_live_files >= 90`
guard defers new names anyway). `access_token=` is scrubbed from every stored
reason (`_scrub_secret`).

`CacheManifest.get_file` hides tombstones (`size == 0`) but **returns
`unverified` entries** — a stale-but-existing ZIP beats a cache miss.

## Reconcile manifest ↔ deposit (`reconcile_manifest`)

```bash
python3 zenodo_cache.py reconcile           # dry run, prints diff
python3 zenodo_cache.py reconcile --apply   # takes the fleet Zenodo lease (purpose=reconcile), writes atomically
```

Lists the deposit and, ignoring `chkpt_*` (tile-checkpoint registry) and
non-`.zip` keys:

* live entry absent from deposit → tombstone `reconcile: missing from deposit`
* tombstoned entry present → `_probe_zip_usable` (fresh central directory,
  count entries matching `<product>_<S>_<W>_<N>_<E>[_<year>].npz`, range-read
  the last member's local header: `PK\x03\x04` magic + ends inside the file).
  Usable → `restore` (size/md5 from listing, `tile_count` from the central
  directory, `updated_at=now`, `restored_from_tombstone:{reason,at,tombstoned_at}`,
  `tombstone_reason` dropped). Corrupt → stays tombstoned, reason retagged
  `reconcile: present but corrupt (<why>)`, WARNING.
* `unverified` live entries present → flag cleared, size/md5 adopted; size drift
  fixed likewise.
* deposit files unknown to the manifest → logged only.

Summary persisted to `data/austria_processor/cache_reconcile_last.json`
(`cache_reconcile:` line in `/process.txt`). The director runs it hourly
(`_reconcile_cache_manifest_if_due`) and immediately when the Zenodo-degraded
circuit clears. Restores propagate to peers through the normal 5-min
`cache_manifest` sync because both merge paths are "newest `updated_at` wins"
with no tombstone preference — a restore stamped `now` beats the older
tombstone on every peer.

## Zenodo-degraded circuit (peer side)

`zenodo_degraded()` reads `cache_manifest.json → zenodo_circuit.degraded`
(director-written, rides the manifest sync like `prewarm`; 30 s TTL). While
set, `upload_all` skips *replacement* uploads of ZIPs that already have a live
entry (`zips_deferred_degraded`), still performs first-time uploads, and no
code path tombstones. KG product uploads (`zenodo_client`) are unaffected.

**Download**: On local cache miss, `tile_cache` calls `ZenodoCache.fetch_copernicus()`
or `fetch_hansen()`. Uses 2-3 HTTP range requests to read individual NPZ entries
from remote ZIP files via the cached central directory index.

**When uploads happen**:
- After each completed tile in the child subprocess (throttled to 30 min)
- After KG completion (forced)
- Before disk eviction of expensive tiles (forced)
- Before each KG in the parent process (throttled to 30 min)

**Key invariant**: Every `.npz` tile file MUST have a `.meta.json` sidecar.
Orphan tiles (no sidecar) are invisible to the uploader and waste disk.
`cleanup_orphan_tiles()` runs at processor startup to delete them.

**Manifest files** (don't confuse them):
- `data/austria_processor/cache_manifest.json` — Zenodo cache deposit (tiles)
- `data/austria_processor/zenodo_manifest.json` — KG product uploads (GPKGs, JSONs)

**Cached ZIP indices**: `data/austria_processor/zenodo_zip_index/*.json` — cached
central directories of remote ZIPs, keyed by MD5 of download URL. Invalidated
automatically after each upload. Stale indices cause false "local ⊆ remote" and
skip uploads — delete the directory to force re-fetch.

**Why not BEV/ortho?** BEV DTM/DSM/ortho are already COGs with efficient HTTP range
reads. At 1m resolution, all Austria = ~4TB (infeasible for Zenodo).

```bash
python3 zenodo_cache.py status      # show local + Zenodo tile counts
python3 zenodo_cache.py dry-run     # build ZIPs without uploading
python3 zenodo_cache.py upload       # upload local tiles to Zenodo
```

**Troubleshooting**:
- `Upload complete: 0 ZIPs, 0 tiles` — either no new tiles (normal when same
  Copernicus cells are reused across KGs), or all local tiles already on Zenodo.
  Check `python3 zenodo_cache.py status` for local vs remote counts.
- Tiles not uploading — check `.meta.json` sidecars exist alongside `.npz` files.
  Missing sidecars = orphans. Run `python3 -c "from zenodo_cache import cleanup_orphan_tiles; cleanup_orphan_tiles()"`
- Stale indices — `rm -rf data/austria_processor/zenodo_zip_index/` and re-flush.


---

*See `AGENTS.md` for the project map.*

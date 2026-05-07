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
5. Invalidates cached ZIP index after upload.

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

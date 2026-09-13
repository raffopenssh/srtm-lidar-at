"""Fetch a KG's Zenodo products (full / light GPKG, JSON) to local disk and
audit them.

Zenodo bucket downloads run ~0.6 MB/s single-stream from exe.dev but scale
near-linearly with parallel range requests (8 streams ≈ 4.6 MB/s measured),
so we split every file into N byte ranges.

Audit (appended to ``data/segv2/audit.jsonl``, one record per file):
  * md5 vs manifest checksum, size vs manifest
  * SQLite ``PRAGMA integrity_check`` / ``application_id`` for GPKGs
  * ``gpkg_contents`` inventory (layer names + data types), missing layers
  * per-raster-layer: tile count, matrix size, fill fraction
The audit doubles as the "audit files while we touch them" step for the v2
reprocessing pass.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import zenodo_client  # noqa: E402

log = logging.getLogger("segv2.fetch")

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/austria_processor/zenodo_manifest.json"
AUDIT_LOG = ROOT / "data/segv2/audit.jsonl"
CACHE_DIR = Path(os.environ.get("SEGV2_GPKG_DIR", "/tmp/segv2_gpkg"))
N_STREAMS = 8
CHUNK = 1 << 20

EXPECTED_FULL_LAYERS = {"DTM", "DSM", "nDSM", "segment_type", "segment_height",
                        "segments", "segment_points"}
EXPECTED_LIGHT_LAYERS = {"segments", "segment_points", "parcels", "buildings",
                         "new_buildings", "infrastructure", "segment_type",
                         "segment_height"}

_MANIFEST_CACHE: tuple[float, dict] | None = None


def _manifest() -> dict:
    global _MANIFEST_CACHE
    mt = MANIFEST.stat().st_mtime
    if _MANIFEST_CACHE is None or _MANIFEST_CACHE[0] != mt:
        _MANIFEST_CACHE = (mt, json.load(open(MANIFEST))["entries"])
    return _MANIFEST_CACHE[1]


def manifest_entry(kg: str, product: str) -> dict | None:
    """product ∈ {'full_gpkg','light_gpkg','json'}."""
    return _manifest().get(f"{kg}_{product}")


def product_codes(product: str = "full_gpkg") -> list[str]:
    """All codes (incl. split blocks like '49006-south') with this product."""
    suf = f"_{product}"
    return sorted(k[: -len(suf)] for k in _manifest() if k.endswith(suf))


def _headers():
    return {"Authorization": f"Bearer {zenodo_client.DEFAULT_TOKEN}"}


def download(entry: dict, dest: Path, n_streams: int = N_STREAMS) -> dict:
    """Parallel range download. Returns {seconds, bytes, md5}."""
    url = entry["bucket_url"] + "/" + entry["filename"]
    size = int(entry["size"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        f.truncate(size)
    chunk = (size + n_streams - 1) // n_streams
    t0 = time.time()

    def _get(i):
        a, b = i * chunk, min(size, (i + 1) * chunk) - 1
        if a > b:
            return 0
        for attempt in range(12):   # Zenodo 504 storms last 10-30 min; back off up to 5 min/try
            try:
                r = requests.get(url, headers={**_headers(), "Range": f"bytes={a}-{b}"},
                                 timeout=900, stream=True)
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                n = 0
                with open(tmp, "r+b") as f:
                    f.seek(a)
                    for c in r.iter_content(CHUNK):
                        f.write(c)
                        n += len(c)
                if n != b - a + 1:
                    raise RuntimeError(f"short range {n} != {b-a+1}")
                return n
            except Exception as e:  # noqa: BLE001
                log.warning("range %d attempt %d failed: %s", i, attempt, e)
                time.sleep(min(300, 5 * 2 ** attempt))
        raise RuntimeError(f"range {i} failed")

    with cf.ThreadPoolExecutor(n_streams) as ex:
        total = sum(ex.map(_get, range(n_streams)))
    dt = time.time() - t0
    h = hashlib.md5()
    with open(tmp, "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    os.replace(tmp, dest)
    log.info("downloaded %s: %.0f MB in %.0fs (%.1f MB/s)", dest.name, total / 1e6, dt,
             total / 1e6 / max(dt, 1e-6))
    return {"seconds": round(dt, 1), "bytes": total, "md5": h.hexdigest()}


def audit_gpkg(path: Path, kind: str) -> dict:
    rec: dict = {}
    try:
        c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rec["integrity"] = c.execute("PRAGMA integrity_check").fetchone()[0]
        rec["app_id_ok"] = c.execute("PRAGMA application_id").fetchone()[0] == 0x47504B47
        contents = c.execute("SELECT table_name, data_type FROM gpkg_contents").fetchall()
        rec["layers"] = {n: t for n, t in contents}
        expected = EXPECTED_FULL_LAYERS if kind == "full_gpkg" else EXPECTED_LIGHT_LAYERS
        rec["missing_layers"] = sorted(expected - set(rec["layers"]))
        rasters = {}
        for name, dt in contents:
            if dt in ("tiles", "2d-gridded-coverage"):
                try:
                    n_tiles = c.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                    z = c.execute("SELECT MAX(zoom_level) FROM gpkg_tile_matrix WHERE table_name=?",
                                  (name,)).fetchone()[0]
                    mw, mh = c.execute("SELECT matrix_width, matrix_height FROM gpkg_tile_matrix "
                                       "WHERE table_name=? AND zoom_level=?", (name, z)).fetchone()
                    rasters[name] = {"tiles": n_tiles, "grid": [mw, mh],
                                     "fill": round(n_tiles / max(mw * mh, 1), 3)}
                except Exception as e:  # noqa: BLE001
                    rasters[name] = {"error": str(e)}
        rec["rasters"] = rasters
        for key, tbl in (("n_segments", "segments"), ("n_points", "segment_points")):
            try:
                rec[key] = c.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
            except Exception:  # noqa: BLE001
                rec[key] = None
        try:
            rec["styled_layers"] = c.execute(
                "SELECT COUNT(DISTINCT f_table_name) FROM layer_styles").fetchone()[0]
        except Exception:  # noqa: BLE001
            rec["styled_layers"] = None
        c.close()
    except Exception as e:  # noqa: BLE001
        rec["error"] = str(e)
    return rec


def _append_audit(rec: dict):
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def fetch(kg_code: str, product: str = "full_gpkg", *, force: bool = False,
          audit: bool = True) -> Path | None:
    """Download + audit one product. Returns local path or None if not in manifest
    or download failed."""
    e = manifest_entry(kg_code, product)
    if not e:
        return None
    dest = CACHE_DIR / e["filename"]
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kg": kg_code,
           "product": product, "depo_id": e.get("depo_id"), "manifest_size": e["size"],
           "manifest_md5": (e.get("checksum") or "").replace("md5:", ""),
           "version": e.get("version", "v1")}
    if dest.exists() and dest.stat().st_size == e["size"] and not force:
        rec["cached"] = True
    else:
        try:
            rec.update(download(e, dest))
        except Exception as ex:  # noqa: BLE001
            rec["error"] = f"download: {ex}"
            _append_audit(rec)
            return None
        rec["md5_ok"] = rec["md5"] == rec["manifest_md5"]
        rec["size_ok"] = rec["bytes"] == e["size"]
    if audit:
        if product.endswith("gpkg"):
            rec["gpkg"] = audit_gpkg(dest, product)
            rec["ok"] = bool(rec["gpkg"].get("integrity") == "ok"
                             and not rec["gpkg"].get("missing_layers")
                             and rec.get("md5_ok", True))
        else:
            rec["ok"] = rec.get("md5_ok", True)
        _append_audit(rec)
        if not rec["ok"]:
            log.warning("AUDIT FAIL %s %s: md5_ok=%s integrity=%s missing=%s", kg_code, product,
                        rec.get("md5_ok"), rec.get("gpkg", {}).get("integrity"),
                        rec.get("gpkg", {}).get("missing_layers"))
    return dest


def release(kg_code: str, product: str = "full_gpkg"):
    e = manifest_entry(kg_code, product)
    if e:
        (CACHE_DIR / e["filename"]).unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for code in sys.argv[1:]:
        print(code, fetch(code))

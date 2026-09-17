"""Parcel-outline elevation profiles (v2 product).

For every cadastre parcel we keep DTM (and DSM) sampled *along the parcel
boundary* — the legal outline is where terraces, retaining walls, hedges,
fences, road crowns and building edges sit, so a boundary profile is the
cheapest signal for later terrain / building / boundary-structure work
without touching the rasters again.

Sampling (deterministic, so the points can be re-derived from the parcel
polygon alone):

* exterior ring only (holes are other parcels), start at the ring's first
  vertex, walk the ring at constant ``spacing`` metres; the spacing is chosen
  per parcel so that ``n = ceil(perimeter / spacing)`` lands in
  ``[MIN_PTS, MAX_PTS]`` (default 8–120; base spacing 5 m).  Every sample is
  the DTM/DSM value at the pixel under the point (1 m raster, no interpolation).

Encoding (shared by the light-GPKG table and the KG JSON):

* ``z``   int16 decimetres DTM, little-endian, base64 in JSON / BLOB in GPKG
* ``zs``  int16 decimetres DSM−DTM (nDSM, clipped 0..3276 m) — optional
* ``n``   number of points, ``sp`` spacing (m, 2 decimals), ``p0`` first
  point (E, N EPSG:3035, 1 decimal) as a sanity anchor, ``miss`` count of
  points that fell on nodata (encoded as ``NODATA = -32768``).

Budget: 2 B/pt/layer → a 60-pt parcel is 240 B raw (~330 B base64 for both
layers) — ≈ 0.3 MB per 1000-parcel KG in the JSON, well inside the v2 size
budget (README “v2 product design notes”).  xy is *not* stored: re-derive with
:func:`outline_points` from the cadastre polygon + ``sp``.

API:

    prof = ParcelOutlineProfiler(parcels)          # [(parcel_id, geom3035)]
    prof.sample(dtm, dsm, transform)               # once per tile (any order)
    rec = prof.records()                           # {parcel_id: dict}  (JSON-ready)
    rows = prof.gpkg_rows()                        # for the `parcel_outline_z` table
    z_m, ndsm_m = decode(rec)                       # float arrays (NaN for nodata)
    pts = outline_points(geom, rec['sp'], rec['n']) # (n, 2) EPSG:3035
"""
from __future__ import annotations

import base64
import math
from typing import Iterable

import numpy as np

BASE_SPACING_M = 5.0
MIN_PTS = 8
MAX_PTS = 120
NODATA = -32768
GPKG_TABLE = "parcel_outline_z"
GPKG_SCHEMA = (
    f"CREATE TABLE IF NOT EXISTS {GPKG_TABLE} ("
    "parcel_id TEXT PRIMARY KEY, n INTEGER, spacing_m REAL, e0 REAL, n0 REAL, "
    "miss INTEGER, z_dtm_dm BLOB, ndsm_dm BLOB)"
)


def choose_spacing(perimeter_m: float, base: float = BASE_SPACING_M) -> tuple[float, int]:
    """(spacing, n) so that n ∈ [MIN_PTS, MAX_PTS] and spacing ≥ base where possible."""
    if perimeter_m <= 0:
        return base, 0
    n = int(math.ceil(perimeter_m / base))
    if n > MAX_PTS:
        n = MAX_PTS
    elif n < MIN_PTS:
        n = MIN_PTS
    sp = round(perimeter_m / n, 2)
    return max(sp, 0.01), n


def outline_points(geom, spacing: float | None = None, n: int | None = None) -> np.ndarray:
    """(n, 2) EPSG:3035 sample points along the exterior ring, deterministic.

    Multi-polygons use the largest part.  With ``spacing``/``n`` omitted the
    per-parcel defaults from :func:`choose_spacing` apply (that is what the
    encoder used, so decoders may pass the stored ``sp``/``n`` or nothing)."""
    if geom is None or geom.is_empty:
        return np.zeros((0, 2))
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon":
        return np.zeros((0, 2))
    ring = geom.exterior
    per = ring.length
    if spacing is None or n is None:
        spacing, n = choose_spacing(per)
    if n <= 0:
        return np.zeros((0, 2))
    d = np.arange(n, dtype=np.float64) * spacing
    d = d[d < per] if per > 0 else d[:1]
    pts = [ring.interpolate(float(x)) for x in d]
    out = np.array([(p.x, p.y) for p in pts], dtype=np.float64)
    if len(out) < n:  # rounding of spacing left us short — pad with the last point
        out = np.vstack([out, np.repeat(out[-1:], n - len(out), 0)])
    return out


def _encode(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<i2").tobytes()).decode("ascii")


def _decode(s: str | bytes | None) -> np.ndarray | None:
    if s is None:
        return None
    raw = s if isinstance(s, (bytes, bytearray)) else base64.b64decode(s)
    return np.frombuffer(raw, dtype="<i2").astype(np.int32)


def decode(rec: dict) -> tuple[np.ndarray, np.ndarray | None]:
    """JSON/GPKG record → (dtm_m float32 [NaN=nodata], ndsm_m or None)."""
    z = _decode(rec.get("z") if "z" in rec else rec.get("z_dtm_dm"))
    zs = _decode(rec.get("zs") if "zs" in rec else rec.get("ndsm_dm"))
    zf = np.where(z == NODATA, np.nan, z / 10.0).astype(np.float32)
    sf = None if zs is None else np.where(zs == NODATA, np.nan, zs / 10.0).astype(np.float32)
    return zf, sf


class ParcelOutlineProfiler:
    """Accumulates outline samples across tiles; parcels may straddle tiles."""

    def __init__(self, parcels: Iterable[tuple[str, object]], with_ndsm: bool = True):
        self.ids: list[str] = []
        self.sp: list[float] = []
        self.pts: list[np.ndarray] = []
        self.z: list[np.ndarray] = []
        self.zs: list[np.ndarray] | None = [] if with_ndsm else None
        for pid, geom in parcels:
            if geom is None or geom.is_empty:
                continue
            g = max(geom.geoms, key=lambda x: x.area) if geom.geom_type == "MultiPolygon" else geom
            if g.geom_type != "Polygon":
                continue
            sp, n = choose_spacing(g.exterior.length)
            p = outline_points(g, sp, n)
            if len(p) == 0:
                continue
            self.ids.append(str(pid)); self.sp.append(sp); self.pts.append(p)
            self.z.append(np.full(len(p), NODATA, np.int16))
            if self.zs is not None:
                self.zs.append(np.full(len(p), NODATA, np.int16))
        self._all = np.vstack(self.pts) if self.pts else np.zeros((0, 2))
        self._off = np.cumsum([0] + [len(p) for p in self.pts])

    def __len__(self):
        return len(self.ids)

    def sample(self, dtm: np.ndarray, dsm: np.ndarray | None, transform) -> int:
        """Fill samples that fall inside this tile. Returns #points filled."""
        if len(self._all) == 0:
            return 0
        h, w = dtm.shape
        col = np.floor((self._all[:, 0] - transform.c) / transform.a).astype(np.int64)
        row = np.floor((self._all[:, 1] - transform.f) / transform.e).astype(np.int64)
        inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        if not inside.any():
            return 0
        idx = np.flatnonzero(inside)
        zv = dtm[row[idx], col[idx]].astype(np.float64)
        ok = np.isfinite(zv)
        idx = idx[ok]; zv = zv[ok]
        zdm = np.clip(np.round(zv * 10), -32767, 32767).astype(np.int16)
        nd = None
        if dsm is not None and self.zs is not None:
            dv = dsm[row[idx], col[idx]].astype(np.float64)
            nd = np.clip(dv - zv, 0, 3276.7)
            nd = np.where(np.isfinite(nd), np.round(nd * 10), NODATA).astype(np.int16)
        # scatter back per parcel
        par = np.searchsorted(self._off, idx, side="right") - 1
        for k in np.unique(par):
            sel = par == k
            loc = idx[sel] - self._off[k]
            # only overwrite nodata so an earlier tile's value wins on overlaps
            cur = self.z[k]
            fill = cur[loc] == NODATA
            cur[loc[fill]] = zdm[sel][fill]
            if nd is not None:
                self.zs[k][loc[fill]] = nd[sel][fill]
        return int(len(idx))

    # --- outputs --------------------------------------------------------------------
    def record(self, k: int) -> dict:
        z = self.z[k]
        rec = {"n": int(len(z)), "sp": self.sp[k],
               "p0": [round(float(self.pts[k][0, 0]), 1), round(float(self.pts[k][0, 1]), 1)],
               "miss": int((z == NODATA).sum()), "z": _encode(z)}
        if self.zs is not None:
            rec["zs"] = _encode(self.zs[k])
        return rec

    def records(self, skip_empty: bool = True) -> dict[str, dict]:
        """{parcel_id: record} — JSON-ready (`outline_z` per parcel)."""
        out = {}
        for k, pid in enumerate(self.ids):
            if skip_empty and (self.z[k] == NODATA).all():
                continue
            out[pid] = self.record(k)
        return out

    def gpkg_rows(self, skip_empty: bool = True) -> list[tuple]:
        """Rows for ``GPKG_SCHEMA`` (blobs raw int16 LE, not base64)."""
        rows = []
        for k, pid in enumerate(self.ids):
            z = self.z[k]
            if skip_empty and (z == NODATA).all():
                continue
            rows.append((pid, int(len(z)), self.sp[k], round(float(self.pts[k][0, 0]), 2),
                         round(float(self.pts[k][0, 1]), 2), int((z == NODATA).sum()),
                         np.ascontiguousarray(z, dtype="<i2").tobytes(),
                         None if self.zs is None else np.ascontiguousarray(self.zs[k], dtype="<i2").tobytes()))
        return rows

    def summary(self) -> dict:
        n_pts = int(sum(len(z) for z in self.z))
        miss = int(sum(int((z == NODATA).sum()) for z in self.z))
        return {"parcels": len(self.ids), "points": n_pts, "missing": miss,
                "bytes_raw": n_pts * (4 if self.zs is not None else 2)}


def write_gpkg_table(gpkg_path, rows: list[tuple]) -> int:
    """Create/replace the attribute-only ``parcel_outline_z`` table in a GPKG and
    register it in ``gpkg_contents`` (data_type 'attributes')."""
    import sqlite3
    c = sqlite3.connect(str(gpkg_path))
    try:
        c.execute(GPKG_SCHEMA)
        c.execute(f"DELETE FROM {GPKG_TABLE}")
        c.executemany(f"INSERT INTO {GPKG_TABLE} VALUES (?,?,?,?,?,?,?,?)", rows)
        try:
            c.execute("INSERT OR REPLACE INTO gpkg_contents (table_name, data_type, identifier, description) "
                      "VALUES (?, 'attributes', ?, ?)",
                      (GPKG_TABLE, GPKG_TABLE,
                       "DTM (dm) and nDSM (dm) sampled along each parcel's exterior ring at spacing_m "
                       "starting at ring vertex 0 (segv2.parcel_elevation.outline_points); -32768 = nodata"))
        except Exception:  # noqa: BLE001  — no gpkg_contents (plain sqlite) is fine
            pass
        c.commit()
        return len(rows)
    finally:
        c.close()

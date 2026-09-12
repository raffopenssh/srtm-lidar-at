"""Read the 1 m raster layers of one of our *full* GPKGs into numpy — the
input side of the v2 feature pipeline (replaces BEV/openEO/UMD reads).

Handles the quirks of how ``austria_processor.build_full_gpkg_tiled`` wrote
them:

* float layers (DTM/DSM/nDSM/DTM_YYYY/DSM_YYYY/NDVI/SAR_*) are
  ``2d-gridded-coverage`` with nodata -9999 → NaN.
* uint8 layers (Ortho_YYYY 3/4-band, CIR_YYYY, WorldCover, Hansen_*) are PNG
  ``tiles``; read via rasterio.
* ``segment_type`` is a *palette* PNG — GDAL expands it to RGBA, so the type
  code must be read back from the raw PNG palette indices (sqlite tile blobs).

All windows are returned on the full-GPKG grid (EPSG:3035, 1 m) so layers
line up pixel-for-pixel.
"""
from __future__ import annotations

import io
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

NODATA = -9999.0


class FullGpkg:
    def __init__(self, path: str | Path):
        self.path = str(path)
        c = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.layers = {n: t for n, t in c.execute("SELECT table_name, data_type FROM gpkg_contents")}
        c.close()
        with rasterio.open(f"GPKG:{self.path}:DTM") as ds:
            self.transform = ds.transform
            self.shape = (ds.height, ds.width)
            self.bounds = ds.bounds
            self.crs = ds.crs

    # ---- inventory helpers -------------------------------------------------
    @property
    def raster_layers(self) -> list[str]:
        return [n for n, t in self.layers.items() if t in ("tiles", "2d-gridded-coverage")]

    def years(self, prefix: str) -> list[int]:
        rx = re.compile(rf"^{prefix}_(\d{{4}})$")
        return sorted(int(m.group(1)) for n in self.layers for m in [rx.match(n)] if m)

    # ---- readers -----------------------------------------------------------
    def read(self, layer: str, window: Window | None = None, bands=None) -> np.ndarray | None:
        if layer not in self.layers:
            return None
        with rasterio.open(f"GPKG:{self.path}:{layer}") as ds:
            arr = ds.read(bands, window=window) if bands else ds.read(window=window)
            if ds.dtypes[0].startswith("float"):
                arr = arr.astype(np.float32)
                arr[arr <= NODATA + 1] = np.nan
        return arr[0] if arr.shape[0] == 1 else arr

    def read_ortho(self, year: int | None = None, window: Window | None = None):
        """(rgb (3,h,w) uint8, nir (h,w) uint8 | None, year). Newest year by default."""
        ys = self.years("Ortho")
        if not ys:
            return None, None, None
        y = year if year in ys else ys[-1]
        arr = self.read(f"Ortho_{y}", window)
        if arr is None:
            return None, None, None
        if arr.ndim == 2:
            arr = arr[None]
        rgb = arr[:3]
        nir = arr[3] if arr.shape[0] >= 4 else None
        if nir is None and f"CIR_{y}" in self.layers:
            cir = self.read(f"CIR_{y}", window)
            nir = cir[0] if cir is not None and cir.ndim == 3 else None
        return rgb, nir, y

    def read_segment_type(self, window: Window | None = None) -> np.ndarray:
        """v1 segment_type codes (uint8).

        Interior tiles are palette PNGs (index == type code). Edge tiles were
        written RGBA by GDAL; those are mapped back to codes through the
        SEGMENT_COLORS table (exact RGBA match, alpha ignored)."""
        from PIL import Image
        h, w = self.shape
        out = np.zeros((h, w), np.uint8)
        c = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        lut = _color_to_code_lut()
        try:
            z = c.execute("SELECT MAX(zoom_level) FROM gpkg_tile_matrix WHERE table_name='segment_type'").fetchone()[0]
            tw, th = c.execute("SELECT tile_width, tile_height FROM gpkg_tile_matrix "
                               "WHERE table_name='segment_type' AND zoom_level=?", (z,)).fetchone()
            for col, row, blob in c.execute("SELECT tile_column, tile_row, tile_data FROM segment_type "
                                            "WHERE zoom_level=?", (z,)):
                im = Image.open(io.BytesIO(blob))
                if im.mode == "P":
                    a = np.array(im, dtype=np.uint8)
                else:
                    rgba = np.array(im.convert("RGB"), dtype=np.uint32)
                    key = (rgba[..., 0] << 16) | (rgba[..., 1] << 8) | rgba[..., 2]
                    a = np.zeros(key.shape, np.uint8)
                    for k, code in lut.items():
                        a[key == k] = code
                r0, c0 = row * th, col * tw
                hh, ww = min(th, h - r0), min(tw, w - c0)
                if hh > 0 and ww > 0:
                    out[r0:r0 + hh, c0:c0 + ww] = a[:hh, :ww]
        finally:
            c.close()
        if window is not None:
            r0, c0 = int(window.row_off), int(window.col_off)
            out = out[r0:r0 + int(window.height), c0:c0 + int(window.width)]
        return out

    # ---- tiling ------------------------------------------------------------
    def windows(self, tile_px: int = 1500, overlap_px: int = 0):
        h, w = self.shape
        step = tile_px - overlap_px
        for r in range(0, h, step):
            for c in range(0, w, step):
                yield Window(c, r, min(tile_px, w - c), min(tile_px, h - r))

    def window_transform(self, window: Window):
        return rasterio.windows.transform(window, self.transform)

    def window_bounds(self, window: Window):
        return rasterio.windows.bounds(window, self.transform)


@lru_cache(maxsize=1)
def _color_to_code_lut() -> dict[int, int]:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from austria_processor import SEGMENT_COLORS
    from object_segmentation import OBJECT_TYPES
    lut = {}
    for name, rgba in SEGMENT_COLORS.items():
        code = OBJECT_TYPES.get(name)
        if code is not None:
            r, g, b = rgba[:3]
            lut[(r << 16) | (g << 8) | b] = code
    return lut


@lru_cache(maxsize=4)
def open_full(path: str) -> FullGpkg:
    return FullGpkg(path)

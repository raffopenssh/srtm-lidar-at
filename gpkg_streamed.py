"""Strip-streamed full-GPKG builder for large KGs.

When a KG's raster grid exceeds _STREAM_PIXEL_THRESHOLD pixels, the normal
in-memory stitching would require many GB of RAM.  This module writes each
raster layer in horizontal strips so peak memory stays bounded (~500 MB per
layer regardless of KG size).

Used by austria_processor.build_full_gpkg_tiled() — not imported elsewhere.
"""
import logging
import os

import numpy as np

log = logging.getLogger("austria_processor")

# Switch to streamed writing above this pixel count (full_h * full_w).
# 100 Mpx ≈ 10 km × 10 km.  Below this the in-memory path is fine.
_STREAM_PIXEL_THRESHOLD = 100_000_000

# Skip full-array boundary merge above this (saves ~20 bytes/px).  At 120 Mpx
# the categorical arrays alone are ~1.5 GB and adding ndsm merge (+2.3 GB)
# would push a 7 GB cgroup over the edge.
_CATEGORICAL_PIXEL_LIMIT = 120_000_000

# Above this pixel count, skip the full GPKG entirely.  The full GPKG
# contains DTM/DSM/ortho raster layers that require streaming re-reads of
# remote BEV data — at 200+ Mpx that means 200+ remote tile reads per layer
# and hours of wall time even without memory issues.  The light GPKG + JSON
# summary are still produced and carry all the analytical value.
_MAX_FULL_GPKG_PIXELS = 200_000_000


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _tile_offsets(tr, full_left, full_top, full_h, full_w, res=1.0):
    """Return (row_off, col_off, r_end, c_end, th_eff, tw_eff) or None."""
    th, tw = tr["shape"]
    tile_left = tr["bounds_3035"][0]
    tile_top = tr["bounds_3035"][3]
    col_off = int(round((tile_left - full_left) / res))
    row_off = int(round((full_top - tile_top) / res))
    r_end = min(row_off + th, full_h)
    c_end = min(col_off + tw, full_w)
    th_eff = r_end - row_off
    tw_eff = c_end - col_off
    if th_eff <= 0 or tw_eff <= 0:
        return None
    return row_off, col_off, r_end, c_end, th_eff, tw_eff


def _feather_weight(rows, cols, margin=100):
    """2-D feather weight: 1.0 at centre, tapers to 0.0 at edges."""
    wy = np.ones(rows, dtype=np.float32)
    wx = np.ones(cols, dtype=np.float32)
    m = min(margin, rows // 2, cols // 2)
    if m > 0:
        ramp = np.linspace(0.0, 1.0, m, endpoint=False, dtype=np.float32)
        wy[:m] = np.minimum(wy[:m], ramp)
        wy[-m:] = np.minimum(wy[-m:], ramp[::-1])
        wx[:m] = np.minimum(wx[:m], ramp)
        wx[-m:] = np.minimum(wx[-m:], ramp[::-1])
    return wy[:, None] * wx[None, :]


def _precompute_tile_info(tile_seg_results, full_left, full_top, full_h, full_w, res=1.0):
    """Pre-compute offsets for all tiles.  Returns list of (idx, tr, offsets)."""
    info = []
    for ti_idx, tr in enumerate(tile_seg_results):
        off = _tile_offsets(tr, full_left, full_top, full_h, full_w, res)
        if off is not None:
            info.append((ti_idx, tr, off))
    return info


# ---------------------------------------------------------------------------
# Generic strip-streamed layer writer
# ---------------------------------------------------------------------------

def _streamed_write_layer(
    out_path, table_count,
    tile_info,          # from _precompute_tile_info
    full_h, full_w, full_tf,
    layer_name, band_descs,
    fetch_fn,           # callable(tr, th_eff, tw_eff) → list[ndarray] | None
    dtype='float32',
    blend='feather',    # 'feather' | 'categorical' | 'overwrite'
):
    """Write one GPKG raster layer in horizontal strips.

    *fetch_fn(tr, th_eff, tw_eff)* must return a list of 2-D arrays (one per
    band, shape ``(th_eff, tw_eff)``), or ``None`` to skip the tile.

    Returns the new *table_count*.
    """
    import rasterio
    import rasterio.windows

    _NODATA = -9999.0
    n_bands = len(band_descs)

    # Adaptive strip height — target ~400 MB accumulator.
    if blend == 'feather':
        bpp = n_bands * 8 + 8          # f64 sums + f64 weight
    elif blend == 'categorical':
        bpp = n_bands * 4 + 4          # data + best_weight
    else:
        bpp = n_bands * 4
    strip_h = max(256, min(full_h, int(400_000_000 / max(full_w * bpp, 1))))

    opts = dict(
        driver='GPKG', width=full_w, height=full_h, count=n_bands,
        dtype=dtype, crs='EPSG:3035', transform=full_tf,
        RASTER_TABLE=layer_name, RASTER_IDENTIFIER=layer_name,
    )
    if dtype == 'float32':
        opts['nodata'] = _NODATA
    if table_count > 0:
        opts['APPEND_SUBDATASET'] = 'YES'

    with rasterio.open(out_path, 'w', **opts) as dst:
        for b_idx, desc in enumerate(band_descs, 1):
            dst.set_band_description(b_idx, desc)

        for strip_start in range(0, full_h, strip_h):
            strip_end = min(strip_start + strip_h, full_h)
            sh = strip_end - strip_start

            # Allocate strip accumulators
            if blend == 'feather':
                accum = np.zeros((n_bands, sh, full_w), dtype=np.float64)
                wsum = np.zeros((sh, full_w), dtype=np.float64)
            elif blend == 'categorical':
                if dtype == 'uint8':
                    accum = np.zeros((n_bands, sh, full_w), dtype=np.uint8)
                else:
                    accum = np.full((n_bands, sh, full_w), np.nan, dtype=np.float32)
                best_w = np.zeros((sh, full_w), dtype=np.float32)
            else:
                if dtype == 'uint8':
                    accum = np.zeros((n_bands, sh, full_w), dtype=np.uint8)
                else:
                    accum = np.full((n_bands, sh, full_w), np.nan, dtype=np.float32)

            # Accumulate tiles that intersect this strip
            for _ti_idx, tr, (row_off, col_off, r_end, c_end, th_eff, tw_eff) in tile_info:
                t_row_start = max(row_off, strip_start)
                t_row_end = min(r_end, strip_end)
                if t_row_start >= t_row_end:
                    continue

                tile_r0 = t_row_start - row_off
                tile_r1 = t_row_end - row_off
                strip_r0 = t_row_start - strip_start
                strip_r1 = t_row_end - strip_start

                bands = fetch_fn(tr, th_eff, tw_eff)
                if bands is None:
                    continue

                if blend == 'feather':
                    fw = _feather_weight(th_eff, tw_eff, margin=100)
                    fw_strip = fw[tile_r0:tile_r1, :]
                    for b in range(n_bands):
                        tile_sl = bands[b][tile_r0:tile_r1, :tw_eff]
                        if dtype == 'float32':
                            valid = ~np.isnan(tile_sl)
                        else:
                            valid = np.ones(tile_sl.shape, dtype=bool)
                        w = np.where(valid, fw_strip, 0.0).astype(np.float64)
                        accum[b, strip_r0:strip_r1, col_off:col_off + tw_eff] += \
                            np.where(valid, tile_sl, 0.0).astype(np.float64) * w
                    # Shared weight from first band
                    b0 = bands[0][tile_r0:tile_r1, :tw_eff]
                    v0 = ~np.isnan(b0) if dtype == 'float32' else np.ones(b0.shape, dtype=bool)
                    w0 = np.where(v0, fw_strip, 0.0).astype(np.float64)
                    wsum[strip_r0:strip_r1, col_off:col_off + tw_eff] += w0

                elif blend == 'categorical':
                    fw = _feather_weight(th_eff, tw_eff, margin=100)
                    fw_strip = fw[tile_r0:tile_r1, :]
                    # Validity gate (same rationale as Phase-A categorical
                    # stitch / build_full_gpkg_tiled): a high-feather tile's
                    # empty reprojected corner must not win the overlap seam
                    # and block a valid lower-feather tile. Validity = band-0
                    # has data (uint8 != 0 / non-NaN for float).
                    b0 = bands[0][tile_r0:tile_r1, :tw_eff]
                    valid = (b0 != 0) if dtype == 'uint8' else ~np.isnan(b0)
                    _bw = best_w[strip_r0:strip_r1, col_off:col_off + tw_eff]
                    wins = valid & (fw_strip > _bw)
                    for b in range(n_bands):
                        tile_sl = bands[b][tile_r0:tile_r1, :tw_eff]
                        accum[b, strip_r0:strip_r1, col_off:col_off + tw_eff] = np.where(
                            wins, tile_sl,
                            accum[b, strip_r0:strip_r1, col_off:col_off + tw_eff])
                    best_w[strip_r0:strip_r1, col_off:col_off + tw_eff] = np.where(
                        valid, np.maximum(_bw, fw_strip), _bw)

                else:  # overwrite
                    for b in range(n_bands):
                        tile_sl = bands[b][tile_r0:tile_r1, :tw_eff]
                        if dtype == 'uint8':
                            mask = tile_sl != 0
                        else:
                            mask = ~np.isnan(tile_sl)
                        accum[b, strip_r0:strip_r1, col_off:col_off + tw_eff] = np.where(
                            mask, tile_sl,
                            accum[b, strip_r0:strip_r1, col_off:col_off + tw_eff])

            # Normalise feather
            if blend == 'feather':
                has = wsum > 0
                for b in range(n_bands):
                    if dtype == 'float32':
                        accum[b] = np.where(has, accum[b] / np.where(has, wsum, 1.0), np.nan)
                    else:
                        accum[b] = np.where(
                            has,
                            np.clip(accum[b] / np.where(has, wsum, 1.0), 0, 255),
                            0)

            # Write to GPKG
            window = rasterio.windows.Window(0, strip_start, full_w, sh)
            for b in range(n_bands):
                out = accum[b].astype(np.float32 if dtype == 'float32' else dtype)
                if dtype == 'float32':
                    out = np.where(np.isnan(out), _NODATA, out)
                dst.write(out, b + 1, window=window)

    return table_count + 1


# ---------------------------------------------------------------------------
# Main streamed builder
# ---------------------------------------------------------------------------

def build_full_gpkg_streamed(
    kg_code, tile_seg_results, all_objects, obs_year,
    mark_uncertain, out_path, full_h, full_w, full_left, full_top,
    full_right, full_bottom, full_tf, res,
    # These are needed for segment vectors / boundary merge:
    _write_segment_vectors, _write_segment_points, _write_gpkg_all_styles,
    _fix_gpkg_raster_crs, _merge_boundary_segments,
    SEGMENT_COLORS,
):
    """Build the full GPKG using strip-streamed raster writing.

    Called by build_full_gpkg_tiled() when pixel count exceeds threshold.
    Returns out_path.
    """
    import rasterio
    import rasterio.transform

    table_count = 0
    _GPKG_NODATA = -9999.0
    obj_map = {o.obj_id: o for o in all_objects}
    tile_info = _precompute_tile_info(
        tile_seg_results, full_left, full_top, full_h, full_w, res)
    n_pixels = full_h * full_w

    log.info("  STREAMED GPKG: %d x %d px (%.0f Mpx), %d tiles, strip mode",
             full_w, full_h, n_pixels / 1e6, len(tile_seg_results))

    # ===================================================================
    # Phase A — Categorical layers (labels / seg_type / seg_height)
    # Need full arrays for boundary merge + vectorisation.
    # For monster KGs (>350 Mpx) we skip boundary merge to stay in RAM.
    # ===================================================================
    skip_boundary_merge = (n_pixels > _CATEGORICAL_PIXEL_LIMIT)
    if skip_boundary_merge:
        log.info("  STREAMED GPKG: %d Mpx exceeds categorical limit — "
                 "skipping boundary merge", n_pixels // 1_000_000)

    labels_full = np.zeros((full_h, full_w), dtype=np.int32)
    seg_type_full = np.zeros((full_h, full_w), dtype=np.uint8)
    seg_height_full = np.full((full_h, full_w), np.nan, dtype=np.float32)
    best_cat_weight = np.zeros((full_h, full_w), dtype=np.float32)

    for _ti_idx, tr, (row_off, col_off, r_end, c_end, th_eff, tw_eff) in tile_info:
        if tr.get("labels") is None:
            continue
        labels_tile = tr["labels"][:th_eff, :tw_eff].astype(np.int32)
        type_tile = np.zeros((th_eff, tw_eff), dtype=np.uint8)
        for uid in np.unique(labels_tile):
            if uid == 0:
                continue
            obj = obj_map.get(int(uid))
            if obj:
                type_tile[labels_tile == uid] = obj.type_code

        fw = _feather_weight(th_eff, tw_eff, margin=100)
        # Only a pixel this tile actually segmented (label > 0) may win or raise
        # the best-weight bar — see austria_processor.build_full_gpkg_tiled for
        # the rotated-quad NaN-corner rationale (triangle-sliver fix).
        tile_valid = labels_tile > 0
        _bw = best_cat_weight[row_off:r_end, col_off:c_end]
        wins = tile_valid & (fw > _bw)
        labels_full[row_off:r_end, col_off:c_end] = np.where(
            wins, labels_tile, labels_full[row_off:r_end, col_off:c_end])
        seg_type_full[row_off:r_end, col_off:c_end] = np.where(
            wins, type_tile, seg_type_full[row_off:r_end, col_off:c_end])

        if tr.get("ndsm") is not None:
            tile_sh = tr["ndsm"][:th_eff, :tw_eff].astype(np.float32)
            seg_height_full[row_off:r_end, col_off:c_end] = np.where(
                wins, tile_sh, seg_height_full[row_off:r_end, col_off:c_end])

        best_cat_weight[row_off:r_end, col_off:c_end] = np.where(
            tile_valid, np.maximum(_bw, fw), _bw)

    del best_cat_weight

    # Boundary merge (in-place on labels_full / seg_type_full)
    _label_remap = {}
    if not skip_boundary_merge:
        try:
            # Need a blended nDSM for the merge — stream-stitch just the nDSM
            # into a temporary array.  This is 8 bytes/px extra but brief.
            ndsm_full = np.full((full_h, full_w), np.nan, dtype=np.float32)
            ndsm_sum = np.zeros((full_h, full_w), dtype=np.float64)
            ndsm_w = np.zeros((full_h, full_w), dtype=np.float64)
            for _ti_idx, tr, (row_off, col_off, r_end, c_end, th_eff, tw_eff) in tile_info:
                try:
                    tdata = _read_dtm_for_tile(tr)
                    tile_ndsm = tdata["ndsm"][:th_eff, :tw_eff].astype(np.float32)
                    del tdata
                    fw = _feather_weight(th_eff, tw_eff, margin=100)
                    valid = ~np.isnan(tile_ndsm)
                    w = np.where(valid, fw, 0.0).astype(np.float64)
                    ndsm_sum[row_off:r_end, col_off:c_end] += \
                        np.where(valid, tile_ndsm, 0.0).astype(np.float64) * w
                    ndsm_w[row_off:r_end, col_off:c_end] += w
                except Exception:
                    pass
            has = ndsm_w > 0
            ndsm_full = np.where(has, ndsm_sum / ndsm_w, np.nan).astype(np.float32)
            del ndsm_sum, ndsm_w

            _, _label_remap = _merge_boundary_segments(
                labels_full, seg_type_full, all_objects,
                tile_seg_results, full_left, full_top, res,
                ndsm_full=ndsm_full, mark_uncertain=mark_uncertain)
            del ndsm_full
        except Exception as e:
            log.warning("Streamed GPKG boundary merge failed: %s", e)

    # Write categorical rasters via _write_table (small helper)
    def _write_table(name, arrays, h, w, tf, dtype='float32', descs=None):
        nonlocal table_count
        opts = dict(driver='GPKG', width=w, height=h, count=len(arrays),
                    dtype=dtype, crs='EPSG:3035', transform=tf,
                    RASTER_TABLE=name, RASTER_IDENTIFIER=name)
        if dtype == 'float32':
            opts['nodata'] = _GPKG_NODATA
        if table_count > 0:
            opts['APPEND_SUBDATASET'] = 'YES'
        with rasterio.open(out_path, 'w', **opts) as dst:
            if name == 'segment_type' and dtype == 'uint8':
                # Colormap BEFORE pixels → palette PNG tiles. See the same block
                # in austria_processor.build_full_gpkg_tiled._write_table;
                # order is load-bearing (post-band write is ignored by GDAL).
                try:
                    from austria_processor import _segment_type_colormap
                    dst.write_colormap(1, _segment_type_colormap())
                except Exception as _e:
                    log.warning("segment_type colormap write failed: %s", _e)
            for i, arr in enumerate(arrays, 1):
                band = arr[:h, :w]
                if dtype == 'float32':
                    band = np.where(np.isnan(band), _GPKG_NODATA, band)
                dst.write(band, i)
                if descs and i <= len(descs):
                    dst.set_band_description(i, descs[i - 1])
        table_count += 1

    _write_table('segment_type', [seg_type_full], full_h, full_w, full_tf,
                 dtype='uint8', descs=['Object type code'])
    del seg_type_full

    _write_table('segment_height', [seg_height_full], full_h, full_w, full_tf,
                 descs=['Segment height (m)'])
    del seg_height_full

    # Segment vectors from labels_full
    if all_objects and tile_seg_results:
        try:
            mask_full = labels_full > 0
            _write_segment_vectors(
                out_path, labels_full, all_objects,
                mask_full, full_tf, layer_name='segments', obs_year=obs_year)
        except Exception as e:
            log.warning("Streamed GPKG segment vectors failed: %s", e)
    del labels_full

    if all_objects:
        try:
            _write_segment_points(out_path, all_objects,
                                  layer_name='segment_points', obs_year=obs_year)
        except Exception as e:
            log.warning("Streamed GPKG segment points failed: %s", e)

    # ===================================================================
    # Phase B — Continuous raster layers, strip-streamed
    # ===================================================================

    # --- B1: DTM ---
    def _fetch_dtm(tr, th_eff, tw_eff):
        try:
            tdata = _read_dtm_for_tile(tr)
            return [tdata["dtm"][:th_eff, :tw_eff].astype(np.float32)]
        except Exception:
            return None

    table_count = _streamed_write_layer(
        out_path, table_count, tile_info, full_h, full_w, full_tf,
        'DTM', ['Digital Terrain Model (m)'], _fetch_dtm,
        dtype='float32', blend='feather')
    log.info("  STREAMED GPKG: DTM written")

    # --- B2: DSM ---
    def _fetch_dsm(tr, th_eff, tw_eff):
        try:
            tdata = _read_dtm_for_tile(tr)
            return [tdata["dsm"][:th_eff, :tw_eff].astype(np.float32)]
        except Exception:
            return None

    table_count = _streamed_write_layer(
        out_path, table_count, tile_info, full_h, full_w, full_tf,
        'DSM', ['Digital Surface Model (m)'], _fetch_dsm,
        dtype='float32', blend='feather')
    log.info("  STREAMED GPKG: DSM written")

    # --- B3: nDSM ---
    def _fetch_ndsm(tr, th_eff, tw_eff):
        try:
            tdata = _read_dtm_for_tile(tr)
            return [tdata["ndsm"][:th_eff, :tw_eff].astype(np.float32)]
        except Exception:
            return None

    table_count = _streamed_write_layer(
        out_path, table_count, tile_info, full_h, full_w, full_tf,
        'nDSM', ['Normalised DSM (m)'], _fetch_ndsm,
        dtype='float32', blend='feather')
    log.info("  STREAMED GPKG: nDSM written")

    # --- B4: Multi-date DTM/DSM ---
    import tile_index as _ti
    import raster_io as _rio
    from shapely.geometry import box

    other_dates = sorted(d for d in _ti.DATASETS if d != _ti.DEFAULT_DATASET)
    for date_key in other_dates:
        year = _ti.dataset_to_year(date_key)

        def _make_fetch_dtm_date(dk):
            def _fetch(tr, th_eff, tw_eff):
                try:
                    d2 = _rio.read_dtm_dsm(box(*tr["bounds_3035"]), dk)
                    return [d2["dtm"][:th_eff, :tw_eff].astype(np.float32)]
                except Exception:
                    return None
            return _fetch

        def _make_fetch_dsm_date(dk):
            def _fetch(tr, th_eff, tw_eff):
                try:
                    d2 = _rio.read_dtm_dsm(box(*tr["bounds_3035"]), dk)
                    return [d2["dsm"][:th_eff, :tw_eff].astype(np.float32)]
                except Exception:
                    return None
            return _fetch

        try:
            table_count = _streamed_write_layer(
                out_path, table_count, tile_info, full_h, full_w, full_tf,
                f'DTM_{year}', [f'DTM {year} (m)'],
                _make_fetch_dtm_date(date_key),
                dtype='float32', blend='feather')
            table_count = _streamed_write_layer(
                out_path, table_count, tile_info, full_h, full_w, full_tf,
                f'DSM_{year}', [f'DSM {year} (m)'],
                _make_fetch_dsm_date(date_key),
                dtype='float32', blend='feather')
            log.info("  STREAMED GPKG: DTM_%d + DSM_%d written", year, year)
        except Exception as e:
            log.warning("STREAMED GPKG: multi-date %d failed: %s", year, e)

    # --- B5: Ortho + CIR per year ---
    import ortho_io as _oio
    import concurrent.futures as _cf
    _ORTHO_YEARS = {2024: 2024, 2023: 2023, 2020: 2020}
    ORTHO_TIMEOUT = 180

    for o_year in sorted(_ORTHO_YEARS.keys(), reverse=True):
        # First pass: check if any tile has data for this year
        def _make_fetch_ortho(yr):
            def _fetch(tr, th_eff, tw_eff):
                try:
                    tdata = _read_dtm_for_tile(tr)
                    with _cf.ThreadPoolExecutor(max_workers=1) as exe:
                        fut = exe.submit(_oio.read_ortho_for_als, tdata, year=yr)
                        rgb, nir = fut.result(timeout=ORTHO_TIMEOUT)
                    del tdata
                    if rgb is None:
                        return None
                    bands = [
                        rgb[0][:th_eff, :tw_eff].astype(np.float32),
                        rgb[1][:th_eff, :tw_eff].astype(np.float32),
                        rgb[2][:th_eff, :tw_eff].astype(np.float32),
                    ]
                    if nir is not None:
                        bands.append(nir[:th_eff, :tw_eff].astype(np.float32))
                    return bands
                except Exception:
                    return None
            return _fetch

        fetch_ortho = _make_fetch_ortho(_ORTHO_YEARS[o_year])

        # Probe first tile to determine band count (RGB vs RGBI)
        got_nir = False
        probe_bands = None
        for _ti, tr, (_, _, _, _, th_eff, tw_eff) in tile_info:
            probe_bands = fetch_ortho(tr, th_eff, tw_eff)
            if probe_bands is not None:
                got_nir = len(probe_bands) == 4
                break

        if probe_bands is None:
            log.info("  STREAMED GPKG: no ortho data for year %d", o_year)
            continue

        try:
            descs_rgb = ['Red', 'Green', 'Blue']
            if got_nir:
                descs_rgb.append('NIR')

            # For ortho we need a fetch that always returns the right band count
            def _make_fetch_ortho_fixed(yr, n_bands_expected):
                def _fetch(tr, th_eff, tw_eff):
                    try:
                        tdata = _read_dtm_for_tile(tr)
                        with _cf.ThreadPoolExecutor(max_workers=1) as exe:
                            fut = exe.submit(_oio.read_ortho_for_als, tdata, year=yr)
                            rgb, nir = fut.result(timeout=ORTHO_TIMEOUT)
                        del tdata
                        if rgb is None:
                            return None
                        bands = [
                            rgb[0][:th_eff, :tw_eff].astype(np.float32),
                            rgb[1][:th_eff, :tw_eff].astype(np.float32),
                            rgb[2][:th_eff, :tw_eff].astype(np.float32),
                        ]
                        if n_bands_expected == 4:
                            if nir is not None:
                                bands.append(nir[:th_eff, :tw_eff].astype(np.float32))
                            else:
                                bands.append(np.zeros((th_eff, tw_eff), dtype=np.float32))
                        return bands
                    except Exception:
                        return None
                return _fetch

            n_expected = 4 if got_nir else 3
            table_count = _streamed_write_layer(
                out_path, table_count, tile_info, full_h, full_w, full_tf,
                f'Ortho_{o_year}', descs_rgb,
                _make_fetch_ortho_fixed(_ORTHO_YEARS[o_year], n_expected),
                dtype='uint8', blend='feather')
            log.info("  STREAMED GPKG: Ortho_%d written (NIR=%s)", o_year, got_nir)

            if got_nir:
                # CIR: NIR→Red, Red→Green, Green→Blue
                def _make_fetch_cir(yr):
                    def _fetch(tr, th_eff, tw_eff):
                        try:
                            tdata = _read_dtm_for_tile(tr)
                            with _cf.ThreadPoolExecutor(max_workers=1) as exe:
                                fut = exe.submit(_oio.read_ortho_for_als, tdata, year=yr)
                                rgb, nir = fut.result(timeout=ORTHO_TIMEOUT)
                            del tdata
                            if rgb is None or nir is None:
                                return None
                            return [
                                nir[:th_eff, :tw_eff].astype(np.float32),
                                rgb[0][:th_eff, :tw_eff].astype(np.float32),
                                rgb[1][:th_eff, :tw_eff].astype(np.float32),
                            ]
                        except Exception:
                            return None
                    return _fetch

                table_count = _streamed_write_layer(
                    out_path, table_count, tile_info, full_h, full_w, full_tf,
                    f'CIR_{o_year}',
                    ['NIR\u2192Red', 'Red\u2192Green', 'Green\u2192Blue'],
                    _make_fetch_cir(_ORTHO_YEARS[o_year]),
                    dtype='uint8', blend='feather')
                log.info("  STREAMED GPKG: CIR_%d written", o_year)
        except Exception as e:
            log.warning("STREAMED GPKG: ortho year %d failed: %s", o_year, e)

    # --- B6: Copernicus NDVI ---
    try:
        from austria_processor import _get_cop_cache
        cop_cache = _get_cop_cache()

        def _fetch_ndvi(tr, th_eff, tw_eff):
            from rasterio.warp import reproject, Resampling
            bbox_wgs = tr["bbox_wgs"]
            bbox_dict = {"west": bbox_wgs[0], "south": bbox_wgs[1],
                         "east": bbox_wgs[2], "north": bbox_wgs[3]}
            nd = cop_cache.get_ndvi(bbox_dict, year=obs_year)
            if not nd or nd.get("ndvi") is None:
                return None
            src_arr = nd["ndvi"].astype(np.float32)
            th_t, tw_t = tr["shape"]
            tile_left = tr["bounds_3035"][0]
            tile_top = tr["bounds_3035"][3]
            dst_tf = rasterio.transform.from_bounds(
                tile_left, tr["bounds_3035"][1],
                tr["bounds_3035"][2], tile_top, tw_t, th_t)
            dst_arr = np.full((th_t, tw_t), np.nan, dtype=np.float32)
            reproject(
                src_arr, dst_arr,
                src_transform=nd["transform"],
                src_crs=nd.get("crs", "EPSG:4326"),
                dst_transform=dst_tf, dst_crs='EPSG:3035',
                resampling=Resampling.bilinear)
            return [dst_arr[:th_eff, :tw_eff]]

        table_count = _streamed_write_layer(
            out_path, table_count, tile_info, full_h, full_w, full_tf,
            'NDVI', ['Sentinel-2 NDVI composite'], _fetch_ndvi,
            dtype='float32', blend='feather')
        log.info("  STREAMED GPKG: NDVI written")
    except Exception as e:
        from tile_cache import CacheMissError as _CacheMissError
        if isinstance(e, _CacheMissError) or isinstance(getattr(e, '__cause__', None), _CacheMissError):
            raise
        log.warning("STREAMED GPKG: NDVI layer failed: %s", e)

    # --- B7: ESA WorldCover ---
    try:
        from austria_processor import _get_cop_cache
        cop_cache = _get_cop_cache()

        def _fetch_worldcover(tr, th_eff, tw_eff):
            from rasterio.warp import reproject, Resampling
            bbox_wgs = tr["bbox_wgs"]
            bbox_dict = {"west": bbox_wgs[0], "south": bbox_wgs[1],
                         "east": bbox_wgs[2], "north": bbox_wgs[3]}
            lc = cop_cache.get_landcover(bbox_dict)
            if not lc or lc.get("map") is None:
                return None
            src_arr = lc["map"].astype(np.uint8)
            th_t, tw_t = tr["shape"]
            tile_left = tr["bounds_3035"][0]
            tile_top = tr["bounds_3035"][3]
            dst_tf = rasterio.transform.from_bounds(
                tile_left, tr["bounds_3035"][1],
                tr["bounds_3035"][2], tile_top, tw_t, th_t)
            dst_arr = np.zeros((th_t, tw_t), dtype=np.uint8)
            reproject(
                src_arr, dst_arr,
                src_transform=lc["transform"],
                src_crs=lc.get("crs", "EPSG:4326"),
                dst_transform=dst_tf, dst_crs='EPSG:3035',
                resampling=Resampling.nearest)
            return [dst_arr[:th_eff, :tw_eff]]

        table_count = _streamed_write_layer(
            out_path, table_count, tile_info, full_h, full_w, full_tf,
            'WorldCover', ['ESA WorldCover 2021 class'], _fetch_worldcover,
            dtype='uint8', blend='overwrite')
        log.info("  STREAMED GPKG: WorldCover written")
    except Exception as e:
        from tile_cache import CacheMissError as _CacheMissError
        if isinstance(e, _CacheMissError) or isinstance(getattr(e, '__cause__', None), _CacheMissError):
            raise
        log.warning("STREAMED GPKG: WorldCover layer failed: %s", e)

    # --- B8: Sentinel-1 SAR (VV + VH) ---
    try:
        from austria_processor import _get_cop_cache
        cop_cache = _get_cop_cache()

        def _reproject_sar_band(sar, band_name, tr, th_eff, tw_eff):
            from rasterio.warp import reproject, Resampling
            src = sar.get(band_name)
            if src is None:
                return np.full((th_eff, tw_eff), np.nan, dtype=np.float32)
            th_t, tw_t = tr["shape"]
            tile_left = tr["bounds_3035"][0]
            tile_top = tr["bounds_3035"][3]
            dst_tf = rasterio.transform.from_bounds(
                tile_left, tr["bounds_3035"][1],
                tr["bounds_3035"][2], tile_top, tw_t, th_t)
            dst_arr = np.full((th_t, tw_t), np.nan, dtype=np.float32)
            reproject(
                src.astype(np.float32), dst_arr,
                src_transform=sar["transform"],
                src_crs=sar.get("crs", "EPSG:4326"),
                dst_transform=dst_tf, dst_crs='EPSG:3035',
                resampling=Resampling.bilinear)
            return dst_arr[:th_eff, :tw_eff]

        def _make_fetch_sar_band(band_name):
            def _fetch(tr, th_eff, tw_eff):
                bbox_wgs = tr["bbox_wgs"]
                bbox_dict = {"west": bbox_wgs[0], "south": bbox_wgs[1],
                             "east": bbox_wgs[2], "north": bbox_wgs[3]}
                sar = cop_cache.get_sar(bbox_dict, year=obs_year)
                if not sar:
                    return None
                return [_reproject_sar_band(sar, band_name, tr, th_eff, tw_eff)]
            return _fetch

        table_count = _streamed_write_layer(
            out_path, table_count, tile_info, full_h, full_w, full_tf,
            'SAR_VV', ['Sentinel-1 VV (dB)'], _make_fetch_sar_band('vv'),
            dtype='float32', blend='overwrite')
        table_count = _streamed_write_layer(
            out_path, table_count, tile_info, full_h, full_w, full_tf,
            'SAR_VH', ['Sentinel-1 VH (dB)'], _make_fetch_sar_band('vh'),
            dtype='float32', blend='overwrite')
        log.info("  STREAMED GPKG: SAR_VV + SAR_VH written")
    except Exception as e:
        from tile_cache import CacheMissError as _CacheMissError
        if isinstance(e, _CacheMissError) or isinstance(getattr(e, '__cause__', None), _CacheMissError):
            raise
        log.warning("STREAMED GPKG: SAR layer failed: %s", e)

    # --- B9: Hansen Global Forest Change ---
    try:
        from austria_processor import _get_hansen_cache
        hc = _get_hansen_cache()

        def _fetch_hansen_tc(tr, th_eff, tw_eff):
            bbox_wgs = tr["bbox_wgs"]
            th_t, tw_t = tr["shape"]
            tile_left = tr["bounds_3035"][0]
            tile_top = tr["bounds_3035"][3]
            dst_tf = rasterio.transform.from_bounds(
                tile_left, tr["bounds_3035"][1],
                tr["bounds_3035"][2], tile_top, tw_t, th_t)
            hd = hc.get_forest_prior(bbox_wgs, dst_tf, (th_t, tw_t))
            if not hd or hd.get("treecover2000") is None:
                return None
            return [hd["treecover2000"][:th_eff, :tw_eff]]

        def _fetch_hansen_ly(tr, th_eff, tw_eff):
            bbox_wgs = tr["bbox_wgs"]
            th_t, tw_t = tr["shape"]
            tile_left = tr["bounds_3035"][0]
            tile_top = tr["bounds_3035"][3]
            dst_tf = rasterio.transform.from_bounds(
                tile_left, tr["bounds_3035"][1],
                tr["bounds_3035"][2], tile_top, tw_t, th_t)
            hd = hc.get_forest_prior(bbox_wgs, dst_tf, (th_t, tw_t))
            if not hd or hd.get("loss_year") is None:
                return None
            return [hd["loss_year"][:th_eff, :tw_eff]]

        table_count = _streamed_write_layer(
            out_path, table_count, tile_info, full_h, full_w, full_tf,
            'Hansen_treecover', ['Tree cover 2000 (%)'], _fetch_hansen_tc,
            dtype='uint8', blend='overwrite')
        table_count = _streamed_write_layer(
            out_path, table_count, tile_info, full_h, full_w, full_tf,
            'Hansen_lossyear',
            ['Forest loss year (0=none, 1=2001, ..., 23=2023)'],
            _fetch_hansen_ly,
            dtype='uint8', blend='overwrite')
        log.info("  STREAMED GPKG: Hansen (treecover + lossyear) written")
    except Exception as e:
        from tile_cache import CacheMissError as _CacheMissError
        if isinstance(e, _CacheMissError) or isinstance(getattr(e, '__cause__', None), _CacheMissError):
            raise
        log.warning("STREAMED GPKG: Hansen layer failed: %s", e)

    # ===================================================================
    # Finish: styles + CRS fix
    # ===================================================================
    try:
        _write_gpkg_all_styles(
            out_path,
            has_segments=bool(all_objects),
            has_points=bool(all_objects))
    except Exception as e:
        log.warning("Streamed GPKG styles failed: %s", e)

    _fix_gpkg_raster_crs(out_path)

    fsize = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    log.info("  STREAMED GPKG: %.1f MB, %d tables, %d tiles",
             fsize / 1e6, table_count, len(tile_seg_results))
    return out_path, _label_remap


# Re-read helper — same as austria_processor._read_dtm_for_tile
def _read_dtm_for_tile(tr):
    import raster_io as _rio
    import tile_index as _ti
    from shapely.geometry import box
    return _rio.read_dtm_dsm(box(*tr["bounds_3035"]), _ti.DEFAULT_DATASET)

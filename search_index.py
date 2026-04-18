"""search_index.py — SQLite FTS5 + R-tree search index for Austrian landscape analysis.

Builds a fast queryable index over all ~8440 KGs from kg_list.json,
enriched with landscape summary data from processed JSON files and
Zenodo download links from the manifest.

Tables: kg, kg_landcover, kg_hansen, kg_rtree (R-tree), fts_kg (FTS5), index_meta
"""
import fcntl
import json
import os
import sqlite3
import threading
import time
import logging
from pathlib import Path

log = logging.getLogger(__name__)

STATE_CODES = {
    '1': 'Burgenland', '2': 'Kärnten', '3': 'Niederösterreich',
    '4': 'Oberösterreich', '5': 'Salzburg', '6': 'Steiermark',
    '7': 'Tirol', '8': 'Vorarlberg', '9': 'Wien',
}

DISTRICT_NAMES = {
    "101": "Eisenstadt (Stadt)", "102": "Rust (Stadt)", "103": "Eisenstadt-Umgebung",
    "104": "Güssing", "105": "Jennersdorf", "106": "Mattersburg",
    "107": "Neusiedl am See", "108": "Oberpullendorf", "109": "Oberwart",
    "201": "Klagenfurt (Stadt)", "202": "Villach (Stadt)", "203": "Hermagor",
    "204": "Klagenfurt-Land", "205": "Sankt Veit an der Glan",
    "206": "Spittal an der Drau", "207": "Villach-Land", "208": "Völkermarkt",
    "209": "Wolfsberg", "210": "Feldkirchen",
    "301": "Krems an der Donau (Stadt)", "302": "Sankt Pölten (Stadt)",
    "303": "Waidhofen an der Ybbs (Stadt)", "304": "Wiener Neustadt (Stadt)",
    "305": "Amstetten", "306": "Baden", "307": "Bruck an der Leitha",
    "308": "Gänserndorf", "309": "Gmünd", "310": "Hollabrunn",
    "311": "Horn", "312": "Korneuburg", "313": "Krems (Land)",
    "314": "Lilienfeld", "315": "Melk", "316": "Mistelbach",
    "317": "Mödling", "318": "Neunkirchen", "319": "Sankt Pölten (Land)",
    "320": "Scheibbs", "321": "Tulln", "322": "Waidhofen an der Thaya",
    "323": "Wiener Neustadt (Land)", "324": "Zwettl",
    "401": "Linz (Stadt)", "402": "Steyr (Stadt)", "403": "Wels (Stadt)",
    "404": "Braunau am Inn", "405": "Eferding", "406": "Freistadt",
    "407": "Gmunden", "408": "Grieskirchen", "409": "Kirchdorf an der Krems",
    "410": "Linz-Land", "411": "Perg", "412": "Ried im Innkreis",
    "413": "Rohrbach", "414": "Schärding", "415": "Steyr-Land",
    "416": "Urfahr-Umgebung", "417": "Vöcklabruck", "418": "Wels-Land",
    "501": "Salzburg (Stadt)", "502": "Hallein", "503": "Salzburg-Umgebung",
    "504": "Sankt Johann im Pongau", "505": "Tamsweg", "506": "Zell am See",
    "601": "Graz (Stadt)", "603": "Deutschlandsberg", "606": "Graz-Umgebung",
    "610": "Leibnitz", "611": "Leoben", "612": "Liezen",
    "614": "Murau", "616": "Voitsberg", "617": "Weiz",
    "620": "Murtal", "621": "Bruck-Mürzzuschlag", "622": "Hartberg-Fürstenfeld",
    "623": "Südoststeiermark",
    "701": "Innsbruck (Stadt)", "702": "Imst", "703": "Innsbruck-Land",
    "704": "Kitzbühel", "705": "Kufstein", "706": "Landeck",
    "707": "Lienz", "708": "Reutte", "709": "Schwaz",
    "801": "Bludenz", "802": "Bregenz", "803": "Dornbirn", "804": "Feldkirch",
    "900": "Wien", "901": "Wien",
}

BASE_URL = 'https://srtm-lidar-at.exe.xyz:8000'


class SearchIndex:
    """SQLite FTS5 + R-tree search index for Austrian landscape KGs."""

    def __init__(self, db_path='data/search_index.db'):
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self.db_path = db_path
        self._write_lock = threading.Lock()
        self._file_lock_path = db_path + '.lock'
        self._local = threading.local()
        self._migrate()

    def _conn(self):
        """Thread-local connection."""
        c = getattr(self._local, 'conn', None)
        if c is None:
            c = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
            c.row_factory = sqlite3.Row
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('PRAGMA synchronous=NORMAL')
            c.execute('PRAGMA cache_size=-64000')
            c.execute('PRAGMA foreign_keys=ON')
            c.execute('PRAGMA busy_timeout=30000')
            self._local.conn = c
        return c

    def _schema_stmts(self):
        """Return list of CREATE statements for current schema."""
        return [
            # === Core KG table ===
            '''CREATE TABLE IF NOT EXISTS kg (
                kg_code TEXT PRIMARY KEY,
                kg_name TEXT NOT NULL,
                gemeinde_code TEXT,
                gemeinde_name TEXT,
                district_code TEXT,
                district_name TEXT,
                state_code TEXT,
                state_name TEXT,
                min_lon REAL, min_lat REAL, max_lon REAL, max_lat REAL,
                centroid_lon REAL, centroid_lat REAL,
                parcel_count INTEGER DEFAULT 0,
                building_count INTEGER DEFAULT 0,
                total_area_sqm REAL DEFAULT 0,
                processed INTEGER DEFAULT 0,
                generated_at TEXT,
                primary_year INTEGER,
                dominant_type TEXT,
                vegetated_fraction REAL,
                shannon_diversity REAL,
                n_segments INTEGER DEFAULT 0,
                elevation_min_m REAL,
                elevation_max_m REAL,
                elevation_mean_m REAL,
                slope_mean_deg REAL,
                aspect_dominant TEXT,
                ndvi_mean REAL,
                tree_count INTEGER DEFAULT 0,
                tree_canopy_sqm REAL DEFAULT 0,
                tree_mean_height_m REAL,
                tree_stem_volume_m3 REAL DEFAULT 0,
                net_volume_change_m3 REAL,
                temporal_stability REAL,
                new_building_count INTEGER DEFAULT 0,
                infrastructure_count INTEGER DEFAULT 0,
                -- building aggregates (from building_footprints + new_buildings)
                building_footprint_sqm REAL DEFAULT 0,
                building_mean_height_m REAL,
                building_max_height_m REAL,
                building_stories_mean REAL,
                building_stories_max INTEGER,
                building_pitched_pct REAL,
                new_building_footprint_sqm REAL DEFAULT 0,
                new_building_mean_height_m REAL,
                new_building_stories_mean REAL,
                -- SAR
                sar_vv_mean_db REAL,
                sar_vh_mean_db REAL,
                -- NDVI harmonics
                ndvi_harm_mean REAL,
                ndvi_harm_amplitude REAL,
                ndvi_harm_phase REAL,
                -- temporal change
                dtm_change_mean_m REAL,
                n_changed_segments INTEGER,
                total_disturbed_volume_m3 REAL,
                -- phenology
                phenology_dominant TEXT,
                -- coverage
                n_tiles INTEGER,
                building_height_coverage_pct REAL,
                --
                quality_score REAL,
                quality_grade TEXT,
                zenodo_json_url TEXT,
                zenodo_json_size INTEGER,
                zenodo_light_gpkg_url TEXT,
                zenodo_light_gpkg_size INTEGER,
                zenodo_full_gpkg_url TEXT,
                zenodo_full_gpkg_size INTEGER,
                zenodo_depo_id INTEGER
            )''',
            # === Per-type landcover breakdown ===
            '''CREATE TABLE IF NOT EXISTS kg_landcover (
                kg_code TEXT NOT NULL,
                object_type TEXT NOT NULL,
                area_sqm REAL DEFAULT 0,
                fraction REAL DEFAULT 0,
                n_objects INTEGER DEFAULT 0,
                height_min REAL, height_max REAL, height_mean REAL, height_p90 REAL,
                PRIMARY KEY (kg_code, object_type)
            )''',
            # === Hansen forest loss by year ===
            '''CREATE TABLE IF NOT EXISTS kg_hansen (
                kg_code TEXT NOT NULL,
                loss_year INTEGER NOT NULL,
                loss_pixels INTEGER DEFAULT 0,
                PRIMARY KEY (kg_code, loss_year)
            )''',
            # === Index metadata ===
            '''CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )''',
            # === R-tree spatial index ===
            '''CREATE VIRTUAL TABLE IF NOT EXISTS kg_rtree USING rtree(
                id, min_lon, max_lon, min_lat, max_lat
            )''',
            # === FTS5 text search ===
            '''CREATE VIRTUAL TABLE IF NOT EXISTS fts_kg USING fts5(
                kg_code, kg_name, gemeinde_name, district_name, state_name,
                tokenize="unicode61 remove_diacritics 2"
            )''',
            # === Indexes ===
            'CREATE INDEX IF NOT EXISTS idx_kg_gemeinde ON kg(gemeinde_code)',
            'CREATE INDEX IF NOT EXISTS idx_kg_district ON kg(district_code)',
            'CREATE INDEX IF NOT EXISTS idx_kg_state ON kg(state_code)',
            'CREATE INDEX IF NOT EXISTS idx_kg_processed ON kg(processed)',
            'CREATE INDEX IF NOT EXISTS idx_kg_dominant ON kg(dominant_type)',
            'CREATE INDEX IF NOT EXISTS idx_kg_quality ON kg(quality_score)',
            'CREATE INDEX IF NOT EXISTS idx_kg_year ON kg(primary_year)',
            'CREATE INDEX IF NOT EXISTS idx_kg_tree_count ON kg(tree_count)',
            'CREATE INDEX IF NOT EXISTS idx_kg_new_bldg ON kg(new_building_count)',
            'CREATE INDEX IF NOT EXISTS idx_lc_type ON kg_landcover(object_type)',
            'CREATE INDEX IF NOT EXISTS idx_hansen_year ON kg_hansen(loss_year)',
        ]

    def _migrate(self):
        c = self._conn()
        for s in self._schema_stmts():
            try:
                c.execute(s)
            except Exception as e:
                log.warning('migrate: %s: %s', s[:60], e)
        c.commit()

    # ════════════════════════════════════════════════════════════════
    # Build / Update
    # ════════════════════════════════════════════════════════════════

    def build(self, kg_list_path='data/austria_processor/kg_list.json',
              json_dir='data/austria_processor/json',
              manifest_path='data/austria_processor/zenodo_manifest.json'):
        """Full rebuild from scratch. Completes in <2s for 8440 KGs."""
        t0 = time.time()
        # File lock prevents concurrent rebuilds across gunicorn workers
        lock_fd = open(self._file_lock_path, 'w')
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            return self._build_inner(kg_list_path, json_dir, manifest_path, t0)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def _build_inner(self, kg_list_path, json_dir, manifest_path, t0):
        with self._write_lock:
            c = self._conn()
            # Drop and recreate — ensures schema changes are picked up
            for t in ('kg', 'kg_landcover', 'kg_hansen', 'index_meta'):
                c.execute(f'DROP TABLE IF EXISTS {t}')
            for t in ('kg_rtree', 'fts_kg'):
                c.execute(f'DROP TABLE IF EXISTS {t}')
            c.commit()
            # Recreate tables with current schema (inline, same connection)
            for s in self._schema_stmts():
                try:
                    c.execute(s)
                except Exception as e:
                    log.warning('build schema: %s: %s', s[:60], e)
            c.commit()

            # Load sources
            kg_list = json.loads(Path(kg_list_path).read_text()) if Path(kg_list_path).exists() else []
            manifest = {}
            if Path(manifest_path).exists():
                md = json.loads(Path(manifest_path).read_text())
                manifest = md.get('entries', md)

            # Index all KGs from kg_list
            kg_rows = []
            fts_rows = []
            rtree_rows = []

            for i, kg in enumerate(kg_list):
                code = kg['kg_code']
                gc = kg.get('gemeinde_code', '')
                sc = gc[:1] if gc else ''
                dc = gc[:3] if len(gc) >= 3 else ''
                sn = kg.get('state_name', STATE_CODES.get(sc, ''))
                dn = kg.get('district_name', DISTRICT_NAMES.get(dc, ''))
                bb = kg.get('bbox', {})
                mn_lon = bb.get('min_lon', 0)
                mn_lat = bb.get('min_lat', 0)
                mx_lon = bb.get('max_lon', 0)
                mx_lat = bb.get('max_lat', 0)
                cx = kg.get('lon', (mn_lon + mx_lon) / 2 if mn_lon else 0)
                cy = kg.get('lat', (mn_lat + mx_lat) / 2 if mn_lat else 0)

                # Zenodo links
                zj = manifest.get(f'{code}_json', {})
                zl = manifest.get(f'{code}_light_gpkg', {})
                zf = manifest.get(f'{code}_full_gpkg', {})

                kg_rows.append((
                    code, kg.get('kg_name', ''), gc, kg.get('gemeinde_name', ''),
                    dc, dn, sc, sn,
                    mn_lon, mn_lat, mx_lon, mx_lat, cx, cy,
                    kg.get('parcel_count', 0), kg.get('building_count', 0),
                    kg.get('total_area_sqm', 0),
                    0, None, None,  # processed, generated_at, primary_year
                    None, None, None, 0,  # landscape
                    None, None, None, None, None,  # terrain
                    None,  # ndvi
                    0, 0, None, 0,  # tree_stats
                    None, None,  # temporal
                    0, 0,  # new_building, infrastructure
                    0, None, None, None, None, None,  # building aggregates
                    0, None, None,  # new_building aggregates
                    None, None,  # sar
                    None, None, None,  # ndvi harmonics
                    None, None, None,  # temporal change
                    None,  # phenology
                    None, None,  # coverage
                    None, None,  # quality
                    _zenodo_url(zj), zj.get('size'),
                    _zenodo_url(zl), zl.get('size'),
                    _zenodo_url(zf), zf.get('size'),
                    zj.get('depo_id') or zl.get('depo_id') or zf.get('depo_id'),
                ))
                fts_rows.append((code, kg.get('kg_name', ''), kg.get('gemeinde_name', ''), dn, sn))
                rowid = i + 1
                if mn_lon and mx_lon and mn_lat and mx_lat:
                    rtree_rows.append((rowid, mn_lon, mx_lon, mn_lat, mx_lat))

            c.executemany(
                '''INSERT OR REPLACE INTO kg VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ''', kg_rows)
            c.executemany('INSERT INTO fts_kg VALUES (?,?,?,?,?)', fts_rows)
            c.executemany('INSERT INTO kg_rtree VALUES (?,?,?,?,?)', rtree_rows)

            # Enrich processed KGs from JSON files
            json_dir_p = Path(json_dir)
            n_processed = 0
            if json_dir_p.exists():
                for jf in json_dir_p.glob('*.json'):
                    code = jf.stem
                    if not code.isdigit():
                        continue
                    try:
                        data = json.loads(jf.read_text())
                        self._enrich_kg(c, code, data)
                        n_processed += 1
                    except Exception as e:
                        log.warning('enrich %s: %s', code, e)

            elapsed = time.time() - t0
            c.execute('INSERT OR REPLACE INTO index_meta VALUES (?, ?)',
                      ('built_at', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
            c.execute('INSERT OR REPLACE INTO index_meta VALUES (?, ?)',
                      ('build_time_ms', str(int(elapsed * 1000))))
            c.execute('INSERT OR REPLACE INTO index_meta VALUES (?, ?)',
                      ('kg_count', str(len(kg_rows))))
            c.execute('INSERT OR REPLACE INTO index_meta VALUES (?, ?)',
                      ('processed_count', str(n_processed)))
            c.commit()
            log.info('🔍 Search index built: %d KGs (%d processed) in %.1fs',
                     len(kg_rows), n_processed, elapsed)

    def _enrich_kg(self, c, code, data):
        """Enrich a KG row with data from its JSON summary."""
        ls = data.get('landscape', {})
        tr = data.get('terrain', {})
        ts = data.get('tree_stats', {})
        tc = data.get('temporal_change', {})
        nd = data.get('ndvi', {})
        dq = data.get('data_quality', {})
        op = data.get('observation_period', {})
        nb = data.get('new_buildings', {})
        inf = data.get('infrastructure', {})
        bf = data.get('building_footprints', {})
        sar = data.get('sar', {})
        harm = data.get('ndvi_harmonics', {})
        phen = data.get('phenology', {})
        cov = data.get('coverage', {})

        # --- Building aggregate stats ---
        bf_details = bf.get('details', [])
        bf_heights = [b['max_height_m'] for b in bf_details if b.get('max_height_m')]
        bf_stories = [b['stories_est'] for b in bf_details if b.get('stories_est')]
        bf_roofs = [b.get('roof_type_hint') for b in bf_details if b.get('roof_type_hint')]
        bf_areas = [b.get('footprint_area_sqm', 0) for b in bf_details]

        nb_features = nb.get('features', [])
        nb_heights = [b['max_height_m'] for b in nb_features if b.get('max_height_m')]
        nb_stories = [b['stories_est'] for b in nb_features if b.get('stories_est')]
        nb_areas = [b.get('area_sqm', 0) for b in nb_features]

        # Phenology dominant class
        phen_dist = phen.get('distribution', {})
        phen_dominant = max(phen_dist, key=phen_dist.get) if phen_dist else None

        c.execute('''UPDATE kg SET
            processed=1, generated_at=?, primary_year=?,
            total_area_sqm=?, parcel_count=?,
            dominant_type=?, vegetated_fraction=?, shannon_diversity=?, n_segments=?,
            building_count=?,
            elevation_min_m=?, elevation_max_m=?, elevation_mean_m=?,
            slope_mean_deg=?, aspect_dominant=?,
            ndvi_mean=?,
            tree_count=?, tree_canopy_sqm=?, tree_mean_height_m=?, tree_stem_volume_m3=?,
            net_volume_change_m3=?, temporal_stability=?,
            new_building_count=?, infrastructure_count=?,
            building_footprint_sqm=?, building_mean_height_m=?, building_max_height_m=?,
            building_stories_mean=?, building_stories_max=?, building_pitched_pct=?,
            new_building_footprint_sqm=?, new_building_mean_height_m=?, new_building_stories_mean=?,
            sar_vv_mean_db=?, sar_vh_mean_db=?,
            ndvi_harm_mean=?, ndvi_harm_amplitude=?, ndvi_harm_phase=?,
            dtm_change_mean_m=?, n_changed_segments=?, total_disturbed_volume_m3=?,
            phenology_dominant=?,
            n_tiles=?, building_height_coverage_pct=?,
            quality_score=?, quality_grade=?
            WHERE kg_code=?''',
            (
                data.get('generated_at'), op.get('primary_year'),
                data.get('total_area_sqm', 0),
                data.get('parcels', {}).get('count', 0),
                ls.get('dominant_type'), ls.get('vegetated_fraction'),
                ls.get('shannon_diversity'), ls.get('n_segments'),
                bf.get('count', 0),
                tr.get('elevation_min_m'), tr.get('elevation_max_m'), tr.get('elevation_mean_m'),
                tr.get('steepness_mean_deg'), tr.get('aspect_dominant'),
                nd.get('copernicus_mean') or nd.get('bev_nir_mean'),
                ts.get('count', 0), ts.get('total_canopy_sqm', 0),
                ts.get('mean_height_m'), ts.get('est_stem_volume_m3', 0),
                tc.get('net_volume_change_m3'), tc.get('mean_stability'),
                nb.get('count', 0), inf.get('total', 0),
                # building aggregates
                sum(bf_areas) if bf_areas else 0,
                (sum(bf_heights) / len(bf_heights)) if bf_heights else None,
                max(bf_heights) if bf_heights else None,
                (sum(bf_stories) / len(bf_stories)) if bf_stories else None,
                max(bf_stories) if bf_stories else None,
                (sum(1 for r in bf_roofs if r == 'pitched') / len(bf_roofs) * 100) if bf_roofs else None,
                # new building aggregates
                sum(nb_areas) if nb_areas else 0,
                (sum(nb_heights) / len(nb_heights)) if nb_heights else None,
                (sum(nb_stories) / len(nb_stories)) if nb_stories else None,
                # sar
                sar.get('vv_mean_db'), sar.get('vh_mean_db'),
                # ndvi harmonics
                harm.get('mean_mean'), harm.get('amplitude_mean'), harm.get('phase_mean'),
                # temporal change
                tc.get('dtm_change_mean_m'), tc.get('n_changed_segments'), tc.get('total_disturbed_volume_m3'),
                # phenology
                phen_dominant,
                # coverage
                cov.get('n_tiles'), cov.get('building_height_coverage_pct'),
                # quality
                dq.get('quality_score'), dq.get('quality_grade'),
                code,
            ))

        # Landcover breakdown
        area_sum = data.get('area_summary', {})
        height_dist = data.get('height_distribution', {})
        lc_rows = []
        for otype, info in area_sum.items():
            hd = height_dist.get(otype, {})
            lc_rows.append((
                code, otype,
                info.get('area_sqm', 0), info.get('fraction', 0), info.get('n_objects', 0),
                hd.get('min'), hd.get('max'), hd.get('mean'), hd.get('p90'),
            ))
        if lc_rows:
            c.execute('DELETE FROM kg_landcover WHERE kg_code=?', (code,))
            c.executemany(
                'INSERT INTO kg_landcover VALUES (?,?,?,?,?,?,?,?,?)', lc_rows)

        # Hansen loss
        hansen = data.get('hansen', {})
        loss_by_year = hansen.get('loss_by_year', {})
        if loss_by_year:
            c.execute('DELETE FROM kg_hansen WHERE kg_code=?', (code,))
            h_rows = []
            for yr, val in loss_by_year.items():
                try:
                    px = val.get('pixels', val) if isinstance(val, dict) else val
                    h_rows.append((code, int(yr), int(px)))
                except (ValueError, TypeError):
                    pass
            if h_rows:
                c.executemany('INSERT INTO kg_hansen VALUES (?,?,?)', h_rows)

    def update_kg(self, kg_code, json_path=None, manifest=None):
        """Incremental update for a single KG after processing."""
        with self._write_lock:
            c = self._conn()
            if json_path and Path(json_path).exists():
                data = json.loads(Path(json_path).read_text())
                self._enrich_kg(c, kg_code, data)
            if manifest:
                for suffix, col_url, col_size in [
                    ('_json', 'zenodo_json_url', 'zenodo_json_size'),
                    ('_light_gpkg', 'zenodo_light_gpkg_url', 'zenodo_light_gpkg_size'),
                    ('_full_gpkg', 'zenodo_full_gpkg_url', 'zenodo_full_gpkg_size'),
                ]:
                    entry = manifest.get(f'{kg_code}{suffix}', {})
                    if entry:
                        url = _zenodo_url(entry)
                        c.execute(f'UPDATE kg SET {col_url}=?, {col_size}=? WHERE kg_code=?',
                                  (url, entry.get('size'), kg_code))
            c.commit()

    # ════════════════════════════════════════════════════════════════
    # Stats
    # ════════════════════════════════════════════════════════════════

    def stats(self):
        """Index statistics."""
        c = self._conn()
        s = {}
        for name, q in [
            ('kg_count', 'SELECT COUNT(*) FROM kg'),
            ('processed_count', 'SELECT COUNT(*) FROM kg WHERE processed=1'),
            ('total_area_km2', 'SELECT COALESCE(SUM(total_area_sqm),0)/1e6 FROM kg'),
            ('processed_area_km2', 'SELECT COALESCE(SUM(total_area_sqm),0)/1e6 FROM kg WHERE processed=1'),
            ('total_parcels', 'SELECT COALESCE(SUM(parcel_count),0) FROM kg'),
            ('total_buildings', 'SELECT COALESCE(SUM(building_count),0) FROM kg'),
            ('total_tree_count', 'SELECT COALESCE(SUM(tree_count),0) FROM kg WHERE processed=1'),
            ('total_new_buildings', 'SELECT COALESCE(SUM(new_building_count),0) FROM kg WHERE processed=1'),
            ('zenodo_kgs', 'SELECT COUNT(*) FROM kg WHERE zenodo_json_url IS NOT NULL'),
            ('states_covered', 'SELECT COUNT(DISTINCT state_code) FROM kg WHERE state_code != ""'),
            ('districts_covered', 'SELECT COUNT(DISTINCT district_code) FROM kg WHERE district_code != ""'),
            ('landcover_types', 'SELECT COUNT(DISTINCT object_type) FROM kg_landcover'),
            ('avg_quality_score', 'SELECT ROUND(AVG(quality_score),2) FROM kg WHERE quality_score IS NOT NULL'),
        ]:
            r = c.execute(q).fetchone()
            s[name] = round(r[0], 2) if r[0] is not None else 0
        # Meta
        for row in c.execute('SELECT key, value FROM index_meta'):
            s[f'_meta_{row[0]}'] = row[1]
        return s

    # ════════════════════════════════════════════════════════════════
    # Query: single KG
    # ════════════════════════════════════════════════════════════════

    def query_kg(self, kg_code):
        """Full KG record with landcover, hansen, and links."""
        c = self._conn()
        row = c.execute('SELECT * FROM kg WHERE kg_code=?', (kg_code,)).fetchone()
        if not row:
            return None
        d = dict(row)
        # Landcover
        d['landcover'] = [dict(r) for r in
            c.execute('SELECT * FROM kg_landcover WHERE kg_code=? ORDER BY area_sqm DESC', (kg_code,))]
        # Hansen
        d['hansen_loss'] = [dict(r) for r in
            c.execute('SELECT loss_year, loss_pixels FROM kg_hansen WHERE kg_code=? ORDER BY loss_year', (kg_code,))]
        # Links
        d['_links'] = self._build_links(d)
        return d

    def _build_links(self, d):
        """Build _links object for a KG dict."""
        code = d['kg_code']
        links = {'json': f'{BASE_URL}/api/v1/kg/{code}'}
        if d.get('zenodo_json_url'):
            links['zenodo_json'] = d['zenodo_json_url']
        if d.get('zenodo_light_gpkg_url'):
            links['zenodo_light_gpkg'] = d['zenodo_light_gpkg_url']
        if d.get('zenodo_full_gpkg_url'):
            links['zenodo_full_gpkg'] = d['zenodo_full_gpkg_url']
        if d.get('processed'):
            links['segment'] = f'{BASE_URL}/api/v1/segment'
            links['terrain'] = f'{BASE_URL}/api/v1/terrain'
        return links

    # ════════════════════════════════════════════════════════════════
    # Query: parcel
    # ════════════════════════════════════════════════════════════════

    def query_parcel(self, parcel_id):
        """Look up a parcel. Returns KG summary + tries to find parcel in JSON."""
        if '-' not in parcel_id:
            return None
        kg_code = parcel_id.split('-')[0]
        kg = self.query_kg(kg_code)
        if not kg:
            return None
        result = {'kg': kg, 'parcel_id': parcel_id, 'parcel_detail': None}
        # Try to read detail from local JSON
        jp = Path(f'data/austria_processor/json/{kg_code}.json')
        if jp.exists():
            try:
                data = json.loads(jp.read_text())
                for p in data.get('parcels', {}).get('details', []):
                    if p.get('parcel_id') == parcel_id:
                        result['parcel_detail'] = p
                        break
            except Exception:
                pass
        return result

    # ════════════════════════════════════════════════════════════════
    # Query: spatial (bbox / point)
    # ════════════════════════════════════════════════════════════════

    def query_bbox(self, min_lon, min_lat, max_lon, max_lat,
                   processed_only=False, limit=500):
        """Find KGs intersecting a bounding box via R-tree."""
        c = self._conn()
        # R-tree query: find KGs whose bbox overlaps the query bbox
        q = '''SELECT k.* FROM kg k
               JOIN kg_rtree r ON r.id = (SELECT rowid FROM kg WHERE kg_code=k.kg_code)
               WHERE r.max_lon >= ? AND r.min_lon <= ?
                 AND r.max_lat >= ? AND r.min_lat <= ?'''
        params = [min_lon, max_lon, min_lat, max_lat]
        if processed_only:
            q += ' AND k.processed=1'
        q += ' LIMIT ?'
        params.append(limit)
        # Faster approach: use R-tree directly then join
        q2 = '''SELECT k.* FROM kg_rtree r
                JOIN kg k ON k.rowid = r.id
                WHERE r.max_lon >= ? AND r.min_lon <= ?
                  AND r.max_lat >= ? AND r.min_lat <= ?'''
        params2 = [min_lon, max_lon, min_lat, max_lat]
        if processed_only:
            q2 += ' AND k.processed=1'
        q2 += ' LIMIT ?'
        params2.append(limit)
        return [self._kg_summary(r) for r in c.execute(q2, params2)]

    def query_point(self, lon, lat, radius_km=5):
        """Find KGs containing or near a point."""
        # Approximate degree offset for radius
        dlat = radius_km / 111.0
        dlon = radius_km / (111.0 * max(0.5, __import__('math').cos(__import__('math').radians(lat))))
        results = self.query_bbox(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
        # Sort by distance to centroid
        import math
        for r in results:
            cx = r.get('centroid_lon', 0) or 0
            cy = r.get('centroid_lat', 0) or 0
            r['_distance_km'] = round(math.sqrt(
                ((lon - cx) * 111 * math.cos(math.radians(lat))) ** 2 +
                ((lat - cy) * 111) ** 2
            ), 2)
        results.sort(key=lambda r: r.get('_distance_km', 999))
        return results

    # ════════════════════════════════════════════════════════════════
    # Query: admin hierarchy
    # ════════════════════════════════════════════════════════════════

    def query_admin(self, level, code=None, name=None, processed_only=False, limit=500):
        """Query by admin level: state/district/gemeinde/kg.
        code: exact match. name: FTS search."""
        c = self._conn()
        if name:
            return self.query_text(name, limit=limit)

        col_map = {
            'state': ('state_code', 'state_name'),
            'bundesland': ('state_code', 'state_name'),
            'district': ('district_code', 'district_name'),
            'bezirk': ('district_code', 'district_name'),
            'gemeinde': ('gemeinde_code', 'gemeinde_name'),
            'municipality': ('gemeinde_code', 'gemeinde_name'),
            'kg': ('kg_code', 'kg_name'),
        }
        if level not in col_map:
            return []
        code_col, name_col = col_map[level]
        q = f'SELECT * FROM kg WHERE {code_col}=?'
        params = [code]
        if processed_only:
            q += ' AND processed=1'
        q += f' ORDER BY {name_col} LIMIT ?'
        params.append(limit)
        return [self._kg_summary(r) for r in c.execute(q, params)]

    def query_text(self, q, limit=20):
        """Full-text search across KG/gemeinde/district/state names."""
        c = self._conn()
        # Tokenize query for FTS prefix matching
        terms = q.strip().split()
        if not terms:
            return []
        fts_q = ' '.join(t + '*' for t in terms)
        try:
            rows = c.execute('''
                SELECT k.* FROM fts_kg f
                JOIN kg k ON k.kg_code = f.kg_code
                WHERE fts_kg MATCH ?
                ORDER BY rank LIMIT ?
            ''', (fts_q, limit)).fetchall()
            return [self._kg_summary(r) for r in rows]
        except Exception:
            # Fallback to LIKE
            like = f'%{q}%'
            rows = c.execute('''
                SELECT * FROM kg
                WHERE kg_name LIKE ? OR gemeinde_name LIKE ?
                   OR district_name LIKE ? OR state_name LIKE ?
                LIMIT ?
            ''', (like, like, like, like, limit)).fetchall()
            return [self._kg_summary(r) for r in rows]

    # ════════════════════════════════════════════════════════════════
    # Query: aggregates
    # ════════════════════════════════════════════════════════════════

    def aggregate_district(self, district_code):
        """Aggregate landscape stats for a Bezirk."""
        return self._aggregate('district_code', district_code)

    def aggregate_gemeinde(self, gemeinde_code):
        """Aggregate landscape stats for a Gemeinde."""
        return self._aggregate('gemeinde_code', gemeinde_code)

    def aggregate_state(self, state_code_or_name):
        """Aggregate landscape stats for a Bundesland."""
        # Accept either code or name
        if len(state_code_or_name) <= 2 and state_code_or_name.isdigit():
            return self._aggregate('state_code', state_code_or_name)
        # Reverse lookup
        for code, name in STATE_CODES.items():
            if name.lower() == state_code_or_name.lower():
                return self._aggregate('state_code', code)
        return self._aggregate('state_name', state_code_or_name)

    def aggregate_country(self):
        """Aggregate landscape stats for all of Austria."""
        return self._aggregate(None, None)

    def _aggregate(self, col, val):
        c = self._conn()
        where = f'WHERE {col}=?' if col else ''
        params = [val] if col else []

        # Basic counts
        r = c.execute(f'''
            SELECT COUNT(*) as total_kgs,
                   SUM(CASE WHEN processed=1 THEN 1 ELSE 0 END) as processed_kgs,
                   ROUND(SUM(total_area_sqm)/1e6, 2) as total_area_km2,
                   SUM(parcel_count) as total_parcels,
                   SUM(building_count) as total_buildings,
                   SUM(tree_count) as total_trees,
                   ROUND(SUM(tree_canopy_sqm)/1e6, 2) as tree_canopy_km2,
                   ROUND(SUM(tree_stem_volume_m3), 0) as total_stem_volume_m3,
                   SUM(new_building_count) as total_new_buildings,
                   SUM(infrastructure_count) as total_infrastructure,
                   ROUND(AVG(CASE WHEN processed=1 THEN elevation_mean_m END), 1) as avg_elevation_m,
                   ROUND(AVG(CASE WHEN processed=1 THEN slope_mean_deg END), 2) as avg_slope_deg,
                   ROUND(AVG(CASE WHEN processed=1 THEN ndvi_mean END), 3) as avg_ndvi,
                   ROUND(AVG(CASE WHEN processed=1 THEN quality_score END), 2) as avg_quality_score,
                   ROUND(SUM(CASE WHEN processed=1 THEN net_volume_change_m3 ELSE 0 END), 0) as total_net_volume_change_m3,
                   MIN(min_lon) as min_lon, MIN(min_lat) as min_lat,
                   MAX(max_lon) as max_lon, MAX(max_lat) as max_lat
            FROM kg {where}
        ''', params).fetchone()
        agg = dict(r)

        # Admin name
        if col == 'state_code':
            agg['name'] = STATE_CODES.get(val, val)
            agg['level'] = 'state'
        elif col == 'district_code':
            agg['name'] = DISTRICT_NAMES.get(val, val)
            agg['level'] = 'district'
        elif col == 'gemeinde_code':
            row = c.execute('SELECT gemeinde_name FROM kg WHERE gemeinde_code=? LIMIT 1', [val]).fetchone()
            agg['name'] = row[0] if row else val
            agg['level'] = 'gemeinde'
        elif col == 'state_name':
            agg['name'] = val
            agg['level'] = 'state'
        else:
            agg['name'] = 'Austria'
            agg['level'] = 'country'

        # Top landcover types (processed KGs only)
        lc_where = f'WHERE k.{col}=?' if col else ''
        lc_params = [val] if col else []
        lc_rows = c.execute(f'''
            SELECT lc.object_type,
                   ROUND(SUM(lc.area_sqm)/1e6, 3) as total_area_km2,
                   ROUND(AVG(lc.fraction), 4) as avg_fraction,
                   SUM(lc.n_objects) as total_objects
            FROM kg_landcover lc
            JOIN kg k ON k.kg_code = lc.kg_code
            {lc_where}
            GROUP BY lc.object_type
            ORDER BY total_area_km2 DESC
            LIMIT 10
        ''', lc_params).fetchall()
        agg['top_landcover'] = [dict(r) for r in lc_rows]

        # Hansen loss summary
        h_rows = c.execute(f'''
            SELECT h.loss_year, SUM(h.loss_pixels) as total_pixels
            FROM kg_hansen h
            JOIN kg k ON k.kg_code = h.kg_code
            {lc_where}
            GROUP BY h.loss_year
            ORDER BY h.loss_year
        ''', lc_params).fetchall()
        agg['hansen_loss_by_year'] = {r['loss_year']: r['total_pixels'] for r in h_rows}

        return agg

    # ════════════════════════════════════════════════════════════════
    # Query: rankings and filters
    # ════════════════════════════════════════════════════════════════

    def query_type_ranking(self, object_type, metric='area', limit=20):
        """Rank KGs by a specific object type."""
        c = self._conn()
        order = {'area': 'lc.area_sqm', 'fraction': 'lc.fraction',
                 'count': 'lc.n_objects', 'height': 'lc.height_mean'}
        col = order.get(metric, 'lc.area_sqm')
        rows = c.execute(f'''
            SELECT k.kg_code, k.kg_name, k.gemeinde_name, k.district_name, k.state_name,
                   lc.area_sqm, lc.fraction, lc.n_objects,
                   lc.height_min, lc.height_max, lc.height_mean, lc.height_p90
            FROM kg_landcover lc
            JOIN kg k ON k.kg_code = lc.kg_code
            WHERE lc.object_type=?
            ORDER BY {col} DESC
            LIMIT ?
        ''', (object_type, limit)).fetchall()
        return [dict(r) for r in rows]

    def query_hansen_loss(self, year_from=None, year_to=None, min_loss=0, limit=50):
        """Find KGs with forest loss in a year range."""
        c = self._conn()
        where_parts = []
        params = []
        if year_from:
            where_parts.append('h.loss_year >= ?')
            params.append(year_from)
        if year_to:
            where_parts.append('h.loss_year <= ?')
            params.append(year_to)
        where = ('WHERE ' + ' AND '.join(where_parts)) if where_parts else ''
        rows = c.execute(f'''
            SELECT k.kg_code, k.kg_name, k.gemeinde_name, k.state_name,
                   SUM(h.loss_pixels) as total_loss
            FROM kg_hansen h
            JOIN kg k ON k.kg_code = h.kg_code
            {where}
            GROUP BY h.kg_code
            HAVING total_loss >= ?
            ORDER BY total_loss DESC
            LIMIT ?
        ''', params + [min_loss, limit]).fetchall()
        return [dict(r) for r in rows]

    def query_new_buildings(self, min_count=1, limit=50):
        """Find KGs with new (uncadastred) buildings."""
        c = self._conn()
        rows = c.execute('''
            SELECT kg_code, kg_name, gemeinde_name, district_name, state_name,
                   new_building_count, total_area_sqm, quality_score
            FROM kg WHERE processed=1 AND new_building_count >= ?
            ORDER BY new_building_count DESC LIMIT ?
        ''', (min_count, limit)).fetchall()
        return [dict(r) for r in rows]

    def query_processed(self, limit=500, offset=0):
        """List all processed KGs."""
        c = self._conn()
        rows = c.execute('''
            SELECT * FROM kg WHERE processed=1
            ORDER BY kg_name LIMIT ? OFFSET ?
        ''', (limit, offset)).fetchall()
        return [self._kg_summary(r) for r in rows]

    # ════════════════════════════════════════════════════════════════
    # Lazy-load detail from light GPKG (Zenodo or local)
    # ════════════════════════════════════════════════════════════════

    def query_buildings(self, kg_code, bbox=None, limit=500, offset=0):
        """Return height-enriched building footprints for a KG.
        Lazy-loads the light GPKG from Zenodo if needed."""
        gpkg = self._resolve_gpkg(kg_code, 'light')
        if not gpkg:
            return None
        return _query_gpkg_layer(gpkg, 'buildings', bbox=bbox, limit=limit, offset=offset)

    def query_new_buildings_detail(self, kg_code, bbox=None, limit=500, offset=0):
        """Return detected new building footprints for a KG."""
        gpkg = self._resolve_gpkg(kg_code, 'light')
        if not gpkg:
            return None
        return _query_gpkg_layer(gpkg, 'new_buildings', bbox=bbox, limit=limit, offset=offset)

    def query_infrastructure_detail(self, kg_code, bbox=None, limit=500, offset=0):
        """Return detected infrastructure for a KG."""
        gpkg = self._resolve_gpkg(kg_code, 'light')
        if not gpkg:
            return None
        return _query_gpkg_layer(gpkg, 'infrastructure', bbox=bbox, limit=limit, offset=offset)

    def query_segments_detail(self, kg_code, bbox=None, type_filter=None,
                              limit=500, offset=0):
        """Return segment polygons for a KG, optionally filtered by type."""
        gpkg = self._resolve_gpkg(kg_code, 'light')
        if not gpkg:
            return None
        return _query_gpkg_layer(gpkg, 'segments', bbox=bbox,
                                 type_filter=type_filter, limit=limit, offset=offset)

    def query_segment_points(self, kg_code, bbox=None, type_filter=None,
                             limit=500, offset=0):
        """Return segment centroid points for a KG."""
        gpkg = self._resolve_gpkg(kg_code, 'light')
        if not gpkg:
            # Try full GPKG which has segment_points
            gpkg = self._resolve_gpkg(kg_code, 'full')
            if not gpkg:
                return None
        return _query_gpkg_layer(gpkg, 'segment_points', bbox=bbox,
                                 type_filter=type_filter, limit=limit, offset=offset)

    def gpkg_layers(self, kg_code, variant='light'):
        """List available vector layers in a KG's GPKG."""
        gpkg = self._resolve_gpkg(kg_code, variant)
        if not gpkg:
            return None
        try:
            import fiona
            layers = fiona.listlayers(gpkg)
            result = []
            for l in layers:
                try:
                    with fiona.open(gpkg, layer=l) as src:
                        result.append({
                            'name': l,
                            'count': len(src),
                            'geometry_type': src.schema.get('geometry'),
                            'properties': list(src.schema.get('properties', {}).keys()),
                        })
                except Exception:
                    result.append({'name': l, 'count': 0, 'geometry_type': None, 'properties': []})
            return result
        except Exception as e:
            log.warning('gpkg_layers %s %s: %s', kg_code, variant, e)
            return None

    def _resolve_gpkg(self, kg_code, variant='light'):
        """Find or download a GPKG for a KG. Returns local path or None.

        Search order:
        1. Local processor output  (data/austria_processor/gpkg/)
        2. GPKG cache              (data/gpkg_cache/)
        3. Download from Zenodo    (cached for future use)
        """
        # 1. Local processor output
        local = Path(f'data/austria_processor/gpkg/{kg_code}_{variant}.gpkg')
        if local.exists() and local.stat().st_size > 0:
            return str(local)

        # 2. GPKG cache
        cache = get_gpkg_cache()
        cached = cache.get(kg_code, variant)
        if cached:
            return cached

        # 3. Zenodo URL from index
        c = self._conn()
        col = f'zenodo_{variant}_gpkg_url'
        row = c.execute(f'SELECT {col} FROM kg WHERE kg_code=?', (kg_code,)).fetchone()
        if not row or not row[0]:
            return None
        url = row[0]

        # Download in background? No — caller needs it now. Download synchronously.
        path = cache.download(kg_code, variant, url)
        return path

    # ════════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════════

    def _kg_summary(self, row):
        """Convert a kg Row to a summary dict (no landcover/hansen detail)."""
        d = dict(row)
        # Round float columns for cleaner output
        for k, v in d.items():
            if isinstance(v, float):
                if 'pct' in k or 'fraction' in k:
                    d[k] = round(v, 1)
                elif 'stories' in k:
                    d[k] = round(v, 1)
                else:
                    d[k] = round(v, 2)
        d['_links'] = self._build_links(d)
        return d


def _zenodo_url(entry):
    """Build Zenodo download URL from manifest entry."""
    if not entry or not entry.get('bucket_url'):
        return None
    return f"{entry['bucket_url']}/{entry['filename']}"


# ════════════════════════════════════════════════════════════════════════
# GPKG lazy-load cache
# ════════════════════════════════════════════════════════════════════════

GPKG_CACHE_DIR = 'data/gpkg_cache'
GPKG_CACHE_MAX_BYTES = 1_000_000_000  # 1 GB


class GpkgCache:
    """LRU disk cache for light/full GPKGs downloaded from Zenodo."""

    def __init__(self, cache_dir=GPKG_CACHE_DIR, max_bytes=GPKG_CACHE_MAX_BYTES):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def _path(self, kg_code, variant):
        return self.cache_dir / f'{kg_code}_{variant}.gpkg'

    def get(self, kg_code, variant='light'):
        """Return cached path if it exists, else None. Touch for LRU."""
        p = self._path(kg_code, variant)
        if p.exists() and p.stat().st_size > 0:
            try:
                p.touch()  # update mtime for LRU
            except OSError:
                pass
            return str(p)
        return None

    def download(self, kg_code, variant, url):
        """Download a GPKG from Zenodo. Returns local path or None."""
        import urllib.request
        with self._lock:
            # Double-check after lock
            existing = self.get(kg_code, variant)
            if existing:
                return existing
            self._evict_if_needed()
            dest = self._path(kg_code, variant)
            tmp = dest.with_suffix('.gpkg.tmp')
            try:
                log.info('gpkg_cache: downloading %s %s from %s', kg_code, variant, url[:80])
                req = urllib.request.Request(url, headers={'User-Agent': 'srtm-lidar/1.0'})
                with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, 'wb') as f:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp.rename(dest)
                log.info('gpkg_cache: cached %s (%d MB)', dest.name, dest.stat().st_size // (1024*1024))
                return str(dest)
            except Exception as e:
                log.warning('gpkg_cache: download failed %s %s: %s', kg_code, variant, e)
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                return None

    def _evict_if_needed(self):
        """Remove oldest files until under max_bytes."""
        files = sorted(self.cache_dir.glob('*.gpkg'), key=lambda p: p.stat().st_mtime)
        total = sum(f.stat().st_size for f in files)
        while total > self.max_bytes and files:
            victim = files.pop(0)
            sz = victim.stat().st_size
            try:
                victim.unlink()
                total -= sz
                log.info('gpkg_cache: evicted %s (%d MB)', victim.name, sz // (1024*1024))
            except OSError:
                pass

    def size_bytes(self):
        """Total cache size."""
        return sum(f.stat().st_size for f in self.cache_dir.glob('*.gpkg'))

    def count(self):
        """Number of cached files."""
        return len(list(self.cache_dir.glob('*.gpkg')))


_gpkg_cache = None
_gpkg_cache_lock = threading.Lock()


def get_gpkg_cache() -> GpkgCache:
    global _gpkg_cache
    if _gpkg_cache is None:
        with _gpkg_cache_lock:
            if _gpkg_cache is None:
                _gpkg_cache = GpkgCache()
    return _gpkg_cache


def _query_gpkg_layer(gpkg_path, layer_name, bbox=None, type_filter=None,
                      limit=500, offset=0):
    """Read features from a GPKG layer. Returns {layer, count, features} or None.

    bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84
    type_filter: list of object type strings to include
    """
    import fiona
    try:
        layers = fiona.listlayers(gpkg_path)
        if layer_name not in layers:
            return None
    except Exception:
        return None

    features = []
    total = 0
    try:
        with fiona.open(gpkg_path, layer=layer_name) as src:
            if bbox:
                it = src.filter(bbox=(bbox[0], bbox[1], bbox[2], bbox[3]))
            else:
                it = iter(src)

            for f in it:
                if type_filter:
                    ft = f.get('properties', {}).get('type')
                    if ft not in type_filter:
                        continue
                total += 1
                if total <= offset:
                    continue
                if len(features) >= limit:
                    continue  # keep counting total
                feat = {
                    'type': 'Feature',
                    'geometry': dict(f.get('geometry', {})) if f.get('geometry') else None,
                    'properties': dict(f.get('properties', {})),
                }
                features.append(feat)
    except Exception as e:
        log.warning('_query_gpkg_layer %s/%s: %s', gpkg_path, layer_name, e)
        return None

    return {
        'layer': layer_name,
        'total': total,
        'offset': offset,
        'limit': limit,
        'features': features,
    }


# ════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ════════════════════════════════════════════════════════════════════════

_index = None
_index_lock = threading.Lock()


def get_index() -> SearchIndex:
    """Get or create the global search index singleton."""
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = SearchIndex()
    return _index


def init_index():
    """Initialize and build the search index on startup."""
    idx = get_index()
    idx.build()
    return idx

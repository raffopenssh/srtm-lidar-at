"""Single source of truth for data-source licences and attribution text.

Every place that redistributes or displays derived data (API responses,
Zenodo deposit metadata, GeoPackage ``gpkg_metadata``, KG JSON summaries,
the web UIs, ``llm.txt``) must pull its attribution wording from here so
the obligations stay consistent.  See ``docs/attributions.md`` for the
legal assessment behind each entry.

Summary of the obligations we carry:

* **BEV** (ALS DTM/DSM, DOP orthophoto, cadastre) — Open Government Data,
  CC BY 4.0.  Attribution "Datenquelle: BEV – Bundesamt für Eich- und
  Vermessungswesen" (+ Stichtag), licence link, mark as modified, no
  implied endorsement.
* **Copernicus Sentinel-1/-2** — free, full and open (EU Reg. 377/2014,
  Delegated Reg. 1159/2013).  Modified data must carry "Contains modified
  Copernicus Sentinel data [year]".
* **ESA WorldCover** — CC BY 4.0, cite Zanaga et al. + DOI.
* **Hansen GFC** — CC BY 4.0, cite Hansen et al. 2013 (Science).
* **OpenStreetMap** — ODbL 1.0.  "© OpenStreetMap contributors".  OSM-derived
  fields merged into a redistributed database trigger share-alike, so they
  are kept in separate layers / flagged (see ``OSM_DERIVED_LAYERS``).
* **Our derivatives** (Zenodo, API) — CC BY 4.0, compatible with all of the
  above except that OSM-derived layers remain ODbL.
"""
from __future__ import annotations

import json
from typing import Iterable, Optional

# === SECTION: Licence registry ===

CC_BY_4 = "CC BY 4.0"
CC_BY_4_URL = "https://creativecommons.org/licenses/by/4.0/"
ODBL_1 = "ODbL 1.0"
ODBL_1_URL = "https://opendatacommons.org/licenses/odbl/1-0/"
COPERNICUS_LEGAL_URL = (
    "https://sentinels.copernicus.eu/documents/247904/690755/"
    "Sentinel_Data_Legal_Notice"
)

# What this project publishes its own derivatives under.
OUTPUT_LICENSE = CC_BY_4
OUTPUT_LICENSE_URL = CC_BY_4_URL
OUTPUT_LICENSE_ZENODO_ID = "cc-by-4.0"

# Free-form "modified" statement required by CC BY 4.0 §3(a)(1)(B) and the
# Copernicus legal notice.  Kept short so it fits GPKG / Zenodo fields.
MODIFICATION_NOTE = (
    "Bearbeitet / modified: re-projected (EPSG:3035), tiled, resampled, "
    "segmented, classified and enriched by the srtm-lidar-at pipeline. "
    "The original providers are not responsible for and do not endorse "
    "this derivative work."
)

# Each source: id → dict(name, provider, license, license_url, attribution,
# citation?, url, note?, share_alike?).  ``attribution`` is the exact
# one-line credit that must appear wherever the derived data is shown.
SOURCES: dict[str, dict] = {
    "bev_als": {
        "name": "BEV ALS DTM/DSM (Airborne Laserscanning, 1 m)",
        "provider": "BEV – Bundesamt für Eich- und Vermessungswesen",
        "license": CC_BY_4,
        "license_url": CC_BY_4_URL,
        "attribution": (
            "Datenquelle: BEV – Bundesamt für Eich- und Vermessungswesen, "
            "ALS DGM/DOM 1 m, Stichtag {stichtag}, CC BY 4.0, bearbeitet"
        ),
        "stichtag": "2022-09-15 / 2023-09-15 / 2024-09-15 (mosaic epochs; "
                    "real flight years per block in meta.acquisition)",
        "url": "https://data.bev.gv.at/",
        "note": "Open Government Data. data.bev.gv.at metadata: 'Für dieses "
                "Produkt gilt die Standardlizenz CC-BY-4.0'.",
    },
    "bev_dop": {
        "name": "BEV DOP RGBI Orthophoto (0.2 m)",
        "provider": "BEV – Bundesamt für Eich- und Vermessungswesen",
        "license": CC_BY_4,
        "license_url": CC_BY_4_URL,
        "attribution": (
            "Datenquelle: BEV – Bundesamt für Eich- und Vermessungswesen, "
            "Orthophoto (DOP RGBI), Stichtag {stichtag}, CC BY 4.0, bearbeitet"
        ),
        "stichtag": "2022-01-28 / 2022-10-27 / 2024-06-25 / 2025-04-15 "
                    "(series; operate flight year in ORTHO_EPOCH tag)",
        "url": "https://data.bev.gv.at/",
    },
    "bev_cadastre": {
        "name": "BEV Kataster (parcels, building footprints, KG boundaries)",
        "provider": "BEV – Bundesamt für Eich- und Vermessungswesen",
        "license": CC_BY_4,
        "license_url": CC_BY_4_URL,
        "attribution": (
            "Datenquelle: BEV – Bundesamt für Eich- und Vermessungswesen, "
            "Kataster (INSPIRE / kataster.bev.gv.at), CC BY 4.0, bearbeitet"
        ),
        "url": "https://kataster.bev.gv.at/",
        "note": "BEV confirmed to the OSM community (2023) that the "
                "kataster.bev.gv.at service is CC BY 4.0. Owner data is not "
                "part of the open dataset and is not in this API; EZ numbers "
                "are public cadastre attributes.",
    },
    "basemap_at": {
        "name": "basemap.at Orthofoto (web-map display only)",
        "provider": "basemap.at / geoland.at (Austrian Länder + BEV)",
        "license": CC_BY_4,
        "license_url": CC_BY_4_URL,
        "attribution": "Grundkarte: basemap.at, CC BY 4.0",
        "url": "https://basemap.at/",
        "note": "Display-only background tiles in query.html; not redistributed.",
    },
    "copernicus_s2": {
        "name": "Copernicus Sentinel-2 L2A (NDVI composites, phenology)",
        "provider": "European Union / ESA / Copernicus, via Copernicus Data "
                    "Space Ecosystem (openEO)",
        "license": "Copernicus Sentinel data — free, full and open access",
        "license_url": COPERNICUS_LEGAL_URL,
        "attribution": "Contains modified Copernicus Sentinel data {year}",
        "url": "https://dataspace.copernicus.eu/",
        "note": "Regulation (EU) No 377/2014, Commission Delegated Regulation "
                "(EU) No 1159/2013. Modified products must say 'Contains "
                "modified Copernicus Sentinel data [Year]'.",
    },
    "copernicus_s1": {
        "name": "Copernicus Sentinel-1 IW GRD (VV/VH backscatter)",
        "provider": "European Union / ESA / Copernicus, via Copernicus Data "
                    "Space Ecosystem (openEO)",
        "license": "Copernicus Sentinel data — free, full and open access",
        "license_url": COPERNICUS_LEGAL_URL,
        "attribution": "Contains modified Copernicus Sentinel data {year}",
        "url": "https://dataspace.copernicus.eu/",
    },
    "esa_worldcover": {
        "name": "ESA WorldCover 10 m 2021 v200",
        "provider": "ESA WorldCover consortium (VITO et al.)",
        "license": CC_BY_4,
        "license_url": CC_BY_4_URL,
        "attribution": "© ESA WorldCover project 2021 / Contains modified "
                       "Copernicus Sentinel data (2021) processed by ESA "
                       "WorldCover consortium",
        "citation": "Zanaga, D., Van De Kerchove, R., Daems, D., De Keersmaecker, "
                    "W., Brockmann, C., Kirches, G., Wevers, J., Cartus, O., "
                    "Santoro, M., Fritz, S., Lesiv, M., Herold, M., Tsendbazar, "
                    "N.E., Xu, P., Ramoino, F., Arino, O., 2022. ESA WorldCover "
                    "10 m 2021 v200. https://doi.org/10.5281/zenodo.7254221",
        "url": "https://esa-worldcover.org/",
    },
    "hansen_gfc": {
        "name": "Hansen Global Forest Change 2000–2024 v1.12",
        "provider": "Hansen/UMD/Google/USGS/NASA",
        "license": CC_BY_4,
        "license_url": CC_BY_4_URL,
        "attribution": "Hansen/UMD/Google/USGS/NASA Global Forest Change "
                       "2000–2024 v1.12, CC BY 4.0",
        "citation": "Hansen, M. C., P. V. Potapov, R. Moore, M. Hancher, S. A. "
                    "Turubanova, A. Tyukavina, D. Thau, S. V. Stehman, S. J. "
                    "Goetz, T. R. Loveland, A. Kommareddy, A. Egorov, L. Chini, "
                    "C. O. Justice, and J. R. G. Townshend. 2013. High-Resolution "
                    "Global Maps of 21st-Century Forest Cover Change. Science "
                    "342 (15 November): 850–53. "
                    "https://glad.earthengine.app/view/global-forest-change",
        "url": "https://storage.googleapis.com/earthenginepartners-hansen/"
               "GFC-2024-v1.12/download.html",
    },
    "osm": {
        "name": "OpenStreetMap (roads, paths, water, landcover, power polygons)",
        "provider": "OpenStreetMap contributors",
        "license": ODBL_1,
        "license_url": ODBL_1_URL,
        "attribution": "© OpenStreetMap contributors, ODbL 1.0",
        "url": "https://www.openstreetmap.org/copyright",
        "share_alike": True,
        "note": "ODbL is NOT CC BY. Layers/fields derived from OSM are flagged "
                "(see osm_derived_layers) and remain ODbL; a redistributed "
                "database that merges them is subject to ODbL share-alike. "
                "OSM is also used as RF training label source — a trained "
                "model is not a 'derivative database' under the ODbL "
                "Community Guideline on produced works.",
    },
    "austria_power": {
        "name": "austria-power API (wind turbines, masts, substations)",
        "provider": "austria-power.exe.xyz aggregating Austro Control "
                    "obstacle data, IG Windkraft and OSM",
        "license": "Mixed — per-feature `source` attribute: austrocontrol / "
                   "igwindkraft (open data), osm_power (ODbL)",
        "license_url": "https://austria-power.exe.xyz:8000/",
        "attribution": "Infrastructure: Austro Control, IG Windkraft, "
                       "© OpenStreetMap contributors (ODbL)",
        "url": "https://austria-power.exe.xyz:8000/",
        "share_alike": True,
        "note": "OSM-derived features carry source=osm_power and fall under "
                "ODbL; verify upstream licence for austrocontrol / igwindkraft "
                "features before commercial redistribution.",
    },
}

# Layers / JSON keys in our products that contain OSM-derived (ODbL) data.
# Everything else in our outputs is CC BY 4.0.
OSM_DERIVED_LAYERS = ("infrastructure",)

# Sources baked into the standard KG products (JSON / full+light GPKG).
KG_PRODUCT_SOURCES = (
    "bev_als", "bev_dop", "bev_cadastre", "copernicus_s2", "copernicus_s1",
    "esa_worldcover", "hansen_gfc", "osm", "austria_power",
)


# === SECTION: Renderers ===

def _fmt(src: dict, year: Optional[int] = None, stichtag: Optional[str] = None) -> str:
    s = src["attribution"]
    return s.format(
        year=year or "2022–2025",
        stichtag=stichtag or src.get("stichtag", ""),
    )


def attribution_lines(sources: Iterable[str] = KG_PRODUCT_SOURCES,
                      year: Optional[int] = None) -> list[str]:
    """One credit line per source, ready for display."""
    return [_fmt(SOURCES[k], year) for k in sources if k in SOURCES]


def attribution_text(sources: Iterable[str] = KG_PRODUCT_SOURCES,
                     year: Optional[int] = None, sep: str = "\n") -> str:
    """Plain-text attribution block (for GPKG metadata, logs, footers)."""
    lines = attribution_lines(sources, year)
    lines.append(f"Licence of this derivative: {OUTPUT_LICENSE} ({OUTPUT_LICENSE_URL}). "
                 f"OSM-derived layers ({', '.join(OSM_DERIVED_LAYERS)}) remain ODbL.")
    lines.append(MODIFICATION_NOTE)
    return sep.join(lines)


def attribution_short() -> str:
    """Compact one-liner (map footers, API meta)."""
    return ("Datenquelle: BEV – Bundesamt für Eich- und Vermessungswesen (CC BY 4.0, "
            "bearbeitet) · Contains modified Copernicus Sentinel data 2022–2025 · "
            "© ESA WorldCover 2021 · Hansen/UMD/Google/USGS/NASA GFC · "
            "© OpenStreetMap contributors (ODbL)")


def attribution_dict(sources: Iterable[str] = KG_PRODUCT_SOURCES,
                     year: Optional[int] = None) -> dict:
    """Structured block for JSON responses / KG summaries."""
    return {
        "license": OUTPUT_LICENSE,
        "license_url": OUTPUT_LICENSE_URL,
        "modification_note": MODIFICATION_NOTE,
        "osm_derived_layers": list(OSM_DERIVED_LAYERS),
        "osm_derived_license": ODBL_1,
        "sources": {
            k: {
                "name": SOURCES[k]["name"],
                "provider": SOURCES[k]["provider"],
                "license": SOURCES[k]["license"],
                "license_url": SOURCES[k]["license_url"],
                "attribution": _fmt(SOURCES[k], year),
                **({"citation": SOURCES[k]["citation"]} if SOURCES[k].get("citation") else {}),
                **({"share_alike": True} if SOURCES[k].get("share_alike") else {}),
                "url": SOURCES[k]["url"],
            }
            for k in sources if k in SOURCES
        },
    }


def zenodo_description_footer(year: Optional[int] = None) -> str:
    """HTML block appended to every Zenodo deposit description."""
    items = "".join(f"<li>{_fmt(SOURCES[k], year)}</li>" for k in KG_PRODUCT_SOURCES)
    cites = "".join(f"<li>{SOURCES[k]['citation']}</li>"
                    for k in KG_PRODUCT_SOURCES if SOURCES[k].get("citation"))
    return (
        "<p><b>Licence &amp; attribution.</b> This derivative dataset is "
        f"published under <a href=\"{OUTPUT_LICENSE_URL}\">{OUTPUT_LICENSE}</a>. "
        f"{MODIFICATION_NOTE}</p>"
        f"<p>Source data:</p><ul>{items}</ul>"
        f"<p>References:</p><ul>{cites}</ul>"
        "<p>Layers derived from OpenStreetMap ("
        f"{', '.join(OSM_DERIVED_LAYERS)}) are licensed "
        f"<a href=\"{ODBL_1_URL}\">{ODBL_1}</a> (share-alike) and are kept in "
        "separate layers.</p>"
    )


def zenodo_keywords() -> list[str]:
    return ["CC-BY-4.0", "BEV", "Copernicus", "Sentinel-2", "Sentinel-1",
            "ESA WorldCover", "Hansen GFC", "OpenStreetMap"]


def write_gpkg_metadata(conn, year: Optional[int] = None,
                        layers: Optional[Iterable[str]] = None) -> None:
    """Write standard ``gpkg_metadata`` / ``gpkg_metadata_reference`` rows.

    Idempotent: replaces any prior rows with our ``mime_type='text/plain'``
    dataset-scope entry.  ``conn`` is an open sqlite3 connection to a GPKG.
    Follows OGC GeoPackage 1.3 §2.4 (Metadata extension).
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS gpkg_metadata ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "md_scope TEXT NOT NULL DEFAULT 'dataset', "
        "md_standard_uri TEXT NOT NULL, "
        "mime_type TEXT NOT NULL DEFAULT 'text/xml', "
        "metadata TEXT NOT NULL DEFAULT '')"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS gpkg_metadata_reference ("
        "reference_scope TEXT NOT NULL, table_name TEXT, column_name TEXT, "
        "row_id_value INTEGER, "
        "timestamp DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')), "
        "md_file_id INTEGER NOT NULL, md_parent_id INTEGER)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS gpkg_extensions ("
        "table_name TEXT, column_name TEXT, extension_name TEXT NOT NULL, "
        "definition TEXT NOT NULL, scope TEXT NOT NULL, "
        "CONSTRAINT ge_tce UNIQUE (table_name, column_name, extension_name))"
    )
    for tbl in ("gpkg_metadata", "gpkg_metadata_reference"):
        conn.execute(
            "INSERT OR IGNORE INTO gpkg_extensions "
            "(table_name, column_name, extension_name, definition, scope) "
            "VALUES (?, NULL, 'gpkg_metadata', "
            "'http://www.geopackage.org/spec120/#extension_metadata', 'read-write')",
            (tbl,),
        )
    # Drop previous rows we wrote (re-runs / GPKG reuse).
    old = [r[0] for r in conn.execute(
        "SELECT id FROM gpkg_metadata WHERE md_standard_uri = ?",
        (OUTPUT_LICENSE_URL,)).fetchall()]
    if old:
        q = ",".join("?" * len(old))
        conn.execute(f"DELETE FROM gpkg_metadata_reference WHERE md_file_id IN ({q})", old)
        conn.execute(f"DELETE FROM gpkg_metadata WHERE id IN ({q})", old)

    # Row 1: human-readable text; Row 2: structured JSON.
    cur = conn.execute(
        "INSERT INTO gpkg_metadata (md_scope, md_standard_uri, mime_type, metadata) "
        "VALUES ('dataset', ?, 'text/plain', ?)",
        (OUTPUT_LICENSE_URL, attribution_text(year=year)),
    )
    txt_id = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO gpkg_metadata (md_scope, md_standard_uri, mime_type, metadata) "
        "VALUES ('dataset', ?, 'application/json', ?)",
        (OUTPUT_LICENSE_URL, json.dumps(attribution_dict(year=year), ensure_ascii=False)),
    )
    json_id = cur.lastrowid
    conn.execute(
        "INSERT INTO gpkg_metadata_reference (reference_scope, md_file_id) "
        "VALUES ('geopackage', ?)", (txt_id,))
    conn.execute(
        "INSERT INTO gpkg_metadata_reference (reference_scope, md_file_id) "
        "VALUES ('geopackage', ?)", (json_id,))
    # Per-layer ODbL flag for OSM-derived layers.
    for lyr in (layers or ()):
        if lyr in OSM_DERIVED_LAYERS:
            cur = conn.execute(
                "INSERT INTO gpkg_metadata (md_scope, md_standard_uri, mime_type, metadata) "
                "VALUES ('dataset', ?, 'text/plain', ?)",
                (ODBL_1_URL, f"Layer '{lyr}' contains data derived from OpenStreetMap "
                             f"({SOURCES['osm']['attribution']}); share-alike applies."),
            )
            conn.execute(
                "INSERT INTO gpkg_metadata_reference (reference_scope, table_name, md_file_id) "
                "VALUES ('table', ?, ?)", (lyr, cur.lastrowid))

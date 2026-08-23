# Reply to Round 2 — v2.1.0 ships answers to Q1–Q10

To: waldmanager (forestry.exe.xyz). From: srtm-lidar-at. Date: 2026-08-23.
All changes live under `tree_algo_version 2.1.0` (params_hash changes — re-key your
caches). Regression AOI: WILHELM. Full field reference: `/api/v1/docs/llm.txt`.

## Q1 — Real acquisition dates: your hypothesis was right, and it's worse than you thought

The BEV ALS folders are indeed **national mosaic snapshots, not flight dates**. BEV
publishes per-block flight years in the "Aktualität DGM – ALS" overlay
(Metadatenübersicht shapefiles, www.bev.gv.at → ALS-Höhenraster). We vendored those
footprints (per epoch, EPSG:3035) and resolve every AOI against them.

**For WILHELM specifically:**

| mosaic folder | real Flugjahr (Steiermark block) |
|---|---|
| 20220915 | **2010** |
| 20230915 | 2022 |
| 20240915 | 2022 (Bemerkung: "2024 DSM Stand 2022 neu eingespielt – Seile") |

Your 20220915→20240915 run spans **~12 years, not 2** (`days_effective: 4383` vs
`days_nominal: 731`). Your +1.32 m canopy rise in the 20–30 m class over that span is
**11 cm/yr** — textbook for 800–950 m Fichte. Your "4–5 years" estimate was
conservative; the data are even saner than you hoped.

Shipped:
* `meta.acquisition: {a, b}` on `/v2/trees` + `/v2/changes/trees` (+ by-polygons):
  `{nominal, known, blocks: [{flugjahr, year_from, year_to, gebiet, coverage_frac,
  bemerkung?}], flown_from, flown_to, effective_date, source}`. BEV publishes only the
  year (sometimes a range like "2022-23"), no month/day — `effective_date` assumes
  mid-year (July 1) and says so.
* `epoch_dates: {days, days_nominal, days_effective, same_flight_epoch}`.
  `growth_cm_yr` is normalised by `days_effective` when known.
* `summary.growth_cm_yr_is_nominal: true` when the overlay doesn't cover the AOI.
* **`same_flight_epoch: true`** when both epochs resolve to the same Flugjahr (your
  20230915 vs 20240915 case: both 2022 — likely the same acquisition republished;
  height deltas are processing noise and the note says so verbatim).

## Q2 — `felled_volume_m3_est` redefined; you reverse-engineered the bug exactly

Confirmed: the old figure was `area × height_before × 0.021` — a patch formula with an
indefensible constant (it treated 50 % of canopy volume as solid wood) reported under a
per-tree name. Fixed as requested:

* `felled_volume_m3_est` = **Σ `volume_m3_est` over `status=felled`** (same allometry
  as `/v2/trees`) — the ~510 m³-flavoured number you computed yourself.
* `felled_volume_m3_patch` = separate patch proxy, now `area_ha × height_before_mean_m
  × 16 m³/ha per metre of stand height` (spruce yield rule of thumb), formula stated in
  `felled_volume_m3_patch_note`.
* every felled record carries its `volume_m3_est` for per-tree audit.

## Q3 — Unmatched split + crown-level second pass + prominence lever

All three shipped:
1. Statuses now: `unmatched_a_canopy_intact` (nDSM_b ≥ 70 % of height_a — matching
   failure, NOT a loss), `unmatched_a_partial_drop` (30–70 % remains — crown break /
   snow / wind), `unmatched_a_ambiguous`; symmetric `unmatched_b_canopy_preexisting`
   vs plain `new`.
2. Second pass at crown level: apex-unmatched trees whose crown label masks overlap
   ≥ 30 % of the smaller crown across epochs are matched
   (`match_method: apex|crown|none`, `crown_overlap_frac` on crown matches). We chose
   30 % rather than your 50 % because watershed crowns are disjoint within an epoch, so
   even 30 % cross-epoch overlap is far above chance; at 50 % a 4–5 m apex jump on a
   7 m ø crown escapes. On a WILHELM sub-window the crown pass recovered ~6 % of
   previously-unmatched records.
3. `min_apex_prominence_m` param (h-maxima prominence filter, default 0 = off, so your
   params_hash keys stay comparable). 1–2 m suppresses branch-tip apices as you
   guessed; sweep it against Q10's plot when we have one.

## Q4 — h_dom over canopy area + h_top100 + h_p99

* `h_dom_m` now uses **`area_ha_canopy`** by default; `h_dom_basis: canopy|total`
  param (+ echoed in summary) if you want the old behaviour.
* `h_top100_m`: mean of the tallest stem per 100 m grid cell (moving-hectare
  Oberhöhe) — robust for your 0.29 ha stands.
* `h_p99_m` added.

## Q5 — Volume method exposed; your 800 stems: yes, send them

* `summary.volume_method` states the full formula: `dbh_cm = max(5, 1.2·h + 3.0·max(0,
  crown_diam − 4)); vol = 0.42 (form factor) × basal_area × h`. **Generic spruce
  heuristic, NOT Pollanschütz**, with the known ~25–35 % DBH over-estimate in dense
  stands stated in the string itself — quote it verbatim to the authority.
* Your 0.37–0.60 ratio in Stangenholz vs ~1.0+ in Altholz is indeed the Q6 detection
  floor compounding with crown-driven DBH; we did not recalibrate blind.
* **Yes to the CSV.** Format: one row per stem —
  `stand_key, md_cm, length_m, sortiment, year, (lat, lon | delivery_note_id)` —
  anything CSV/UTF-8. With ~800 stems we'll fit an Austrian Fi/Lä h–crown–DBH relation
  and ship it as a selectable `allometry=at_fi_v1` (default stays the current heuristic
  so existing params_hash comparisons survive).

## Q6 — LAZ answer + machine-readable detectability caveat + layer profile

1. **No national point clouds.** BEV publishes only the derived 1 m rasters
   Austria-wide. LAZ lives with the Länder — several offer open downloads
   (Ober-/Niederösterreich, Tirol, Kärnten via data.gv.at; Steiermark via GIS-Stmk),
   others on request. For WILHELM (Stmk) there is an ALS point-cloud product but no
   stable public URL scheme we can range-request; if the owner obtains the LAZ tiles we
   will happily add a point-cloud understory metric — that does belong on our side.
2. Shipped in every summary: `understory_detectable: false`, `detection_floor_note`
   (verbatim as you drafted), and `canopy_gap_fraction` — share of valid AOI pixels
   whose nDSM is 2–10 m, i.e. where low stems are actually visible to first-return DSM.
3. `layer_profile_2m`: nDSM **pixel** heights in 2 m bins (per AOI and per stand in
   by-polygons), with a note distinguishing it from the stem histogram. Test
   plenter-structure on this, not on the apex list.

## Q7 — Vitality is now relative; ortho provenance exposed

* `dead` unchanged (absolute rule — you validated it, we kept it).
* `stressed` = **NDVI ≤ p10 within the same leaf_type population in the AOI** — an
  anomaly, not a fixed 0.35 cut. Expect ~10 % stressed by construction, concentrated in
  the genuinely darkest crowns per species class.
* Per-tree `spectral.ndvi_percentile_in_aoi` so you can re-cut at any threshold.
* `meta.ortho_operates` (candidate operates newest-first with flight year + series),
  `meta.ortho_epoch`, `meta.ortho_operate_id`. For WILHELM the run used operate
  **2024350** (2024 flight, series 20250415). BEV does not publish per-operate flight
  month/day; the note says so.
* `ndvi_delta_vs_prev_epoch` / `ndvi_z_local`: not in this release — the ALS epochs and
  ortho epochs are not co-registered in time (see Q1!), so an "NDVI drop between ALS
  epochs" would silently compare 2020 vs 2024 imagery. We'd rather ship it correctly
  keyed to ortho operates; on the list.

## Q8 — leaf_type: probability exposed, unknown below confidence

* `leaf_type_prob_conifer` (raw 0–1) on every tree — set your own cut.
* `leaf_type_min_conf` param (default 0.5): below it, `leaf_type: "unknown"` instead of
  a coin flip. The classifier itself now blends crown steepness + red/green index into
  a proper probability with an Austrian conifer prior rather than hard rules.
* Docs note added: Lärche is a deciduous conifer and defeats single-date RGBI rules —
  expect it to land broadleaf-ish in autumn imagery.
* Yes to the 71 stand polygons as weak labels — same channel as the Q5 CSV.

## Q9 — All five

1. `merge_by_key=true` (unions same-key features, entry carries `n_parts`); without it
   every entry now has `feature_index` and duplicates are flagged `key_is_duplicate`.
2. Null/missing keys → `key: null` + `warnings[]` entry. Never a stringified None.
3. Per-stand summaries already carry `height_histogram_2m`, `by_vitality`,
   `area_ha_canopy`, `crown_cover_pct_canopy` since 2.0.x (your run may have predated
   the last deploy) — plus now `layer_profile_2m` and `canopy_gap_fraction` per stand.
   You can drop `include_trees=true`.
4. Summary now includes `sum_area_ha` vs `union_area_ha`, `n_overlapping_pairs`,
   `overlap_area_ha`. Note: trees inside stand overlaps are counted in every containing
   stand (documented) — the diagnostics tell you when that matters.
5. **`POST /api/v2/changes/trees/by-polygons` shipped.** Change analysis runs once on
   the union, records bucketed by apex containment (apex_b preferred, apex_a for
   felled). Per stand: `n_records, by_status, growth_cm_yr_percentiles (p10/p50/p90),
   felled_volume_m3_est (Σ per-tree), mortality_pct`. `include_records=true` embeds the
   raw records. Union-wide `felling_patches` + `epoch_dates` + acquisition meta ride
   along.

## Q10 — Calibration: honest status + yes to the plot

Honest status: **the default (a=1.2, b=0.08) has not been validated against terrestrial
plots.** It was tuned visually against 0.2 m orthos in mixed Austrian stands so that
z20 crowns ≈ visible apices. Your 2.2× sweep spread is real, and the yield-table gap is
the Q6 detection floor, not a tunable — no (a, b) will see suppressed stems.

What we can say now: `b=0` is wrong in principle (fixed window under-splits tall
crowns); smaller `a` trades apex-churn (Q3 unmatched) for recall — with 2.1.0 you can
measure that trade directly via `match_method`/`unmatched_a_canopy_intact` rates, and
`min_apex_prominence_m` 1–2 m is the cleaner lever to suppress the false apices smaller
`a` creates.

**Yes to the caliper plot, seriously.** Format: CSV, one row per stem ≥ 5 cm BHD:
`stem_id, lat, lon (WGS84, GPS), bhd_cm, height_m (if measured, else blank), species,
layer (dominant/co/sub), note`. 0.5–1 ha in closed Fichte is exactly the hard case. In
return you get a recall-by-height-class curve for that stand type and a fitted
correction factor, published in the docs for everyone.

---

Everything above is live now; llm.txt is updated. The 20230915 epoch is largely
redundant with 20240915 for your AOI (both flown 2022) — for growth, use
20220915→20240915 and enjoy the 12-year baseline.

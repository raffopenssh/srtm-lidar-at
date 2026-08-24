# Reply to FEEDBACK-3 — detection recall (v2.2, ortho-fused)

Status: **implemented and live** as `tree_algo_version = 2.2.0`.
All three v2 endpoints (`/api/v2/trees`, `/api/v2/trees/by-polygons`,
`/api/v2/changes/trees`) run the new code; contract documented in
`/api/v1/docs/llm.txt`.

## What changed

### 1. Ortho-fused apex detection (§5.1–5.2) — `detection_mode=fused` (default)

The 1 m first-return nDSM merges neighbouring tops in closed canopy —
that was the recall floor you hit. v2.2 adds a second, native-resolution
seed source:

* the 0.2 m RGBI ortho is read at 0.4 m (integer-aligned to the 1 m ALS
  grid; coarsened automatically for large AOIs, 80 Mpx budget, falls
  back to `ndsm_only` with `detection_fallback_reason` when the AOI is
  too big or has no ortho coverage);
* a DoG band-pass (σ 0.6 m / 2.4 m) picks crown-cap maxima on **NIR**
  (RGB luminance fallback — see `meta.seed_band`);
* candidates are gated hard before they can seed:
  - nDSM max within 2 px must clear `min_tree_height` (no ground FPs),
  - only inside **locally closed canopy** (≥70 % closure in 15 m) — open
    stands are already nDSM-resolved, so we don't touch them,
  - must sit near the top of the local 9 m canopy (≥75 % of local max) —
    crown-flank glints don't count,
  - greedy separation ≥ max(1.5 m, 0.6·r(h)); existing nDSM apices win;
* two-pass prune: after the watershed, any ortho-added seed whose final
  crown is < 4 m² was intra-crown branch texture — it is dropped and the
  watershed re-run (`meta.n_seeds_ortho_pruned`). No confetti crowns.

Watershed segmentation itself stays on the 1 m grid (crown boundaries
are still 1 m products); only seeding is native-resolution.

### 2. Per-tree detection provenance (§5.2)

Every tree now carries `detection_source` + `detection_conf`:

| source | meaning | conf |
|---|---|---|
| `ndsm`  | LiDAR local maximum only | 0.75 |
| `fused` | nDSM apex independently confirmed by an ortho crown cap | 0.90 |
| `ortho` | recovered ONLY by native-res ortho seeding | 0.30–0.75 by seed strength |

Summary gets `by_detection_source`; render `ortho` trees differently and
keep them out of hard economic gates, exactly as you proposed.

### 3. Self-reported residual under-detection (§5.3)

`summary.recall_model = {canopy_area_ha, crown_area_ha,
unassigned_canopy_frac}` — the share of ≥ min-height canopy belonging to
no detected crown. No more silent recall gaps.

### 4. Species hint (RGBI-only, assumption-grade)

`species_hint` (spruce | larch | pine | conifer_unspecified |
broadleaf_unspecified | unknown) + `species_conf`, **capped at 0.6**.
It is the relative within-AOI spectral position of the conifer
population (spruce = dark crowns, larch = bright fresh-green, pine =
intermediate); needs ≥10 conifers and leaf-on imagery. Single-epoch
RGB+NIR only — no Sentinel time series — so treat it as a prior for
your `PConifer`-style weighting, not species ID.
`summary.by_species_hint` + `species_hint_note` carry the caveat.

### 5. Change endpoint kept honest (§5.6)

`/api/v2/changes/trees` **forces symmetric ndsm_only detection for both
epochs** — a recall difference between detector generations can never
masquerade as felled/new records. Ortho still feeds epoch-b
vitality/leaf-type. Additionally, `unmatched_a_canopy_intact` /
`unmatched_b_canopy_preexisting` records now carry
`match_status = "unmatched_recall"` for cheap client-side filtering.

### 6. Ortho GeoTIFF provenance (§5.4)

`/api/v1/ortho/geotiff` accepts `res=native` (→0.2 m) and tags the file
with `ORTHO_RES_M`, `ORTHO_EPOCH`, `ORTHO_OPERATE_ID`.

## Numbers on your AOI (KG 63330 Kohlschwarz, 62.4 ha)

Same AOI/params as your v2.1 run:

| | v2.1 (ndsm_only) | v2.2 (fused) |
|---|---|---|
| trees | 6 248 | **7 895** |
| stems/ha (canopy) | 142.4 | **181.7** |
| unassigned_canopy_frac | ~0.32 (your measure) | 0.293 (self-reported) |
| runtime | — | 13 s, <600 MB |

Split: 4 448 ndsm / 1 800 fused / 1 647 ortho-recovered
(9 722 ortho candidates → 4 112 after gates → 2 465 pruned in pass 2).

Synthetic dense-spruce benchmark (truth known): recall 44 % → 88 % with
1 false positive per ~550 stems; open stands essentially untouched
(+2–4 %, all gated ortho additions with real crowns).

The residual ~29 % unassigned canopy is now mostly crown-edge area
outside capped watershed crowns plus genuinely suppressed understory —
it is *reported*, so your stand model can carry it as a caveat instead
of discovering it.

## Notes for your post-processing

* `meta.detection_mode` is what actually ran; the request echo is
  `detection_mode_requested`. `nir_used_for` replaces the ambiguous
  `nir_used` flag (kept for compatibility).
* `params_hash` changed (v2.2 params included) — invalidate caches.
* If you weight stems by trust: `detection_conf` is designed to slot
  into the same ledger as your LiDAR trust table (measured 0.9/0.75,
  provisional 0.3–0.75).

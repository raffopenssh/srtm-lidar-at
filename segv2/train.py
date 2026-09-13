#!/usr/bin/env python3
"""segv2/train.py — the model harness (HANDOVER step 1).

Loads `data/segv2/dataset/*.parquet` (built by build_dataset.py), applies the
row filter + class merges, runs GroupKFold-by-parent-KG CV for a set of
candidate models and writes `data/segv2/report.{md,json}`.

Models
  A  v1 baseline   deployed RF (data/best_model/rf_model.joblib) on FEATURE_KEYS
  B  RF relabelled same hyper-params as v1, trained on v2 labels, FEATURE_KEYS
  C  LGBM          LightGBM on FEATURE_KEYS
  D  LGBM+ctx      LightGBM on ALL_KEYS (v1 + V2_EXTRA_KEYS)
  E  LGBM+ctx+nb   D + second stage: mean OOF class-proba of the k nearest
                   segments (centroid kNN within the same kg/tile — adjacency
                   is not stored in the parquet, so kNN is the proxy)
  P  v1 pipeline   `v1_type` column = what the deployed processor wrote on the
                   same pixels (no training, reference for "what is on Zenodo")

Fleet-safety: nothing here is imported by the live stack. Read-only on
`data/best_model/`. Writes only under `data/segv2/`.

Usage
  python3 segv2/train.py                       # all models, 5 folds
  python3 segv2/train.py --models A,C,D --quick # 20 % row subsample
  python3 segv2/train.py --save-final D         # also fit D on all rows → data/segv2/models/
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from learned_classifier import FEATURE_KEYS, _downsample  # noqa: E402
from features import ALL_KEYS, V2_EXTRA_KEYS  # noqa: E402
from labels import MIN_NDVI as MIN_NDVI_SEG  # noqa: E402  segment-level replay of the label veto

log = logging.getLogger("segv2.train")

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data/segv2/dataset"
OUT_DIR = ROOT / "data/segv2"
MODEL_DIR = OUT_DIR / "models"
REPORT_SUFFIX = ""
V1_MODEL = ROOT / "data/best_model/rf_model.joblib"
V1_META = ROOT / "data/best_model/rf_meta.json"

# --- row filter / class policy ------------------------------------------------
MIN_PURITY = 0.6
MIN_KG_FRAC = 0.5
MIN_CLASS_ROWS = 300          # classes below this are dropped from train+eval
MERGE = {                    # applied to y, v1_type and v1-model predictions
    "excavation": "earthwork",
    "fill": "earthwork",
    "hedge": "shrub",        # 280 rows in the 143-KG build → too thin alone
}
MAX_PER_CLASS_TRAIN = 0       # 0 = no cap (operator OK'd full machine use); --cap N to limit
MAX_NDVI_SEG = {"earthwork": 0.45}  # cadastre 84 (Abbau/Halden/Deponie) that has re-greened is not earthwork
CLASS_WEIGHT = "sqrt"        # "balanced" (LGBM built-in) or "sqrt" (tempered: w_c ∝ (n_max/n_c)^0.5)
# ALL_KEYS minus the OSM/cadastre distance features — the honest "recognition
# only" ablation (dist_* share geometry with the road/rail/water/roof labels).
NODIST_KEYS = [k for k in ALL_KEYS if not k.startswith("dist_")]
KEY_CLASSES = ["tree", "roof", "grass", "crop", "water", "road"]  # promotion rule
PROMO_MACRO_F1_GAIN = 0.05
PROMO_MAX_KEY_LOSS = 0.02

LGBM_PARAMS = dict(
    n_estimators=400, learning_rate=0.08, num_leaves=63, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    class_weight="balanced", n_jobs=2, verbose=-1, random_state=42, max_bin=127,
)
RF_PARAMS = dict(n_estimators=200, max_depth=20, min_samples_leaf=5,
                 class_weight="balanced", n_jobs=2, random_state=42)
KNN_K = 8


# === SECTION: data ===

def load_dataset(quick: bool = False, seed: int = 0) -> pd.DataFrame:
    files = sorted(glob.glob(str(DATASET_DIR / "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files under {DATASET_DIR}")
    parts = []
    for f in files:
        d = pd.read_parquet(f)
        if len(d):
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["kg_parent"] = df["kg"].astype(str).str.split("-").str[0]
    df["y"] = df["y"].fillna("").replace(MERGE)
    df["v1_type"] = df["v1_type"].fillna("").replace(MERGE)
    # harmonics: 0 means "not available" for most KGs → NaN so GBMs treat it as missing
    hk = [k for k in ALL_KEYS if k.startswith("harm_")]
    miss = (df[hk].abs().sum(axis=1) == 0)
    df.loc[miss, hk] = np.nan
    if quick:
        df = df.sample(frac=0.2, random_state=seed).reset_index(drop=True)
    log.info("loaded %d rows from %d files (%d parent KGs)", len(df), len(files), df.kg_parent.nunique())
    return df


def select_labelled(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    m = (df["y"] != "") & (df["y_purity"] >= MIN_PURITY) & (df["kg_frac"] >= MIN_KG_FRAC)
    # replay labels.MIN_NDVI / MAX_NDVI_SEG on the segment mean BEV NDVI so parquets
    # built before the veto existed get the same treatment (alpine "grass" on scree, ...)
    vet = np.zeros(len(df), dtype=bool)
    for ty, mn in MIN_NDVI_SEG.items():
        vet |= (df["y"] == ty) & (df["ndvi_mean"] < mn)
    for ty, mx in MAX_NDVI_SEG.items():
        vet |= (df["y"] == ty) & (df["ndvi_mean"] > mx)
    n_vet = int((m & vet).sum())
    lab = df[m & ~vet].copy()
    counts = lab["y"].value_counts()
    keep = sorted(c for c, n in counts.items() if n >= MIN_CLASS_ROWS)
    dropped = {c: int(n) for c, n in counts.items() if n < MIN_CLASS_ROWS}
    lab = lab[lab["y"].isin(keep)].copy()
    lab["w"] = (lab["y_weight"].fillna(1.0) * lab["y_purity"]).astype(np.float32)
    if CLASS_WEIGHT == "sqrt":
        # tempered balancing: LGBM's "balanced" gives bare_soil (332) ~700× the weight of
        # tree (232k) which wrecks calibration on the majority classes; sqrt keeps the
        # ordering but caps the spread at ~26×.
        cw = {c: float(np.sqrt(counts.max() / counts[c])) for c in keep}
        lab["w"] = (lab["w"] * lab["y"].map(cw).astype(np.float32)).astype(np.float32)
    info = {
        "n_rows_all": int(len(df)), "n_rows_labelled": int(len(lab)),
        "classes": keep, "class_counts": {c: int(counts[c]) for c in keep},
        "dropped_classes": dropped, "merge": MERGE,
        "filter": {"min_purity": MIN_PURITY, "min_kg_frac": MIN_KG_FRAC, "min_class_rows": MIN_CLASS_ROWS},
        "n_parent_kgs": int(lab.kg_parent.nunique()),
        "n_ndvi_vetoed": n_vet, "min_ndvi_seg": MIN_NDVI_SEG, "max_ndvi_seg": MAX_NDVI_SEG,
        "class_weight": CLASS_WEIGHT,
        "label_sources": {k: int(v) for k, v in lab["y_src"].value_counts().items()},
    }
    log.info("labelled rows: %d (ndvi-vetoed %d), classes: %s, dropped: %s", len(lab), n_vet, keep, dropped)
    return lab, keep, info


def _X(df: pd.DataFrame, keys: list[str], nan_to_zero: bool) -> np.ndarray:
    X = df.reindex(columns=keys).to_numpy(dtype=np.float32)
    if nan_to_zero:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        X[~np.isfinite(X)] = np.nan
    return X


def _cap_train(idx: np.ndarray, y: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Cap each class at `cap` rows (indices into the fold's train set)."""
    out = []
    if not cap:
        return idx
    for c in np.unique(y):
        ii = idx[y == c]
        if len(ii) > cap:
            ii = rng.choice(ii, cap, replace=False)
        out.append(ii)
    return np.sort(np.concatenate(out))


# === SECTION: metrics ===

def ece_score(y_true: np.ndarray, proba: np.ndarray, classes: list[str], n_bins: int = 10) -> float:
    conf = proba.max(axis=1)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    acc = (pred == y_true).astype(float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def evaluate(y_true, y_pred, classes, proba=None, proba_classes=None, w=None) -> dict:
    from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        "per_class": {c: {"p": float(p[i]), "r": float(r[i]), "f1": float(f[i]), "n": int(s[i])}
                      for i, c in enumerate(classes)},
        "confusion": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "n": int(len(y_true)),
    }
    if w is not None:
        out["weighted_accuracy"] = float(np.average(np.asarray(y_true) == np.asarray(y_pred), weights=w))
    if proba is not None:
        out["ece"] = ece_score(np.asarray(y_true), proba, proba_classes)
        out["mean_conf"] = float(proba.max(axis=1).mean())
    return out


# === SECTION: models ===

class V1Baseline:
    """Deployed RF; predicts on the 66 FEATURE_KEYS with NaN→0 like the processor."""

    def __init__(self):
        import joblib
        self.rf = joblib.load(V1_MODEL)
        self.classes_ = [MERGE.get(c, c) for c in self.rf.classes_]

    def predict_proba(self, df):
        P = self.rf.predict_proba(_X(df, FEATURE_KEYS, True))
        # merge columns mapped to the same class (excavation/fill → earthwork)
        cls = sorted(set(self.classes_))
        out = np.zeros((len(df), len(cls)), dtype=np.float32)
        for j, c in enumerate(self.classes_):
            out[:, cls.index(c)] += P[:, j]
        self.classes_ = cls
        return out


def make_model(kind: str):
    if kind == "B":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**RF_PARAMS)
    import lightgbm as lgb
    params = dict(LGBM_PARAMS)
    if CLASS_WEIGHT == "sqrt":
        params["class_weight"] = None   # tempered weights are already in sample_weight
    return lgb.LGBMClassifier(**params)


MODEL_SPECS = {
    "A": ("v1 baseline (deployed RF, FEATURE_KEYS)", FEATURE_KEYS, True),
    "B": ("RF relabelled (v1 hyper-params, FEATURE_KEYS)", FEATURE_KEYS, True),
    "C": ("LGBM FEATURE_KEYS", FEATURE_KEYS, False),
    "D": ("LGBM ALL_KEYS (+context)", ALL_KEYS, False),
    "E": ("LGBM ALL_KEYS + neighbour OOF proba (kNN stacking)", ALL_KEYS, False),
    "F": ("LGBM ALL_KEYS minus dist_* (no OSM-geometry features)", NODIST_KEYS, False),
    "P": ("v1 pipeline output on Zenodo (v1_type column)", [], True),
}


def fit_predict_cv(kind, lab, classes, folds, all_df=None, extra=None, seed=0):
    """Return (oof_pred, oof_proba, fold_stats, importances). `extra` = extra
    feature matrix aligned with `lab` (model E)."""
    from sklearn.model_selection import GroupKFold
    _, keys, nan0 = MODEL_SPECS[kind]
    y = lab["y"].to_numpy()
    groups = lab["kg_parent"].to_numpy()
    X = _X(lab, keys, nan0)
    if extra is not None:
        X = np.hstack([X, extra.astype(np.float32)])
    w = lab["w"].to_numpy()
    oof_proba = np.zeros((len(lab), len(classes)), dtype=np.float32)
    rng = np.random.default_rng(seed)
    stats, imps = [], []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
        t0 = time.time()
        tr_c = _cap_train(tr, y[tr], MAX_PER_CLASS_TRAIN, rng)
        if kind == "B":
            Xtr, ytr = _downsample(X[tr_c], y[tr_c])
            wtr = None
        else:
            Xtr, ytr, wtr = X[tr_c], y[tr_c], w[tr_c]
        m = make_model(kind)
        m.fit(Xtr, ytr, sample_weight=wtr)
        P = m.predict_proba(X[te])
        for j, c in enumerate(m.classes_):
            oof_proba[te, classes.index(c)] = P[:, j]
        fpred = np.asarray(classes)[oof_proba[te].argmax(1)]
        f1 = evaluate(y[te], fpred, classes)["macro_f1"]
        stats.append({"fold": k, "n_train": int(len(Xtr)), "n_test": int(len(te)),
                      "test_kgs": int(len(set(groups[te]))), "macro_f1": f1, "seconds": round(time.time() - t0, 1)})
        if hasattr(m, "feature_importances_"):
            imps.append(np.asarray(m.feature_importances_, dtype=float))
        log.info("%s fold %d: train %d test %d macroF1=%.3f (%.0fs)", kind, k, len(Xtr), len(te), f1, time.time() - t0)
    names = list(keys) + ([f"nb_p_{c}" for c in classes] + ["nb_k"] if extra is not None else [])
    imp = {}
    if imps:
        v = np.mean(imps, axis=0)
        v = v / v.sum() if v.sum() else v
        imp = {n: float(x) for n, x in sorted(zip(names, v), key=lambda t: -t[1])}
    pred = np.asarray(classes)[oof_proba.argmax(1)]
    return pred, oof_proba, stats, imp


def knn_neighbour_proba(lab: pd.DataFrame, proba: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Mean OOF proba of the k nearest other segments in the same kg/tile
    (+ count actually found). Proxy for RAG adjacency."""
    from scipy.spatial import cKDTree
    out = np.zeros((len(lab), proba.shape[1] + 1), dtype=np.float32)
    xy = lab[["centroid_e", "centroid_n"]].to_numpy(dtype=float)
    for _, idx in lab.groupby(["kg", "tile"]).indices.items():
        if len(idx) < 2:
            continue
        kk = min(k, len(idx) - 1)
        tree = cKDTree(xy[idx])
        d, nn = tree.query(xy[idx], k=kk + 1)
        nn = nn[:, 1:]  # drop self
        out[idx, :-1] = proba[idx][nn].mean(axis=1)
        out[idx, -1] = kk
    return out


# === SECTION: report ===

def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def write_report(res: dict, info: dict, classes: list[str]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"report{REPORT_SUFFIX}.json").write_text(json.dumps({"info": info, "classes": classes, "models": res}, indent=1))
    order = [m for m in "ABCDEFP" if m in res]
    L = ["# segv2 harness report", "",
         f"generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} · {info['n_rows_labelled']:,} labelled "
         f"segments from {info['n_parent_kgs']} parent KGs · {info.get('folds')}-fold GroupKFold by parent KG",
         "", f"Filter: purity ≥ {MIN_PURITY}, kg_frac ≥ {MIN_KG_FRAC}, class ≥ {MIN_CLASS_ROWS} rows. "
         f"Merged: {MERGE}. Dropped: {info['dropped_classes']}. Train cap {MAX_PER_CLASS_TRAIN or 'none'}/class/fold. "
         f"Segment NDVI veto (min {info.get('min_ndvi_seg')}, max {info.get('max_ndvi_seg')}) removed {info.get('n_ndvi_vetoed', 0):,} rows. "
         f"Class weighting: {info.get('class_weight')}.", ""]
    L += ["## Class counts", "", "| class | rows |", "|---|---:|"]
    L += [f"| {c} | {n:,} |" for c, n in sorted(info["class_counts"].items(), key=lambda t: -t[1])]
    L += ["", "## Summary", "", "| model | macro-F1 | weighted-F1 | acc | ECE | mean conf | Δ macro-F1 vs A |", "|---|---:|---:|---:|---:|---:|---:|"]
    base = res.get("A", {}).get("metrics", {}).get("macro_f1")
    for m in order:
        mt = res[m]["metrics"]
        d = f"{mt['macro_f1'] - base:+.3f}" if base is not None and m != "A" else ""
        L.append(f"| {m} {res[m]['name']} | {mt['macro_f1']:.3f} | {mt['weighted_f1']:.3f} | {mt['accuracy']:.3f} | "
                 f"{_fmt(mt.get('ece', '—'))} | {_fmt(mt.get('mean_conf', '—'))} | {d} |")
    L += ["", "## Per-class F1", "", "| class | n | " + " | ".join(order) + " |", "|---|---:|" + "---:|" * len(order)]
    for c in classes:
        n = res[order[0]]["metrics"]["per_class"][c]["n"]
        L.append(f"| {c} | {n:,} | " + " | ".join(f"{res[m]['metrics']['per_class'][c]['f1']:.3f}" for m in order) + " |")
    if "promotion" in info:
        L += ["", "## Promotion check (README rule)", ""]
        for m, p in info["promotion"].items():
            L.append(f"- **{m}**: {'PASS' if p['pass'] else 'fail'} — Δmacro-F1 {p['macro_gain']:+.3f} (need ≥ +{PROMO_MACRO_F1_GAIN}), "
                     f"worst key-class Δ {p['worst_key_loss']:+.3f} on {p['worst_key_class']} (limit −{PROMO_MAX_KEY_LOSS}), "
                     f"ECE {p['ece']:.3f} vs A {p['ece_a']:.3f}")
    for m in order:
        r = res[m]
        L += ["", f"## {m} — {r['name']}", ""]
        if r.get("folds"):
            L += ["| fold | n_train | n_test | test KGs | macro-F1 | s |", "|---:|---:|---:|---:|---:|---:|"]
            L += [f"| {f['fold']} | {f['n_train']:,} | {f['n_test']:,} | {f['test_kgs']} | {f['macro_f1']:.3f} | {f['seconds']} |" for f in r["folds"]]
            L.append("")
        if r.get("importances"):
            top = list(r["importances"].items())[:20]
            L += ["Top-20 features: " + ", ".join(f"{k} {v:.3f}" for k, v in top), ""]
        L += ["Confusion (rows = truth, cols = pred):", "", "| | " + " | ".join(classes) + " |", "|---|" + "---:|" * len(classes)]
        for i, c in enumerate(classes):
            L.append(f"| **{c}** | " + " | ".join(str(v) for v in r["metrics"]["confusion"][i]) + " |")
    (OUT_DIR / f"report{REPORT_SUFFIX}.md").write_text("\n".join(L) + "\n")
    log.info("wrote %s and report%s.json", OUT_DIR / f"report{REPORT_SUFFIX}.md", REPORT_SUFFIX)


def promotion_check(res: dict, classes: list[str]) -> dict:
    if "A" not in res:
        return {}
    a = res["A"]["metrics"]
    out = {}
    for m in ("B", "C", "D", "E", "F"):
        if m not in res:
            continue
        mt = res[m]["metrics"]
        deltas = {c: mt["per_class"][c]["f1"] - a["per_class"][c]["f1"] for c in KEY_CLASSES if c in classes}
        worst_c = min(deltas, key=deltas.get)
        gain = mt["macro_f1"] - a["macro_f1"]
        out[m] = {"macro_gain": gain, "worst_key_class": worst_c, "worst_key_loss": deltas[worst_c],
                  "ece": mt.get("ece", 1.0), "ece_a": a.get("ece", 1.0),
                  "pass": bool(gain >= PROMO_MACRO_F1_GAIN and deltas[worst_c] >= -PROMO_MAX_KEY_LOSS
                               and mt.get("ece", 1.0) <= a.get("ece", 1.0) + 1e-9)}
    return out


# === SECTION: main ===

def main():
    global MAX_PER_CLASS_TRAIN, CLASS_WEIGHT, REPORT_SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="A,B,C,D,F,P")
    ap.add_argument("--class-weight", default=CLASS_WEIGHT, choices=["balanced", "sqrt"])
    ap.add_argument("--report-suffix", default="", help="write report<suffix>.md/json instead of report.md")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="20%% row subsample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-final", default="", help="model letter(s) to refit on all rows → data/segv2/models/")
    ap.add_argument("--cap", type=int, default=MAX_PER_CLASS_TRAIN)
    a = ap.parse_args()
    MAX_PER_CLASS_TRAIN = a.cap
    CLASS_WEIGHT = a.class_weight
    REPORT_SUFFIX = a.report_suffix
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("lightgbm").setLevel(logging.WARNING)

    df = load_dataset(a.quick, a.seed)
    lab, classes, info = select_labelled(df)
    info["folds"] = a.folds
    y = lab["y"].to_numpy()
    w = lab["w"].to_numpy()
    models = [m.strip().upper() for m in a.models.split(",") if m.strip()]
    res: dict = {}
    oof_D = None

    for m in models:
        name = MODEL_SPECS[m][0]
        t0 = time.time()
        if m == "A":
            v1 = V1Baseline()
            P = v1.predict_proba(lab)
            pred = np.asarray(v1.classes_)[P.argmax(1)]
            met = evaluate(y, pred, classes, P, v1.classes_, w)
            met["unpredictable_classes"] = sorted(set(classes) - set(v1.classes_))
            res[m] = {"name": name, "metrics": met}
        elif m == "P":
            pred = lab["v1_type"].to_numpy()
            met = evaluate(y, pred, classes, w=w)
            met["frac_empty_or_unclassified"] = float(np.isin(pred, ["", "unclassified"]).mean())
            res[m] = {"name": name, "metrics": met}
        elif m == "E":
            if oof_D is None:
                log.info("E needs D's OOF proba — running D first")
                _, oof_D, _, _ = fit_predict_cv("D", lab, classes, a.folds, seed=a.seed)
            nb = knn_neighbour_proba(lab, oof_D)
            pred, P, folds, imp = fit_predict_cv("E", lab, classes, a.folds, extra=nb, seed=a.seed)
            met = evaluate(y, pred, classes, P, classes, w)
            met["note"] = ("neighbour features are kNN(k=%d) means of stage-1 OOF proba within kg/tile; "
                           "mild stacking leak (stage-1 folds ≠ stage-2 train/test split)" % KNN_K)
            res[m] = {"name": name, "metrics": met, "folds": folds, "importances": imp}
        else:
            pred, P, folds, imp = fit_predict_cv(m, lab, classes, a.folds, seed=a.seed)
            if m == "D":
                oof_D = P
            met = evaluate(y, pred, classes, P, classes, w)
            res[m] = {"name": name, "metrics": met, "folds": folds, "importances": imp}
        res[m]["seconds"] = round(time.time() - t0, 1)
        log.info("%s %s: macroF1=%.3f wF1=%.3f acc=%.3f ece=%s (%.0fs)", m, name, res[m]["metrics"]["macro_f1"],
                 res[m]["metrics"]["weighted_f1"], res[m]["metrics"]["accuracy"],
                 _fmt(res[m]["metrics"].get("ece", "—")), time.time() - t0)
        info["promotion"] = promotion_check(res, classes)
        write_report(res, info, classes)  # incremental — survives a kill mid-run

    for m in [s.strip().upper() for s in a.save_final.split(",") if s.strip()]:
        if m not in ("B", "C", "D", "F"):
            log.warning("--save-final only supports B/C/D (E needs stage-1 at inference); skipping %s", m)
            continue
        import joblib
        _, keys, nan0 = MODEL_SPECS[m]
        rng = np.random.default_rng(a.seed)
        idx = _cap_train(np.arange(len(lab)), y, MAX_PER_CLASS_TRAIN, rng)
        mdl = make_model(m)
        mdl.fit(_X(lab.iloc[idx], keys, nan0), y[idx], sample_weight=None if m == "B" else w[idx])
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(mdl, MODEL_DIR / f"model_{m}.joblib")
        (MODEL_DIR / f"model_{m}.meta.json").write_text(json.dumps({
            "model": m, "classes": list(map(str, mdl.classes_)), "feature_keys": keys, "nan_to_zero": nan0,
            "n_train": int(len(idx)), "merge": MERGE, "class_weight": CLASS_WEIGHT,
            "min_ndvi_seg": MIN_NDVI_SEG, "max_ndvi_seg": MAX_NDVI_SEG, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cv_metrics": res.get(m, {}).get("metrics", {}).get("macro_f1"),
        }, indent=1))
        log.info("saved final %s → %s", m, MODEL_DIR / f"model_{m}.joblib")

    log.info("FINISHED")


if __name__ == "__main__":
    main()

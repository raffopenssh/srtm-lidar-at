"""segv2 model v2 — inference wrapper around the LightGBM ship candidate.

Mirrors ``learned_classifier.LearnedClassifier`` for the v2 pipeline:

    m = ModelV2.load()                 # data/segv2/models/model_G.joblib (+ .meta.json)
    types, conf, proba = m.predict(df) # df = segv2.features.extract() output

* Feature order comes from the meta (``feature_keys``), never from the caller.
* LightGBM sees NaN as "missing" — no NaN→0 fill (``nan_to_zero`` is False for
  every LGBM model; the RF variants B get the v1 fill).
* The train-time NDVI vetoes (``min_ndvi_seg`` / ``max_ndvi_seg`` in the meta)
  are applied post-hoc as *class masks* on the probability matrix: a segment
  whose real-NIR ``ndvi_mean`` sits below a vegetation class's floor cannot be
  that class, above a sealed class's ceiling cannot be that class; the argmax
  is taken over the remaining classes.  Rows without real NIR (NaN) are exempt,
  exactly as in ``train.select_labelled``.
* ``report()`` writes gain/split importances + a LightGBM ``pred_contrib``
  (TreeSHAP) summary over a sample → ``data/segv2/report_model_v2.md``.

Fleet-safety: importing this module does not touch v1 (``data/best_model``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("segv2.model_v2")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "data/segv2/models"
DEFAULT_MODEL = os.environ.get("SEG_V2_MODEL", "G")          # letter or path

# classes the v2 model never predicts (rule-only in the product taxonomy)
RULE_ONLY = {"earthwork", "excavation", "fill", "path", "bare_soil", "tree_loss", "construction",
             "wind_turbine", "substation", "solar_panel", "mast", "fence", "wall", "bridge",
             "greenhouse", "hedge"}
MANMADE = {"road", "path", "parking", "roof", "wall", "fence", "mast", "greenhouse", "solar_panel",
           "bridge", "excavation", "fill", "construction", "substation", "wind_turbine", "rail"}


def _resolve(model: str | os.PathLike) -> tuple[Path, Path]:
    p = Path(model)
    if p.suffix == ".joblib" and p.exists():
        return p, p.with_suffix(".meta.json")
    p = MODEL_DIR / f"model_{model}.joblib"
    return p, MODEL_DIR / f"model_{model}.meta.json"


class ModelV2:
    def __init__(self, model, meta: dict, path: Path):
        self.model = model
        self.meta = meta
        self.path = path
        self.classes = [str(c) for c in model.classes_]
        self.feature_keys = list(meta["feature_keys"])
        self.nan_to_zero = bool(meta.get("nan_to_zero", False))
        self.min_ndvi = {k: float(v) for k, v in (meta.get("min_ndvi_seg") or {}).items()}
        self.max_ndvi = {k: float(v) for k, v in (meta.get("max_ndvi_seg") or {}).items()}
        self.merge = dict(meta.get("merge") or {})
        self.model_hash = hashlib.sha1(path.read_bytes()).hexdigest()[:12] if path.exists() else ""
        self._ci = {c: i for i, c in enumerate(self.classes)}

    # --- loading ---------------------------------------------------------------
    @classmethod
    def load(cls, model: str | os.PathLike | None = None) -> "ModelV2":
        import joblib
        mp, metap = _resolve(model or DEFAULT_MODEL)
        if not mp.exists():
            raise FileNotFoundError(f"v2 model not found: {mp}")
        meta = json.loads(metap.read_text()) if metap.exists() else {}
        mdl = joblib.load(mp)
        if "feature_keys" not in meta:
            # RF/LGBM both expose feature_names_in_ when fitted on a DataFrame; we fit on
            # ndarray, so the meta is the only source of truth.
            raise RuntimeError(f"{metap} missing feature_keys — cannot order features safely")
        m = cls(mdl, meta, mp)
        log.info("ModelV2 loaded %s: %d classes, %d features, hash=%s, cv_macroF1=%s",
                 mp.name, len(m.classes), len(m.feature_keys), m.model_hash, meta.get("cv_metrics"))
        return m

    # --- inference ----------------------------------------------------------------
    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.feature_keys).to_numpy(dtype=np.float32)
        X[~np.isfinite(X)] = np.nan
        if self.nan_to_zero:
            X = np.nan_to_num(X, nan=0.0)
        return X

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if len(df) == 0:
            return np.zeros((0, len(self.classes)), np.float32)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = self.model.predict_proba(self.matrix(df))
        return np.asarray(P, dtype=np.float32)

    def veto_mask(self, df: pd.DataFrame) -> np.ndarray:
        """bool (n, n_classes): True where the class is vetoed for the row."""
        n = len(df)
        vet = np.zeros((n, len(self.classes)), bool)
        if "ndvi_mean" not in df or n == 0:
            return vet
        nd = df["ndvi_mean"].to_numpy(dtype=np.float32)
        fin = np.isfinite(nd)
        for c, v in self.min_ndvi.items():
            if c in self._ci:
                vet[:, self._ci[c]] |= fin & (nd < v)
        for c, v in self.max_ndvi.items():
            if c in self._ci:
                vet[:, self._ci[c]] |= fin & (nd > v)
        return vet

    def predict(self, df: pd.DataFrame, *, apply_veto: bool = True):
        """→ (types: np.ndarray[str], confidence: np.ndarray[f32], proba: np.ndarray).

        ``proba`` is the raw model output (before vetoes) so callers can keep
        second-best classes; ``types``/``confidence`` are after the vetoes.
        """
        P = self.predict_proba(df)
        if len(P) == 0:
            return np.array([], dtype=object), np.zeros(0, np.float32), P
        Q = P.copy()
        if apply_veto:
            vet = self.veto_mask(df)
            Q[vet] = 0.0
            # a fully-vetoed row (cannot happen with disjoint veg/sealed sets, but be safe)
            dead = Q.sum(1) <= 0
            Q[dead] = P[dead]
        idx = Q.argmax(1)
        conf = Q[np.arange(len(Q)), idx]
        types = np.asarray(self.classes, dtype=object)[idx]
        return types, conf.astype(np.float32), P

    def top2(self, P: np.ndarray):
        """(second-best class, its proba) per row from a raw proba matrix."""
        if len(P) == 0:
            return np.array([], dtype=object), np.zeros(0, np.float32)
        o = np.argsort(-P, axis=1)
        j = o[:, 1] if P.shape[1] > 1 else o[:, 0]
        return np.asarray(self.classes, dtype=object)[j], P[np.arange(len(P)), j]

    # --- explainability ------------------------------------------------------------
    def importances(self) -> pd.DataFrame:
        b = self.model.booster_
        gain = b.feature_importance("gain"); split = b.feature_importance("split")
        d = pd.DataFrame({"feature": self.feature_keys, "gain": gain, "split": split})
        d["gain_frac"] = d["gain"] / max(d["gain"].sum(), 1e-9)
        return d.sort_values("gain", ascending=False).reset_index(drop=True)

    def shap_summary(self, df: pd.DataFrame, n: int = 5000, seed: int = 0) -> pd.DataFrame:
        """Mean |SHAP| per feature (TreeSHAP via LightGBM pred_contrib), overall and
        per class, over a random sample of ``df``."""
        if len(df) > n:
            df = df.sample(n=n, random_state=seed)
        X = self.matrix(df)
        C = self.model.booster_.predict(X, pred_contrib=True)  # (n, n_classes*(F+1))
        F = len(self.feature_keys); K = len(self.classes)
        C = C.reshape(len(X), K, F + 1)[:, :, :F]
        out = pd.DataFrame({"feature": self.feature_keys, "mean_abs_shap": np.abs(C).mean((0, 1))})
        for k, c in enumerate(self.classes):
            out[f"shap_{c}"] = np.abs(C[:, k, :]).mean(0)
        return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def report(self, df: pd.DataFrame | None = None, out: Path | None = None) -> Path:
        out = out or (ROOT / "data/segv2/report_model_v2.md")
        imp = self.importances()
        L = [f"# model v2 — {self.path.name}", "",
             f"hash `{self.model_hash}` · trained `{self.meta.get('trained_at')}` · n_train {self.meta.get('n_train'):,} · "
             f"CV macro-F1 {self.meta.get('cv_metrics')}", "",
             f"classes ({len(self.classes)}): {', '.join(self.classes)}", "",
             f"merge: `{self.merge}` · rule-only: `{sorted(self.meta.get('drop_classes') or [])}`", "",
             f"NDVI vetoes — min: `{self.min_ndvi}` · max: `{self.max_ndvi}`", "",
             "## Feature importance (gain)", "", "| # | feature | gain % | splits |", "|--:|---|--:|--:|"]
        for i, r in imp.head(40).iterrows():
            L.append(f"| {i+1} | {r.feature} | {100*r.gain_frac:.2f} | {int(r.split)} |")
        groups = {"dist_*": "dist_", "nb_*": "nb_", "harm_*": "harm_", "sar_*": "sar_", "esa_*": "esa_",
                  "hansen_*": "hansen_", "glcm/texture": "glcm_", "ndvi_*": "ndvi_"}
        L += ["", "### Gain by feature family", "", "| family | gain % |", "|---|--:|"]
        for name, pre in groups.items():
            L.append(f"| {name} | {100*imp.loc[imp.feature.str.startswith(pre), 'gain_frac'].sum():.1f} |")
        if df is not None and len(df):
            try:
                sh = self.shap_summary(df)
                L += ["", f"## Mean |SHAP| (TreeSHAP, {min(len(df), 5000)} sampled segments)", "",
                      "| # | feature | mean abs SHAP | top class |", "|--:|---|--:|---|"]
                cc = [c for c in sh.columns if c.startswith("shap_")]
                for i, r in sh.head(40).iterrows():
                    top = max(cc, key=lambda c: r[c])[5:]
                    L.append(f"| {i+1} | {r.feature} | {r.mean_abs_shap:.3f} | {top} |")
                L += ["", "### Top-5 features per class (mean |SHAP|)", ""]
                for c in cc:
                    t = sh.nlargest(5, c)
                    L.append(f"* **{c[5:]}**: " + ", ".join(f"{f} ({v:.2f})" for f, v in zip(t.feature, t[c])))
            except Exception as e:  # noqa: BLE001
                L += ["", f"_SHAP summary failed: {e}_"]
        out.write_text("\n".join(L) + "\n")
        log.info("model v2 report → %s", out)
        return out


# --- process-wide singleton (thread-safe) ------------------------------------------
_LOCK = threading.Lock()
_SINGLETON: ModelV2 | None = None


def get_model(model: str | None = None) -> ModelV2:
    global _SINGLETON
    with _LOCK:
        if _SINGLETON is None or (model and Path(_SINGLETON.path) != _resolve(model)[0]):
            _SINGLETON = ModelV2.load(model)
        return _SINGLETON


if __name__ == "__main__":
    import argparse, glob
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--shap-files", type=int, default=8, help="parquets to sample for SHAP")
    a = ap.parse_args()
    m = ModelV2.load(a.model)
    files = sorted(glob.glob(str(ROOT / "data/segv2/dataset/*.parquet")))
    rng = np.random.default_rng(0)
    pick = list(rng.choice(files, size=min(a.shap_files, len(files)), replace=False)) if files else []
    df = pd.concat([pd.read_parquet(f) for f in pick], ignore_index=True) if pick else None
    if df is not None:
        df = df[df["y"].fillna("") != ""]
    p = m.report(df)
    print(p.read_text()[:3000])

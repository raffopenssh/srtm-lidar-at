"""Learned classifier: Random Forest trained on cadastre ground truth.

Replaces the hand-tuned decision tree in classify_object() with a
data-driven Random Forest that learns from cadastre land-use labels.

Architecture:
  1. Feature extraction: same per-segment features as before
     (height, slope, roughness, shape, NDVI, texture, harmonics, SAR)
  2. Training: cadastre parcel codes provide noisy labels
  3. Prediction: RF predicts type + probability per segment
  4. Disagreement detection: segments where RF disagrees with cadastre
     are flagged as likely cadastre errors

The classifier falls back to the rule-based system when no trained
model is available, so there's zero risk of regression.

Usage::

    from learned_classifier import LearnedClassifier
    clf = LearnedClassifier.load()       # load trained model
    pred = clf.predict(feature_dict)     # returns (type, conf)
    clf.train(features, labels)          # retrain from new data
    disagreements = clf.find_disagreements(features, cadastre_labels)
"""
from __future__ import annotations

import logging
import pathlib
import json
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_DIR = pathlib.Path("/tmp/learned_classifier")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "rf_model.joblib"
META_PATH = MODEL_DIR / "rf_meta.json"

# Best model from curve evaluation (preferred over live training model)
BEST_MODEL_DIR = pathlib.Path("data/best_model")
BEST_MODEL_PATH = BEST_MODEL_DIR / "rf_model.joblib"
BEST_META_PATH = BEST_MODEL_DIR / "rf_meta.json"

# Features used by the RF.  Order matters (must match training).
# These are the per-segment features from extract_object_features().
FEATURE_KEYS = [
    # Height & terrain (DTM/DSM 1m)
    "h_mean", "h_max", "h_std", "h_p90",
    "slope_mean", "slope_std",
    "dsm_roughness", "dtm_roughness",
    # Shape
    "compactness", "elongation", "solidity", "extent", "area",
    # DSM edges
    "dsm_edge_strength",
    # Spectral (BEV ortho 1m)
    "ndvi_mean", "ndvi_std", "brightness_mean", "nir_mean",
    "red_mean", "green_mean", "blue_mean", "green_ratio", "rg_index",
    "nir_brightness_ratio", "nir_red_ratio",
    # Copernicus spectral
    "cop_ndvi_mean", "fused_ndvi_mean", "fused_ndvi_std",
    # ESA WorldCover
    "esa_built_frac", "esa_tree_frac", "esa_crop_frac",
    "esa_grass_frac", "esa_water_frac",
    # Temporal
    "h_change", "dtm_change", "dtm_change_abs",
    "temporal_h_std", "stability",
    # Volume change (m³ at 1m res)
    "volume_change_m3", "volume_change_abs_m3",
    "dtm_change_max", "dtm_change_frac_03m",
    # GLCM texture
    "glcm_contrast", "glcm_homogeneity", "glcm_entropy",
    "glcm_dissimilarity", "glcm_energy", "texture_complexity",
    # SAR
    "sar_vv", "sar_vh", "sar_ratio",
    # NDVI harmonics
    "harm_mean", "harm_amplitude", "harm_phase", "harm_rmse",
    # Hansen Global Forest Change
    "hansen_treecover2000", "hansen_loss_frac", "hansen_recent_loss_frac", "hansen_loss_3yr_frac",
    "hansen_gain_frac", "hansen_current_forest_frac",
    # Additional discriminators
    "ndvi_max", "slope_max", "h_p10", "perimeter", "esa_dominant_lc",
]

# Mapping from cadastre land-use codes (Benützungsart) to our types.
# These are the Austrian BEV cadastre "BA" codes.
CADASTRE_TO_TYPE = {
    # Agricultural
    51: "crop",       # Acker (arable)
    62: "crop",       # Acker (field)
    52: "grass",      # Wiese (meadow)
    53: "grass",      # Hutweide (pasture)
    54: "grass",      # Alpe (alpine pasture)
    55: "grass",      # Weide (grazing)
    58: "grass",      # Alpe (alpine meadow)
    61: "grass",      # Grünland (grassland)
    # Forest
    56: "tree",       # Wald (forest)
    57: "shrub",      # Krummholz/Strauchwald
    # Orchards/vineyards
    63: "vineyard",   # Weingarten
    64: "garden",     # Hausgarten
    65: "orchard",    # Obstgarten
    # Buildings
    40: "garden",     # Baufläche begrünt (green built-up)
    42: "roof",       # Gebäude
    43: "roof",       # Gebäude (other)
    44: "roof",       # Gebäude (farm)
    45: "roof",       # Gewächshaus (greenhouse)
    46: "roof",       # Gebäude
    47: "roof",       # Gebäude
    41: "parking",    # Baufläche (paved)
    # Transport
    48: "road",       # Straße (road)
    73: "road",       # Straße (road)
    74: "path",       # Weg (path)
    75: "road",       # Brücke (bridge)
    # Water
    70: "water",      # Gewässer
    71: "water",      # Stehende Gewässer
    96: "water",      # Feuchtgebiet
    # Terrain
    59: "bare_soil",  # Ödland
    60: "water",      # Sumpf/Moor (wetland)
    72: "water",      # Quelle/Brunnen (spring)
    80: "earthwork",  # Abbaufläche (quarry)
    81: "earthwork",  # Deponie
    83: "rock",       # Felsen (rock)
    84: "rock",       # Geröll (scree)
    90: "bare_soil",  # sonstige Fläche
    93: "earthwork",  # Abbaufläche
}

# Simplified target classes for the RF (merge rare types).
# Dropped (unlearnable, rule-based only): wind_turbine, substation, solar_panel.
# Merged: excavation + fill → earthwork.
TYPE_CLASSES = [
    "tree", "shrub", "grass", "crop", "road", "path", "parking",
    "roof", "water", "bare_soil", "rock", "earthwork",
    "garden", "orchard", "vineyard",
    "tree_loss",
]

# Classes excluded from RF training — detected by rule-based logic instead.
# Keep in CADASTRE_TO_TYPE so they're still recognised during ground truth
# extraction, but filter them out before training.
RF_EXCLUDED_CLASSES = {"wind_turbine", "substation", "solar_panel"}


def feature_vector(feat: dict) -> np.ndarray:
    """Extract fixed-length feature vector from a segment feature dict."""
    return np.array([feat.get(k, 0.0) for k in FEATURE_KEYS], dtype=np.float32)


def _downsample(
    X: np.ndarray,
    y: np.ndarray,
    cap_multiplier: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample dominant classes to reduce class imbalance.

    Caps each class at ``cap_multiplier * median_class_count`` samples.
    Classes below the cap are kept intact.

    Parameters
    ----------
    X : (N, F) feature matrix
    y : (N,) label array
    cap_multiplier : how many multiples of the median count to allow

    Returns
    -------
    X_ds, y_ds : downsampled arrays
    """
    classes, counts = np.unique(y, return_counts=True)
    median_count = int(np.median(counts))
    cap = cap_multiplier * median_count
    log.info("Downsample: median class size=%d, cap=%d (x%d)",
             median_count, cap, cap_multiplier)

    rng = np.random.RandomState(42)
    keep_idx: list[np.ndarray] = []

    for cls, cnt in zip(classes, counts):
        idx = np.where(y == cls)[0]
        if cnt > cap:
            idx = rng.choice(idx, size=cap, replace=False)
            log.info("  %s: %d -> %d (-%d)", cls, cnt, cap, cnt - cap)
        keep_idx.append(idx)

    keep = np.concatenate(keep_idx)
    keep.sort()  # preserve original ordering
    log.info("Downsample: %d -> %d samples (%.0f%% kept)",
             len(y), len(keep), 100 * len(keep) / len(y))
    return X[keep], y[keep]


class LearnedClassifier:
    """Random Forest classifier trained on cadastre ground truth."""

    def __init__(self):
        self.model = None
        self.classes: list[str] = []
        self.feature_importances: dict[str, float] = {}
        self.n_train: int = 0
        self.oob_score: float = 0.0
        self.trained_at: str = ""
        self.n_kgs: int = 0

    def train(
        self,
        features: list[dict],
        labels: list[str],
        *,
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_leaf: int = 5,
        n_kgs: int = 0,
    ) -> dict:
        """Train RF on feature dicts + string labels.

        Returns training stats dict.
        """
        from sklearn.ensemble import RandomForestClassifier
        import joblib

        # Filter valid
        X_list, y_list = [], []
        for feat, label in zip(features, labels):
            if label not in TYPE_CLASSES:
                continue
            X_list.append(feature_vector(feat))
            y_list.append(label)

        if len(X_list) < 20:
            raise ValueError(f"Need >= 20 training samples, got {len(X_list)}")

        X = np.stack(X_list)
        y = np.array(y_list)

        # Replace NaN/inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ------------------------------------------------------------------
        # Break label circularity for tree_loss samples.
        #
        # tree_loss labels are created in train_rf_4000kg.py using
        # hansen_recent_loss_frac and hansen_treecover2000 as the labelling
        # criteria.  If the RF sees these same features alongside the label
        # it just learns "hansen_recent_loss_frac > 0.15 → tree_loss" —
        # circular, hence the inflated 99.6% OOB.  Zero out only the two
        # criteria features for tree_loss rows so the model must learn from
        # independent signals (height drop, NDVI change, spectral, etc.).
        # Other Hansen features (loss_frac, loss_3yr_frac, gain_frac,
        # current_forest_frac) are NOT zeroed — they are independent.
        # ------------------------------------------------------------------
        _CIRCULAR_HANSEN_COLS = [
            FEATURE_KEYS.index("hansen_recent_loss_frac"),
            FEATURE_KEYS.index("hansen_treecover2000"),
        ]
        tree_loss_mask = (y == "tree_loss")
        n_tl = int(tree_loss_mask.sum())
        if n_tl > 0:
            X[np.ix_(tree_loss_mask, _CIRCULAR_HANSEN_COLS)] = 0.0
            log.info(
                "Label-circularity fix: zeroed hansen_recent_loss_frac & "
                "hansen_treecover2000 for %d tree_loss samples", n_tl,
            )

        # Downsample dominant classes to reduce imbalance
        X, y = _downsample(X, y)

        log.info("Training RF: %d samples, %d features, %d classes",
                 len(X), X.shape[1], len(set(y)))

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            oob_score=True,
            n_jobs=2,  # limit parallelism to avoid memory issues + sklearn warnings
            random_state=42,
            class_weight="balanced",  # handle imbalanced classes
        )
        rf.fit(X, y)

        self.model = rf
        self.classes = list(rf.classes_)
        self.n_train = len(X)
        self.oob_score = rf.oob_score_
        self.feature_importances = {
            k: float(v) for k, v in zip(FEATURE_KEYS, rf.feature_importances_)
        }

        import datetime
        self.trained_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        self.n_kgs = n_kgs

        # Save
        joblib.dump(rf, MODEL_PATH)
        meta = {
            "classes": self.classes,
            "n_train": self.n_train,
            "oob_score": self.oob_score,
            "feature_importances": self.feature_importances,
            "feature_keys": FEATURE_KEYS,
            "trained_at": self.trained_at,
            "n_kgs": self.n_kgs,
        }
        META_PATH.write_text(json.dumps(meta, indent=2))

        # Top features
        top = sorted(self.feature_importances.items(), key=lambda x: -x[1])[:10]
        log.info("RF trained: OOB=%.3f, top features: %s",
                 self.oob_score,
                 ", ".join(f"{k}={v:.3f}" for k, v in top))

        return {
            "n_train": self.n_train,
            "n_classes": len(self.classes),
            "oob_score": self.oob_score,
            "top_features": dict(top),
        }

    def predict(self, feat: dict) -> tuple[str, float]:
        """Predict type and confidence for a single segment.

        Returns (type_name, confidence).
        """
        if self.model is None:
            return "", 0.0

        x = feature_vector(feat).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        pred = self.model.predict(x)[0]
        proba = self.model.predict_proba(x)[0]
        conf = float(proba.max())

        return pred, conf

    def predict_batch(self, features: list[dict]) -> list[tuple[str, float]]:
        """Predict for multiple segments."""
        if self.model is None:
            return [("", 0.0)] * len(features)

        X = np.stack([feature_vector(f) for f in features])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        preds = self.model.predict(X)
        probas = self.model.predict_proba(X)
        confs = probas.max(axis=1)

        return [(str(p), float(c)) for p, c in zip(preds, confs)]

    def find_disagreements(
        self,
        features: list[dict],
        cadastre_labels: list[str],
    ) -> list[dict]:
        """Find segments where RF disagrees with cadastre.

        These are candidate cadastre errors.

        Returns list of {segment_features, cadastre_label, rf_prediction,
        rf_confidence, likely_correct}.
        """
        if self.model is None:
            return []

        results = []
        for feat, cad_label in zip(features, cadastre_labels):
            pred, conf = self.predict(feat)
            if pred != cad_label and conf > 0.6:
                results.append({
                    "label": feat.get("label", 0),
                    "cadastre_type": cad_label,
                    "rf_prediction": pred,
                    "rf_confidence": round(conf, 3),
                    "likely_correct": "rf" if conf > 0.75 else "uncertain",
                    "area": feat.get("area", 0),
                    "centroid_e": feat.get("centroid_e", 0),
                    "centroid_n": feat.get("centroid_n", 0),
                })

        results.sort(key=lambda x: -x["rf_confidence"])
        log.info("Found %d disagreements (RF vs cadastre)", len(results))
        return results

    @classmethod
    def load(cls, model_path=None, meta_path=None) -> "LearnedClassifier":
        """Load a previously trained model, or return empty classifier.

        Prefers the best model from curve evaluation (data/best_model/)
        over the live training model (/tmp/learned_classifier/) when
        available, since the curve eval selects the checkpoint count
        and seed that maximise composite score.
        """
        inst = cls()

        # Determine which model to load: best_model > live
        if model_path and meta_path:
            mp, mtp = pathlib.Path(model_path), pathlib.Path(meta_path)
        elif BEST_MODEL_PATH.exists() and BEST_META_PATH.exists():
            mp, mtp = BEST_MODEL_PATH, BEST_META_PATH
        elif MODEL_PATH.exists() and META_PATH.exists():
            mp, mtp = MODEL_PATH, META_PATH
        else:
            return inst

        try:
            import joblib
            inst.model = joblib.load(mp)
            meta = json.loads(mtp.read_text())
            inst.classes = meta.get("classes", [])
            inst.n_train = meta.get("n_train", 0)
            inst.oob_score = meta.get("oob_score", 0)
            inst.feature_importances = meta.get("feature_importances", {})
            inst.trained_at = meta.get("trained_at", "")
            inst.n_kgs = meta.get("n_kgs", 0)
            source = "best_model" if mp == BEST_MODEL_PATH else "live"
            log.info("Loaded RF model (%s): %d classes, OOB=%.3f, n=%d, kgs=%d, trained=%s",
                     source, len(inst.classes), inst.oob_score, inst.n_train,
                     inst.n_kgs, inst.trained_at)
        except Exception as e:
            log.warning("Failed to load RF model from %s: %s", mp, e)
        return inst

    @property
    def is_trained(self) -> bool:
        return self.model is not None


# ===================================================================
# Integration with object_segmentation pipeline
# ===================================================================

_cached_classifier: Optional[LearnedClassifier] = None
_cached_model_mtime: float = 0.0


def get_classifier() -> LearnedClassifier:
    """Get or load the singleton classifier.  Reloads when model file changes.

    Watches both data/best_model/ and /tmp/learned_classifier/ — prefers
    whichever has the newer mtime (best_model is updated by curve eval,
    live model by the training service).
    """
    global _cached_classifier, _cached_model_mtime
    try:
        best_mtime = BEST_MODEL_PATH.stat().st_mtime if BEST_MODEL_PATH.exists() else 0.0
    except OSError:
        best_mtime = 0.0
    try:
        live_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
    except OSError:
        live_mtime = 0.0
    mtime = max(best_mtime, live_mtime)
    if _cached_classifier is None or mtime != _cached_model_mtime:
        _cached_classifier = LearnedClassifier.load()
        _cached_model_mtime = mtime
    return _cached_classifier


def classify_with_rf(
    feat: dict,
    *,
    fallback_fn=None,
    has_spectral: bool = False,
    min_confidence: float = 0.4,
) -> tuple[str, int, float, bool]:
    """Classify segment: try RF first, fall back to rules.

    Parameters
    ----------
    feat : feature dict from extract_object_features
    fallback_fn : callable(feat, has_spectral=bool) -> (name, code, conf, is_mm)
    has_spectral : whether ortho was available
    min_confidence : minimum RF confidence to trust

    Returns
    -------
    (type_name, type_code, confidence, is_manmade)
    """
    from object_segmentation import OBJECT_TYPES, classify_object

    clf = get_classifier()
    if clf.is_trained:
        pred, conf = clf.predict(feat)
        if pred and conf >= min_confidence:
            # Post-split: earthwork → excavation or fill based on dtm_change sign
            if pred == "earthwork":
                dtm_ch = feat.get("dtm_change", 0.0)
                pred = "excavation" if dtm_ch < 0 else "fill"
            if pred in OBJECT_TYPES:
                code = OBJECT_TYPES[pred]
                is_mm = pred in {
                    "road", "path", "parking", "roof", "wall", "fence",
                    "mast", "greenhouse", "solar_panel", "bridge",
                    "excavation", "fill", "construction", "substation",
                    "wind_turbine",
                }
                return pred, code, conf, is_mm

    # Fallback to rule-based
    if fallback_fn:
        return fallback_fn(feat, has_spectral=has_spectral)
    return classify_object(feat, has_spectral=has_spectral)


def classify_with_rf_batch(
    features: list[dict],
    *,
    has_spectral: bool = False,
    min_confidence: float = 0.4,
) -> dict[int, tuple[str, int, float, bool]]:
    """Batch-classify segments with RF, falling back to rules per-segment.

    Returns dict mapping label -> (type_name, type_code, confidence, is_manmade).
    Uses predict_batch for a single RF call instead of per-segment calls.
    """
    from object_segmentation import OBJECT_TYPES, classify_object

    results = {}
    clf = get_classifier()
    if not clf.is_trained or not features:
        # All rule-based
        for feat in features:
            results[feat["label"]] = classify_object(feat, has_spectral=has_spectral)
        return results

    # Batch predict
    batch_preds = clf.predict_batch(features)

    MANMADE_TYPES = {
        "road", "path", "parking", "roof", "wall", "fence",
        "mast", "greenhouse", "solar_panel", "bridge",
        "excavation", "fill", "construction", "substation",
        "wind_turbine",
    }

    for feat, (pred, conf) in zip(features, batch_preds):
        if pred and conf >= min_confidence:
            # Post-split: earthwork → excavation or fill based on dtm_change
            if pred == "earthwork":
                dtm_ch = feat.get("dtm_change", 0.0)
                pred = "excavation" if dtm_ch < 0 else "fill"
            if pred in OBJECT_TYPES:
                code = OBJECT_TYPES[pred]
                is_mm = pred in MANMADE_TYPES
                results[feat["label"]] = (pred, code, conf, is_mm)
            else:
                results[feat["label"]] = classify_object(feat, has_spectral=has_spectral)
        else:
            # Fall back to rule-based for low-confidence predictions
            results[feat["label"]] = classify_object(feat, has_spectral=has_spectral)

    return results
